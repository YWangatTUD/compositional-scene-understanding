import copy
import importlib
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from torchvision.utils import make_grid
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from src.data.dataset import CLEVR2DPosBlender
from src.pipeline.composable_stable_diffusion_pipeline import ComposableStableDiffusionPipeline

from src.parser import parse_args

if is_wandb_available():
    import wandb

logger = get_logger(__name__)

@torch.no_grad()
def log_visualization(
        vis_dataset,
        unet,
        vae,
        scheduler,
        args,
        accelerator,
        weight_dtype,
        global_step,
):
    logger.info(
        f"Running visualization... \n."
    )

    unet = accelerator.unwrap_model(unet)

    vis_dataloader = torch.utils.data.DataLoader(
        vis_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in scheduler.config:
        variance_type = scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    # use a more efficient scheduler at test time
    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    scheduler = scheduler_class.from_config(
        scheduler.config, **scheduler_args)

    pipeline = ComposableStableDiffusionPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=None,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    num_digits = len(str(args.max_train_steps))
    folder_name = f"image_logging_{global_step:0{num_digits}}"
    image_log_dir = os.path.join(accelerator.logging_dir, folder_name, )
    os.makedirs(image_log_dir, exist_ok=True)

    images = []
    image_count = 0

    for step, batch in enumerate(vis_dataloader):
        pixel_values = batch[0].to(device=accelerator.device, dtype=weight_dtype)
        label = batch[1]['y']
        mask_orig = batch[1]['mask'].squeeze(1).to(device=accelerator.device)
        num_comp = torch.sum(mask_orig, dim=1)
        num_comp[num_comp == 0.] = 1.
        num_comp = num_comp
        mask = mask_orig / num_comp.unsqueeze(1)
        mask = mask.to(device=accelerator.device)

        with torch.autocast("cuda"):
            images_gen = []
            for slot_i in range(label.shape[1]):
                generator = torch.Generator(
                    device=accelerator.device).manual_seed(args.seed)
                images_gen_i = pipeline(
                    prompt_embeds=label[:, slot_i, :].unsqueeze(1),
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=25,
                    generator=generator,
                    guidance_scale=1.,
                    output_type="pt",
                ).images
                images_gen.append(images_gen_i.unsqueeze(1))
            images_gen = mask_orig.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * torch.cat(images_gen, dim=1)

            generator = torch.Generator(
                device=accelerator.device).manual_seed(args.seed)
            images_recon = pipeline(
                prompt_embeds=label,
                mask = mask,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=25,
                generator=generator,
                guidance_scale=1.,
                output_type="pt",
            ).images

        grid_image = torch.cat(
            [pixel_values.unsqueeze(1) * 0.5 + 0.5, images_gen, images_recon.unsqueeze(1)], dim=1)
        grid_image = make_grid(
            grid_image.view(grid_image.shape[0] * grid_image.shape[1], grid_image.shape[2], grid_image.shape[3],
                            grid_image.shape[4], ), nrow=grid_image.shape[1])
        ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        images.append(im)
        img_path = os.path.join(image_log_dir, f"image_02.jpg")
        im.save(img_path, optimize=True, quality=95)
        image_count += pixel_values.shape[0]
        if image_count >= args.num_validation_images:
            break

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "visualization", np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "visualization": [
                        wandb.Image(image, caption=f"{i}") for i, image in enumerate(images)
                    ]
                }
            )
    torch.cuda.empty_cache()
    return images


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler and models
    noise_scheduler_config = DDPMScheduler.load_config(args.scheduler_config)
    noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name, subfolder="vae")

    if os.path.exists(args.unet_config):
        unet_config = UNet2DConditionModel.load_config(args.unet_config)
        unet = UNet2DConditionModel.from_config(unet_config)
    else:
        raise ValueError(
            f"Unknown unet config {args.unet_config}")

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:

                # continue if not one of [UNet2DConditionModel]
                if not isinstance(model, (UNet2DConditionModel)):
                    continue
                sub_dir = model._get_name().lower()
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()
            sub_dir = model._get_name().lower()
            if isinstance(model, UNet2DConditionModel):
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder=sub_dir)
                model.register_to_config(**load_model.config)
            else:
                raise ValueError(
                    f"Unknown model type {type(model)}")
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps *
                args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = list(unet.parameters())
    params_group = [
        {'params': unet.parameters(), "lr": args.learning_rate}
    ]

    optimizer = optimizer_class(
        params_group,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda _: 1]
    )

    train_dataset = CLEVR2DPosBlender(
            resolution=args.resolution,
            data_root=args.dataset_root,
            split="train",
            ood=False,
            use_captions=False,
            random_crop=False,
            random_flip=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=False, drop_last=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler, unet = accelerator.prepare(optimizer, train_dataloader, lr_scheduler, unet)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            args.tracker_project_name, config=tracker_config
        )

    # Train!
    total_batch_size = args.train_batch_size * \
                       accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    accumulate_steps = 0  # necessary for args.gradient_accumulation_steps > 1

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(
                args.resume_from_checkpoint.rstrip('/'))  # only the checkpoint folder name is needed, not the full path
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            accumulate_steps = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        position=0, leave=True
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch[0].to(dtype=weight_dtype)
            label = batch[1]['y'].to(dtype=weight_dtype)
            mask_orig = batch[1]['mask'].to(dtype=weight_dtype).squeeze(1)

            num_comp = torch.sum(mask_orig, dim=1)
            num_comp[num_comp == 0.] = 1.
            num_comp = num_comp.to(dtype=weight_dtype,device=accelerator.device)
            mask = mask_orig/num_comp.unsqueeze(1)

            # Convert images to latent space
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            # Sample noise that we'll add to the model input
            if args.offset_noise:
                noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                    model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
                )
            else:
                noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(
                model_input, noise, timesteps)

            noisy_model_input_expand = noisy_model_input[:, None, :].expand(-1, label.shape[1], -1, -1, -1)
            timesteps_expand = timesteps[:, None].expand(-1, label.shape[1])

            noisy_model_input_expand = torch.flatten(noisy_model_input_expand, 0, 1)
            timesteps_expand = torch.flatten(timesteps_expand, 0, 1)
            text_emb_batch = torch.flatten(label, 0, 1)

            #conditional pixel prediction of components
            cond_pred = unet(noisy_model_input_expand, timesteps_expand, text_emb_batch.unsqueeze(1)).sample
            cond_pred = cond_pred.view(pixel_values.shape[0], -1, noisy_model_input_expand.shape[1],
                                           noisy_model_input_expand.shape[2], noisy_model_input_expand.shape[3])

            # Compute instance loss without uncond prediction
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    model_input, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Compute instance loss
            #unconditional pixel prediction of entire image
            uncond_label = torch.zeros(1,2).to(dtype=weight_dtype).to(accelerator.device)
            uncond_encoder_hidden_states = uncond_label.repeat(label.shape[0], 1, 1)
            uncond_pred = unet(noisy_model_input, timesteps, uncond_encoder_hidden_states).sample
            composed_pred = uncond_pred + torch.sum(mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*(cond_pred - uncond_pred.unsqueeze(1)), dim=1)
            loss = F.mse_loss(composed_pred.float(), target.float(), reduction="mean")

            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)
            accumulate_steps += 1
            # if accelerator.sync_gradients:
            if (accumulate_steps + 1) % args.gradient_accumulation_steps == 0:
                params_to_clip = params_to_optimize
                accelerator.clip_grad_norm_(
                    params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if (accumulate_steps + 1) % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    images = []
                    if global_step % args.validation_steps == 0: #
                        num_digits = len(str(args.max_train_steps))
                        folder_name = f"image_logging_{global_step:0{num_digits}}"
                        image_log_dir = os.path.join(accelerator.logging_dir, folder_name, )
                        os.makedirs(image_log_dir, exist_ok=True)

                        vis_pix = pixel_values[:args.num_validation_images]
                        #vis_recon = model_pred[:args.num_validation_images]/vae.config.scaling_factor
                        vis_recon = composed_pred[:args.num_validation_images]/vae.config.scaling_factor
                        vis_recon = vae.decode(vis_recon.to(torch.float16)).sample
                        vis_output = mask_orig[:args.num_validation_images].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*cond_pred[:args.num_validation_images]/vae.config.scaling_factor
                        vis_comps = []
                        for comp_i in range(cond_pred.shape[1]):
                            comp = vae.decode(vis_output[:, comp_i, :, :, :].to(torch.float16)).sample
                            vis_comps.append(comp.unsqueeze(1))
                        vis_comps = torch.cat(vis_comps, dim=1)

                        vis_recon = vis_recon * 0.5 + 0.5
                        vis_comps = vis_comps * 0.5 + 0.5

                        grid_image = torch.cat(
                            [vis_pix.unsqueeze(1) * 0.5 + 0.5, vis_comps.clamp(0, 1).to(torch.float32), vis_recon.clamp(0, 1).to(torch.float32).unsqueeze(1)], dim=1)
                        grid_image = make_grid(
                            grid_image.view(grid_image.shape[0] * grid_image.shape[1], grid_image.shape[2],
                                            grid_image.shape[3],
                                            grid_image.shape[4], ), nrow=grid_image.shape[1])
                        ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                 torch.uint8).numpy()
                        im = Image.fromarray(ndarr)
                        images.append(im)
                        img_path = os.path.join(image_log_dir, f"image_{step:02}.jpg")
                        im.save(img_path, optimize=True, quality=95)

                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                tracker.log(
                                    {
                                        "train": [
                                            wandb.Image(image, caption=f"{i}") for i, image in enumerate(images)
                                        ]
                                    }
                                )
                        torch.cuda.empty_cache()

                        #visualilze generations of individual components
                        images = log_visualization(
                            vis_dataset=train_dataset,
                            unet=unet,
                            vae=vae,
                            scheduler=noise_scheduler,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            global_step=global_step,
                        )

            logs = {"loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(
            args.output_dir, f"checkpoint-{global_step}-last")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
