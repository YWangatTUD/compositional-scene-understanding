import importlib
import os
import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torchvision.utils import make_grid
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from src.parser import parse_args
from src.hungarian_match import HungarianMatcher
from src.data.dataset import CLEVR2DPosBlender


def main(args):
    device = 'cuda:0'
    weight_dtype = torch.float32

    noise_scheduler_config = DDPMScheduler.load_config(args.scheduler_config)
    noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name, subfolder="vae")
    vae.to(device = device, dtype=weight_dtype)
    vae.requires_grad_(False)

    ckpt_step = 80000
    unet = UNet2DConditionModel.from_pretrained(
        f'/space/ywang86/ckpts/2DPos_ckpts/compositional_text_to_img/checkpoint-{ckpt_step}', subfolder="unet2dconditionmodel".lower())
    unet = unet.to(device=device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.eval()
    print("loaded a trained unet2dconditionmodel")

    scheduler_args = {}
    if "variance_type" in noise_scheduler.config:
        variance_type = noise_scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    # use a more efficient scheduler at test time
    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    scheduler = scheduler_class.from_config(
        noise_scheduler.config, **scheduler_args)

    image_log_dir = f"./outputs/clevr_test_ood_{ckpt_step}/"
    os.makedirs(image_log_dir, exist_ok=True)

    eval_dataset = CLEVR2DPosBlender(
        resolution=args.resolution,
        data_root=args.dataset_root,
        split="val",
        ood=True,
        use_captions=False,
        random_crop=False,
        random_flip=False)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1,
        shuffle=False, drop_last=True
    )
    optimizer_class = torch.optim.AdamW

    total_num = 0.
    for step_data, batch in enumerate(eval_dataloader): # evaluate data point i
        pixel_values = batch[0].to(device=device, dtype=weight_dtype)
        label_gt = batch[1]['y'].to(dtype=weight_dtype)
        mask_orig = batch[1]['mask'].to(device=device,dtype=weight_dtype).squeeze(1)
        bernoulli_probs = torch.zeros_like(mask_orig) + 1.0
        bernoulli = torch.bernoulli(bernoulli_probs)
        mask_orig = mask_orig * bernoulli

        num_comp = torch.sum(mask_orig, dim=1)
        num_comp[num_comp == 0.] = 1.
        num_comp = num_comp.to(dtype=weight_dtype, device=device)
        mask = mask_orig / num_comp.unsqueeze(1)

        total_num += num_comp.squeeze(0)

        # save current image
        grid_image = pixel_values * 0.5 + 0.5
        grid_image = make_grid(grid_image, nrow=1)
        ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        img_path = os.path.join(image_log_dir, f'validation_{step_data}.jpg')
        im.save(img_path, optimize=True, quality=95)


        label = torch.randn([1, int(num_comp.squeeze(0)), 2], device=device, dtype=weight_dtype) * 0.13 + 0.5 ### initialize the labels to make them around the center of images
        label = torch.nn.Parameter(label, requires_grad=True)

        params_group = [
            {'params': [label], "lr": args.learning_rate}
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
        for step_opt in range(2000): # stochastic gradient descent steps
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
            # Sample a random timestep for each image  noise_scheduler.config.num_train_timesteps
            timesteps = torch.randint(0, 500, (bsz,), device=model_input.device) # try sampling time step t from [0,500] or [0,1000]
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

            #composed predictions
            cond_pred = unet(noisy_model_input_expand, timesteps_expand, text_emb_batch.unsqueeze(1)).sample
            cond_pred = cond_pred.view(pixel_values.shape[0], -1, noisy_model_input_expand.shape[1],
                                           noisy_model_input_expand.shape[2], noisy_model_input_expand.shape[3])
            uncond_label = torch.zeros(1, 2).to(dtype=weight_dtype).to(device) # define an unconditional label for calculating unconditional score
            uncond_encoder_hidden_states = uncond_label.repeat(label.shape[0], 1, 1) # unconditional score; using it or not doesn't make much difference
            uncond_pred = unet(noisy_model_input, timesteps, uncond_encoder_hidden_states).sample
            composed_pred = uncond_pred + torch.sum(
                mask[:,:int(num_comp.squeeze(0))].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (cond_pred - uncond_pred.unsqueeze(1)), dim=1)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    model_input, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(composed_pred.float(), target.float(), reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(label, 1.)
            optimizer.step()
            label.data.clamp_(0, 1)
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

        plt.figure().set_figwidth(15, 15)
        plt.figure().patch.set_facecolor('grey')
        plt.scatter(label_gt[:,:int(num_comp.squeeze(0)), 0].squeeze(0), 1 - label_gt[:,:int(num_comp.squeeze(0)), 1].squeeze(0), color='blue', marker='o', facecolors='none', s=350,
                    label='ground-truth location')
        plt.scatter(label[:, :, 0].detach().cpu().numpy(), 1 - label[:, :, 1].detach().cpu().numpy(),
                    color='red', marker='+', s=350, label='predicted location')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
        plt.axis('off')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(image_log_dir, f'inference_prediction_{step_data}.png'), bbox_inches='tight')
        plt.close('all')



if __name__ == "__main__":
    args = parse_args()
    main(args)
