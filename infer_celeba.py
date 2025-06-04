import os.path as osp
import pandas as pd
import imageio
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
import os
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from src.parser import parse_args
from itertools import product

def main(args):
    device = 'cuda:0'
    weight_dtype = torch.float32

    noise_scheduler_config = DDPMScheduler.load_config(args.scheduler_config)
    noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name, subfolder="vae")
    vae.to(device = device, dtype=weight_dtype)
    vae.requires_grad_(False)

    ckpt_step = 30000
    unet = UNet2DConditionModel.from_pretrained(
        f'/space/ywang86/ckpts/celeba_female_3features/compositional_text_to_img/checkpoint-{ckpt_step}', subfolder="unet2dconditionmodel".lower())
    unet = unet.to(device=device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.eval()
    print("loaded a trained unet2dconditionmodel")

    img_path =  args.dataset_root + "/img_align_celeba/img_align_celeba"
    labels = pd.read_csv(args.dataset_root + "/list_attr_celeba.csv", sep="\s+", skiprows=1)
    labels = labels.reset_index()

    total_num=0
    for image_i in range(100000):
        info = labels.iloc[image_i + 90000]
        fname = info.iloc[1].split(",")[0]
        path = osp.join(img_path, fname)
        im = imageio.imread(path)
        im = np.array(Image.fromarray(im).resize((128, 128)))
        im = im / 255.

        label_gender = int(info.iloc[1].split(",")[21])
        if label_gender == 1: #-1 female; 0 male
            continue
        total_num += 1
        if total_num == 1001:
            break

        label_blackhair = int(info.iloc[1].split(",")[9])
        if label_blackhair == -1:
            label_blackhair = 0
        label_blackhair = np.eye(6)[label_blackhair]

        label_glasses = int(info.iloc[1].split(",")[16])
        if label_glasses == -1:
            label_glasses = 0
        label_glasses = np.eye(6)[label_glasses + 2]

        label_smile = int(info.iloc[1].split(",")[32])
        if label_smile == -1:
            label_smile = 0
        label_smile = np.eye(6)[label_smile + 4]

        label = np.array([label_blackhair, label_glasses, label_smile])

        pixel_values = torch.tensor(im).unsqueeze(0).permute(0, 3, 1, 2).to(device=device, dtype=weight_dtype)
        label_true = torch.tensor(label).unsqueeze(0).to(device=device, dtype=weight_dtype)

        allowed_indices = [
            (0, 1),  # Row 1: Positions 0 or 1
            (2, 3),  # Row 2: Positions 2 or 3
            (4, 5),  # Row 3: Positions 4 or 5
        ]
        # Generate all possible false labels
        possible_labels = []
        for bitmask in product([0, 1], repeat=3):
            # Construct a new label as a tensor with batch dimension
            new_label = torch.zeros((1, 3, 6)).to(device=device, dtype=weight_dtype)
            for row_idx, flip_bit in enumerate(bitmask):
                pos1, pos2 = allowed_indices[row_idx]
                if flip_bit == 0:
                    new_label[0, row_idx, pos1] = 1.
                else:
                    new_label[0, row_idx, pos2] = 1.
            possible_labels.append(new_label)

        label_cat = torch.cat(possible_labels, dim=0).unsqueeze(0).to(device=device, dtype=weight_dtype)  # Shape: (1, 8, 3, 6)

        denoising_acc = torch.zeros(label_cat.shape[1]).to(device=device)
        for step in range(1000):
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
                0, 1000, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(
                model_input, noise, timesteps)

            noisy_model_input_expand = noisy_model_input.unsqueeze(1).unsqueeze(1).expand(-1, label_cat.shape[1], label_true.shape[1], -1, -1, -1)
            timesteps_expand = timesteps.unsqueeze(1).unsqueeze(1).expand(-1, label_cat.shape[1], label_true.shape[1])

            noisy_model_input_expand = torch.flatten(noisy_model_input_expand, 0, 2)
            timesteps_expand = torch.flatten(timesteps_expand, 0, 2)

            text_emb_batch_true = torch.flatten(label_cat, 0, 2)

            #conditional pixel prediction of components
            cond_pred = unet(noisy_model_input_expand, timesteps_expand, text_emb_batch_true.unsqueeze(1)).sample
            cond_pred = cond_pred.view(pixel_values.shape[0], label_cat.shape[1], label_true.shape[1], noisy_model_input_expand.shape[1],
                                           noisy_model_input_expand.shape[2], noisy_model_input_expand.shape[3])
            composed_pred = cond_pred.mean(dim=2)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    model_input, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            # Compute instance loss
            for i in range(label_cat.shape[1]):
                denoising_acc[i] += F.mse_loss(composed_pred[:,i,:,:,:].float(), target.float(), reduction="mean")


        predicted_label = label_cat[:,torch.argmin(denoising_acc),:,:]

        if predicted_label[0,0,0] == 1.:
            pred_label_hair = "noblackhair"
        else:
            pred_label_hair = "blackhair"

        if predicted_label[0,1,2] == 1.:
            pred_label_glasses = "noglasses"
        else:
            pred_label_glasses = "glasses"

        if predicted_label[0,2,4] == 1.:
            pred_label_smile = "nosmiling"
        else:
            pred_label_smile = "smiling"

        #label the image with the file name and save it
        image_save = Image.fromarray((im * 255).astype(np.uint8))
        image_log_dir = f'./outputs/celeba_test_id_{ckpt_step}'
        os.makedirs(image_log_dir, exist_ok=True)
        img_save_path = os.path.join(image_log_dir, f"image_{total_num}_{pred_label_hair}_{pred_label_glasses}_{pred_label_smile}.jpg")
        image_save.save(img_save_path, optimize=True, quality=95)
        print(f"image_{total_num}_{pred_label_hair}_{pred_label_glasses}_{pred_label_smile}.jpg")

if __name__ == "__main__":
    args = parse_args()
    main(args)
