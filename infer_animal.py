import os
import torch

torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.utils.checkpoint

from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from torchvision.transforms import transforms
import torchvision

def main():
    device = 'cuda:0'
    weight_dtype = torch.float32

    noise_scheduler = DDPMScheduler.from_pretrained(
        'stabilityai/stable-diffusion-2-1', subfolder="scheduler")
    print("loaded a SD scheduler")

    vae = AutoencoderKL.from_pretrained(
        'stabilityai/stable-diffusion-2-1', subfolder="vae")
    vae.to(device = device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()
    print("loaded a SD vae")

    unet = UNet2DConditionModel.from_pretrained(
        'stabilityai/stable-diffusion-2-1', subfolder="unet", revision=None)
    unet = unet.to(device=device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.eval()
    print("loaded a SD unet")

    tokenizer = CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="tokenizer")
    print("loaded a CLIP Tokenizer")

    text_encoder = CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="text_encoder")
    text_encoder = text_encoder.to(device=device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    print("loaded a CLIP TextEncoder")

    input_img_dir = './data/dog-cat-bunny-dataset'
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    size =512
    interpolation = torchvision.transforms.functional.InterpolationMode.BICUBIC
    img_transforms = transforms.Compose([
        transforms.Resize(size, interpolation=interpolation),
        transforms.CenterCrop(size),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    acc_num_CDC =0.
    total_num = 0.
    for images in os.listdir(input_img_dir):
        if images.endswith('.jpg') or images.endswith('.png') or images.endswith('.jpeg'):
            print(f"image name: {images}")
            img = Image.open(os.path.join(input_img_dir, images))
            img = img_transforms(img)
            pixel_values = img.unsqueeze(0).to(device=device, dtype=weight_dtype)

            prompt_1 = 'a photo of a cat'
            prompt_2 = 'a photo of a dog'
            prompt_3 = 'a photo of a rabbit'

            text_input_1 = tokenizer(prompt_1, padding="max_length",
                                   max_length=tokenizer.model_max_length, truncation=True,
                                   return_tensors="pt").to(device)
            text_embeddings_1 = text_encoder(text_input_1.input_ids)[0]

            text_input_2 = tokenizer(prompt_2, padding="max_length",
                                     max_length=tokenizer.model_max_length, truncation=True,
                                     return_tensors="pt").to(device)
            text_embeddings_2 = text_encoder(text_input_2.input_ids)[0]

            text_input_3 = tokenizer(prompt_3, padding="max_length",
                                     max_length=tokenizer.model_max_length, truncation=True,
                                     return_tensors="pt").to(device)
            text_embeddings_3 = text_encoder(text_input_3.input_ids)[0]

            loss_compose_1 = 0.
            loss_compose_2 = 0.
            loss_compose_3 = 0.

            for step in range(1000):
                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # Sample noise that we'll add to the model input
                noise = torch.randn_like(model_input)
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, 1000, (1,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(
                    model_input, noise, timesteps)

                #composed prediction
                composed_pred_1 = (unet(noisy_model_input, timesteps, text_embeddings_1).sample + unet(noisy_model_input, timesteps, text_embeddings_2).sample)/2
                composed_pred_2 = (unet(noisy_model_input, timesteps, text_embeddings_1).sample + unet(noisy_model_input, timesteps, text_embeddings_3).sample)/2
                composed_pred_3 = (unet(noisy_model_input, timesteps, text_embeddings_2).sample + unet(noisy_model_input, timesteps, text_embeddings_3).sample)/2

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        model_input, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                # Compute instance loss
                loss_compose_1 += F.mse_loss(composed_pred_1.float(), target.float(), reduction="mean")
                loss_compose_2 += F.mse_loss(composed_pred_2.float(), target.float(), reduction="mean")
                loss_compose_3 += F.mse_loss(composed_pred_3.float(), target.float(), reduction="mean")
            total_num+=1
            if loss_compose_1 < loss_compose_2 and loss_compose_1 < loss_compose_3:
                print("the image contains: a cat and a dog")
            if loss_compose_2 < loss_compose_1 and loss_compose_2 < loss_compose_3:
                print("the image contains: a cat and a rabbit")
            if loss_compose_3 < loss_compose_1 and loss_compose_3 < loss_compose_2:
                print("the image contains: a dog and a rabbit")


if __name__ == "__main__":
    main()
