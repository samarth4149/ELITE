import os
import numpy as np
import torch
from diffusers import LMSDiscreteScheduler
from PIL import Image
import torch.nn as nn
from datasets import CustomDatasetWithBG
import inference_global
from pathlib import Path
import torchvision
import cv2
import copy
import PIL
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from tqdm import tqdm
from train_global import th2image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor

def prepare_control_image(
    control_image_processor,
    image,
    width,
    height,
):
    image = control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
    image = image.squeeze(0)
    
    return image

@torch.no_grad()
def validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, controlnet, device, guidance_scale, token_index='full', seed=None):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    uncond_input = tokenizer(
        [''] * example["pixel_values"].shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder({'input_ids':uncond_input.input_ids.to(device)})[0]

    if seed is None:
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.config.in_channels, 64, 64)
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.config.in_channels, 64, 64), generator=generator,
        )

    control_image = example['control_image']
    control_image = torch.cat([control_image, control_image]) # for classifier free guidance

    latents = latents.to(example["pixel_values_clip"])
    scheduler.set_timesteps(100)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["index"]
    image = F.interpolate(example["pixel_values_clip"], (224, 224), mode='bilinear')

    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12],
                        image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = mapper(image_embeddings)

    if token_index != 'full':
        token_index = int(token_index)
        inj_embedding = inj_embedding[:, token_index:token_index + 1, :]

    encoder_hidden_states = text_encoder({'input_ids': example["input_ids"],
                                          "inj_embedding": inj_embedding,
                                          "inj_index": placeholder_idx})[0]
    
    encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states]) # "batch_dim" gets doubled here
    
    controlnet_cond_scale = 1.
    controlnet_guidance_start = 0.
    controlnet_guidance_end = 1.
    
    controlnet_keep = []
    for i in range(len(scheduler.timesteps)):
        s, e = controlnet_guidance_start, controlnet_guidance_end
        keep = 1.0 - float(i / len(scheduler.timesteps) < s or (i + 1) / len(scheduler.timesteps) > e)
        controlnet_keep.append(keep)

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        latent_model_input = torch.cat([latents] * 2) # Classifier free guidance is done
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        control_model_input = latent_model_input
        
        controlnet_prompt_embeds = encoder_hidden_states
        cond_scale = controlnet_cond_scale * controlnet_keep[i]
        down_block_res_samples, mid_block_res_sample = controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=control_image,
            conditioning_scale=cond_scale,
            guess_mode=False,
            return_dict=False,
        )
        
        noise_pred = unet(
            latent_model_input,
            t,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            encoder_hidden_states={
                "CONTEXT_TENSOR": encoder_hidden_states,
            }
        ).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    _latents = 1 / 0.18215 * latents.clone()
    images = vae.decode(_latents).sample
    ret_pil_images = [th2image(image) for image in images]

    return ret_pil_images
