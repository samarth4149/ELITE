import os
from typing import Optional, Tuple
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
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    do_classifier_free_guidance=True,
    guess_mode=False,
):
    image = control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)

    return image

@torch.no_grad()
def validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, controlnet, control_image_processor, device, guidance_scale, token_index='full', seed=None):
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
    control_image = prepare_control_image(
        control_image_processor=control_image_processor,
        image=control_image,
        width=512,
        height=512,
        batch_size=example["pixel_values"].shape[0],
        num_images_per_prompt=1,
        device=device,
        dtype=controlnet.dtype,
        do_classifier_free_guidance=True,
        guess_mode=False,
    )

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
        
        # noise_pred_text = unet(
        #     latent_model_input,
        #     t,
        #     encoder_hidden_states={
        #         "CONTEXT_TENSOR": encoder_hidden_states,
        #     }
        # ).sample
        
        # noise_pred_uncond = unet(
        #     latent_model_input,
        #     t,
        #     down_block_additional_residuals=down_block_res_samples,
        #     mid_block_additional_residual=mid_block_res_sample,
        #     encoder_hidden_states={
        #         "CONTEXT_TENSOR": uncond_embeddings,
        #     }
        # ).sample

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    _latents = 1 / 0.18215 * latents.clone()
    images = vae.decode(_latents).sample
    ret_pil_images = [th2image(image) for image in images]

    return ret_pil_images

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = [torchvision.transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)



pt_model_name = 'CompVis/stable-diffusion-v1-4'
token_index = 0
seed = 42
vae_scale_factor = 8



# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     pt_model_name, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
# )

# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


global_mapper_path = 'checkpoints/global_mapper.pt'
device = "cuda:0"

vae, unet, text_encoder, tokenizer, image_encoder, mapper, scheduler = inference_global.pww_load_tools(
    device,
    UniPCMultistepScheduler,
    diffusion_model_path=pt_model_name,
    mapper_model_path=global_mapper_path,
)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
controlnet.to(device)
control_image_processor = VaeImageProcessor(
    vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False
)


placeholder_token = 'S'
template = 'A painting of a S'

# Preparing example
example = {}

# Text
placeholder_string = placeholder_token
text = template.format(placeholder_string)

placeholder_index = 0
words = text.strip().split(' ')
for idx, word in enumerate(words):
    if word == placeholder_string:
        placeholder_index = idx + 1


example["index"] = torch.tensor(placeholder_index).unsqueeze(0)

example["input_ids"] = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=tokenizer.model_max_length,
    return_tensors="pt",
).input_ids

img_path = Path('/usr4/cs591/samarthm/projects/synthetic/data/synthetic-cdm/domainnet/sketch/aircraft_carrier/sketch_001_000041.jpg')
# img_path = Path('/usr4/cs591/samarthm/projects/synthetic/data/synthetic-cdm/domainnet/sketch/cake/sketch_052_000073.jpg')
# img_path = Path('/gpfs/u/home/LMTM/LMTMsmms/scratch/data/synthetic-cdm/domainnet/sketch/aircraft_carrier/sketch_001_000041.jpg')

# for i, idx in enumerate(rand_idxs):
image = Image.open(img_path).convert('RGB').resize((512, 512))
image.save(f'domainnet_examples_controlnet/orig_aircraft_carrier.jpg')

low_threshold = 100
high_threshold = 200

npimg = np.array(image)

canny_image = cv2.Canny(npimg, low_threshold, high_threshold)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image = Image.fromarray(canny_image)


example["pixel_values_clip"] = get_tensor_clip()(image).unsqueeze(0)
example["pixel_values"] = copy.deepcopy(example["pixel_values_clip"]) # NOTE : only used for shape information
example["pixel_values"] = example["pixel_values"].to("cuda:0") 
example["pixel_values_clip"] = example["pixel_values_clip"].to("cuda:0").half()
example["input_ids"] = example["input_ids"].to("cuda:0")
example["index"] = example["index"].to("cuda:0").long()
example["control_image"] = canny_image

ret_imgs = validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, controlnet, control_image_processor, example["pixel_values_clip"].device, 5,
                      token_index=token_index, seed=seed)

# ret_imgs[0].save(f'cub_examples/edited_geococcyx.jpg')
ret_imgs[0].save(f'domainnet_examples_controlnet/edited_aircraft_carrier_painting.jpg')

