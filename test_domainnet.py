import os
from typing import Optional, Tuple
import numpy as np
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from train_global import Mapper, th2image
from train_global import inj_forward_text, inj_forward_crossattention, validation
import torch.nn as nn
from datasets import CustomDatasetWithBG
import inference_global
from pathlib import Path
import torchvision
import cv2
import copy
import PIL
from torchvision.datasets import ImageFolder

def process(image):
    img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    img = np.array(img).astype(np.float32)
    img = img / 127.5 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

token_index = 0
seed = 42
global_mapper_path = 'checkpoints/global_mapper.pt'
placeholder_token = 'S'
pt_model_name = 'CompVis/stable-diffusion-v1-4'
template = 'A painting of a S'

# load components
vae, unet, text_encoder, tokenizer, image_encoder, mapper, scheduler = inference_global.pww_load_tools(
    "cuda:0",
    LMSDiscreteScheduler,
    diffusion_model_path=pt_model_name,
    mapper_model_path=global_mapper_path,
)

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

# Image
# cub_dset = ImageFolder('/gpfs/u/home/LMTM/LMTMsmms/scratch/data/synthetic-cdm/cub/Real')

RNG = np.random.RandomState(44)
# rand_idxs = RNG.choice(len(cub_dset), 2, replace=False)
img_path = Path('/usr4/cs591/samarthm/projects/synthetic/data/synthetic-cdm/domainnet/sketch/cat/sketch_065_000060.jpg')

# for i, idx in enumerate(rand_idxs):
image = Image.open(img_path).convert('RGB').resize((512, 512))
image.save(f'domainnet_examples/orig_cat.jpg')
# mask_path = self.image_paths[i % self.num_images].replace('.jpeg', '.png').replace('.jpg', '.png').replace('.JPEG', '.png')[:-4] + '_bg.png'
# mask = np.array(Image.open(mask_path))

# mask = np.where(mask > 0, 1, 0)

image_np = np.array(image)
# object_tensor = image_np * mask
# example["pixel_values"] = process(image_np)

# ref_object_tensor = Image.fromarray(object_tensor.astype('uint8')).resize((224, 224), resample=self.interpolation)
ref_image_tenser = Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=PIL.Image.Resampling.BICUBIC)
image_tensor_vae = Image.fromarray(image_np.astype('uint8')).resize((512, 512), resample=PIL.Image.Resampling.BICUBIC)
# example["pixel_values_obj"] = self.get_tensor_clip()(ref_object_tensor)
example["pixel_values_clip"] = get_tensor_clip()(ref_image_tenser).unsqueeze(0)
example["pixel_values"] = 2. * get_tensor_clip(normalize=False, toTensor=True)(image_tensor_vae).unsqueeze(0) - 1. # this normalization is apparently needed


example["pixel_values"] = example["pixel_values"].to("cuda:0").half()
example["pixel_values_clip"] = example["pixel_values_clip"].to("cuda:0").half()
example["input_ids"] = example["input_ids"].to("cuda:0")
example["index"] = example["index"].to("cuda:0").long()

ret_imgs = validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, example["pixel_values_clip"].device, 5,
                    token_index=token_index, seed=seed, strength=0.5)

ret_imgs[0].save(f'domainnet_examples/edited_cat.jpg')

    # ref_seg_tensor = Image.fromarray(mask.astype('uint8') * 255)
    # ref_seg_tensor = self.get_tensor_clip(normalize=False)(ref_seg_tensor)
    # example["pixel_values_seg"] = torch.nn.functional.interpolate(ref_seg_tensor.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)