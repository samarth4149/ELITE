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
    transform_list = [torchvision.transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)]
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
# template = 'An image of a S in the style of domain_clipart'
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

# Loading textual inversion token
# textinv_path = '/usr4/cs591/samarthm/projects/synthetic/synthetic-cdm/expts/textual_inversion_sd_v1-4/train/domainnet/clipart/learned_embeds-steps-5000.bin'
# learned_embeds = torch.load(textinv_path)
# new_token, new_token_embed = next(iter(learned_embeds.items()))
# assert new_token == f'domain_clipart'
# num_added_tokens = tokenizer.add_tokens(new_token)
# assert num_added_tokens > 0, f'Token {new_token} already exists in tokenizer'

# text_encoder.resize_token_embeddings(len(tokenizer))
# added_token_id = tokenizer.convert_tokens_to_ids(new_token)

# # get the old word embeddings
# embeddings = text_encoder.get_input_embeddings()

# # get the id for the token and assign new embeds
# embeddings.weight.data[added_token_id] = \
#     new_token_embed.to(embeddings.weight.dtype)
####################################



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
# img_path = Path('/usr4/cs591/samarthm/projects/synthetic/data/synthetic-cdm/cub/Real/110.Geococcyx/Geococcyx_0038_104266.jpg')
img_path = Path('/usr4/cs591/samarthm/projects/synthetic/data/synthetic-cdm/domainnet/sketch/aircraft_carrier/sketch_001_000041.jpg')

# for i, idx in enumerate(rand_idxs):
image = Image.open(img_path).convert('RGB').resize((512, 512))
# image2 = Image.open(img_path2).convert('RGB')
# image.save(f'cub_examples/orig_geococcyx.jpg')
image.save(f'domainnet_examples_old/orig_aircraft_carrier.jpg')
# image2.save(f'domainnet_examples/orig_aircraft_carrier2.jpg')
# mask_path = self.image_paths[i % self.num_images].replace('.jpeg', '.png').replace('.jpg', '.png').replace('.JPEG', '.png')[:-4] + '_bg.png'
# mask = np.array(Image.open(mask_path))

# mask = np.where(mask > 0, 1, 0)

# image_np = np.array(image)
# object_tensor = image_np * mask
# example["pixel_values"] = process(image_np)

# ref_object_tensor = Image.fromarray(object_tensor.astype('uint8')).resize((224, 224), resample=self.interpolation)
# ref_image_tenser = Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=PIL.Image.Resampling.BICUBIC)
example["pixel_values_clip"] = get_tensor_clip()(image).unsqueeze(0)
# example["pixel_values_clip"] = torch.stack([get_tensor_clip()(image), get_tensor_clip()(image2)], dim=0)
example["pixel_values"] = copy.deepcopy(example["pixel_values_clip"])


example["pixel_values"] = example["pixel_values"].to("cuda:0")
example["pixel_values_clip"] = example["pixel_values_clip"].to("cuda:0").half()
example["input_ids"] = example["input_ids"].to("cuda:0")
example["index"] = example["index"].to("cuda:0").long()

ret_imgs = validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, example["pixel_values_clip"].device, 5,
                    token_index=token_index, seed=seed)

# ret_imgs[0].save(f'cub_examples/edited_geococcyx.jpg')
ret_imgs[0].save(f'domainnet_examples_old/edited_aircraft_carrier_clipart.jpg')

    # ref_seg_tensor = Image.fromarray(mask.astype('uint8') * 255)
    # ref_seg_tensor = self.get_tensor_clip(normalize=False)(ref_seg_tensor)
    # example["pixel_values_seg"] = torch.nn.functional.interpolate(ref_seg_tensor.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)