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

from data_list import Imagelist
from subset_w_idx import SubsetWIdx
import argparse

from pathlib import Path

# TEMPLATES = {
#     'office_home' : {
#         'Art' : 'A painting/artistic photo of a S',
#         'Clipart' : 'A clipart image of S',
#         'Product' : 'A product image of S on a white background',
#         'Real' : 'A colored realistic photo of a S',
#     },
#     'cub' : {
#         'Real' : 'A colored realistic photo of S',
#         'Painting' : 'A painting of S',
#     },
#     'domainnet': {
#         'clipart' : 'A clipart image of S',
#         'painting': 'A painting of a S',
#         'sketch': 'A pencil/charcoal sketch of S',
#     },
# }

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Generate ELITE data for CUB.')
    parser.add_argument('--source', type=str, default='Real', help='src domain to generate data for')
    parser.add_argument('--target', type=str, default='Painting', help='target domain')
    parser.add_argument('--num_jobs', type=int, default=1, help='number of jobs to run in parallel')
    parser.add_argument('--job_idx', type=int, default=0, help='job index')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset', type=str, default='cub', help='dataset to generate data for')
    parser.add_argument('--root_dir', type=str, default='/projectnb/ivc-ml/samarth/projects/synthetic/data/synthetic-cdm/synthetic_data/hparams/elite_global')
    
    parser.add_argument('--filelist_root', type=str, default='/usr4/cs591/samarthm/projects/synthetic/synthetic-cdm/CDS_pretraining/data')
    parser.add_argument('--filelist', type=str, default='')
    return parser.parse_args(args)

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

def main(args):
    scenario = f'{args.source[0]}2{args.target[0]}'
    token_index = 0
    global_mapper_path = 'checkpoints/global_mapper.pt'
    placeholder_token = 'S'
    pt_model_name = 'CompVis/stable-diffusion-v1-4'
    template = f'An image of S in the style of domain_{args.target}'
    # template = TEMPLATES[args.dataset][args.target]

    # load components
    vae, unet, text_encoder, tokenizer, image_encoder, mapper, scheduler = inference_global.pww_load_tools(
        "cuda:0",
        LMSDiscreteScheduler,
        diffusion_model_path=pt_model_name,
        mapper_model_path=global_mapper_path,
    )
    
    # Loading textual inversion token
    textinv_path = f'/usr4/cs591/samarthm/projects/synthetic/synthetic-cdm/expts/textual_inversion_sd_v1-4/train/domainnet/{args.target}/learned_embeds-steps-5000.bin'
    learned_embeds = torch.load(textinv_path)
    new_token, new_token_embed = next(iter(learned_embeds.items()))
    assert new_token == f'domain_{args.target}'
    num_added_tokens = tokenizer.add_tokens(new_token)
    assert num_added_tokens > 0, f'Token {new_token} already exists in tokenizer'

    text_encoder.resize_token_embeddings(len(tokenizer))
    added_token_id = tokenizer.convert_tokens_to_ids(new_token)

    # get the old word embeddings
    embeddings = text_encoder.get_input_embeddings()

    # get the id for the token and assign new embeds
    embeddings.weight.data[added_token_id] = \
        new_token_embed.to(embeddings.weight.dtype)
    ####################################

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

    orig_index = torch.tensor(placeholder_index).unsqueeze(0).repeat(args.batch_size)

    orig_input_ids = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.repeat(args.batch_size, 1)

    # Image
    if args.filelist:
        dset = Imagelist(args.filelist, transform=get_tensor_clip())
    else:
        dset = Imagelist(Path(args.filelist_root) / f'{args.dataset}/{args.source}_train.txt', transform=get_tensor_clip())
    # dset = Imagelist(f'/gpfs/u/home/LMTM/LMTMsmms/scratch/projects/synthetic-cdm/CDS_pretraining/data/{args.dataset}/{args.source}_train.txt', transform=get_tensor_clip())
    args.root_dir = Path(args.root_dir) / args.dataset / scenario

    if args.num_jobs > 1:
        dset.mode_self = False
        curr_idxs = np.array_split(np.arange(len(dset)), args.num_jobs)[args.job_idx]
        dset = SubsetWIdx(dset, curr_idxs)
        dset.imgs = [dset.dataset.imgs[i] for i in curr_idxs]
        dset.labels = dset.dataset.labels[curr_idxs]
        
    loader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    for i, batch in enumerate(loader):
        example["pixel_values_clip"] = batch[0]
        example["pixel_values"] = copy.deepcopy(example["pixel_values_clip"])
        
        # img_path = Path('/usr4/cs591/samarthm/projects/synthetic/data/synthetic-cdm/domainnet/sketch/aircraft_carrier/sketch_001_000035.jpg')
        # img_path2 = Path('/usr4/cs591/samarthm/projects/synthetic/data/synthetic-cdm/domainnet/sketch/aircraft_carrier/sketch_001_000041.jpg')

        # # for i, idx in enumerate(rand_idxs):
        # image = Image.open(img_path).convert('RGB')
        # image2 = Image.open(img_path2).convert('RGB')
        # tmp = torch.stack([get_tensor_clip()(image), get_tensor_clip()(image2)], dim=0) # should be equal to batch[0]

        
        example["index"] = orig_index[:len(batch[0])]
        example["input_ids"] = orig_input_ids[:len(batch[0])]
        
        example["pixel_values"] = example["pixel_values"].to("cuda:0")
        example["pixel_values_clip"] = example["pixel_values_clip"].to("cuda:0").half()
        example["input_ids"] = example["input_ids"].to("cuda:0")
        example["index"] = example["index"].to("cuda:0").long()
        
        ret_imgs = validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, example["pixel_values_clip"].device, 5, token_index=token_index, seed=args.seed)
        
        src_paths = [loader.dataset.imgs[idx] for idx in batch[2]]
        src_paths = [str(Path(p).relative_to(Path(p).parents[1])) for p in src_paths]
        
        for path, img in zip(src_paths, ret_imgs):
            if str(path).startswith('/'):
                raise Exception(f'Path {path} should be relative')
            out_path = Path(args.root_dir) / path
            os.makedirs(out_path.parent, exist_ok=True)
            img.save(out_path)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)