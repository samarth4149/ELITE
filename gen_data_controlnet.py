import os
import numpy as np
import torch
from diffusers import LMSDiscreteScheduler
from pathlib import Path
import torchvision
import cv2
import copy
import PIL

from data_list import Imagelist
from subset_w_idx import SubsetWIdx
import argparse

from pathlib import Path

from diffusers import ControlNetModel
from diffusers.image_processor import VaeImageProcessor
from controlnet_fns import prepare_control_image, validation
import inference_global # for pww_load_tools
from functools import partial

TEMPLATES = {
    'office_home' : {
        'Art' : 'A painting/artistic photo of a <S>',
        'Clipart' : 'A clipart image of <S>',
        'Product' : 'A catalog image of <S> on a white background',
        'Real' : 'A realistic photo of a <S>',
    },
    'cub' : {
        'Real' : 'A colored realistic photo of <S>',
        'Painting' : 'A painting of <S>',
    },
    'domainnet': {
        'clipart' : 'A clipart image of <S>',
        'painting': 'A painting of a <S>',
        'sketch': 'A pencil/charcoal sketch of <S>',
    },
}

RNG = np.random.RandomState(44)

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Generate ELITE data for CUB.')
    parser.add_argument('--source', type=str, default='Real', help='src domain to generate data for')
    parser.add_argument('--target', type=str, default='Painting', help='target domain')
    parser.add_argument('--num_jobs', type=int, default=1, help='number of jobs to run in parallel')
    parser.add_argument('--job_idx', type=int, default=0, help='job index')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset', type=str, default='cub', help='dataset to generate data for')
    parser.add_argument('--root_dir', type=str, default='/projectnb/ivc-ml/samarth/projects/synthetic/data/synthetic-cdm/synthetic_data/elite_plus_controlnet')
    
    parser.add_argument('--filelist_root', type=str, default='/usr4/cs591/samarthm/projects/synthetic/synthetic-cdm/CDS_pretraining/data')
    parser.add_argument('--filelist', type=str, default='')
    return parser.parse_args(args)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = [torchvision.transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def control_transform(image : PIL.Image.Image, 
                      control_image_processor, 
                      width, 
                      height,):
    low_threshold = 100
    high_threshold = 200

    npimg = np.array(image)
    
    canny_image = cv2.Canny(npimg, low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = PIL.Image.fromarray(canny_image)
    
    control_image = prepare_control_image(
        control_image_processor, canny_image, width, height)
    return control_image
    

def main(args):
    scenario = f'{args.source[0]}2{args.target[0]}'
    token_index = 0
    global_mapper_path = 'checkpoints/global_mapper.pt'
    placeholder_token = '<S>'
    pt_model_name = 'CompVis/stable-diffusion-v1-4'
    template = TEMPLATES[args.dataset][args.target]
    controlnet_model_name = 'lllyasviel/sd-controlnet-canny'
    device = 'cuda:0'
    vae_scale_factor = 8

    # load components
    vae, unet, text_encoder, tokenizer, image_encoder, mapper, scheduler = inference_global.pww_load_tools(
        device,
        LMSDiscreteScheduler,
        diffusion_model_path=pt_model_name,
        mapper_model_path=global_mapper_path,
    )
    
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_name, torch_dtype=torch.float16
    )
    controlnet.to(device)
    control_image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False
    )

    # Update tokenizer with new token
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    added_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    
    # Just so some initial embeddings are assigned to the new token
    # These will be replaced by inj_embedding later
    text_encoder.resize_token_embeddings(len(tokenizer)) 

    # Preparing example
    example = {}

    orig_input_ids = tokenizer(
        template,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    
    placeholder_index = torch.arange(orig_input_ids.shape[1])[(orig_input_ids == added_token_id).flatten()]
    
    orig_input_ids = orig_input_ids.repeat(args.batch_size, 1)
    orig_index = placeholder_index.repeat(args.batch_size)

    control_image_transform = partial(control_transform, 
                                      control_image_processor=control_image_processor, 
                                      width=512, 
                                      height=512)

    # Image
    if args.filelist:
        dset = Imagelist(args.filelist, transform=get_tensor_clip())
        dset_control = Imagelist(args.filelist, transform=control_image_transform)
    else:
        dset = Imagelist(Path(args.filelist_root) / f'{args.dataset}/{args.source}_train.txt', transform=get_tensor_clip())
        dset_control = Imagelist(Path(args.filelist_root) / f'{args.dataset}/{args.source}_train.txt', transform=control_image_transform)
        
    args.root_dir = Path(args.root_dir) / args.dataset / scenario

    if args.num_jobs > 1:
        dset.mode_self = False
        curr_idxs = np.array_split(np.arange(len(dset)), args.num_jobs)[args.job_idx]
        dset = SubsetWIdx(dset, curr_idxs)
        dset.imgs = [dset.dataset.imgs[i] for i in curr_idxs]
        dset.labels = dset.dataset.labels[curr_idxs]
        
    loader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)
    
    loader_control = torch.utils.data.DataLoader(
        dset_control, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    for i, (batch, batch_control) in enumerate(zip(loader, loader_control)):
        example["pixel_values_clip"] = batch[0]
        example["pixel_values"] = copy.deepcopy(example["pixel_values_clip"])
        example["index"] = orig_index[:len(batch[0])]
        example["input_ids"] = orig_input_ids[:len(batch[0])]
        example["pixel_values"] = example["pixel_values"].to(device)
        example["pixel_values_clip"] = example["pixel_values_clip"].to(device).half()
        example["input_ids"] = example["input_ids"].to(device)
        example["index"] = example["index"].to(device).long()
        example["control_image"] = batch_control[0].to(device=device, dtype=controlnet.dtype)
        
        ret_imgs = validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, controlnet, example["pixel_values_clip"].device, 5, token_index=token_index, seed=args.seed)
        
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