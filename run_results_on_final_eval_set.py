CUDA_VISIBLE_DEVICES = 2
import diffusers  # 0.12.1
import math
import fire
import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import sys
import time
import torch
import json, os, random
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms
import pdb
import imageio
from tqdm import tqdm
import cv2
from PIL import Image
import argparse 
import matplotlib.pyplot as plt
torch_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'h w c-> h w c')),
])

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, all_cond_images_and_handmasks):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            input_im_encoded = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            c = model.cc_projection(input_im_encoded)
            cond = {}
            cond['c_crossattn'] = [c]
            all_cond_images_and_handmasks_encoded = torch.zeros(1,2*int((config.model.params.unet_config.params.in_channels-8)/4/2)*4 + 4, 32, 32) #max 5 frames of cond, with 1 image and 1 handmask per frame. 4 channels per image once encoded
            for idx in range(int(2*(config.model.params.unet_config.params.in_channels-8)/4/2) + 1):
                if all_cond_images_and_handmasks[idx*3:idx*3+3,:,:].to(device).all() != 0:
                    all_cond_images_and_handmasks_encoded[:,idx*4:idx*4+4, :, :] = model.encode_first_stage(all_cond_images_and_handmasks[None,idx*3:idx*3+3,:,:].to(device)).mode().detach()
                else:
                    all_cond_images_and_handmasks_encoded[:,idx*4:idx*4+4, :, :] = torch.zeros(1, 4, 32, 32)
            
            c_concat = all_cond_images_and_handmasks_encoded.to(device)
            cond['c_concat'] = [c_concat.repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, config.model.params.unet_config.params.in_channels-4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            x_samples_ddim = model.decode_first_stage(samples_ddim)

            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def resize_image(path, target_width, target_height):
    img = plt.imread(path)
    if len(img.shape)==2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # Calculate aspect ratios
    original_height, original_width = img.shape[0:2]
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    # Crop to correct aspect ratio
    if original_aspect > target_aspect:
        # Crop width
        new_width = int(original_height * target_aspect)
        start = (original_width - new_width) // 2
        cropped_img = img[:, start+50: start + new_width+50, :]
    else:
        # Crop height
        new_height = int(original_width / target_aspect)
        start = (original_height - new_height) // 2
        cropped_img = img[start: start + new_height, :, :]

    # Resize video
    resized_img = cv2.resize(cropped_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    resized_img = resized_img / 255.0

    return resized_img

def get_samples(root_dir, xid, save_path, file_list, target_width=256, target_height=256):
    save_path = os.path.join(save_path,xid)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    target_im = resize_image(os.path.join(root_dir, 'rawframes', xid, file_list[-1]), target_width, target_height)
    target_im=(target_im*255.).astype(np.uint8)

    plot_images = []
    plot_handmasks = []
    all_cond_images_and_handmasks = torch.zeros((3*2*int((config.model.params.unet_config.params.in_channels-8)/4/2) + 3,256,256)) #last +3 is for the target handmask
    for idx in range(len(file_list)-1):
        hand_masks0 = resize_image(os.path.join(root_dir, 'handmasks', xid, file_list[idx]), target_width, target_height) #path to handmask of this videoid+image_frame
        all_cond_images_and_handmasks[idx*6:idx*6+3,:,:] = torch_transforms(hand_masks0).unsqueeze(0).to(device)

        input_im = resize_image(os.path.join(root_dir, 'rawframes', xid, file_list[idx]), target_width, target_height) #path to rgb image of this videoid+image_frame
        all_cond_images_and_handmasks[idx*6+3:idx*6+6,:,:] = torch_transforms(input_im).unsqueeze(0).to(device)

        hand_masks0=(hand_masks0*255.).astype(np.uint8)
        plot_handmasks.extend([hand_masks0,hand_masks0,hand_masks0,hand_masks0,hand_masks0])
        input_im=(input_im*255.).astype(np.uint8)
        plot_images.extend([input_im,input_im,input_im,input_im,input_im])

    final_hm = resize_image(os.path.join(root_dir, 'handmasks', xid, file_list[-1]), target_width, target_height) #path to target handmask of this videoid+image_frame
    all_cond_images_and_handmasks[-3:,:,:] = torch_transforms(final_hm).unsqueeze(0).to(device)

    final_hm=(final_hm*255.).astype(np.uint8)
    plot_handmasks.extend([final_hm,final_hm,final_hm,final_hm,final_hm])
    imageio.mimsave(f'{save_path}/{file_list[0]}_{file_list[-1]}_handmasks.gif', plot_handmasks)
    
    plot_inputs_target = plot_images+[target_im, target_im, target_im, target_im, target_im]
    imageio.mimsave(f'{save_path}/{file_list[0]}_{file_list[-1]}_input_target.gif', plot_inputs_target)

    sampler = DDIMSampler(models['turncam'])
    x_samples_ddim = sample_model(torch_transforms(input_im).unsqueeze(0).to(device), models['turncam'], sampler, 'fp32', target_height, target_width,
                                    50, 1, 1.5, 1.0, all_cond_images_and_handmasks)
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    for i in range(len(output_ims)):
        samples = plot_images + [output_ims[i], output_ims[i], output_ims[i], output_ims[i], output_ims[i]]
        imageio.mimsave(f'{save_path}/{file_list[0]}_{file_list[-1]}_sample{i}_gs1.5.gif', samples)
    
    output_ims[0].save(f'{save_path}/{file_list[0]}_{file_list[-1]}_sample{i}_gs1.5.png')





parser = argparse.ArgumentParser()
args = parser.parse_args()

device = 'cuda'
config = '/proj/vondrick/www/sruthi/coshand/assets/coshandrelease_config.yaml'
ckpt='/proj/vondrick/www/sruthi/coshand/assets/coshandrelease.ckpt'

sspp = './evaluation_results' #set path for output of results
if not os.path.exists(sspp):
    print('CREATING DIRECTORY')
    os.makedirs(sspp)


config = OmegaConf.load(config)
models = dict()
print('Instantiating LatentDiffusion...')
models['turncam'] = load_model_from_config(config, ckpt, device=device)
rr = "/proj/vondrick3/datasets/FullSSv2/data/" #path to SSV2 folder

final_eval_set = json.load(open('/proj/vondrick3/sruthi/sairam_evaluation/final_eval_set.json')) # json file where key=video id and value=before and after frame ids. ex below:
'''
{
    "20376": ["img_00049.jpg", "img_00052.jpg"], 
    "71955": ["img_00003.jpg", "img_00006.jpg"],
    "video_id": ["before_image.jpg", "after_image.jpg"]...
}
'''

print(len(final_eval_set))

#iterate through each video id in the json file 
for xid in tqdm(final_eval_set):
    value = [final_eval_set[xid][-2],final_eval_set[xid][-1]]
    get_samples(rr,xid,sspp, value, target_height=256, target_width=256)
    print('processed', xid, value)
