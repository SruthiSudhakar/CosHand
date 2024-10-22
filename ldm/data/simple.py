from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
import pdb
import json
import time
from tqdm import tqdm
from lovely_numpy import lo

class SomethingSomethingDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, labels_dir, batch_size, train=None, validation=None, \
                 test=None, num_workers=4, max_number_of_conditioning_frames=5, \
                 frame_width =256, frame_height=256, frame_stride=3, one_ex=False, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_number_of_conditioning_frames = max_number_of_conditioning_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_stride = frame_stride
        self.one_ex = one_ex

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

    def train_dataloader(self):

        dataset = SomethingSomethingData(root_dir=self.root_dir, labels_dir=self.labels_dir,\
             validation=False, test_set=False, max_number_of_conditioning_frames=self.max_number_of_conditioning_frames,\
             frame_height=self.frame_height, frame_width=self.frame_width, frame_stride=self.frame_stride, one_ex=self.one_ex)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = SomethingSomethingData(root_dir=self.root_dir, labels_dir=self.labels_dir,\
             validation=True, test_set=False, max_number_of_conditioning_frames=self.max_number_of_conditioning_frames,\
             frame_height=self.frame_height, frame_width=self.frame_width, frame_stride=self.frame_stride, one_ex=self.one_ex)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(SomethingSomethingData(root_dir=self.root_dir, labels_dir=self.labels_dir, \
             validation=False, test_set=True, max_number_of_conditioning_frames=self.max_number_of_conditioning_frames,\
             frame_height=self.frame_height, frame_width=self.frame_width, frame_stride=self.frame_stride, one_ex=self.one_ex),\
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

def resize_transform(img, target_height, target_width):
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
        cropped_img = img[:, start: start + new_width, :]
    else:
        # Crop height
        new_height = int(original_width / target_aspect)
        start = (original_height - new_height) // 2
        cropped_img = img[start: start + new_height, :, :]

    # Resize video
    resized_img = cv2.resize(cropped_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    # print(lo(resized_img))
    resized_img = resized_img / 255.0
    # print(lo(resized_img))

    return resized_img

class SomethingSomethingData(Dataset):
    def __init__(self,
        root_dir='/local/vondrick/sruthi/FullSSv2/rawframes/rawframes',
        labels_dir='',
        postprocess=None,
        return_paths=False,
        validation=False,
        test_set=False,
        max_number_of_conditioning_frames: int=5,
        frame_width: int = 256,
        frame_height: int = 256,
        frame_stride: int = 3,
        one_ex=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        start_time = time.time()
        
        self.root_dir=root_dir
        if 'proj' in root_dir:
            self.rawframes_dir = Path(f'{root_dir}/rawframes')
            self.handmasks_dir = Path(f'{root_dir}/handmasks')
        else:
            self.rawframes_dir = Path(f'{root_dir}/rawframes/rawframes')
            self.handmasks_dir = Path(f'{root_dir}/handmasks/handmasks')
            
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_stride = frame_stride

        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.max_number_of_conditioning_frames = max_number_of_conditioning_frames
        self.torch_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c')),
        ])

        self.data = []
        if validation:
            data = json.load(open(f"{root_dir}/validation_filterd_byenoughhandmask_and_pretending_with_object_bboxes.json"))
        elif test_set:
            all_data = json.load(open(f"{root_dir}/validation_filterd_byenoughhandmask_and_pretending_with_object_bboxes.json"))
            final_eval_set = json.load(open('/proj/vondrick3/sruthi/sairam_evaluation/final_eval_set.json'))
            for datapt in all_data:
                if datapt['id'] in final_eval_set:
                    new_entry = datapt.copy()      
                    if max_number_of_conditioning_frames<=1:
                        new_entry['frames_list'] = [final_eval_set[datapt['id']][0],final_eval_set[datapt['id']][3]]
                    elif max_number_of_conditioning_frames==5:              
                        new_entry['frames_list'] = final_eval_set[datapt['id']]
                    else:
                        pdb.set_trace()
                        raise Exception("which frames do you want to use?", max_number_of_conditioning_frames)
                    self.data.append(new_entry)
            if one_ex:
                self.data=self.data[:6]

            print(f"============= length of dataset {len(self.data)} =============")
            print("data prep time:", time.time() - start_time)
            return
        else:
            data = json.load(open(f"{root_dir}/train_filterd_byenoughhandmask_and_pretending_with_object_bboxes.json"))            
        print(f"============= number of videos: {len(data)} =============")
        if one_ex:
            data=[data[7]]

        max_number_of_conditioning_frames_for_frames_list=max_number_of_conditioning_frames
        if max_number_of_conditioning_frames_for_frames_list == 0:
            max_number_of_conditioning_frames_for_frames_list=1 #this is so that we get 2 frames for unconditional prediction so that we have the start frame to make the future prediction. bc you always have a start frame so you know the context of the scene is
        for entry in data:
            entry['frames_list'].sort()
            for frame_indx in range(0,len(entry['frames_list'])-max_number_of_conditioning_frames_for_frames_list*self.frame_stride):
                new_entry = entry.copy()                    
                new_entry['frames_list'] = entry['frames_list'][frame_indx:frame_indx+max_number_of_conditioning_frames_for_frames_list*self.frame_stride+1:self.frame_stride]
                # new_entry['frames_list'][0] = entry['frames_list'][random.randrange(0,frame_indx+1,self.frame_stride)] #replcaed 1st frame with a history frame
                # new_entry['target_objects'] = entry['objects'][new_entry['frames_list'][-1]]
                self.data.append(new_entry)

        print(f"============= length of dataset {len(self.data)} =============")
        print("data prep time:", time.time() - start_time)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = {}
        if self.return_paths:
            data["path"] = str(os.path.join(self.rawframes_dir, self.data[index]['id']))

        if self.max_number_of_conditioning_frames != 0 :
            # print('LOTS OF FRAMES', self.max_number_of_conditioning_frames)
            all_cond_images_and_handmasks = torch.zeros((self.frame_height,self.frame_width,3*2*self.max_number_of_conditioning_frames +3)) #last +3 is for the target handmask
            for idx in range(len(self.data[index]['frames_list'])-1):
                # print('ONE', lo(resize_transform(plt.imread(os.path.join(self.handmasks_dir, self.data[index]['id'], self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width)))
                # print('TWO', lo(self.torch_transforms(resize_transform(plt.imread(os.path.join(self.handmasks_dir, self.data[index]['id'], self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width)).numpy()))
                # print('THREE', lo((self.torch_transforms((resize_transform(plt.imread(os.path.join(self.handmasks_dir, self.data[index]['id'], self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width))).numpy()+1)/2))
                all_cond_images_and_handmasks[:,:,(idx)*6:(idx)*6+3] = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.handmasks_dir, self.data[index]['id'], self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width))
                # plt.imsave(f'contextmask{idx}.png', (self.torch_transforms((resize_transform(plt.imread(os.path.join(self.handmasks_dir, self.data[index]['id'], self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width))).numpy()+1)/2)
                all_cond_images_and_handmasks[:,:,(idx)*6+3:(idx)*6+6] = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.rawframes_dir, self.data[index]['id'], self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width))
                # plt.imsave(f'contextimg{idx}.png', (self.torch_transforms((resize_transform(plt.imread(os.path.join(self.rawframes_dir, self.data[index]['id'], self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width))).numpy()+1)/2)
            all_cond_images_and_handmasks[:,:,-3:] = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.handmasks_dir, self.data[index]['id'], self.data[index]['frames_list'][-1])), self.frame_height, self.frame_width))

            cond_im = all_cond_images_and_handmasks[:,:,3:6]
            # plt.imsave('condimg.png', ((cond_im.numpy())+1)/2)
        else:
            # print('0 COND FRAMES', self.data[index]['frames_list'])
            all_cond_images_and_handmasks = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.rawframes_dir, self.data[index]['id'], self.data[index]['frames_list'][0])), self.frame_height, self.frame_width))
            cond_im = all_cond_images_and_handmasks
            # plt.imsave('condimg.png', ((cond_im.numpy())+1)/2)

        target_im = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.rawframes_dir, self.data[index]['id'], self.data[index]['frames_list'][-1])), self.frame_height, self.frame_width))
        # plt.imsave('targetim.png', ((target_im.numpy())+1)/2)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["all_cond_images_and_handmasks"] = all_cond_images_and_handmasks
        data["max_number_of_conditioning_frames"] = self.max_number_of_conditioning_frames
        data["caption"] = self.data[index]['label']

        if self.postprocess is not None:
            data = self.postprocess(data)
        return data























class BerkeleyBridgeDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, labels_dir, batch_size, train=None, validation=None, \
                 test=None, num_workers=4, max_number_of_conditioning_frames=5, \
                 frame_width =256, frame_height=256, frame_stride=3, one_ex=False, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_number_of_conditioning_frames = max_number_of_conditioning_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_stride = frame_stride
        self.one_ex = one_ex

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

    def train_dataloader(self):

        dataset = BerkeleyBridgeData(root_dir=self.root_dir, labels_dir=self.labels_dir,\
             validation=False, test_set=False, max_number_of_conditioning_frames=self.max_number_of_conditioning_frames,\
             frame_height=self.frame_height, frame_width=self.frame_width, frame_stride=self.frame_stride, one_ex=self.one_ex)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = BerkeleyBridgeData(root_dir=self.root_dir, labels_dir=self.labels_dir,\
             validation=True, test_set=False, max_number_of_conditioning_frames=self.max_number_of_conditioning_frames,\
             frame_height=self.frame_height, frame_width=self.frame_width, frame_stride=self.frame_stride, one_ex=self.one_ex)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(BerkeleyBridgeData(root_dir=self.root_dir, labels_dir=self.labels_dir, \
             validation=False, test_set=True, max_number_of_conditioning_frames=self.max_number_of_conditioning_frames,\
             frame_height=self.frame_height, frame_width=self.frame_width, frame_stride=self.frame_stride, one_ex=self.one_ex),\
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class BerkeleyBridgeData(Dataset):
    def __init__(self,
        root_dir='./',
        labels_dir='',
        postprocess=None,
        return_paths=False,
        validation=False,
        test_set=False,
        max_number_of_conditioning_frames: int=5,
        frame_width: int = 256,
        frame_height: int = 256,
        frame_stride: int = 3,
        one_ex=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        start_time = time.time()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_stride = frame_stride

        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.max_number_of_conditioning_frames = max_number_of_conditioning_frames

        self.data = []
        if validation:
            data = json.load(open(labels_dir))
        elif test_set:
            data = json.load(open(labels_dir))
        else:
            data = json.load(open(labels_dir))
        # random.shuffle(data)

        if one_ex:
            for entry in data:
                if len(entry['frames_list'])>32:
                    data=[entry]
                    print('PATH OF DATA POINT:', entry['path'])
                    break
                    
        for entry in data:
            entry['frames_list'].sort(key=self.get_number)
            for frame_indx in range(0,len(entry['frames_list'])-max_number_of_conditioning_frames*self.frame_stride):
                new_entry = entry.copy()
                new_entry['frames_list'] = entry['frames_list'][frame_indx:frame_indx+max_number_of_conditioning_frames*self.frame_stride+1:self.frame_stride]
                # new_entry['frames_list'][0] = entry['frames_list'][random.randrange(0,frame_indx+1,self.frame_stride)] #replcaed 1st frame with a history frame
                # new_entry['target_objects'] = entry['objects'][new_entry['frames_list'][-1]]
                self.data.append(new_entry)

        print(f"============= length of dataset {len(self.data)} =============")
        print("data prep time:", time.time() - start_time)
        self.torch_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c')),
        ])
    def __len__(self):
        return len(self.data)
    def get_number(self, img_path):
        return int(img_path.split('/')[-1].split('_')[1][:-4])

    def __getitem__(self, index):
        data = {}

        if self.max_number_of_conditioning_frames != 0 :
            all_cond_images_and_handmasks = torch.zeros((self.frame_height,self.frame_width,3*2*self.max_number_of_conditioning_frames +3)) #last +3 is for the target handmask
            for idx in range(len(self.data[index]['frames_list'])-1):
                # print('ONE', lo(resize_transform(plt.imread(os.path.join(self.data[index]['path'],'handmasks',self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width)))
                # print('TWO', lo(self.torch_transforms(resize_transform(plt.imread(os.path.join(self.data[index]['path'],'handmasks',self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width)).numpy()))
                # print('THREE', lo((self.torch_transforms((resize_transform(plt.imread(os.path.join(self.data[index]['path'],'handmasks',self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width))).numpy()+1)/2))
                all_cond_images_and_handmasks[:,:,(idx)*6:(idx)*6+3] = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.data[index]['path'],'handmasks',self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width))
                # plt.imsave(f'contextmask{idx}.png', (self.torch_transforms((resize_transform(plt.imread(os.path.join(self.data[index]['path'],'handmasks',self.data[index]['frames_list'][isx])), self.frame_height, self.frame_width))).numpy()+1)/2)
                all_cond_images_and_handmasks[:,:,(idx)*6+3:(idx)*6+6] = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.data[index]['path'],self.data[index]['frames_list'][idx])), self.frame_height, self.frame_width))
                # plt.imsave(f'contextimg{idx}.png', (self.torch_transforms((resize_transform(plt.imread(os.path.join(self.data[index]['path'],self.data[index]['frames_list'][isx])), self.frame_height, self.frame_width))).numpy()+1)/2)
            all_cond_images_and_handmasks[:,:,-3:] = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.data[index]['path'],'handmasks',self.data[index]['frames_list'][-1])), self.frame_height, self.frame_width))
            cond_im = all_cond_images_and_handmasks[:,:,3:6]
            # plt.imsave('condimg.png', ((cond_im.numpy())+1)/2)
        else:
            all_cond_images_and_handmasks = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.data[index]['path'],self.data[index]['frames_list'][0])), self.frame_height, self.frame_width))
            cond_im = all_cond_images_and_handmasks

        target_im = self.torch_transforms(resize_transform(plt.imread(os.path.join(self.data[index]['path'],self.data[index]['frames_list'][-1])), self.frame_height, self.frame_width))
        # plt.imsave('target_im.png', ((target_im.numpy())+1)/2)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["all_cond_images_and_handmasks"] = all_cond_images_and_handmasks
        data["max_number_of_conditioning_frames"] = self.max_number_of_conditioning_frames
        # data["object_bboxes"] = mask

        if self.postprocess is not None:
            data = self.postprocess(data)
        return data