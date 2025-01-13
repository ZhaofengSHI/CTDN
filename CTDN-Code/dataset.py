import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import numpy as np

import cv2
from PIL import Image
from torchvision import transforms

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip

BICUBIC = Image.BICUBIC
NEAREST = Image.NEAREST

import torch.nn.functional as F


def _transform_all(h, w, crop_scale):
    return Compose([
        transforms.RandomResizedCrop((h,w), scale=(crop_scale,1.0), interpolation=NEAREST)
    ])

def _transform_img():
    return Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class Dataset_ft(Dataset):

    def __init__(self, data_root, data_list, class_names, crop_scale, dataset, w=640, h=320):
    # def __init__(self, data_root, data_list, class_names, crop_scale, dataset, w=448, h=448):
        super(Dataset_ft, self).__init__()
        self.list = data_list
        self.data_root = data_root
        self.img_dir = 'JPEGImages'
        self.label_dir = 'SegmentationClassgray'
        if dataset == 'egohos_hands':
            self.label_dir = 'SegmentationClassgray_twohands'
        if dataset == 'egohos_handobject':
            self.label_dir = 'SegmentationClassgray_handobject'

        self.h = h
        self.w = w
        self.crop_scale = crop_scale

        self.class_names = class_names


    def __getitem__(self, index):

        # transforms
        transform_all = _transform_all(self.h,self.w,self.crop_scale)
        transform_img = _transform_img()

        # image
        img_name = self.list[index]
        img_path = os.path.join(self.data_root,self.img_dir,img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        # label
        label_name = img_name[:-4]+'.png'
        label_path = os.path.join(self.data_root,self.label_dir,label_name)
        # label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = np.array(Image.open(label_path))

        # image label concat and random crop togeter
        label = np.expand_dims(label, axis=2)
        image_label = np.append(image,label,axis=2)
        image_label = transform_all(Image.fromarray(image_label))
        image = transform_img(Image.fromarray(np.array(image_label)[:,:,:3]))

        label = np.array(image_label)[:,:,3]
        label = np.asarray(label, dtype=np.int32)

        # text
        label_id_list = []
        for ii in range(len(self.class_names)):
            label_id = ii + 1 # except bkground
            if np.where(label==label_id)[0].size > 0:
                label_id_list.append(label_id-1)
        
        if len(label_id_list) != 0:
            label_id_list = torch.tensor(label_id_list)
            multi_hot = torch.zeros(len(self.class_names)).scatter_(0, label_id_list, 1)
        else:
            multi_hot = torch.zeros(len(self.class_names))


        return image, multi_hot


    def __len__(self):
        return len(self.list)

