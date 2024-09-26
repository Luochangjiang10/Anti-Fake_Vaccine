from __future__ import print_function
import argparse
import os
from math import log10
from PIL import Image
import random
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image,save_image_cv2
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from material.models.generators import ResnetGenerator, weights_init
import torch.backends.cudnn as cudnn
import math
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as TF
import cv2



def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def normalize_and_scale(delta_im,bs,strenght):
    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5 # now 0..1

    # normalize image color channels
    for c in range(15):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean[0]) / std[0]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = bs
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(15):
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = strenght/(255.0*stddev_arr[ci])
            
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im

def BGR2RGB(im):
        im1 = im.clone()
        im[:,0, :, :] = im1[:,2, :, :]
        im[:,2, :, :] = im1[:,0, :, :]
        return im
cudnn.benchmark = True
torch.cuda.manual_seed(123)


gpulist = [0]
n_gpu = len(gpulist)
mean=[0.5, 0.5, 0.5]
std=[0.5, 0.5, 0.5]
stddev_arr = [0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.229, 0.224, 0.225]
stddev_arr= [i+0.3 for i in stddev_arr]
transform1 = []
transform1.append(transforms.Resize(256))
transform1.append(transforms.ToTensor())
transform1.append(transforms.Normalize(mean=mean, std=std))
transform = transforms.Compose(transform1)

transform2 = []
transform2.append(transforms.Resize(256))
transform2.append(transforms.ToTensor())
transform2 = transforms.Compose(transform2)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=gpulist)
netG.to(device)
netG.apply(weights_init)
netG.load_state_dict(torch.load('./weight/004067.pth', map_location=lambda storage, loc: storage))
netG.eval()
optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0.5, 0.999))

mag_in = 5

source_path = "./data/imgs/004067.jpg"          #待批量重命名的文件夹 
mask_weight_path = './data/mask_weight/004067.jpg'
mask_path = './data/masks/004067'
save_path = '/exdata/LCj/Anti-Fake_Vaccine/noise/004067.jpg'
 
image_source = cv2.imread(source_path)
image_source = Image.fromarray(image_source)
image_source = transform(image_source)
image_source = image_source.to(device)
image_source= image_source.unsqueeze(0)

image_mask_weight = cv2.imread(mask_weight_path)
image_mask_weight = Image.fromarray(image_mask_weight)
image_mask_weight= transform2(image_mask_weight)
image_mask_weight = image_mask_weight.unsqueeze(0)

image_mask = []
for x1 in['eyebrows','eyes','mouth','nose','skin']:
    image_m = cv2.imread(mask_path+x1+'.jpg')
    image_m = Image.fromarray(image_m)
    image_m = transform2(image_m)
    image_m  = image_m.unsqueeze(0)
    image_m = image_m[0:1,:,:,:]
    image_m[image_m>0.5]=1
    image_m[image_m<0.5]=0
    image_mask.append(image_m.to(device))

image_source = image_source.to(device)
delta_im = netG(image_source) 
bs = image_source.size(0)
delta_im = normalize_and_scale(delta_im,bs,mag_in)
delta_im_list = []
for i in range(5):
    delta_im_part = delta_im[:,i*3:(i+1)*3,:]*image_mask[i]
    delta_im_list.append(delta_im_part)
all_noise = torch.zeros(1,3,256,256).cuda()
for l in range(len(delta_im_list)):
    all_noise = all_noise+delta_im_list[l]     
image_mask_weight[image_mask_weight>0]=1
image_mask_weight[image_mask_weight<=0]=0
all_noise = all_noise*image_mask_weight.cuda()
     
recons = torch.add(image_source.to(device), all_noise.to(device))
                
for cii in range(3):
    recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image_source[:,cii,:,:].min(), image_source[:,cii,:,:].max())
recons = torch.flip( recons, dims=[1])
save_image(denorm(recons.data.cpu()), save_path,nrow=1, padding=0)                
                