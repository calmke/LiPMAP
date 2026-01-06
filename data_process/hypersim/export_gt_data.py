import sys
sys.path.append('.')

import numpy as np
from utils.dataIO_util import glob_data, load_rgb
from utils import model_util
import os

import shutil
from pathlib import Path
import cv2
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torchvision import transforms

from data_process.hypersim.hypersim_dataset import HypersimDataset


parser = ArgumentParser("process rgb images and poses")
parser.add_argument("--data_root", type=str, default="./data/Hypersim", help='root path to scannet dataset')
parser.add_argument("--save_root", type=str, default="./data/general_data/Hypersim", help='path to save processed results')
parser.add_argument("--scene_id", type=str, default='ai_001_001', help='name of one scene')
args = parser.parse_args()

data_path = os.path.join(args.data_root, args.scene_id)
save_path = os.path.join(args.save_root, args.scene_id)
os.makedirs(save_path, exist_ok=True)

dataset = HypersimDataset(args.data_root, args.scene_id)
trans_topil = transforms.ToPILImage()

# intrinsics
np.savetxt(f'{save_path}/intrinsics.txt', dataset.K_full)

image_root = f'{save_path}/images'
os.makedirs(image_root, exist_ok=True)
normal_g_root = f'{save_path}/gt_global_normal'
os.makedirs(normal_g_root, exist_ok=True)
pose_root = f'{save_path}/poses'
os.makedirs(pose_root, exist_ok=True)
depth_root = f'{save_path}/gt_depth'
os.makedirs(depth_root, exist_ok=True)

for i in tqdm(range(len(dataset.image_paths))):
    iname = f'{i:06d}'
    # image
    shutil.copy2(dataset.image_paths[i], f'{image_root}/{iname}.jpg')
    # pose
    np.savetxt(f'{pose_root}/{iname}.txt', dataset.pose_all[i].cpu().numpy())
    # global normal
    normal_g = torch.tensor(dataset.normal_gs[i]).permute(2,0,1)
    np.save(f'{normal_g_root}/{iname}.npy', normal_g)
    shutil.copy2(dataset.normal_image_paths[i], f'{normal_g_root}/{iname}.png')
    # depth
    depth = dataset.mono_depths[i]
    np.save(f'{depth_root}/{iname}.npy', depth)
    shutil.copy2(dataset.depth_image_paths[i], f'{depth_root}/{iname}.png')

    
