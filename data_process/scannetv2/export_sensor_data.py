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

parser = ArgumentParser("process rgb images and poses")
parser.add_argument("--data_root", type=str, default="./data/ScanNetV2/scans", help='root path to scannet dataset')
parser.add_argument("--save_root", type=str, default="./data/general_data/ScanNetV2", help='path to save processed results')
parser.add_argument("--scene_id", type=str, required=True, default='', help='name of one scene')
parser.add_argument("--frame_step", type=int, default=10, help='frame interval for data sampling')
args = parser.parse_args()

data_path = os.path.join(args.data_root, args.scene_id)
save_path = os.path.join(args.save_root, args.scene_id)
os.makedirs(save_path, exist_ok=True)
step = args.frame_step

# intrinsics
intrinsics = np.loadtxt(os.path.join(data_path, "intrinsic/intrinsic_depth.txt"))
np.savetxt(os.path.join(save_path, "intrinsics.txt"), intrinsics)

image_path = sorted(glob_data(os.path.join(data_path, "sensor_data", "*.color.jpg")))[::step]
depth_path = sorted(glob_data(os.path.join(data_path, "sensor_data", "*.depth.png")))[::step]
pose_path = sorted(glob_data(os.path.join(data_path, "sensor_data", "*.pose.txt")))[::step]

dest_image_root = os.path.join(save_path, 'images')
os.makedirs(dest_image_root, exist_ok=True)
dest_depth_root = os.path.join(save_path, 'sensor_depth')
os.makedirs(dest_depth_root, exist_ok=True)
dest_pose_root = os.path.join(save_path, 'poses')
os.makedirs(dest_pose_root, exist_ok=True)

for i in tqdm(range(len(image_path))):
    image_file = image_path[i]
    depth_file = depth_path[i]
    pose_file = pose_path[i]

    iname = f'{i:06d}'

    # valid pose
    data = np.loadtxt(pose_file)
    if np.isfinite(data).all():
        # pose
        np.savetxt(os.path.join(dest_pose_root, f'{iname}.txt'), data)
        # image
        image = cv2.imread(image_file)
        cv2.imwrite(os.path.join(dest_image_root, f'{iname}.png'), cv2.resize(image, (640, 480)))
        # sensor depth
        shutil.copy2(depth_file, os.path.join(dest_depth_root, f'{iname}.png'))

    else:
        continue