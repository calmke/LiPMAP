import os
import sys
sys.path.append('.')
# sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))

import argparse
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils.run_metric3d import extract_mono_geo_demo
from utils.run_vggt import run_vggt
from utils.misc_util import save_video, is_video_file, save_frames_from_video
from utils.run_detector import run_line_detector

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

import numpy as np
import os
import glob
from tqdm import tqdm
import open3d as o3d

import matplotlib.pyplot as plt
from pathlib import Path
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_path", type=str, default='examples/living_room/images', help='path of input data')
    parser.add_argument("-s", "--frame_step", type=int, default=10, help='sampling step of video frames')
    parser.add_argument("--depth_conf", type=float, default=2.0, help='depth confidence threshold of vggt')
    parser.add_argument("--conf_path", type=str, default='utils_demo/demo.conf', help='path of configure file')
    parser.add_argument("--detector", type=str, default='deeplsd', help='line detector: lsd, scalelsd, hawpv3, deeplsd')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f'The input data path {data_path} does not exist.')
    
    image_path = None
    if os.path.isdir(data_path):
        image_path = data_path
    else:
        if is_video_file(data_path):
            image_path = os.path.join(*(data_path.split('/')[:-1]), 'images')
            save_frames_from_video(data_path, image_path, args.frame_step)
        else:
            raise ValueError(f'The input file {data_path} is not a video.')
    assert image_path is not None, f"Can not find images or videos from {data_path}."

    root_path = os.path.join(*(image_path.split('/')[:-1]))
    precomputed_data_path = os.path.join('/', root_path, 'data.pth')
    

    # data = run_vggt(image_path, depth_conf_thresh=args.depth_conf)
    data = run_vggt(image_path, step=args.frame_step, depth_conf_thresh=args.depth_conf)
    # run metric3dv2
    _, normal_maps_list = extract_mono_geo_demo(data['color'], data['intrinsics'])
    data['normal'] = normal_maps_list
    # torch.save(data, precomputed_data_path)
    data = run_line_detector(data, args.detector)

    K = data['intrinsics'][0]
    K_full = np.eye(4)
    K_full[:3, :3] = K

    image_paths = data['image_paths']
    n_images = len(image_paths)

    # load camera
    intrinsics_all = [torch.from_numpy(intrinsic).cuda() for intrinsic in data['intrinsics']]
    poses_all = [torch.from_numpy(extrinsic).cuda() for extrinsic in data['extrinsics']]
    
    # load rgbs
    rgbs = data['color']
    depths = data['depth']
    normals = data['normal']
    wireframes = data['wireframes']

    print("start to save data ......")
    # posr-processing
    vggt_dir = os.path.join(os.path.join(root_path, 'vggt'))
    if not os.path.exists(vggt_dir):
        os.makedirs(vggt_dir, exist_ok=True)
        image_dir = os.path.join(vggt_dir, 'images')
        os.makedirs(image_dir, exist_ok=True)
        depth_dir = os.path.join(vggt_dir, 'vggt_depth')
        os.makedirs(depth_dir, exist_ok=True)
        normal_dir = os.path.join(vggt_dir, 'vggt_normal')
        os.makedirs(normal_dir, exist_ok=True)
        pose_dir = os.path.join(vggt_dir, 'poses')
        os.makedirs(pose_dir, exist_ok=True)
        intrinsic_path = os.path.join(vggt_dir, 'intrinsics.txt')
        np.savetxt(intrinsic_path, K_full)
        wireframe_dir = os.path.join(vggt_dir, args.detector)
        os.makedirs(wireframe_dir, exist_ok=True)
        for i in tqdm(range(n_images), desc='processing data'):  
            # iname = image_paths[i].split('/')[-1]
            # idx = iname.split('.')[0]
            idx = i

            rgb = rgbs[i]
            plt.imsave(os.path.join(image_dir, f'{idx:06d}.jpg'), rgb)

            pose = poses_all[i].cpu().numpy()
            np.savetxt(os.path.join(pose_dir, f'{idx:06d}.txt'), pose)

            # resize depth
            depth = depths[i]
            # depth = cv2.resize(depth, [1024, 768], interpolation=cv2.INTER_NEAREST)
            np.save(os.path.join(depth_dir, f'{idx:06d}_depth.npy'), depth)
            plt.imsave(os.path.join(depth_dir, f'{idx:06d}_depth.png'), depth, cmap='viridis')

            # resize normal
            normal = normals[i]
            # normal_resized = cv2.resize(normal.transpose(1,2,0), [1024, 768], interpolation=cv2.INTER_LINEAR)
            # norm = np.linalg.norm(normal, axis=2, keepdims=True)
            # normal_resized = np.divide(normal, norm, where=(norm > 1e-10))
            np.save(os.path.join(normal_dir, f'{idx:06d}_normal.npy'), normal)
            plt.imsave(os.path.join(normal_dir, f'{idx:06d}_normal.png'), normal.transpose(1,2,0), cmap='viridis')

            wireframe = wireframes[i]
            outpath = os.path.join(wireframe_dir, f'{idx:06d}.json')
            with open(outpath,'w') as f:
                json.dump(wireframe.jsonize(),f)


    