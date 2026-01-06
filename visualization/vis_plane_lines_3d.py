import sys
sys.path.append('.')
import argparse
import os
from utils.misc_util import fix_seeds
from utils.conf_util import get_class
from utils.model_util import project2D
from utils.hawp_util import WireframeGraph
from utils.dataIO_util import glob_data, load_rgb
from utils import model_util
from utils import mesh_util
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
import numpy as np
import tqdm
from pathlib import Path

import torch
from run.network import RecWrapper
import open3d as o3d 

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=str, default='exps_results', help='the exps path')
    parser.add_argument('--epoch', type=int, default=6000)
    parser.add_argument('--ckpt', default='latest', type=str, help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--dist_th', type=float, default=5, help='threshold for select lines by distance')
    parser.add_argument('--theta_th', type=float, default=0.01, help='threshold for select lines by theta')
    parser.add_argument('--save3d', default=True, action="store_false", help='save 3d results')
    parser.add_argument('--ext', default='npy', type=str, help='the extension of the output file', choices=['json', 'npy'])
    parser.add_argument('--plot2d', default=False, action="store_true", help='plot 2d results')
    parser.add_argument('--nv', default=0, type=int, help='n views for lines tracking')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    fix_seeds()
    args = parse_args()

    conf = os.path.join(args.workdir, 'train.conf')
    cfg = ConfigFactory.parse_file(conf)

    print('Loading data...')
    cfg.put('dataset.inference', True)
    dataset = get_class(cfg.get_string('train.dataset_class'))(**cfg.get_config('dataset'))

    net = RecWrapper(cfg)
    model = net.cuda()

    ckpt_path = os.path.join(args.workdir, 'checkpoints/Parameters', f'{args.epoch}.pth')
    print(f'loading model from {ckpt_path}')
    saved_model_state = torch.load(ckpt_path)
    plane_num = saved_model_state["model_state_dict"]['planarSplat._plane_center'].shape[0]
    model.planarSplat.initialize_as_zero(plane_num)
    model.load_state_dict(saved_model_state["model_state_dict"])
    
    final_lines_3d = model.forward(
        dataset,
        distance_threshold=args.dist_th, 
        theta_threshold=args.theta_th
    )
    output_path = os.path.join(args.workdir, f'line_plots/final_lines3d_{args.epoch}.ply')
    mesh_util.lines3d2ply(final_lines_3d, output_path)
    mesh_util.vis_plane_rays_juncs_lines(lines=final_lines_3d)

    final_lines_3d = model.forward(
        dataset,
        distance_threshold=args.dist_th, 
        theta_threshold=args.theta_th,
        merge='global'
    )
    output_path = os.path.join(args.workdir, f'line_plots/final_lines3d_merge.ply')
    mesh_util.lines3d2ply(final_lines_3d, output_path)
    mesh_util.vis_plane_rays_juncs_lines(lines=final_lines_3d)

    # plane_corners = model.planarSplat.get_plane_vertex()
    # mesh_util.vis_plane_rays_juncs_lines(plane_corners=plane_corners, lines=final_lines_3d)
