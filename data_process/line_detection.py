
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
from utils_gradio.run_detector import run_line_detector

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
import cv2

import matplotlib.pyplot as plt
from pathlib import Path
import json

import argparse
import copy
import hashlib
import os.path as osp
from typing import Optional

from easylsd.base import show, WireframeGraph

URLS_CKPT = {
    'scalelsd': 'https://huggingface.co/cherubicxn/scalelsd/resolve/main/scalelsd-vitbase-v2-train-sa1b.pt',
    'hawpv3': 'https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv3/hawpv3-fdc5487a.pth',
    'deeplsd': 'https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_wireframe.tar',
}

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def try_download(url: Optional[str], dst_path: str) -> bool:
    """Attempt to download a file to dst_path. Returns True on success.
    Requires network connectivity. If url is None, returns False.
    """
    if url is None:
        return False
    try:
        import urllib.request  # lazy import
        ensure_dir(osp.dirname(dst_path))
        print(f'Downloading: {url} -> {dst_path}')
        urllib.request.urlretrieve(url, dst_path)
        return True
    except Exception as e:
        print(f'Warning: download failed for {url}: {e}')
        return False

def prepare_model(detector):
    # Resolve device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

    # Prepare checkpoint paths
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    scalelsd_ckpt = osp.join(ckpt_dir, os.path.basename(URLS_CKPT['scalelsd']))
    deeplsd_ckpt = osp.join(ckpt_dir, os.path.basename(URLS_CKPT['deeplsd']))
    hawpv3_ckpt = osp.join(ckpt_dir, os.path.basename(URLS_CKPT['hawpv3']))

    # Detector-specific setup
    if detector == 'scalelsd':
        # Optional auto-download
        if not osp.exists(scalelsd_ckpt):
            ok = try_download(URLS_CKPT['scalelsd'], scalelsd_ckpt)
            if not ok:
                print('Warning: ScaleLSD checkpoint not found and could not be downloaded.')
        # Import and load model
        try:
            # from scalelsd.ssl.models.detector import ScaleLSD  # type: ignore
            from easylsd.models.scalelsd import ScaleLSD  # type: ignore
        except Exception as e:
            raise ImportError('scalelsd package not found. Please install it from https://github.com/ant-research/ScaleLSD') from e
        threshold = 10.0
        ScaleLSD.junction_threshold_hm = 0.1
        ScaleLSD.num_junctions_inference = 512
        model = ScaleLSD(gray_scale=True, use_layer_scale=True).eval().to(device)
        if osp.exists(scalelsd_ckpt):
            state_dict = torch.load(scalelsd_ckpt, map_location='cpu')
            try:
                model.load_state_dict(state_dict['model_state'])
            except Exception:
                model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f'ScaleLSD checkpoint not found: {scalelsd_ckpt}. Provide --scalelsd-ckpt or enable --download with --scalelsd-ckpt-url.')
            # print(f'Note: ScaleLSD checkpoint not found at {scalelsd_ckpt}. Running uninitialized weights.')

    elif detector == 'hawpv3':
        if not osp.exists(hawpv3_ckpt):
            ok = try_download(URLS_CKPT['hawpv3'], hawpv3_ckpt)
            if not ok:
                print('Warning: HAWPv3 checkpoint not found and could not be downloaded.')
        try:
            from easylsd.models import HAWPv3
        except Exception as e:
            raise ImportError('hawp package not found. Please install submodule and package from https://github.com/cherubicXN/hawp') from e
        threshold = 0.05

        model = HAWPv3(gray_scale=True).eval().to(device)
        if not osp.exists(hawpv3_ckpt):
            raise FileNotFoundError(f'HAWPv3 checkpoint not found: {hawpv3_ckpt}. Provide --hawpv3-ckpt or enable --download with --hawpv3-ckpt-url.')
        state_dict = torch.load(hawpv3_ckpt, map_location='cpu')
        model.load_state_dict(state_dict)

    elif detector == 'lsd':
        model = cv2.createLineSegmentDetector(0)
        # model runs on CPU via OpenCV; keep device for consistency

    elif detector == 'deeplsd':
        if not osp.exists(deeplsd_ckpt):
            ok = try_download(URLS_CKPT['deeplsd'], deeplsd_ckpt)
            if not ok:
                print('Warning: DeepLSD checkpoint not found and could not be downloaded.')
        try:
            # from deeplsd.models.deeplsd import DeepLSD  # type: ignore
            from easylsd.models import DeepLSD  # type: ignore
        except Exception as e:
            raise ImportError('deeplsd package not found. Please install from https://github.com/cvg/DeepLSD') from e
        conf = {
            'sharpen': True,
            'detect_lines': True,
            'line_detection_params': {
                'merge': False,
                'optimize': False,
                'use_vps': True,
                'optimize_vps': True,
                'filtering': True,
                'grad_thresh': 3,
                'grad_nfa': True,
            }
        }
        model = DeepLSD(conf).eval().to(device)
        if not osp.exists(deeplsd_ckpt):
            raise FileNotFoundError(f'DeepLSD checkpoint not found: {deeplsd_ckpt}. Provide --deeplsd-ckpt or enable --download with --deeplsd-ckpt-url.')
        state_dict = torch.load(str(deeplsd_ckpt), map_location='cpu')
        model.load_state_dict(state_dict['model'])

    else:
        raise TypeError(f'Unknown detector <{detector}>')
    
    return model, device

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--data_path", type=str, default='', help='path of input data')
    parser.add_argument("--detector", type=str, default='lsd', help='line detector: lsd, scalelsd, hawpv3, deeplsd')
    parser.add_argument("--save_detected_images", default=False, action='store_true')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f'The input data path {data_path} does not exist.')
    detector = args.detector
    
    save_root = f'{data_path}/../{detector}'
    os.makedirs(save_root, exist_ok=True)

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

    if isinstance(image_path, list):
        img_name_list = image_path
    elif os.path.isdir(image_path):
        img_name_list = glob.glob(image_path + '/*')
    else:
        raise ValueError(f'The input data_path {image_path} should be either a list which consist of image paths or a directory which contain images.')

    model, device = prepare_model(detector)
    width, height = 512, 512
    wireframe_list = []
    threshold = 0.05

    painter = show.painters.HAWPainter()
    edge_color = 'midnightblue' # 'midnightblue'
    vertex_color = 'deeppink' # 'deeppink'
    show.Canvas.white_overlay = 0.3

    for image_path in tqdm(sorted(img_name_list)):

        image = cv2.imread(image_path,0)

        ori_shape = image.shape[:2]
        image_resized = cv2.resize(image, (width, height))
        image_t = torch.from_numpy(image_resized).float() / 255.0
        image_t = image_t[None, None].to(device)

        with torch.no_grad():
            if detector == 'scalelsd':
                meta = {
                    'width': ori_shape[1],
                    'height':ori_shape[0],
                    'filename': '',
                    'use_lsd': False,
                    'use_nms': True,
                }
                outputs, _ = model(image_t, meta)
                outputs = outputs[0]
                threshold = 10.0

            elif detector == 'hawpv3':
                meta = {
                    'width': ori_shape[1],
                    'height': ori_shape[0],
                    'filename': '',
                    'use_lsd': False,
                }
                outputs, _ = model(image_t, [meta])

            elif detector == 'deeplsd':
                inputs = {'image': torch.tensor(image, dtype=torch.float, device=device)[None, None] / 255.}
                out = model(inputs)
                pred_lines = out['lines'][0]
                lines = torch.from_numpy(pred_lines)
                juncs = torch.unique(lines.reshape(-1, 2), dim=0)
                outputs = {
                    'width': ori_shape[1],
                    'height': ori_shape[0],
                    'lines_pred': lines.reshape(-1, 4),
                    'juncs_pred': juncs,
                    'lines_score': torch.ones(lines.shape[0]),
                    'juncs_score': torch.ones(juncs.shape[0]),
                }

            elif detector == 'lsd':
                # OpenCV LSD expects uint8 grayscale image
                lsd = model.detect(image)
                if lsd is None or lsd[0] is None:
                    lines = torch.zeros((0, 4), dtype=torch.float32)
                else:
                    lines_arr = lsd[0].reshape(-1, 4)
                    lines = torch.from_numpy(lines_arr)
                juncs = torch.unique(lines.reshape(-1, 2), dim=0) if lines.numel() > 0 else torch.zeros((0, 2), dtype=torch.float32)
                outputs = {
                    'width': ori_shape[1],
                    'height': ori_shape[0],
                    'lines_pred': lines,
                    'juncs_pred': juncs,
                    'lines_score': torch.ones(lines.shape[0]),
                    'juncs_score': torch.ones(juncs.shape[0]),
                }
            else:
                raise TypeError('Please input the correct detector!')

            show.painters.HAWPainter.confidence_threshold = threshold
            indices = WireframeGraph.xyxy2indices(outputs['juncs_pred'], outputs['lines_pred'])
            wireframe = WireframeGraph(outputs['juncs_pred'], outputs['juncs_score'], indices, outputs['lines_score'], outputs['width'], outputs['height'])
            pname = Path(image_path)
            outpath = osp.join(save_root, f'{pname.stem}.json')
            with open(outpath,'w') as f:
                json.dump(wireframe.jsonize(),f)

            if args.save_detected_images:
                pname = Path(image_path)
                fig_file = osp.join(save_root, f'{pname.stem}.png')
                with show.image_canvas(image_path, fig_file=fig_file) as ax:
                    painter.draw_wireframe(ax, outputs, edge_color=edge_color, vertex_color=vertex_color)
        

