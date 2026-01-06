# adapted from https://github.com/EPFL-VILAB/omnidata

import sys
import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
from tqdm import tqdm

omnidata_path='./third_party/omnidata/omnidata_tools/torch/'
sys.path.append(omnidata_path)
from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

trans_topil = transforms.ToPILImage()
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img

def save_outputs(img_path, output_path, output_file_name):
    with torch.no_grad():
        save_path = os.path.join(output_path, f'{output_file_name}_{args.task}.png')

        img = Image.open(img_path)
        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
        [h, w] = np.array(img).shape[:2]

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            output = output.clamp(0,1)
            if w != 384:
                padding = (w- h) // 2
                output = F.interpolate(output[None], (w, w), mode='nearest').squeeze()
                output = output[padding:-padding, :]
            np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
            plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
        else:
            if w != 384:
                padding = (w- h) // 2
                output = F.interpolate(output, (w, w), mode='nearest').squeeze()
                output = output[:, padding:-padding, :]
            np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy())
            trans_topil(output).save(save_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export normal maps from Omnidata')
    parser.add_argument('--task', type=str, default='normal', help="normal or depth")
    parser.add_argument("--images_root", type=str, default='', help='images to generate depth')
    parser.add_argument("--img_res", default=None, help='target image resolution')
    args = parser.parse_args()

    assert args.images_root!='', "Please provide --images_root"

    model_root = "./third_party/omnidata/omnidata_tools/torch/pretrained_models"
    image_size = 384
    # get target task and model
    if args.task == 'normal':
        pretrained_weights_path = os.path.join(model_root, 'omnidata_dpt_normal_v2.ckpt')
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([
            transforms.Pad((0, 80, 0, 80)),
            transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(image_size),
            get_transform('rgb', image_size=None)])

    elif args.task == 'depth':
        pretrained_weights_path = os.path.join(model_root, 'omnidata_dpt_depth_v2.ckpt')
        model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([
            transforms.Pad((0, 80, 0, 80)),
            transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)])

    else:
        print("task should be one of the following: normal, depth")
        sys.exit()

    img_path = args.images_root
    if not os.path.exists(img_path):
        raise ValueError(f'The path {img_path} does not exist')

    out_path = os.path.join(f'{img_path}/../', f'omnidata_{args.task}')
    os.makedirs(out_path, exist_ok=True)

    img_list = glob.glob(img_path+'/*')
    img_list = sorted(img_list)
    for f in tqdm(img_list, desc=f"export {args.task} from Omnidata"):
        save_outputs(f, out_path, os.path.splitext(os.path.basename(f))[0])