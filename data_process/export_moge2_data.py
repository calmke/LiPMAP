import os
import sys
import cv2
import glob
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import utils3d
from torchvision import transforms
import argparse

# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

trans_topil = transforms.ToPILImage()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export normal/depth maps from MoGe-2')
    parser.add_argument("--images_root", type=str, default='', help='images to generate depth')
    args = parser.parse_args()
    assert args.images_root!='', "Please provide --images_root"

    image_path = args.images_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model from huggingface hub (or load from local).
    moge_ckpt_path = 'checkpoints/moge2-vitl-normal.pt'
    model = MoGeModel.from_pretrained(moge_ckpt_path).to(device)   
    model.eval()                          

    depth_dir = os.path.join(os.path.join(f'{image_path}/../', f'moge2_depth'))
    os.makedirs(depth_dir, exist_ok=True)
    normal_dir = os.path.join(os.path.join(f'{image_path}/../', f'moge2_normal'))
    os.makedirs(normal_dir, exist_ok=True)

    img_name_list = glob.glob(image_path + '/*')
    image_paths = sorted(img_name_list)
    for i in tqdm.tqdm(range(len(image_paths)), desc=f"export normal/depth maps from MoGe-2"):
        image_path = image_paths[i]
        # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
        input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)                       
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

        # Infer 
        output = model.infer(input_image)

        depth_pred = output['depth'].squeeze().detach()
        depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        np.save(os.path.join(depth_dir, f'{i:06d}_depth.npy'), depth_pred)
        plt.imsave(os.path.join(depth_dir, f'{i:06d}_depth.png'), depth_pred, cmap='viridis')

        normal_pred = output['normal'].squeeze().detach().permute(2,0,1)
        np.save(os.path.join(normal_dir, f'{i:06d}_normal.npy'), normal_pred.cpu().numpy())
        trans_topil((normal_pred+1)/2).save(os.path.join(normal_dir, f'{i:06d}_normal.png'))
