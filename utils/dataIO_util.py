import numpy as np
import cv2
import imageio
import skimage
import os
from datetime import datetime
from pyhocon import HOCONConverter
import shutil
from glob import glob

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths

def load_rgb(path, normalize_rgb = False):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if normalize_rgb: # [-1,1] --> [0,1]
        img -= 0.5
        img *= 2.
    img = img.transpose(2, 0, 1)
    return img

def prepare_folders(kwargs, expname, timestamp=''):
    # =======================================  create experiment folder
    exps_folder_name = kwargs['exps_folder_name']
    # expdir = os.path.join(exps_folder_name, expname, timestamp)
    expdir = os.path.join(exps_folder_name, expname)
    mkdir_ifnotexists(expdir)
    # =======================================  create plot folder
    plane_plots_dir = os.path.join(expdir, 'plane_plots')
    mkdir_ifnotexists(plane_plots_dir)
    line_plots_dir = os.path.join(expdir, 'line_plots')
    mkdir_ifnotexists(line_plots_dir)
    # =======================================  create checkpoint folder
    checkpoints_path = os.path.join(expdir, 'checkpoints')
    model_subdir = "Parameters"
    mkdir_ifnotexists(os.path.join(checkpoints_path, model_subdir))

    return expdir, plane_plots_dir, line_plots_dir, checkpoints_path, model_subdir

def save_config_files(expdir, conf):
    timestamp_ = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    cfg_path = os.path.join(expdir, f'train.conf')
    with open(cfg_path, "w") as fd:
        fd.write(HOCONConverter.to_json(conf))
        
    # source_tain_file = os.path.join(*(conf.get_string('train.train_runner_class').split('.')[:-1])) + '.py'
    # if os.path.exists(source_tain_file):
    #     shutil.copy(source_tain_file, os.path.join(expdir, f'trainer_{timestamp_}.py'))

def save_images_to_video(images, saveto='lines.mp4', fps=30):
    """
    Save a list of images to a video file.
    
    Args:
        images (list): Data of images. [path of folder] or [list of numpy arrays].
        fps (int): Frames per second for the video.
    """
    if isinstance(images, str):
        # If a directory is given, load all images from that directory
        image_files = sorted(glob(os.path.join(images, '*.png')), key=os.path.getmtime)
        images = [cv2.imread(img_file) for img_file in image_files]
    else:
        # Ensure images is a list of numpy arrays
        if len(images) == 0:
            print("No images to save.")
            return
        else:
            images = [img for img in images if isinstance(img, np.ndarray)]
    
    height, width, layers = images[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(saveto, fourcc, fps, (width, height))
    for img in images:
        out.write(img)
    out.release()
    print(f"Video saved to {saveto}")

def save_images_to_gif(images, saveto='lines.gif', fps=30):
    """
    Save a list of images to a GIF file.
    
    Args:
        images (list): Data of images. [path of folder] or [list of numpy arrays].
        gif_filename (str): Name of the output GIF file.
        duration (int): Duration for each frame in milliseconds.
    """
    if isinstance(images, str):
        # If a directory is given, load all images from that directory
        image_files = sorted(glob(os.path.join(images, '*.png')), key=os.path.getmtime)
        images = [imageio.imread(img_file) for img_file in image_files]
    else:
        # Ensure images is a list of numpy arrays
        if len(images) == 0:
            print("No images to save.")
            return
        else:
            images = [img for img in images if isinstance(img, np.ndarray)]
    
    imageio.mimsave(saveto, images, duration=1/fps)
    print(f"GIF saved to {saveto}")