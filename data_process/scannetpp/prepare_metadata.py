import os
from colmap_io import read_extrinsics_text, read_intrinsics_text
from colmap_io import qvec2rotmat
import numpy as np
import shutil
from argparse import ArgumentParser

parser = ArgumentParser("process rgb images and poses")
parser.add_argument("--data_root", type=str, default="./data/ScanNet++", help='root path to scannet dataset')
parser.add_argument("--save_root", type=str, default="./data/ScanNetPP/scans", help='path to save processed results')
parser.add_argument("--scene_id", type=str, required=True, default='', help='name of one scene')
args = parser.parse_args()

data_path = os.path.join(args.data_root, args.scene_id)
save_path = os.path.join(args.save_root, args.scene_id)
os.makedirs(save_path, exist_ok=True)

rgb_path = os.path.join(data_path, 'iphone/rgb')
assert not os.path.exists(rgb_path)

rgb_frame_names = os.listdir(rgb_path)
assert len(rgb_frame_names) == 0

colmap_cam_file_path = os.path.join(data_path, 'iphone/colmap/cameras.txt')
assert not os.path.exists(colmap_cam_file_path)
    
colmap_image_file_path = os.path.join(data_path, 'iphone/colmap/images.txt')
assert not os.path.exists(colmap_image_file_path)

cameras = read_intrinsics_text(colmap_cam_file_path)
camera = next(iter(cameras.values()))
fx, fy, cx, cy = camera.params[:4]
intrinsic_out_path = os.path.join(save_path, 'intrinsic')
os.makedirs(intrinsic_out_path, exist_ok=True)
intrinsic_color = np.array([[fx, 0., cx, 0.],
                            [0.,fy, cy, 0.],
                            [0.,0.,1.0,0.],
                            [0.,0.,0.,1.0]])
np.savetxt(os.path.join(intrinsic_out_path, 'intrinsic_color.txt'), intrinsic_color, fmt='%.6f')

hr = 480 / camera.height
wr = 640 / camera.width
fx_ = fx * wr
fy_ = fy * hr
cx_ = cx * wr
cy_ = cy * hr
intrinsic_depth = np.array([[fx_, 0., cx_, 0.],
                            [0.,fy_, cy_, 0.],
                            [0.,0.,1.0,0.],
                            [0.,0.,0.,1.0]])
np.savetxt(os.path.join(intrinsic_out_path, 'intrinsic_depth.txt'), intrinsic_depth, fmt='%.6f')

images_meta = read_extrinsics_text(colmap_image_file_path)

i = 0
for img_id, img_meta in images_meta.items():
    image_meta = img_meta
    image_id = image_meta.id
    frame_name = image_meta.name
    q = image_meta.qvec
    t = image_meta.tvec
    r = qvec2rotmat(q)
    rt = np.eye(4)
    rt[:3,:3] = r
    rt[:3, 3] = t
    c2w = np.linalg.inv(rt)

    out_path = os.path.join(save_path, 'sensor_data')
    os.makedirs(out_path, exist_ok=True)
    shutil.copy2(os.path.join(rgb_path, frame_name), os.path.join(out_path, 'frame-%06d.color.jpg'%(i)))

    np.savetxt(os.path.join(out_path, 'frame-%06d.pose.txt'%(i)), c2w, fmt='%.6f')
    i += 1

