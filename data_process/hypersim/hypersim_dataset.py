import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.dataIO_util import glob_data, load_rgb
from pathlib import Path
import csv
import h5py
import cv2


class HypersimDataset:
    def __init__(
        self,
        data_root='./data',
        scan_id='',
    ):
        self.scene_dir = os.path.join(data_root, '{0}'.format(scan_id))
        assert os.path.exists(self.scene_dir), f"scene path ({self.scene_dir}) does not exist"
        
        self.init_intrinsic()

        self.load_file_paths()

        self.n_images = len(self.index_list)
        
        self.load_data()


    def init_intrinsic(self, max_dim=-1):
        self.R180x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.mpau = self.read_mpau()

        self.default_h, self.default_w = 768, 1024
        if max_dim > 0:
            ratio = max_dim / max(self.default_h, self.default_w)
            h, w = int(round(self.default_h * ratio)), int(round(self.default_w * ratio))
        else:
            h, w = self.default_h, self.default_w

        fov_x = np.pi / 3  # set fov_x to pi/3 to match DIODE dataset (60 degrees)
        f = w / (2 * np.tan(fov_x / 2))
        fov_y = 2 * np.arctan(h / (2 * f))
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        self.K = K
        self.K_full = np.eye(4)
        self.K_full[:3, :3] = K
        self.img_res = [h, w] 

    def read_mpau(self):
        fname_metascene = os.path.join(
            self.scene_dir, "_detail", "metadata_scene.csv"
        )

        param_dict = {}
        with open(fname_metascene) as f:
            reader = csv.DictReader(f)
            for row in reader:
                param_dict[row["parameter_name"]] = row["parameter_value"]
        key = "meters_per_asset_unit"
        if key in param_dict:
            return float(param_dict[key])
        else:
            raise ValueError(f"Key {key} not exists in {fname_metascene}")
        
    def __len__(self):
        return self.n_images
    
    def load_file_paths(self, n=1, cam_id=0):
        index_list = np.arange(0, 100, n).tolist()
        self.index_list = []
        self.image_paths = []
        self.depth_paths = []
        self.depth_image_paths = []
        self.normal_paths_global = []
        self.normal_image_paths = []

        positions_fname = os.path.join(
            self.scene_dir,
            "_detail",
            f"cam_{cam_id:02d}",
            "camera_keyframe_positions.hdf5",
        )
        self.t_path = positions_fname

        orientations_fname = os.path.join(
            self.scene_dir,
            "_detail",
            f"cam_{cam_id:02d}",
            "camera_keyframe_orientations.hdf5",
        )
        self.R_path = orientations_fname

        for image_id in index_list:
                
            image_fname = os.path.join(
                self.scene_dir,
                "images",
                f"scene_cam_{cam_id:02d}_final_preview",
                f"frame.{image_id:04d}.color.jpg",
            )
            if os.path.exists(image_fname):
                self.image_paths.append(image_fname)
                self.index_list.append(image_id)
            else:
                continue

            raydepth_fname = os.path.join(
                self.scene_dir,
                "images",
                f"scene_cam_{cam_id:02d}_geometry_hdf5",
                f"frame.{image_id:04d}.depth_meters.hdf5",
            )
            if os.path.exists(raydepth_fname):
                self.depth_paths.append(raydepth_fname)
                self.depth_image_paths.append(
                    os.path.join(
                        self.scene_dir,
                        "images",
                        f"scene_cam_{cam_id:02d}_geometry_preview",
                        f"frame.{image_id:04d}.depth_meters.png",
                    )
                )

            raynormal_fname_global = os.path.join(
                self.scene_dir,
                "images",
                f"scene_cam_{cam_id:02d}_geometry_hdf5",
                f"frame.{image_id:04d}.normal_bump_world.hdf5",
            )
            if os.path.exists(raynormal_fname_global):
                self.normal_paths_global.append(raynormal_fname_global)
                self.normal_image_paths.append(
                    os.path.join(
                        self.scene_dir,
                        "images",
                        f"scene_cam_{cam_id:02d}_geometry_preview",
                        f"frame.{image_id:04d}.normal_bump_world.png",
                    )
                )


    def raydepth2depth(self, raydepth, K, img_hw):
        K_inv = np.linalg.inv(K)
        h, w = raydepth.shape[0], raydepth.shape[1]
        grids = np.meshgrid(np.arange(w), np.arange(h))
        coords_homo = [grids[0].reshape(-1), grids[1].reshape(-1), np.ones(h * w)]
        coords_homo = np.stack(coords_homo)
        coeffs = np.linalg.norm(K_inv @ coords_homo, axis=0)
        coeffs = coeffs.reshape(h, w)
        depth = raydepth / coeffs
        return depth

    def inpaint_normal(self, normal):
        nan_mask = np.isnan(normal).any(axis=2).astype(np.uint8) * 255
        normal_clean = np.nan_to_num(normal, nan=0.0)
        normal_8bit = np.clip((normal_clean + 1) / 2.0 * 255, 0, 255).astype(np.uint8)
        restored_channels = []
        for ch in range(3):
            restored_ch = cv2.inpaint(
                normal_8bit[:, :, ch], nan_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
            )
            restored_channels.append(restored_ch)
        restored_8bit = np.stack(restored_channels, axis=2)
        restored = (restored_8bit / 255.0) * 2 - 1  # [-1, 1]
        normal_new = np.where(nan_mask[..., None], restored, normal)
        return normal_new

    def inpaint_depth(self, depth):
        depth_clean = depth.copy()
        nan_mask = np.isnan(depth_clean).astype(np.uint8) * 255  # 0 或 255，uint8
        depth_clean[np.isnan(depth_clean)] = 0.0
        depth_min, depth_max = depth_clean.min(), depth_clean.max()
        if depth_max - depth_min == 0:
            depth_8bit = np.zeros_like(depth_clean, dtype=np.uint8)
        else:
            depth_8bit = np.clip(
                (depth_clean - depth_min) / (depth_max - depth_min) * 255,
                0, 255
            ).astype(np.uint8)
        depth_8bit_repaired = cv2.inpaint(
            depth_8bit, nan_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
        )
        repaired_depth = depth_8bit_repaired.astype(np.float32) / 255.0 * (depth_max - depth_min) + depth_min
        depth_new = depth.copy()
        depth_new[np.isnan(depth)] = repaired_depth[np.isnan(depth)]
        return depth_new

    def load_data(self):
        self.pose_all = []
        self.rgbs = []
        self.mono_depths = []   
        self.normal_gs = []

        with h5py.File(self.t_path, "r") as f:
            Tvecs = np.array(f["dataset"]).astype(np.float32)
        with h5py.File(self.R_path, "r") as f:
            Rvecs = np.array(f["dataset"]).astype(np.float32)
            
        for i in tqdm(range(self.n_images), desc='loading data'):
            rgbpath = self.image_paths[i]
            mdpath = self.depth_paths[i]
            mnpath_g = self.normal_paths_global[i]

            # load pose
            ip = self.index_list[i]
            R_w2c = Rvecs[ip]
            T_w2c = Tvecs[ip] * self.mpau
            R_c2w = self.R180x @ R_w2c.T
            T_c2w = -R_c2w @ T_w2c
            pose = np.eye(4)
            pose[:3, :3] = R_c2w
            pose[:3, 3] = T_c2w
            pose = np.linalg.inv(pose)
            self.pose_all.append(torch.from_numpy(pose).float().cuda())

            rgb = torch.from_numpy(load_rgb(rgbpath)).squeeze().float().cuda()  # 3, h, w
            self.rgbs.append(rgb.permute(1,2,0))
            
            # load mono-depth
            with h5py.File(mdpath, "r") as f:
                raydepth = np.array(f["dataset"]).astype(np.float32)
            depth = self.raydepth2depth(raydepth, self.K, self.img_res).astype(np.float32)  # h, w
            if np.isnan(depth).any():
                depth = self.inpaint_depth(depth)
            self.mono_depths.append(depth)

            # load mono-normal-lglobal
            with h5py.File(mnpath_g, "r") as f:
                normal_g = np.array(f["dataset"]).astype(np.float32)  # h, w, 3
            if np.isnan(normal_g).any():
                normal_g = self.inpaint_normal(normal_g)
            self.normal_gs.append(normal_g)

    def load_cameras(self, cam_id=0):
        R180x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.mpau = self.read_mpau()

        positions_fname = os.path.join(
            self.scene_dir,
            "_detail",
            f"cam_{cam_id:02d}",
            "camera_keyframe_positions.hdf5",
        )
        with h5py.File(positions_fname, "r") as f:
            Tvecs = np.array(f["dataset"]).astype(np.float32)

        orientations_fname = os.path.join(
            self.scene_dir,
            "_detail",
            f"cam_{cam_id:02d}",
            "camera_keyframe_orientations.hdf5",
        )
        with h5py.File(orientations_fname, "r") as f:
            Rvecs = np.array(f["dataset"]).astype(np.float32)

        # # change to world-frame R and t following Iago Suarez
        # # [LINK] https://github.com/iago-suarez/powerful-lines/blob/main/src/evaluation/consensus_3dsegs_detection.py
        # N_images = self.n_images
        # self.Rvecs = Rvecs[:N_images].copy()
        # self.Tvecs = Tvecs[:N_images].copy()
        # for image_id in range(N_images):
        #     self.Rvecs[image_id] = self.R180x @ self.Rvecs[image_id].T
        #     self.Tvecs[image_id] = self.Tvecs[image_id] * self.mpau
        #     self.Tvecs[image_id] = -self.Rvecs[image_id] @ self.Tvecs[image_id]
        
        self.pose_all = []
        for image_id in range(self.n_images):
            R_w2c = Rvecs[image_id]
            T_w2c = Tvecs[image_id] * self.mpau
            
            R_c2w = R180x @ R_w2c.T
            T_c2w = -R_c2w @ T_w2c

            pose = np.eye(4)
            pose[:3, :3] = R_c2w
            pose[:3, 3] = T_c2w
            pose = np.linalg.inv(pose)

            self.pose_all.append(torch.from_numpy(pose).float().cuda())

    def get_point_cloud(self):
        all_points = []
        with h5py.File(self.t_path, "r") as f:
            Tvecs = np.array(f["dataset"]).astype(np.float32)
        with h5py.File(self.R_path, "r") as f:
            Rvecs = np.array(f["dataset"]).astype(np.float32)
        ip = self.index_list[i]

        h, w = self.img_res
        for i in range(self.n_images):
            depth = self.depths[i]
            K, R, T = self.K, Rvecs[ip], Tvecs[ip]
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            homo_2d = np.vstack([np.array([xv.flatten(), yv.flatten()]), np.ones((1, arr.shape[1]))])
            points = np.linalg.inv(K) @ (homo_2d * depth.flatten())
            points = R.T @ (points - T[:, None])
            all_points.append(points)
        all_points = np.concatenate(all_points, 0)
        return all_points

