import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.dataIO_util import glob_data, load_rgb
from utils import model_util
from utils import graphics_utils
from typing import NamedTuple
import math
from loguru import logger
from pathlib import Path

from utils.hawp_util import WireframeGraph
from hawp.base import _C
from skimage import transform
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

def _normalize(inp):
    mag = torch.sqrt(inp[0]*inp[0]+inp[1]*inp[1])
    return inp/(mag+1e-6)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class ViewInfo(nn.Module):
    def __init__(self, cam_info, gt_info, inference=False):
        super().__init__()
        self.inference = inference

        if not inference:
            self.gt_info = gt_info
            self.cam_info = cam_info

            self.K = cam_info['K']
            # self.R = cam_info['R']
            # self.t = cam_info['t']

            self.mono_mask = cam_info['mask'].cuda()
            self.intrinsic = cam_info['intrinsic'].cuda()
            self.pose = cam_info['pose'].cuda()
            self.raster_cam_w2c = cam_info['raster_cam_w2c'].cuda()
            self.raster_cam_proj = cam_info['raster_cam_proj'].cuda()
            self.raster_cam_fullproj = cam_info['raster_cam_fullproj'].cuda()
            self.raster_cam_center = cam_info['raster_cam_center'].cuda()
            self.raster_cam_FovX = cam_info['raster_cam_FovX'].cpu().item()
            self.raster_cam_FovY = cam_info['raster_cam_FovY'].cpu().item()
            self.tanfovx = math.tan(self.raster_cam_FovX  * 0.5)
            self.tanfovy = math.tan(self.raster_cam_FovY * 0.5)
            self.raster_img_center = cam_info['raster_img_center'].cuda()

            self.ray_dirs = cam_info['ray_dirs'].cuda()
            self.cam_loc = cam_info['cam_loc'].cuda()
            self.depth_scale = cam_info['depth_scale'].cuda()

            # self.max_depth = gt_info['max_depth']
            self.img_path = gt_info['rgb_path']
            self.img_size = gt_info['img_size']
            self.rgb = gt_info['rgb'].cuda()
            self.md_path = gt_info['md_path']
            self.mono_depth = gt_info['mono_depth'].cuda()
            # self.mono_normal_local = gt_info['mono_normal_local'].cuda()
            self.mono_normal_global = gt_info['mono_normal_global'].cuda()
            self.index = gt_info['index']

            #
            self.uv = gt_info['uv']
            self.uv_proj = gt_info['uv_proj']
            self.juncs2d = gt_info['juncs2d']
            self.wireframe = gt_info['wireframe']
            self.mask = gt_info['mask']
            self.labels = gt_info['labels']
            self.lines = gt_info['lines']
            self.lines_uniq = gt_info['lines_uniq']

            self.groups_ids = gt_info['groups_ids']
            self.groups_nums = gt_info['groups_nums']
            self.padded_group_ids = gt_info['padded_group_ids']
            self.max_num = self.groups_nums.max()
            self.group_num = len(self.groups_ids)
            self.rows = np.arange(self.padded_group_ids.shape[0])[:, None]
            self.lines = self.lines[self.mask]
            self.uv = self.uv[self.mask]

        else:
            self.K = cam_info['K']
            self.intrinsic = cam_info['intrinsic'].cuda()
            self.img_path = gt_info['rgb_path']
            self.pose = cam_info['pose'].cuda()
            self.mask = gt_info['mask']
            self.labels = gt_info['labels']
            self.lines = gt_info['lines']
            self.lines_uniq = gt_info['lines_uniq']
            self.uv = gt_info['uv']

    def get_gt_dict(self):
        return {**self.cam_info, **self.gt_info}

    def get_sampling_gt(self, sampling_num=None):
        if not self.inference:
            sample_num = 1
            sample_idx = np.random.randint(0, self.max_num, self.group_num*sample_num).reshape(-1,sample_num) # [N, sample_num]

            sampling_idx = self.padded_group_ids[self.rows, sample_idx] # [N, sample_num]
            # sampling_idx = [self.groups_ids[i][np.random.randint(0, self.groups_ids[i].shape[0], sample_num)] for i in range(len(self.groups_ids))]
            sampling_idx = np.array(sampling_idx).reshape(-1)

            sampled_dict = {
                "intrinsics": self.intrinsic,
                "pose": self.pose,
                "lines": self.lines[sampling_idx],
                "uv": self.uv[sampling_idx],
            }
        else:
            # sampling
            sampling_idx = self.mask.nonzero().flatten()
            if sampling_num is not None:
                sampling_idx = sampling_idx[torch.randperm(sampling_idx.numel())[:sampling_num]] 
            sampled_dict = {
                # "rgb": self.rgb.reshape(-1,3)[sampling_idx, :],
                "intrinsics": self.intrinsic,
                "pose": self.pose,
                # "wireframe": self.wireframe,
                "lines": self.lines[sampling_idx],
                "labels": self.labels[sampling_idx],
                "uv": self.uv[sampling_idx],
                # "uv_proj": self.uv_proj[sampling_idx]
            }

        return sampled_dict


class GeneralDataset:
    def __init__(
        self,
        data_root='../data',
        scan_id='',
        depth_type='sensor',
        normal_type='omnidata',
        initial_mesh_root='',
        line_detector='scalelsd',
        distance_threshold=1.0,
        score_threshold=0.05,
        img_res = [],
        inference = False,
        scene_bounding_sphere=-1,
        depth_trunc=10.,
    ):
        self.detector = line_detector
        if depth_type == 'vggt':
            self.scene_dir = os.path.join(data_root, f'{scan_id}/vggt')
        else:
            self.scene_dir = os.path.join(data_root, f'{scan_id}')
        assert os.path.exists(self.scene_dir), f"scene path ({self.scene_dir}) does not exist"
        self.scene_bounding_sphere = scene_bounding_sphere
        assert self.scene_bounding_sphere > 0.
        self.distance = distance_threshold
        self.score_threshold = score_threshold
        self.inference = inference
        self.img_res = img_res
        self.depth_type = depth_type
        self.normal_type = normal_type
        self.depth_trunc = float(depth_trunc)

        self.image_root = os.path.join(self.scene_dir, 'images')
        self.depth_root = os.path.join(self.scene_dir, f'{depth_type}_depth')
        self.normal_root = os.path.join(self.scene_dir, f'{normal_type}_normal')
        self.pose_root = os.path.join(self.scene_dir, 'poses')
        self.wireframe_root = os.path.join(self.scene_dir, f'{line_detector}')

        # intrinsics
        intrinsic_path = os.path.join(self.scene_dir, 'intrinsics.txt')
        self.K_full = np.loadtxt(intrinsic_path)
        self.K = self.K_full[:3, :3]

        self.load_file_paths()
        self.load_data()

        # initial planar mesh
        self.mesh_path = initial_mesh_root + f'/{scan_id}_pred_mesh.ply'
        if not self.inference:
            if not os.path.exists(self.mesh_path):
                # self.get_mesh(self.depths, self.img_res, dest=self.mesh_path, depth_trunc=self.depth_trunc, voxel_length=0.02, sdf_trunc=0.04)
                self.get_mesh(self.depths, self.img_res, dest=self.mesh_path, depth_trunc=self.depth_trunc, voxel_length=0.01, sdf_trunc=0.02)
                
        
        if not self.inference:
            self.raster_cam_w2c_list, self.raster_cam_proj_list, self.raster_cam_fullproj_list, self.raster_cam_center_list, self.raster_cam_FovX_list, self.raster_cam_FovY_list, self.raster_img_center_list = self.get_raster_cameras(
                self.pose_all, self.img_res[0], self.img_res[1])

        # pre-compute attraction fields
        # logger.info('precomputing the support regions of 2D attraction fields...')
        for lines in tqdm(self.lines, desc='precomputing the support regions of 2D attraction fields...'):
            mask, labels, att_points = self.compute_point_line_attraction(lines)
            self.masks.append(mask)
            self.labels.append(labels)
            self.att_points.append(att_points)

        # build uv map
        logger.info('building pre-computed pixel uv map...')
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float().cuda()
        self.uv = uv.reshape(2, -1).transpose(1, 0)  # h*w, 2

        # prepare view list
        self.view_info_list = []
        for idx in tqdm(range(self.n_images), desc='building view list...'):
            pose = self.pose_all[idx].clone()
            intrinsic = torch.from_numpy(self.K).float().cuda()

            if not self.inference:
                ray_dirs, cam_loc = model_util.get_raydir_camloc(self.uv[None], pose[None], intrinsic[None])
                ray_dirs_tmp, _ = model_util.get_raydir_camloc(self.uv[None], torch.eye(4).to(pose.device)[None], intrinsic[None])
                '''
                dist_along_ray * depth_scale = depth
                '''
                depth_scale = ray_dirs_tmp[0, :, 2:]  # h*w, 1

                valid_labels = self.labels[idx][self.masks[idx]].cpu().numpy()
                uniq_labels = np.unique(valid_labels)
                group_ids = [np.where(valid_labels == ul)[0] for ul in uniq_labels]
                group_nums = np.array([len(gid) for gid in group_ids])
                max_len = group_nums.max() if group_nums.size > 0 else 0
                padded_group_ids = np.empty((len(group_ids), max_len), dtype=np.int64)
                for i, gid in enumerate(group_ids):
                    L = gid.shape[0]
                    if L == max_len:
                        padded_group_ids[i] = gid
                    else:
                        reps = (max_len + L - 1) // L
                        padded_group_ids[i] = np.tile(gid, reps)[:max_len]

                cam_info = {
                    "mask": self.mono_masks[idx].clone(),
                    "intrinsic": intrinsic,
                    "pose": self.pose_all[idx].clone(),  # camera to world
                    "raster_cam_w2c": self.raster_cam_w2c_list[idx].clone(),
                    "raster_cam_proj": self.raster_cam_proj_list[idx].clone(),
                    "raster_cam_fullproj": self.raster_cam_fullproj_list[idx].clone(),
                    "raster_cam_center": self.raster_cam_center_list[idx].clone(),
                    "raster_cam_FovX": self.raster_cam_FovX_list[idx].clone(),
                    "raster_cam_FovY": self.raster_cam_FovY_list[idx].clone(),
                    "raster_img_center": self.raster_img_center_list[idx].clone(),
                    "ray_dirs": ray_dirs.squeeze(0),
                    "cam_loc": cam_loc.squeeze(0),
                    "depth_scale": depth_scale,
                    "K": self.K,
                    # "R": self.R_all[idx],
                    # "t": self.t_all[idx],
                }

                gt_info = {
                    # "max_depth": self.max_depth,
                    "rgb_path": self.image_paths[idx],
                    "img_size": self.img_res,
                    "rgb": self.rgbs[idx].clone(),
                    "md_path": self.depth_paths[idx],
                    "mono_depth": self.mono_depths[idx].clone(),
                    # "mono_normal_local": self.mono_normals_local[idx].clone(),
                    "mono_normal_global": self.mono_normals_global[idx].clone(),
                    'index': idx, ##
                    "uv": self.uv.clone(),
                    # "uv": self.uv[self.masks[idx]].clone(),
                    "uv_proj": self.att_points[idx].clone(),
                    "juncs2d": self.wireframes[idx].vertices.clone(),
                    'wireframe': self.wireframes[idx],
                    "mask": self.masks[idx].clone(),
                    "labels": self.labels[idx].clone(),
                    "lines": self.lines[idx][self.labels[idx]].clone(),
                    # "lines": self.lines[idx][self.labels[idx]][self.masks[idx]].clone(),
                    "lines_uniq": self.lines[idx].clone(), ##
                    "groups_ids": group_ids,
                    "groups_nums": group_nums,
                    "padded_group_ids": padded_group_ids,
                }
            else:
                cam_info = {
                    "intrinsic": intrinsic,
                    "pose": pose,  # camera to world
                    "K": self.K,
                }
                gt_info = {
                    "rgb_path": self.image_paths[idx],
                    "mask": self.masks[idx].clone(),
                    "labels": self.labels[idx].clone(),
                    "lines": self.lines[idx][self.labels[idx]].clone(),
                    "lines_uniq": self.lines[idx].clone(), ##
                    "uv": self.uv.clone(),
                }

            self.view_info_list.append(ViewInfo(cam_info, gt_info, self.inference))            

        logger.info('data loader finished')


    def refuse_2(self, depths, poses, img_res, depth_trunc=10.0, voxel_length=0.03, sdf_trunc=0.05):
        H, W = img_res
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # for pose, depth_pred in tqdm(zip(poses, depths)):
        for i in range(len(poses)):
            intrinsic = self.K_full
            pose = poses[i].cpu().numpy()
            depth_pred = depths[i]
            
            rgb = np.ones((H, W, 3))
            rgb = (rgb * 255).astype(np.uint8)
            rgb = o3d.geometry.Image(rgb)
            
            depth_pred = o3d.geometry.Image(depth_pred)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, depth_pred, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
            )
            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
            extrinsic = np.linalg.inv(pose)
            volume.integrate(rgbd, intrinsic, extrinsic)
        
        return volume.extract_triangle_mesh()
        
    def get_mesh(self, depths, img_res, dest=None, depth_trunc=10., voxel_length=0.08, sdf_trunc=0.18):
        mesh = self.refuse_2(depths, self.pose_all, img_res, depth_trunc=depth_trunc, voxel_length=voxel_length, sdf_trunc=sdf_trunc)
        
        o3d.io.write_triangle_mesh(dest, mesh)   


    def __len__(self):
        return self.n_images
    
    def load_file_paths(self, n=1):
        cam_paths = glob_data(os.path.join(self.pose_root, "*.txt"))
        image_paths = glob_data(os.path.join(self.image_root, "*"))
        mono_normal_paths = glob_data(os.path.join(self.normal_root, "*.npy"))
        mono_depth_paths = glob_data(os.path.join(self.depth_root, "*.npy"))
        if len(mono_depth_paths)==0:
            mono_depth_paths = glob_data(os.path.join(self.depth_root, "*.png"))
            if len(mono_depth_paths)==0:
                print('Please provide avaliable depths!')
        wireframe_paths = glob_data(os.path.join(self.wireframe_root, "*.json"))

        n_images = len(mono_depth_paths)
        self.index_list = np.arange(0, n_images, n).tolist()
        self.n_images = len(self.index_list)

        if not self.inference:
            self.depth_paths = [mono_depth_paths[i] for i in self.index_list]
            self.normal_paths = [mono_normal_paths[i] for i in self.index_list]
        
        self.cam_paths = [cam_paths[i] for i in self.index_list]
        self.image_paths = [image_paths[i] for i in self.index_list]
        self.wireframe_paths = [wireframe_paths[i] for i in self.index_list]
        self.img_res = load_rgb(image_paths[0]).shape[1:]

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

    def load_cameras(self, cam_paths):
        poses_all = []
        for cam_path in cam_paths:
            pose = np.loadtxt(cam_path)
            poses_all.append(torch.from_numpy(pose).float().cuda())
        return poses_all

    def load_data(self):
        # load pose
        self.pose_all = self.load_cameras(self.cam_paths)
        
        # load image/depth/normal
        self.rgbs = []
        self.mono_depths = []   
        # self.mono_normals_local = []
        self.mono_normals_global = []
        self.mono_masks = []
        # load wireframe data
        self.masks = []
        self.wireframes = []
        self.lines = []
        self.labels = []
        self.att_points = []
        self.depths = []

        for i in tqdm(range(self.n_images), desc='loading data'):

            if not self.inference:
                rgbpath = self.image_paths[i]
                mdpath = self.depth_paths[i]
                mnpath = self.normal_paths[i]

                rgb = torch.from_numpy(load_rgb(rgbpath)).squeeze().float().cuda()  # 3, h, w
                assert rgb.shape[1] == self.img_res[0]
                assert rgb.shape[2] == self.img_res[1]
                # self.rgbs.append(rgb.reshape(3, -1).transpose(1, 0))
                self.rgbs.append(rgb.permute(1,2,0))
                
                # load mono-depth
               
                if 'sensor' in self.depth_type:
                    depth = cv2.imread(mdpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    if np.isnan(depth).any():
                        depth = self.inpaint_depth(depth)
                    depth = depth.astype(np.float32) / 1000.0
                else:
                    depth = np.load(mdpath)
                    if np.isnan(depth).any():
                        depth = self.inpaint_depth(depth)
                    # depth = depth.astype(np.float32) / 1000.

                depth[depth > self.depth_trunc] = 0
                mono_depth = torch.from_numpy(depth).squeeze().float().cuda() # h, w
                # remove black boundary
                mono_depth = mono_depth.reshape(-1)

                assert depth.shape[0] == self.img_res[0]
                assert depth.shape[1] == self.img_res[1]
                self.depths.append(depth)
                self.mono_depths.append(mono_depth.reshape(-1))

                # load mono-normal
                mono_normal = torch.from_numpy(np.load(mnpath)).squeeze().float().cuda()  # 3, h, w
                assert mono_normal.shape[1] == self.img_res[0]
                assert mono_normal.shape[2] == self.img_res[1]
                mono_normal = mono_normal.reshape(3, -1).transpose(1, 0)  # n, 3
                if self.normal_type == 'omnidata' or mono_normal.min()>=0:
                    # Note: the output normal of omnidata is normalized to [0,1]
                    mono_normal = mono_normal * 2. - 1.
                mono_normal = F.normalize(mono_normal, dim=-1)
                if 'global' in self.normal_type:
                    self.mono_normals_global.append(mono_normal)
                else:
                    mono_normal.reshape(self.img_res[0], self.img_res[1], 3)[:15, :] = 0
                    mono_normal.reshape(self.img_res[0], self.img_res[1], 3)[-15:, :] = 0
                    mono_normal.reshape(self.img_res[0], self.img_res[1], 3)[:, :15] = 0
                    mono_normal.reshape(self.img_res[0], self.img_res[1], 3)[:, -15:] = 0
                    c2w = self.pose_all[i]
                    mono_normal_global = mono_normal @ c2w[:3, :3].T
                    self.mono_normals_global.append(mono_normal_global)
                
                mono_mask = (self.mono_depths[-1] > 0) & (self.mono_normals_global[-1].abs().sum(dim=-1) > 0)
                self.mono_masks.append(mono_mask)

            # load wireframe  
            wfpath = self.wireframe_paths[i]
            wireframe = WireframeGraph.load_json(wfpath)
            self.wireframes.append(wireframe)
            lines = wireframe.line_segments(self.score_threshold).float().cuda()  # n, 4
            self.lines.append(lines)
        
    def get_raster_cameras(self, poses_all, height, width):
        zfar = 10.
        znear = 0.01
        raster_cam_w2c_list = []
        raster_cam_proj_list = []
        raster_cam_fullproj_list = []
        raster_cam_center_list = []
        raster_cam_FovX_list = []
        raster_cam_FovY_list = []
        raster_img_center_list = []
        intrinsic = torch.from_numpy(self.K).float().cuda()

        for i in range(self.n_images):
            focal_length_x = intrinsic[0,0]
            focal_length_y = intrinsic[1,1]
            FovY = graphics_utils.focal2fov(focal_length_y, height)
            FovX = graphics_utils.focal2fov(focal_length_x, width)

            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]

            c2w = poses_all[i]  # 4, 4
            w2c = c2w.inverse()  # 4, 4
            w2c_right = w2c.T

            world_view_transform = w2c_right.clone()
            projection_matrix = graphics_utils.getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            raster_cam_w2c_list.append(world_view_transform)
            raster_cam_proj_list.append(projection_matrix)
            raster_cam_fullproj_list.append(full_proj_transform)
            raster_cam_center_list.append(camera_center)
            raster_cam_FovX_list.append(torch.tensor([FovX]).cuda())
            raster_cam_FovY_list.append(torch.tensor([FovY]).cuda())

            raster_img_center_list.append(torch.tensor([cx, cy]).cuda())
        
        return raster_cam_w2c_list, raster_cam_proj_list, raster_cam_fullproj_list, raster_cam_center_list, raster_cam_FovX_list, raster_cam_FovY_list, raster_img_center_list

    def compute_point_line_attraction(self, lines):
        # lines_ = lines[:512,:-1].cuda()
        lines_ = lines[:,:-1].cuda()
        lmap, labels_onehot, _ = _C.encodels(lines_,self.img_res[0],self.img_res[1],self.img_res[0],self.img_res[1],lines_.shape[0])

        mask, labels = labels_onehot.max(dim=0)

        dismap = torch.sqrt(lmap[0]**2+lmap[1]**2)
        md_map = _normalize(lmap[:2])
        st_map = _normalize(lmap[2:4])
        ed_map = _normalize(lmap[4:])
        st_map = lmap[2:4]
        ed_map = lmap[4:]

        md_ = md_map.reshape(2,-1).t()
        st_ = st_map.reshape(2,-1).t()
        ed_ = ed_map.reshape(2,-1).t()
        Rt = torch.cat(
                (torch.cat((md_[:,None,None,0],md_[:,None,None,1]),dim=2),
                 torch.cat((-md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
        R = torch.cat(
                (torch.cat((md_[:,None,None,0], -md_[:,None,None,1]),dim=2),
                 torch.cat((md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
        #Rtst_ = torch.matmul(Rt, st_[:,:,None]).squeeze(-1).t()
        #Rted_ = torch.matmul(Rt, ed_[:,:,None]).squeeze(-1).t()
        Rtst_ = torch.bmm(Rt, st_[:,:,None]).squeeze(-1).t()
        Rted_ = torch.bmm(Rt, ed_[:,:,None]).squeeze(-1).t()
        swap_mask = (Rtst_[1]<0)*(Rted_[1]>0)
        pos_ = Rtst_.clone()
        neg_ = Rted_.clone()
        temp = pos_[:,swap_mask]
        pos_[:,swap_mask] = neg_[:,swap_mask]
        neg_[:,swap_mask] = temp

        pos_[0] = pos_[0].clamp(min=1e-9)
        pos_[1] = pos_[1].clamp(min=1e-9)
        neg_[0] = neg_[0].clamp(min=1e-9)
        neg_[1] = neg_[1].clamp(max=-1e-9)
        
        mask = (dismap<=self.distance)*mask

        pos_map = pos_.reshape(-1,self.img_res[0],self.img_res[1])
        neg_map = neg_.reshape(-1,self.img_res[0],self.img_res[1])

        md_angle  = torch.atan2(md_map[1], md_map[0])
        pos_angle = torch.atan2(pos_map[1],pos_map[0])
        neg_angle = torch.atan2(neg_map[1],neg_map[0])

        mask *= (pos_angle>0)
        mask *= (neg_angle<0)

        offsets = lmap[:2].permute(1,2,0)
        proj_points = torch.zeros((*mask.shape,2),device=mask.device,dtype=torch.float32)
        proj_points[mask,:] = offsets[mask] + mask.nonzero()[:,[1,0]].float()
        return mask.cpu().reshape(-1), labels.cpu().reshape(-1), proj_points.reshape(-1,2)

    def get_point_cloud(self):
        all_points = []
        with h5py.File(self.t_path, "r") as f:
            Tvecs = np.array(f["dataset"]).astype(np.float32)
        with h5py.File(self.R_path, "r") as f:
            Rvecs = np.array(f["dataset"]).astype(np.float32)
        ip = self.index_list[i]

        h, w = self.img_res
        for i in range(self.n_images):
            depth = self.mono_depths[i]
            K, R, T = self.K, Rvecs[ip], Tvecs[ip]
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            homo_2d = np.vstack([np.array([xv.flatten(), yv.flatten()]), np.ones((1, arr.shape[1]))])
            points = np.linalg.inv(K) @ (homo_2d * depth.flatten())
            points = R.T @ (points - T[:, None])
            all_points.append(points)
        all_points = np.concatenate(all_points, 0)
        return all_points

