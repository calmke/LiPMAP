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

import utils_gradio.general as utils
from utils_gradio import rend_util
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
        self.gt_info = gt_info
        self.cam_info = cam_info

        self.K = cam_info['K']
        # self.R = cam_info['R']
        # self.t = cam_info['t']

        # self.mask = cam_info['mask'].cuda()
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

        self.img_path = gt_info['rgb_path']
        self.img_size = gt_info['img_size']
        self.rgb = gt_info['rgb'].cuda()
        self.mono_depth = gt_info['mono_depth'].cuda()
        self.mono_normal_local = gt_info['mono_normal_local'].cuda()
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

        self.scale = 1.0
        self.shift = 0.0
        self.plane_depth = None

    def get_gt_dict(self):
        return {**self.cam_info, **self.gt_info}

    def get_sampling_gt(self, sampling_num=None):
        # sampling
        sampling_idx = self.mask.nonzero().flatten()
        sampling_num = sampling_idx.shape[0] // 20
        if sampling_num is not None:
            sampling_idx = sampling_idx[torch.randperm(sampling_idx.numel())[:sampling_num]] 

        sampled_dict = {
            "rgb": self.rgb.reshape(-1,3)[sampling_idx, :],
            "intrinsics": self.intrinsic,
            "pose": self.pose,
            "wireframe": self.wireframe,
            "lines": self.lines[sampling_idx],
            "labels": self.labels[sampling_idx],
            "uv": self.uv[sampling_idx],
            "uv_proj": self.uv_proj[sampling_idx]
        }
        return sampled_dict


class Dataset:
    def __init__(
        self,
        data,
        img_res: list,
        scan_id: str = 'example',
        dataset_name: str = 'demo',
        scene_bounding_sphere: float = 5.0,
        line_detector: str = 'scalelsd',
        distance_threshold: float = 1.0,
        score_threshold: float = 0.05,
        voxel_length: float = 0.05,
        sdf_trunc: float = 0.08,
        depth_trunc: float = 5.0,
        initial_mesh_root = '',
        **kwargs
    ):
        self.detector = line_detector
        self.dataset_name = dataset_name
        self.scan_id = scan_id
        self.scene_bounding_sphere = scene_bounding_sphere
        assert self.scene_bounding_sphere > 0.
        self.score_threshold = data['threshold'] if 'threshold' in data.keys() else score_threshold
        self.distance = distance_threshold

        self.img_res = img_res
        image_paths = data['image_paths']
        self.image_paths = image_paths
        self.n_images = len(image_paths)

        # load camera
        self.intrinsics_all = [torch.from_numpy(intrinsic).cuda() for intrinsic in data['intrinsics']]
        self.poses_all = [torch.from_numpy(extrinsic).cuda() for extrinsic in data['extrinsics']]
        
        # load rgbs
        rgbs = [torch.from_numpy(rgb).cuda().float()/255. for rgb in data['color']]
        rgbs = torch.stack(rgbs, dim=0).contiguous()  # n, h, w, 3
        assert rgbs.shape[0] == self.n_images
        assert rgbs.shape[1] == img_res[0]
        assert rgbs.shape[2] == img_res[1]
        rgbs = rgbs.reshape(self.n_images, -1, 3)  # n, hw, 3

        # load depths
        mono_depths = [torch.from_numpy(depth).cuda() for depth in data['depth']]
        mono_depths = torch.stack(mono_depths, dim=0).contiguous()   # n, h, w
        assert mono_depths.shape[0] == self.n_images
        assert mono_depths.shape[1] == img_res[0]
        assert mono_depths.shape[2] == img_res[1]
        mono_depths[mono_depths > 2.0 * self.scene_bounding_sphere] = 0.
        mono_depths = mono_depths.reshape(self.n_images, -1)  # n, hw

        # mono_mask = (self.mono_depths[-1] > 0) & (self.mono_normals_global[-1].abs().sum(dim=-1) > 0)
        # self.mono_masks.append(mono_mask)

        # load normals
        mono_normals = [torch.from_numpy(normal).cuda()*2.-1. for normal in data['normal']]
        mono_normals = torch.stack(mono_normals, dim=0).permute(0, 2, 3, 1).contiguous()  # n, h, w, 3
        assert mono_normals.shape[0] == self.n_images
        assert mono_normals.shape[1] == img_res[0]
        assert mono_normals.shape[2] == img_res[1]
        mono_normals = mono_normals.reshape(self.n_images, -1, 3)  # n, hw, 3

        # load wireframe  
        self.wireframes = data['wireframes']
        self.lines = []
        for wireframe in data['wireframes']:
            assert wireframe.frame_height == self.img_res[0] and wireframe.frame_width == self.img_res[1]
            self.lines.append(wireframe.line_segments(self.score_threshold).cuda())

        self.K = data['intrinsics'][0]
        self.K_full = np.eye(4)
        self.K_full[:3, :3] = self.K

        # get cam parameters for rasterization
        self.raster_cam_w2c_list, self.raster_cam_proj_list, self.raster_cam_fullproj_list, self.raster_cam_center_list, self.raster_cam_FovX_list, self.raster_cam_FovY_list, self.raster_img_center_list = self.get_raster_cameras(
            self.poses_all, self.img_res[0], self.img_res[1])
        
        # generate mesh for initialization
        mesh = self.refuse_2(data['depth'], self.poses_all, img_res, depth_trunc=depth_trunc, voxel_length=voxel_length, sdf_trunc=sdf_trunc)
        absolute_img_path = os.path.abspath(image_paths[0])
        current_dir = os.path.dirname(absolute_img_path)
        parent_dir = os.path.dirname(current_dir)
        self.mesh_path = os.path.join(initial_mesh_root, f'{scan_id}_pred_mesh.ply')
        o3d.io.write_triangle_mesh(self.mesh_path, mesh) 

        # load wireframe data
        self.masks = []
        self.labels = []
        self.att_points = []
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
            pose = self.poses_all[idx].clone()
            # intrinsic = self.intrinsics_all[idx].clone()
            intrinsic = torch.from_numpy(self.K).float().cuda()
            ray_dirs, cam_loc = model_util.get_raydir_camloc(self.uv[None], pose[None], intrinsic[None])
        
            ray_dirs_tmp, _ = model_util.get_raydir_camloc(self.uv[None], torch.eye(4).to(pose.device)[None], intrinsic[None])
            '''
            dist_along_ray * depth_scale = depth
            '''
            depth_scale = ray_dirs_tmp[0, :, 2:]  # h*w, 1

            normal_local = mono_normals[idx].clone().cuda()
            normal_global = normal_local @ self.poses_all[idx][:3, :3].T

            cam_info = {
                # "mask": self.mono_masks[idx].clone(),
                "intrinsic": intrinsic,
                "pose": self.poses_all[idx].clone(),  # camera to world
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
                "rgb_path": self.image_paths[idx],
                "img_size": self.img_res,
                "rgb": rgbs[idx].clone(),
                "mono_depth": mono_depths[idx].clone(),
                "mono_normal_local": normal_local,
                "mono_normal_global": normal_global,
                'index': idx, ##
                "uv": self.uv.clone(),
                "uv_proj": self.att_points[idx].clone(),
                "juncs2d": self.wireframes[idx].vertices.clone(),
                'wireframe': self.wireframes[idx],
                "mask": self.masks[idx].clone(),
                "labels": self.labels[idx].clone(),
                "lines": self.lines[idx][self.labels[idx]].clone(),
                "lines_uniq": self.lines[idx].clone(), ##
            }
            self.view_info_list.append(ViewInfo(cam_info, gt_info))            

        logger.info('data loader finished')


    def refuse_2(self, depths, poses, img_res, depth_trunc=5.0, voxel_length=0.03, sdf_trunc=0.05):
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

    def __len__(self):
        return self.n_images
    
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

