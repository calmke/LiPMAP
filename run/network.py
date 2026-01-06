import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import KDTree
import open3d as o3d

from utils import conf_util
from utils import model_util
from utils import loss_util
from utils import mesh_util
from utils import plot_util

from loguru import logger
import matplotlib.pyplot as plt

import time
import json
import h5py
# import bloscpack as bp

# import line_edge_assignment as lea

class RecWrapper(nn.Module):
    def __init__(self, conf, plane_plots_dir='', view_info_list=None):
        super().__init__()
        self.conf = conf

        # plane model
        self.plane_model_conf = self.conf.get_config('plane_model')
        self.planarSplat = conf_util.get_class(self.plane_model_conf.get_string('plane_model_class'))(conf, plot_dir=plane_plots_dir)
        self.init_type = self.plane_model_conf.get_string('init_type', default='mesh')

        scan_id = self.conf.get_string('dataset.scan_id')
        if self.init_type == 'mesh':
            if self.conf.get_string('dataset.initial_mesh_root', None) is not None:
                initial_mesh_root = self.conf.get_string('dataset.initial_mesh_root')
                mesh_path = os.path.join(initial_mesh_root, f'{scan_id}_pred_mesh.ply')
                if not os.path.exists(mesh_path):
                    mesh_path = os.path.join(initial_mesh_root, f'{scan_id}/{scan_id}_pred_mesh.ply')
            else:
                raise ValueError('Not a correct mesh path provided for mesh initialization')
            logger.info(f'initialization with mesh {mesh_path}')
            self.planarSplat.initialize_from_mesh(mesh_path)

        self.plane_vis_denom = torch.zeros((self.planarSplat.get_plane_num(), 1), device="cuda")
        self.radii_grad_denom = torch.zeros((self.planarSplat.get_plane_num(), 1), device="cuda")
        self.radii_grad_accum = torch.zeros((self.planarSplat.get_plane_num(), 4), device="cuda")
        self.split_thres = self.plane_model_conf.get_float('split_thres')
        self.radii_dir_type = self.plane_model_conf.get_string('radii_dir_type')

        self.plane_depth_loss = loss_util.metric_depth_loss
        self.plane_normal_loss = loss_util.normal_loss        
        # ======================================= loss settings
        loss_plane_conf = self.plane_model_conf.get_config('plane_loss')
        self.weight_plane_normal = loss_plane_conf.get_float('weight_mono_normal')
        self.weight_plane_depth = loss_plane_conf.get_float('weight_mono_depth')

        self.weight_plane = 10.0
        self.weight_line = 0.1

    def set_global_junctions(self, num=None, array=None):
        if num is not None:
            self.global_juncs = nn.Parameter(torch.zeros(num,3).cuda(), requires_grad=True)
        elif array is not None:
            self.global_juncs = nn.Parameter(array.cuda(), requires_grad=True)
        else:
            self.global_juncs = nn.Parameter(self.planarSplat._plane_center.detach(), requires_grad=True)

    def build_plane_optimizer_and_LRscheduler(self):
        plane_model_conf = self.conf.get_config('plane_model')
        opt_dict = [
            {'params': [self.planarSplat._plane_center], 'lr': plane_model_conf.lr_center, "name": "plane_center", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_radii_xy_p], 'lr': plane_model_conf.lr_radii, "name": "plane_radii_xy_p", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_radii_xy_n], 'lr': plane_model_conf.lr_radii, "name": "plane_radii_xy_n", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_rot_q_normal_wxy], 'lr': plane_model_conf.lr_rot_normal, "name": "plane_rot_q_normal_wxy", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_rot_q_xyAxis_z], 'lr': plane_model_conf.lr_rot_xy, "name": "plane_rot_q_xyAxis_z", "weight_decay": 0.},
            {'params': [self.planarSplat._plane_rot_q_xyAxis_w], 'lr': plane_model_conf.lr_rot_xy, "name": "plane_rot_q_xyAxis_w", "weight_decay": 0.},
        ]
        self.optimizer_plane = torch.optim.Adam(opt_dict, betas=(0.9, 0.99), eps=1e-15)

    def build_global_juncs_optimizer_and_LRscheduler(self):
        plane_model_conf = self.conf.get_config('plane_model')
        self.optimizer_global_juncs = torch.optim.Adam([self.global_juncs], lr=plane_model_conf.lr_global_juncs, betas=(0.9, 0.99), eps=1e-15)

    def build_optimizer_and_LRscheduler(self):
        self.build_plane_optimizer_and_LRscheduler()
        self.optimizer_global_juncs = None
        self.scheduler_global_juncs = None

    def optimizer_zero_grad(self):
        self.optimizer_plane.zero_grad()
        if self.optimizer_global_juncs is not None:
            self.optimizer_global_juncs.zero_grad()

    def optimizer_update(self):
        self.optimizer_plane.step()
        self.update_grad_stats()
        self.regularize_plane_shape(False)
        if self.optimizer_global_juncs is not None:
            self.optimizer_global_juncs.step()
            # self.scheduler_global_juncs.step()
        
    def prune_optimizer(self, valid_mask):
        valid_mask = valid_mask.squeeze()
        optimizable_tensors = {}
        for group in self.optimizer_plane.param_groups:
            stored_state = self.optimizer_plane.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]

                del self.optimizer_plane.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][valid_mask].requires_grad_(True)))
                self.optimizer_plane.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][valid_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer_plane.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer_plane.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer_plane.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer_plane.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        assert len(optimizable_tensors) > 0
        return optimizable_tensors
    
    def prune_core(self, invalid_mask):
        invalid_mask = invalid_mask.squeeze()
        valid_plane_mask = ~invalid_mask
        opt_tensors = self.prune_optimizer(valid_plane_mask)

        self.planarSplat._plane_center = opt_tensors["plane_center"]
        self.planarSplat._plane_radii_xy_p = opt_tensors['plane_radii_xy_p']
        self.planarSplat._plane_radii_xy_n = opt_tensors['plane_radii_xy_n']
        self.planarSplat._plane_rot_q_normal_wxy = opt_tensors['plane_rot_q_normal_wxy']
        self.planarSplat._plane_rot_q_xyAxis_z = opt_tensors['plane_rot_q_xyAxis_z']
        self.planarSplat._plane_rot_q_xyAxis_w = opt_tensors['plane_rot_q_xyAxis_w']

        self.plane_vis_denom = self.plane_vis_denom[valid_plane_mask]
        self.radii_grad_denom = self.radii_grad_denom[valid_plane_mask]
        self.radii_grad_accum = self.radii_grad_accum[valid_plane_mask]

        if self.planarSplat.rot_delta is not None:
            self.planarSplat.rot_delta = self.planarSplat.rot_delta[valid_plane_mask]

    def prune_invisible_plane(self,):
        prune_mask = self.plane_vis_denom == 0
        self.prune_core(prune_mask.detach())
        self.planarSplat.check_model()
        self.reset_plane_vis()
        torch.cuda.empty_cache()


    def prune_invalid_plane(self,):
        plane_radii = self.planarSplat.get_plane_radii()
        prune_mask = ~(plane_radii.abs().min(-1)[0] >= 0)

        self.prune_core(prune_mask.detach())
        self.planarSplat.check_model()
        torch.cuda.empty_cache()
    
    def prune_uninteracted_plane(self, view_info_list):
        self.reset_plane_vis()
        valid_plane_mask = torch.ones(self.planarSplat.get_plane_num(), dtype=torch.bool).cuda()
        for view_info in view_info_list:
            mask = self.check_interacted_plane_mask(view_info)
            # valid_plane_mask = valid_plane_mask & mask
            valid_plane_mask = valid_plane_mask | mask

        prune_mask = ~valid_plane_mask
        self.prune_core(prune_mask.detach())
        # self.plane_vis_denom = self.plane_vis_denom[valid_plane_mask]
        # self.radii_grad_denom = self.radii_grad_denom[valid_plane_mask]
        # self.radii_grad_accum = self.radii_grad_accum[valid_plane_mask]
        # if self.planarSplat.rot_delta is not None:
        #     self.planarSplat.rot_delta = self.planarSplat.rot_delta[valid_plane_mask]
        # self.planarSplat.check_model()
        torch.cuda.empty_cache()


    def prune_small_plane(self,):
        plane_radii = self.planarSplat.get_plane_radii()
        prune_mask = plane_radii.abs().min(-1)[0] <= self.planarSplat.radii_min_list[-1] * 1.25
        self.prune_core(prune_mask.detach())
        self.planarSplat.check_model()
        torch.cuda.empty_cache()
    
    def prune_overlapped_plane(self):
        plane_normal, plane_offset, plane_center, plane_radii, plane_rot_q, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_geometry()
        prune_mask = model_util.get_overlapped_mask(plane_center, plane_normal, plane_offset, plane_radii, plane_xAxis, plane_yAxis)
        self.prune_core(prune_mask.detach().cuda().squeeze())
        self.planarSplat.check_model()
        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer_plane.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer_plane.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer_plane.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer_plane.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self, new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z, new_rot_delta=[]):
        d = {"plane_center": new_plane_center,
            "plane_radii_xy_p": new_plane_radii_xy_p,
            "plane_radii_xy_n": new_plane_radii_xy_n,
            "plane_rot_q_normal_wxy": new_plane_rot_q_normal_wxy,
            "plane_rot_q_xyAxis_w" : new_plane_rot_q_xyAxis_w,
            "plane_rot_q_xyAxis_z" : new_plane_rot_q_xyAxis_z,
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self.planarSplat._plane_center = optimizable_tensors["plane_center"]
        self.planarSplat._plane_radii_xy_p = optimizable_tensors["plane_radii_xy_p"]
        self.planarSplat._plane_radii_xy_n = optimizable_tensors["plane_radii_xy_n"]
        self.planarSplat._plane_rot_q_normal_wxy = optimizable_tensors["plane_rot_q_normal_wxy"]
        self.planarSplat._plane_rot_q_xyAxis_w = optimizable_tensors["plane_rot_q_xyAxis_w"]
        self.planarSplat._plane_rot_q_xyAxis_z = optimizable_tensors["plane_rot_q_xyAxis_z"]

        # reset all accum stats after densification
        self.reset_grad_stats()

        if self.planarSplat.rot_delta is not None and len(new_rot_delta) > 0:
            self.planarSplat.rot_delta = torch.cat([self.planarSplat.rot_delta, new_rot_delta], dim=0)

    def split_planes_via_radii_grad(self, radii_ratio=1.5):
        grads_radii = self.radii_grad_accum / self.radii_grad_denom
        grads_radii[grads_radii.isnan()] = 0.0
        assert self.planarSplat.get_plane_num() == grads_radii.shape[0]
        _, _, plane_center, plane_radii, _, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_geometry()
        plane_radii_xy_p = plane_radii[:, :2]
        plane_radii_xy_n = plane_radii[:, 2:]
        plane_rot_q_normal_wxy = self.planarSplat.get_plane_rot_q_normal_wxy
        plane_rot_q_xyAxis_w = self.planarSplat.get_plane_rot_q_xyAxis_w
        plane_rot_q_xyAxis_z = self.planarSplat.get_plane_rot_q_xyAxis_z

        # Extract planes that satisfy the gradient condition
        x_split_mask, y_split_mask = model_util.get_split_mask_via_radii_grad(
            grads_radii, plane_radii_xy_p, plane_radii_xy_n, radii_ratio, self.planarSplat.radii_min_list[-1], self.split_thres)
        selected_mask_1 = torch.logical_and(y_split_mask, ~x_split_mask)
        selected_mask_2 = torch.logical_and(x_split_mask, ~y_split_mask)
        selected_mask_3 = torch.logical_and(x_split_mask, y_split_mask)
        selected_mask = selected_mask_1 | selected_mask_2 | selected_mask_3

        new_plane_center,new_plane_radii_xy_p,new_plane_radii_xy_n,new_plane_rot_q_normal_wxy,new_plane_rot_q_xyAxis_w,new_plane_rot_q_xyAxis_z,new_rot_delta = model_util.split_planes_via_mask(
            selected_mask_1, selected_mask_2, selected_mask_3, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z, self.planarSplat.rot_delta)

        if len(new_plane_center) > 0:
            if self.planarSplat.rot_delta is not None:
                new_rot_delta = torch.cat(new_rot_delta, dim=0)
            self.densification_postfix(new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z, new_rot_delta)
        
        self.planarSplat.check_model()
        self.reset_grad_stats()
        self.reset_plane_vis()
        if len(new_plane_center) > 0:
            prune_mask = torch.cat((selected_mask, torch.zeros(new_plane_center.shape[0], device="cuda", dtype=bool)))
            self.prune_core(prune_mask)
        torch.cuda.empty_cache()

    def split_all_plane(self,):
        self.regularize_plane_shape()
        _, _, plane_center, plane_radii, _, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_geometry()
        plane_num = self.planarSplat.get_plane_num()
        plane_radii_xy_p = plane_radii[:, :2]
        plane_radii_xy_n = plane_radii[:, 2:]
        plane_rot_q_normal_wxy = self.planarSplat.get_plane_rot_q_normal_wxy
        plane_rot_q_xyAxis_w = self.planarSplat.get_plane_rot_q_xyAxis_w
        plane_rot_q_xyAxis_z = self.planarSplat.get_plane_rot_q_xyAxis_z

        x_split_mask = torch.ones(plane_num).cuda() > 0
        y_split_mask = torch.ones(plane_num).cuda() > 0
        selected_mask_1 = torch.logical_and(y_split_mask, ~x_split_mask)
        selected_mask_2 = torch.logical_and(x_split_mask, ~y_split_mask)
        selected_mask_3 = torch.logical_and(x_split_mask, y_split_mask)
        selected_mask = selected_mask_1 | selected_mask_2 | selected_mask_3

        new_plane_center,new_plane_radii_xy_p,new_plane_radii_xy_n,new_plane_rot_q_normal_wxy,new_plane_rot_q_xyAxis_w,new_plane_rot_q_xyAxis_z,new_rot_delta = model_util.split_planes_via_mask(
            selected_mask_1, selected_mask_2, selected_mask_3, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z, self.planarSplat.rot_delta)

        if len(new_plane_center) > 0:
            if self.planarSplat.rot_delta is not None:
                new_rot_delta = torch.cat(new_rot_delta, dim=0)
            self.densification_postfix(new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z, new_rot_delta)
        
        self.planarSplat.check_model()
        self.reset_grad_stats()
        self.reset_plane_vis()
        if len(new_plane_center) > 0:
            prune_mask = torch.cat((selected_mask, torch.zeros(new_plane_center.shape[0], device="cuda", dtype=bool)))
            self.prune_core(prune_mask)
        self.prune_small_plane()
        torch.cuda.empty_cache()


    def split_plane(self,):
        self.regularize_plane_shape()
        self.split_planes_via_radii_grad()
        self.prune_small_plane()
        self.reset_plane_vis()
        self.reset_grad_stats()
        torch.cuda.empty_cache()

    def reset_plane_vis(self):
        self.plane_vis_denom = torch.zeros((self.planarSplat.get_plane_num(), 1), device="cuda")

    def update_plane_vis(self, vis_mask=None):
        if vis_mask is None:
            vis_mask = self.planarSplat._plane_center.grad.abs().detach().sum(dim=-1) > 0

        assert vis_mask.shape[0] == self.plane_vis_denom.shape[0]
        self.plane_vis_denom[vis_mask] += 1
   
    def reset_grad_stats(self):
        self.radii_grad_denom = torch.zeros((self.planarSplat.get_plane_num(), 1), device="cuda")
        self.radii_grad_accum = torch.zeros((self.planarSplat.get_plane_num(), 4), device="cuda")

    def update_grad_stats(self, vis_mask=None):
        if self.planarSplat._plane_radii_xy_p.grad is not None:
            if vis_mask is None:
                if self.planarSplat._plane_center.grad is not None:
                    vis_mask = self.planarSplat._plane_center.grad.abs().detach().sum(dim=-1) > 0
                else:
                    vis_mask = torch.ones(self.planarSplat.get_plane_num()).cuda() > 0
            # self.radii_grad_denom[vis_mask] += 1
            self.radii_grad_denom = self.radii_grad_denom + vis_mask.float().view(-1, 1)
            if self.radii_dir_type == 'double':
                # self.radii_grad_accum[vis_mask] += torch.cat([self.planarSplat._plane_radii_xy_p.grad, self.planarSplat._plane_radii_xy_n.grad], dim=-1).abs().detach()[vis_mask]
                self.radii_grad_accum = self.radii_grad_accum + torch.cat([self.planarSplat._plane_radii_xy_p.grad, self.planarSplat._plane_radii_xy_n.grad], dim=-1).abs().detach() * vis_mask.float().view(-1, 1)
            elif self.radii_dir_type == 'single':
                self.radii_grad_accum[vis_mask] += torch.cat([self.planarSplat._plane_radii_xy_p.grad, self.planarSplat._plane_radii_xy_p.grad], dim=-1).abs().detach()[vis_mask]
            else:
                raise NotImplementedError
            
    def regularize_plane_shape(self, empty_cache=False):
        with torch.no_grad():
            plane_radii, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_geometry_for_regularize()
            plane_center_x_offset = (plane_radii[:, 0] - plane_radii[:, 2]).unsqueeze(-1) / 2.
            plane_center_y_offset = (plane_radii[:, 1] - plane_radii[:, 3]).unsqueeze(-1) / 2.
            plane_center_offset_new = plane_center_x_offset * plane_xAxis + plane_center_y_offset * plane_yAxis
            plane_radii_x_new = (plane_radii[:, 0] + plane_radii[:, 2]) / 2.
            plane_radii_y_new = (plane_radii[:, 1] + plane_radii[:, 3]) / 2.
            plane_radii_new = torch.stack([plane_radii_x_new, plane_radii_y_new], dim=-1)
            plane_center_new = self.planarSplat._plane_center + plane_center_offset_new

        # reset plane parameters
        opt_tensors = self.replace_tensor_to_optimizer(plane_center_new.detach(), 'plane_center')
        self.planarSplat._plane_center = opt_tensors['plane_center']
        opt_tensors = self.replace_tensor_to_optimizer(plane_radii_new.detach(), 'plane_radii_xy_p')
        self.planarSplat._plane_radii_xy_p = opt_tensors['plane_radii_xy_p']
        opt_tensors = self.replace_tensor_to_optimizer(plane_radii_new.detach(), 'plane_radii_xy_n')
        self.planarSplat._plane_radii_xy_n = opt_tensors['plane_radii_xy_n']
        if empty_cache:
            torch.cuda.empty_cache()

    def find_min_theta_dist_line(self, lines, plane_corners_2d, use_theta=True, use_dist=True):
        if use_theta and use_dist:
            # theta
            N = lines.shape[0]
            vec_lines = lines[:, 2:4] - lines[:, :2]
            vec_planes = []
            vec_planes = torch.concat([plane_corners_2d[:,(i+1)%4].unsqueeze(1)-plane_corners_2d[:,i].unsqueeze(1) for i in range(4)], dim=1)

            vec_lines_exp = vec_lines.unsqueeze(1).cuda()
            cos_theta = torch.abs((vec_lines_exp * vec_planes).sum(dim=-1)) / (torch.norm(vec_lines_exp, dim=-1) * torch.norm(vec_planes, dim=-1) + 1e-8)
            theta_all = torch.acos(cos_theta.clamp(-1,1))
            theta_, indices = torch.topk(-theta_all, k=2, dim=1)

            # dist
            plane_lines_2d = torch.concat([torch.concat([plane_corners_2d[:,i],plane_corners_2d[:,(i+1)%4]], dim=1).unsqueeze(1) for i in range(4)], dim=1)
            dist1 = model_util.calculate_lines_2d_dist(plane_lines_2d[torch.arange(N), indices[:,0]], lines[:,:4].cuda(), 'l').mean(dim=1)
            dist2 = model_util.calculate_lines_2d_dist(plane_lines_2d[torch.arange(N), indices[:,1]], lines[:,:4].cuda(), 'l').mean(dim=1)
            dists = torch.stack([dist1, dist2], dim=1)
            dist, dist_indix = dists.min(dim=1)
            final_indix = indices[torch.arange(indices.shape[0], device=indices.device), dist_indix]
            theta = theta_all[torch.arange(theta_all.shape[0], device=theta_all.device), final_indix]

        elif use_theta and not use_dist:
            # theta
            N = lines.shape[0]
            vec_lines = lines[:, 2:4] - lines[:, :2]
            vec_planes = []
            vec_planes = torch.concat([plane_corners_2d[:,(i+1)%4].unsqueeze(1)-plane_corners_2d[:,i].unsqueeze(1) for i in range(4)], dim=1)

            vec_lines_exp = vec_lines.unsqueeze(1).cuda()
            cos_theta = torch.abs((vec_lines_exp * vec_planes).sum(dim=-1)) / (torch.norm(vec_lines_exp, dim=-1) * torch.norm(vec_planes, dim=-1) + 1e-8)
            theta_all = torch.acos(cos_theta.clamp(-1,1))

            theta, theta_indix = theta_all.min(dim=-1)

            # dist
            plane_lines_2d = torch.concat([torch.concat([plane_corners_2d[:,i],plane_corners_2d[:,(i+1)%4]], dim=1).unsqueeze(1) for i in range(4)], dim=1)
            dist = model_util.calculate_lines_2d_dist(plane_lines_2d[torch.arange(N),theta_indix], lines[:,:4].cuda(), 'l').mean(dim=1)
            final_indix = theta_indix

        elif use_dist and not use_theta:
            plane_lines_2d = torch.concat([torch.concat([plane_corners_2d[:,i],plane_corners_2d[:,(i+1)%4]], dim=1).unsqueeze(1) for i in range(4)], dim=1)
            dists = torch.stack([model_util.calculate_lines_2d_dist(plane_lines_2d[:,i], lines[:,:4].cuda(), 'l').mean(dim=1) for i in range(4)], dim=1)
            dist, final_indix = dists.min(dim=1)

            N = lines.shape[0]
            vec_lines = lines[:, 2:4] - lines[:, :2]
            vec_lines = vec_lines.cuda()
            vec_planes = torch.concat([plane_corners_2d[torch.arange(N),(final_indix+1)%4]-plane_corners_2d[torch.arange(N),final_indix]], dim=1)
            cos_theta = torch.abs((vec_lines * vec_planes).sum(dim=-1)) / (torch.norm(vec_lines, dim=-1) * torch.norm(vec_planes, dim=-1) + 1e-8)
            theta = torch.acos(cos_theta.clamp(-1,1))

        # random selection
        else:
            AssertionError('Either use_theta or use_dist should be True!')

        '''
        # for i in range(lines.shape[0]):
        #     import pdb;pdb.set_trace()
        #     mid = theta_indix[i]
        #     li = lines[i][:4].numpy()
        #     pli = plane_corners_2d[i].cpu().numpy()
        #     mpli = np.concatenate([pli[mid], pli[(mid+1)%4]])
        #     plt.plot([li[0],li[2]], [li[1],li[3]], 'r-')
        #     for k in range(4):
        #         plt.plot([pli[k][0], pli[(k+1)%4][0]], [pli[k][1], pli[(k+1)%4][1]], 'b-')
        #     plt.plot([mpli[0],mpli[2]], [mpli[1],mpli[3]], 'y-')
        #     plt.show()
        '''

        return theta, dist, final_indix

    def get_inter_points_lines_theta_dist(self, view_info, ite=-1):
        H, W = view_info.img_size
        mask_data = view_info.get_sampling_gt()
        uv = mask_data['uv'][None]
        lines = mask_data['lines']
        pose = mask_data['pose'][None]
        intrinsics = mask_data['intrinsics'][None]
        image = view_info.rgb.cpu().numpy()

        rays_d, rays_o = model_util.get_raydir_camloc(uv, pose, intrinsics)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = rays_o.expand(rays_d.shape[0], 3)

        # prune Nan planes
        # self.prune_invalid_plane()
        plane_corners = self.planarSplat.get_plane_vertex(ite)

        with torch.no_grad():
            plane_normal, plane_center, plane_radii, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_property(ite)
            if ~(plane_radii.max()>0):
                import pdb;pdb.set_trace()
            #     self.prune_invalid_plane()
            inter, o2inter_SD = model_util.compute_ray2plane_intersections_v2(rays_o, rays_d, plane_center, plane_normal)
            # check if the plane is intersected with rays
            mask_i = model_util.check_intersections_in_plane(inter, plane_center, plane_radii, plane_xAxis, plane_yAxis)

            # check if the plane is intersected with rays in front of the camera (s.t. positive direction)
            mask_p = o2inter_SD > 0
            mask_ip = mask_i * mask_p

            # only optimize the plane first hitted by rays
            mask, min_idx = model_util.find_positive_min_mask(o2inter_SD * mask_ip)  # [N_ray, N_plane] -> [N_ray]

            line_plane_corners = plane_corners[min_idx[mask]]
            lines_gt = lines[mask.cpu()]

            line_plane_corners_2d = model_util.project2D(pose, intrinsics, line_plane_corners)
            theta, dist, final_indix = self.find_min_theta_dist_line(lines_gt, line_plane_corners_2d)

        all_plane_lines_3d = model_util.get_plane_lines_from_corners(plane_corners[min_idx[mask]])
        plane_lines_3d = all_plane_lines_3d[torch.arange(len(line_plane_corners)), final_indix]
        plane_lines_2d = model_util.project2D(pose, intrinsics, plane_lines_3d.reshape(-1,2,3)).reshape(-1,4)
        lines_gt = lines_gt[:,:4].cuda()

        with torch.no_grad():
            dist1 = torch.sum((plane_lines_2d-lines_gt)**2,dim=-1,keepdim=True).detach()
            dist2 = torch.sum((plane_lines_2d[:,[2,3,0,1]]-lines_gt)**2,dim=-1,keepdim=True).detach()

        plane_lines_3d = torch.where(dist1<dist2, plane_lines_3d, plane_lines_3d[:,[3,4,5,0,1,2]])
        plane_lines_2d = torch.where(dist1<dist2, plane_lines_2d, plane_lines_2d[:,[2,3,0,1]])

        outputs = {
            "inter_points_3d": inter,
            "inter_SD": o2inter_SD,
            "mask_inlier": mask_i,
            "mask_positive": mask_p,
            "mask_inter": mask,
            "mask_plane": min_idx,
            "plane_lines_3d": plane_lines_3d,
            "plane_lines_2d": plane_lines_2d,
            "lines_gt_2d": lines_gt,
            "theta": theta,
            "dist": dist,
        }

        return outputs

    def calculate_line_inter_loss(self, view_info, ite=-1):
        loss_line = 0.
        loss_line_dict = {}

        outputs = self.get_inter_points_lines_theta_dist(view_info, ite)
        plane_lines_2d = outputs["plane_lines_2d"]
        plane_lines_3d = outputs["plane_lines_3d"]
        lines_tgt = outputs["lines_gt_2d"]

        if plane_lines_2d.shape[0] == 0:
            loss_juncs_2d = torch.tensor(0., device='cuda:0')
            loss_lines_2d = torch.tensor(0., device='cuda:0')
            # loss_lines_3d = torch.tensor(0., device='cuda:0')
        else:
            loss_juncs_2d = loss_util.lines_edps_loss_2d(plane_lines_2d, lines_tgt, view_info.img_size) # 2d .. loss
            loss_lines_2d = loss_util.lines_loss_2d(plane_lines_2d, lines_tgt) # 2d // loss
            # loss_lines_3d = loss_util.lines_loss_3d(plane_lines_2d, lines_tgt, plane_lines_3d) * 0.1 # 3d // loss
        loss_line += loss_juncs_2d
        loss_line += loss_lines_2d
        # loss_line += loss_lines_3d

        loss_line_dict.update({
            'loss_line': loss_line.item(),
            'loss_juncs_2d': loss_juncs_2d.item(),
            'loss_lines_2d': loss_lines_2d.item(),
            # 'loss_lines_3d': loss_lines_3d.item(),
        })

        return loss_line, loss_line_dict

    def calculate_plane_loss(self, view_info, iter=-1):
        allmap = self.planarSplat(view_info, iter)
        
        # get rendered maps
        raster_cam_w2c = view_info.raster_cam_w2c
        depth = allmap[0:1].view(-1)
        self.depth = depth
        normal_local_ = allmap[2:5]
        normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
        # get aux maps
        vis_weight = allmap[1:2].view(-1)
        valid_ray_mask = vis_weight > 0.00001
        
        loss_mono_depth = self.plane_depth_loss(depth, view_info.mono_depth, valid_ray_mask)
        loss_normal_l1, loss_normal_cos = self.plane_normal_loss(normal_global, view_info.mono_normal_global, valid_ray_mask)
        loss_plane = (loss_mono_depth * 1.0) * self.weight_plane_depth \
                            + (loss_normal_l1 + loss_normal_cos) * self.weight_plane_normal
        loss_plane_dict = {
            'loss_plane': loss_plane.item(),
            'loss_plane_depth': loss_mono_depth.item(),
            'loss_plane_normal_l1': loss_normal_l1.item(),
            'loss_plane_normal_cos': loss_normal_cos.item(),
        }

        return loss_plane, loss_plane_dict

    def calculate_loss(self, view_info, decay, iter=-1):
        loss_final = 0.
        loss_final_dict = {}

        # ======================================= calculate plane losses
        loss_plane, loss_plane_dict = self.calculate_plane_loss(view_info, iter)
        loss_final += loss_plane * decay * self.weight_plane
        loss_final_dict.update({'loss_plane': loss_plane.item()})

        # ======================================= calculate line intersection losses
        loss_line, loss_line_dict = self.calculate_line_inter_loss(view_info, iter)
        loss_final += loss_line * self.weight_line
        loss_final_dict.update(loss_line_dict)

        loss_final_dict.update({'total_loss': loss_final.item()})
        torch.cuda.empty_cache()
    
        return loss_final, loss_final_dict

    def check_interacted_plane_mask(self, view_info):
        mask_data = view_info.get_sampling_gt()
        uv = mask_data['uv'][None]
        pose = mask_data['pose'][None]
        intrinsics = mask_data['intrinsics'][None]

        rays_d, rays_o = model_util.get_raydir_camloc(uv, pose, intrinsics)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = rays_o.expand(rays_d.shape[0], 3)

        plane_normal, plane_center, plane_radii, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_property()
        inter, o2inter_SD = model_util.compute_ray2plane_intersections_v2(rays_o, rays_d, plane_center, plane_normal)
        # check if the plane is intersected with rays
        mask_i = model_util.check_intersections_in_plane(inter, plane_center, plane_radii, plane_xAxis, plane_yAxis)

        # check if the plane is intersected with rays in front of the camera (s.t. positive direction)
        mask_p = o2inter_SD > 0
        mask_ip = mask_i * mask_p
        valid_plane_mask = mask_ip.sum(dim=0) > 0

        return valid_plane_mask
    
    def forward_global_3dlines_theta_dist(self, view_info_list, distance_threshold=5, theta_threshold=0.01, merge=0, depth_trunc=0):
        '''
        Get the final 3D lines from all views and plane primitives based on the thresholds of 2D distance and angle (theta).
        Args:
            view_info_list: list of ViewInfo objects containing view information.
            distance_threshold: threshold for 2D distance between projected plane lines and ground truth lines.
            theta_threshold: threshold for angle (theta) between projected plane lines and ground truth lines.
            merge: value in [0,'local', 'global']. Default is 0, meaning no merging.
                'local': if set, merge lines from different planes within the same view based on their labels.
                'global': if set, merge lines from different planes and views based on their global labels.
        Returns:
            all_lines_3d: list of tensors containing final 3D lines for each view
        '''
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=1e-2, min_samples=1, metric='precomputed')
        all_lines_3d = []
        lines_dict = []
        group_dict = []
        group_num = 0

        with torch.no_grad():
            plane_corners = self.planarSplat.get_plane_vertex()
            plane_normal, plane_center, plane_radii, plane_xAxis, plane_yAxis = self.planarSplat.get_plane_property()
            
            self.planarSplat.mask=torch.zeros(plane_center.shape[0],1)

            for view_info in tqdm(view_info_list, desc="inference process..."):
                mask_data = view_info.get_sampling_gt()
                uv = mask_data['uv'][None]
                lines = mask_data['lines']
                pose = mask_data['pose'][None]
                intrinsics = mask_data['intrinsics'][None]
                labels = mask_data['labels']

                rays_d, rays_o = model_util.get_raydir_camloc(uv, pose, intrinsics)
                rays_d = rays_d.reshape(-1, 3)
                rays_o = rays_o.expand(rays_d.shape[0], 3)

                # inter, o2inter_SD = model_util.compute_ray2plane_intersections_v2(rays_o, rays_d, plane_center, plane_normal)
                # # check if the plane is intersected with rays
                # mask_i = model_util.check_intersections_in_plane(inter, plane_center, plane_radii, plane_xAxis, plane_yAxis)
                # # check if the plane is intersected with rays in front of the camera (s.t. positive direction)
                # mask_p = o2inter_SD > 0
                # mask_ip = mask_i * mask_p
                # # only optimize the plane first hitted by rays
                # mask, min_idx = model_util.find_positive_min_mask(o2inter_SD * mask_ip)  # [N_ray, N_plane] -> [N_ray]

                mask = []
                min_idx = []
                 # process in chunks to avoid OOM
                chunk = 10000 * 10000 // plane_corners.shape[0]
                for i in range(0, rays_o.shape[0], chunk):
                    inter, o2inter_SD = model_util.compute_ray2plane_intersections_v2(rays_o[i:i+chunk], rays_d[i:i+chunk], plane_center, plane_normal)
                    # check if the plane is intersected with rays
                    mask_i = model_util.check_intersections_in_plane(inter, plane_center, plane_radii, plane_xAxis, plane_yAxis)
                    if depth_trunc > 0:
                        mask_p = (o2inter_SD > 0) & (o2inter_SD < depth_trunc)
                    else:
                        mask_p = o2inter_SD > 0
                    mask_ip = mask_i * mask_p
                    # only optimize the plane first hitted by rays
                    mask_c, min_idx_c = model_util.find_positive_min_mask(o2inter_SD * mask_ip)  # [N_ray, N_plane] -> [N_ray]
                    mask.append(mask_c)
                    min_idx.append(min_idx_c)
                mask = torch.cat(mask, dim=0)
                min_idx = torch.cat(min_idx, dim=0)

                line_plane_corners = plane_corners[min_idx[mask]]
                lines_gt = lines[mask.cpu()]
                labels_gt = labels[mask.cpu()]

                line_plane_corners_2d = model_util.project2D(pose, intrinsics, line_plane_corners)
                theta, dist, final_indix = self.find_min_theta_dist_line(lines_gt, line_plane_corners_2d)

                all_plane_lines_3d = model_util.get_plane_lines_from_corners(plane_corners[min_idx[mask]])
                plane_lines_3d = all_plane_lines_3d[torch.arange(len(line_plane_corners)), final_indix]
                    
                final_mask = (dist<=distance_threshold) & (theta<=theta_threshold)

                self.planarSplat.mask[min_idx[mask][final_mask].unique(dim=0).detach().cpu()]=1


                if not merge:
                    final_plane_lines_3d = plane_lines_3d[final_mask]
                    final_plane_lines_3d = torch.unique(final_plane_lines_3d, dim=0)
                    all_lines_3d.append(final_plane_lines_3d)
                else:
                    final_labels_gt = labels_gt[final_mask.cpu()]
                    uniq_labels = final_labels_gt.unique()
                    counts = (final_labels_gt[:,None]==uniq_labels[None]).sum(dim=0)
                    temp_lines_3d = []
                    for i in range(len(counts)):
                        c = counts[i]
                        temp_mask = final_labels_gt==uniq_labels[i]
                        segments = plane_lines_3d[final_mask][temp_mask].cpu().numpy()
                        if np.unique(segments, axis=0).shape[0] >= 2:
                            p1 = segments[:, :3]  # (N, 3)
                            p2 = segments[:, 3:]  # (N, 3)
                            # ================= #
                            # # coarse strategy 1: only use two endpoints
                            # vec1 = p1[:,None] - p1[None] # (N, N, 3)
                            # vec2 = p1[:,None] - p2[None] # (N, N, 3)
                            # vecl = p1 - p2  # (N, 3)
                            # d_p1 = np.linalg.norm(np.cross(vec1, vecl[None], axis=-1), axis=-1) / np.linalg.norm(vecl, axis=-1)[None]  # (N, N)
                            # d_p2 = np.linalg.norm(np.cross(vec2, vecl[None], axis=-1), axis=-1) / np.linalg.norm(vecl, axis=-1)[None]  # (N, N)                            
                            # d_matrix = np.maximum(d_p1, d_p2)
                            # d_matrix = np.minimum(d_p1, d_p2)
                            # ================= #
                            # stable strategy 2: uniformly sample points along the line segments
                            t = np.linspace(0, 1, num=10)
                            p = p1[:, None, :] * (1 - t[None, :, None]) + p2[:, None, :] * t[None, :, None]  # (N, 100, 3)
                            vec1 = p[:, :, None, :] - p1[None, None]  # (N, 100, N, 3)
                            vec2 = p[:, :, None, :] - p2[None, None]  # (N, 100, N, 3)
                            vecl = p1 - p2  # (N, 3)
                            d_matrix = np.mean(np.linalg.norm(np.cross(vec1, vec2, axis=-1), axis=-1) / np.linalg.norm(vecl, axis=-1)[None,None], axis=1)  # (N, N)
                            clusters = clustering.fit_predict(d_matrix)
                            for clus_id in np.unique(clusters):
                                segs_cluster = segments[clusters==clus_id]
                                if len(segs_cluster) >= 2:
                                    # locally merge lines in the cluster
                                    if merge == 'local':
                                        merged_line = model_util.merge_segments(segs_cluster, use_pca=True)
                                        temp_lines_3d.append(merged_line)
                                    # record lines and group id for finally global merging
                                    else:
                                        if len(lines_dict) == 0:
                                            for seg in np.unique(segs_cluster, axis=0):
                                                lines_dict.append(seg)
                                                group_dict.append(group_num)
                                            group_num += 1
                                        else:
                                            flag = 0
                                            idx_num = 0
                                            add_lines = []
                                            for seg in np.unique(segs_cluster, axis=0):
                                                if (np.all(np.array(lines_dict) == seg, axis=1)).sum() > 0:
                                                    flag += 1
                                                    idx_num = np.array(group_dict)[np.all(np.array(lines_dict) == seg, axis=1)][0]
                                                else:
                                                    add_lines.append(seg)
                                            if flag == 0:
                                                for k in range(len(add_lines)):
                                                    lines_dict.append(add_lines[k])
                                                    group_dict.append(group_num)
                                                group_num += 1
                                            else:
                                                for k in range(len(add_lines)):
                                                    lines_dict.append(add_lines[k])
                                                    group_dict.append(idx_num)
                                else:
                                    if merge == 'local':
                                        temp_lines_3d.append(segs_cluster[0])
                                    else:
                                        seg = segs_cluster[0]
                                        if len(lines_dict) == 0 or (np.all(np.array(lines_dict) == seg, axis=1)).sum() <= 0:
                                            lines_dict.append(seg)
                                            group_dict.append(group_num)
                                            group_num += 1
                        else:
                            if merge == 'local':
                                temp_lines_3d.append(segments[0])
                            else:
                                seg = segments[0]
                                if len(lines_dict) == 0 or (np.all(np.array(lines_dict) == seg, axis=1)).sum() <= 0:
                                    lines_dict.append(seg)
                                    group_dict.append(group_num)
                                    group_num += 1

                    if merge == 'local':
                        final_plane_lines_3d = torch.unique(torch.tensor(np.array(temp_lines_3d)).cuda(), dim=0)
                        all_lines_3d.append(final_plane_lines_3d)
                
            if not merge:
                final_lines_3d = torch.unique(torch.concat(all_lines_3d, dim=0), dim=0)
            else:
                if merge == 'local':
                    final_lines_3d = torch.unique(torch.concat(all_lines_3d, dim=0), dim=0)
                else:
                    for gid in np.unique(group_dict):
                        group_lines = np.array(lines_dict)[np.array(group_dict)==gid]
                        if len(group_lines) >= 2:
                            merged_line = model_util.merge_segments(group_lines, use_pca=True)
                        else:
                            merged_line = group_lines[0]
                        all_lines_3d.append(merged_line)
                    final_lines_3d = torch.unique(torch.tensor(np.array(all_lines_3d)), dim=0)

            torch.cuda.empty_cache()

        return final_lines_3d
       
    def refinement(self, lines3d, mesh_path):
        mesh_gt = o3d.io.read_triangle_mesh(mesh_path)
        pcd_gt = o3d.geometry.PointCloud(mesh_gt.vertices)
        points = np.asarray(pcd_gt.points)
        kdtree = KDTree(points)
        edps = lines3d.reshape(-1,3).cpu().numpy()
        distances, indices = kdtree.query(edps)
        edps_rf = points[indices[:,0]].astype(np.float32)
        lines3d_rf = torch.from_numpy(edps_rf).to(lines3d.device).view(-1,6)

        return lines3d_rf

    def forward(self, dataset, distance_threshold=5, theta_threshold=0.01, merge=0, refine=False):
        final_lines_3d = self.forward_global_3dlines_theta_dist(dataset.view_info_list, distance_threshold, theta_threshold, merge)

        if refine:
            if os.path.exists(dataset.mesh_path) and dataset.mesh_path is not None:
                final_lines_3d = self.refinement(final_lines_3d, dataset.mesh_path)

        return final_lines_3d