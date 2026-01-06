import os
from datetime import datetime
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import glob
from .network import RecWrapper
from random import randint
import math
import shutil
import open3d as o3d
import matplotlib.pyplot as plt

from utils import misc_util
from utils import loss_util
from utils import conf_util
from utils import dataIO_util
from utils import mesh_util

from loguru import logger
from collections import defaultdict

class AverageMeter(object):
    def __init__(self):
        self.loss_dict = defaultdict(list)
        
    @torch.no_grad()
    def push(self, loss_dict):
        with torch.no_grad():
            for key, val in loss_dict.items():
                self.loss_dict[key].append(val)
    @torch.no_grad()
    def __call__(self):
        out_dict = {}
        for key, val in self.loss_dict.items():
            out_dict[key] = sum(val)/len(val)
        
        return out_dict

class Trainer():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        self.debug = kwargs.get('debug', False)
        self.conf = kwargs['conf']
        self.mode = self.conf.get_string('train.mode', default='joint')
        self.use_plane = True
        self.expname, _, self.timestamp, is_continue = conf_util.get_train_param(kwargs, self.conf)
        self.expdir, self.plane_plots_dir, self.line_plots_dir, self.checkpoints_path, self.model_subdir = dataIO_util.prepare_folders(kwargs, self.expname, self.timestamp)
        misc_util.setup_logging(os.path.join(self.expdir, 'train.log'))

        ###
        self.conf.put('dataset.initial_mesh_root', self.expdir)
        
        # =======================================  loading dataset
        logger.info('Loading data for training...')
        self.dataset = conf_util.get_class(self.conf.get_string('train.dataset_class'))(**self.conf.get_config('dataset'))
        self.ds_len = self.dataset.n_images
        self.H, self.W = self.dataset.img_res
        self.conf.put("dataset.img_res", self.dataset.img_res)
        logger.info('Data loaded. Frame number = {0}'.format(self.ds_len))
        logger.info('Shell command : {0}'.format(' '.join(sys.argv)))
        dataIO_util.save_config_files(self.expdir, self.conf)

        # =======================================  build plane model
        self.plane_model_conf = self.conf.get_config('plane_model')
        self.net = RecWrapper(self.conf, self.plane_plots_dir)
        self.net.expdir = self.expdir

        if 'resume_path' in kwargs and kwargs['resume_path'] is not None:
            if isinstance(kwargs['resume_path'], int) or kwargs['resume_path']=='latest':
                resume_path = os.path.join(self.checkpoints_path, self.model_subdir, f'{kwargs["resume_path"]}.pth')
            else:
                resume_path = kwargs['resume_path']
            # self.start_iter = self.load_model(kwargs['resume_path'])
            self.start_iter = self.load_model(resume_path)
        else:
            resume_path_latest = os.path.join(self.checkpoints_path, self.model_subdir, 'latest.pth')
            if os.path.exists(os.path.join(resume_path_latest)):
                self.start_iter = self.load_model(resume_path_latest)
            else:
                self.start_iter = 0
        self.iter_step = self.start_iter

        self.net = self.net.cuda()
        self.net.build_optimizer_and_LRscheduler()

        # ======================================= plot settings
        self.do_vis = kwargs['do_vis']
        self.plot_freq = self.conf.get_int('train.plot_freq')        
        
        # ======================================= training settings
        self.max_total_iters = self.conf.get_int('train.max_total_iters')
        self.process_plane_freq_ite = self.ds_len
        self.split_start_ite = self.ds_len
        self.check_vis_freq_ite = self.conf.get_int('train.check_plane_vis_freq_ite')
        self.data_order = self.conf.get_string('train.data_order')
        self.save_ckpt_freq_ite = self.conf.get_int('train.save_ckpt_freq_ite')
    
    def load_model(self, ckpt):
        logger.info(f'resuming model from PATH: {ckpt}')
        assert os.path.exists(ckpt)
        saved_model_state = torch.load(ckpt) 
        if self.use_plane:
            plane_num = saved_model_state["model_state_dict"]['planarSplat._plane_center'].shape[0]
            self.net.planarSplat.initialize_as_zero(plane_num)
            self.net.reset_plane_vis()
            self.net.reset_grad_stats()
        self.net.load_state_dict(saved_model_state["model_state_dict"])
        latest_iter = saved_model_state["iter"]
        return latest_iter + 1

    def save_checkpoints(self, iter, only_latest=False):
        torch.save(
                {"iter": iter, "model_state_dict": self.net.state_dict()},
                os.path.join(self.checkpoints_path, self.model_subdir, "latest.pth"))
        if not only_latest:
            torch.save(
                {"iter": iter, "model_state_dict": self.net.state_dict()},
                os.path.join(self.checkpoints_path, self.model_subdir, str(iter) + ".pth"))

    def run(self):
        self.train()

    def train(self):
        weight_decay_list = []
        for i in tqdm(range(self.max_total_iters+1), desc="generating sampling idx list..."):
            weight_decay_list.append(max(math.exp(-i / self.max_total_iters), 0.1))
        logger.info("Training...")
        logger.info('Start training at {:%Y_%m_%d_%H_%M_%S}'.format(datetime.now()))
        self.net.train()

        view_info_list = None
        progress_bar = tqdm(range(self.start_iter, self.max_total_iters+1), desc="Training progress")
        self.calculate_plane_depth()
        loss_final_meters = AverageMeter()

        for iter in tqdm(range(self.start_iter, self.max_total_iters + 1)):

            self.iter_step = iter

            if iter == 0:
                self.check_plane_visibility_cuda()

            # ======================================= process planes
            if iter % self.process_plane_freq_ite==0:
                self.net.regularize_plane_shape()
                self.net.prune_small_plane()
                if iter > self.split_start_ite and iter <= self.max_total_iters - 1000:
                    logger.info('splitting...')
                    ori_num = self.net.planarSplat.get_plane_num()
                    self.net.split_plane()
                    new_num = self.net.planarSplat.get_plane_num()
                    logger.info(f'plane num: {ori_num} ---> {new_num}')

            # ======================================= get view info
            if not view_info_list:
                view_info_list = self.dataset.view_info_list.copy()
            if self.data_order == 'rand':
                view_info = view_info_list.pop(randint(0, len(view_info_list)-1))
            else:
                view_info = view_info_list.pop(0)
                           
            # # ======================================= zero grad
            self.net.optimizer_zero_grad()
            # # ======================================= calculate losses
            decay = weight_decay_list[iter]
            loss_final, loss_final_dict = self.net.calculate_loss(view_info, decay, self.iter_step)
            loss_final_meters.push(loss_final_dict)
            loss_final.backward()
            self.net.optimizer_update()
           
            # ======================================= print loss dict
            if iter % self.ds_len == 0:
                loss_final_msg = []
                for key in loss_final_dict.keys():
                    loss_final_msg.append('{}={:.4f}({:.4f})'.format(key, loss_final_dict[key], loss_final_meters()[key]))
                logger.info('{0}/{1} {2} plane_num:{3}'.format(
                    self.iter_step, self.max_total_iters, loss_final_msg, self.net.planarSplat.get_plane_num()
                ))

            with torch.no_grad():
                # Progress bar
                plane_num = self.net.planarSplat.get_plane_num()
                if iter % 10 == 0:
                    loss_dict = {
                        "Planes": f"{plane_num}",
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)
                if iter == self.max_total_iters:
                    progress_bar.close()
        
            # ======================================= plot model outputs
            if self.do_vis and iter % self.plot_freq == 0:
                self.net.regularize_plane_shape()
                self.net.eval()
                self.net.planarSplat.draw_plane(epoch=iter)
                self.plot_plane_img()
                self.net.train()

            if iter > 0 and iter % self.check_vis_freq_ite == 0:
                self.check_plane_visibility_cuda()
            # ======================================= save model
            if iter > 0 and iter % self.save_ckpt_freq_ite == 0 or iter == self.max_total_iters:
                self.save_checkpoints(iter=self.iter_step, only_latest=False)

            torch.cuda.empty_cache()
        

    def check_plane_visibility_cuda(self, joint=False):   
        self.net.regularize_plane_shape(False)     
        logger.info('checking plane visibility...')
        self.net.eval()
        self.net.reset_plane_vis()
        view_info_list = self.dataset.view_info_list.copy()
        for iter in tqdm(range(self.ds_len)):
            # ========================= get view info
            view_info = view_info_list.pop(randint(0, len(view_info_list)-1))

            if joint:
                # jointly
                loss_final, loss_final_dict = self.net.calculate_loss(view_info, self.iter_step)
                loss_final.backward()
            else:
                # only plane
                loss_plane, loss_plane_dict = self.net.calculate_plane_loss(view_info, self.iter_step)
                loss_plane.backward()

            # update plane visibility
            self.net.update_plane_vis() 
            self.net.optimizer_plane.zero_grad()

        self.net.optimizer_plane.zero_grad()
        self.net.train()
        self.net.prune_invisible_plane()
        self.net.planarSplat.draw_plane(epoch=self.iter_step)
    
    def calculate_plane_depth(self):   
        self.net.regularize_plane_shape(False)     
        self.net.eval()
        view_info_list = self.dataset.view_info_list.copy()
        # for iter in tqdm(range(self.ds_len)):
        for iter in range(self.ds_len):
            # ========================= get view info
            view_info = view_info_list[iter]
            # ----------- plane forward
            with torch.no_grad():
                allmap = self.net.planarSplat(view_info, self.iter_step)
            # get rendered maps
            depth = allmap[0:1].view(-1)
            self.dataset.view_info_list[iter].plane_depth = depth.detach()
        self.net.train()
    
    def plot_plane_img(self, plot_img_idx=0, prefix='', pcd_on=True):
        self.net.regularize_plane_shape(False)
        view_info_list = self.dataset.view_info_list.copy()
        if plot_img_idx < 0:
            plot_img_idx = randint(0, len(view_info_list)-1)
        view_info = view_info_list[plot_img_idx]
        raster_cam_w2c = view_info.raster_cam_w2c
        ray_dirs = view_info.ray_dirs
        cam_loc = view_info.cam_loc
        depth_scale = view_info.depth_scale
        allmap, rendered_rgb  = self.net.planarSplat(view_info, self.iter_step, return_rgb=True)

        # get rendered maps
        rendered_depth = allmap[0:1].view(-1)
        normal_local_ = allmap[2:5]
        plane_surf = cam_loc.view(1, 3) + rendered_depth.view(-1, 1) / depth_scale.view(-1, 1) * ray_dirs.view(-1, 3)
        rendered_normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
        rendered_normal_np = (F.normalize(rendered_normal_global, dim=-1).reshape(self.H, self.W, 3)).detach().cpu().numpy()
        rendered_normal_color = ((rendered_normal_np + 1.0) / 2.0 * 255).astype(np.uint8)
        rendered_rgb_np = (rendered_rgb.permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint8)
        rendered_depth_np = rendered_depth.reshape(self.H, self.W).detach().cpu().numpy()

        plane_surf_np = plane_surf.detach().cpu().numpy()

        # get gt
        gt_rgb_np = (view_info.rgb.reshape(self.H, self.W, 3) * 255).cpu().numpy().astype(np.uint8)
        gt_normal_np = (F.normalize(view_info.mono_normal_global, dim=-1).reshape(self.H, self.W, 3)).cpu().numpy()
        gt_normal_color = ((gt_normal_np + 1.0) / 2.0 * 255).astype(np.uint8)
        gt_depth_np = view_info.mono_depth.reshape(self.H, self.W).cpu().numpy()

        n_r = 1
        n_c = 5
        plt.figure(figsize=(8, 2.5))  # width, height
        plt.subplot(n_r, n_c, 1)
        plt.title("rgb gt")
        plt.imshow(gt_rgb_np)
        plt.axis('off')

        plt.subplot(n_r, n_c, 2)
        plt.title("mono normal")
        plt.imshow(gt_normal_color)
        plt.axis('off')

        plt.subplot(n_r, n_c, 3)
        plt.title("rendered normal")
        plt.imshow(rendered_normal_color)
        plt.axis('off')

        plt.subplot(n_r, n_c, 4)
        plt.title("mono depth")
        plt.imshow(gt_depth_np)
        plt.axis('off')

        plt.subplot(n_r, n_c, 5)
        plt.title("rendered depth")
        plt.imshow(rendered_depth_np)
        plt.axis('off')

        root_dir = self.plane_plots_dir
        os.makedirs(root_dir, exist_ok=True)
        save_dir = os.path.join(root_dir, '%svis_%d_%d_cuda.jpg'%(prefix, self.iter_step, plot_img_idx))
        plt.savefig(save_dir)
        logger.info(f"saving to {save_dir}")

        save_dir = os.path.join(root_dir, 'debug_vis_%d_cuda.jpg'%(plot_img_idx))
        plt.savefig(save_dir)
        logger.info(f"saving to {save_dir}")

        if pcd_on:
            points_plane_o3d = o3d.geometry.PointCloud()
            points_plane_o3d.points = o3d.utility.Vector3dVector(plane_surf_np)
            save_dir = os.path.join(root_dir, 'debug_vis_%d_plane_surf.ply'%(plot_img_idx))
            logger.info(f'saving to {save_dir}')
            o3d.io.write_point_cloud(save_dir, points_plane_o3d)

    