import copy
import torch
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from utils import model_util


def compute_robust_range(arr, range_robust=None, k_stretch=2.0):
    if range_robust is None:
        range_robust = [0.05, 0.95]
    N = arr.shape[0]
    start_idx = int(round((N - 1) * range_robust[0]))
    end_idx = int(round((N - 1) * range_robust[1]))
    arr_sorted = np.sort(arr)
    start = arr_sorted[start_idx]
    end = arr_sorted[end_idx]
    start_stretched = (start + end) / 2.0 - k_stretch * (end - start) / 2.0
    end_stretched = (start + end) / 2.0 + k_stretch * (end - start) / 2.0
    return start_stretched, end_stretched

def compute_robust_range_lines(lines, range_robust=None, k_stretch=2.0):
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    if range_robust is None:
        range_robust = [0.05, 0.95]
    x_array = lines.reshape(-1, 3)[:, 0]
    y_array = lines.reshape(-1, 3)[:, 1]
    z_array = lines.reshape(-1, 3)[:, 2]

    x_start, x_end = compute_robust_range(
        x_array, range_robust=range_robust, k_stretch=k_stretch
    )
    y_start, y_end = compute_robust_range(
        y_array, range_robust=range_robust, k_stretch=k_stretch
    )
    z_start, z_end = compute_robust_range(
        z_array, range_robust=range_robust, k_stretch=k_stretch
    )
    ranges = np.array([[x_start, y_start, z_start], [x_end, y_end, z_end]])
    return ranges

def open3d_get_cameras(
    view_info_list,
    color=None,
    ranges=None,
    scale_cam_geometry=1.0,
    scale=1.0,
):
    if color is None:
        color = [1.0, 0.0, 0.0]
    cameras = o3d.geometry.LineSet()

    for cam_id in range(len(view_info_list)):
        view_info = view_info_list[cam_id]
        h, w = view_info.img_size
        K = view_info.intrinsic[:3,:3].cpu().numpy()
        pose = view_info.pose.cpu().numpy()
        # K = view_info.K

        camera_lines = o3d.geometry.LineSet.create_camera_visualization(
            w, h, K, np.eye(4),
            scale=0.005 * scale_cam_geometry * scale,
        )
        # T = np.eye(4)
        # T[:3, :3] = view_info.R
        # T[:3, 3] = view_info.t * scale
        # T = np.linalg.inv(T)
        T = pose
        cam = copy.deepcopy(camera_lines).transform(T)
        cam.paint_uniform_color(color)
        cameras += cam
    return cameras

def linesToOpen3d(lines, color=None):
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    
    num_lines = lines.shape[0]
    points = lines.reshape(-1,3)
    edges = np.arange(num_lines*2).reshape(-1,2)

    lineset = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(edges),
    )

    if color is not None:
        colors = np.repeat(np.array(color)[None,:], num_lines, axis=0)
        lineset.colors = o3d.utility.Vector3dVector(colors)

    return lineset

def juncsToOpen3d(juncs, color=None):
    if isinstance(juncs, torch.Tensor):
        juncs = juncs.cpu().numpy()
    
    num_juncs = juncs.shape[0]
    points = juncs.reshape(-1,3)

    pointset = o3d.geometry.PointCloud()
    pointset.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        colors = np.repeat(np.array(color)[None,:], num_juncs, axis=0)
        pointset.colors = o3d.utility.Vector3dVector(colors)

    return pointset

def linesetToPly(lineset, path):
    points = np.asarray(lineset.points)
    lines = np.asarray(lineset.lines)
    
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(lines)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        
        for l in lines:
            f.write(f"{l[0]} {l[1]}\n")
    
    print(f"Saved to PLY file: {path}")

def vis_open3d(open3d_set, name=''):
    # o3d.visualization.draw_geometries(open3d_set)
    vis = o3d.visualization.Visualizer()
    if name != '':
        vis.create_window(window_name=name)
    else:
        vis.create_window(window_name='temp')
    # vis.add_geometry(open3d_set)
    for s in open3d_set:
        vis.add_geometry(s)

    vis.run()
    vis.destroy_window()

def vis_lines(lines, color=None, vis=True):
    if isinstance(lines, torch.Tensor):
        if lines.requires_grad:
            lines = lines.detach()
    lineset = linesToOpen3d(lines, color)
    if not vis:
        return lineset
    else:
        vis_open3d([lineset])

def vis_plane(plane_corners, color=None, vis=True):
    planes = []
    for i in range(4):
        planes.append(torch.concat([plane_corners[:,i],plane_corners[:,(i+1)%4]], dim=-1))
    planes = torch.concat(planes, dim=0)
    if planes.requires_grad:
        planes = planes.detach()
    lineset = linesToOpen3d(planes, color)
    if not vis:
        return lineset
    else:
        vis_open3d([lineset])

def vis_juncs(juncs, mask=None, color=None, vis=True):
    if mask is not None:
        juncs = juncs[mask]
    if isinstance(juncs,torch.Tensor) and juncs.requires_grad:
        juncs = juncs.detach()
    juncset = juncsToOpen3d(juncs, color)
    if not vis:
        return juncset
    else:
        vis_open3d([juncset])

def vis_rays(rays, t=10, color=None, vis=True):
    rays_o, rays_d = rays
    edp1 = rays_o
    edp2 = rays_o + t * rays_d
    lines = torch.concat([edp1, edp2], dim=-1)
    lineset = linesToOpen3d(lines, color)
    if not vis:
        return lineset
    else:
        vis_open3d(lineset)

def vis_plane_rays_juncs_lines(plane_corners=None, rays=None, juncs=None, lines=None, lines2=None, t=10, use_color=True, view_info_list=None, return_o3d=False):
    '''
    Visualize lines, rays, junctions, and planes in Open3D.
    Args:
        plane_corners: the vertex of the planes, used to draw the planes. [N_p, 4, 3]
        rays: [rays_o, rays_d], used to draw the rays. rays_d: [N_r, 3], rays_o: [N_r, 3]
        juncs: used to draw the selected junctions. [N_j, 3]
        lines, lines2: used to together or separately draw two sets of lines. [N_l, 4]
        t: length of rays, which is used to control how long the rays drawn in open3d.
        use_color: if True, use color for visualization
        view_info_list: list of input view information, used to draw the cameras.
    '''    
    if use_color:
        color_p = [0,0,1]
        color_r = [1,0,0]
        color_i = [0,1,0]
        color_l = [0,0,0]
    else:
        color_p = None
        color_r = None
        color_i = None
        color_l = None

    name=''
    open3d_set = []
    if plane_corners is not None:
        plane_lineset = vis_plane(plane_corners, color=color_p, vis=False)
        open3d_set.append(plane_lineset)
        name += f'Plane_Num: {plane_corners.shape[0]} '
    if rays is not None:
        rays_lineset = vis_rays(rays, t=t, color=color_r, vis=False)
        open3d_set.append(rays_lineset)
        name += f'Rays_Num: {rays[0].shape[0]} '
    if juncs is not None:
        juncset = vis_juncs(juncs, color=color_i, vis=False)
        open3d_set.append(juncset)
        name += f'Junc_Num: {juncs.shape[0]} '
    if lines is not None:
        if juncs is None:
            juncs_num = torch.unique(lines.reshape(-1,3),dim=0).shape[0]
            name += f'Junc_Num: {juncs_num} '
        lineset = vis_lines(lines, color=color_l, vis=False)
        open3d_set.append(lineset)
        name += f'Line_Num: {lines.shape[0]} '

    ####################
    # for visual comparison
    if lines2 is not None:
        lineset2 = vis_lines(lines2, color=color_r, vis=False)
        open3d_set.append(lineset2)
    ####################

    if view_info_list is not None:
        lranges = compute_robust_range_lines(lines)
        scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()
        cam_scale = 1.0
        camera_set = open3d_get_cameras(
            view_info_list,
            scale_cam_geometry=scale_cam_geometry * cam_scale,
        )
        open3d_set.append(camera_set)

    if return_o3d:
        return open3d_set
    else:
        vis_open3d(open3d_set, name=name)


def save_plot_image_lines(image, final_lines, lines=None, lines_gt=None, saveto=None, show=False):

    def plot_ax(ax, lines, lines_gt=None):
        ax.imshow(image)
        if lines_gt is not None:
            lines_gt = np.unique(lines_gt, axis=0)
            ax.plot([lines_gt[:,0],lines_gt[:,2]], [lines_gt[:,1],lines_gt[:,3]], 'r-', linewidth=2, alpha=0.5)
        ax.plot([lines[:,0],lines[:,2]], [lines[:,1],lines[:,3]], color='yellow', linestyle='-', linewidth=2, alpha=0.5)
        ax.plot(lines[:,0], lines[:,1], 'bo', markersize=3)
        ax.plot(lines[:,2], lines[:,3], 'bo', markersize=3)

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    if isinstance(final_lines, torch.Tensor):
        final_lines = final_lines.cpu().numpy()
    if isinstance(lines_gt, torch.Tensor):
        lines_gt = lines_gt.cpu().numpy()
    
    if len(lines.shape)==3:
        lines = lines.reshape(-1,4)    
    if len(final_lines.shape)==3:
        final_lines = final_lines.reshape(-1,4)    
    if len(lines_gt.shape)==3:
        lines_gt = lines_gt.reshape(-1,4)

    final_lines = np.unique(final_lines, axis=0)
    if lines is not None:
        lines = np.unique(lines, axis=0)
        fig, axs = plt.subplots(1, 2, figsize=(28, 12))
        plot_ax(axs[0], lines, lines_gt)
        plot_ax(axs[1], final_lines, lines_gt)
    else:
        fig, ax = plt.subplots(figsize=(28, 12))
        plot_ax(ax, final_lines, lines_gt)

    if saveto is not None:
        plt.savefig(saveto)
    if show:
        plt.show()
        import pdb;pdb.set_trace()
    plt.close()

def plot_image_lines_juncs(image, lines=None, lines_gt=None, juncs=None):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()
    if isinstance(lines_gt, torch.Tensor):
        lines_gt = lines_gt.cpu().numpy()
    if isinstance(juncs, torch.Tensor):
        juncs = juncs.cpu().numpy()

    if len(lines.shape)>2:
        lines = lines.reshape(-1, 4)
    if len(lines_gt.shape)>2:
        lines_gt = lines_gt.reshape(-1, 4)
    
    plt.imshow(image)
    if lines is not None:
        plt.plot([lines[:,0],lines[:,2]],[lines[:,1],lines[:,3]], color='yellow', linestyle='-', alpha=0.5)
    if lines_gt is not None:
        plt.plot([lines_gt[:,0],lines_gt[:,2]],[lines_gt[:,1],lines_gt[:,3]], color='red', linestyle='-', alpha=0.5)
    if juncs is not None:
        plt.plot(juncs[:,0], juncs[1], 'bo')
    plt.show()
    plt.close()


def plot_image_plane_lines_2d(image, plane_corners_2d=None, lines_gt=None):
    plane_lines_2d = torch.concat([torch.concat([plane_corners_2d[:,(i+1)%4],plane_corners_2d[:,i]], dim=1) for i in range(4)], dim=0)
    plot_image_lines_juncs(image, lines_gt=lines_gt, lines=plane_lines_2d)
    

def plot_lines_gt_2d_to_3d(view_info, lines_tgt, plane_corners):
    K = view_info.intrinsic[:3,:3].cpu()
    pose = view_info.pose.cpu()
    depth = torch.from_numpy(np.load(view_info.md_path)).squeeze().float()*view_info.scene_scale
    lines_gt_2d = lines_tgt.long()
    lines_gt_2d_ps = lines_gt_2d[:,[1,0,3,2]].reshape(-1,2)
    depths = depth[lines_gt_2d_ps[:,0],lines_gt_2d_ps[:,1]].reshape(-1,2)
    lines_gt_3d = model_util.project_2d_to_3d(lines_tgt.cpu(), K, pose, depths)
    vis_plane_rays_juncs_lines(plane_corners=plane_corners, lines=lines_gt_3d, view_inof_list=[view_info])


def plot_lines_gt_2d_to_3d_list(view_info_list, plane_corners=None):
    all_lines_gt_3d = []
    for view_info in view_info_list:
        K = view_info.intrinsic[:3,:3].cpu()
        pose = view_info.pose.cpu()
        depth = torch.from_numpy(np.load(view_info.md_path)).squeeze().float()*view_info.scene_scale
        lines_tgt = view_info.lines_uniq[:,:4].cpu()
        lines_gt_2d = lines_tgt.long()
        lines_gt_2d_ps = lines_gt_2d[:,[1,0,3,2]].reshape(-1,2)
        depths = depth[lines_gt_2d_ps[:,0],lines_gt_2d_ps[:,1]].reshape(-1,2)
        lines_gt_3d = model_util.project_2d_to_3d(lines_tgt.cpu(), K, pose, depths)
        all_lines_gt_3d.append(lines_gt_3d)
    all_lines_gt_3d = torch.cat(all_lines_gt_3d, dim=0)
    vis_plane_rays_juncs_lines(plane_corners=plane_corners, lines=all_lines_gt_3d, view_inof_list=view_info_list)


def lines3d2ply(lines3d, path):
    line_set = vis_lines(lines3d, vis=False)
    o3d.io.write_line_set(path, line_set)
    print(f".ply file saved to {path}")

def draw_3dline_with_pose_select(save_dir, lines3d, iter):
    default_camera_path = os.path.join(save_dir, 'camera_pose.json')
    lineset = linesToOpen3d(lines3d)

    def save_view(vis):
        # set and save a camera pose for observation
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(default_camera_path, param)
        print("Pose Parameters Saved!")
        return False

    # select view
    if not os.path.exists(default_camera_path):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=2000, height=2000)
        vis.add_geometry(lineset)
        vis.register_key_callback(ord('S'), save_view)
        vis.run()
        vis.destroy_window()


    camera_params = o3d.io.read_pinhole_camera_parameters(default_camera_path) 
    # get the original window size
    original_width = camera_params.intrinsic.width
    original_height = camera_params.intrinsic.height

    # apply the view and capture the image
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=original_width, height=original_height, visible=False)
    vis.add_geometry(lineset)
    
    view_ctl = vis.get_view_control()
    view_ctl.convert_from_pinhole_camera_parameters(camera_params)
    
    # vis.update_geometry(lineset)
    vis.poll_events()
    vis.update_renderer()

    # vis.capture_screen_image(os.path.join(save_dir, f'{name}_{detector}.png'))
    image = vis.capture_screen_float_buffer(do_render=True)
    image = (np.asarray(image)*255).astype(np.uint8)
    img_path = os.path.join(save_dir, f'{iter}.png')
    plt.imsave(img_path, np.asarray(image))

    # remove geometry
    vis.remove_geometry(lineset)
    # reset view
    vis.reset_view_point(True)
    vis.destroy_window()

    print(f"Image of 3D lines saved to {img_path}")

def create_camera_set(pose, w, h , K, scale, color=None):
    if color is None:
        color = [1.0, 0.0, 0.0]

    camera_lines = o3d.geometry.LineSet.create_camera_visualization(
        w, h, K, np.eye(4),
        scale=scale,
    )
    cam = copy.deepcopy(camera_lines).transform(pose)
    cam.paint_uniform_color(color)
    return cam

def draw_camera_pose(lines, poses1, poses2, w, h , K):
    open3d_set = []

    lineset = vis_lines(lines, vis=False)
    open3d_set.append(lineset)

    lranges = compute_robust_range_lines(lines)
    scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()
    cam_scale = 1.0
    scale_cam_geometry=0.005 * scale_cam_geometry * cam_scale

    color1 = [1,0,0]
    color2 = [0,0,1]
    for i, (pose1, pose2) in enumerate(zip(poses1, poses2)):
        cam1 = create_camera_set(pose1, w, h, K, scale_cam_geometry, color=color1)
        open3d_set.append(cam1)
        cam2 = create_camera_set(pose2, w, h, K, scale_cam_geometry, color=color2)
        open3d_set.append(cam2)

    vis_open3d(open3d_set)