import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import quaternion
import math
import cv2
from tqdm import tqdm
from utils import mesh_util
import time

RAY_IDX = torch.arange(480*640).reshape(-1, 1).repeat(1, 15).cuda()
TMP_IDX = torch.arange(480*640).reshape(-1, 1).repeat(1, 15).cuda()
PLANE_IDX = torch.arange(50000).cuda()

def get_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def project2D(pose, intrinsics, points3d):
    if len(pose.size()) == 3:
        pose = pose[0]
    if len(intrinsics.size()) == 3:
        intrinsics = intrinsics[0]
    assert points3d.shape[-1] == 3

    ori_shape = points3d.shape

    K = intrinsics[:3,:3]
    proj_mat = pose.inverse()[:3]
    R = proj_mat[:,:3]
    T = proj_mat[:,3:]

    X = points3d.reshape(-1,3)
    x = K@(R@X.t()+T)
    x = x.t()
    denominator = x[:,-1:]
    sign = torch.where(denominator>=0, torch.ones_like(denominator), -torch.ones_like(denominator))
    eps = torch.where(denominator.abs()<1e-8, torch.ones_like(denominator)*1e-8, torch.zeros_like(denominator))
    x = x/(denominator+eps*sign)
    x = x.reshape(*ori_shape)[...,:2]

    return x

def project_2d_to_3d(L, K, pose, depth):
    """
    Project 2D line segments to 3D space using camera intrinsic matrix K, pose, and depth information.

    Parameters:
    - L: Tensor of shape [N, 4], where each row is [u1, v1, u2, v2] representing 2D line segments.
    - K: Tensor of shape [3, 3], the camera intrinsic matrix.
    - pose: Tensor of shape [4, 4], the camera pose matrix [R | t].
    - depth: Tensor of shape [N, 2], where each row is [d1, d2] representing the depth of the two endpoints.

    Returns:
    - L_3d: Tensor of shape [N, 6], where each row is [x1, y1, z1, x2, y2, z2] representing 3D line segments.
    """
    # Extract rotation matrix R and translation vector t from pose
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Inverse of the intrinsic matrix
    K_inv = torch.inverse(K)

    # Convert 2D points to homogeneous coordinates
    L_homogeneous = torch.cat([L[:, :2], torch.ones(L.size(0), 1)], dim=1)  # [N, 3]
    L_homogeneous_2 = torch.cat([L[:, 2:], torch.ones(L.size(0), 1)], dim=1)  # [N, 3]

    # Convert to normalized image plane coordinates
    L_normalized = torch.bmm(K_inv.unsqueeze(0).expand(L.size(0), 3, 3), L_homogeneous.unsqueeze(2)).squeeze(2)
    L_normalized_2 = torch.bmm(K_inv.unsqueeze(0).expand(L.size(0), 3, 3), L_homogeneous_2.unsqueeze(2)).squeeze(2)

    # Convert to camera coordinates using depth information
    L_camera = depth[:, 0].unsqueeze(1) * L_normalized
    L_camera_2 = depth[:, 1].unsqueeze(1) * L_normalized_2

    # Convert to world coordinates
    L_world = torch.bmm(R.unsqueeze(0).expand(L.size(0), 3, 3), L_camera.unsqueeze(2)).squeeze(2) + t.unsqueeze(0)
    L_world_2 = torch.bmm(R.unsqueeze(0).expand(L.size(0), 3, 3), L_camera_2.unsqueeze(2)).squeeze(2) + t.unsqueeze(0)

    # Combine into 3D line segments
    L_3d = torch.cat([L_world, L_world_2], dim=1)

    return L_3d

def quat_to_rot(q):
    assert isinstance(q, torch.Tensor)
    assert q.shape[-1] == 4
    if q.dim() == 1:
        q = q.unsqueeze(0)  # 1, 4
    elif q.dim() == 2:
        pass  # bs, 4
    else:
        raise NotImplementedError

    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    # R = torch.zeros((batch_size, 3, 3), device='cuda')
    qr = q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q

def quaternion_mult(q1, q2):
    '''
    q1 x q2

    q1 = w1+i*x1+j*y1+k*z1
    q2 = w2+i*x2+j*y2+k*z2
    q1*q2 =
     (w1w2 - x1x2 - y1y2 - z1z2)
    +(w1x2 + x1w2 + y1z2 - z1y2)i
    +(w1y2 - x1z2 + y1w2 + z1x2)j
    +(w1z2 + x1y2 - y1x2 + z1w2)k

    :param q1:
    :param q2:
    :return:
    '''

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    w1 = q1[:, 0]  # bs
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]

    w2 = q2[:, 0]  # bs
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    q = torch.stack([w, x, y, z], dim=-1)

    return q

def get_raydir_camloc(uv, pose, intrinsics):
    if pose.shape[1] == 7:
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else:
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc

def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)

def get_sphere_intersections(cam_loc, ray_directions, r = 1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r ** 2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print('BOUNDING SPHERE PROBLEM!')
        exit()

    sphere_intersections = torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).cuda().float() - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points

def get_rotation_quaternion_of_normal(plane_normal_init, standard_normal=None):
    if standard_normal is None:
        standard_normal = torch.tensor([0., 0., 1.]).reshape(1, 3).expand(plane_normal_init.shape[0], 3).to(plane_normal_init.device)
    angle_diff = torch.acos((standard_normal * plane_normal_init).sum(dim=-1).clamp(-1, 1)).reshape(-1, 1)
    rot_axis = torch.cross(standard_normal, plane_normal_init, dim=-1)  # n_plane, 3
    rot_axis = F.normalize(rot_axis, dim=-1)
    rot_vec = (rot_axis * angle_diff).cpu().numpy()
    rot_q = quaternion.as_float_array(quaternion.from_rotation_vector(rot_vec))
    rot_q = torch.from_numpy(rot_q).float()
    return rot_q

def get_rotation_quaternion_of_xyAxis(plane_num, angle=None):
    rot_axis = torch.tensor([0., 0., 1.]).reshape(1, 3)
    if angle is None:
        rand_angle = (torch.rand(plane_num, 1) - 0.5) * 5. / 180. * np.pi
    else:
        rand_angle = angle / 180. * np.pi
    rot_vec = (rot_axis * rand_angle.cpu()).numpy()
    rot_q = quaternion.as_float_array(quaternion.from_rotation_vector(rot_vec))
    rot_q = torch.from_numpy(rot_q).float()
    return rot_q

def get_overlapped_mask(plane_center, plane_normal, plane_offset, plane_radii, plane_xAxis, plane_yAxis, normal_thres=15, dist_thres=0.02):
    plane_num = plane_center.shape[0]
    corner1 = plane_center + plane_radii[:, 0:1] * plane_xAxis + plane_radii[:, 1:2] * plane_yAxis  # n, 3
    corner2 = plane_center + plane_radii[:, 0:1] * plane_xAxis - plane_radii[:, 1:2] * plane_yAxis
    corner3 = plane_center - plane_radii[:, 0:1] * plane_xAxis + plane_radii[:, 1:2] * plane_yAxis
    corner4 = plane_center - plane_radii[:, 0:1] * plane_xAxis - plane_radii[:, 1:2] * plane_yAxis

    pts = torch.stack([corner1, corner2, corner3, corner4], dim=1)  # n, 4, 3
    nc = pts.shape[1]
    pts = pts.reshape(-1, 3)  # n * 4, 3

    dists = compute_pts2planes_dist(pts, plane_normal, plane_offset)  # n * 4, n, 1
    pts_proj = pts.reshape(-1, 1, 3) - dists * plane_normal.reshape(1, -1, 3)  # n * 4, n, 3 

    vec = pts_proj - plane_center.reshape(1, -1, 3)  # n * 4, n, 3
    if_in_x = ((vec * plane_xAxis.reshape(1, -1, 3)).sum(dim=-1).abs() -  plane_radii[:, 0:1].reshape(1, -1)) <= 0.0005   # n * 4, n
    if_in_y = ((vec * plane_yAxis.reshape(1, -1, 3)).sum(dim=-1).abs() -  plane_radii[:, 1:2].reshape(1, -1)) <= 0.0005   # n * 4, n

    if_in = if_in_x & if_in_y   # n * 4, n
    if_in = if_in.reshape(-1, nc, plane_num)   # n, 4, n
    if_in = if_in.sum(dim=1) == nc  # n~, n

    normal_diff = torch.acos((plane_normal.reshape(-1, 1, 3) * plane_normal.reshape(1, -1, 3)).sum(dim=-1).clamp(-1., 1.)) * 180. / np.pi
    normal_mask = normal_diff < normal_thres

    dist_mask = (dists.reshape(-1, nc, plane_num).abs() < dist_thres).sum(dim=1) == nc

    if_in_final = if_in & normal_mask & dist_mask
    prune_mask = if_in_final.sum(dim=-1) > 1

    return prune_mask

def compute_offset(pts, normals):
    v1 = pts
    v2 = F.normalize(normals, dim=-1)
    offset = (v1 * v2).sum(-1)
    return offset

def compute_ray2plane_intersections_v2(rays_o, rays_d, plane_center, plane_normal):
    vec_o2c = plane_center.unsqueeze(0) - rays_o.unsqueeze(1)  # n_ray, n_plane, 3
    signed_dist_o2p = -(plane_normal.unsqueeze(0) * vec_o2c).sum(dim=-1)  # n_ray, n_plane
    o2inter_SD = - signed_dist_o2p / ((rays_d.unsqueeze(1) * plane_normal.unsqueeze(0)).sum(dim=-1) + 1e-10)
    inter = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * o2inter_SD.unsqueeze(-1)
    return inter, o2inter_SD

def compute_pairwise_ray2plane_intersection_v2(rays_o, rays_d, plane_center, plane_normal, grads=None):
    rays_o = rays_o.reshape(-1, 3)
    rays_d = F.normalize(rays_d.reshape(-1, 3), dim=-1)
    plane_normal = F.normalize(plane_normal.reshape(-1, 3), dim=-1)
    plane_center = plane_center.reshape(-1, 3)
    
    vec_o2c = plane_center - rays_o  # n, 3
    signed_dist_o2p = -(plane_normal * vec_o2c).sum(dim=-1)  # n
    o2inter_SD = - signed_dist_o2p / ((rays_d * plane_normal).sum(dim=-1) + 1e-10) # n
    inter = rays_o + rays_d * o2inter_SD.unsqueeze(-1)
    return inter, o2inter_SD

def check_intersections_in_plane(inter, plane_center, plane_radii, plane_xAxis, plane_yAxis):
    vec_c2i = plane_center - inter
    signed_dist_c2i_xAxis = torch.sum(vec_c2i * plane_xAxis, dim=-1)
    signed_dist_c2i_yAxis = torch.sum(vec_c2i * plane_yAxis, dim=-1)

    plane_radii_exp = plane_radii.unsqueeze(0).expand(inter.shape[0], -1, -1)
    mask_x = (signed_dist_c2i_xAxis >= -(plane_radii_exp[...,2])) * (signed_dist_c2i_xAxis <= plane_radii_exp[...,0])
    mask_y = (signed_dist_c2i_yAxis >= -(plane_radii_exp[...,3])) * (signed_dist_c2i_yAxis <= plane_radii_exp[...,1])
    mask = mask_x * mask_y

    return mask

def find_positive_min_mask_for(A):
    assert isinstance(A, torch.Tensor)
    mask = torch.zeros_like(A, dtype=torch.bool)
    for i in range(A.shape[0]):
        row = A[i]
        positive_values = row[row > 0]
        
        if positive_values.numel() > 0:
            min_positive_value = positive_values.min()
            mask[i] = (row == min_positive_value) & (row > 0)
    return mask

def find_positive_min_mask(A):
    assert isinstance(A, torch.Tensor)
    A[A==0]=float('inf')
    min_values, min_indix = torch.min(A, dim=1)
    mask = min_values < float('inf')
    
    return mask, min_indix

def find_positive_min_mask_org(A):
    assert isinstance(A, torch.Tensor)
    positive_values = A[A > 0]
    if positive_values.numel() == 0:
        return torch.zeros_like(A, dtype=torch.bool)
    
    min_positive_values, _ = torch.min(torch.where(A > 0, A, torch.inf), dim=1, keepdim=True)
    mask = (A == min_positive_values) & (A > 0)
    
    return mask
    
    
def cluster_dbscan(points, eps=0.01, min_samples=2):
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clustered_points = []
    for i in range(labels.max()+1):
        clustered_points.append(points[labels==i].mean(axis=0))
    clustered_points = torch.tensor(np.array(clustered_points)).float().cuda()
    
    return clustered_points

def get_plane_lines_from_corners(plane_corners):
    plane_lines = torch.concat([torch.concat([plane_corners[:,i],plane_corners[:,(i+1)%4]], dim=-1).unsqueeze(1) for i in range(4)], dim=1)
    return plane_lines

def compute_pairewise_pts2planes_dist(pts, planes_normal, planes_offset):
    """
    pts: [..., 3]
    planes_normal: [..., 3]
    planes_offset: [..., 1]
    """
    planes_normal = planes_normal.reshape(-1, 3)
    planes_offset = planes_offset.reshape(-1, 1)
    pts = pts.reshape(-1, 3)
    pts_origin2planes = planes_normal * planes_offset  # N, 3
    dist_field = ((pts - pts_origin2planes) * planes_normal).sum(-1, keepdim=True)  # N, 1

    return dist_field

def compute_pairewise_pts2planes_dist_v2(pts, planes_normal, planes_pt):
    """
    pts: [..., 3]
    planes_normal: [..., 3]
    planes_pt: [..., 3]
    """
    planes_normal = planes_normal.reshape(-1, 3)
    planes_pt = planes_pt.reshape(-1, 3)
    pts = pts.reshape(-1, 3)

    vec_p2pp = planes_pt - pts  # n, 3
    signed_dist_p2p = -(planes_normal * vec_p2pp).sum(dim=-1)  # n
    return signed_dist_p2p

def compute_pts2planes_dist(pts, planes_normal, planes_offset):
    """
    pts: [..., 3]
    planes_normal: [N, 3]
    planes_offset: [N, 1]
    """
    planes_normal = planes_normal.unsqueeze(0)  # 1, N, 3
    planes_offset = planes_offset.unsqueeze(0)
    pts = pts.reshape(-1, 1, 3)
    pts_origin2planes = planes_normal * planes_offset  # 1, N, 3
    dist_field = ((pts - pts_origin2planes) * planes_normal).sum(-1, keepdim=True)  # npt, N, 1

    return dist_field

def rbf_rectangle(pts, centers, plane_radii, rot_q, plane_xAxis, plane_yAxis, plane_radii_weight=None, scale=30., RBF_dist_type='abs', RBF_weight_type='soft', grads=None):
    assert plane_radii.shape[-1] == 4
    pts = pts.reshape(-1, 3).detach()
    centers = centers.reshape(-1, 3)
    plane_radii = plane_radii.reshape(-1, 4)
    rot_q = rot_q.reshape(-1, 4)
    plane_xAxis = plane_xAxis.reshape(-1, 3)
    plane_yAxis = plane_yAxis.reshape(-1, 3)
    plane_radii_weight = plane_radii_weight.reshape(-1, 4)

    assert pts.shape[0] == centers.shape[0]
    assert pts.shape[0] == plane_radii.shape[0]
    assert pts.shape[0] == rot_q.shape[0]
    assert pts.shape[0] == plane_xAxis.shape[0]
    assert pts.shape[0] == plane_yAxis.shape[0]

    plane_xAxis = F.normalize(plane_xAxis, dim=-1)  # related to rotation
    plane_yAxis = F.normalize(plane_yAxis, dim=-1)  # related to rotation

    radii_xp = plane_radii[:, 0:1]
    radii_xn = plane_radii[:, 2:3]
    radii_yp = plane_radii[:, 1:2]
    radii_yn = plane_radii[:, 3:4]

    diff = pts - centers  # n_pt, 3
    proj_x = torch.sum(diff * plane_xAxis, dim=-1, keepdim=True)
    proj_y = torch.sum(diff * plane_yAxis, dim=-1, keepdim=True)

    if RBF_dist_type == 'abs':
        w_x = torch.where(proj_x > 0, 
                        torch.sigmoid((radii_xp - proj_x.abs())*plane_radii_weight[:, 0:1]) * 2.,
                        torch.sigmoid((radii_xn - proj_x.abs())*plane_radii_weight[:, 2:3]) * 2.).clamp(max=1.0)
        w_y = torch.where(proj_y > 0,
                        torch.sigmoid((radii_yp - proj_y.abs())*plane_radii_weight[:, 1:2]) * 2.,
                        torch.sigmoid((radii_yn - proj_y.abs())*plane_radii_weight[:, 3:4]) * 2.).clamp(max=1.0)
    elif RBF_dist_type == 'rel':
        raise
    else:
        raise

    if RBF_weight_type == 'hard':
        w_min = torch.where(w_x < w_y, w_x, w_y)  # n_pt, 1
    elif RBF_weight_type == 'soft': 
        w_min = torch.where(w_x < w_y, w_x+w_y*0.001, w_y+w_x*0.001)  # n_pt, 1
    else:
        raise
    
    return w_min.clamp(min=0., max=1.)

def rbf_hard(pts, centers, plane_radii, rot_q, plane_xAxis, plane_yAxis):
    assert plane_radii.shape[-1] == 4
    pts = pts.reshape(-1, 3).detach()
    centers = centers.reshape(-1, 3)
    plane_radii = plane_radii.reshape(-1, 4)
    rot_q = rot_q.reshape(-1, 4)
    plane_xAxis = plane_xAxis.reshape(-1, 3)
    plane_yAxis = plane_yAxis.reshape(-1, 3)

    assert pts.shape[0] == centers.shape[0]
    assert pts.shape[0] == plane_radii.shape[0]
    assert pts.shape[0] == rot_q.shape[0]
    assert pts.shape[0] == plane_xAxis.shape[0]
    assert pts.shape[0] == plane_yAxis.shape[0]

    plane_xAxis = F.normalize(plane_xAxis, dim=-1)  # related to rotation
    plane_yAxis = F.normalize(plane_yAxis, dim=-1)  # related to rotation

    radii_xp = plane_radii[:, 0:1]
    radii_xn = plane_radii[:, 2:3]
    radii_yp = plane_radii[:, 1:2]
    radii_yn = plane_radii[:, 3:4]

    diff = pts - centers  # n_pt, 3
    proj_x = torch.sum(diff * plane_xAxis, dim=-1, keepdim=True)
    proj_y = torch.sum(diff * plane_yAxis, dim=-1, keepdim=True)
        
    w_x = ((proj_x < radii_xp) & (proj_x > -radii_xn)).float().detach()
    w_y = ((proj_y < radii_yp) & (proj_y > -radii_yn)).float().detach()
        
    w_min = torch.where(w_x < w_y, w_x, w_y)  # n_pt, 1
       
    return w_min.clamp(min=0., max=1.)

def decode_covariance_rpy(rot_q, plane_radii, invert=True):
    DIV_EPSILON = 1e-8
    d = 1.0 / (plane_radii + DIV_EPSILON) if invert else plane_radii
    diag = torch.diag_embed(d*d)
    rot_q = F.normalize(rot_q, dim=-1)
    rot_matrix = quat_to_rot(rot_q)
    return torch.matmul(torch.matmul(rot_matrix, diag), rot_matrix.transpose(1, 2))

def rbf_gaussian(pts, centers, plane_radii, rot_q, scale):
    pts = pts.reshape(-1, 3)
    centers = centers.reshape(-1, 3)
    plane_radii = plane_radii.reshape(-1, 4)
    plane_radii = plane_radii[:, :2]
    plane_radii = torch.cat([plane_radii, torch.ones_like(plane_radii[:, 0:1])], dim=-1)
    rot_q = rot_q.reshape(-1, 4)
    assert pts.shape[0] == centers.shape[0]
    assert pts.shape[0] == plane_radii.shape[0]
    assert pts.shape[0] == rot_q.shape[0]

    diff = pts - centers  # n_pt, 3

    x, y, z = diff.unbind(-1)

    inv_cov = decode_covariance_rpy(rot_q, plane_radii * scale, invert=True)
    inv_cov = inv_cov.view(*inv_cov.shape[:-2], 9)

    c00, c01, c02, _, c11, c12, _, _, c22 = inv_cov.unbind(-1)
    dist = (x * (c00 * x + c01 * y + c02 * z) +
            y * (c01 * x + c11 * y + c12 * z) +
            z * (c02 * x + c12 * y + c22 * z))
    dist = torch.exp(-0.5 * dist)[Ellipsis, None]
    return dist

def eval_pairwise_rbf(pts, centers, plane_radii, rot_q, plane_xAxis, plane_yAxis, plane_cfg, ite=-1, grads=None):
    assert plane_radii.shape[-1] == 4
    RBF_type = plane_cfg.RBF_type
    if RBF_type == 'rectangle':
        if ite == -1 or plane_cfg.RBF_weight_change_type == 'max':
            weight = 300.
            plane_radii_weight = torch.ones_like(plane_radii) * weight
        elif plane_cfg.RBF_weight_change_type == 'increase':
            max_weight = 300.
            ratio = ite / plane_cfg.coarse_stage_ite
            weight = min(math.exp(-(1 - ratio)) * 20, max_weight)
            plane_radii_weight = torch.ones_like(plane_radii) * weight
        elif plane_cfg.RBF_weight_change_type == 'min':
            max_weight = 300.
            ratio = 0.
            weight = min(math.exp(-(1 - ratio)) * 20, max_weight)
            plane_radii_weight = torch.ones_like(plane_radii) * weight
        else:
            raise NotImplementedError
        rbf = rbf_rectangle(pts, centers, plane_radii, rot_q, plane_xAxis, plane_yAxis, plane_radii_weight=plane_radii_weight * 5., 
                            scale=30., RBF_dist_type=plane_cfg.RBF_dist_type, RBF_weight_type=plane_cfg.RBF_weight_type, grads=grads)
        return rbf, weight
    elif RBF_type == 'gaussian':
        assert plane_cfg.get_string('radii_dir_type') == 'single'
        ratio = ite / plane_cfg.coarse_stage_ite
        scale = 1. + min(ratio, 1.)
        rbf = rbf_gaussian(pts, centers, plane_radii, rot_q, scale)
        return rbf, scale
    elif RBF_type == 'none':
        rbf = rbf_hard(pts, centers, plane_radii, rot_q, plane_xAxis, plane_yAxis)
        return rbf, 0.
    else:
        raise ValueError

def get_max_and_min_radii(ite, radii_max_list, radii_min_list, radii_milestone_list):
    if ite == -1:
        max_radii = radii_max_list[-1]
        min_radii = radii_min_list[-1]
    else:
        ms_i = -1
        for ms in radii_milestone_list:
                if ite >= ms:
                    ms_i += 1
                else:
                    break
        assert ms_i >= 0
        max_radii = radii_max_list[ms_i]
        min_radii = radii_min_list[ms_i]
    
    return max_radii, min_radii

def get_plane_param_from_sphere(plane_num, radius):
    points = []
    for _ in range(plane_num):
        z = np.random.uniform(-1, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        x *= radius
        y *= radius
        z *= radius  
        points.append([x, y, z])
    points = np.array(points)
    init_centers = torch.from_numpy(points).float()
    init_normals = F.normalize(-init_centers, dim=-1)

    init_rot_q_normal = get_rotation_quaternion_of_normal(init_normals)
    init_rot_angle_xyAxis = torch.tensor([0.]).reshape(1, 1).repeat(init_normals.shape[0], 1)
    init_rot_q_xyAxis = get_rotation_quaternion_of_xyAxis(init_normals.shape[0], angle=init_rot_angle_xyAxis)

    return init_centers, init_rot_q_normal, init_rot_q_xyAxis

def get_topK_plane_index_core(rays_o, rays_d, plane_normal, plane_offset, plane_center, plane_radii, plane_rot_q, plane_xAxis, plane_yAxis, topK, topK_out, plane_cfg, ite=-1, ray_id=None):
    N_rays = rays_o.shape[0]
    plane_num = plane_normal.shape[0]
    assert plane_radii.shape[-1] == 4

    plane_radii_padded = plane_radii.unsqueeze(0).expand(N_rays, plane_num, 4)
    plane_center_padded = plane_center.unsqueeze(0).expand(N_rays, plane_num, 3)
    plane_rot_q_padded = plane_rot_q.unsqueeze(0).expand(N_rays, plane_num, 4)
    plane_xAxis_padded = plane_xAxis.unsqueeze(0).expand(N_rays, plane_num, 3)
    plane_yAxis_padded = plane_yAxis.unsqueeze(0).expand(N_rays, plane_num, 3)

    # get ray-to-plane intersection
    # inter, o2inter_SD, _, _ = compute_ray2plane_intersections(rays_o, rays_d, plane_normal, plane_offset)
    inter, o2inter_SD = compute_ray2plane_intersections_v2(rays_o, rays_d, plane_center, plane_normal)
    
    # calculate valid inter mask
    # inter_valid_mask = (o2inter_SD > 0.1) & (inter2center_dist < 0.5)  # n_ray, n_plane
    inter_valid_mask = o2inter_SD > 0.1  # n_ray, n_plane
    
    # get RBF weight of inter
    inter2center_RBF, _ = eval_pairwise_rbf(
        inter, plane_center_padded, plane_radii_padded, plane_rot_q_padded, plane_xAxis_padded, plane_yAxis_padded,
        plane_cfg, ite=ite)
    inter2center_RBF = inter2center_RBF.reshape(N_rays, -1) # n_ray, n_plane
    
    # get masked RBF weight of inter
    inter2center_RBF_masked = inter2center_RBF * inter_valid_mask.float()

    # get top K inters' index
    sorted_inter_RBF, sorted_inter_index = inter2center_RBF_masked.sort(dim=-1, descending=True)
    topK_inter_RBF, topK_inter_index = sorted_inter_RBF[:, :topK], sorted_inter_index[:, :topK]  # n_ray, topK
    # ray_idx = torch.arange(N_rays).reshape(-1, 1).repeat(1, topK).cuda()
    ray_idx = RAY_IDX[:N_rays, :topK]
    topK_inter_SD = o2inter_SD[ray_idx, topK_inter_index]
    topK_inter_SD[topK_inter_SD < 0.2] = 1e10
    topK_inter_SD[topK_inter_RBF < 0.35] = 1e10
    _, sorted_topK_inter_SD_index = topK_inter_SD.sort(dim=-1)

    # tmp_idx = torch.arange(N_rays).reshape(-1, 1).repeat(1, topK_out)
    tmp_idx = TMP_IDX[:N_rays, :topK_out]
    final_index = topK_inter_index[tmp_idx, sorted_topK_inter_SD_index[:, :topK_out]]

    return final_index

def get_topK_interPlane_index(plane_model, rays_o, rays_d, c2w, intrinsic, ite=-1, topK=-1):
    H, W = plane_model.H, plane_model.W
    N_rays = rays_d.shape[0]
    assert topK > 0

    # get visible plane idx
    with torch.no_grad():
        # get plane parameters
        plane_normal, plane_offset, plane_center, plane_radii, plane_rot_q, plane_xAxis, plane_yAxis = plane_model.get_plane_geometry(ite=ite)
        
        plane_corner_1 = plane_center + plane_xAxis * plane_radii[:, 0:1] + plane_yAxis * plane_radii[:, 1:2]
        plane_corner_2 = plane_center + plane_xAxis * plane_radii[:, 0:1] - plane_yAxis * plane_radii[:, 3:4]
        plane_corner_3 = plane_center - plane_xAxis * plane_radii[:, 2:3] + plane_yAxis * plane_radii[:, 1:2]
        plane_corner_4 = plane_center - plane_xAxis * plane_radii[:, 2:3] - plane_yAxis * plane_radii[:, 3:4]
        pt_world = torch.stack([plane_corner_1, plane_corner_2, plane_corner_3, plane_corner_4, plane_center], dim=1).reshape(-1, 3)
        pt_local = torch.matmul(torch.inverse(c2w[:3, :3]), pt_world.permute(1, 0) - c2w[:3, 3:4])
        pt_depth = pt_local[2:3]  # 1, n_plane
        pt_uv1 = torch.matmul(intrinsic, pt_local / (pt_depth + 1e-10))
        pt_uv = pt_uv1[:2].permute(1, 0)  # n_plane, 2
        valid_proj_mask = (pt_uv[:, 0] >= 0) & (pt_uv[:, 0] <= W) & (pt_uv[:, 1] >= 0) & (pt_uv[:, 1] <= H) & (pt_depth[0] > 0)
        valid_proj_mask = valid_proj_mask.reshape(-1, 5)
        vis_plane_mask_pre = valid_proj_mask.sum(dim=-1) > 0

        vis_plane_mask_pre = (vis_plane_mask_pre + 1.) > 0

        # vis_plane_id_pre = torch.arange(plane_model.get_plane_num()).cuda()[vis_plane_mask_pre]
        vis_plane_id_pre = PLANE_IDX[:plane_model.get_plane_num()][vis_plane_mask_pre]
        if vis_plane_mask_pre.sum() <= 500:
            chunk_size = 40960 * 10
        elif vis_plane_mask_pre.sum() <= 1000:
            chunk_size = 40960 * 5
        else:
            chunk_size = 40960
        topK_inter_index = []

        for i in range(0, N_rays, chunk_size):
            topK = min(topK, vis_plane_mask_pre.sum())
            topK_upper = min(max(10, topK),vis_plane_mask_pre.sum())
            if topK == 0:
                return None
            topK_inter_index_i = get_topK_plane_index_core(
                    rays_o[i:i+chunk_size], 
                    rays_d[i:i+chunk_size],
                    plane_normal[vis_plane_mask_pre], 
                    plane_offset[vis_plane_mask_pre], 
                    plane_center[vis_plane_mask_pre], 
                    plane_radii[vis_plane_mask_pre], 
                    plane_rot_q[vis_plane_mask_pre], 
                    plane_xAxis[vis_plane_mask_pre], 
                    plane_yAxis[vis_plane_mask_pre],
                    topK_upper, topK, plane_model.plane_cfg, ite=ite)
            # torch.cuda.empty_cache()
            topK_inter_index.append(topK_inter_index_i)

        topK_inter_index = torch.cat(topK_inter_index, dim=0)  # n_ray, topk
        topK_inter_index = vis_plane_id_pre[topK_inter_index.reshape(-1)].reshape(N_rays, -1)
    return topK_inter_index

def get_vis_weight(Ralpha, clamp_min=0.):
    device = Ralpha.device
    transmittance = (
            torch.cumprod(
                torch.cat(
                    [torch.ones([*Ralpha.shape[:-1], 1], device=device), (1 - Ralpha).clamp(clamp_min)], dim=-1),
                dim=-1)[..., :-1]
    )
    vis_weight = (Ralpha + 1e-10) * transmittance
    return vis_weight

def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb: # [-1,1] --> [0,1]
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1. ) / 2.

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cuda())

    return psnr

def split_y_axis(selected_mask, plane_y_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z):
        selected_plane_center = plane_center[selected_mask]
        selected_plane_y_axis = plane_y_axis[selected_mask]
        selected_plane_radii_x_p = plane_radii_xy_p[:, 0:1][selected_mask]
        selected_plane_radii_y_p = plane_radii_xy_p[:, 1:2][selected_mask]
        selected_plane_radii_x_n = plane_radii_xy_n[:, 0:1][selected_mask]
        selected_plane_radii_y_n = plane_radii_xy_n[:, 1:2][selected_mask]

        new1_plane_center = selected_plane_center + selected_plane_y_axis * 0.5 * selected_plane_radii_y_p
        new1_plane_radii_xy_p = torch.cat([selected_plane_radii_x_p, 0.5 * selected_plane_radii_y_p], dim=-1)
        new1_plane_radii_xy_n = torch.cat([selected_plane_radii_x_n, 0.5 * selected_plane_radii_y_p], dim=-1)
        
        new2_plane_center = selected_plane_center - selected_plane_y_axis * 0.5 * selected_plane_radii_y_n
        new2_plane_radii_xy_p = torch.cat([selected_plane_radii_x_p, 0.5 * selected_plane_radii_y_n], dim=-1)
        new2_plane_radii_xy_n = torch.cat([selected_plane_radii_x_n, 0.5 * selected_plane_radii_y_n], dim=-1)
        
        new_plane_center = torch.cat([new1_plane_center, new2_plane_center], dim=0)
        new_plane_radii_xy_p = torch.cat([new1_plane_radii_xy_p, new2_plane_radii_xy_p], dim=0)
        new_plane_radii_xy_n = torch.cat([new1_plane_radii_xy_n, new2_plane_radii_xy_n], dim=0)

        new_plane_rot_q_normal_wxy = plane_rot_q_normal_wxy[selected_mask].repeat(2, 1)
        new_plane_rot_q_xyAxis_w = plane_rot_q_xyAxis_w[selected_mask].repeat(2, 1)
        new_plane_rot_q_xyAxis_z = plane_rot_q_xyAxis_z[selected_mask].repeat(2, 1)

        return new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z

def split_x_axis(selected_mask, plane_x_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z):
        selected_plane_center = plane_center[selected_mask]
        selected_plane_x_axis = plane_x_axis[selected_mask]
        selected_plane_radii_x_p = plane_radii_xy_p[:, 0:1][selected_mask]
        selected_plane_radii_y_p = plane_radii_xy_p[:, 1:2][selected_mask]
        selected_plane_radii_x_n = plane_radii_xy_n[:, 0:1][selected_mask]
        selected_plane_radii_y_n = plane_radii_xy_n[:, 1:2][selected_mask]

        new1_plane_center = selected_plane_center + selected_plane_x_axis * 0.5 * selected_plane_radii_x_p
        new1_plane_radii_xy_p = torch.cat([selected_plane_radii_x_p * 0.5, selected_plane_radii_y_p], dim=-1)
        new1_plane_radii_xy_n = torch.cat([selected_plane_radii_x_p * 0.5, selected_plane_radii_y_n], dim=-1)
        
        new2_plane_center = selected_plane_center - selected_plane_x_axis * 0.5 * selected_plane_radii_x_n
        new2_plane_radii_xy_p = torch.cat([selected_plane_radii_x_n * 0.5, selected_plane_radii_y_p], dim=-1)
        new2_plane_radii_xy_n = torch.cat([selected_plane_radii_x_n * 0.5, selected_plane_radii_y_n], dim=-1)

        new_plane_center = torch.cat([new1_plane_center, new2_plane_center], dim=0)
        new_plane_radii_xy_p = torch.cat([new1_plane_radii_xy_p, new2_plane_radii_xy_p], dim=0)
        new_plane_radii_xy_n = torch.cat([new1_plane_radii_xy_n, new2_plane_radii_xy_n], dim=0)

        new_plane_rot_q_normal_wxy = plane_rot_q_normal_wxy[selected_mask].repeat(2, 1)
        new_plane_rot_q_xyAxis_w = plane_rot_q_xyAxis_w[selected_mask].repeat(2, 1)
        new_plane_rot_q_xyAxis_z = plane_rot_q_xyAxis_z[selected_mask].repeat(2, 1)

        return new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q_normal_wxy, new_plane_rot_q_xyAxis_w, new_plane_rot_q_xyAxis_z

def split_xy_axis(selected_mask, plane_x_axis, plane_y_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z):
        new1_plane_center, new1_plane_radii_xy_p, new1_plane_radii_xy_n, new1_plane_rot_q_normal_wxy, new1_plane_rot_q_xyAxis_w, new1_plane_rot_q_xyAxis_z = split_y_axis(
            selected_mask, plane_y_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z)

        selected_mask_1 = torch.ones(new1_plane_center.shape[0]).cuda() > 0
        new3_plane_center, new3_plane_radii_xy_p, new3_plane_radii_xy_n, new3_plane_rot_q_normal_wxy, new3_plane_rot_q_xyAxis_w, new3_plane_rot_q_xyAxis_z = split_x_axis(
            selected_mask_1, plane_x_axis[selected_mask].repeat(2, 1), new1_plane_center, new1_plane_radii_xy_p, new1_plane_radii_xy_n, new1_plane_rot_q_normal_wxy, new1_plane_rot_q_xyAxis_w, new1_plane_rot_q_xyAxis_z)

        return new3_plane_center, new3_plane_radii_xy_p, new3_plane_radii_xy_n, new3_plane_rot_q_normal_wxy, new3_plane_rot_q_xyAxis_w, new3_plane_rot_q_xyAxis_z

def get_split_mask_via_radii_grad(grads_radii, plane_radii_xy_p, plane_radii_xy_n, radii_ratio, radii_min, split_thres):
    grads_radii_max = grads_radii.max(dim=-1)[0]
    assert grads_radii_max.dim() == 1
    grads_radii_x_p = grads_radii[:, 0].contiguous()
    grads_radii_y_p = grads_radii[:, 1].contiguous()
    grads_radii_x_n = grads_radii[:, 2].contiguous()
    grads_radii_y_n = grads_radii[:, 3].contiguous()
    grad_radii_y_max = torch.max(grads_radii_y_p, grads_radii_y_n)
    grad_radii_x_max = torch.max(grads_radii_x_p, grads_radii_x_n)
    radii_x_split_mask = (plane_radii_xy_p[:, 0] + plane_radii_xy_n[:, 0]) > radii_ratio * radii_min
    radii_y_split_mask = (plane_radii_xy_p[:, 1] + plane_radii_xy_n[:, 1]) > radii_ratio * radii_min
    grad_radii_x_split_mask = grad_radii_x_max >= split_thres
    grad_radii_y_split_mask = grad_radii_y_max >= split_thres
    x_split_mask = radii_x_split_mask & grad_radii_x_split_mask
    y_split_mask = radii_y_split_mask & grad_radii_y_split_mask
    return x_split_mask, y_split_mask

def split_planes_via_mask(split_y_mask, split_x_mask, split_xy_mask, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z, rot_delta):
    new_plane_center = []
    new_plane_radii_xy_p = []
    new_plane_radii_xy_n = []
    new_plane_rot_q_normal_wxy = []
    new_plane_rot_q_xyAxis_w = []
    new_plane_rot_q_xyAxis_z = []
    new_rot_delta = []
    
    if split_y_mask.sum() > 0:
            new1_plane_center, new1_plane_radii_xy_p, new1_plane_radii_xy_n, new1_plane_rot_q_normal_wxy, new1_plane_rot_q_xyAxis_w, new1_plane_rot_q_xyAxis_z = split_y_axis(
                split_y_mask, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z)
            new_plane_center.append(new1_plane_center)
            new_plane_radii_xy_p.append(new1_plane_radii_xy_p)
            new_plane_radii_xy_n.append(new1_plane_radii_xy_n)
            new_plane_rot_q_normal_wxy.append(new1_plane_rot_q_normal_wxy)
            new_plane_rot_q_xyAxis_w.append(new1_plane_rot_q_xyAxis_w)
            new_plane_rot_q_xyAxis_z.append(new1_plane_rot_q_xyAxis_z)
            if rot_delta is not None:
                new_rot_delta.append(torch.cat([rot_delta[split_y_mask], rot_delta[split_y_mask]], dim=0))
    if split_x_mask.sum() > 0:
            new2_plane_center, new2_plane_radii_xy_p, new2_plane_radii_xy_n, new2_plane_rot_q_normal_wxy, new2_plane_rot_q_xyAxis_w, new2_plane_rot_q_xyAxis_z = split_x_axis(
                split_x_mask, plane_xAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z)
            new_plane_center.append(new2_plane_center)
            new_plane_radii_xy_p.append(new2_plane_radii_xy_p)
            new_plane_radii_xy_n.append(new2_plane_radii_xy_n)
            new_plane_rot_q_normal_wxy.append(new2_plane_rot_q_normal_wxy)
            new_plane_rot_q_xyAxis_w.append(new2_plane_rot_q_xyAxis_w)
            new_plane_rot_q_xyAxis_z.append(new2_plane_rot_q_xyAxis_z)
            if rot_delta is not None:
                new_rot_delta.append(torch.cat([rot_delta[split_x_mask], rot_delta[split_x_mask]], dim=0))
    if split_xy_mask.sum() > 0:
            new3_plane_center, new3_plane_radii_xy_p, new3_plane_radii_xy_n, new3_plane_rot_q_normal_wxy, new3_plane_rot_q_xyAxis_w, new3_plane_rot_q_xyAxis_z = split_xy_axis(
                split_xy_mask, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q_normal_wxy, plane_rot_q_xyAxis_w, plane_rot_q_xyAxis_z)
            new_plane_center.append(new3_plane_center)
            new_plane_radii_xy_p.append(new3_plane_radii_xy_p)
            new_plane_radii_xy_n.append(new3_plane_radii_xy_n)
            new_plane_rot_q_normal_wxy.append(new3_plane_rot_q_normal_wxy)
            new_plane_rot_q_xyAxis_w.append(new3_plane_rot_q_xyAxis_w)
            new_plane_rot_q_xyAxis_z.append(new3_plane_rot_q_xyAxis_z)
            if rot_delta is not None:
                new_rot_delta.append(torch.cat([rot_delta[split_xy_mask], rot_delta[split_xy_mask], rot_delta[split_xy_mask], rot_delta[split_xy_mask]], dim=0))

    if len(new_plane_center) > 0:
            new_plane_center = torch.cat(new_plane_center, dim=0)
            new_plane_radii_xy_p = torch.cat(new_plane_radii_xy_p, dim=0)
            new_plane_radii_xy_n = torch.cat(new_plane_radii_xy_n, dim=0)
            new_plane_rot_q_normal_wxy = torch.cat(new_plane_rot_q_normal_wxy, dim=0)
            new_plane_rot_q_xyAxis_w = torch.cat(new_plane_rot_q_xyAxis_w, dim=0)
            new_plane_rot_q_xyAxis_z = torch.cat(new_plane_rot_q_xyAxis_z, dim=0)
            if rot_delta is not None:
                new_rot_delta = torch.cat(new_rot_delta, dim=0)

    return new_plane_center,new_plane_radii_xy_p,new_plane_radii_xy_n,new_plane_rot_q_normal_wxy,new_plane_rot_q_xyAxis_w,new_plane_rot_q_xyAxis_z,new_rot_delta

def split_y_axis2(selected_mask, plane_y_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q):
        selected_plane_center = plane_center[selected_mask]
        selected_plane_y_axis = plane_y_axis[selected_mask]
        selected_plane_radii_x_p = plane_radii_xy_p[:, 0:1][selected_mask]
        selected_plane_radii_y_p = plane_radii_xy_p[:, 1:2][selected_mask]
        selected_plane_radii_x_n = plane_radii_xy_n[:, 0:1][selected_mask]
        selected_plane_radii_y_n = plane_radii_xy_n[:, 1:2][selected_mask]

        new1_plane_center = selected_plane_center + selected_plane_y_axis * 0.5 * selected_plane_radii_y_p
        new1_plane_radii_xy_p = torch.cat([selected_plane_radii_x_p, 0.5 * selected_plane_radii_y_p], dim=-1)
        new1_plane_radii_xy_n = torch.cat([selected_plane_radii_x_n, 0.5 * selected_plane_radii_y_p], dim=-1)
        
        new2_plane_center = selected_plane_center - selected_plane_y_axis * 0.5 * selected_plane_radii_y_n
        new2_plane_radii_xy_p = torch.cat([selected_plane_radii_x_p, 0.5 * selected_plane_radii_y_n], dim=-1)
        new2_plane_radii_xy_n = torch.cat([selected_plane_radii_x_n, 0.5 * selected_plane_radii_y_n], dim=-1)
        
        new_plane_center = torch.cat([new1_plane_center, new2_plane_center], dim=0)
        new_plane_radii_xy_p = torch.cat([new1_plane_radii_xy_p, new2_plane_radii_xy_p], dim=0)
        new_plane_radii_xy_n = torch.cat([new1_plane_radii_xy_n, new2_plane_radii_xy_n], dim=0)

        new_plane_rot_q = plane_rot_q[selected_mask].repeat(2, 1)

        return new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q

def split_x_axis2(selected_mask, plane_x_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q):
        selected_plane_center = plane_center[selected_mask]
        selected_plane_x_axis = plane_x_axis[selected_mask]
        selected_plane_radii_x_p = plane_radii_xy_p[:, 0:1][selected_mask]
        selected_plane_radii_y_p = plane_radii_xy_p[:, 1:2][selected_mask]
        selected_plane_radii_x_n = plane_radii_xy_n[:, 0:1][selected_mask]
        selected_plane_radii_y_n = plane_radii_xy_n[:, 1:2][selected_mask]

        new1_plane_center = selected_plane_center + selected_plane_x_axis * 0.5 * selected_plane_radii_x_p
        new1_plane_radii_xy_p = torch.cat([selected_plane_radii_x_p * 0.5, selected_plane_radii_y_p], dim=-1)
        new1_plane_radii_xy_n = torch.cat([selected_plane_radii_x_p * 0.5, selected_plane_radii_y_n], dim=-1)
        
        new2_plane_center = selected_plane_center - selected_plane_x_axis * 0.5 * selected_plane_radii_x_n
        new2_plane_radii_xy_p = torch.cat([selected_plane_radii_x_n * 0.5, selected_plane_radii_y_p], dim=-1)
        new2_plane_radii_xy_n = torch.cat([selected_plane_radii_x_n * 0.5, selected_plane_radii_y_n], dim=-1)

        new_plane_center = torch.cat([new1_plane_center, new2_plane_center], dim=0)
        new_plane_radii_xy_p = torch.cat([new1_plane_radii_xy_p, new2_plane_radii_xy_p], dim=0)
        new_plane_radii_xy_n = torch.cat([new1_plane_radii_xy_n, new2_plane_radii_xy_n], dim=0)

        new_plane_rot_q = plane_rot_q[selected_mask].repeat(2, 1)

        return new_plane_center, new_plane_radii_xy_p, new_plane_radii_xy_n, new_plane_rot_q

def split_xy_axis2(selected_mask, plane_x_axis, plane_y_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q):
        new1_plane_center, new1_plane_radii_xy_p, new1_plane_radii_xy_n, new1_plane_rot_q = split_y_axis(
            selected_mask, plane_y_axis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q)

        selected_mask_1 = torch.ones(new1_plane_center.shape[0]).cuda() > 0
        new3_plane_center, new3_plane_radii_xy_p, new3_plane_radii_xy_n, new3_plane_rot_q = split_x_axis(
            selected_mask_1, plane_x_axis[selected_mask].repeat(2, 1), new1_plane_center, new1_plane_radii_xy_p, new1_plane_radii_xy_n, new1_plane_rot_q)

        return new3_plane_center, new3_plane_radii_xy_p, new3_plane_radii_xy_n, new3_plane_rot_q

def split_planes_via_mask2(split_y_mask, split_x_mask, split_xy_mask, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q, rot_delta):
    new_plane_center = []
    new_plane_radii_xy_p = []
    new_plane_radii_xy_n = []
    new_plane_rot_q = []
    new_rot_delta = []
    
    if split_y_mask.sum() > 0:
            new1_plane_center, new1_plane_radii_xy_p, new1_plane_radii_xy_n, new1_plane_rot_q = split_y_axis2(
                split_y_mask, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q)
            new_plane_center.append(new1_plane_center)
            new_plane_radii_xy_p.append(new1_plane_radii_xy_p)
            new_plane_radii_xy_n.append(new1_plane_radii_xy_n)
            new_plane_rot_q.append(new1_plane_rot_q)
            if rot_delta is not None:
                new_rot_delta.append(torch.cat([rot_delta[split_y_mask], rot_delta[split_y_mask]], dim=0))
    if split_x_mask.sum() > 0:
            new2_plane_center, new2_plane_radii_xy_p, new2_plane_radii_xy_n, new2_plane_rot_q = split_x_axis2(
                split_x_mask, plane_xAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q)
            new_plane_center.append(new2_plane_center)
            new_plane_radii_xy_p.append(new2_plane_radii_xy_p)
            new_plane_radii_xy_n.append(new2_plane_radii_xy_n)
            new_plane_rot_q.append(new2_plane_rot_q)
            if rot_delta is not None:
                new_rot_delta.append(torch.cat([rot_delta[split_x_mask], rot_delta[split_x_mask]], dim=0))
    if split_xy_mask.sum() > 0:
            new3_plane_center, new3_plane_radii_xy_p, new3_plane_radii_xy_n, new3_plane_rot_q = split_xy_axis2(
                split_xy_mask, plane_xAxis, plane_yAxis, plane_center, plane_radii_xy_p, plane_radii_xy_n, plane_rot_q)
            new_plane_center.append(new3_plane_center)
            new_plane_radii_xy_p.append(new3_plane_radii_xy_p)
            new_plane_radii_xy_n.append(new3_plane_radii_xy_n)
            new_plane_rot_q.append(new3_plane_rot_q)
            if rot_delta is not None:
                new_rot_delta.append(torch.cat([rot_delta[split_xy_mask], rot_delta[split_xy_mask], rot_delta[split_xy_mask], rot_delta[split_xy_mask]], dim=0))

    if len(new_plane_center) > 0:
            new_plane_center = torch.cat(new_plane_center, dim=0)
            new_plane_radii_xy_p = torch.cat(new_plane_radii_xy_p, dim=0)
            new_plane_radii_xy_n = torch.cat(new_plane_radii_xy_n, dim=0)
            new_plane_rot_q = torch.cat(new_plane_rot_q, dim=0)
            if rot_delta is not None:
                new_rot_delta = torch.cat(new_rot_delta, dim=0)

    return new_plane_center,new_plane_radii_xy_p,new_plane_radii_xy_n,new_plane_rot_q,new_rot_delta

# def check_point_inter_plane(point)

def clip_2d_lines_image_region(image_size, lines, image=None):
    [h, w] = image_size
    # new_lines = torch.unique(lines, dim=0)
    # new_lines = lines.clone()
    left = 0
    right = w - 1
    top = 0
    bottom = h - 1

    clipped_lines = []
    N = lines.shape[0]
    # Iterate over each line
    for i in range(N):
        x1, y1, x2, y2 = lines[i]

        if x1 < left:
            t = (x1 - left) / (x1 - x2)
            if 0<t<1:
                y1 -= t * (y1 - y2)
                x1 = left
            # else:
            #     continue
        elif x1 > right:
            t = (x1 - right) / (x1 - x2)
            if 0<t<1:
                y1 -= t * (y1 - y2)
                x1 = right
            # else:
            #     continue
        if y1 < top:
            t = (y1 - top) / (y1 - y2)
            if 0<t<1:
                x1 -= t * (x1 - x2)
                y1 = top
            # else:
            #     continue
        if y1 > bottom:
            t = (y1 - bottom) / (y1 - y2)
            if 0<t<1:
                x1 -= t * (x1 - x2)
                y1 = bottom
            # else:
            #     continue

        if x2 < left:
            t = (x2 - left) / (x2 - x1)
            if 0<t<1:
                y2 -= t * (y2 - y1)
                x2 = left
            # else:
            #     continue
        elif x2 > right:
            t = (x2 - right) / (x2 - x1)
            if 0<t<1:
                y2 -= t * (y2 - y1)
                x2 = right
            # else:
            #     continue
        if y2 < top:
            t = (y2 - top) / (y2 - y1)
            if 0<t<1:
                x2 -= t * (x2 - x1)
                y2 = top
            # else:
            #     continue
        if y2 > bottom:
            t = (y2 - bottom) / (y2 - y1)
            if 0<t<1:
                x2 -= t * (x2 - x1)
                y2 = bottom
            # else:
            #     continue

        clipped_lines.append([x1, y1, x2, y2])

    return torch.tensor(clipped_lines)


def calculate_lines_2d_dist(lines, lines_gt, dtype='e'):
    if dtype == 'e':
        dist = torch.sqrt(torch.sum((lines.reshape(-1,2)-lines_gt.reshape(-1,2))**2,dim=-1)).reshape(-1,2)
    elif dtype == 'l':
        def cross(A, B):
            return A[...,0] * B[...,1] - A[...,1] * B[...,0]
        line_vector = lines_gt[..., :2] - lines_gt[..., 2:]
        edp1_vector = lines[..., :2] - lines_gt[..., :2]
        edp2_vector = lines[..., 2:] - lines_gt[..., 2:]
        dist1 = torch.abs(cross(edp1_vector, line_vector)) / torch.linalg.norm(line_vector, dim=-1) 
        dist2 = torch.abs(cross(edp2_vector, line_vector)) / torch.linalg.norm(line_vector, dim=-1) 
        dist = torch.stack([dist1,dist2], dim=-1)
    else:
        raise TypeError("Please use dtype=['e','l'] for the calculation of distance.")

    return dist

def check_wireframe_from_lines_juncs(lines, juncs, threshold=1.0):
    cost1 = torch.sqrt(torch.sum((lines[:,:3]-juncs[:,None])**2, dim=-1))
    cost2 = torch.sqrt(torch.sum((lines[:,3:]-juncs[:,None])**2,dim=-1))
    
    dis1, idx_junc_to_end1 = cost1.min(dim=0)
    dis2, idx_junc_to_end2 = cost2.min(dim=0)

    idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
    idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

    iskeep = idx_junc_to_end_min < idx_junc_to_end_max ## not the same junction
    iskeep *= (dis1<threshold) & (dis2<threshold)
    
    idx_lines_for_junctions = torch.stack((idx_junc_to_end_min[iskeep],idx_junc_to_end_max[iskeep]),dim=1)#.unique(dim=0)
    wireframe = juncs[idx_lines_for_junctions]

    return wireframe

def merge_segments(segments, use_pca=False):
    from sklearn.decomposition import PCA
    """
    Merge multiple 3D line segments into a single long line segment.
    Parameters:
        segments (np.ndarray): Array of shape (N, 6) or (N, 2, 3) representing N line segments.
        use_pca (bool): Whether to use PCA for merging. If False, uses average direction method.
    Returns:
        merged_line (np.ndarray): Merged line segment endpoints, shape (6,).
    """

    if use_pca:
        points = segments.reshape(-1, 3)  # [2N, 3]
        pca = PCA(n_components=3)
        pca.fit(points)  # directly fit all points

        # find the main direction (first principal component)
        main_direction = pca.components_[0]  # shape: (3,)
        # find the mean point
        mean_point = pca.mean_  # shape: (3,)
        # project all points onto the main direction
        centered_points = points - mean_point
        projections = np.dot(centered_points, main_direction)  # shape: [2N]
        # find min and max projection values
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        # calculate the merged line segment endpoints
        start_point = mean_point + min_proj * main_direction
        end_point = mean_point + max_proj * main_direction
        # merge as a single line segment [x1,y1,z1,x2,y2,z2]
        merged_line = np.array([start_point, end_point]).flatten()
    else:
        segments = segments.reshape(-1, 2, 3)  # [N, 2, 3]
        # find the centroid of all line segments
        midpoints = np.mean(segments, axis=1)
        avg_midpoint = np.mean(midpoints, axis=0)
        # Calculate the direction vectors of each line segment and normalize them
        directions = segments[:, 1, :] - segments[:, 0, :]
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        # Ensure the orientation of the direction vectors is consistent 
        if directions.size == 0:
            return np.array([avg_midpoint, avg_midpoint])
        # use the first line segment as a reference
        ref_dir = directions[0]
        for i in range(len(directions)):
            if np.dot(directions[i], ref_dir) < 0:
                directions[i] *= -1
        # calculate the normalized average direction as the main direction
        avg_dir = np.sum(directions, axis=0)/np.sum(norms, axis=0)
        main_dir = avg_dir / np.linalg.norm(avg_dir)
        # Project all endpoints to determine the range of the line segments
        all_points = segments.reshape(-1, 3)
        vectors_to_avg = all_points - avg_midpoint
        ts = np.dot(vectors_to_avg, main_dir)
        t_min, t_max = np.min(ts), np.max(ts)
        # Calculate the merged endpoints
        start = avg_midpoint + t_min * main_dir
        end = avg_midpoint + t_max * main_dir
        merged_line = np.array([start, end]).flatten()
    
    return merged_line

def extract_endpoints_and_build_adjacency_matrix(segments):
    """
    Extract the set of unique endpoints and build an adjacency matrix based on endpoint indices.
    Input:
        segments: Collection of 3D line segments, shape [N, 2, 3], representing N line segments, each with two endpoints.
    Output:
        endpoints: Set of unique endpoints, shape [M, 3], representing M unique endpoints.
        adjacency_matrix: Adjacency matrix, shape [M, M], representing the connectivity between endpoints.
    """
    # Flatten all endpoints
    all_points = segments.reshape(-1, 3)
    # Get the unique endpoints and their indices
    unique_points, indices = np.unique(all_points, axis=0, return_inverse=True)
    # Get the number of unique endpoints
    num_endpoints = len(unique_points)
    # Initialize the adjacency matrix
    # adjacency_matrix = np.zeros((num_endpoints, num_endpoints), dtype=int)
    adjacency_matrix = []
    # Iterate over all line segments and fill the adjacency matrix
    for segment in segments:
        point1 = segment[0:3]
        point2 = segment[3:6]
        idx1 = np.where((unique_points == point1).all(axis=1))[0][0]  # Find the index of endpoint 1
        idx2 = np.where((unique_points == point2).all(axis=1))[0][0]  # Find the index of endpoint 2
        # Set the corresponding position in the adjacency matrix to 1
        # adjacency_matrix[idx1, idx2] = 1
        # adjacency_matrix[idx2, idx1] = 1  # Undirected graph, need to set the symmetric position
        adjacency_matrix.append([int(idx1), int(idx2)])
    return unique_points, adjacency_matrix  
       
def cluster_and_merge_segments_(lines):
    N = len(lines)
    processed = np.zeros(N, dtype=bool)
    all_merged_lines = []
    all_merged_dist = []
    final_merged_lines = []
    counts = []
    lines_list = lines.tolist()
    # while len(lines_list)>0:
    while (~processed).sum()>0:
        line = np.array(lines_list[0])
        processed[0] = True
        merged_lines = [line]
        merged_dist = []
        pop_list = []
        for i in range(len(lines_list)-1):
            if ~processed[i]:
                line_new = np.array(lines_list[i])
                processed[i] = True
                for line_mg in merged_lines:
                    vec1 = line_new[:3]-line_new[3:6]
                    vec1_norm = np.linalg.norm(vec1)
                    vec2 = line_mg[:3]-line_mg[3:6]
                    vec2_norm = np.linalg.norm(vec2)
                    cos_theta = np.abs(np.dot(vec1, vec2) / (vec1_norm * vec2_norm))
                    cos_theta = np.clip(cos_theta, 0, 1)
                    theta_rad = np.arccos(cos_theta)

                    if theta_rad < 0.2:
                        vector1 = line_new[:3]-line_mg[:3]
                        vector2 = line_new[3:6]-line_mg[:3]
                        line_vector = line_mg[3:6]-line_mg[:3]
                        t1 = np.dot(vector1, line_vector) / np.linalg.norm(line_vector)**2
                        t2 = np.dot(vector2, line_vector) / np.linalg.norm(line_vector)**2
                        dist1 = np.linalg.norm(vector1-t1*line_vector)
                        dist2 = np.linalg.norm(vector2-t2*line_vector)
                        if (0<=t1<=1 or 0<=t2<=1) and (dist1<0.05 and dist2<0.05):
                            merged_lines.append(line_new)
                            merged_dist.append(np.max([dist1, dist2]))
                            break

        merged_lines_array = np.array(merged_lines)
        all_merged_lines.append(merged_lines_array.tolist())
        merged_dist_array = np.array(merged_dist)
        all_merged_dist.append(merged_dist_array.tolist())
        counts.append(len(merged_lines))
        if len(merged_lines) > 1:
            merged_line = merge_segments(merged_lines_array.reshape(-1, 2, 3)).reshape(6)
            # final_merged_lines.append(merge_segments(merged_lines_array.reshape(-1, 2, 3)))
        else:
            merged_line = merged_lines_array[0]
        final_merged_lines.append(merged_line)

    final_merged_lines = np.array(final_merged_lines)

    data = {
        "all_merged_lines": all_merged_lines,
        "final_merged_lines": final_merged_lines.tolist(),
        "all_merged_dist": all_merged_dist,
        "counts": counts,
    }

    return data
    
def cluster_and_merge_segments_org(lines):
    all_merged_lines = []
    all_merged_dist = []
    final_merged_lines = []
    counts = []
    lines_list = lines.tolist()
    while len(lines_list)>0:
        line = np.array(lines_list.pop(0))
        merged_lines = [line]
        merged_dist = []
        pop_list = []
        for i in tqdm(range(len(lines_list)-1, -1, -1)):
            line_new = np.array(lines_list[i])
            for line_mg in merged_lines:
                vec1 = line_new[:3]-line_new[3:6]
                vec1_norm = np.linalg.norm(vec1)
                vec2 = line_mg[:3]-line_mg[3:6]
                vec2_norm = np.linalg.norm(vec2)
                cos_theta = np.abs(np.dot(vec1, vec2) / (vec1_norm * vec2_norm))
                cos_theta = np.clip(cos_theta, 0, 1)
                theta_rad = np.arccos(cos_theta)

                if theta_rad < 0.2:
                    vector1 = line_new[:3]-line_mg[:3]
                    vector2 = line_new[3:6]-line_mg[:3]
                    line_vector = line_mg[3:6]-line_mg[:3]
                    t1 = np.dot(vector1, line_vector) / np.linalg.norm(line_vector)**2
                    t2 = np.dot(vector2, line_vector) / np.linalg.norm(line_vector)**2
                    dist1 = np.linalg.norm(vector1-t1*line_vector)
                    dist2 = np.linalg.norm(vector2-t2*line_vector)
                    if (0<=t1<=1 or 0<=t2<=1) and (dist1<0.05 and dist2<0.05):
                        merged_lines.append(line_new)
                        merged_dist.append(np.max([dist1, dist2]))
                        # import pdb;pdb.set_trace()
                        lines_list.pop(i)
                        break

        merged_lines_array = np.array(merged_lines)
        all_merged_lines.append(merged_lines_array.tolist())
        merged_dist_array = np.array(merged_dist)
        all_merged_dist.append(merged_dist_array.tolist())
        counts.append(len(merged_lines))
        if len(merged_lines) > 1:
            merged_line = merge_segments(merged_lines_array.reshape(-1, 2, 3)).reshape(6)
            # final_merged_lines.append(merge_segments(merged_lines_array.reshape(-1, 2, 3)))
        else:
            merged_line = merged_lines_array[0]
        final_merged_lines.append(merged_line)

    final_merged_lines = np.array(final_merged_lines)

    # unique_points, adjacency_matrix = extract_endpoints_and_build_adjacency_matrix(final_merged_lines)
    data = {
        "all_merged_lines": all_merged_lines,
        "final_merged_lines": final_merged_lines.tolist(),
        "all_merged_dist": all_merged_dist,
        "counts": counts,
        # "unique_points": unique_points.tolist(),
        # "adjacency_matrix": adjacency_matrix,
    }

    # return all_merged_lines, final_merged_lines, all_merged_dist, counts, unique_points, adjacency_matrix
    return data
    
# def cluster_and_merge_segments(lines, rad_th=0.2, dist_th=0.05):
def cluster_and_merge_segments(lines, rad_th=0.1, dist_th=0.05):
    # 计算两两线段之间的夹角
    starts = lines[:, :3]
    ends = lines[:, 3:]
    vecs = ends - starts
    norms = torch.norm(vecs, dim=1)
    norms_sq = torch.sum(vecs ** 2, dim=1)
    dot_matrix = torch.matmul(vecs, vecs.T)  # 点积矩阵
    norm_product = norms.unsqueeze(1) * norms.unsqueeze(0)
    cos_theta = torch.abs(dot_matrix / (norm_product + 1e-8))
    cos_theta = torch.clamp(cos_theta, 0, 1)
    theta_rad = torch.acos(cos_theta)
    angle_mask = theta_rad < rad_th
    # angle_mask.fill_diagonal_(False)

    vec1 = starts.unsqueeze(1) - starts.unsqueeze(0)  # (N,N,3)
    vec2 = ends.unsqueeze(1) - starts.unsqueeze(0)    # (N,N,3)
    # t1, t2
    dot1 = torch.sum(vec1 * vecs.unsqueeze(0), dim=2)  # (N,N)
    dot2 = torch.sum(vec2 * vecs.unsqueeze(0), dim=2)
    t1 = dot1 / (norms_sq.unsqueeze(0) + 1e-8)
    t2 = dot2 / (norms_sq.unsqueeze(0) + 1e-8)
    t1_valid = (t1 >= 0) & (t1 <= 1)
    t2_valid = (t2 >= 0) & (t2 <= 1)
    t_mask = t1_valid | t2_valid
    # t_mask.fill_diagonal_(False)
    # dist1, dist2
    proj_vec1 = t1.unsqueeze(2) * vecs.unsqueeze(0)
    proj_vec2 = t2.unsqueeze(2) * vecs.unsqueeze(0)
    dist1 = torch.norm(vec1 - proj_vec1, dim=2)
    dist2 = torch.norm(vec2 - proj_vec2, dim=2)
    dist_mask = (dist1 < dist_th) & (dist2 < dist_th)
    # dist_mask.fill_diagonal_(False)

    M = angle_mask & t_mask & dist_mask
    groups = gpu_merge_overlapped_planes(M)

    # ####
    # print(len(groups), dist_th)
    # checker = SimpleGroupChecker(dist_thresh=dist_th*1.0)
    # groups = checker.check(groups, lines)
    # print(len(groups))

    all_merged_lines = []
    final_merged_lines = []
    counts = []
    for group in groups:
        counts.append(len(group))
        try:
            merged_lines_array = lines[torch.tensor(group)].numpy()
        except:
            print('skip empty group...')
            continue
        all_merged_lines.append(merged_lines_array.tolist())
    
        if len(group) > 1:
            merged_line = merge_segments(merged_lines_array.reshape(-1, 2, 3)).reshape(6)
        else:
            merged_line = merged_lines_array[0]
        final_merged_lines.append(merged_line)

    final_merged_lines = np.array(final_merged_lines)

    data = {
        "all_merged_lines": all_merged_lines,
        # "final_merged_lines": final_merged_lines.tolist(),
        "final_merged_lines": final_merged_lines,
        "counts": counts,
    }

    return data

def gpu_merge_overlapped_planes(M):
    M = M | M.t()
    adj = M.to(torch.float32)
    n = adj.size(0)
    label = torch.eye(n, dtype=torch.float32, device=M.device)
    while True:
        new_label = label @ adj
        new_label = (new_label > 0).to(torch.float32)
        if torch.allclose(new_label, label):
            break
        label = new_label
    mask = label.bool()
    unique_masks = torch.unique(mask, dim=0, sorted=False)
    groups = []
    for m in unique_masks:
        group = torch.where(m)[0].cpu().tolist()
        groups.append(group)
    return groups

def differentiable_unique(x):
    """
    Differentiable version of np.unique.
    """
    x_np = x.cpu().detach().numpy()
    sorted_indices = np.lexsort(x_np.T)
    x_sorted = x[sorted_indices]

    x_shifted = torch.roll(x_sorted, 1, 0)
    x_shifted[0] = x_shifted[0] - 1
    mask = torch.eq(x_sorted, x_shifted).all(dim=1)
    mask = ~mask
    unique_rows = x_sorted[mask]
    return x_sorted[mask]

# def differentiable_unique(batched_lines):
#     """
#     Differentiable version of np.unique.
#     """
#     lines_unique, idxs = torch.unique(batched_lines, dim=0, return_inverse=True)
#     if lines_unique.shape[0] >1:
#         idxs_s = torch.roll(idxs, 1)
#         lines_unique = batched_lines[(idxs-idxs_s)!=0]
    
#     return lines_unique

def calculate_lines_theta(vec_lines_exp, vec_planes_exp):
    vec_lines_exp = vec_lines_exp.cpu().numpy() if isinstance(vec_lines_exp, torch.Tensor) else vec_lines_exp
    vec_planes_exp = vec_planes_exp.cpu().numpy() if isinstance(vec_planes_exp, torch.Tensor) else vec_planes_exp
    import pdb;pdb.set_trace()
    cos_theta = numpy.abs((vec_lines_exp * vec_planes_exp).sum(dim=-1)) / (torch.norm(vec_lines_exp, dim=-1) * torch.norm(vec_planes_exp, dim=-1) + 1e-8)
    theta = torch.acos(cos_theta.clamp(-1,1))


from sklearn.cluster import DBSCAN
from collections import defaultdict

class SimpleGroupChecker:
    def __init__(self, dist_thresh=0.01):
        self.dist_thresh = dist_thresh

    def check(self, base_groups, lines):
        valid_groups = []
        for group in base_groups:
            if len(group) > 0:
                if self._validate_group(group, lines):
                    valid_groups.append(group)
                else:
                    valid_groups.extend(self._split_invalid_group(group, lines))
        return valid_groups

    def _validate_group(self, group, lines):
        group_lines = lines[group]
        
        dist_diff = self._calc_max_dist_diff(group_lines)
        
        if dist_diff > self.dist_thresh:
            return False
        
        return True

    def _calc_proj_dist_matrix(self, lines):
        starts = lines[:, :3]
        ends = lines[:, 3:]
        vecs = ends - starts
        norms_sq = torch.sum(vecs ** 2, dim=1)

        vec1 = starts.unsqueeze(1) - starts.unsqueeze(0)  # (N,N,3)
        vec2 = ends.unsqueeze(1) - starts.unsqueeze(0)    # (N,N,3)
        # t1, t2
        dot1 = torch.sum(vec1 * vecs.unsqueeze(0), dim=2)  # (N,N)
        dot2 = torch.sum(vec2 * vecs.unsqueeze(0), dim=2)
        t1 = dot1 / (norms_sq.unsqueeze(0) + 1e-8)
        t2 = dot2 / (norms_sq.unsqueeze(0) + 1e-8)

        # dist1, dist2
        proj_vec1 = t1.unsqueeze(2) * vecs.unsqueeze(0)
        proj_vec2 = t2.unsqueeze(2) * vecs.unsqueeze(0)
        dist1 = torch.norm(vec1 - proj_vec1, dim=2)
        dist2 = torch.norm(vec2 - proj_vec2, dim=2)

        dist_res = (dist1 + dist2) / 2.
        return dist_res

    def _calc_max_dist_diff(self, lines):
        starts = lines[:, :3]
        ends = lines[:, 3:]
        vecs = ends - starts
        norms_sq = torch.sum(vecs ** 2, dim=1)

        vec1 = starts.unsqueeze(1) - starts.unsqueeze(0)  # (N,N,3)
        vec2 = ends.unsqueeze(1) - starts.unsqueeze(0)    # (N,N,3)
        # t1, t2
        dot1 = torch.sum(vec1 * vecs.unsqueeze(0), dim=2)  # (N,N)
        dot2 = torch.sum(vec2 * vecs.unsqueeze(0), dim=2)
        t1 = dot1 / (norms_sq.unsqueeze(0) + 1e-8)
        t2 = dot2 / (norms_sq.unsqueeze(0) + 1e-8)

        # dist1, dist2
        proj_vec1 = t1.unsqueeze(2) * vecs.unsqueeze(0)
        proj_vec2 = t2.unsqueeze(2) * vecs.unsqueeze(0)
        dist1 = torch.norm(vec1 - proj_vec1, dim=2)
        dist2 = torch.norm(vec2 - proj_vec2, dim=2)

        max_dist = max(dist1.max().item(), dist2.max().item())
        return max_dist

    def _split_invalid_group(self, invalid_group, lines):
        from sklearn.cluster import DBSCAN

        dist_matrix = self._calc_proj_dist_matrix(lines[invalid_group])
        cpu_distance = dist_matrix.cpu().numpy()

        db = DBSCAN(
            eps=self.dist_thresh, 
            min_samples=1,
            metric='precomputed'
        )
        labels = db.fit_predict(cpu_distance)
        
        subgroups = defaultdict(list)
        for i, label in enumerate(labels):
            subgroups[label].append(invalid_group[i])
        return list(subgroups.values())
