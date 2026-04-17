"""
network_acc.py – Accelerated training network wrapper.

Subclasses RecWrapper and overrides get_inter_points_lines_theta_dist to use
the diff_assignment CUDA extension, eliminating the O(N_ray * N_plane)
intermediate tensors that dominate training time.

Usage (drop-in replacement for RecWrapper):

    from run.network_acc import AccRecWrapper as RecWrapper
"""

import os
import sys
import torch

# Ensure submodules/diff-assignment is on the path so the package can be
# found both before and after `pip install -e`.
# _HERE = os.path.dirname(os.path.abspath(__file__))
# _DIFF_ASSIGN_DIR = os.path.join(_HERE, '..', 'submodules', 'diff-assignment')
# if _DIFF_ASSIGN_DIR not in sys.path:
#     sys.path.insert(0, _DIFF_ASSIGN_DIR)

import diff_assignment

from utils import model_util
from run.network import RecWrapper


class AccRecWrapper(RecWrapper):
    """
    Drop-in replacement for RecWrapper with CUDA-accelerated
    get_inter_points_lines_theta_dist.

    The two bottleneck stages replaced by CUDA kernels are:

    1. Ray-plane intersection + nearest plane selection  (lines 480-489)
       Original: allocates [N_ray, N_plane, 3] intermediate tensors.
       Accelerated: one thread per ray, inner loop over planes —
                    O(N_ray + N_plane) memory.

    2. Line-plane edge assignment  (line 495 / find_min_theta_dist_line)
       Original: 4 separate Python list-comprehension passes over edges.
       Accelerated: single kernel, all 4 edges computed per thread.
    """

    def get_inter_points_lines_theta_dist(self, view_info, ite=-1):
        H, W = view_info.img_size
        mask_data = view_info.get_sampling_gt()
        uv        = mask_data['uv'][None]
        lines     = mask_data['lines']
        pose      = mask_data['pose'][None]
        intrinsics = mask_data['intrinsics'][None]

        rays_d, rays_o = model_util.get_raydir_camloc(uv, pose, intrinsics)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = rays_o.expand(rays_d.shape[0], 3)

        # Plane corners are needed outside no_grad for the gradient path
        plane_corners = self.planarSplat.get_plane_vertex(ite)

        with torch.no_grad():
            plane_normal, plane_center, plane_radii, plane_xAxis, plane_yAxis = \
                self.planarSplat.get_plane_property(ite)

            if ~(plane_radii.max() > 0):
                import pdb; pdb.set_trace()

            # ------------------------------------------------------------------
            # Stage 1 (CUDA): fused ray-plane intersection + nearest plane
            # ------------------------------------------------------------------
            # Returns tensors of shape [N_ray] / [N_ray, 3].
            # valid_mask  – bool [N_ray]: True where at least one plane was hit
            # min_idx     – int64 [N_ray]: index of nearest valid plane (-1 if none)
            # inter_pts   – float [N_ray, 3]: intersection point of nearest plane
            # min_sd      – float [N_ray]: signed distance to nearest intersection
            inter_pts, min_sd, min_idx, valid_mask = diff_assignment.ray_plane_intersect(
                rays_o, rays_d,
                plane_center, plane_normal, plane_radii,
                plane_xAxis, plane_yAxis,
            )

            # Gather corners and GT lines for the matched rays
            line_plane_corners = plane_corners[min_idx[valid_mask]]   # [M, 4, 3]
            lines_gt           = lines[valid_mask.cpu()]               # [M, L]

            # Project plane corners to 2-D image space
            line_plane_corners_2d = model_util.project2D(             # [M, 4, 2]
                pose, intrinsics, line_plane_corners
            )

            # ------------------------------------------------------------------
            # Stage 2 (CUDA): fused line-plane edge assignment
            # ------------------------------------------------------------------
            # For each matched ray, find the rectangle edge (0-3) whose
            # direction and position best match the GT line segment.
            theta, dist, final_indix = diff_assignment.line_plane_assign(
                lines_gt, line_plane_corners_2d
            )

        # ------------------------------------------------------------------
        # Gradient path: index into differentiable plane_corners
        # ------------------------------------------------------------------
        all_plane_lines_3d = model_util.get_plane_lines_from_corners(
            plane_corners[min_idx[valid_mask]]
        )
        plane_lines_3d = all_plane_lines_3d[
            torch.arange(len(line_plane_corners), device=all_plane_lines_3d.device),
            final_indix
        ]
        plane_lines_2d = model_util.project2D(
            pose, intrinsics, plane_lines_3d.reshape(-1, 2, 3)
        ).reshape(-1, 4)
        lines_gt = lines_gt[:, :4].cuda()

        # Resolve endpoint ordering ambiguity
        with torch.no_grad():
            dist1 = torch.sum(
                (plane_lines_2d - lines_gt) ** 2, dim=-1, keepdim=True
            ).detach()
            dist2 = torch.sum(
                (plane_lines_2d[:, [2, 3, 0, 1]] - lines_gt) ** 2,
                dim=-1, keepdim=True
            ).detach()

        plane_lines_3d = torch.where(
            dist1 < dist2, plane_lines_3d, plane_lines_3d[:, [3, 4, 5, 0, 1, 2]]
        )
        plane_lines_2d = torch.where(
            dist1 < dist2, plane_lines_2d, plane_lines_2d[:, [2, 3, 0, 1]]
        )

        outputs = {
            # Nearest valid intersection per ray (shape [N_ray, 3] instead of
            # the original [N_ray, N_plane, 3] — only the selected plane is kept)
            "inter_points_3d": inter_pts,
            "inter_SD":        min_sd,
            # mask_inlier / mask_positive are not stored separately in the
            # accelerated path; valid_mask combines both conditions plus
            # nearest-plane selection (equivalent to original 'mask').
            "mask_inlier":     None,
            "mask_positive":   None,
            "mask_inter":      valid_mask,
            "mask_plane":      min_idx,
            "plane_lines_3d":  plane_lines_3d,
            "plane_lines_2d":  plane_lines_2d,
            "lines_gt_2d":     lines_gt,
            "theta":           theta,
            "dist":            dist,
        }

        return outputs
