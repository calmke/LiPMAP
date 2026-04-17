"""
diff_assignment – CUDA-accelerated ray-plane intersection and line assignment.

Two main entry points:

    ray_plane_intersect(rays_o, rays_d, plane_center, plane_normal,
                        plane_radii, plane_xAxis, plane_yAxis)
        -> (inter_pts [N_ray,3], min_sd [N_ray], min_idx [N_ray], valid_mask [N_ray])

    line_plane_assign(lines_gt, corners_2d)
        -> (theta [N], dist [N], final_edge_idx [N])
"""

import torch
from . import _C


def ray_plane_intersect(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    plane_center: torch.Tensor,
    plane_normal: torch.Tensor,
    plane_radii: torch.Tensor,
    plane_xAxis: torch.Tensor,
    plane_yAxis: torch.Tensor,
):
    """
    Fused ray-plane intersection + nearest valid plane selection.

    Replaces the combination of:
        compute_ray2plane_intersections_v2  +
        check_intersections_in_plane        +
        find_positive_min_mask

    in a single CUDA kernel that avoids the O(N_ray * N_plane) intermediate
    tensors.

    Args:
        rays_o        : [N_ray, 3]   Ray origins (float32, CUDA).
        rays_d        : [N_ray, 3]   Ray directions (float32, CUDA).
        plane_center  : [N_plane, 3] Plane center positions (float32, CUDA).
        plane_normal  : [N_plane, 3] Plane normal vectors (float32, CUDA).
        plane_radii   : [N_plane, 4] Plane radii (rx+, ry+, rx-, ry-).
                        If shape is [N_plane, 2] it is expanded to 4 columns.
        plane_xAxis   : [N_plane, 3] Plane local x-axis directions.
        plane_yAxis   : [N_plane, 3] Plane local y-axis directions.

    Returns:
        inter_pts  : [N_ray, 3]  Nearest valid intersection point per ray
                                  (zeros for invalid rays).
        min_sd     : [N_ray]     Signed distance to nearest valid intersection
                                  (0 for invalid rays).
        min_idx    : [N_ray]     Index of nearest valid plane (-1 = invalid).
        valid_mask : [N_ray]     Boolean mask: True where a valid plane was hit.
    """
    # Normalise plane_radii from [N, 2] to [N, 4] if necessary
    if plane_radii.shape[-1] == 2:
        plane_radii = plane_radii.repeat(1, 2)  # [rx, ry] -> [rx, ry, rx, ry]

    # Ensure float32 and contiguous CUDA tensors
    def _prep(t):
        return t.float().contiguous()

    rays_o       = _prep(rays_o)
    rays_d       = _prep(rays_d)
    plane_center = _prep(plane_center)
    plane_normal = _prep(plane_normal)
    plane_radii  = _prep(plane_radii)
    plane_xAxis  = _prep(plane_xAxis)
    plane_yAxis  = _prep(plane_yAxis)

    return _C.ray_plane_intersect(
        rays_o, rays_d,
        plane_center, plane_normal, plane_radii,
        plane_xAxis, plane_yAxis,
    )


def line_plane_assign(
    lines_gt: torch.Tensor,
    corners_2d: torch.Tensor,
):
    """
    Line-plane edge assignment using top-2 theta + min dist tiebreak.

    Replaces find_min_theta_dist_line(use_theta=True, use_dist=True) with a
    single CUDA kernel that fuses the 4-edge loop.

    Args:
        lines_gt   : [N, >=4]  GT line segments; first 4 cols are x1,y1,x2,y2.
                               Must be float32 on CUDA.
        corners_2d : [N, 4, 2] 2D corner positions of the matched plane
                               rectangles.  Must be float32 on CUDA.

    Returns:
        theta      : [N]  Angular error (radians) for the selected edge.
        dist       : [N]  Mean perpendicular endpoint distance for the
                           selected edge.
        final_idx  : [N]  Index (0-3) of the selected plane edge.
    """
    lines_gt   = lines_gt.float().contiguous()
    # corners_2d expected as [N, 4, 2]; flatten to [N, 8] for contiguous access
    corners_2d = corners_2d.float().reshape(lines_gt.shape[0], 4, 2).contiguous()

    return _C.line_plane_assign(lines_gt, corners_2d)
