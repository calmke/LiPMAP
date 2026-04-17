#pragma once
#include <torch/extension.h>
#include <tuple>

// Ray-plane intersection + nearest valid plane selection.
// Fuses compute_ray2plane_intersections_v2 + check_intersections_in_plane
// + find_positive_min_mask into a single kernel, eliminating the O(N_ray*N_plane)
// intermediate tensors.
//
// Returns: (inter_pts [N_ray,3], min_sd [N_ray], min_idx [N_ray], valid_mask [N_ray])
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RayPlaneIntersectCUDA(
    const torch::Tensor& rays_o,       // [N_ray, 3]
    const torch::Tensor& rays_d,       // [N_ray, 3]
    const torch::Tensor& plane_center, // [N_plane, 3]
    const torch::Tensor& plane_normal, // [N_plane, 3]
    const torch::Tensor& plane_radii,  // [N_plane, 4]  (x+, y+, x-, y-)
    const torch::Tensor& plane_xAxis,  // [N_plane, 3]
    const torch::Tensor& plane_yAxis   // [N_plane, 3]
);

// Line-plane edge assignment.
// For each matched (line_gt, plane_corners_2d) pair, finds the plane edge
// (among 4 rectangular edges) that best matches the GT line segment by:
//   1. Computing angular error (theta) to all 4 edges
//   2. Taking the top-2 smallest-angle edges
//   3. Among those two, selecting the one with smaller perpendicular endpoint distance
//
// Matches the behaviour of find_min_theta_dist_line(use_theta=True, use_dist=True).
//
// Returns: (theta [N], dist [N], final_edge_idx [N])
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
LinePlaneAssignCUDA(
    const torch::Tensor& lines_gt,    // [N, >=4]  (x1,y1,x2,y2,...)
    const torch::Tensor& corners_2d   // [N, 4, 2]
);
