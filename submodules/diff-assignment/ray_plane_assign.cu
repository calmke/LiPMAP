#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include "ray_plane_assign.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ---------------------------------------------------------------------------
// Kernel 1: Ray-plane intersection + nearest valid plane selection
//
// One thread per ray. Inner loop over all planes avoids storing an
// [N_ray, N_plane, 3] intermediate tensor.
//
// The intersection logic mirrors model_util.compute_ray2plane_intersections_v2
// and the bounds check mirrors model_util.check_intersections_in_plane exactly
// (including the sign convention: vec_c2i = plane_center - inter_point).
// ---------------------------------------------------------------------------
__global__ void ray_plane_intersect_kernel(
    const float* __restrict__ rays_o,        // [N_ray, 3]
    const float* __restrict__ rays_d,        // [N_ray, 3]
    const float* __restrict__ plane_center,  // [N_plane, 3]
    const float* __restrict__ plane_normal,  // [N_plane, 3]
    const float* __restrict__ plane_radii,   // [N_plane, 4]  (rx+, ry+, rx-, ry-)
    const float* __restrict__ plane_xAxis,   // [N_plane, 3]
    const float* __restrict__ plane_yAxis,   // [N_plane, 3]
    float*       __restrict__ inter_pts,     // [N_ray, 3]   output
    float*       __restrict__ min_sd_out,    // [N_ray]      output
    int64_t*     __restrict__ min_idx_out,   // [N_ray]      output  (-1 = invalid)
    bool*        __restrict__ valid_mask,    // [N_ray]      output
    int N_ray,
    int N_plane
)
{
    int ray_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_i >= N_ray) return;

    const float ox = rays_o[ray_i * 3 + 0];
    const float oy = rays_o[ray_i * 3 + 1];
    const float oz = rays_o[ray_i * 3 + 2];
    const float dx = rays_d[ray_i * 3 + 0];
    const float dy = rays_d[ray_i * 3 + 1];
    const float dz = rays_d[ray_i * 3 + 2];

    float best_t  = 1e30f;
    int   best_p  = -1;
    float best_ix = 0.f, best_iy = 0.f, best_iz = 0.f;

    for (int p = 0; p < N_plane; p++) {
        const float cx = plane_center[p * 3 + 0];
        const float cy = plane_center[p * 3 + 1];
        const float cz = plane_center[p * 3 + 2];
        const float nx = plane_normal[p * 3 + 0];
        const float ny = plane_normal[p * 3 + 1];
        const float nz = plane_normal[p * 3 + 2];

        // vec_o2c = plane_center - ray_origin
        const float o2cx = cx - ox;
        const float o2cy = cy - oy;
        const float o2cz = cz - oz;

        // signed_dist_o2p = -dot(normal, vec_o2c)
        // o2inter_SD = -signed_dist_o2p / dot(ray_d, normal)
        //            = dot(normal, vec_o2c) / dot(ray_d, normal)
        const float signed_dist_o2p = -(nx * o2cx + ny * o2cy + nz * o2cz);
        const float denom = dx * nx + dy * ny + dz * nz;
        const float t = -signed_dist_o2p / (denom + 1e-10f);

        // Must be in front of camera (positive direction)
        if (t <= 0.f) continue;
        // Discard if farther than current best
        if (t >= best_t) continue;

        // Intersection point
        const float ix = ox + dx * t;
        const float iy = oy + dy * t;
        const float iz = oz + dz * t;

        // vec_c2i = plane_center - inter  (same sign convention as check_intersections_in_plane)
        const float c2ix = cx - ix;
        const float c2iy = cy - iy;
        const float c2iz = cz - iz;

        // Project onto plane local axes
        const float xx = plane_xAxis[p * 3 + 0];
        const float xy = plane_xAxis[p * 3 + 1];
        const float xz = plane_xAxis[p * 3 + 2];
        const float yx = plane_yAxis[p * 3 + 0];
        const float yy = plane_yAxis[p * 3 + 1];
        const float yz = plane_yAxis[p * 3 + 2];

        const float proj_x = c2ix * xx + c2iy * xy + c2iz * xz;
        const float proj_y = c2ix * yx + c2iy * yy + c2iz * yz;

        // Bounds: radii layout is [rx+, ry+, rx-, ry-]
        const float rx_p = plane_radii[p * 4 + 0];
        const float ry_p = plane_radii[p * 4 + 1];
        const float rx_n = plane_radii[p * 4 + 2];
        const float ry_n = plane_radii[p * 4 + 3];

        if (proj_x < -rx_n || proj_x > rx_p) continue;
        if (proj_y < -ry_n || proj_y > ry_p) continue;

        best_t  = t;
        best_p  = p;
        best_ix = ix;
        best_iy = iy;
        best_iz = iz;
    }

    const bool valid = (best_p >= 0);
    valid_mask[ray_i]        = valid;
    min_idx_out[ray_i]       = (int64_t)best_p;
    min_sd_out[ray_i]        = valid ? best_t : 0.f;
    inter_pts[ray_i * 3 + 0] = best_ix;
    inter_pts[ray_i * 3 + 1] = best_iy;
    inter_pts[ray_i * 3 + 2] = best_iz;
}

// ---------------------------------------------------------------------------
// Kernel 2: Line-plane edge assignment
//
// One thread per matched (GT line, plane corners 2D) pair.
// Computes theta and dist for all 4 rectangular edges, selects the best match
// using top-2 theta + min dist tiebreak — identical to
// find_min_theta_dist_line(use_theta=True, use_dist=True).
//
// corners_2d layout: [N, 4, 2] stored as flat float* in row-major order.
//   corners_2d[i, e, d] = corners_2d_ptr[i*8 + e*2 + d]
// lines_gt layout:   [N, L] stored row-major, first 4 cols = x1,y1,x2,y2.
//   lines_gt[i, k]   = lines_gt_ptr[i*lines_stride + k]
// ---------------------------------------------------------------------------
__global__ void line_plane_assign_kernel(
    const float*   __restrict__ lines_gt,      // [N, lines_stride]
    const float*   __restrict__ corners_2d,    // [N, 4, 2]
    float*         __restrict__ theta_out,     // [N]
    float*         __restrict__ dist_out,      // [N]
    int64_t*       __restrict__ final_idx_out, // [N]
    int N,
    int lines_stride
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // GT line endpoints
    const float lx1 = lines_gt[i * lines_stride + 0];
    const float ly1 = lines_gt[i * lines_stride + 1];
    const float lx2 = lines_gt[i * lines_stride + 2];
    const float ly2 = lines_gt[i * lines_stride + 3];

    // GT line direction vector (for angle calculation: p2 - p1)
    const float lvx_ang = lx2 - lx1;
    const float lvy_ang = ly2 - ly1;
    const float lv_ang_len = sqrtf(lvx_ang * lvx_ang + lvy_ang * lvy_ang) + 1e-8f;

    // GT line direction vector (for distance calculation: p1 - p2, matching
    // calculate_lines_2d_dist which uses lines_gt[:2] - lines_gt[2:])
    const float lvx_dist = lx1 - lx2;
    const float lvy_dist = ly1 - ly2;
    const float lv_dist_len = sqrtf(lvx_dist * lvx_dist + lvy_dist * lvy_dist) + 1e-8f;

    float theta_vals[4];
    float dist_vals[4];

    for (int e = 0; e < 4; e++) {
        const int e_next = (e + 1) & 3;  // % 4

        const float px1 = corners_2d[i * 8 + e      * 2 + 0];
        const float py1 = corners_2d[i * 8 + e      * 2 + 1];
        const float px2 = corners_2d[i * 8 + e_next * 2 + 0];
        const float py2 = corners_2d[i * 8 + e_next * 2 + 1];

        // Edge direction vector (p2 - p1)
        const float evx = px2 - px1;
        const float evy = py2 - py1;
        const float ev_len = sqrtf(evx * evx + evy * evy) + 1e-8f;

        // Angle: cos(theta) = |dot(line_vec, edge_vec)| / (|line_vec| * |edge_vec|)
        const float cos_t = fabsf((lvx_ang * evx + lvy_ang * evy) / (lv_ang_len * ev_len));
        const float cos_clamped = fminf(1.f, fmaxf(0.f, cos_t));
        theta_vals[e] = acosf(cos_clamped);

        // Perpendicular distance of plane edge endpoints from GT line.
        // Mirrors calculate_lines_2d_dist with dtype='l':
        //   line_vector = lines_gt[:2] - lines_gt[2:]  (= lvx_dist, lvy_dist)
        //   edp1_vector = plane_p1 - lines_gt[:2]
        //   edp2_vector = plane_p2 - lines_gt[2:]
        //   d1 = |cross(edp1, line_vector)| / |line_vector|
        //   d2 = |cross(edp2, line_vector)| / |line_vector|
        const float edp1x = px1 - lx1;
        const float edp1y = py1 - ly1;
        const float d1 = fabsf(edp1x * lvy_dist - edp1y * lvx_dist) / lv_dist_len;

        const float edp2x = px2 - lx2;
        const float edp2y = py2 - ly2;
        const float d2 = fabsf(edp2x * lvy_dist - edp2y * lvx_dist) / lv_dist_len;

        dist_vals[e] = (d1 + d2) * 0.5f;
    }

    // Find top-2 indices with smallest theta (ascending)
    int idx0 = 0, idx1 = 1;
    if (theta_vals[idx0] > theta_vals[idx1]) {
        int tmp = idx0; idx0 = idx1; idx1 = tmp;
    }
    for (int e = 2; e < 4; e++) {
        if (theta_vals[e] < theta_vals[idx0]) {
            idx1 = idx0;
            idx0 = e;
        } else if (theta_vals[e] < theta_vals[idx1]) {
            idx1 = e;
        }
    }

    // Among top-2, choose the one with smaller mean perpendicular distance
    const int final_e = (dist_vals[idx0] <= dist_vals[idx1]) ? idx0 : idx1;

    theta_out[i]     = theta_vals[final_e];
    dist_out[i]      = dist_vals[final_e];
    final_idx_out[i] = (int64_t)final_e;
}

// ---------------------------------------------------------------------------
// C++ host wrappers
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RayPlaneIntersectCUDA(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    const torch::Tensor& plane_center,
    const torch::Tensor& plane_normal,
    const torch::Tensor& plane_radii,
    const torch::Tensor& plane_xAxis,
    const torch::Tensor& plane_yAxis)
{
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(plane_center);
    CHECK_INPUT(plane_normal);
    CHECK_INPUT(plane_radii);
    CHECK_INPUT(plane_xAxis);
    CHECK_INPUT(plane_yAxis);

    const int N_ray   = rays_o.size(0);
    const int N_plane = plane_center.size(0);

    auto opts_f = rays_o.options().dtype(torch::kFloat32);
    auto opts_i = rays_o.options().dtype(torch::kInt64);
    auto opts_b = rays_o.options().dtype(torch::kBool);

    torch::Tensor inter_pts  = torch::zeros({N_ray, 3}, opts_f);
    torch::Tensor min_sd     = torch::zeros({N_ray},    opts_f);
    torch::Tensor min_idx    = torch::full ({N_ray}, -1, opts_i);
    torch::Tensor valid_mask = torch::zeros({N_ray},    opts_b);

    if (N_ray > 0 && N_plane > 0) {
        const int threads = 256;
        const int blocks  = (N_ray + threads - 1) / threads;

        ray_plane_intersect_kernel<<<blocks, threads>>>(
            rays_o.contiguous().data_ptr<float>(),
            rays_d.contiguous().data_ptr<float>(),
            plane_center.contiguous().data_ptr<float>(),
            plane_normal.contiguous().data_ptr<float>(),
            plane_radii.contiguous().data_ptr<float>(),
            plane_xAxis.contiguous().data_ptr<float>(),
            plane_yAxis.contiguous().data_ptr<float>(),
            inter_pts.data_ptr<float>(),
            min_sd.data_ptr<float>(),
            min_idx.data_ptr<int64_t>(),
            valid_mask.data_ptr<bool>(),
            N_ray,
            N_plane
        );
    }

    return std::make_tuple(inter_pts, min_sd, min_idx, valid_mask);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
LinePlaneAssignCUDA(
    const torch::Tensor& lines_gt,
    const torch::Tensor& corners_2d)
{
    CHECK_INPUT(lines_gt);
    CHECK_INPUT(corners_2d);

    const int N            = lines_gt.size(0);
    const int lines_stride = lines_gt.size(1);

    auto opts_f = lines_gt.options().dtype(torch::kFloat32);
    auto opts_i = lines_gt.options().dtype(torch::kInt64);

    torch::Tensor theta     = torch::zeros({N}, opts_f);
    torch::Tensor dist      = torch::zeros({N}, opts_f);
    torch::Tensor final_idx = torch::zeros({N}, opts_i);

    if (N > 0) {
        const int threads = 256;
        const int blocks  = (N + threads - 1) / threads;

        line_plane_assign_kernel<<<blocks, threads>>>(
            lines_gt.contiguous().data_ptr<float>(),
            corners_2d.contiguous().data_ptr<float>(),
            theta.data_ptr<float>(),
            dist.data_ptr<float>(),
            final_idx.data_ptr<int64_t>(),
            N,
            lines_stride
        );
    }

    return std::make_tuple(theta, dist, final_idx);
}
