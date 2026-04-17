#include <torch/extension.h>
#include "ray_plane_assign.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ray_plane_intersect", &RayPlaneIntersectCUDA,
          "Fused ray-plane intersection + nearest valid plane selection (CUDA). "
          "Returns (inter_pts [N_ray,3], min_sd [N_ray], min_idx [N_ray], valid_mask [N_ray]).");

    m.def("line_plane_assign", &LinePlaneAssignCUDA,
          "Line-plane edge assignment using top-2 theta + min dist (CUDA). "
          "Returns (theta [N], dist [N], final_edge_idx [N]).");
}
