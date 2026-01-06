import torch
import torch.nn.functional as F
from utils import model_util
from utils import mesh_util

def metric_depth_loss(depth_pred, depth_gt, mask, max_depth=10.0, weight=None):
    depth_mask = torch.logical_and(depth_gt<=max_depth, depth_gt>0)
    depth_mask = torch.logical_and(depth_mask, mask)
    # depth_mask = mask & (0 < depth_gt) & (depth_gt <= max_depth)

    if depth_mask.sum() == 0:
        depth_loss = torch.tensor([0.]).mean().cuda()
    else:
        if weight is None:
            depth_loss = torch.mean(torch.abs((depth_pred - depth_gt)[depth_mask]))
        else:
            depth_loss = torch.mean((weight * torch.abs(depth_pred - depth_gt))[depth_mask])
    return depth_loss

def normal_loss(normal_pred, normal_gt, mask):
    # if mask.sum() == 0:
    #     l1 = torch.tensor([0.]).mean().cuda()
    #     cos = torch.tensor([0.]).mean().cuda()
    # else:
    #     normal_pred = F.normalize(normal_pred, dim=-1)
    #     normal_gt = F.normalize(normal_gt, dim=-1)
    #     l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)[mask].mean()
    #     cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1))[mask].mean()

    normal_pred = F.normalize(normal_pred, dim=-1)
    normal_gt = F.normalize(normal_gt, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)[mask].mean()
    cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1))[mask].mean()
    return l1, cos

def cluster_loss(lines2d, lines3d, lines_tgt):
    loss = torch.tensor([0.])
    loss_dict = []
    all_loss = []
    unique_lines_tgt, inverse_indices = torch.unique(lines_tgt, dim=0, return_inverse=True)
    for i in range(unique_lines_tgt.size(0)):
        indices = torch.where(inverse_indices == i)[0]
        _, indices_ = lines3d[indices].unique(dim=0, return_inverse=True)
        diff = indices_[1:] - indices_[:-1]
        diff = torch.cat((torch.tensor([1]).cuda(),diff))
        lines_3d = lines3d[indices][diff!=0]

        center = torch.mean(lines_3d, dim=0, keepdim=True)
        # distances = torch.sum((lines_3d - center)**2, dim=1)
        distances = torch.abs(lines_3d - center)
        # loss += torch.mean(distances)
        loss += distances.sum()
        all_loss.append(distances.sum())
    return loss

def lines_edps_loss_2d(lines2d, lines2d_gt, img_size, dtype='e'):
    if dtype == 'e':    
        H, W = img_size
        edps_input = lines2d.reshape(-1,2)
        edps_gt = lines2d_gt.reshape(-1,2)
        # edps_mask = (edps_input[:,0]>=0)*(edps_input[:,0]<=H-1) * (edps_input[:,1]>=0)*(edps_input[:,1]<=W-1)
        edps_mask = (edps_input[:,0]>=0)*(edps_input[:,0]<=W-1) * (edps_input[:,1]>=0)*(edps_input[:,1]<=H-1)
        juncs_input = edps_input[edps_mask]
        juncs_gt = edps_gt[edps_mask]
        if juncs_input.shape[0] == 0:
            loss_juncs = torch.tensor([0.]).mean().cuda()
        else:
            loss_juncs = torch.mean(torch.sqrt(torch.sum((juncs_input - juncs_gt)**2, dim=-1)))
    elif dtype == 'l':
        def cross(A, B):
            return A[:,0] * B[:,1] - A[:,1] * B[:,0]
        line_vector = lines2d_gt[:, :2] - lines2d_gt[:, 2:]
        edp1_vector = lines2d[:, :2] - lines2d_gt[:, :2]
        edp2_vector = lines2d[:, 2:] - lines2d_gt[:, 2:]
        dist1 = torch.abs(cross(edp1_vector, line_vector)) / torch.linalg.norm(line_vector, dim=-1) 
        dist2 = torch.abs(cross(edp2_vector, line_vector)) / torch.linalg.norm(line_vector, dim=-1) 
        loss_juncs = torch.mean((dist1 + dist2) / 2)
    else:
        raise TypeError("Please use dtype=['e','l'] for the calculation of distance.")
    return loss_juncs

def lines_edps_loss_2d_(lines2d, lines2d_gt, img_size, dtype='e'):
    H, W = img_size
    edps_input = lines2d.reshape(-1,2)
    edps_gt = lines2d_gt.reshape(-1,2)
    edps_clip = torch.cat((edps_input[:,0].clamp(0, W-1).unsqueeze(-1), edps_input[:,1].clamp(0, H-1).unsqueeze(-1)), dim=-1)
    # import pdb;pdb.set_trace()
    # edps_input[:,0] = torch.clamp(edps_input[:,0], 0, W-1)
    # edps_input[:,1] = torch.clamp(edps_input[:,1], 0, H-1)
    # loss_juncs = torch.mean(torch.sqrt(torch.sum((edps_gt - edps_input)**2, dim=-1)))
    loss_juncs = torch.mean(torch.sqrt(torch.sum((edps_gt - edps_clip)**2, dim=-1)))

    return loss_juncs

def lines_edps_loss(lines2d, lines3d, lines2d_gt, img_size, dtype='e'):
    H, W = img_size
    edps_2d = lines2d.reshape(-1,2)
    edps_gt = lines2d_gt.reshape(-1,2)
    edps_mask = (edps_2d[:,0]>=0)*(edps_2d[:,0]<=H-1) * (edps_2d[:,1]>=0)*(edps_2d[:,1]<=W-1)
    juncs_2d = edps_2d[edps_mask]
    juncs_gt = edps_gt[edps_mask]
    loss_juncs_2d = torch.mean(torch.sqrt(torch.sum((juncs_2d - juncs_gt)**2, dim=-1)))

    edps_3d = lines3d.reshape(-1,3)
    juncs_3d = edps_3d[edps_mask]
    loss_juncs_3d = (juncs_3d[1:]-juncs_3d[:-1]).abs().mean(dim=-1).sum()


    return loss_juncs_2d, loss_juncs_3d

def lines_loss_2d(lines2d, lines2d_gt):
    def cross(A, B):
        return A[:,0] * B[:,1] - A[:,1] * B[:,0]
    line_vector = lines2d_gt[:, :2] - lines2d_gt[:, 2:]
    edp1_vector = lines2d[:, :2] - lines2d_gt[:, :2]
    edp2_vector = lines2d[:, 2:] - lines2d_gt[:, 2:]
    dist1 = torch.abs(cross(edp1_vector, line_vector)) / torch.linalg.norm(line_vector, dim=-1) 
    dist2 = torch.abs(cross(edp2_vector, line_vector)) / torch.linalg.norm(line_vector, dim=-1)
    dist = torch.stack([dist1,dist2], dim=1)
    # max dist loss
    loss = torch.max(dist, dim=-1)[0].mean()
    # mean dist loss
    # loss = torch.mean(dist, dim=-1).mean()

    return loss

def lines_loss_3d_org(lines2d, lines2d_gt, lines3d, unique=False):
    loss = torch.tensor(0.).cuda()
    batched_lines = []
    lines2d_gt_uniq, indices = torch.unique(lines2d_gt, dim=0, return_inverse=True)
    for i in range(indices.unique(dim=0).shape[0]):
        batched_lines = lines3d[indices==i]
        ###### remove duplicate lines
        lines_unique = model_util.differentiable_unique(batched_lines)
        # lines_unique, idxs = torch.unique(batched_lines, dim=0, return_inverse=True)
        if lines_unique.shape[0] >1:
            # idxs_s = torch.roll(idxs, 1)
            # mask = (idxs-idxs_s)!=0
            # lines_unique = batched_lines[mask]

            lines_shifted = torch.roll(lines_unique, 1, 0)
            lines_vector = lines_unique[:, :3] - lines_unique[:, 3:]
            vector1 = lines_shifted[:, :3] - lines_unique[:, 3:]
            vector2 = lines_shifted[:, 3:] - lines_unique[:, 3:]
            cross1 = torch.cross(lines_vector, vector1, dim=-1)
            cross2 = torch.cross(lines_vector, vector2, dim=-1)
            norm_lines_vector = torch.norm(lines_vector, dim=-1)
            dist1 = torch.norm(cross1, dim=-1) / norm_lines_vector
            dist2 = torch.norm(cross2, dim=-1) / norm_lines_vector
            loss += ((dist1+dist2)/2).sum()
    return loss

def lines_loss_3d(lines2d, lines2d_gt, lines3d, unique=False):
    import time
    loss = torch.tensor(0.).cuda()
    lines2d_gt_uniq, indices = torch.unique(lines2d_gt, dim=0, return_inverse=True)

    num_indices = indices.max().item() + 1

    batched_lines = lines3d[None, :, :].expand(num_indices, -1, -1)
    mask = torch.arange(num_indices).cuda().unsqueeze(-1)==indices.unsqueeze(0)
    grouped_lines3d = batched_lines[mask]
    
    counts = mask.sum(dim=1)
    counts = torch.cat((torch.tensor([0]).cuda(), counts))
    group_shifted = [(torch.arange(counts[:i+1].sum(),counts[:i+1+1].sum())+1).cuda()%counts[:i+1+1].sum() for i in range(counts.shape[0]-1)]
    mask_shifted = torch.cat(group_shifted, dim=0)
    grouped_lines3d_shifted = grouped_lines3d[mask_shifted]

    lines_vector = grouped_lines3d[:, :3] - grouped_lines3d[:, 3:]
    vector1 = grouped_lines3d_shifted[:, :3] - grouped_lines3d[:, 3:]
    vector2 = grouped_lines3d_shifted[:, 3:] - grouped_lines3d[:, 3:]
    cross1 = torch.cross(lines_vector, vector1, dim=-1)
    cross2 = torch.cross(lines_vector, vector2, dim=-1)
    norm_lines_vector = torch.norm(lines_vector, dim=-1)
    dist1 = torch.norm(cross1, dim=-1) / norm_lines_vector
    dist2 = torch.norm(cross2, dim=-1) / norm_lines_vector
    loss += ((dist1+dist2)/2).sum()

    return loss


def lines_theta_loss_2d(lines2d, lines2d_gt):
    vector_l = lines2d[:, :2] - lines2d[:, 2:]
    vector_gt = lines2d_gt[:, :2] - lines2d_gt[:, 2:]
    dot_product = torch.sum(vector_l * vector_gt, dim=-1).abs()
    norm_l = torch.norm(vector_l, dim=-1)
    norm_gt = torch.norm(vector_gt, dim=-1)
    cos_theta = dot_product / (norm_l * norm_gt + 1e-15)
    theta = torch.acos(cos_theta.clamp(0,1))
    loss = theta.mean()

    return loss