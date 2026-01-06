import os
from glob import glob
import torch
from datetime import datetime

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        # os.mkdir(directory)
        os.makedirs(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels, n_pixels=10000, keys = ['uv', 'uv_proj']):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        for key in keys:
            data[key] = torch.index_select(model_input[key], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)


def get_train_param(opt, conf):
    expname = conf.get_string('train.expname')
    scan_id = opt['scan_id'] if opt['scan_id'] != '-1' else conf.get_string('dataset.scan_id', default='-1')
    if scan_id != '-1':
        expname = expname + '_{0}'.format(scan_id)
    if opt['is_continue'] and opt['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', opt['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', opt['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                is_continue = False
                timestamp = None
            else:
                timestamp = sorted(timestamps)[-1]  # use latest timestamp
                is_continue = True
        else:
            is_continue = False
            timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    else:
        is_continue = False
        timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    
    return expname, scan_id, timestamp, is_continue


def clip_2d_lines_image_region(image_size, lines, image):
    [h, w] = image_size
    # new_lines = torch.unique(lines, dim=0)
    new_lines = lines.clone()
    left = 0
    right = w - 1
    top = 0
    bottom = h - 1

    clipped_lines = []
    N = new_lines.shape[0]
    # Iterate over each line
    for i in tqdm.tqdm(range(N)):
        x1, y1, x2, y2 = new_lines[i].cpu().numpy()

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
