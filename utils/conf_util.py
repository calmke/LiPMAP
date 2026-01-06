import os
from datetime import datetime

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_train_param(opt, conf):
    scan_id = opt['scan_id'] if opt['scan_id'] != '-1' else conf.get_string('dataset.scan_id', default='-1')
    expname = conf.get_string('train.expname')
    # if scan_id != '-1':
    #     expname = expname + '_{0}'.format(scan_id)
    if opt['is_continue']:
        if opt['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../', opt['exps_folder_name'], expname)):
                timestamps = os.listdir(os.path.join('../', opt['exps_folder_name'], expname))
                if (len(timestamps)) == 0:
                    raise ValueError('There are no experiments in the target folder!')
                else:
                    timestamp = sorted(timestamps)[-1]  # use latest timestamp
                    is_continue = True
            else:
                raise ValueError('Target folder does not exist!')
        else:
            if os.path.exists(os.path.join('../', opt['exps_folder_name'], expname, opt['timestamp'])):
                is_continue = True
                timestamp = opt['timestamp']
            else:
                raise ValueError('Target folder does not exist!')
    else:
        is_continue = False
        timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    
    return expname, scan_id, timestamp, is_continue