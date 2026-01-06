import sys
sys.path.append('.')
import argparse
import os
from utils.misc_util import fix_seeds
from utils.conf_util import get_class
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
import numpy as np

fix_seeds(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str, default='temp', help='path to config file')
    parser.add_argument('--conf', type=str, default='confs/scannetv2_debug.conf', help='path to config file')
    parser.add_argument('--base_conf', type=str, default='confs/base_conf_planarSplatCuda.conf', help='path to base config file')
    parser.add_argument('--run_task', type=str, default='train', help='run task: train, eval')
    parser.add_argument("--exps_folder_name", type=str, default="exps_result", help='folder to save model results')
    parser.add_argument('--is_continue', default=False, action="store_true", help='if set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--resume_path', default=None, type=str, help='The path of resumed checkpoint.')
    parser.add_argument('--resume_epoch', default=None, type=int, help='The epoch of resumed checkpoint.')
    parser.add_argument('--cancel_vis', default=False, action="store_true", help='cancel visualization durning training')
    parser.add_argument('--scan_id', type=str, default='-1', help='If set, taken to be the scan id.')
    parser.add_argument('--eval_method', type=str, default='planarSplat', help='method used to get mesh')

    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--glj_iter', type=int, default=0)
    args = parser.parse_args()

    # load base conf at first
    cfg = ConfigFactory.parse_file(args.conf)

    # process scan_id if needed
    cfg.put('dataset.scan_id', args.scan_id)
    scan_id = cfg.get_string('dataset.scan_id', default='-1')
    if scan_id == '':
        assert args.scan_id != '', "scan_id should be given!"
    
    data_root = cfg.get_string('dataset.data_root', default='../data')

    # update conf about scene bounding sphere
    scene_bounding_sphere = cfg.get_float('train.scene_bounding_sphere')
    sphere_ratio = cfg.get_float('train.sphere_ratio')
    # add to plot conf
    cfg.put('plot.grid_boundary', [-scene_bounding_sphere, scene_bounding_sphere])
    cfg.put('sdf_model.scene_bounding_sphere', scene_bounding_sphere)
    # cfg.put('sdf_model.scene_scale', scene_scale.item())
    cfg.put('sdf_model.implicit_network.divide_factor', scene_bounding_sphere)
    cfg.put('dataset.scene_bounding_sphere', scene_bounding_sphere)
    cfg.put('plane_model.sphere_ratio', sphere_ratio)
    cfg.put('train.expname', args.expname)
    cfg.put('train.glj_iter', args.glj_iter)

    if args.resume_path is not None:
        resume_path = args.resume_path
    elif args.resume_epoch is not None:
        resume_path = args.resume_epoch
    else:
        resume_path = None

    runner = get_class(cfg.get_string('train.train_runner_class'))(
                                conf=cfg,
                                batch_size=1,
                                exps_folder_name=args.exps_folder_name,
                                is_continue=args.is_continue,
                                timestamp=args.timestamp,
                                do_vis=not args.cancel_vis,
                                scan_id=args.scan_id,
                                debug=args.debug,
                                resume_path=args.resume_path,
                                )
    
    runner.run()
