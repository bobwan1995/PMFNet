"""
Test InteractNet.
Perform inference on one or more datasets.
"""

import argparse
import cv2
import os
import pprint
import sys
import time

import torch
import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
# from core.test_engine import run_inference
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--freeze_at',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        default=0, type=int)

    parser.add_argument(
        '--heatmap_kernel_size', help='default gaussian size of heatmap',
        default=7, type=int)

    parser.add_argument(
        '--part_crop_size', help='default part crop size of union feature map',
        default=7, type=int)
    
    parser.add_argument(
        '--use_kps17', help='use 17 keypoints for PartAlign',
        action='store_true')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')
    parser.add_argument(
        '--vcoco_kp_on', help='vcoco keypoint multi task',
        action='store_true')
    parser.add_argument(
        '--net_name', default='PMFNet_Final', help='which network')
    parser.add_argument(
        '--mlp_head_dim', help='hidden feature dimension',
        default=512, type=int)
    parser.add_argument(
        '--vcoco_use_spatial', help='vcoco keypoint multi task',
        action='store_true')
    parser.add_argument(
        '--vcoco_use_union_feat', help='vcoco keypoint multi task',
        action='store_true')
    parser.add_argument(
        '--use_precomp_box', help='use boxes get from detection directly',
        action='store_true')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    args.multi_gpu_testing = (torch.cuda.device_count() > 1)
    # assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "coco2014":
        cfg.TEST.DATASETS = ('coco_2014_val',),
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == 'vcoco_test':
        cfg.TEST.DATASETS = ('vcoco_test',)
        cfg.MODEL.NUM_CLASSES = 81
        logger.info('Val PMFNet...')
    elif args.dataset == 'vcoco_val':
        cfg.TEST.DATASETS = ('vcoco_val',)
        cfg.MODEL.NUM_CLASSES = 81
        logger.info('Test PMFNet...')
    elif args.dataset == 'vcoco_trainval':
        cfg.TEST.DATASETS = ('vcoco_trainval',)
        cfg.MODEL.NUM_CLASSES = 81
        logger.info('Collect detection result on VCOCO train set')
    elif args.dataset == 'vcoco_all':
        cfg.TEST.DATASETS = ('vcoco_train','vcoco_val') #, 'vcoco_test')
        cfg.MODEL.NUM_CLASSES = 81
        logger.info('Test on vcoco_trainval to check if overfitting')
    elif args.dataset == 'keypoints_coco2014':
        cfg.TEST.DATASETS = ('keypoints_coco_2014_val',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    
    if args.vcoco_use_spatial:
        cfg.VCOCO.USE_SPATIAL = True

    if args.vcoco_use_union_feat:
        cfg.VCOCO.USE_UNION_FEAT = True

    if args.use_precomp_box:
        cfg.VCOCO.USE_PRECOMP_BOX = True

    cfg.NETWORK_NAME=args.net_name # network name
    print('Network name:', args.net_name)

    cfg.VCOCO.MLP_HEAD_DIM = args.mlp_head_dim
    print('MLP head dim: ', args.mlp_head_dim)

    cfg.RESNETS.FREEZE_AT=args.freeze_at
    print('Freeze at: ', args.freeze_at)

    cfg.VCOCO.HEATMAP_KERNEL_SIZE=args.heatmap_kernel_size
    print('heatmap_kernel_size: ', cfg.VCOCO.HEATMAP_KERNEL_SIZE)

    cfg.VCOCO.PART_CROP_SIZE=args.part_crop_size
    print('part_crop_size: ', cfg.VCOCO.PART_CROP_SIZE)
        
    print('use use_kps17 for part Align: ', args.use_kps17)
    if args.use_kps17:
        cfg.VCOCO.USE_KPS17 = True
    else:
        cfg.VCOCO.USE_KPS17 = False

    '''****** test with gt action ******'''
    cfg.DEBUG_TEST_WITH_GT = True
    # cfg.DEBUG_TEST_GT_ACTION = True

    assert_and_infer_cfg()

    logger.info('Testing with config:')
    # logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    #print('args.output_dir:', args.output_dir)
    from core.test_engine import run_inference
    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True)
