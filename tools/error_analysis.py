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
from core.test_engine import run_inference, get_inference_dataset
import utils.logging
from datasets.json_dataset import JsonDataset
import pickle
import ipdb

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
        '--vcoco_use_spatial', help='vcoco keypoint multi task',
        action='store_true')
    parser.add_argument(
        '--vcoco_use_union_feat', help='vcoco keypoint multi task',
        action='store_true')
    parser.add_argument(
        '--use_precomp_box', help='use boxes get from detection directly',
        action='store_true')
    parser.add_argument(
        '--use_origin', help='Use original InteractNet as paper',
        action='store_true')
    parser.add_argument(
        '--late_fusion', help='where to late fusion',
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

    # assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
    #     'Exactly one of --load_ckpt and --load_detectron should be specified.'
    # if args.output_dir is None:
    #     ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
    #     args.output_dir = os.path.join(
    #         os.path.dirname(os.path.dirname(ckpt_path)), 'test')
    #     logger.info('Automatically set output directory to %s', args.output_dir)
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "coco2014":
        cfg.TEST.DATASETS = ('coco_2014_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == 'vcoco_test':
        cfg.TEST.DATASETS = ('vcoco_test',)
        cfg.MODEL.NUM_CLASSES = 81
        logger.info('Test InteractNet...')
    elif args.dataset == 'vcoco_val':
        cfg.TEST.DATASETS = ('vcoco_val',)
        cfg.MODEL.NUM_CLASSES = 81
        logger.info('Test InteractNet...')
    elif args.dataset == 'keypoints_coco2014':
        cfg.TEST.DATASETS = ('keypoints_coco_2014_val',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'

    if args.vcoco_kp_on:
        cfg.VCOCO.KEYPOINTS_ON = True

    if args.vcoco_use_spatial:
        cfg.VCOCO.USE_SPATIAL = True

    if args.vcoco_use_union_feat:
        cfg.VCOCO.USE_UNION_FEAT = True

    if args.use_precomp_box:
        cfg.VCOCO.USE_PRECOMP_BOX = True

    if args.use_origin:
        cfg.VCOCO.USE_ORIGIN = True

    if args.late_fusion:
        cfg.VCOCO.LATE_FUSION = True

    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = False

    dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
    dataset = JsonDataset(dataset_name)
    file_name = (args.load_ckpt).split('/')[:-2]
    det_name = file_name + ['test', 'hoi_vcoco_test_results.pkl']
    det_name = "/".join(det_name)
    ''' for error analysis '''
    save_file = file_name + ['test', 'hoi_vcoco_triplet_results_ican.pkl']
    # for replace different error
    # save_file = file_name + ['test', 'hoi_vcoco_test_results_no_misgrouping.pkl']
    # save_file = file_name + ['test', 'hoi_vcoco_test_results_no_actmiscls.pkl']
    # save_file = file_name + ['test', 'hoi_vcoco_test_results_no_objmisloc.pkl']
    # save_file = file_name + ['test', 'hoi_vcoco_test_results_no_34error.pkl']
    save_file = "/".join(save_file)
    # det_name = './ican/300000_iCAN_ResNet50_VCOCO_Early.pkl'
    # save_file = './ican/hoi_vcoco_triplet_results_ican.pkl'
    all_hoi = pickle.load(open(det_name, 'rb') , encoding='latin1')
    vcocodb = dataset.get_roidb(gt=True)
    dataset.VCOCO._save_error(vcocodb, all_hoi, ovr_thresh=0.5, eval_type='scenario_1', save_file=save_file)
