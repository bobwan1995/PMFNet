""" Training script for steps_with_decay policy"""


import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test_engine import test_net, evaluate_affinity
from datasets.json_dataset import JsonDataset
from datasets import task_evaluation
from datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
# from modeling.model_builder import Generalized_RCNN
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats import TrainingStats
import ipdb
import subprocess
import json
import os.path as osp
import random

def random_init(seed=0):
    """ Set the seed for random sampling of pytorch related random packages
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic=True

random_init(0) ## set seed to make results reproduceable 

# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='Dataset to use')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=20, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)
    parser.add_argument(
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)
    parser.add_argument(
        '--fg_thresh',
        help='foreground threshold for training sampling',
        default=0.5, type=float)
    parser.add_argument(
        '--freeze_at',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        default=5, type=int)
    parser.add_argument(
        '--max_iter', help='default maximum iteration',
        default=8000, type=int)
    parser.add_argument(
        '--snapshot', help='default snapshot interval',
        default=1000, type=int)
    parser.add_argument(
        '--triplets_num_per_im', help='default snapshot interval',
        default=16, type=int)
    parser.add_argument(
        '--heatmap_kernel_size', help='default gaussian size of heatmap',
        default=7, type=int)
    parser.add_argument(
        '--part_crop_size', help='default part crop size of union feature map',
        default=7, type=int)
    parser.add_argument(
        '--use_heatmap2', help='whether use heatmap 2 ',
        action='store_true')
    parser.add_argument(
        '--solver_steps', help='default solver lr decay steps',
        nargs='+', type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)

    # Epoch
    parser.add_argument(
        '--start_step',
        help='Starting step count for training epoch. 0-indexed.',
        default=0, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')
    parser.add_argument(
        '--krcnn_from_faster', help='load krcnn weights from faster-rcnn model',
        action='store_true')
    parser.add_argument(
        '--mlp_head_dim',
        help='hidden feature dimension',
        default=256, type=int)

    parser.add_argument(
        '--vcoco_kp_on', help='vcoco keypoint multi task',
        action='store_true')
    parser.add_argument(
        '--use_kps17', help='use 17 keypoints for PartAlign',
        action='store_true')
    parser.add_argument(
        '--conv_body', default='FPN.fpn_ResNet50_conv5_body', help='which backbone network')
    parser.add_argument(
        '--net_name', default='InteractNetUnion', help='which network')
    parser.add_argument(
        '--expID', default='default', help='experiment id')
    parser.add_argument(
        '--expDir', default='exp', help='experiment dir')
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
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()


def save_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


def main():
    """Main function"""

    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if args.dataset == "coco2017":
        cfg.TRAIN.DATASETS = ('coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "coco2014":
        cfg.TRAIN.DATASETS = ('coco_2014_train',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == 'vcoco_trainval':
        cfg.TRAIN.DATASETS = ('vcoco_trainval',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == 'vcoco_train':
        cfg.TRAIN.DATASETS = ('vcoco_train',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == 'vcoco_val':
        cfg.TRAIN.DATASETS = ('vcoco_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == 'keypoints_coco2014':
        cfg.TRAIN.DATASETS = ('keypoints_coco_2014_train',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "keypoints_coco2017":
        cfg.TRAIN.DATASETS = ('keypoints_coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.vcoco_kp_on:
        cfg.VCOCO.KEYPOINTS_ON = True

    cfg.NETWORK_NAME=args.net_name # network name
    print('Network name:', args.net_name)

    cfg.MODEL.CONV_BODY=args.conv_body # backbone network name
    print('Conv_body name:', args.conv_body)

    cfg.TRAIN.FG_THRESH=args.fg_thresh
    print('Train fg thresh:', args.fg_thresh)

    cfg.RESNETS.FREEZE_AT=args.freeze_at
    print('Freeze at: ', args.freeze_at)

    cfg.VCOCO.MLP_HEAD_DIM=args.mlp_head_dim
    print('MLP head dim: ', args.mlp_head_dim)

    cfg.SOLVER.MAX_ITER=args.max_iter
    print('MAX iter: ', args.max_iter)

    cfg.TRAIN.SNAPSHOT_ITERS=args.snapshot
    print('Snapshot Iters: ', args.snapshot)

    if args.solver_steps is not None:
        cfg.SOLVER.STEPS=args.solver_steps
    print('Solver_steps: ', cfg.SOLVER.STEPS)

    cfg.VCOCO.TRIPLETS_NUM_PER_IM=args.triplets_num_per_im
    print('triplets_num_per_im: ', cfg.VCOCO.TRIPLETS_NUM_PER_IM)

    cfg.VCOCO.HEATMAP_KERNEL_SIZE=args.heatmap_kernel_size
    print('heatmap_kernel_size: ', cfg.VCOCO.HEATMAP_KERNEL_SIZE)

    cfg.VCOCO.PART_CROP_SIZE=args.part_crop_size
    print('part_crop_size: ', cfg.VCOCO.PART_CROP_SIZE)

    print('use use_kps17 for part Align: ', args.use_kps17)
    if args.use_kps17:
        cfg.VCOCO.USE_KPS17 = True
    else:
        cfg.VCOCO.USE_KPS17 = False

    print('MULTILEVEL_ROIS: ', cfg.FPN.MULTILEVEL_ROIS)

    if args.vcoco_use_spatial:
        cfg.VCOCO.USE_SPATIAL = True

    if args.vcoco_use_union_feat:
        cfg.VCOCO.USE_UNION_FEAT = True

    if args.use_precomp_box:
        cfg.VCOCO.USE_PRECOMP_BOX = True

    cfg.DEBUG_TEST_WITH_GT = True

    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    ### Adaptively adjust some configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH # 16
    original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
    original_num_gpus = cfg.NUM_GPUS
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    effective_batch_size = args.iter_size * args.batch_size
    print('effective_batch_size = batch_size * iter_size = %d * %d' % (args.batch_size, args.iter_size))

    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' % (original_batch_size, effective_batch_size))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))
    print('    FG_THRESH: ', cfg.TRAIN.FG_THRESH)
    ### Adjust learning based on batch size change linearly
    # For iter_size > 1, gradients are `accumulated`, so lr is scaled based
    # on batch_size instead of effective_batch_size
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr, cfg.SOLVER.BASE_LR))

    ### Adjust solver steps
    step_scale = original_batch_size / effective_batch_size
    old_solver_steps = cfg.SOLVER.STEPS
    old_max_iter = cfg.SOLVER.MAX_ITER
    cfg.SOLVER.STEPS = list(map(lambda x: int(x * step_scale + 0.5), cfg.SOLVER.STEPS))
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * step_scale + 0.5)
    cfg.SOLVER.VAL_ITER = int(cfg.SOLVER.VAL_ITER * step_scale + 0.5) 
    cfg.TRAIN.SNAPSHOT_ITERS = int(cfg.TRAIN.SNAPSHOT_ITERS * step_scale + 0.5)
    print('Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
          '    SOLVER.STEPS: {} --> {}\n'
          '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps, cfg.SOLVER.STEPS,
                                                  old_max_iter, cfg.SOLVER.MAX_ITER))

    # Scale FPN rpn_proposals collect size (post_nms_topN) in `collect` function
    # of `collect_and_distribute_fpn_rpn_proposals.py`
    #
    # post_nms_topN = int(cfg[cfg_key].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5)
    if cfg.FPN.FPN_ON and cfg.MODEL.FASTER_RCNN:
        cfg.FPN.RPN_COLLECT_SCALE = cfg.TRAIN.IMS_PER_BATCH / original_ims_per_batch
        print('Scale FPN rpn_proposals collect size directly propotional to the change of IMS_PER_BATCH:\n'
              '    cfg.FPN.RPN_COLLECT_SCALE: {}'.format(cfg.FPN.RPN_COLLECT_SCALE))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    # ipdb.set_trace()
    ### Overwrite some solver settings from command line arguments
    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer

    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma
    assert_and_infer_cfg()

    timers = defaultdict(Timer)

    ### Dataset ###
    timers['roidb'].tic()
    roidb, ratio_list, ratio_index = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb'].toc()
    roidb_size = len(roidb)
    logger.info('{:d} roidb entries'.format(roidb_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

    # Effective training sample size for one epoch
    train_size = roidb_size // args.batch_size * args.batch_size
    # ToDo: shuffle?
    batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index),
        batch_size=args.batch_size,
        drop_last=True)

    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batchSampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)
    # dataiterator = iter(dataloader)

    ### Model ###
    from modeling.model_builder import Generalized_RCNN
    maskRCNN = Generalized_RCNN()

    if cfg.CUDA:
        maskRCNN.cuda()

    ### Optimizer ###
    bias_hoi_params = []
    bias_hoi_param_names = []
    bias_faster_params = []
    bias_faster_param_names = []
    nobias_hoi_params = []
    nobias_hoi_param_names = []
    nobias_faster_params = []
    nobias_faster_param_names = []

    # bias_params = []
    # bias_param_names = []
    # nonbias_params = []
    # nonbias_param_names = []


    #base_model = torch.load('Outputs/baseline/baseline_512_32_nogt_1o3/ckpt/model_step47999.pth')

    nograd_param_names = []
    for key, value in maskRCNN.named_parameters():
        #if key in base_model['model'].keys():
        #   value.requires_grad = False

        #print('the key xxx:', key)
        # Fix RPN module same as the paper
        # ToDo: or key.startswith('Box')
        # if 'affinity' not in key:
        #     value.requires_grad = False

        print(key, value.size(), value.requires_grad)
        if value.requires_grad:
            if 'bias' in key:
                if 'HOI_Head' in key:
                    bias_hoi_params.append(value)
                    bias_hoi_param_names.append(key)
                else:
                    bias_faster_params.append(value)
                    bias_faster_param_names.append(key)
            else:
                if 'HOI_Head' in key:
                    nobias_hoi_params.append(value)
                    nobias_hoi_param_names.append(key)
                else:
                    nobias_faster_params.append(value)
                    nobias_faster_param_names.append(key)
        else:
            nograd_param_names.append(key)

    #del base_model
    #ipdb.set_trace()

    # Learning rate of 0 is a dummy value to be set properly at the start of training
    params = [
        {'params': nobias_hoi_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': nobias_faster_params,
         'lr': 0 * cfg.SOLVER.FASTER_RCNN_WEIGHT,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_hoi_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': bias_faster_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1) * cfg.SOLVER.FASTER_RCNN_WEIGHT,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
    ]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)

    ### Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        if args.krcnn_from_faster:
            net_utils.load_krcnn_from_faster(maskRCNN, checkpoint['model'])
        else:
            net_utils.load_ckpt(maskRCNN, checkpoint['model'])
            print('Original model loaded....')
        if args.resume:
            print('Resume, loaded step\n\n\n: ', checkpoint['step'])
            args.start_step = checkpoint['step'] + 1
            if 'train_size' in checkpoint:  # For backward compatibility
                if checkpoint['train_size'] != train_size:
                    print('train_size value: %d different from the one in checkpoint: %d'
                          % (train_size, checkpoint['train_size']))

            # reorder the params in optimizer checkpoint's params_groups if needed
            # misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)

            # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
            # However it's fixed on master.
            optimizer.load_state_dict(checkpoint['optimizer'])
            # misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()

    if args.load_detectron:  #TODO resume for detectron weights (load sgd momentum values)
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True)

    ### Training Setups ###
    args.run_name = misc_utils.get_run_name() + '_step'
    #output_dir = misc_utils.get_output_dir(args, args.run_name)
    output_dir = os.path.join('Outputs', args.expDir, args.expID)
    os.makedirs(output_dir, exist_ok=True)

    args.cfg_filename = os.path.basename(args.cfg_file)

    tblogger = None
    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)

    ### Training Loop ###
    train_val(maskRCNN, args, optimizer, lr, dataloader, train_size, output_dir, tblogger)


def train_val(model, args, optimizer, lr, dataloader, train_size, output_dir, tblogger=None):

    dataiterator = iter(dataloader)
    model.train()

    CHECKPOINT_PERIOD = cfg.TRAIN.SNAPSHOT_ITERS

    # Set index for decay steps
    decay_steps_ind = None
    for i in range(1, len(cfg.SOLVER.STEPS)):
        if cfg.SOLVER.STEPS[i] >= args.start_step:
            decay_steps_ind = i
            break
    if decay_steps_ind is None:
        decay_steps_ind = len(cfg.SOLVER.STEPS)

    training_stats = TrainingStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None)

    try:
        logger.info('Training starts !')
        step = args.start_step

        best_ap = 0
        best_step = 0
        running_tr_loss = 0.
        DRAW_STEP = args.start_step

        for step in range(args.start_step, cfg.SOLVER.MAX_ITER):
            #print('stepppp: ', step)
            # Warm up
            if step < cfg.SOLVER.WARM_UP_ITERS:
                method = cfg.SOLVER.WARM_UP_METHOD
                if method == 'constant':
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
                elif method == 'linear':
                    alpha = step / cfg.SOLVER.WARM_UP_ITERS
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                else:
                    raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                lr_new = cfg.SOLVER.BASE_LR * warmup_factor
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
            elif step == cfg.SOLVER.WARM_UP_ITERS:
                net_utils.update_learning_rate(optimizer, lr, cfg.SOLVER.BASE_LR)
                lr = optimizer.param_groups[0]['lr']
                assert lr == cfg.SOLVER.BASE_LR

            # Learning rate decay
            if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
                    step == cfg.SOLVER.STEPS[decay_steps_ind]:
                logger.info('Decay the learning on step %d', step)
                lr_new = lr * cfg.SOLVER.GAMMA
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
                decay_steps_ind += 1

            training_stats.IterTic()
            optimizer.zero_grad()

            for inner_iter in range(args.iter_size):
                try:
                    input_data = next(dataiterator)
                except StopIteration:
                    dataiterator = iter(dataloader)
                    input_data = next(dataiterator)

                for key in input_data:
                    if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))

                net_outputs = model(**input_data)
                training_stats.UpdateIterStats(net_outputs, inner_iter)
                loss = net_outputs['total_loss']
                #print('555')
                running_tr_loss += loss.item()
                if loss.requires_grad:
                    loss.backward()

            optimizer.step()
            training_stats.IterToc()
            training_stats.LogIterStats(step, lr)
            # print('CHECKPOINT_PERIOD: ', CHECKPOINT_PERIOD)

            # if (step+1) % 800==0 or step==cfg.SOLVER.MAX_ITER-1:
            #     print('\tAverage Training Runing loss of step {}: {:.8f}'.format(step+1, running_tr_loss/(step+1-DRAW_STEP)))
            #     tblogger.add_scalar('Runing_Training_loss', running_tr_loss/(step+1-DRAW_STEP), step+1)
            #     DRAW_STEP = step+1
            #     running_tr_loss = 0.

            CHECKPOINT_PERIOD = 2000
            if ((step+1) % CHECKPOINT_PERIOD == 0) or step==cfg.SOLVER.MAX_ITER-1:
                if(step+1)>15000:
                    save_ckpt(output_dir, args, step, train_size, model, optimizer)

        # ---- Training ends ----
        # Save last checkpoint
        #save_ckpt(output_dir, args, step, train_size, model, optimizer)

    except (RuntimeError, KeyboardInterrupt):
        del dataiterator
        logger.info('Save ckpt on exception ...')
        save_ckpt(output_dir, args, step, train_size, model, optimizer)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if args.use_tfboard and not args.no_save:
            tblogger.close()


def val(model, root_dir, step, mode='val'):
    if mode == 'train':
        dataset_name = 'vcoco_train'
    elif mode == 'val':
        dataset_name = 'vcoco_val'
    elif mode == 'test':
        dataset_name = 'vcoco_test'

    logger.info("start test {}_set while training".format(mode))
    model.eval()

    ids_back = model.device_ids
    model.device_ids = [0]

    output_dir = os.path.join(root_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    from core.test_engine import test_net


    ## all_hois here includes several outputs
    all_boxes, all_segms, all_keyps, all_hois, all_keyps_vcoco, all_losses \
        = test_net(args=None, dataset_name=dataset_name, proposal_file=None,
                   output_dir=output_dir, active_model=model, step=step)

    # pos_acc, neg_acc, AP, total_action_num, recall_action_num, \
    #     total_affinity_num, recall_affinity_num = evaluate_affinity(all_losses, thresh=0.5)
    # print('pos acc:{}, neg acc:{}, AP:{}'.format(pos_acc, neg_acc, AP))
    # print('total_action_num:{}, recall_action_num:{}, action_recall:{}'.format(total_action_num, recall_action_num, recall_action_num/total_action_num))
    # print('total_affinity_num:{}, recall_affinity_num:{}, affinity_recall:{}'.format(total_affinity_num, recall_affinity_num, recall_affinity_num/total_affinity_num))

    interaction_cls_loss = dict()

    interaction_action_loss_list = all_losses['interaction_action_loss']
    val_interaction_action_loss = sum(interaction_action_loss_list)/len(interaction_action_loss_list)

    interaction_affinity_loss_list = all_losses['interaction_affinity_loss']
    val_interaction_affinity_loss = sum(interaction_affinity_loss_list)/len(interaction_affinity_loss_list)

    interaction_cls_loss['interaction_action_loss'] = val_interaction_action_loss
    interaction_cls_loss['interaction_affinity_loss'] = val_interaction_affinity_loss

    dataset = JsonDataset(dataset_name)

    hois_keys_roidb = list(all_hois.keys())
    all_hois_3 = dict()
    all_hois_13 = dict()
    all_hois_23 = dict()
    all_hois_123 = dict()

    for roidb_id in hois_keys_roidb:
        multi_hois = all_hois[roidb_id]
        # print(multi_hois.keys())
        all_hois_3[roidb_id] =  dict(agents=multi_hois['agents'],
                                    roles = multi_hois['roles'])

        all_hois_13[roidb_id] = dict(agents=multi_hois['agents'],
                                    roles=multi_hois['roles1'])

        all_hois_23[roidb_id] = dict(agents=multi_hois['agents'],
                                    roles=multi_hois['roles2'])

        all_hois_123[roidb_id] = dict(agents=multi_hois['agents'],
                                    roles=multi_hois['roles3'])


    hois_list = [all_hois_3, all_hois_13, all_hois_23, all_hois_123]
    role_ap_list = []

    for idx, hoi_step in enumerate(hois_list):
        output_dir_step = osp.join(output_dir, 'role{}'.format(idx))
        if not osp.exists(output_dir_step):
            os.makedirs(output_dir_step)
        role_ap = task_evaluation.evaluate_hoi_vcoco(dataset, hoi_step, output_dir_step)
        role_ap_list.append(role_ap)

    model.device_ids = ids_back
    model.train()

    return role_ap_list, interaction_cls_loss
    
if __name__ == '__main__':
    main()
