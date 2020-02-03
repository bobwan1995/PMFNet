# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os.path as osp
from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml
from tqdm import tqdm
import torch

from core.config import cfg
# from core.rpn_generator import generate_rpn_on_dataset  #TODO: for rpn only case
# from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all, im_detect_all_precomp_box
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling.model_builder import Generalized_RCNN
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils
from utils.io import save_object
from utils.timer import Timer
from sklearn.metrics import average_precision_score
import ipdb
import json

logger = logging.getLogger(__name__)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    if cfg.MODEL.RPN_ONLY:
        raise NotImplementedError
        # child_func = generate_rpn_on_range
        # parent_func = generate_rpn_on_dataset
    else:
        # Generic case that handles all network types other than RPN-only nets
        # and RetinaNet
        child_func = test_net
        parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None
    final_results = {}

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset

            print(cfg.TEST.DATASETS)
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results, all_losses = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )

                result_list = results
                final_results[dataset_name+'_ap_3'] = result_list[0]
                final_results[dataset_name+'_ap_13'] = result_list[1]
                final_results[dataset_name+'_action_loss'] = all_losses['interaction_action_loss']
                final_results[dataset_name+'_affinity_loss'] = all_losses['interaction_affinity_loss']

                print('results saved')

            return results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    saved_path = os.path.join(args.output_dir, 'saved_AP_results.json')
    json.dump(final_results, open(saved_path, 'w'), cls=MyEncoder)

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        all_boxes, all_segms, all_keyps, all_hois, all_keyps_vcoco, all_losses = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_boxes, all_segms, all_keyps, all_hois, all_keyps_vcoco, all_losses = test_net(
            args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))

    interaction_cls_loss = dict()
    interaction_action_loss_list = all_losses['interaction_action_loss']
    val_interaction_action_loss = sum(interaction_action_loss_list) / len(interaction_action_loss_list)
    interaction_affinity_loss_list = all_losses['interaction_affinity_loss']
    val_interaction_affinity_loss = sum(interaction_affinity_loss_list) / len(interaction_affinity_loss_list)

    interaction_cls_loss['interaction_action_loss'] = val_interaction_action_loss
    interaction_cls_loss['interaction_affinity_loss'] = val_interaction_affinity_loss

    dataset = JsonDataset(dataset_name)

    hois_keys_roidb = list(all_hois.keys())
    all_hois_3 = dict()
    all_hois_13 = dict()

    for roidb_id in hois_keys_roidb:
        multi_hois = all_hois[roidb_id]
        all_hois_3[roidb_id] = dict(agents=multi_hois['agents'],
                                    roles=multi_hois['roles'])

        all_hois_13[roidb_id] = dict(agents=multi_hois['agents'],
                                     roles=multi_hois['roles1'])

    hois_list = [all_hois_3, all_hois_13]
    role_ap_list = []

    for idx, hoi_step in enumerate(hois_list):
        output_dir_step = osp.join(output_dir, 'role{}'.format(idx))
        if not osp.exists(output_dir_step):
            os.makedirs(output_dir_step)
        role_ap = task_evaluation.evaluate_hoi_vcoco(dataset, hoi_step, output_dir_step)
        role_ap_list.append(role_ap)

    pos_acc, neg_acc, AP, total_action_num, recall_action_num, total_affinity_num, recall_affinity_num = evaluate_affinity(all_losses)
    print('pos acc:{}, neg acc:{}, AP:{}'.format(pos_acc, neg_acc, AP))
    print('total_action_num:{}, recall_recall_num:{}, action_recall:{}'.format(total_action_num, recall_action_num, recall_action_num/total_action_num))
    print('total_affinity_num:{}, recall_affinity_num:{}, affinity_recall:{}'.format(total_affinity_num, recall_affinity_num, recall_affinity_num/total_affinity_num))

    return  role_ap_list, interaction_cls_loss


def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir,
        args.load_ckpt, args.load_detectron, args.net_name, args.mlp_head_dim, 
        args.heatmap_kernel_size, args.part_crop_size, args.use_kps17,
        opts)

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_hois = {}
    all_losses = defaultdict(list)
    all_keyps_vcoco = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        all_hois = {**all_hois, **det_data['all_hois']}
        for k, v in det_data['all_losses'].items():
            all_losses[k].extend(v)

        all_keyps_vcoco_batch = det_data['all_keyps_vcoco']
        for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[cls_idx] += all_boxes_batch[cls_idx]
            all_segms[cls_idx] += all_segms_batch[cls_idx]
            all_keyps[cls_idx] += all_keyps_batch[cls_idx]
            all_keyps_vcoco[cls_idx] += all_keyps_vcoco_batch[cls_idx]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            all_hois=all_hois,
            all_keyps_vcoco=all_keyps_vcoco,
            all_losses=all_losses,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps, all_hois, all_keyps_vcoco, all_losses


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0,
        active_model=None,
        step=None):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )
    if active_model is None:
        model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    else:
        model = active_model

    if 'train' in dataset_name:
        mode = 'train'
    elif 'val' in dataset_name:
        mode = 'val'
    elif 'test' in dataset_name:
        mode = 'test'
    else:
        raise Exception

    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    # num_images = 5
    all_boxes, all_segms, all_keyps, all_hois, all_keyps_vcoco = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)
    all_losses = defaultdict(list)
    for i, entry in enumerate(roidb):

        if cfg.TEST.PRECOMPUTED_PROPOSALS:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            if len(box_proposals) == 0:
                continue
        else:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN; 1-stage models don't require proposals.
            box_proposals = None
        # h, w, c
        im = cv2.imread(entry['image'])

        if not cfg.VCOCO.USE_PRECOMP_BOX:
            cls_boxes_i, cls_segms_i, cls_keyps_i, hoi_res_i, vcoco_cls_keyps_i, loss_i = \
                im_detect_all(model, im, box_proposals, timers, entry)
        else:
            cls_boxes_i, cls_segms_i, cls_keyps_i, hoi_res_i, vcoco_cls_keyps_i, loss_i = \
                im_detect_all_precomp_box(model, im, timers, entry, mode, dataset.json_category_id_to_contiguous_id)

        extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)
        if hoi_res_i is not None:
            all_hois[entry['id']] = hoi_res_i
        if vcoco_cls_keyps_i is not None:
            extend_results(i, all_keyps_vcoco, vcoco_cls_keyps_i)

        if loss_i['interaction_action_loss'] is not None:
            for k, v in loss_i.items():
                all_losses[k].append(v)

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        if step is None:
            det_name = 'detections.pkl'
        else:
            det_name = 'detections_step{}.pkl'.format(step)
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            all_hois=all_hois,
            all_keyps_vcoco=all_keyps_vcoco,
            all_losses=all_losses,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes, all_segms, all_keyps, all_hois, all_keyps_vcoco, all_losses


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])
        # model.load_state_dict(checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb(gt=cfg.DEBUG_TEST_WITH_GT)

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    Human-object-interaction predictions is collected into:
      list of dict, dict include image id, human box, action score, role objects
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_hois  = {}
    all_keyps_vcoco = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps, all_hois, all_keyps_vcoco


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]


def evaluate_affinity(loss, thresh=0.1):
    # ipdb.set_trace()
    interaction_affinity_label = np.concatenate(loss['interaction_affinity_label'])
    interaction_affinity_score  = np.concatenate(loss['interaction_affinity_score'])
    # gt_action_num = np.concatenate(loss['gt_action_num'])
    pos_ind = np.where(interaction_affinity_label)[0]
    neg_ind = np.where(interaction_affinity_label == False)[0]

    tp_ind = np.where(interaction_affinity_score[pos_ind] >= thresh)[0]
    tf_ind = np.where(interaction_affinity_score[neg_ind] <= thresh)[0]
    # recall_num = np.sum(gt_action_num[pos_ind][tp_ind])
    total_action_num = sum(loss['total_action_num'])
    recall_action_num = sum(loss['recall_action_num'])
    total_affinity_num = sum(loss['total_affinity_num'])
    recall_affinity_num = sum(loss['recall_affinity_num'])

    pos_acc = tp_ind.shape[0]/pos_ind.shape[0]
    neg_acc = tf_ind.shape[0]/neg_ind.shape[0]
    AP = average_precision_score(interaction_affinity_label, interaction_affinity_score)
    return pos_acc, neg_acc, AP, total_action_num, recall_action_num, total_affinity_num, recall_affinity_num
