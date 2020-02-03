# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import numpy as np
import pycocotools.mask as mask_util

from torch.autograd import Variable
import torch
import ipdb
import math

from core.config import cfg
from utils.timer import Timer
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import utils.image as image_utils
import utils.keypoints as keypoint_utils
from roi_data.hoi_data import get_hoi_blob_names
from roi_data.hoi_data_union import get_hoi_union_blob_names, generate_union_mask, generate_joints_heatmap
from roi_data.hoi_data_union import generate_pose_configmap, generate_part_box_from_kp, generate_part_box_from_kp17
from datasets import json_dataset
import torch.nn.functional as F
import time

def im_detect_all(model, im, box_proposals=None, timers=None, entry=None):
    """Process the outputs of model for testing
    Args:
      model: the network module
      im_data: Pytorch variable. Input batch to the model.
      im_info: Pytorch variable. Input batch to the model.
      gt_boxes: Pytorch variable. Input batch to the model.
      num_boxes: Pytorch variable. Input batch to the model.
      args: arguments from command line.
      timer: record the cost of time for different steps
    The rest of inputs are of type pytorch Variables and either input to or output from the model.
    """
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        # boxes is in origin img size
        scores, boxes, im_scale, blob_conv = im_detect_bbox_aug(
            model, im, box_proposals)
    else:
        scores, boxes, im_scale, blob_conv = im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals)
    timers['im_detect_bbox'].toc()
    im_info = np.array([im.shape[:2]+(im_scale[0],)])

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class) (numpy.ndarray)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            masks = im_detect_mask_aug(model, im, boxes, im_scale, blob_conv)
        else:
            masks = im_detect_mask(model, im_scale, boxes, blob_conv)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.KPS_AUG.ENABLED:
            heatmaps = im_detect_keypoints_aug(model, im, boxes, im_scale, blob_conv)
        else:
            heatmaps = im_detect_keypoints(model, im_scale, boxes, blob_conv)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        cls_keyps = keypoint_results(cls_boxes, heatmaps, boxes)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None

    vcoco_heatmaps = None
    if cfg.MODEL.VCOCO_ON:
        if cfg.VCOCO.KEYPOINTS_ON:
            # ipdb.set_trace()
            vcoco_heatmaps, vcoco_heatmaps_np = im_detect_keypoints_vcoco(model, im_scale[0], cls_boxes[1][:, :4], blob_conv)
            vcoco_cls_keyps = keypoint_results_vcoco(cls_boxes, vcoco_heatmaps_np)
        else:
            vcoco_cls_keyps = None

        hoi_res = im_detect_hoi_union(model, boxes, scores, cls_boxes[1].shape[0],
                                      im_info, blob_conv, entry,
                                      vcoco_heatmaps)
    else:
        hoi_res = None
        vcoco_cls_keyps = None

    return cls_boxes, cls_segms, cls_keyps, hoi_res, vcoco_cls_keyps


def im_detect_all_precomp_box(model, im, timers=None, entry=None, mode='val', category_id_to_contiguous_id=None):
    """Process the outputs of model for testing
    Args:
      model: the network module
      im_data: Pytorch variable. Input batch to the model.
      im_info: Pytorch variable. Input batch to the model.
      gt_boxes: Pytorch variable. Input batch to the model.
      num_boxes: Pytorch variable. Input batch to the model.
      args: arguments from command line.
      timer: record the cost of time for different steps
    The rest of inputs are of type pytorch Variables and either input to or output from the model.
    """
    if timers is None:
        timers = defaultdict(Timer)
    blob_conv, im_scale = im_conv_body_only(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    im_info = np.array([im.shape[:2] + (im_scale[0],)])
    scores, boxes, cates, cls_boxes = im_detect_bbox_precomp_box(entry, category_id_to_contiguous_id)

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            masks = im_detect_mask_aug(model, im, boxes, im_scale, blob_conv)
        else:
            masks = im_detect_mask(model, im_scale, boxes, blob_conv)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.KPS_AUG.ENABLED:
            heatmaps = im_detect_keypoints_aug(model, im, boxes, im_scale, blob_conv)
        else:
            heatmaps = im_detect_keypoints(model, im_scale, boxes, blob_conv)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        cls_keyps = keypoint_results(cls_boxes, heatmaps, boxes)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None

    vcoco_heatmaps = None
    vcoco_cls_keyps = None
    loss = None
    if cfg.MODEL.VCOCO_ON:

        hoi_res, loss = im_detect_hoi_union(model, boxes, scores, cates, cls_boxes[1].shape[0],
                                      im_info, blob_conv, entry, mode,
                                      vcoco_heatmaps)
    else:
        hoi_res = None
        vcoco_cls_keyps = None

    return cls_boxes, cls_segms, cls_keyps, hoi_res, vcoco_cls_keyps, loss


def im_conv_body_only(model, im, target_scale, target_max_size):
    inputs, im_scale = _get_blobs(im, None, target_scale, target_max_size)

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = Variable(torch.from_numpy(inputs['data']), volatile=True).cuda()
    else:
        inputs['data'] = torch.from_numpy(inputs['data']).cuda()
    inputs.pop('im_info')

    blob_conv = model.module.convbody_net(**inputs)

    return blob_conv, im_scale


def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None):
    """Prepare the bbox for testing"""

    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
    else:
        inputs['data'] = [torch.from_numpy(inputs['data'])]
        inputs['im_info'] = [torch.from_numpy(inputs['im_info'])]

    time1 = time.time()
    return_dict = model(**inputs)
    time2 = time.time()
    print('model_time:', time2-time1)

    if cfg.MODEL.FASTER_RCNN:
        rois = return_dict['rois'].data.cpu().numpy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scale

    # cls prob (activations after softmax)
    scores = return_dict['cls_score'].data.cpu().numpy().squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = return_dict['bbox_pred'].data.cpu().numpy().squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # (legacy) Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                         + cfg.TRAIN.BBOX_NORMALIZE_MEANS
        pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scale, return_dict['blob_conv']


def im_detect_bbox_precomp_box(entry, category_id_to_contiguous_id):
    """Prepare the bbox for testing"""
    # box in origin image
    pred_boxes = entry['precomp_boxes']
    scores = entry['precomp_score']
    cates = entry['precomp_cate'].astype(np.int32)

    contiguous_cate = list()
    for cls in cates:
        # ipdb.set_trace()
        if category_id_to_contiguous_id.get(cls) is None:
            contiguous_cate.append(80)
        else:
            contiguous_cate.append(category_id_to_contiguous_id[cls])
    cates = np.array(contiguous_cate, dtype=cates.dtype)


    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]

    box_sc = np.concatenate([pred_boxes, scores[:, None]], 1)
    unique_cates = np.unique(cates)
    for c in unique_cates:
        if category_id_to_contiguous_id.get(c) is not None:
            inds = np.where(cates == c)
            cls_boxes[category_id_to_contiguous_id[c]] = box_sc[inds]

    if len(cls_boxes[1]) == 0:
        cls_boxes[1] = np.empty((0,5), dtype=np.float32)

    return scores, pred_boxes, cates, cls_boxes


def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _ = im_detect_bbox_hflip(
            model,
            im,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals
        )
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals
        )
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True
            )
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals
        )
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True
            )
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scale_i, blob_conv_i = im_detect_bbox(
        model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
    )
    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
        )

    return scores_c, boxes_c, im_scale_i, blob_conv_i


def im_detect_bbox_hflip(
        model, im, target_scale, target_max_size, box_proposals=None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_hf = im[:, ::-1, :]
    im_width = im.shape[1]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scale, _ = im_detect_bbox(
        model, im_hf, target_scale, target_max_size, boxes=box_proposals_hf
    )

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scale


def im_detect_bbox_scale(
        model, im, target_scale, target_max_size, box_proposals=None, hflip=False):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposals
        )
    else:
        scores_scl, boxes_scl, _, _ = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals
        )
    return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(
        model, im, aspect_ratio, box_proposals=None, hflip=False):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _, _ = im_detect_bbox(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            boxes=box_proposals_ar
        )

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def im_detect_mask(model, im_scale, boxes, blob_conv):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scale (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)
        blob_conv (Variable): base features from the backbone network.

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    pred_masks = model.module.mask_net(blob_conv, inputs)
    pred_masks = pred_masks.data.cpu().numpy().squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    return pred_masks


def im_detect_mask_aug(model, im, boxes, im_scale, blob_conv):
    """Performs mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes
        im_scale (list): image blob scales as returned by im_detect_bbox
        blob_conv (Tensor): base features from the backbone network.

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    masks_ts = []

    # Compute masks for the original image (identity transform)
    masks_i = im_detect_mask(model, im_scale, boxes, blob_conv)
    masks_ts.append(masks_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        masks_hf = im_detect_mask_hflip(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes
        )
        masks_ts.append(masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        masks_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
        masks_ts.append(masks_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            masks_scl_hf = im_detect_mask_scale(
                model, im, scale, max_size, boxes, hflip=True
            )
            masks_ts.append(masks_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        masks_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        masks_ts.append(masks_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            masks_ar_hf = im_detect_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            masks_ts.append(masks_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        masks_c = np.mean(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        masks_c = np.amax(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

        logit_masks = [logit(y) for y in masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR)
        )

    return masks_c


def im_detect_mask_hflip(model, im, target_scale, target_max_size, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    blob_conv, im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    masks_hf = im_detect_mask(model, im_scale, boxes_hf, blob_conv)

    # Invert the predicted soft masks
    masks_inv = masks_hf[:, :, :, ::-1]

    return masks_inv


def im_detect_mask_scale(
        model, im, target_scale, target_max_size, boxes, hflip=False):
    """Computes masks at the given scale."""
    if hflip:
        masks_scl = im_detect_mask_hflip(
            model, im, target_scale, target_max_size, boxes
        )
    else:
        blob_conv, im_scale = im_conv_body_only(model, im, target_scale, target_max_size)
        masks_scl = im_detect_mask(model, im_scale, boxes, blob_conv)
    return masks_scl


def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        masks_ar = im_detect_mask_hflip(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar
        )
    else:
        blob_conv, im_scale = im_conv_body_only(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
        )
        masks_ar = im_detect_mask(model, im_scale, boxes_ar, blob_conv)

    return masks_ar


def im_detect_keypoints_vcoco(model, im_scale, human_boxes, blob_conv):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scale (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    M = cfg.KRCNN.HEATMAP_SIZE

    if human_boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return None, pred_heatmaps

    # project boxes to re-sized image size
    human_boxes = np.hstack((np.zeros((human_boxes.shape[0], 1), dtype=human_boxes.dtype),
                              human_boxes * im_scale))

    inputs = {'human_boxes': human_boxes}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'human_boxes')

    pred_heatmaps = model.module.vcoco_keypoint_net(blob_conv, inputs)
    np_pred_heatmaps = pred_heatmaps.data.cpu().numpy().squeeze()

    # In case of 1
    if np_pred_heatmaps.ndim == 3:
        np_pred_heatmaps = np.expand_dims(np_pred_heatmaps, axis=0)

    return pred_heatmaps, np_pred_heatmaps


def keypoint_results_vcoco(cls_boxes, pred_heatmaps):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()
    xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, cls_boxes[person_idx])

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils.nms_oks(xy_preds, cls_boxes[person_idx], 0.3)
        xy_preds = xy_preds[keep, :, :]
        # ref_boxes = ref_boxes[keep, :]
        # pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def im_detect_keypoints(model, im_scale, boxes, blob_conv):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scale (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    pred_heatmaps = model.module.keypoint_net(blob_conv, inputs)
    pred_heatmaps = pred_heatmaps.data.cpu().numpy().squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def im_detect_keypoints_aug(model, im, boxes, im_scale, blob_conv):
    """Computes keypoint predictions with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes
        im_scale (list): image blob scales as returned by im_detect_bbox
        blob_conv (Tensor): base features from the backbone network.

    Returns:
        heatmaps (ndarray): R x J x M x M array of keypoint location logits
    """
    # Collect heatmaps predicted under different transformations
    heatmaps_ts = []
    # Tag predictions computed under downscaling and upscaling transformations
    ds_ts = []
    us_ts = []

    def add_heatmaps_t(heatmaps_t, ds_t=False, us_t=False):
        heatmaps_ts.append(heatmaps_t)
        ds_ts.append(ds_t)
        us_ts.append(us_t)

    # Compute the heatmaps for the original image (identity transform)
    heatmaps_i = im_detect_keypoints(model, im_scale, boxes, blob_conv)
    add_heatmaps_t(heatmaps_i)

    # Perform keypoints detection on the horizontally flipped image
    if cfg.TEST.KPS_AUG.H_FLIP:
        heatmaps_hf = im_detect_keypoints_hflip(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.KPS_AUG.SCALES:
        ds_scl = scale < cfg.TEST.SCALE
        us_scl = scale > cfg.TEST.SCALE
        heatmaps_scl = im_detect_keypoints_scale(
            model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_scl, ds_scl, us_scl)

        if cfg.TEST.KPS_AUG.SCALE_H_FLIP:
            heatmaps_scl_hf = im_detect_keypoints_scale(
                model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_scl_hf, ds_scl, us_scl)

    # Compute keypoints at different aspect ratios
    for aspect_ratio in cfg.TEST.KPS_AUG.ASPECT_RATIOS:
        heatmaps_ar = im_detect_keypoints_aspect_ratio(
            model, im, aspect_ratio, boxes
        )
        add_heatmaps_t(heatmaps_ar)

        if cfg.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP:
            heatmaps_ar_hf = im_detect_keypoints_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_ar_hf)

    # Select the heuristic function for combining the heatmaps
    if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
        np_f = np.mean
    elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
        np_f = np.amax
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR)
        )

    def heur_f(hms_ts):
        return np_f(hms_ts, axis=0)

    # Combine the heatmaps
    if cfg.TEST.KPS_AUG.SCALE_SIZE_DEP:
        heatmaps_c = combine_heatmaps_size_dep(
            heatmaps_ts, ds_ts, us_ts, boxes, heur_f
        )
    else:
        heatmaps_c = heur_f(heatmaps_ts)

    return heatmaps_c


def im_detect_keypoints_hflip(model, im, target_scale, target_max_size, boxes):
    """Computes keypoint predictions on the horizontally flipped image.
    Function signature is the same as for im_detect_keypoints_aug.
    """
    # Compute keypoints for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    blob_conv, im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    heatmaps_hf = im_detect_keypoints(model, im_scale, boxes_hf, blob_conv)

    # Invert the predicted keypoints
    heatmaps_inv = keypoint_utils.flip_heatmaps(heatmaps_hf)

    return heatmaps_inv


def im_detect_keypoints_scale(
    model, im, target_scale, target_max_size, boxes, hflip=False):
    """Computes keypoint predictions at the given scale."""
    if hflip:
        heatmaps_scl = im_detect_keypoints_hflip(
            model, im, target_scale, target_max_size, boxes
        )
    else:
        blob_conv, im_scale = im_conv_body_only(model, im, target_scale, target_max_size)
        heatmaps_scl = im_detect_keypoints(model, im_scale, boxes, blob_conv)
    return heatmaps_scl


def im_detect_keypoints_aspect_ratio(
    model, im, aspect_ratio, boxes, hflip=False):
    """Detects keypoints at the given width-relative aspect ratio."""

    # Perform keypoint detectionon the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        heatmaps_ar = im_detect_keypoints_hflip(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar
        )
    else:
        blob_conv, im_scale = im_conv_body_only(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
        )
        heatmaps_ar = im_detect_keypoints(model, im_scale, boxes_ar, blob_conv)

    return heatmaps_ar


def combine_heatmaps_size_dep(hms_ts, ds_ts, us_ts, boxes, heur_f):
    """Combines heatmaps while taking object sizes into account."""
    assert len(hms_ts) == len(ds_ts) and len(ds_ts) == len(us_ts), \
        'All sets of hms must be tagged with downscaling and upscaling flags'

    # Classify objects into small+medium and large based on their box areas
    areas = box_utils.boxes_area(boxes)
    sm_objs = areas < cfg.TEST.KPS_AUG.AREA_TH
    l_objs = areas >= cfg.TEST.KPS_AUG.AREA_TH

    # Combine heatmaps computed under different transformations for each object
    hms_c = np.zeros_like(hms_ts[0])

    for i in range(hms_c.shape[0]):
        hms_to_combine = []
        for hms_t, ds_t, us_t in zip(hms_ts, ds_ts, us_ts):
            # Discard downscaling predictions for small and medium objects
            if sm_objs[i] and ds_t:
                continue
            # Discard upscaling predictions for large objects
            if l_objs[i] and us_t:
                continue
            hms_to_combine.append(hms_t[i])
        hms_c[i] = heur_f(hms_to_combine)

    return hms_c


def box_results_with_nms_and_limit(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.05,
                # score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = (ref_box[2] - ref_box[0] + 1)
            h = (ref_box[3] - ref_box[1] + 1)
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            # For dumping to json, need to decode the byte string.
            # https://github.com/cocodataset/cocoapi/issues/70
            rle['counts'] = rle['counts'].decode('ascii')
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms


def keypoint_results(cls_boxes, pred_heatmaps, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()
    xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, ref_boxes)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_utils.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn_utils.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale


# -------------------------- HOI ----------------------------

def im_detect_hoi(model, boxes, scores, human_count, im_info, blob_conv, entry=None, vcoco_heatmaps=None):
    hoi_blob_in = get_hoi_blob_names(is_training=False)

    # im_info.shape = (1, 3) h, w, scale
    im_scale = im_info[0, 2]
    # project boxes to re-sized image size
    hoi_blob_in['boxes'] = np.hstack((np.zeros((boxes.shape[0], 1), dtype=boxes.dtype),
                                      boxes * im_scale))
    hoi_blob_in['scores'] = scores

    human_index = np.arange(boxes.shape[0])[:human_count]
    object_index = np.arange(boxes.shape[0])[human_count:]

    interaction_human_inds, interaction_target_object_inds \
        = np.repeat(human_index, object_index.size), np.tile(object_index - human_count, human_index.size)

    hoi_blob_in['human_index'] = human_index
    hoi_blob_in['target_object_index'] = object_index
    hoi_blob_in['interaction_human_inds'] = interaction_human_inds
    hoi_blob_in['interaction_target_object_inds'] = interaction_target_object_inds

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(hoi_blob_in, 'boxes')

    # if no human box is detected, not use hoi_head, just return nan
    if human_index.size > 0:
        hoi_blob_out = model.module.hoi_net(blob_conv, hoi_blob_in, im_info, vcoco_heatmaps)
        # ipdb.set_trace()
        # if entry:
        #     test_hoi_fill_hoi_blob_from_gt(hoi_blob_out, entry, im_scale)
        hoi_res = hoi_res_gather(hoi_blob_out, im_scale, entry)
    else:
        # ToDo: any problem here?
        hoi_res = dict(
            agents=np.full((1, 4 + cfg.VCOCO.NUM_ACTION_CLASSES), np.nan),
            roles=np.full((1, 5 * cfg.VCOCO.NUM_ACTION_CLASSES, cfg.VCOCO.NUM_TARGET_OBJECT_TYPES), np.nan),
        )

    return hoi_res


def hoi_res_gather(hoi_blob, im_scale, entry=None):
    '''
    Convert predicted score and location to triplets
    :param hoi_blob:
    :param im_scale:
    :param entry:
    :return:
    '''
    # ToDo: modify comments
    num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
    num_target_object_types = cfg.VCOCO.NUM_TARGET_OBJECT_TYPES

    human_action_score = F.sigmoid(hoi_blob['human_action_score']).cpu().numpy()
    human_action_bbox_pred = hoi_blob['human_action_bbox_pred'].cpu().numpy()
    interaction_action_score = F.sigmoid(hoi_blob['interaction_action_score']).cpu().numpy()

    human_score = hoi_blob['scores'][hoi_blob['human_index']]
    object_score = hoi_blob['scores'][hoi_blob['target_object_index']]
    # scale to original image size when testing
    boxes = hoi_blob['boxes'][:, 1:] / im_scale

    # For actions don't interact with object, action_score is s_h * s^a_h
    # For triplets(interact with objects), action_score is s_h * s_o * s^a_h * g^a_h,o
    # we use mask to choose appropriate score
    action_mask = np.array(cfg.VCOCO.ACTION_MASK)
    triplet_action_mask = np.tile(action_mask.transpose((1, 0)), (human_action_score.shape[0], 1, 1))

    # For actions that that do not interact with any object (e.g., smile, run),
    # we rely on s^a_h and the interaction output s^a_h_o is not used,
    human_action_pair_score = human_score[:, np.newaxis] * human_action_score

    # in case there is no role-objects
    if hoi_blob['target_object_index'].size > 0:
        # transform from (human num, object num, action_num) to
        # (human_num*action_num*num_target_object_types, object_num)
        interaction_action_score = \
            interaction_action_score.reshape(human_score.size, object_score.size, -1).transpose(0, 2, 1)
        interaction_action_score = np.repeat(interaction_action_score, num_target_object_types, axis=1
                                             ).reshape(-1, object_score.size)

        # get target localization term g^a_h,o
        target_localization_term = target_localization(boxes, hoi_blob['human_index'],
                                                       hoi_blob['target_object_index'], human_action_bbox_pred)
        # find the object box that maximizes S^a_h,o
        # `for each human / action pair we find the object box that maximizes S_h_o^a`
        object_action_score = object_score * interaction_action_score * target_localization_term
        choosed_object_inds = np.argmax(object_action_score, axis=-1)

        # choose corresponding target_localization_term
        target_localization_term = target_localization_term[np.arange(choosed_object_inds.size), choosed_object_inds]
        # ToDo: choose top-50
        # triplet score S^a_h,o
        triplet_action_score = \
            np.repeat(human_score, num_action_classes * num_target_object_types) * \
            object_score[choosed_object_inds] * \
            np.repeat(human_action_score, num_target_object_types, axis=1).ravel() * \
            target_localization_term
        # transform to (human_num, action_num, num_target_object_types)
        triplet_action_score = triplet_action_score.reshape(human_action_score.shape[0], num_action_classes,
                                                            num_target_object_types)
        # ToDo: thresh
        # triplet_action_score[triplet_action_mask <= cfg.TEST.SCORE_THRESH] = np.nan
        if entry:
            # assert triplet_action_score.shape == entry['gt_role_id'][hoi_blob['human_index']].shape
            for i in range(len(triplet_action_score.shape)):
                pass
                # assert np.all(np.where(triplet_action_score > 0.9)[i] ==
                #               np.where(entry['gt_role_id'][hoi_blob['human_index']] > -1)[i])

        # choose appropriate score
        # ToDo: any problem here?
        # As not every action that defined interacts with objects will have
        # corresponding objects in one image, and triplet_action_score always
        # have a object box, should I set a thresh or some method to choose
        # score between human_action_pair_score and triplet score???
        # OR wrong result will be excluded when calculate AP??
        # action_score = np.zeros(human_action_score.shape)
        # action_score[human_action_mask == 0] = human_action_pair_score[human_action_mask == 0]
        # action_score[human_action_mask == 1] = np.amax(triplet_action_score, axis=-1)[human_action_mask == 1]

        # set triplet action score don't interact with object to zero
        # triplet_action_score[triplet_action_mask == 0] = np.nan
        triplet_action_score[triplet_action_mask == 0] = -1
        top_k_value = triplet_action_score.flatten()[
            np.argpartition(triplet_action_score, -cfg.VCOCO.KEEP_TOP_NUM, axis=None)[-cfg.VCOCO.KEEP_TOP_NUM]]
        triplet_action_score[triplet_action_score <= top_k_value] = np.nan
        # get corresponding box of role-objects
        choosed_object_inds = choosed_object_inds.reshape(human_action_score.shape[0], num_action_classes,
                                                          num_target_object_types)
        choosed_objects = boxes[hoi_blob['target_object_index']][choosed_object_inds]
    else:
        # if there is no object predicted, triplet action score won't used
        triplet_action_score = np.full((1, num_action_classes, num_target_object_types), np.nan)
        choosed_objects = np.zeros((1, num_action_classes, num_target_object_types, 4))

    action_score = human_action_pair_score

    # ToDo: threshold
    # action_score[action_score <= cfg.TEST.SCORE_THRESH] = np.nan

    # keep consistent with v-coco eval code
    # agents: box coordinates + 26 action score.
    # roles: 26 * (role object coordinates + role-action score) * num_target_object_types
    agents = np.hstack((boxes[hoi_blob['human_index']], action_score))
    roles = np.concatenate((choosed_objects, triplet_action_score[..., np.newaxis]), axis=-1)
    roles = np.stack([roles[:, :, i, :].reshape(-1, num_action_classes * 5) for i in range(num_target_object_types)], axis=-1)

    return_dict = dict(
        # image_id=i
        agents=agents,
        roles=roles
    )
    return return_dict


def target_localization(boxes, human_index, object_index, target_location):
    """
    Target localization term in paper, g^a_h,o
    Measure compatibility between human-object relative location and
    target location, which is predicted by hoi-head
    :param boxes:
    :param human_index:
    :param object_index:
    :param target_location:
    :return:
    """
    human_boxes = boxes[human_index]
    object_boxes = boxes[object_index]
    num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
    num_target_object_types = cfg.VCOCO.NUM_TARGET_OBJECT_TYPES

    # relative location between every human box and object box
    # ToDo: add cfg.MODEL.BBOX_REG_WEIGHTS
    relative_location = box_utils.bbox_transform_inv(
        np.repeat(human_boxes, object_boxes.shape[0], axis=0),
        np.tile(object_boxes, (human_boxes.shape[0], 1))
    ).reshape(human_boxes.shape[0], object_boxes.shape[0], 4)

    # reshape target location same shape as relative location
    target_location = target_location.reshape(-1, num_action_classes * num_target_object_types, 4)

    # tile to human_num * (num_action_classes * num_target_object_types * object_num) * 4
    relative_location, target_location = \
        np.tile(relative_location, (1, num_action_classes * num_target_object_types, 1)), \
        np.repeat(target_location, relative_location.shape[1], axis=1)

    compatibility = np.sum(np.square((relative_location - target_location)), axis=-1)
    # It seems the paper make a mistake here
    compatibility = np.exp(-compatibility / (2 * cfg.VCOCO.TARGET_SIGMA ** 2))

    # reshape to (human_num * num_action_classes * num_target_object_types, object_num)
    compatibility = compatibility.reshape(human_index.size * num_action_classes * num_target_object_types,
                                          object_index.size)
    return compatibility

# ------------------test interact net code ------------------
# ToDo: will be cleaned


def test_hoi_fill_hoi_blob_from_gt(hoi_blob, entry, im_scale):
    """['boxes', 'human_index', 'target_object_index', 'interaction_human_inds',
    'interaction_target_object_inds', 'interaction_batch_idx', 'human_action_labels',
    'human_action_targets', 'action_target_weights', 'interaction_action_labels',
    'boxes_fpn2', 'boxes_fpn3', 'boxes_fpn4', 'boxes_fpn5', 'boxes_idx_restore_int32',
     'human_action_score', 'human_action_bbox_pred', 'interaction_action_score']"""
    hoi_blob['boxes'] = np.hstack((np.zeros((entry['boxes'].shape[0], 1), dtype=hoi_blob['boxes'].dtype),
                                   entry['boxes'])) * im_scale
    hoi_blob['scores'] = np.ones(entry['boxes'].shape[0])

    human_index = np.where(entry['gt_actions'][:, 0] > -1)[0]
    # all object could be target object
    target_object_index = np.arange(entry['boxes'].shape[0], dtype=human_index.dtype)

    interaction_human_inds, interaction_target_object_inds \
        = np.repeat(np.arange(human_index.size), target_object_index.size), \
          np.tile(np.arange(target_object_index.size), human_index.size)

    hoi_blob['human_index'] = human_index
    hoi_blob['target_object_index'] = target_object_index
    hoi_blob['interaction_human_inds'] = interaction_human_inds
    hoi_blob['interaction_target_object_inds'] = interaction_target_object_inds

    human_action_score = entry['gt_actions'][human_index]
    hoi_blob['human_action_score'] = torch.from_numpy(human_action_score).cuda()

    action_label_mat = generate_action_mat(entry['gt_role_id'])
    triplet_label = action_label_mat[human_index[interaction_human_inds],
                                     target_object_index[interaction_target_object_inds]]
    hoi_blob['interaction_action_score'] = torch.from_numpy(triplet_label).cuda()

    human_action_bbox_pred, _ = \
        _compute_action_targets(entry['boxes'][human_index], entry['boxes'],
                                entry['gt_role_id'][human_index])
    hoi_blob['human_action_bbox_pred'] = torch.from_numpy(human_action_bbox_pred).cuda()


def generate_action_mat(gt_role_id):
    '''
    Generate a matrix to store action triplet
    :param gt_role_id:
    :return: action_mat, row is person id, column is role-object id,
             third axis is action id
    '''
    mat = np.zeros((gt_role_id.shape[0], gt_role_id.shape[0], cfg.VCOCO.NUM_ACTION_CLASSES, gt_role_id.shape[-1]), dtype=np.float32)
    obj_ids = gt_role_id[np.where(gt_role_id > -1)]
    human_ids, action_cls, role_cls = np.where(gt_role_id > -1)
    assert role_cls.size == human_ids.size == action_cls.size == obj_ids.size
    mat[human_ids, obj_ids, action_cls, role_cls] = 1
    return mat


def _compute_action_targets(person_rois, gt_boxes, role_ids):
    '''
    Compute action targets
    :param person_rois: rois assigned to gt acting-human, n * 4
    :param gt_boxes: all gt boxes in one image
    :param role_ids: person_rois_num * action_cls_num * num_target_object_types, store person rois corresponding role object ids
    :return:
    '''

    assert person_rois.shape[0] == role_ids.shape[0]
    # should use cfg.MODEL.BBOX_REG_WEIGHTS?
    # calculate targets between every person rois and every gt_boxes
    targets = box_utils.bbox_transform_inv(np.repeat(person_rois, gt_boxes.shape[0], axis=0),
                                           np.tile(gt_boxes, (person_rois.shape[0], 1)),
                                           (1., 1., 1., 1.)).reshape(person_rois.shape[0], gt_boxes.shape[0], -1)
    # human action targets is (person_num: 16, action_num: 26, role_cls: 2, relative_location: 4)
    # don't use np.inf, so that actions without target_objects could kept
    human_action_targets = np.zeros((role_ids.shape[0], role_ids.shape[1],
                                     role_ids.shape[2], 4), dtype=np.float32)
    action_target_weights = np.zeros_like(human_action_targets, dtype=np.float32)
    # get action targets relative location
    human_action_targets[np.where(role_ids > -1)] = \
        targets[np.where(role_ids > -1)[0], role_ids[np.where(role_ids > -1)].astype(int)]
    action_target_weights[np.where(role_ids > -1)] = 1.

    return human_action_targets.reshape(-1, cfg.VCOCO.NUM_ACTION_CLASSES * 2 * 4), \
            action_target_weights.reshape(-1, cfg.VCOCO.NUM_ACTION_CLASSES * 2 * 4)


# ------------------------------- HOI union ------------------------------------

def im_detect_hoi_union(model, boxes, scores, cates, human_count, im_info, blob_conv, entry=None, mode='val', vcoco_heatmaps=None):
    loss = dict(
                interaction_action_loss=None,
                interaction_action_accuray_cls=None)
                
    hoi_blob_in = get_hoi_union_blob_names(is_training=False)

    # im_info.shape = (1, 3)
    im_scale = im_info[0, 2]
    # project boxes to re-sized image size
    scaled_boxes = np.hstack((np.zeros((boxes.shape[0], 1), dtype=boxes.dtype),
                              boxes * im_scale))

    # ToDo: choose top 16 human boxes, top 64 target boxes??
    # ToDo: lower nms thresh, triplet nms
    human_inds = np.where(cates == 1)[0]
    human_boxes = scaled_boxes[human_inds]
    human_scores = scores[human_inds]
    # human_boxes = scaled_boxes[:human_count]
    # human_scores = scores[:human_count]
    # keep_human_inds = np.where(human_scores >= cfg.VCOCO.TEST_HUMAN_SCORE_THRESH)[0][:16]  # ToDo:
    keep_human_inds = np.where(human_scores >= cfg.VCOCO.TEST_HUMAN_SCORE_THRESH)[0]
    human_boxes = human_boxes[keep_human_inds]
    human_scores = human_scores[keep_human_inds]

    # select target objects boxes, all boxes are used as targets, including human
    # ToDo: try different targets number
    # keep_target_inds = np.where(scores >= cfg.VCOCO.TEST_TARGET_OBJECT_SCORE_THRESH)[0][:64]
    keep_target_inds = np.where(scores >= cfg.VCOCO.TEST_TARGET_OBJECT_SCORE_THRESH)[0]
    target_boxes = scaled_boxes[keep_target_inds]
    target_scores = scores[keep_target_inds]
    target_classes = cates[keep_target_inds]

    interaction_human_inds, interaction_object_inds, union_boxes, spatial_info =\
        generate_triplets(human_boxes, target_boxes)
    target_cls_mat = np.zeros((target_boxes.shape[0], cfg.MODEL.NUM_CLASSES)).astype(np.float32)
    target_cls_mat[:, target_classes] = 1.0

    hoi_blob_in['human_boxes'] = human_boxes
    hoi_blob_in['object_boxes'] = target_boxes
    hoi_blob_in['object_classes'] = target_cls_mat
    hoi_blob_in['union_boxes'] = union_boxes
    hoi_blob_in['human_scores'] = human_scores
    hoi_blob_in['object_scores'] = target_scores
    hoi_blob_in['spatial_info'] = spatial_info
    hoi_blob_in['interaction_human_inds'] = interaction_human_inds
    hoi_blob_in['interaction_object_inds'] = interaction_object_inds

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(hoi_blob_in, 'human_boxes')
        _add_multilevel_rois_for_test(hoi_blob_in, 'object_boxes')
        _add_multilevel_rois_for_test(hoi_blob_in, 'union_boxes')
    else:
        blob_conv = blob_conv[-1]

    # if no human box is detected, not use hoi_head, just return nan
    if human_boxes.size > 0:
        rois_keypoints = entry['precomp_keypoints']
        human_keypoints = rois_keypoints[human_inds[keep_human_inds]]

        union_kps, part_boxes, flag = get_pred_keypoints(human_boxes, human_keypoints, interaction_human_inds, im_scale)

        vcoco_heatmaps, union_mask, rescale_kps = generate_joints_heatmap(union_kps, union_boxes[:, 1:]/im_scale,
                                                    human_boxes[interaction_human_inds, 1:]/im_scale,
                                                    target_boxes[interaction_object_inds, 1:]/im_scale)
        pose_configmap = generate_pose_configmap(union_kps, union_boxes[:, 1:]/im_scale,
                                                    human_boxes[interaction_human_inds, 1:]/im_scale,
                                                    target_boxes[interaction_object_inds, 1:]/im_scale)

        hoi_blob_in['union_mask'] = union_mask
        hoi_blob_in['rescale_kps'] = rescale_kps
        hoi_blob_in['part_boxes'] = part_boxes
        hoi_blob_in['flag'] = flag
        hoi_blob_in['poseconfig'] = pose_configmap
        # # Testing. Replace pred action with gt action
        if cfg.DEBUG_TEST_WITH_GT and cfg.DEBUG_TEST_GT_ACTION and entry is not None:
            hoi_blob_out = test_det_bbox_gt_action(hoi_blob_in, entry, im_info)
        else:
            hoi_blob_out = model.module.hoi_net(blob_conv, hoi_blob_in, im_info, vcoco_heatmaps)
        affinity_mat = None
        if entry.get('affinity_mat') is not None:
            affinity_mat = entry['affinity_mat']
            affinity_mat = affinity_mat[human_inds[keep_human_inds]][:, keep_target_inds]

        hoi_res, interaction_affinity_score = hoi_union_res_gather(hoi_blob_out, im_scale, affinity_mat, entry)

        human_action_labels, interaction_action_labels, interaction_affinity_label, \
        total_action_num, recall_action_num, total_affinity_num, recall_affinity_num = \
            get_gt_labels(entry, human_boxes, target_boxes, interaction_human_inds.shape[0], im_scale)

        hoi_blob_out['human_action_labels'] = human_action_labels
        hoi_blob_out['interaction_action_labels'] = interaction_action_labels
        hoi_blob_out['interaction_affinity'] = interaction_affinity_label

        interaction_action_loss, interaction_affinity_loss, \
        interaction_action_accuray_cls, interaction_affinity_cls = model.module.HOI_Head.loss(hoi_blob_out)

        loss = dict(
                    interaction_action_loss=float(interaction_action_loss.cpu()),
                    interaction_action_accuray_cls=float(interaction_action_accuray_cls.cpu()),
                    interaction_affinity_loss=float(interaction_affinity_loss),
                    interaction_affinity_cls=float(interaction_affinity_cls),
                    interaction_affinity_label=interaction_affinity_label,
                    interaction_affinity_score=interaction_affinity_score,
                    total_action_num = total_action_num,
                    total_affinity_num = total_affinity_num,
                    recall_action_num = recall_action_num,
                    recall_affinity_num = recall_affinity_num)
    else:
        # ToDo: any problem here?
        hoi_res = dict(
            agents=np.full((1, 4 + cfg.VCOCO.NUM_ACTION_CLASSES), np.nan),
            roles=np.full((1, 5 * cfg.VCOCO.NUM_ACTION_CLASSES, cfg.VCOCO.NUM_TARGET_OBJECT_TYPES), np.nan),
            roles1=np.full((1, 5 * cfg.VCOCO.NUM_ACTION_CLASSES, cfg.VCOCO.NUM_TARGET_OBJECT_TYPES), np.nan),
        )
    return hoi_res, loss


def get_maxAgent(agents_i, separated_parts_o):
    '''
    given agents_i, choosed best agents by pred score
    N x (4+26) x 3
    '''
    num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
    #num_target_object_types = cfg.VCOCO.NUM_TARGET_OBJECT_TYPES
    agents_i = agents_i.reshape((-1, 4 + num_action_classes, separated_parts_o))
    boxes = agents_i[:,:4, 0]
    scores = agents_i[:, 4:, :] # N x 26 x o
    choose_id = np.argmax(scores, axis=-1) # N x 26
    choose_id_ = choose_id.reshape(-1) #

    scores_ = scores.reshape((-1, separated_parts_o)) # 
    assert scores_.shape[0] == len(choose_id_)

    choosed_score = scores_[np.arange(len(choose_id_)), choose_id_]
    choosed_score = choosed_score.reshape(scores.shape[:-1])

    return np.hstack((boxes, choosed_score)) # N x 30


def get_maxRole(roles_i, separated_parts_o):
    '''
    given roles_i, choose best roles by pred score
    '''

    num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
    num_target_object_types = cfg.VCOCO.NUM_TARGET_OBJECT_TYPES

    roles_i = roles_i.reshape((-1, num_action_classes, 5, num_target_object_types, separated_parts_o))
    role_score = roles_i[:,:,-1:] # N x 26 x 1 x 2 x o
    choose_id = np.argmax(role_score, axis=-1) # N x 26 x 1 x 2

    choose_id = np.tile(choose_id, (1,1,5,1)) # N x 26 x 5 x 2
    choose_id_ = choose_id.reshape(-1)

    roles_i_ = roles_i.reshape((-1, separated_parts_o))
    assert roles_i_.shape[0] == len(choose_id_)

    outs = roles_i_[np.arange(len(choose_id_)), choose_id_] # N 

    return outs.reshape((roles_i.shape[0], num_action_classes*5, num_target_object_types))


def hoi_union_res_gather(hoi_blob, im_scale, interaction_affinity_score=None, entry=None):
    '''
    Convert predicted score and location to triplets
    :param hoi_blob:
    :param im_scale:
    :param entry:
    :return:
    '''
    # ToDo: modify comments
    num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
    num_target_object_types = cfg.VCOCO.NUM_TARGET_OBJECT_TYPES

    # (1) interaction_affinity_score
    interaction_affinity_score = F.sigmoid(hoi_blob['interaction_affinity_score']).cpu().numpy() ##N*1
    # interaction_affinity_score = 1 / (np.exp(-interaction_affinity_score)+1)

    # (2) interaction_action_score
    interaction_action_score = F.sigmoid(hoi_blob['interaction_action_score']).cpu().numpy()  ## N*24

    ## combine interaction_action_score and interaction_affinity_score (1+3)
    interaction_action_score1 = interaction_action_score * interaction_affinity_score

    human_score = hoi_blob['human_scores']
    object_score = hoi_blob['object_scores']

    ## use LIS to human_score
    # human_score = LIS(human_score)
    # object_score = LIS(object_score)

    # scale to original image size when testing
    human_boxes = hoi_blob['human_boxes'][:, 1:] / im_scale
    object_boxes = hoi_blob['object_boxes'][:, 1:] / im_scale

    # we use mask to choose appropriate score
    action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
    # triplet_action_mask = np.tile(action_mask, (human_action_score.shape[0], 1, 1))

    # For actions that do not interact with any object (e.g., smile, run),
    # we rely on s^a_h and the interaction output s^a_h_o is not used,
    # human_action_pair_score = human_score[:, np.newaxis] * human_action_score
    # ToDo: try just use human score as human action pair score
    # we can get better results for `pred bbox and gt action`
    # human_action_pair_score = human_score[:, np.newaxis]

    interaction_score_list = [interaction_action_score, interaction_action_score1]
    role_list = []

    for inter_idx in range((len(interaction_score_list))):

    # in case there is no role-objects
        if hoi_blob['object_boxes'].size > 0:
            # triplets score
            triplet_action_score = interaction_score_list[inter_idx] * \
                                   human_score[hoi_blob['interaction_human_inds']][:, np.newaxis] * \
                                   object_score[hoi_blob['interaction_object_inds']][:, np.newaxis]
            # ToDo: try just use interaction_action_score, better results for `pred bbox and gt action`
            # triplet_action_score = interaction_action_score

            # transform from (human num, object num, action_num) to
            # (human_num*action_num*num_target_object_types, object_num)
            triplet_action_score_tmp = np.zeros(
                (triplet_action_score.shape[0], num_action_classes, num_target_object_types),
                dtype=triplet_action_score.dtype)
            triplet_action_score_tmp[:, np.where(action_mask > 0)[0], np.where(action_mask > 0)[1]] = \
                triplet_action_score
            triplet_action_score = triplet_action_score_tmp
            # interaction_action_score = interaction_action_score_tmp.reshape(human_score.size, object_score.size, -1)
            # interaction_action_score = interaction_action_score.transpose(0, 2, 1).reshape(-1, object_score.size)
            triplet_action_score = triplet_action_score.reshape(human_score.size, object_score.size,
                                                                num_action_classes, num_target_object_types)

            """********** remove misgrouping case before hard nms ************"""
            # triplet_action_score_mask = remove_mis_group(hoi_blob, entry, im_scale)
            # if triplet_action_score_mask is not None:
            #     # ipdb.set_trace()
            #     triplet_action_score = triplet_action_score * triplet_action_score_mask[:,:,None,None]
            # ToDo: one person one action one object
            # ToDo: or one pair three action
            choosed_object_inds = np.argmax(triplet_action_score, axis=1)
            triplet_action_score = np.max(triplet_action_score, axis=1)

            # triplet_action_score[triplet_action_score < 0.3] = -1

            # triplet_action_score[triplet_action_mask == 0] = -1
            """*********  keep top k value **********"""
            # top_k_value = triplet_action_score.flatten()[
            #     np.argpartition(triplet_action_score, -cfg.VCOCO.KEEP_TOP_NUM, axis=None)[-cfg.VCOCO.KEEP_TOP_NUM]]
            # triplet_action_score[triplet_action_score <= top_k_value] = np.nan

            choosed_objects = object_boxes[choosed_object_inds]

        else:
            # if there is no object predicted, triplet action score won't used
            triplet_action_score = np.full((1, num_action_classes, num_target_object_types), np.nan)
            choosed_objects = np.zeros((1, num_action_classes, num_target_object_types, 4))

        # action_score = human_action_pair_score

        # keep consistent with v-coco eval code
        # agents: box coordinates + 26 action score.
        # roles: 26 * (role object coordinates + role-action score) * num_target_object_types
        # agents = np.hstack((human_boxes, action_score))
        agents = np.hstack((human_boxes, np.zeros((human_boxes.shape[0], num_action_classes))))
        roles = np.concatenate((choosed_objects, triplet_action_score[..., np.newaxis]), axis=-1)
        roles = np.stack([roles[:, :, i, :].reshape(-1, num_action_classes * 5) for i in range(num_target_object_types)], axis=-1)

        role_list.append(roles)

    return_dict = dict(
        agents=agents,
        roles=role_list[0],
        roles1=role_list[1],
    )
    return return_dict, interaction_affinity_score


def hoi_union_res_gather_action_first(hoi_blob, im_scale, entry=None):
    """
    A rough try to mitigate targets invisible problem, role_ap1 could achieve 41.5
    :param hoi_blob:
    :param im_scale:
    :param entry:
    :return:
    """
    num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
    num_target_object_types = cfg.VCOCO.NUM_TARGET_OBJECT_TYPES

    human_action_score = F.sigmoid(hoi_blob['human_action_score']).cpu().numpy()
    interaction_action_score = F.sigmoid(hoi_blob['interaction_action_score']).cpu().numpy()

    human_score = hoi_blob['human_scores']
    object_score = hoi_blob['object_scores']
    # scale to original image size when testing
    human_boxes = hoi_blob['human_boxes'][:, 1:] / im_scale
    object_boxes = hoi_blob['object_boxes'][:, 1:] / im_scale

    # we use mask to choose appropriate score
    action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
    triplet_action_mask = np.tile(action_mask, (human_action_score.shape[0], 1, 1))

    # For actions that do not interact with any object (e.g., smile, run),
    # we rely on s^a_h and the interaction output s^a_h_o is not used,
    human_action_pair_score = human_action_score * human_score[:, np.newaxis]

    # in case there is no role-objects
    if hoi_blob['object_boxes'].size > 0:
        # triplets score
        triplet_action_score = interaction_action_score  * \
                               human_score[hoi_blob['interaction_human_inds']][:, np.newaxis] * \
                               object_score[hoi_blob['interaction_object_inds']][:, np.newaxis]
        # transform from (human num, object num, action_num) to
        # (human_num*action_num*num_target_object_types, object_num)
        triplet_action_score_tmp = np.zeros(
            (triplet_action_score.shape[0], num_action_classes, num_target_object_types),
            dtype=triplet_action_score.dtype)
        triplet_action_score_tmp[:, np.where(action_mask > 0)[0], np.where(action_mask > 0)[1]] = \
            triplet_action_score
        triplet_action_score = triplet_action_score_tmp
        # interaction_action_score = interaction_action_score_tmp.reshape(human_score.size, object_score.size, -1)
        # interaction_action_score = interaction_action_score.transpose(0, 2, 1).reshape(-1, object_score.size)
        triplet_action_score = triplet_action_score.reshape(human_score.size, object_score.size,
                                                            num_action_classes, num_target_object_types)
        choosed_object_inds = np.argmax(triplet_action_score, axis=1)
        triplet_action_score = np.max(triplet_action_score, axis=1)
        # triplet_action_score[triplet_action_score < 0.3] = -1

        triplet_topN = cfg.VCOCO.KEEP_TOP_NUM
        # only keep top 7 target bboxes
        triplet_object_topN = 7
        # triplet_action_score[triplet_action_mask == 0] = -1
        top_k_value = triplet_action_score.flatten()[
            np.argpartition(triplet_action_score, -triplet_topN, axis=None)[-triplet_topN]]
        triplet_action_score[triplet_action_score <= top_k_value] = np.nan

        top_k_value_for_object = triplet_action_score.flatten()[
            np.argpartition(triplet_action_score, -triplet_object_topN, axis=None)[-triplet_object_topN]]
        choosed_objects = object_boxes[choosed_object_inds]
        # Other objects except top7 will set nan
        choosed_objects[triplet_action_score < top_k_value_for_object] = np.nan

        # mix human action pair score with triplets score
        # for some actions that targets are invisible, human action score are better than triplets score
        # stack to reshape to (human number, action_classes, targets classes)
        human_centric_score = np.stack((human_action_pair_score, human_action_pair_score), axis=2)
        # Top 25
        human_top_k_value = human_centric_score.flatten()[
            np.argpartition(human_centric_score, -cfg.VCOCO.KEEP_TOP_NUM, axis=None)[-cfg.VCOCO.KEEP_TOP_NUM]]
        human_centric_score[human_centric_score <= human_top_k_value] = 0
        # human action score is the product of two items(human score, action score)
        # multiply 0.75 to approach triplets score
        human_centric_score *= 0.75
        # select maximum score
        triplet_action_score = np.maximum(human_centric_score, triplet_action_score)
        triplet_action_score[triplet_action_score == 0] = np.nan

    else:
        # if there is no object predicted, triplet action score won't used
        triplet_action_score = np.full((1, num_action_classes, num_target_object_types), np.nan)
        choosed_objects = np.zeros((1, num_action_classes, num_target_object_types, 4))

    action_score = human_action_pair_score

    # keep consistent with v-coco eval code
    # agents: box coordinates + 26 action score.
    # roles: 26 * (role object coordinates + role-action score) * num_target_object_types
    agents = np.hstack((human_boxes, action_score))
    roles = np.concatenate((choosed_objects, triplet_action_score[..., np.newaxis]), axis=-1)
    roles = np.stack([roles[:, :, i, :].reshape(-1, num_action_classes * 5) for i in range(num_target_object_types)], axis=-1)

    return_dict = dict(
        # image_id=i
        agents=agents,
        roles=roles
    )
    return return_dict


def generate_triplets(human_boxes, object_boxes):
    human_inds, object_inds = np.meshgrid(np.arange(human_boxes.shape[0]),
                                          np.arange(object_boxes.shape[0]), indexing='ij')
    human_inds, object_inds = human_inds.reshape(-1), object_inds.reshape(-1)

    union_boxes = box_utils.get_union_box(human_boxes[human_inds][:, 1:],
                                          object_boxes[object_inds][:, 1:])
    union_boxes = np.hstack((np.zeros((union_boxes.shape[0], 1), dtype=union_boxes.dtype), union_boxes))
    spatial_info = box_utils.bbox_transform_inv(human_boxes[human_inds][:, 1:],
                                                object_boxes[object_inds][:, 1:])

    return human_inds, object_inds, union_boxes, spatial_info


# --------------------- Check bottleneck ---------------------------


def test_det_bbox_gt_action(hoi_blob_in, entry, im_info):
    # check interaction branch, bbox res from test, interaction from gt
    gt_human_inds = np.where(entry['gt_classes'] == 1)[0]
    gt_human_boxes = entry['boxes'][gt_human_inds]

    pred_human_boxes = hoi_blob_in['human_boxes']/im_info[0, 2]
    human_pred_gt_overlaps = box_utils.bbox_overlaps(
                pred_human_boxes[:, 1:].astype(dtype=np.float32, copy=False),
                gt_human_boxes.astype(dtype=np.float32, copy=False))
    human_pred_to_gt_inds = np.argmax(human_pred_gt_overlaps, axis=1)
    human_ious = human_pred_gt_overlaps.max(axis=1)[:, None]
    human_score = np.zeros(human_ious.shape)
    human_score[np.where(human_ious > 0.5)] = 1

    # assign gt interaction to mapping pred bboxes
    human_action = entry['gt_actions'][gt_human_inds[human_pred_to_gt_inds]]
    # multiply iou to human action, better localization better action score
    # human_action = human_ious * human_action
    human_action = human_score * human_action

    # ------------------------------- Targets -----------------------------------
    # ipdb.set_trace()
    pred_target_boxes = hoi_blob_in['object_boxes']/im_info[0, 2]
    target_pred_gt_overlaps = box_utils.bbox_overlaps(
                pred_target_boxes[:, 1:].astype(dtype=np.float32, copy=False),
                entry['boxes'].astype(dtype=np.float32, copy=False))
    target_pred_to_gt_inds = np.argmax(target_pred_gt_overlaps, axis=1)
    target_ious = target_pred_gt_overlaps.max(axis=1)[:, None]
    target_score = np.zeros(target_ious.shape)
    target_score[np.where(target_ious > 0.5)] = 1

    gt_action_mat = generate_action_mat(entry['gt_role_id'])
    # ToDo: there is a problem, here we ignore `interaction triplets` that
    # targets is invisible
    action_labels = gt_action_mat[gt_human_inds[human_pred_to_gt_inds[hoi_blob_in['interaction_human_inds']]],
                                  target_pred_to_gt_inds[hoi_blob_in['interaction_object_inds']]]
    # triplet_ious = human_ious[hoi_blob_in['interaction_human_inds']] * \
    #                 target_ious[hoi_blob_in['interaction_object_inds']]
    # # multiply iou
    # action_labels = triplet_ious[:, None] * action_labels

    triplet_scores = human_score[hoi_blob_in['interaction_human_inds']] * \
                    target_score[hoi_blob_in['interaction_object_inds']]
    action_labels = triplet_scores[:, None] * action_labels

    # convert to 24-class
    interaction_action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
    action_labels = action_labels[:, np.where(interaction_action_mask > 0)[0], np.where(interaction_action_mask > 0)[1]]

    hoi_blob_in['human_action_score'] = torch.from_numpy(human_action).float().cuda()
    hoi_blob_in['interaction_action_score'] = torch.from_numpy(action_labels).float().cuda()

    return hoi_blob_in


def remove_mis_group(hoi_blob_in, entry, im_scale):
    gt_human_inds = np.where(entry['gt_classes'] == 1)[0]
    gt_human_boxes = entry['boxes'][gt_human_inds]

    pred_human_boxes = hoi_blob_in['human_boxes'][:, 1:]/im_scale
    # if len(pred_human_boxes[0]) == 0:
    #     return None
    human_pred_gt_overlaps = box_utils.bbox_overlaps(
                pred_human_boxes.astype(dtype=np.float32, copy=False),
                gt_human_boxes.astype(dtype=np.float32, copy=False))

    human_pred_to_gt_inds = np.argmax(human_pred_gt_overlaps, axis=1)
    human_ious = human_pred_gt_overlaps.max(axis=1)[:, None]
    valid_human_ind = np.where(human_ious > 0.5)[0]

    # ------------------------------- Targets -----------------------------------
    # ipdb.set_trace()
    pred_obj_boxes = hoi_blob_in['object_boxes'][:, 1:]/im_scale
    obj_pred_gt_overlaps = box_utils.bbox_overlaps(
                pred_obj_boxes.astype(dtype=np.float32, copy=False),
                entry['boxes'].astype(dtype=np.float32, copy=False))
    obj_pred_to_gt_inds = np.argmax(obj_pred_gt_overlaps, axis=1)
    obj_ious = obj_pred_gt_overlaps.max(axis=1)[:, None]
    valid_obj_ind = np.where(obj_ious > 0.5)[0]

    interact_matrix = np.zeros([pred_human_boxes.shape[0], pred_obj_boxes.shape[0]])
    interact_matrix[hoi_blob_in['interaction_human_inds'], hoi_blob_in['interaction_object_inds']] = 1
    valid_matrix = np.zeros([pred_human_boxes.shape[0], pred_obj_boxes.shape[0]]) - 1
    valid_matrix[valid_human_ind, :] += 1
    valid_matrix[:, valid_obj_ind] += 1
    valid_matrix = valid_matrix * interact_matrix
    valid_interaction_human_inds, valid_interaction_obj_inds = np.where(valid_matrix==1)

    gt_action_mat = generate_action_mat(entry['gt_role_id'])
    # ToDo: there is a problem, here we ignore `interaction triplets` that
    # targets is invisible
    action_labels = gt_action_mat[
        gt_human_inds[human_pred_to_gt_inds[valid_interaction_human_inds]],
        obj_pred_to_gt_inds[valid_interaction_obj_inds]]
    # action_labels = action_labels.reshape(action_labels.shape[0], -1)
    no_gt_rel_ind = np.where(action_labels.sum(1).sum(1) == 0)

    ret = np.ones([pred_human_boxes.shape[0], pred_obj_boxes.shape[0]])
    ret[valid_interaction_human_inds[no_gt_rel_ind],
        valid_interaction_obj_inds[no_gt_rel_ind]] = 0
    return ret


def get_gt_labels(entry, human_boxes, obj_boxes, interaction_pair_num, im_scale):
    gt_human_inds = np.where(entry['gt_classes'] == 1)[0]
    gt_human_boxes = entry['boxes'][gt_human_inds]

    gt_boxes = entry['boxes']
    gt_action_mat = generate_action_mat(entry['gt_role_id'])
    tmp_gt_action_mat = np.zeros(gt_action_mat.shape)
    human_action_labels = entry['gt_actions']

    human_to_gt_ov = box_utils.bbox_overlaps(
        (human_boxes[:, 1:]/im_scale).astype(dtype=np.float32, copy=False),
        gt_human_boxes.astype(dtype=np.float32, copy=False))
    obj_to_gt_ov = box_utils.bbox_overlaps(
        (obj_boxes[:, 1:]/im_scale).astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))

    interaction_labels = np.zeros((interaction_pair_num, 24))
    interaction_affinity = np.zeros(interaction_pair_num)
    interaction_action_mask = np.array(cfg.VCOCO.ACTION_MASK).T

    human_to_gt_max_ov = human_to_gt_ov.max(axis=1)
    obj_to_gt_max_ov = obj_to_gt_ov.max(axis=1)
    human_to_gt_inds = human_to_gt_ov.argmax(axis=1)
    obj_to_gt_inds = obj_to_gt_ov.argmax(axis=1)

    human_valid = np.where(human_to_gt_max_ov >= 0.5)[0]
    obj_valid = np.where(obj_to_gt_max_ov >= 0.5)[0]

    action_label = human_action_labels[gt_human_inds[human_to_gt_inds]]

    if len(human_valid) > 0 and len(obj_valid) > 0:
        valid_interaction_human_ind, valid_interaction_obj_ind = np.meshgrid(human_valid, obj_valid, indexing='ij')
        valid_interaction_human_ind, valid_interaction_obj_ind = \
            valid_interaction_human_ind.reshape(-1), valid_interaction_obj_ind.reshape(-1)
        valid_mask = np.zeros((human_boxes.shape[0], obj_boxes.shape[0]))
        valid_mask[valid_interaction_human_ind, valid_interaction_obj_ind] = 1
        valid_ind = np.where(valid_mask.reshape(-1) > 0)[0]

        valid_human_to_gt_box_ind = gt_human_inds[human_to_gt_inds[valid_interaction_human_ind]]
        valid_obj_to_gt_box_ind = obj_to_gt_inds[valid_interaction_obj_ind]

        valid_interaction_labels = gt_action_mat[valid_human_to_gt_box_ind, valid_obj_to_gt_box_ind]
        # ipdb.set_trace()
        tmp_gt_action_mat[valid_human_to_gt_box_ind, valid_obj_to_gt_box_ind] = 1
        tmp_gt_action_mat *= gt_action_mat
        # convert to 24-class
        valid_interaction_labels = valid_interaction_labels[:, np.where(interaction_action_mask > 0)[0],
                                   np.where(interaction_action_mask > 0)[1]]
        valid_interaction_affinity = np.any(valid_interaction_labels.reshape(valid_interaction_labels.shape[0], -1) > 0, 1)

        interaction_labels[valid_ind] = valid_interaction_labels
        interaction_affinity[valid_ind] = valid_interaction_affinity

    # gt_action_num = np.sum(interaction_labels, 1)
    # ipdb.set_trace()
    gt_affinity_mat = gt_action_mat.sum(-1).sum(-1)
    tmp_gt_affinity_mat = tmp_gt_action_mat.sum(-1).sum(-1)
    total_affinity_num = np.where(gt_affinity_mat > 0)[0].shape[0]
    recall_affinity_num = np.where(tmp_gt_affinity_mat > 0)[0].shape[0]

    total_action_num = np.sum(gt_action_mat)
    recall_action_num = np.sum(tmp_gt_action_mat)
    return action_label, interaction_labels, interaction_affinity, total_action_num, recall_action_num, \
            total_affinity_num, recall_affinity_num


def get_gt_keypoints(entry, human_boxes, interaction_human_inds, im_scale):
    gt_human_inds = np.where(entry['gt_classes'] == 1)[0]
    gt_human_boxes = entry['boxes'][gt_human_inds]
    human_to_gt_ov = box_utils.bbox_overlaps(
        (human_boxes[:, 1:]/im_scale).astype(dtype=np.float32, copy=False),
        gt_human_boxes.astype(dtype=np.float32, copy=False))
    human_to_gt_inds = human_to_gt_ov.argmax(axis=1)
    human_to_gt_box_ind = gt_human_inds[human_to_gt_inds[interaction_human_inds]]
    gt_keypoints = entry['gt_keypoints'][human_to_gt_box_ind]
    return gt_keypoints


def get_pred_keypoints(human_boxes, human_keypoints, interaction_human_inds, im_scale):
    assert human_boxes.shape[0] == human_keypoints.shape[0]
    union_kps = human_keypoints[interaction_human_inds]
    if cfg.VCOCO.USE_KPS17:
        part_boxes, flag = generate_part_box_from_kp17(human_keypoints, human_boxes, im_scale,
                                                       body_ratio=cfg.VCOCO.BODY_RATIO, head_ratio=1.5)
    else:
        part_boxes, flag = generate_part_box_from_kp(human_keypoints, human_boxes, im_scale,
                                                     body_ratio=cfg.VCOCO.BODY_RATIO, head_ratio=1.5)
    return union_kps, part_boxes, flag