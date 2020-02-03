from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.random as npr
import ipdb

from core.config import cfg
import utils.boxes as box_utils
import utils.blob as blob_utils
from utils.fpn import distribute_rois_over_fpn_levels
from utils.keypoints import keypoints_to_heatmap_labels
# from core.test import box_results_with_nms_and_limit

'''
Sampling samples for three branches of InteractNet.
'''
# ToDo: check code again, better organization


def get_hoi_blob_names(is_training=True):

    # use `boxes` same as paper to distinguish from rois proposed by rpn,
    # when testing, boxes is refined rois by fast-rcnn head
    # when training, boxes is just sampled from rois
    #
    # human_index store index of acting-human in boxes, the same as target_object_index.
    # interaction_human_inds: store index of acting-human
    # interaction_target_object_inds: index of target_object

    blob_names = ['boxes', 'human_index', 'target_object_index',
                  'interaction_human_inds', 'interaction_target_object_inds',
                  'interaction_batch_idx']
    if is_training:
        blob_names += ['human_action_labels', 'human_action_targets',
                       'action_target_weights', 'interaction_action_labels']
    else:
        blob_names += ['scores']

    if cfg.VCOCO.KEYPOINTS_ON:
        blob_names += ['keypoint_rois', 'keypoint_locations_int32',
                       'keypoint_weights', 'keypoint_loss_normalizer']

    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['boxes_fpn' + str(lvl)]
        blob_names += ['boxes_idx_restore_int32']
        if cfg.VCOCO.KEYPOINTS_ON:
            for lvl in range(k_min, k_max + 1):
                blob_names += ['keypoint_rois_fpn' + str(lvl)]

    blob_names = {k: [] for k in blob_names}
    return blob_names


def sample_for_detection_branch(rpn_ret):
    '''
    This function will only be used when training,  be CAREFUL use it AFTER
    sample_for_hoi_branch, as this will change rpn_net inplace.

    Targets have been calculated and classes have been assigned to rois
    in function *add_fast_rcnn_blobs* in roi_data/fast_rcnn.py
    So we just have to sample small set from rpn_ret for detection branch
    :param rpn_ret:
        rpn_cls_*, rpn_bbox_pred_*: scores and bbox given by fpn_rpn_head,
                                    used to calculate rpn loss, and produce proposals
        rpn_rois*: proposals and scores extracted from rpn_cls_* and rpn_bbox_pred_*,
                   produced by generate proposals op.
                   Used in CollectAndDistributeFpnRpnProposalsOp to collect and select
                   proposals used in next stage.
        rois_fpn*: produced by function add_multilevel_roi_blobs in utils/fpn.py,
                   used in X-transform to get pooled feature of different feature layer.
        rois, label_*, bbox_targets, *_weights: what needed in Fast-RCNN
        rois_to_gt_ind_map, rois_max_overlaps: Used to sampling for Human branch and
                                               Interaction branch

    :return:
    '''
    rois_per_image = int(cfg.VCOCO.TRAIN_BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(np.round(cfg.VCOCO.TRAIN_FG_FRACTION * rois_per_image))

    def sample_for_one_image(labels):
        # Select foreground RoIs as those labels is nonzero
        fg_inds = np.where(labels > 0)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs
        bg_inds = np.where(labels == 0)[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
        # Sample foreground regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)
        return keep_inds

    keep_inds = []
    for im_i in np.unique(rpn_ret['rois'][:, 0]):
        labels = -np.ones_like(rpn_ret['labels_int32'])
        labels[rpn_ret['rois'][:, 0] == im_i] = rpn_ret['labels_int32'][rpn_ret['rois'][:, 0] == im_i]
        keep_ind_this_im = sample_for_one_image(labels)
        keep_inds.append(keep_ind_this_im)
    keep_inds = np.concatenate(keep_inds)

    # we only need these keywords for Fast-RCNN
    keys_for_rcnn = ['rois', 'labels_int32', 'bbox_targets', 'bbox_inside_weights', 'bbox_outside_weights']
    for k in keys_for_rcnn:
        rpn_ret[k] = rpn_ret[k][keep_inds]

    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        distribute_rois_over_fpn_levels(rpn_ret, 'rois')


def sample_for_hoi_branch(rpn_net, roidb, im_info,
                          cls_score=None, bbox_pred=None, is_training=True):
    hoi_blob_names = get_hoi_blob_names(is_training=is_training)
    if is_training:
        # list of sample result
        blobs_list = sample_for_hoi_branch_train(rpn_net, roidb, im_info)
    else:
        raise NotImplementedError
        # NOT USED
        # assert cls_score is not None and bbox_pred is not None
        # blobs_list = sample_for_hoi_branch_test(cls_score, bbox_pred, rpn_net['rois'], im_info)

    hoi_blob_in = merge_hoi_blobs(hoi_blob_names, blobs_list)
    return hoi_blob_in


def merge_hoi_blobs(hoi_blob_in, blobs_list):
    '''
    Merge blob of each image
    :param hoi_blob_in: hoi blob names dict
    :param blobs_list: blob of each image
    :return:
    '''
    # Store human inds of sampled rois temporarily
    hoi_blob_in['human_inds_of_sampled_boxes'] = []

    # support mini-batch
    human_boxes_count = 0
    target_object_boxes_count = 0
    for i in range(len(blobs_list)):
        blob_this_im = blobs_list[i]
        # ensure interaction_*_inds only index right image's human/target_object feature
        blob_this_im['interaction_human_inds'] += human_boxes_count
        blob_this_im['interaction_target_object_inds'] += target_object_boxes_count
        # count human/object rois num
        human_boxes_count += blob_this_im['human_inds_of_sampled_boxes'].sum()
        target_object_boxes_count += (blob_this_im['human_inds_of_sampled_boxes'].size -
                                   blob_this_im['human_inds_of_sampled_boxes'].sum())
        # Append to blob list
        for k, v in blob_this_im.items():
            hoi_blob_in[k].append(v)

    # Concat the training blob lists into tensors
    # np.concatenate default axis=0
    for k, v in hoi_blob_in.items():
        if len(v) > 0:
            hoi_blob_in[k] = np.concatenate(v)
    # get human_index and target_object_index from human_inds_of_sampled_boxes,
    # and delete human_inds_of_sample_rois
    hoi_blob_in['human_index'] = np.where(hoi_blob_in['human_inds_of_sampled_boxes'] == 1)[0]
    hoi_blob_in['target_object_index'] = np.where(
        hoi_blob_in['human_inds_of_sampled_boxes'] == 0)[0]
    del hoi_blob_in['human_inds_of_sampled_boxes']

    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        distribute_rois_over_fpn_levels(hoi_blob_in, 'boxes')
        if cfg.VCOCO.KEYPOINTS_ON:
            distribute_rois_over_fpn_levels(hoi_blob_in, 'keypoint_rois')

    return hoi_blob_in


def _compute_action_targets(person_rois, gt_boxes, role_ids):
    '''
    Compute action targets
    :param person_rois: rois assigned to gt acting-human, n * 4
    :param gt_boxes: all gt boxes in one image
    :param role_ids: person_rois_num * action_cls_num * NUM_TARGET_OBJECT_TYPES,
                     store person rois corresponding role object ids.
    :return:
    '''
    assert person_rois.shape[0] == role_ids.shape[0]
    # ToDo: should use cfg.MODEL.BBOX_REG_WEIGHTS?
    # calculate targets between every person rois and every gt_boxes
    targets = box_utils.bbox_transform_inv(np.repeat(person_rois, gt_boxes.shape[0], axis=0),
                                           np.tile(gt_boxes, (person_rois.shape[0], 1)),
                                           (1., 1., 1., 1.)).reshape(person_rois.shape[0], gt_boxes.shape[0], -1)
    # human action targets is (person_num: 16, action_num: 26, role_cls: 2, relative_location: 4)
    human_action_targets = np.zeros((role_ids.shape[0], role_ids.shape[1],
                                     role_ids.shape[2], 4), dtype=np.float32)
    action_target_weights = np.zeros_like(human_action_targets, dtype=np.float32)
    # get action targets relative location
    human_action_targets[np.where(role_ids > -1)] = \
        targets[np.where(role_ids > -1)[0], role_ids[np.where(role_ids > -1)].astype(int)]
    action_target_weights[np.where(role_ids > -1)] = 1.

    return human_action_targets.reshape(-1, cfg.VCOCO.NUM_ACTION_CLASSES * cfg.VCOCO.NUM_TARGET_OBJECT_TYPES * 4), \
            action_target_weights.reshape(-1, cfg.VCOCO.NUM_ACTION_CLASSES * cfg.VCOCO.NUM_TARGET_OBJECT_TYPES * 4)


def _sample_human_object(rois, rois_to_gt_ind, roidb, im_info):
    '''
    Sample human rois and target_object rois
    :param rois: rois correspond to feature map
    :param rois_to_gt_ind:
    :param roidb: box correspond to origin image
    :return:
    '''
    # ipdb.set_trace()
    human_num_per_image = int(cfg.VCOCO.HUMAN_NUM_PER_IM)
    target_object_num_per_image = int(cfg.VCOCO.TARGET_OBJECT_NUM_PER_IM)
    kp_human_num_per_image = int(cfg.VCOCO.KP_HUMAN_NUM_PER_IM)

    # Add keypoints
    all_human_gt_inds = np.where(roidb['gt_classes'] == 1)[0]
    gt_keypoints = roidb['gt_keypoints']

    # get gt human ids that with action
    # ToDo: name change
    # add all human(even without action) to human-centric branch
    human_with_action_gt_inds = np.where(roidb['gt_actions'][:, 0] >= 0)[0]
    gt_objects_num = roidb['gt_actions'].shape[0]
    # human_with_action_gt_inds = np.where(roidb['gt_classes'][:gt_objects_num] == 1)[0]
    # gt_boxes, for calculating action targets location
    # roidb['boxes'] = gt_boxes + scaled_rois(from RPN module)
    # ipdb.set_trace()
    gt_boxes = roidb['boxes'][:gt_objects_num, :]

    # -------------------------------------------------------------------------
    # Human-Centric Branch: sample human rois and calculate targets
    # -------------------------------------------------------------------------

    # get proposals(rois) that assigned to gt human with action
    # and corresponding target_objects
    rois_human_with_action_inds = []
    rois_human_without_action_inds = []

    for human_gt_i in all_human_gt_inds:
        if human_gt_i in human_with_action_gt_inds:
            rois_human_with_action_inds.append(np.where(rois_to_gt_ind == human_gt_i)[0])
        else:
            rois_human_without_action_inds.append(np.where(rois_to_gt_ind == human_gt_i)[0])

    rois_human_with_action_inds = np.concatenate(rois_human_with_action_inds)

    # select 16 rois of human
    human_num_this_image = min(human_num_per_image, rois_human_with_action_inds.size)
    if rois_human_with_action_inds.size > 0:
        rois_human_with_action_inds = npr.choice(
            rois_human_with_action_inds, size=human_num_this_image, replace=False)

    if cfg.VCOCO.KEYPOINTS_ON:
        if len(rois_human_without_action_inds) > 0:
            rois_human_without_action_inds = np.concatenate(rois_human_without_action_inds)
            human_num_without_action = min(kp_human_num_per_image-rois_human_with_action_inds.size,
                                           rois_human_without_action_inds.size)
            rois_human_without_action_inds = npr.choice(
                rois_human_without_action_inds, size=human_num_without_action, replace=False)
            rois_kp_inds = np.concatenate([rois_human_with_action_inds, rois_human_without_action_inds])
            kp_inds_of_sampled_rois = np.zeros(rois_kp_inds.size, dtype=np.int32)
            kp_inds_of_sampled_rois[:rois_human_with_action_inds.size] = 1
        else:
            rois_kp_inds = rois_human_with_action_inds
            kp_inds_of_sampled_rois = np.ones(rois_human_with_action_inds.size, dtype=np.int32)

        sampled_kp_rois = rois[rois_kp_inds]
        sampled_keypoints = gt_keypoints[rois_to_gt_ind[rois_kp_inds]]
        heats, kp_weights = keypoints_to_heatmap_labels(
            sampled_keypoints, sampled_kp_rois[:, 1:]/float(im_info[2]))

        shape = (sampled_kp_rois.shape[0] * gt_keypoints.shape[2],)
        heats = heats.reshape(shape)
        kp_weights = kp_weights.reshape(shape)


        min_count = cfg.KRCNN.MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH
        num_visible_keypoints = np.sum(kp_weights)
        kp_norm = num_visible_keypoints / (
            cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.BATCH_SIZE_PER_IM * cfg.TRAIN.
            FG_FRACTION * cfg.KRCNN.NUM_KEYPOINTS)

    # get human action targets relative location
    human_rois = rois[rois_human_with_action_inds]
    human_action_labels = roidb['gt_actions'][rois_to_gt_ind[rois_human_with_action_inds]]
    human_action_labels[human_action_labels < 0] = 0

    rois_human_role_ids = roidb['gt_role_id'][rois_to_gt_ind[rois_human_with_action_inds]]
    # scale rois to original image size
    human_action_targets, action_target_weights = \
        _compute_action_targets(human_rois[:, 1:]/float(im_info[2]), gt_boxes, rois_human_role_ids)

    # -------------------------------------------------------------------------
    # Interaction Branch: sample target_object rois and sample positive triplets
    # -------------------------------------------------------------------------

    # Select role objects
    #
    # get gt role object inds
    target_object_gt_inds = np.unique(rois_human_role_ids)
    target_object_gt_inds = target_object_gt_inds[np.where(target_object_gt_inds > -1)]

    # get rois that assigned to gt role object
    if target_object_gt_inds.size > 0:
        rois_target_object_inds = []
        for role_gt_i in target_object_gt_inds:
            rois_target_object_inds.append(np.where(rois_to_gt_ind == role_gt_i)[0])
        rois_target_object_inds = np.concatenate(rois_target_object_inds)
    else:
        # some actions don't have target_objects
        rois_target_object_inds = np.empty((0,), dtype=np.int64)

    # select 32 role objects
    # ToDo: 32 or no limitation?
    # min(target_object_num_per_image, rois_target_object_inds.size)
    target_object_num_this_image = rois_target_object_inds.size
    if rois_target_object_inds.size > 0:
        rois_target_object_inds = npr.choice(
            rois_target_object_inds, size=target_object_num_this_image, replace=False)
    target_object_rois = rois[rois_target_object_inds]
    # target_object_feature_mapping_index = mapping_original_inds[rois_target_object_inds]

    # Sample positive triplets
    #
    human_rois_inds, target_object_rois_inds, interaction_action_labels = \
        generate_positive_triplets(rois_human_with_action_inds, rois_target_object_inds,
                                   rois_to_gt_ind, roidb['gt_role_id'])
    interaction_batch_idx = np.full_like(human_rois_inds, rois[0, 0], dtype=np.int32)

    sampled_rois = np.vstack((human_rois, target_object_rois))
    human_inds_of_sampled_rois = np.zeros(sampled_rois.shape[0], dtype=np.int32)
    human_inds_of_sampled_rois[:human_rois.shape[0]] = 1

    if not cfg.VCOCO.KEYPOINTS_ON:
        return_dict = dict(
            boxes=sampled_rois,
            human_inds_of_sampled_boxes=human_inds_of_sampled_rois,
            human_action_labels=human_action_labels,
            human_action_targets=human_action_targets,
            action_target_weights=action_target_weights,
            interaction_human_inds=human_rois_inds,
            interaction_target_object_inds=target_object_rois_inds,
            interaction_action_labels=interaction_action_labels,
            interaction_batch_idx=interaction_batch_idx
        )
    else:
        return_dict = dict(
            boxes=sampled_rois,
            human_inds_of_sampled_boxes=human_inds_of_sampled_rois,
            human_action_labels=human_action_labels,
            human_action_targets=human_action_targets,
            action_target_weights=action_target_weights,
            interaction_human_inds=human_rois_inds,
            interaction_target_object_inds=target_object_rois_inds,
            interaction_action_labels=interaction_action_labels,
            interaction_batch_idx=interaction_batch_idx,
            keypoint_rois=sampled_kp_rois,
            keypoint_locations_int32=heats.astype(np.int32, copy=False),
            keypoint_weights=kp_weights,
            keypoint_loss_normalizer=np.array([kp_norm], dtype=np.float32),
        )
    return return_dict


def sample_for_hoi_branch_train(rpn_ret, roidb, im_info):
    '''
    hoi: human-object interaction
    Sampling for human-centric branch and interaction branch
    :param rpn_ret:
    :param roidb:
    :return:
    '''
    # Select proposals(rois) that IoU with gt >= 0.5 for human-centric branch
    # and interaction branch
    keep_rois_inds = np.where(rpn_ret['rois_max_overlaps'] >= cfg.TRAIN.FG_THRESH)
    rois = rpn_ret['rois'][keep_rois_inds]
    rois_to_gt_ind = rpn_ret['rois_to_gt_ind_map'][keep_rois_inds]

    train_hoi_blobs = []
    # get blobs of each image
    for i, entry in enumerate(roidb):
        inds_of_this_image = np.where(rois[:, 0] == i)
        blob_this_im = _sample_human_object(rois[inds_of_this_image],
                                            rois_to_gt_ind[inds_of_this_image],
                                            entry,
                                            im_info[i])
        train_hoi_blobs.append(blob_this_im)

    return train_hoi_blobs


def sample_for_hoi_branch_precomp_box_train(roidb, im_info, is_training=True):
    '''
    hoi: human-object interaction
    Sampling for human-centric branch and interaction branch
    :param rpn_ret:
    :param roidb:
    :return:
    '''
    # Select proposals(rois) that IoU with gt >= 0.5 for human-centric branch
    # and interaction branch
    hoi_blob_names = get_hoi_blob_names(is_training=is_training)
    scales = im_info.data.numpy()[:, 2]

    train_hoi_blobs = []
    # get blobs of each image
    for i, entry in enumerate(roidb):
        keep_rois_inds = np.where(entry['max_overlaps'] >= cfg.TRAIN.FG_THRESH)
        # rois is based on enlarged iamge
        rois = entry['boxes'][keep_rois_inds] * scales[i]
        rois = np.concatenate([np.full((keep_rois_inds[0].shape[0], 1),i), rois], 1).astype(np.float32)
        rois_to_gt_ind = entry['box_to_gt_ind_map'][keep_rois_inds]

        blob_this_im = _sample_human_object(rois,
                                            rois_to_gt_ind,
                                            entry,
                                            im_info[i])
        train_hoi_blobs.append(blob_this_im)
    hoi_blob_in = merge_hoi_blobs(hoi_blob_names, train_hoi_blobs)
    return hoi_blob_in


# def _get_refined_boxes(cls_score, bbox_pred, rois, im_info):
#     '''
#     The same as faster-rcnn test code,
#     refine boxes, apply nms and score threshold
#     :param cls_score:
#     :param bbox_pred:
#     :param rois:
#     :param im_info:
#     :return:
#     '''
#     num_classes = cfg.MODEL.NUM_CLASSES
#
#     # unscale back to raw image space
#     boxes = rois[:, 1:5] / im_info[2]
#     batch_idx = rois[0, 0]
#
#     # cls prob (activations after softmax)
#     scores = cls_score.squeeze()
#     # In case there is 1 proposal
#     scores = scores.reshape([-1, scores.shape[-1]])
#
#     if cfg.TEST.BBOX_REG:
#         # Apply bounding-box regression deltas
#         box_deltas = bbox_pred.squeeze()
#         # In case there is 1 proposal
#         box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
#         if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
#             # (legacy) Optionally normalize targets by a precomputed mean and stdev
#             box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
#                          + cfg.TRAIN.BBOX_NORMALIZE_MEANS
#         pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
#         # clip according to raw image size
#         pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im_info[:2] / im_info[2])
#     else:
#         # Simply repeat the boxes, once for each class
#         pred_boxes = np.tile(boxes, (1, scores.shape[1]))
#
#     # apply nms with IoU threshold > 0.3 and scores more than 0.05,
#     # the same as faster-rcnn parameter
#     # ToDo: reset cfg.TEST.DETECTIONS_PER_IM to zero
#     _, _, cls_boxes_score = box_results_with_nms_and_limit(scores, pred_boxes)
#
#     # concat and boxes as input to human-centric branch and interaction branch
#     refined_boxes = np.vstack([cls_boxes_score[j] for j in range(1, num_classes)])[:, :-1]
#     # scale refined boxes to input image size
#     refined_boxes *= im_info[2]
#     # attach batch idx
#     refined_boxes = np.hstack((np.full((refined_boxes.shape[0], 1), batch_idx,
#                                        dtype=refined_boxes.dtype), refined_boxes))
#     # store human inds in refined boxes
#     human_boxes_inds = np.zeros(refined_boxes.shape[0], dtype=np.int32)
#     human_boxes_inds[:cls_boxes_score[1].shape[0]] = 1
#
#     # get interaction inds
#     interaction_human_inds = np.where(human_boxes_inds == 1)[0]
#     interaction_target_object_inds = np.where(human_boxes_inds == 0)[1]
#     # get all possible triplets
#     interaction_human_inds, interaction_target_object_inds \
#         = np.repeat(interaction_human_inds, interaction_target_object_inds.size), \
#            np.tile(interaction_target_object_inds, interaction_target_object_inds)
#     interaction_batch_idx = np.full_like(interaction_human_inds, batch_idx, dtype=np.int32)
#
#     # ToDo: add score
#     # ToDo: interaction_target_object_inds is wrong
#     # ToDo: check boxes size is correct, or not
#     return_dict = dict(
#         boxes=refined_boxes,
#         human_inds_of_sampled_boxes=human_boxes_inds,
#         interaction_human_inds=interaction_human_inds,
#         interaction_target_object_inds=interaction_target_object_inds,
#         interaction_batch_idx=interaction_batch_idx
#     )
#
#     return return_dict
#

# def sample_for_hoi_branch_test(cls_score, bbox_pred, rois, im_info):
#     # ToDo: rois or refined boxes as new proposal for hoi branch
#     # scores have been soft-max, if not training
#     im_info = im_info.data.numpy()
#     cls_score = cls_score.data.cpu().numpy()
#     bbox_pred = bbox_pred.data.cpu().numpy()
#
#     test_blot_list = []
#     for im_i in np.unique(rois[:, 0]):
#         # get rois/score/bbox of this image
#         inds_this_im = np.where(rois[:, 0] == im_i)
#         blob_this_im = _get_refined_boxes(cls_score=cls_score[inds_this_im],
#                                           bbox_pred=bbox_pred[inds_this_im],
#                                           rois=rois[inds_this_im],
#                                           im_info=im_info[im_i])
#         test_blot_list.append(blob_this_im)
#
#     return test_blot_list


def generate_positive_triplets(rois_human_with_action_inds,
                               rois_target_object_inds, rois_to_gt_ind, gt_role_id):
    human_rois_to_gt_ind = rois_to_gt_ind[rois_human_with_action_inds]
    target_object_rois_to_gt_ind = rois_to_gt_ind[rois_target_object_inds]

    # Store inds of sampled human rois and role object rois
    human_rois_inds = np.arange(rois_human_with_action_inds.size, dtype=np.int32)
    target_object_rois_inds = np.arange(rois_target_object_inds.size, dtype=np.int32)

    # generate gt action mat from gt_role_id, see generate_action_mat
    gt_action_mat = generate_action_mat(gt_role_id)
    # repeat and tile to generate all possible triplets
    human_rois_inds, target_object_rois_inds = \
        np.repeat(human_rois_inds, target_object_rois_inds.size), np.tile(target_object_rois_inds, human_rois_inds.size)
    # get action cls of all triplets
    action_labels = gt_action_mat[human_rois_to_gt_ind[human_rois_inds],
                                  target_object_rois_to_gt_ind[target_object_rois_inds]]

    # I think triplets batch size don't need to be fixed, as only need to do
    # element-wise add, so use all positive triplets for training.
    # keep positive triplets for training
    positive_triplet_inds = np.where(np.sum(action_labels, axis=1) > 0)
    human_rois_inds = human_rois_inds[positive_triplet_inds]
    target_object_rois_inds = target_object_rois_inds[positive_triplet_inds]
    action_labels = action_labels[positive_triplet_inds]

    return human_rois_inds, target_object_rois_inds, action_labels


def generate_action_mat(gt_role_id):
    '''
    Generate a matrix to store action triplet
    :param gt_role_id:
    :return: action_mat, row is person id, column is target_object id,
             third axis is action id
    '''
    mat = np.zeros((gt_role_id.shape[0], gt_role_id.shape[0], cfg.VCOCO.NUM_ACTION_CLASSES), dtype=np.float32)
    role_ids = gt_role_id[np.where(gt_role_id > -1)]
    human_ids, action_cls, _ = np.where(gt_role_id > -1)
    assert role_ids.size == human_ids.size == action_cls.size
    mat[human_ids, role_ids, action_cls] = 1
    return mat