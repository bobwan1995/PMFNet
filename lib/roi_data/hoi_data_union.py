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
import utils.keypoints as keypoint_utils
from utils.fpn import distribute_rois_over_fpn_levels
import cv2


def get_hoi_union_blob_names(is_training=True):
    """

    :param is_training:
    :return:
    """
    blob_names = ['human_boxes', 'object_boxes', 'union_boxes', 'rescale_kps', 'union_mask', 'spatial_info',
                  'interaction_human_inds', 'interaction_object_inds', 'interaction_affinity',
                  'gt_union_heatmap', 'interaction_init_part_attens',
                  'poseconfig', 'part_boxes', 'flag']

    if is_training:
        blob_names.extend(['human_action_labels', 'interaction_action_labels'])
    else:
        blob_names.extend(['human_scores', 'object_scores'])

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
            blob_names += ['human_boxes_fpn' + str(lvl), 'object_boxes_fpn' + str(lvl),
                           'union_boxes_fpn' + str(lvl)]
        blob_names += ['human_boxes_idx_restore_int32', 'object_boxes_idx_restore_int32',
                       'union_boxes_idx_restore_int32']
        # if cfg.VCOCO.KEYPOINTS_ON:
        #     for lvl in range(k_min, k_max + 1):
        #         blob_names += ['keypoint_rois_fpn' + str(lvl)]

    blob_names = {k: [] for k in blob_names}
    return blob_names


def sample_for_hoi_branch(rpn_net, roidb, im_info,
                          cls_score=None, bbox_pred=None, is_training=True):
    hoi_blob_names = get_hoi_union_blob_names(is_training=is_training)
    if is_training:
        # list of sample result
        blobs_list = sample_for_hoi_branch_train(rpn_net, roidb, im_info)
    else:
        raise NotImplementedError
    hoi_blob_in = merge_hoi_blobs(hoi_blob_names, blobs_list)
    return hoi_blob_in


def merge_hoi_blobs(hoi_blob_in, blobs_list):
    '''
    Merge blob of each image
    :param hoi_blob_in: hoi blob names dict
    :param blobs_list: blob of each image
    :return:
    '''
    # support mini-batch
    human_boxes_count = 0
    object_boxes_count = 0
    for i in range(len(blobs_list)):
        blob_this_im = blobs_list[i]
        # ensure interaction_*_inds only index correct image's human/target_object feature
        blob_this_im['interaction_human_inds'] += human_boxes_count
        blob_this_im['interaction_object_inds'] += object_boxes_count

        # count human/object rois num
        human_boxes_count += blob_this_im['human_boxes'].shape[0]
        object_boxes_count += blob_this_im['object_boxes'].shape[0]
        # Append to blob list
        for k, v in blob_this_im.items():
            hoi_blob_in[k].append(v)

    # Concat the training blob lists into tensors
    # np.concatenate default axis=0
    for k, v in hoi_blob_in.items():
        if len(v) > 0:
            hoi_blob_in[k] = np.concatenate(v)

    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        distribute_rois_over_fpn_levels(hoi_blob_in, 'human_boxes')
        distribute_rois_over_fpn_levels(hoi_blob_in, 'object_boxes')
        distribute_rois_over_fpn_levels(hoi_blob_in, 'union_boxes')

    return hoi_blob_in


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
    print('cfg.TRAIN.FG_THRESH: ',cfg.TRAIN.FG_THRESH)
    keep_rois_inds = np.where(rpn_ret['rois_max_overlaps'] >= cfg.TRAIN.FG_THRESH)
    rois = rpn_ret['rois'][keep_rois_inds]
    rois_to_gt_ind = rpn_ret['rois_to_gt_ind_map'][keep_rois_inds]

    train_hoi_blobs = []
    # get blobs of each image
    for i, entry in enumerate(roidb):
        inds_of_this_image = np.where(rois[:, 0] == i)
        blob_this_im = _sample_human_union_boxes(rois[inds_of_this_image],
                                                 rois_to_gt_ind[inds_of_this_image],
                                                 entry,
                                                 im_info[i],
                                                 i)
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
    hoi_blob_names = get_hoi_union_blob_names(is_training=is_training)
    scales = im_info.data.numpy()[:, 2]

    train_hoi_blobs = []
    # get blobs of each image
    for i, entry in enumerate(roidb):
    
        keep_rois_inds = np.where(entry['max_overlaps'] >= cfg.TRAIN.FG_THRESH)
        # rois is based on enlarged image
        rois = entry['boxes'][keep_rois_inds] * scales[i]
        rois = np.concatenate([np.full((keep_rois_inds[0].shape[0], 1),i), rois], 1).astype(np.float32)
        rois_to_gt_ind = entry['box_to_gt_ind_map'][keep_rois_inds]

        blob_this_im = _sample_human_union_boxes(rois,
                                                 rois_to_gt_ind,
                                                 entry,
                                                 im_info[i],
                                                 i, keep_rois_inds)

        if blob_this_im is not None:
            train_hoi_blobs.append(blob_this_im)

    if len(train_hoi_blobs)==0:
        return None

    hoi_blob_in = merge_hoi_blobs(hoi_blob_names, train_hoi_blobs)
    return hoi_blob_in


def _sample_human_union_boxes(rois, rois_to_gt_ind, roidb, im_info, batch_idx, keep_rois_inds):
    """
    :param rois: fg rois(gt boxes have been added to rois, see roi_data/fast_rcnn)
    :param rois_to_gt_ind:
    :param roidb:
    :param im_info:
    :return:
    """
    # ipdb.set_trace()
    # ToDo: cfg file, split model and data file type
    human_rois_num_per_image = int(cfg.VCOCO.HUMAN_NUM_PER_IM) # 16
    object_rois_num_per_image = int(cfg.VCOCO.OBJECT_NUM_PER_IM) # 64
    target_object_rois_num_per_image = int(cfg.VCOCO.TARGET_OBJECT_NUM_PER_IM)  # 48. fraction * object_rois_num

    # get gt human inds that with action
    # ToDo: name change
    # add all human(even without action) to human-centric branch
    # gt_objects_num = roidb['gt_actions'].shape[0]
    # human_gt_inds = np.where(roidb['gt_classes'][:gt_objects_num] == 1)[0]
    # if len(human_gt_inds)==0:
    #     return None
    
    # #print('len human_gt_inds: ', len(human_gt_inds))

    # # -------------------------------------------------------------------------
    # # Human-Centric Branch: sample human rois and calculate targets
    # # -------------------------------------------------------------------------

    # # get proposals(rois) that assigned to gt human with action
    # # and corresponding target_objects
    # rois_human_inds = [] ## human proposals
    # for human_gt_i in human_gt_inds:
    #     rois_human_inds.append(np.where(rois_to_gt_ind == human_gt_i)[0])

    # if len(rois_human_inds)==0:
    #     return None
    # rois_human_inds = np.concatenate(rois_human_inds)

    
    rois_human_inds = np.where(roidb['gt_classes'][keep_rois_inds]==1)[0]
    if len(rois_human_inds)==0:
        return None

    # select 16 rois of human
    # human_rois_num_this_image = min(human_rois_num_per_image, rois_human_inds.size)
    # if rois_human_inds.size > 0:
    #     rois_human_inds = npr.choice(
    #         rois_human_inds, size=human_rois_num_this_image, replace=False)

    # get human action targets relative location
    human_rois = rois[rois_human_inds]
    human_action_labels = roidb['gt_actions'][rois_to_gt_ind[rois_human_inds]]
    # human boxes that no-action, their gt_actions is filled with -1
    human_action_labels[human_action_labels < 0] = 0

    # rois_human_role_ids = roidb['gt_role_id'][rois_to_gt_ind[rois_human_inds]]

    # # -------------------------------------------------------------------------
    # # Interaction Branch: sample object rois
    # # -------------------------------------------------------------------------

    # # ToDo: if object boxes are computed parallel, next step is necessary,
    # # ToDo: or else, we could select object boxes from all rois

    # # target objects number is limited by TARGET_OBJECT_NUM_PER_IM, so that I
    # # want to pick gt target objects before other gt objects
    # # get gt target object inds
    # target_object_gt_inds = np.unique(rois_human_role_ids)
    # target_object_gt_inds = target_object_gt_inds[np.where(target_object_gt_inds > -1)]

    # # get rois that assigned to gt role object
    # if target_object_gt_inds.size > 0:
    #     rois_target_object_inds = []
    #     for role_gt_i in target_object_gt_inds:
    #         rois_target_object_inds.append(np.where(rois_to_gt_ind == role_gt_i)[0])
    #     rois_target_object_inds = np.concatenate(rois_target_object_inds)
    # else:
    #     # some actions don't have target_objects
    #     rois_target_object_inds  = np.empty((0,), dtype=np.int64)
    # # rois of other objects, including person, because target object could be person
    # rois_non_target_inds = np.setdiff1d(np.arange(rois.shape[0]), rois_target_object_inds)

    # # select 64 target objects
    # # target object could be person, all objects could be target object
    # target_object_rois_num_this_image = min(target_object_rois_num_per_image, rois_target_object_inds.size)
    # if rois_target_object_inds.size > 0:
    #     rois_target_object_inds = npr.choice(
    #         rois_target_object_inds, size=target_object_rois_num_this_image, replace=False)
    # target_object_rois = rois[rois_target_object_inds]

    # non_target_rois_num_this_image = min(object_rois_num_per_image-rois_target_object_inds.size, rois_non_target_inds.size)
    # if rois_non_target_inds.size > 0:
    #     rois_non_target_inds = npr.choice(
    #         rois_non_target_inds, size=non_target_rois_num_this_image, replace=False)
    # non_target_rois = rois[rois_non_target_inds]
    # rois_object_inds = np.concatenate((rois_target_object_inds, rois_non_target_inds))
    # object_rois = np.concatenate((target_object_rois, non_target_rois), axis=0)

    rois_object_inds = np.arange(rois.shape[0])
    object_rois = rois

    # Sample positive triplets
    if len(rois_human_inds)==0 or len(rois_object_inds)==0:
        return None

    triplets_info = \
        generate_triplets(rois, rois_human_inds, rois_object_inds,
                          rois_to_gt_ind, roidb['gt_role_id'], batch_idx=batch_idx)

    union_mask = generate_union_mask(human_rois, object_rois, triplets_info['union_boxes'],
                triplets_info['human_inds'], triplets_info['object_inds'])

    rois_keypoints = roidb['precomp_keypoints'][keep_rois_inds]
    human_keypoints = rois_keypoints[rois_human_inds]

    if cfg.VCOCO.USE_KPS17:
        part_boxes, flag = generate_part_box_from_kp17(human_keypoints, human_rois, float(im_info[2]),
                                                        body_ratio=cfg.VCOCO.BODY_RATIO, head_ratio=1.5)
    else:
        part_boxes, flag = generate_part_box_from_kp(human_keypoints, human_rois, float(im_info[2]),
                                                        body_ratio=cfg.VCOCO.BODY_RATIO, head_ratio=1.5)

    #union_gt_kps = gt_keypoints[rois_to_gt_ind[rois_human_inds[triplets_info['human_inds']]]]
    
    union_gt_kps = rois_keypoints[rois_human_inds[triplets_info['human_inds']]]

    gt_union_heatmap, union_mask, rescale_kps = generate_joints_heatmap(union_gt_kps, triplets_info['union_boxes'][:, 1:]/float(im_info[2]), \
            rois[rois_human_inds[triplets_info['human_inds']]][:, 1:]/float(im_info[2]), \
            rois[rois_object_inds[triplets_info['object_inds']]][:, 1:]/float(im_info[2]), \
            gaussian_kernel=(cfg.VCOCO.HEATMAP_KERNEL_SIZE,cfg.VCOCO.HEATMAP_KERNEL_SIZE))

    poseconfig = generate_pose_configmap(union_gt_kps, triplets_info['union_boxes'][:, 1:]/float(im_info[2]), \
            rois[rois_human_inds[triplets_info['human_inds']]][:, 1:]/float(im_info[2]), \
            rois[rois_object_inds[triplets_info['object_inds']]][:, 1:]/float(im_info[2]))

    return_dict = dict(
        human_boxes=human_rois,
        object_boxes=object_rois,
        union_boxes=triplets_info['union_boxes'],
        union_mask=union_mask,
        human_action_labels=human_action_labels,
        spatial_info=triplets_info['spatial_info'],
        interaction_human_inds=triplets_info['human_inds'],
        interaction_object_inds=triplets_info['object_inds'],
        interaction_action_labels=triplets_info['action_labels'],
        interaction_affinity=triplets_info['interaction_affinity'].astype(np.int32),
        part_boxes=part_boxes,
        flag=flag,
        gt_union_heatmap=gt_union_heatmap,
        rescale_kps=rescale_kps,
        poseconfig=poseconfig
    )

    return return_dict


def generate_joints_heatmap(union_kps, union_rois, human_rois, obj_rois, gaussian_kernel=(7,7)):
    # ipdb.set_trace()
    num_triplets, _, kp_num = union_kps.shape
    ret = np.zeros((num_triplets, kp_num+2, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
    union_mask = np.zeros((num_triplets, 5, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
    rescale_kps = np.zeros((num_triplets, kp_num, 2)).astype(np.int32)

    ux0 = union_rois[:, 0]
    uy0 = union_rois[:, 1]
    ux1 = union_rois[:, 2]
    uy1 = union_rois[:, 3]

    # offset_x = rois[:, 0]
    # offset_y = rois[:, 1]
    scale_x = cfg.KRCNN.HEATMAP_SIZE / (ux1 - ux0)
    scale_y = cfg.KRCNN.HEATMAP_SIZE / (uy1 - uy0)
    #import pdb
    #pdb.set_trace()
    for i in range(num_triplets):
        for j in range(kp_num):
            vis = union_kps[i, -1, j]
            if vis > 0:
                kpx, kpy = union_kps[i, :2, j]
                if kpx < ux0[i] or kpy < uy0[i] or kpx > ux1[i] or kpy > uy1[i]:
                    continue
                kpx = np.clip(np.round((kpx - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
                kpy = np.clip(np.round((kpy - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
                rescale_kps[i, j] = np.array([kpx, kpy])
                ret[i, j, kpy, kpx] = 1. #1.0
                ret[i, j] = cv2.GaussianBlur(ret[i, j], gaussian_kernel, 0)
                am = np.amax(ret[i, j])
                ret[i, j] /= am

        ox0, oy0, ox1, oy1 = human_rois[i]
        ox0 = np.clip(np.round((ox0 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        oy0 = np.clip(np.round((oy0 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        ox1 = np.clip(np.round((ox1 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        oy1 = np.clip(np.round((oy1 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        ret[i, -2, oy0:oy1, ox0:ox1] = 1.0
        ret[i, -2] -= 0.5
        union_mask[i, 0, oy0:oy1, ox0:ox1] = 1.

        ox0, oy0, ox1, oy1 = obj_rois[i]
        ox0 = np.clip(np.round((ox0 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        oy0 = np.clip(np.round((oy0 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        ox1 = np.clip(np.round((ox1 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        oy1 = np.clip(np.round((oy1 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        ret[i, -1, oy0:oy1, ox0:ox1] = 1.0
        ret[i, -1] -= 0.5
        union_mask[i, 1, oy0:oy1, ox0:ox1] = 1.
        union_mask[i, 2] = np.maximum(union_mask[i, 0], union_mask[i, 1])

        anoise = 0.1*np.random.random((cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
        union_mask[i, 3] = np.maximum(union_mask[i, 2], anoise)

        anoise2 = 0.1 * np.ones((cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
        union_mask[i, 4] = np.maximum(union_mask[i, 2], anoise2)
    
    return ret.astype(np.float32), union_mask.astype(np.float32), rescale_kps


def generate_pose_configmap(union_kps, union_rois, human_rois, obj_rois, gaussian_kernel=(7,7)):
    # generate spatial configuration of pose map: 64 x 64
    # draw lines with different values
    skeletons = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],\
    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2],\
    [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    num_triplets, _, kp_num = union_kps.shape
    ret = np.zeros((num_triplets, 1+2, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))

    ux0 = union_rois[:, 0]
    uy0 = union_rois[:, 1]
    ux1 = union_rois[:, 2]
    uy1 = union_rois[:, 3]

    scale_x = cfg.KRCNN.HEATMAP_SIZE / (ux1 - ux0)
    scale_y = cfg.KRCNN.HEATMAP_SIZE / (uy1 - uy0)
    for i in range(num_triplets):
        cur_kps = np.zeros((kp_num, 2)).astype(np.int32)
        vis = union_kps[i, -1]
        for j in range(kp_num):
            #vis = union_kps[i, -1, j]
            #if vis>0 and vis<1:
            #    print(union_kps[i,:,j])
            if vis[j] > 0:
                kpx, kpy = union_kps[i, :2, j]
                kpx = np.round((kpx - ux0[i]) * scale_x[i]).astype(np.int)
                kpy = np.round((kpy - uy0[i]) * scale_y[i]).astype(np.int)
                cur_kps[j] = np.array([kpx, kpy])

        for j, sk in enumerate(skeletons):
            sk0 =  sk[0] - 1
            sk1 =  sk[1] - 1
            if vis[sk0]>0 and vis[sk1]>0:
                ret[i, 0] = cv2.line(ret[i, 0], tuple(cur_kps[sk0]), tuple(cur_kps[sk1]), 0.05*(j+1), 3)


        ox0, oy0, ox1, oy1 = human_rois[i]
        ox0 = np.clip(np.round((ox0 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        oy0 = np.clip(np.round((oy0 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        ox1 = np.clip(np.round((ox1 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        oy1 = np.clip(np.round((oy1 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        ret[i, 1, oy0:oy1, ox0:ox1] = 1.0

        ox0, oy0, ox1, oy1 = obj_rois[i]
        ox0 = np.clip(np.round((ox0 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        oy0 = np.clip(np.round((oy0 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        ox1 = np.clip(np.round((ox1 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        oy1 = np.clip(np.round((oy1 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE-1)
        ret[i, 2, ox0:ox1, oy0:oy1] = 1.0


    return ret.astype(np.float32)


def generate_part_box_from_kp(all_kps, human_rois, scale, body_ratio=0.1, head_ratio=1.5):
    """
    :param kps: human_roi_num*17*3
    :param human_roi: human_roi_num*5
    :return: human_roi_num*13*5
    """
    assert all_kps.shape[0] == human_rois.shape[0]
    human_num, _, kp_num = all_kps.shape
    # 13 for head and 12 other parts, head in in last channel
    ret = -np.ones([human_num, 13, 4]).astype(human_rois.dtype)
    flag = np.zeros([human_num, 13])

    for h in range(human_num):
        x0,y0,x1,y1 = human_rois[h, 1:]
        width = x1-x0
        height = y1-y0
        length = max(width, height)

        ### TODO: for vis: kps = all_kps[h, :2]
        kps = all_kps[h, :2] * scale

        vis = all_kps[h, -1]
        valid_ind = np.where(vis > 0)[0]
        head_ind = []
        for ind in valid_ind:
            if ind < 5:
                head_ind.append(ind)
            else:
                flag[h, ind-5] = 1
                kp_x, kp_y = kps[:, ind]
                # x_min = np.clip(kp_x - ratio_x*width, x0, x1)
                # x_max = np.clip(kp_x + ratio_x*width, x0, x1)
                # y_min = np.clip(kp_y - ratio_y*height, y0, y1)
                # y_max = np.clip(kp_y + ratio_y*height, y0, y1)
                x_min = np.clip(kp_x - body_ratio*length, x0, x1)
                x_max = np.clip(kp_x + body_ratio*length, x0, x1)
                y_min = np.clip(kp_y - body_ratio*length, y0, y1)
                y_max = np.clip(kp_y + body_ratio*length, y0, y1)
                ret[h, ind-5] = np.array([x_min, y_min, x_max, y_max])

        if len(head_ind) > 0:
            flag[h, -1] = 1
            head_ind = np.array(head_ind)
            # ipdb.set_trace()
            x_head_min = np.min(kps[0, head_ind])
            x_head_max = np.max(kps[0, head_ind])
            y_head_min = np.min(kps[1, head_ind])
            y_head_max = np.max(kps[1, head_ind])
            head_width = x_head_max - x_head_min
            head_height = y_head_max - y_head_min
            x_center = (x_head_max+x_head_min)/2.0
            y_center = (y_head_max+y_head_min)/2.0
            x_min = np.clip(x_center-head_width*head_ratio/2.0, x0, x1)
            x_max = np.clip(x_center+head_width*head_ratio/2.0, x0, x1)
            y_min = np.clip(y_center-head_height*head_ratio/2.0, y0, y1)
            y_max = np.clip(y_center+head_height*head_ratio/2.0, y0, y1)
            ret[h, -1] = np.array([x_min, y_min, x_max, y_max])
    ret = np.concatenate([human_rois[:, [0]].repeat(13, 1)[:,:,None], ret], -1)
    return ret, flag


def generate_part_box_from_kp17(all_kps, human_rois, scale, body_ratio=0.1, head_ratio=1.5):
    """
    :param kps: human_roi_num*17*3
    :param human_roi: human_roi_num*5
    :return: human_roi_num*13*5
    """
    assert all_kps.shape[0] == human_rois.shape[0]
    human_num, _, kp_num = all_kps.shape
    ret = -np.ones([human_num, 17, 4]).astype(human_rois.dtype)
    flag = np.zeros([human_num, 17])

    for h in range(human_num):
        x0,y0,x1,y1 = human_rois[h, 1:]
        width = x1-x0
        height = y1-y0
        length = max(width, height)

        kps = all_kps[h, :2] * scale
        vis = all_kps[h, -1]
        valid_ind = np.where(vis > 0)[0]

        for ind in valid_ind:
            flag[h, ind] = 1
            kp_x, kp_y = kps[:, ind]
            x_min = np.clip(kp_x - body_ratio*length, x0, x1)
            x_max = np.clip(kp_x + body_ratio*length, x0, x1)
            y_min = np.clip(kp_y - body_ratio*length, y0, y1)
            y_max = np.clip(kp_y + body_ratio*length, y0, y1)
            ret[h, ind] = np.array([x_min, y_min, x_max, y_max])
    ret = np.concatenate([human_rois[:, [0]].repeat(17, 1)[:,:,None], ret], -1)
    return ret, flag


def generate_triplets(rois, rois_human_inds, rois_object_inds, rois_to_gt_ind, gt_role_id, batch_idx):
    """
    :param rois:
    :param rois_human_inds: human ind to rois index
    :param rois_object_inds:
    :param rois_to_gt_ind: rois index to gt box index
    :param gt_role_id:
    :param batch_idx:
    :return:
    """
    # ToDo: cfg
    # ipdb.set_trace()
    triplets_num_per_image = cfg.VCOCO.TRIPLETS_NUM_PER_IM
    fg_triplets_num_per_image = int(triplets_num_per_image * cfg.VCOCO.FG_TRIPLETS_FRACTION)

    # label matrix
    gt_action_mat = generate_action_mat(gt_role_id) #N x N x 26 x 2

    # generate combinations
    human_rois_inds, object_rois_inds = np.meshgrid(np.arange(rois_human_inds.size),
                                                    np.arange(rois_object_inds.size), indexing='ij')
    human_rois_inds, object_rois_inds = human_rois_inds.reshape(-1), object_rois_inds.reshape(-1)
    # triplet labels
    action_labels = gt_action_mat[rois_to_gt_ind[rois_human_inds[human_rois_inds]],
                                  rois_to_gt_ind[rois_object_inds[object_rois_inds]]] # (hN' x oN') x 26 x 2
    interaction_action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
    # convert to 24-class 
    # action_labels: (hN' x oN') x 24
    # interaction_affinity: (hN' x oN') x 1
    # init_part_attens: (hN' x oN') x 7 x 17 (last dimension is the holistic atten which is all 1)
    action_labels = action_labels[:, np.where(interaction_action_mask > 0)[0], np.where(interaction_action_mask > 0)[1]]
    interaction_affinity = np.any(action_labels.reshape(action_labels.shape[0], -1) > 0, 1)

    # info for training
    union_boxes = box_utils.get_union_box(rois[rois_human_inds[human_rois_inds]][:, 1:],
                                          rois[rois_object_inds[object_rois_inds]][:, 1:])
    union_boxes = np.concatenate(
        (batch_idx * np.ones((union_boxes.shape[0], 1), dtype=union_boxes.dtype),
         union_boxes), axis=1)
    relative_location = box_utils.bbox_transform_inv(rois[rois_human_inds[human_rois_inds]][:, 1:],
                                                     rois[rois_object_inds[object_rois_inds]][:, 1:])

    # sample fg/bg triplets
    fg_triplets_inds = np.where(np.sum(action_labels, axis=1) > 0)[0]
    bg_triplets_inds = np.setdiff1d(np.arange(action_labels.shape[0]), fg_triplets_inds)

    fg_triplets_num_this_image = min(int(triplets_num_per_image * 1/4.), fg_triplets_inds.size)
    if fg_triplets_inds.size > 0:
        fg_triplets_inds = npr.choice(
            fg_triplets_inds, size=fg_triplets_num_this_image, replace=False)

    bg_triplets_num_this_image = max(fg_triplets_num_this_image * 3, 1)
    bg_triplets_num_this_image = min(bg_triplets_num_this_image, bg_triplets_inds.size)

    if bg_triplets_inds.size > 0 and bg_triplets_num_this_image>0:
        bg_triplets_inds = npr.choice(
            bg_triplets_inds, size=bg_triplets_num_this_image, replace=False)

        keep_triplets_inds = np.concatenate((fg_triplets_inds, bg_triplets_inds))
    else:
        keep_triplets_inds = fg_triplets_inds

    return_dict = dict(
        human_inds=human_rois_inds[keep_triplets_inds],
        object_inds=object_rois_inds[keep_triplets_inds],
        union_boxes=union_boxes[keep_triplets_inds],
        action_labels=action_labels[keep_triplets_inds],
        spatial_info=relative_location[keep_triplets_inds],
        interaction_affinity=interaction_affinity[keep_triplets_inds],
    )

    return return_dict


def generate_action_mat(gt_role_id):
    '''
    Generate a matrix to store action triplet
    :param gt_role_id: N x 26 x 2
    :return: action_mat, row is person id, column is target_object id,
             third axis is action id
    '''
    mat = np.zeros((gt_role_id.shape[0], gt_role_id.shape[0],
                    cfg.VCOCO.NUM_ACTION_CLASSES, cfg.VCOCO.NUM_TARGET_OBJECT_TYPES), dtype=np.float32)
    role_ids = gt_role_id[np.where(gt_role_id > -1)] # t x 1
    human_ids, action_cls, target_cls = np.where(gt_role_id > -1) # t(=interNum) x 1
    assert role_ids.size == human_ids.size == action_cls.size
    mat[human_ids, role_ids, action_cls, target_cls] = 1 # N x N x 26 x 2
    return mat


def get_location_info(human_boxes, object_boxes, union_boxes):
    assert human_boxes.shape[1] == object_boxes.shape[1] == union_boxes.shape[1] == 4
    human_object_loc = box_utils.bbox_transform_inv(human_boxes, object_boxes)
    human_union_loc = box_utils.bbox_transform_inv(human_boxes, union_boxes)
    object_union_loc = box_utils.bbox_transform_inv(object_boxes, union_boxes)
    return np.concatenate((human_object_loc, human_union_loc, object_union_loc), axis=1)


def generate_union_mask(human_rois, object_rois, union_rois, human_inds, object_inds):
    union_mask = np.zeros((human_inds.size, 2, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
    pooling_size = cfg.KRCNN.HEATMAP_SIZE
    for i in range(human_inds.size):
        union_left_top = np.tile(union_rois[i, 1:3], 2)
        w, h = union_rois[i, 3:5] - union_rois[i, 1:3]
        weights_t = pooling_size / np.array([w, h, w, h])
        human_coord = ((human_rois[human_inds][i, 1:] - union_left_top) * weights_t).astype(np.int32)
        object_coord = ((object_rois[object_inds][i, 1:] - union_left_top) * weights_t).astype(np.int32)
        union_mask[i, 0, human_coord[1]:human_coord[3] + 1, human_coord[0]:human_coord[2] + 1] = 1
        union_mask[i, 1, object_coord[1]:object_coord[3] + 1, object_coord[0]:object_coord[2] + 1] = 1
    return union_mask.astype(np.float32)
