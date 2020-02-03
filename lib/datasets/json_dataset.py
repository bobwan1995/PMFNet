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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import json
# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from .vcoco_json import VCOCO

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog import ANN_FN
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX
from .dataset_catalog import VCOCO_ANNS, VCOCO_IMID

logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_FN]), \
            'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.COCO = COCO(DATASETS[name][ANN_FN])

        self.vcoco = False
        if cfg.MODEL.VCOCO_ON:
            self.vcoco = True
            self.VCOCO = VCOCO(DATASETS[name][VCOCO_ANNS])

        self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self._init_keypoints()

        # # Set cfg.MODEL.NUM_CLASSES
        # if cfg.MODEL.NUM_CLASSES != -1:
        #     assert cfg.MODEL.NUM_CLASSES == 2 if cfg.MODEL.KEYPOINTS_ON else self.num_classes, \
        #         "number of classes should equal when using multiple datasets"
        # else:
        #     cfg.MODEL.NUM_CLASSES = 2 if cfg.MODEL.KEYPOINTS_ON else self.num_classes

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map']
        if self.keypoints is not None:
            keys += ['gt_keypoints', 'has_visible_keypoints']
        if self.vcoco:
            keys += ['gt_actions', 'gt_role_id']  # , 'action_mat'
        return keys

    def get_roidb(
            self,
            gt=False,
            proposal_file=None,
            min_proposal_size=2,
            proposal_limit=-1,
            crowd_filter_thresh=0
        ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if cfg.DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:100]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        for entry in roidb:
            self._prep_roidb_entry(entry)

        if self.vcoco and cfg.VCOCO.USE_PRECOMP_BOX:
            # precomp_filepath = os.path.join(self.cache_path, self.name + '_precomp_boxes.json')
            
            #precomp_filepath = os.path.join(self.cache_path, self.name + '_precomp_boxes_ican.json')
            #precomp_boxes = json.load(open(precomp_filepath, 'r'))

            precomp_bbox_kps_filepath = os.path.join(self.cache_path, 'addPredPose', self.name + '_precomp_boxes_keypoints_ican2.json')
            precomp_bbox_keypoints = json.load(open(precomp_bbox_kps_filepath, 'r'))
            
            # TODO: find why len(precomp_boxes) is 4095(should be 4096)
            # import ipdb; ipdb.set_trace()
            affinity_mat_filepath = None
            if self.name == 'vcoco_train':
                affinity_mat_filepath  = os.path.join(self.cache_path, 'train_affinity_score_matrix.pkl')
            elif self.name == 'vcoco_val':
                affinity_mat_filepath = os.path.join(self.cache_path, 'val_affinity_score_matrix.pkl')
            elif self.name == 'vcoco_test':
                affinity_mat_filepath = os.path.join(self.cache_path, 'test_affinity_score_matrix_no_pose.pkl')

            if affinity_mat_filepath is not None:
                affinity_mat = pickle.load(open(affinity_mat_filepath, 'rb'))

            for i, entry in enumerate(roidb):
                #self._add_vcoco_precomp_box(entry, precomp_boxes)
                self._add_vcoco_precomp_bbox_keypoints(entry, precomp_bbox_keypoints)
                if affinity_mat_filepath is not None:
                    self._add_affinity_mat(entry, affinity_mat)

            #print('add precomp box from {}'.format(precomp_filepath))
            print('add precomp keypoints from {}'.format(precomp_bbox_kps_filepath))

        if gt:
            # Include ground-truth object annotations
           # cache_filepath = os.path.join(self.cache_path, self.name+'_gt_roidb.pkl')
            # cache_filepath = os.path.join('cache', self.name+'_gt_roidb.pkl')
            # if os.path.exists(cache_filepath) and not cfg.DEBUG:
            #     self.debug_timer.tic()
            #     self._add_gt_from_cache(roidb, cache_filepath)
            #     logger.debug(
            #         '_add_gt_from_cache took {:.3f}s'.
            #         format(self.debug_timer.toc(average=False))
            #     )
            # else:
            #     self.debug_timer.tic()
            #     for entry in roidb:
            #         self._add_gt_annotations(entry)
            #     logger.debug(
            #         '_add_gt_annotations took {:.3f}s'.
            #         format(self.debug_timer.toc(average=False))
            #     )
            #     if not cfg.DEBUG:
            #         with open(cache_filepath, 'wb') as fp:
            #             pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
            #         logger.info('Cache ground truth roidb to %s', cache_filepath)

            for entry in roidb:
                self._add_gt_annotations(entry)


        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name']
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.float32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]
        # Add v-coco annotations: action and role id
        # 26 different actions, two kinds of role: instrument or direct object
        if self.vcoco:
            entry['gt_actions'] = np.empty((0, self.VCOCO.num_actions), dtype=np.int32)
            entry['gt_role_id'] = np.empty((0, self.VCOCO.num_actions, cfg.VCOCO.NUM_TARGET_OBJECT_TYPES), dtype=np.int32)
            # action mat's size is gt_boxes_num * gt_boxes_num
            # entry['action_mat'] = scipy.sparse.csr_matrix(
            #     np.empty((0, 0, 0), dtype=np.float32)
            # )

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_ann_ids = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for i, obj in enumerate(objs):
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_ann_ids.append(ann_ids[i])
                valid_segms.append(obj['segmentation'])
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.vcoco:
            gt_actions = -np.ones((num_valid_objs, self.VCOCO.num_actions), dtype=entry['gt_actions'].dtype)
            gt_role_id = -np.ones((num_valid_objs, self.VCOCO.num_actions, cfg.VCOCO.NUM_TARGET_OBJECT_TYPES),
                                  dtype=entry['gt_role_id'].dtype)

        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if self.vcoco:
                gt_actions[ix, :], gt_role_id[ix, :, :] = \
                    self.VCOCO.get_vsrl_data(valid_ann_ids[ix], valid_ann_ids, valid_objs)
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints
        if self.vcoco:
            entry['gt_actions'] = np.append(entry['gt_actions'], gt_actions, axis=0)
            entry['gt_role_id'] = np.append(entry['gt_role_id'], gt_role_id, axis=0)
            # entry['action_mat'] = scipy.sparse.csr_matrix(
            #     self.VCOCO.generate_action_mat(gt_role_id)
            # )


    def _add_affinity_mat(self, entry, affinity_mat):
        affinity_value = affinity_mat[entry['id']]
        entry['affinity_mat'] = affinity_value


    def _add_vcoco_precomp_box(self, entry, precomp_boxes):
        im_id = str(entry['id'])
        value = np.array(precomp_boxes[im_id], dtype=np.float32)
        # human_ind = np.where(value[:, -1] == 1)[0]
        # if human_ind.shape == 0:
        #     print(im_id)
        #     import ipdb; ipdb.set_trace()
        entry['precomp_boxes'] = value[:, :4]
        entry['precomp_score'] = value[:, -2]
        entry['precomp_cate'] = value[:, -1]

    def _add_vcoco_precomp_bbox_keypoints(self, entry, precomp_bbox_keypoints):
        im_id = str(entry['id'])
        value = np.array(precomp_bbox_keypoints[im_id], dtype=np.float32)
        # human_ind = np.where(value[:, -1] == 1)[0]
        # if human_ind.shape == 0:
        #     print(im_id)
        #     import ipdb; ipdb.set_trace()
        entry['precomp_boxes'] = value[:, :4]
        entry['precomp_score'] = value[:, 4]
        entry['precomp_cate'] = value[:, 5]
        # entry['precomp_keypoints'] = value[:, 6:]

        kp = value[:, 6:]
        x = kp[:, 0::3]  # 0-indexed x coordinates
        y = kp[:, 1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[:, 2::3]
        num_keypoints = kp.shape[-1] / 3
        #print(num_keypoints, self.num_keypoints)
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((kp.shape[0], 3, self.num_keypoints), dtype=np.float32)
        for i in range(self.num_keypoints):
            gt_kps[:, 0, i] = x[:, i]
            gt_kps[:, 1, i] = y[:, i]
            gt_kps[:, 2, i] = v[:, i]
        entry['precomp_keypoints'] = gt_kps



    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            cached_roidb = pickle.load(fp)

        assert len(roidb) == len(cached_roidb)

        for entry, cached_entry in zip(roidb, cached_roidb):
            values = [cached_entry[key] for key in self.valid_cached_keys]
            boxes, segms, gt_classes, seg_areas, gt_overlaps, is_crowd, \
                box_to_gt_ind_map = values[:7]
            if self.keypoints is not None:
                # key points always in 7:9
                gt_keypoints, has_visible_keypoints = values[7:9]
            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            entry['segms'].extend(segms)
            # To match the original implementation:
            # entry['boxes'] = np.append(
            #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'], box_to_gt_ind_map
            )
            if self.keypoints is not None:
                entry['gt_keypoints'] = np.append(
                    entry['gt_keypoints'], gt_keypoints, axis=0
                )
                entry['has_visible_keypoints'] = has_visible_keypoints
            if self.vcoco:
                # v-coco always in -2:
                gt_actions, gt_role_id = values[-2:]
                entry['gt_actions'] = np.append(entry['gt_actions'], gt_actions, axis=0)
                entry['gt_role_id'] = np.append(entry['gt_role_id'], gt_role_id, axis=0)
                # entry['action_mat'] = scipy.sparse.csr_matrix(action_mat)

    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            if cfg.KRCNN.NUM_KEYPOINTS != -1:
                assert cfg.KRCNN.NUM_KEYPOINTS == self.num_keypoints, \
                    "number of keypoints should equal when using multiple datasets"
            else:
                cfg.KRCNN.NUM_KEYPOINTS = self.num_keypoints
            self.keypoint_flip_map = {
                'left_eye': 'right_eye',
                'left_ear': 'right_ear',
                'left_shoulder': 'right_shoulder',
                'left_elbow': 'right_elbow',
                'left_wrist': 'right_wrist',
                'left_hip': 'right_hip',
                'left_knee': 'right_knee',
                'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return -np.ones((3, self.num_keypoints), dtype=np.float32)
            # return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.float32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps


# def add_proposals(roidb, rois, scales, crowd_thresh):
#     """Add proposal boxes (rois) to an roidb that has ground-truth annotations
#     but no proposals. If the proposals are not at the original image scale,
#     specify the scale factor that separate them in scales.
#     """
#     if cfg.VCOCO.USE_PRECOMP_BOX:
#         assert rois is None
#         box_list = [] # TODO
#         for i in range(len(roidb)):
#             box_list.append(roidb[i]['precomp_boxes'])
#     else:
#         box_list = []
#         for i in range(len(roidb)):
#             inv_im_scale = 1. / scales[i]
#             idx = np.where(rois[:, 0] == i)[0]
#             box_list.append(rois[idx, 1:] * inv_im_scale)
#     _merge_proposal_boxes_into_roidb(roidb, box_list)
#     if crowd_thresh > 0:
#         _filter_crowd_proposals(roidb, crowd_thresh)
#     _add_class_assignments(roidb)


def add_proposals(roidb, rois, im_info, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    scales = im_info[:, 2]
    if cfg.VCOCO.USE_PRECOMP_BOX:
        assert rois is None
        box_list = [] # TODO
        for i in range(len(roidb)):
            data_augmentation(roidb[i], im_info[i])
            if not cfg.TRAIN.USE_GT:
                box_list.append(roidb[i]['precomp_boxes'])
                #print('not use gt')
            else:
                data_augmentation_gt(roidb[i], im_info[i])
                #print(' use gt')
                box_list.append(np.concatenate((roidb[i]['boxes_aug'],roidb[i]['precomp_boxes']),axis=0))
    else:
        box_list = []
        for i in range(len(roidb)):
            inv_im_scale = 1. / scales[i]
            idx = np.where(rois[:, 0] == i)[0]
            box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_compute_boxes_into_roidb(roidb, box_list)

    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)
    #data_augmentation_gt(roidb[i], im_info[i])


def data_augmentation(entry, im_info):
    # for i, entry in enumerate(roidb):
    h, w, r = im_info
    boxes = entry['precomp_boxes']
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    x_center = (x0+x1)/2
    y_center = (y0+y1)/2
    width = x1-x0
    height = y1-y0

    ratio_x = np.random.uniform(0.8*1.1, 1.1*1.1, boxes.shape[0]) # range between 0.7 to 1.3
    ratio_y = np.random.uniform(0.8*1.1, 1.1*1.1, boxes.shape[0]) # range between 0.7 to 1.3

    # ratio_x = np.random.uniform(0.7*1.15, 1.3*1.15, boxes.shape[0]) # range between 0.7 to 1.3
    # ratio_y = np.random.uniform(0.7*1.15, 1.3*1.15, boxes.shape[0]) # range between 0.7 to 1.3

    offset_x = np.random.uniform(-0.05, 0.05, boxes.shape[0]) # range between -0.05 to 0.05
    offset_y = np.random.uniform(-0.05, 0.05, boxes.shape[0]) # range between -0.05 to 0.05

    x_center = x_center + offset_x*width
    y_center = y_center + offset_y*height
    x0_new = np.clip(x_center - width * ratio_x / 2., 0, w/r)
    x1_new = np.clip(x_center + width * ratio_x / 2., 0, w/r)
    y0_new = np.clip(y_center - height * ratio_y / 2., 0, h/r)
    y1_new = np.clip(y_center + height * ratio_y / 2., 0, h/r)

    entry['precomp_boxes'] = np.concatenate(([x0_new], [y0_new], [x1_new], [y1_new]), 0).T


def data_augmentation_gt(entry, im_info):
    h, w, r = im_info
    boxes = entry['boxes']
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    x_center = (x0+x1)/2
    y_center = (y0+y1)/2
    width = x1-x0
    height = y1-y0
    
    ratio_x = np.random.uniform(0.7*1.15, 1.3*1.15, boxes.shape[0]) # range between 0.7 to 1.3
    ratio_y = np.random.uniform(0.7*1.15, 1.3*1.15, boxes.shape[0]) # range between 0.7 to 1.3
    # ratio_x = 1.
    # ratio_y = 1.

    offset_x = np.random.uniform(-0.1, 0.1, boxes.shape[0]) # range between -0.05 to 0.05
    offset_y = np.random.uniform(-0.1, 0.1, boxes.shape[0]) # range between -0.05 to 0.05

    # offset_x = 0.
    # offset_y = 0.
    
    x_center = x_center + offset_x*width
    y_center = y_center + offset_y*height
    x0_new = np.clip(x_center - width * ratio_x / 2., 0, w/r)
    x1_new = np.clip(x_center + width * ratio_x / 2., 0, w/r)
    y0_new = np.clip(y_center - height * ratio_y / 2., 0, h/r)
    y1_new = np.clip(y_center + height * ratio_y / 2., 0, h/r)

    entry['boxes_aug'] = np.concatenate(([x0_new], [y0_new], [x1_new], [y1_new]), 0).T

# def get_precomp_info(roidb, im_info):
#     scales = im_info.data.numpy()[:, 2]
#     all_rois = np.empty((0, 5), np.float32)
#     all_score = np.empty((0, 1), np.float32)
#     all_cate = np.empty((0, 1), np.float32)
#
#     for i in range(len(roidb)):
#         # rois in enlarge image
#         rois = roidb[i]['precomp_boxes'] * scales[i]
#         rois = np.concatenate([np.full((rois.shape[0], 1), i), rois], 1).astype(np.float32)
#         scores = roidb[i]['precomp_score']
#         cates = roidb[i]['precomp_cate']
#         all_rois = np.append(all_rois, rois, axis=0)
#         all_score = np.append(all_score, scores, axis=0)
#         all_cate = np.append(all_cate, cates, axis=0)
#     return all_rois, all_score, all_cate


def _merge_compute_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    
    for i, entry in enumerate(roidb):
        boxes = box_list[i] # gt + det
        #print('len boxes:', len(boxes))
        num_boxes = boxes.shape[0]

        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            # import ipdb; ipdb.set_trace()
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]

        entry['boxes'] = boxes.astype(entry['boxes'].dtype, copy=False)
        entry['box_to_gt_ind_map'] = box_to_gt_ind_map.astype(entry['box_to_gt_ind_map'].dtype, copy=False)

        gt_to_classes = -np.ones(len(entry['box_to_gt_ind_map']))
        matched_ids = np.where(entry['box_to_gt_ind_map']>-1)[0]
        gt_to_classes[matched_ids] = entry['gt_classes'][entry['box_to_gt_ind_map'][matched_ids]]
        entry['gt_classes'] = gt_to_classes

        entry['seg_areas'] = np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        entry['gt_overlaps'] = gt_overlaps
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])

        is_to_crowd = np.ones(len(entry['box_to_gt_ind_map']))
        is_to_crowd[matched_ids] = entry['is_crowd'][entry['box_to_gt_ind_map'][matched_ids]]
        entry['is_crowd'] = is_to_crowd

        # entry['boxes'] = np.append(
        #     entry['boxes'],
        #     boxes.astype(entry['boxes'].dtype, copy=False),
        #     axis=0
        # )

        # gt_to_classes = -np.ones(len(box_to_gt_ind_map))
        # matched_ids = np.where(box_to_gt_ind_map>-1)[0]
        # gt_to_classes[matched_ids] = entry['gt_classes'][box_to_gt_ind_map[matched_ids]]

        # entry['gt_classes'] = np.append(
        #     entry['gt_classes'],
        #     gt_to_classes
        #     # np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        # )
        # entry['seg_areas'] = np.append(
        #     entry['seg_areas'],
        #     np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        # )
        # entry['gt_overlaps'] = np.append(
        #     entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        # )
        # entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        
        # is_to_crowd = np.ones(len(box_to_gt_ind_map))
        # is_to_crowd[matched_ids] = entry['is_crowd'][box_to_gt_ind_map[matched_ids]]

        # entry['is_crowd'] = np.append(
        #     entry['is_crowd'],
        #     is_to_crowd
        #     #np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        # )
        # entry['box_to_gt_ind_map'] = np.append(
        #     entry['box_to_gt_ind_map'],
        #     box_to_gt_ind_map.astype(
        #         entry['box_to_gt_ind_map'].dtype, copy=False
        #     )
        # )



def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb) * 2
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        #print('len boxes:', len(boxes))
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            # import ipdb; ipdb.set_trace()
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]

        # entry['boxes'] = boxes.astype(entry['boxes'].dtype, copy=False)
        # entry['box_to_gt_ind_map'] = box_to_gt_ind_map.astype(entry['box_to_gt_ind_map'].dtype, copy=False)
        
        # gt_to_classes = -np.ones(len(entry['box_to_gt_ind_map']))
        # matched_ids = np.where(entry['box_to_gt_ind_map']>-1)[0]
        # gt_to_classes[matched_ids] = entry['gt_classes'][entry['box_to_gt_ind_map'][matched_ids]]
        # entry['gt_classes'] = gt_to_classes

        # entry['seg_areas'] = np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        # entry['gt_overlaps'] = gt_overlaps
        # entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
    
        # is_to_crowd = np.ones(len(entry['box_to_gt_ind_map']))
        # is_to_crowd[matched_ids] = entry['is_crowd'][entry['box_to_gt_ind_map'][matched_ids]]
        # entry['is_crowd'] = is_to_crowd

        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )

        gt_to_classes = -np.ones(len(box_to_gt_ind_map))
        matched_ids = np.where(box_to_gt_ind_map>-1)[0]
        gt_to_classes[matched_ids] = entry['gt_classes'][box_to_gt_ind_map[matched_ids]]
        
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            gt_to_classes
            # np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        
        is_to_crowd = np.ones(len(box_to_gt_ind_map))
        is_to_crowd[matched_ids] = entry['is_crowd'][box_to_gt_ind_map[matched_ids]]

        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            is_to_crowd
            #np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]
