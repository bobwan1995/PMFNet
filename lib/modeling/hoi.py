import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import nn as mynn
from core.config import cfg
import utils.net as net_utils
from utils.net import SpacialConv, Conv2d, ResidualConv, weighted_binary_cross_entropy_interaction
# from modeling.fast_rcnn_heads import roi_2mlp_head
import ipdb
from torch.autograd import Variable
import pdb
from utils.cbam import SpatialGate, CBAM, CBAM_ks3, ChannelGate
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction


'''
An implementation of PMFNet.
'''

class PMFNet_Baseline(nn.Module):
    """
    Human Object Interaction.
    This module including Human-centric branch and Interaction branch of PMFNet.
    Holistic Part Attention initially
    directly estimate all attentions for each fine grained action
    """

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()

        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.roi_size = roi_size
        # ToDo: cfg
        hidden_dim = cfg.VCOCO.MLP_HEAD_DIM 
        print('.......hidden_dim of VCOCO HOI: ', hidden_dim)
        # num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES  # 26
        action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
        interaction_num_action_classes = action_mask.sum().item()  # 24
        self.interaction_num_action_classes = interaction_num_action_classes

        self.human_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)  # 512
        self.object_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)  # 512
        self.union_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.union_resolution = cfg.KRCNN.HEATMAP_SIZE if cfg.VCOCO.KEYPOINTS_ON else cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        interaction_fc1_dim_in = 3 * hidden_dim

        self.pose_fc3 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2, 512)
        self.pose_fc4 = nn.Linear(512, hidden_dim)

        interaction_fc1_dim_in += hidden_dim

        self.interaction_fc1 = nn.Linear(interaction_fc1_dim_in, hidden_dim)
        self.interaction_action_score = nn.Linear(hidden_dim, interaction_num_action_classes)


    def _init_weights(self):
        # Initialize human centric branch
        mynn.init.XavierFill(self.human_fc1.weight)
        init.constant_(self.human_fc1.bias, 0)
        # mynn.init.XavierFill(self.human_fc2.weight)
        # init.constant_(self.human_fc2.bias, 0)

        init.normal_(self.human_action_score.weight, std=0.01)
        init.constant_(self.human_action_score.bias, 0)
        init.normal_(self.human_action_bbox_pred.weight, std=0.001)
        init.constant_(self.human_action_bbox_pred.bias, 0)

        # Initialize interaction branch(object action score)
        mynn.init.XavierFill(self.interaction_fc1.weight)
        init.constant_(self.interaction_fc1.bias, 0)
        # mynn.init.XavierFill(self.interaction_fc2.weight)
        # init.constant_(self.interaction_fc2.bias, 0)

        init.normal_(self.interaction_action_score.weight, std=0.01)
        init.constant_(self.interaction_action_score.bias, 0)

    def detectron_weight_mapping(self):
        # hc is human centric branch
        # io is interaction branch object part
        detectron_weight_mapping = {
            'human_fc1.weight': 'hc_fc1_w',
            'human_fc1.bias': 'hc_fc1_b',
            'human_fc2.weight': 'hc_fc2_w',
            'human_fc2.bias': 'hc_fc2_b',
            'human_action_score.weight': 'hc_score_w',
            'human_action_score.bias': 'hc_score_b',

            'interaction_fc1.weight': 'inter_fc1_w',
            'interaction_fc1.bias': 'inter_fc1_b',
        }
        return detectron_weight_mapping, []

    def forward(self, x, hoi_blob):
        x_human = self.roi_xform(
            x, hoi_blob,
            blob_rois='human_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x_object = self.roi_xform(
            x, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x_human = x_human.view(x_human.size(0), -1)
        x_object = x_object.view(x_object.size(0), -1)

        # get inds from numpy
        device_id = x_human.get_device()
        interaction_human_inds = torch.from_numpy(
            hoi_blob['interaction_human_inds']).long().cuda(device_id)
        interaction_object_inds = torch.from_numpy(
            hoi_blob['interaction_object_inds']).long().cuda(device_id)
        # human score and bbox predict
        x_human = F.relu(self.human_fc1(x_human), inplace=True)
        x_object = F.relu(self.object_fc1(x_object), inplace=True)
        x_interaction = torch.cat((x_human[interaction_human_inds], x_object[interaction_object_inds]), dim=1)

        x_union = self.roi_xform(
            x, hoi_blob,
            blob_rois='union_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            # resolution=self.union_resolution,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x_union = x_union.view(x_union.size(0), -1)
        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_interaction = torch.cat((x_interaction, x_union), dim=1)

        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)
        # import ipdb
        # ipdb.set_trace()
        # poseconfig = kps_pred.view(kps_pred.size(0), -1)
        x_pose_line = kps_pred.view(kps_pred.size(0), -1)
        x_pose_line = F.relu(self.pose_fc3(x_pose_line), inplace=True)
        x_pose_line = F.relu(self.pose_fc4(x_pose_line), inplace=True)

        x_interaction = torch.cat((x_interaction, x_pose_line), dim=1)
        x_interaction = F.relu(self.interaction_fc1(x_interaction), inplace=True)

        interaction_action_score = self.interaction_action_score(x_interaction)
        hoi_blob['interaction_action_score'] = interaction_action_score
        hoi_blob['interaction_affinity_score']= torch.zeros((interaction_human_inds.shape[0], 1)).cuda(device_id)  ### 2 classisification score

        return hoi_blob

    @staticmethod
    def loss(hoi_blob):
        interaction_action_score = hoi_blob['interaction_action_score']
        device_id = interaction_action_score.get_device()

        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(device_id)
        interaction_action_loss = F.binary_cross_entropy_with_logits(
            interaction_action_score, interaction_action_labels)
        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.VCOCO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)
        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls


class PMFNet_Final(nn.Module):
    """
    add relative coordinate to parts
    Human Object Interaction.
    This module including Human-centric branch and Interaction branch of InteractNet.
    """
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()

        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.crop_size = cfg.VCOCO.PART_CROP_SIZE
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        # ToDo: cfg
        hidden_dim = cfg.VCOCO.MLP_HEAD_DIM
        # num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
        action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
        interaction_num_action_classes = action_mask.sum().item()

        interaction_fc1_dim_in = 3 * hidden_dim
        self.part_num = 17

        self.pose_fc1 = nn.Linear((self.part_num+1) * 258 * self.crop_size ** 2, 1024)
        self.pose_fc2 = nn.Linear(1024, hidden_dim*2)
        interaction_fc1_dim_in += hidden_dim*2

        ## semantic attention
        self.mlp = nn.Sequential(
            nn.Linear(3*64*64, 64),
            nn.ReLU(),
            nn.Linear(64, self.part_num)
        )

        self.pose_fc3 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2, 512)
        self.pose_fc4 = nn.Linear(512, hidden_dim)
        interaction_fc1_dim_in += hidden_dim

        self.human_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.object_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.union_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)

        self.interaction_fc1 = nn.Linear(interaction_fc1_dim_in, hidden_dim)
        self.interaction_action_score = nn.Linear(hidden_dim, interaction_num_action_classes)

        self.global_affinity = nn.Sequential(
            nn.Linear(interaction_fc1_dim_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def detectron_weight_mapping(self):
        # hc is human centric branch
        # io is interaction branch object part
        detectron_weight_mapping = {
            'human_fc1.weight': 'hc_fc1_w',
            'human_fc1.bias': 'hc_fc1_b',
            'human_fc2.weight': 'hc_fc2_w',
            'human_fc2.bias': 'hc_fc2_b',
            'human_action_score.weight': 'hc_score_w',
            'human_action_score.bias': 'hc_score_b',
            'interaction_fc1.weight': 'inter_fc1_w',
            'interaction_fc1.bias': 'inter_fc1_b',
        }
        return detectron_weight_mapping, []

    def forward(self, x, hoi_blob,):

        device_id = x[0].get_device()
        coord_x, coord_y = np.meshgrid(np.arange(x.shape[-1]), np.arange(x.shape[-2]))
        coords = np.stack((coord_x, coord_y), axis=0).astype(np.float32)
        coords = torch.from_numpy(coords).cuda(device_id) 
        x_coords = coords.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # 1 x 2 x H x W

        x_human = self.roi_xform(
            x, hoi_blob,
            blob_rois='human_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_object = self.roi_xform(
            x, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_union = self.roi_xform(
            x, hoi_blob,
            blob_rois='union_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x_union = x_union.view(x_union.size(0), -1)
        x_human = x_human.view(x_human.size(0), -1)
        x_object = x_object.view(x_object.size(0), -1)

        #x_object2 = x_object2.view(x_object2.size(0), -1)
        # get inds from numpy
        interaction_human_inds = torch.from_numpy(
            hoi_blob['interaction_human_inds']).long().cuda(device_id)
        interaction_object_inds = torch.from_numpy(
            hoi_blob['interaction_object_inds']).long().cuda(device_id)
        part_boxes = torch.from_numpy(
            hoi_blob['part_boxes']).cuda(device_id)

        x_human = F.relu(self.human_fc1(x_human[interaction_human_inds]), inplace=True)
        x_object = F.relu(self.object_fc1(x_object[interaction_object_inds]), inplace=True)
        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_interaction = torch.cat((x_human, x_object, x_union), dim=1)

        ## encode the pose information into x_interaction feature
        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)
        poseconfig = kps_pred.view(kps_pred.size(0), -1)
        # x_pose_line = kps_pred.view(kps_pred.size(0), -1)
        x_pose_line = F.relu(self.pose_fc3(poseconfig), inplace=True)
        x_pose_line = F.relu(self.pose_fc4(x_pose_line), inplace=True)
        # x_interaction1 = torch.cat((x_interaction, x_pose_line), dim=1)  ## to get global interaction affinity score

        x_new = torch.cat((x, x_coords), dim=1)
        # x_new = x
        x_object2 = self.roi_xform(
            x_new, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=self.crop_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        ## pose_attention feature, including part feature and object feature and geometry feature
        x_pose = self.crop_pose_map(x_new, part_boxes, hoi_blob['flag'], self.crop_size)
        # x_pose = torch.cat((x_pose, x_coords), dim=2) # N x 17 x 258 x 5 x 5
        x_pose = x_pose[interaction_human_inds]

        # x_object2 = torch.cat((x_object2, x_object2_coord), dim=1) # N x 258 x 5 x 5
        x_object2 = x_object2.unsqueeze(dim=1) # N x 1 x 258 x 5 x 5
        # N x 2 x 5 x 5 

        x_object2 = x_object2[interaction_object_inds]
        center_xy = x_object2[:,:, -2:, 2:3, 2:3] # N x 1 x 2 x 1 x 1
        x_pose[:, :, -2:] = x_pose[:, :, -2:] - center_xy # N x 1 x 2 x 5 x 5
        x_object2[:,:,-2:] = x_object2[:,:, -2:] - center_xy # N x 17 x 2 x 5 x 5
        x_pose = torch.cat((x_pose, x_object2), dim=1) # N x 18 x 258 x 5 x 5
        # N x 18 x 256 x 5 x 5

        semantic_atten = F.sigmoid(self.mlp(poseconfig))
        semantic_atten = semantic_atten.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # N x 17 x 1 x 1 x 1
        x_pose_new = torch.zeros(x_pose.shape).cuda(device_id)
        x_pose_new[:, :17] = x_pose[:, :17] * semantic_atten
        x_pose_new[:, 17] = x_pose[:, 17]

        ## fuse the pose attention information
        x_pose = x_pose_new.view(x_pose_new.shape[0], -1)
        x_pose = F.relu(self.pose_fc1(x_pose), inplace=True)
        x_pose = F.relu(self.pose_fc2(x_pose), inplace=True)
        x_interaction = torch.cat((x_interaction, x_pose, x_pose_line), dim=1)
        interaction_affinity_score = self.global_affinity(x_interaction)

        x_interaction = F.relu(self.interaction_fc1(x_interaction), inplace=True)
        interaction_action_score = self.interaction_action_score(x_interaction)

        hoi_blob['interaction_action_score'] = interaction_action_score  ### multi classification score
        hoi_blob['interaction_affinity_score']= interaction_affinity_score ### binary classisification score

        return hoi_blob

    def crop_pose_map(self, union_feats, part_boxes, flag, crop_size):
        triplets_num, part_num, _ = part_boxes.shape
        ret = torch.zeros((triplets_num, part_num, union_feats.shape[1], crop_size, crop_size)).cuda(
            union_feats.get_device())
        part_feats = RoIAlignFunction(crop_size, crop_size, self.spatial_scale, cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)(
            union_feats, part_boxes.view(-1, part_boxes.shape[-1])).view(ret.shape)

        valid_n, valid_p = np.where(flag > 0)
        if len(valid_n) > 0:
            ret[valid_n, valid_p] = part_feats[valid_n, valid_p]
        return ret

    @staticmethod
    def loss(hoi_blob):

        interaction_action_score = hoi_blob['interaction_action_score']
        interaction_affinity_score = hoi_blob['interaction_affinity_score']
        device_id = interaction_action_score.get_device()

        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(device_id)
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.VCOCO.ACTION_THRESH).type_as(interaction_action_labels)

        interaction_action_loss = F.binary_cross_entropy_with_logits(
            interaction_action_score, interaction_action_labels, size_average=True)

        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()
        interaction_affinity_label = torch.from_numpy(hoi_blob['interaction_affinity'].astype(np.float32)).cuda(
            device_id)
        # interaction_affinity_loss = F.cross_entropy(
        #     interaction_affinity_score, interaction_affinity_label)
        # interaction_affinity_preds = (interaction_affinity[:,1]>interaction_affinity[:,0]).type_as(interaction_affinity_label)
        interaction_affinity_loss = cfg.VCOCO.AFFINITY_WEIGHT * F.binary_cross_entropy_with_logits(
            interaction_affinity_score, interaction_affinity_label.unsqueeze(1), size_average=True)
        interaction_affinity_preds = (interaction_affinity_score.sigmoid() > cfg.VCOCO.ACTION_THRESH).type_as(
            interaction_affinity_label)
        interaction_affinity_cls = interaction_affinity_preds.eq(interaction_affinity_label).float().mean()

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls


class PMFNet_Final_bak(nn.Module):
    """
    add relative coordinate to parts
    Human Object Interaction.
    This module including Human-centric branch and Interaction branch of InteractNet.
    """

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()

        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.crop_size = cfg.VCOCO.PART_CROP_SIZE

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        # ToDo: cfg
        hidden_dim = cfg.VCOCO.MLP_HEAD_DIM
        # num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
        action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
        interaction_num_action_classes = action_mask.sum().item()

        interaction_fc1_dim_in = 3 * hidden_dim
        part_num = 17
        self.pose_fc1 = nn.Linear((part_num + 1) * 258 * self.crop_size ** 2, 1024)
        self.pose_fc2 = nn.Linear(1024, hidden_dim * 2)
        interaction_fc1_dim_in += hidden_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(3 * 64 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, part_num)
        )

        self.pose_fc3 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2, 512)
        self.pose_fc4 = nn.Linear(512, hidden_dim)
        interaction_fc1_dim_in += hidden_dim

        self.human_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.object_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.union_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)

        self.interaction_fc1 = nn.Linear(interaction_fc1_dim_in, hidden_dim)
        self.interaction_action_score = nn.Linear(hidden_dim, interaction_num_action_classes)

    def detectron_weight_mapping(self):
        # hc is human centric branch
        # io is interaction branch object part
        detectron_weight_mapping = {
            'human_fc1.weight': 'hc_fc1_w',
            'human_fc1.bias': 'hc_fc1_b',
            'human_fc2.weight': 'hc_fc2_w',
            'human_fc2.bias': 'hc_fc2_b',
            'human_action_score.weight': 'hc_score_w',
            'human_action_score.bias': 'hc_score_b',
            'interaction_fc1.weight': 'inter_fc1_w',
            'interaction_fc1.bias': 'inter_fc1_b',
        }
        return detectron_weight_mapping, []

    def forward(self, x, hoi_blob):

        device_id = x[0].get_device()

        coord_x, coord_y = np.meshgrid(np.arange(x.shape[-1]), np.arange(x.shape[-2]))
        coords = np.stack((coord_x, coord_y), axis=0).astype(np.float32)
        coords = torch.from_numpy(coords).cuda(device_id)
        x_coords = coords.unsqueeze(0)  # 1 x 2 x H x W

        x_human = self.roi_xform(
            x, hoi_blob,
            blob_rois='human_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_object = self.roi_xform(
            x, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_new = torch.cat((x, x_coords), dim=1)
        x_object2 = self.roi_xform(
            x_new, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=self.crop_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_human = x_human.view(x_human.size(0), -1)
        x_object = x_object.view(x_object.size(0), -1)

        # x_object2 = x_object2.view(x_object2.size(0), -1)
        # get inds from numpy
        interaction_human_inds = torch.from_numpy(
            hoi_blob['interaction_human_inds']).long().cuda(device_id)
        interaction_object_inds = torch.from_numpy(
            hoi_blob['interaction_object_inds']).long().cuda(device_id)
        part_boxes = torch.from_numpy(
            hoi_blob['part_boxes']).cuda(device_id)

        x_human = F.relu(self.human_fc1(x_human[interaction_human_inds]), inplace=True)
        x_object = F.relu(self.object_fc1(x_object[interaction_object_inds]), inplace=True)

        x_union = self.roi_xform(
            x, hoi_blob,
            blob_rois='union_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_union = x_union.view(x_union.size(0), -1)
        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_interaction = torch.cat((x_human, x_object, x_union), dim=1)



        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)

        poseconfig = kps_pred.view(kps_pred.size(0), -1)
        x_pose = self.crop_pose_map(x_new, part_boxes, hoi_blob['flag'], self.crop_size)

        # x_pose = torch.cat((x_pose, x_coords), dim=2) # N x 17 x 258 x 5 x 5
        x_pose = x_pose[interaction_human_inds]

        # x_object2 = torch.cat((x_object2, x_object2_coord), dim=1) # N x 258 x 5 x 5
        x_object2 = x_object2.unsqueeze(dim=1)  # N x 1 x 258 x 5 x 5
        # N x 2 x 5 x 5

        x_object2 = x_object2[interaction_object_inds]
        center_xy = x_object2[:, :, -2:, 2:3, 2:3]  # N x 1 x 2 x 1 x 1
        x_pose[:, :, -2:] = x_pose[:, :, -2:] - center_xy  # N x 1 x 2 x 5 x 5
        x_object2[:, :, -2:] = x_object2[:, :, -2:] - center_xy  # N x 17 x 2 x 5 x 5

        x_pose = torch.cat((x_pose, x_object2), dim=1)  # N x 18 x 258 x 5 x 5

        semantic_atten = F.sigmoid(self.mlp(poseconfig))
        semantic_atten = semantic_atten.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # N x 17 x 1 x 1 x 1
        x_pose_new = torch.zeros(x_pose.shape).cuda(device_id)
        x_pose_new[:, :17] = x_pose[:, :17] * semantic_atten
        x_pose_new[:, 17] = x_pose[:, 17]

        x_pose = x_pose_new.view(x_pose_new.shape[0], -1)
        x_pose = F.relu(self.pose_fc1(x_pose), inplace=True)
        x_pose = F.relu(self.pose_fc2(x_pose), inplace=True)
        x_interaction = torch.cat((x_interaction, x_pose), dim=1)

        x_pose_line = kps_pred.view(kps_pred.size(0), -1)
        x_pose_line = F.relu(self.pose_fc3(x_pose_line), inplace=True)
        x_pose_line = F.relu(self.pose_fc4(x_pose_line), inplace=True)
        x_interaction = torch.cat((x_interaction, x_pose_line), dim=1)

        x_interaction = F.relu(self.interaction_fc1(x_interaction), inplace=True)
        interaction_action_score = self.interaction_action_score(x_interaction)

        hoi_blob['interaction_action_score'] = interaction_action_score
        hoi_blob['interaction_affinity_score'] = torch.zeros((x_union.shape[0], 2)).cuda(device_id)

        return hoi_blob

    def crop_pose_map(self, union_feats, part_boxes, flag, crop_size):
        triplets_num, part_num, _ = part_boxes.shape
        ret = torch.zeros((triplets_num, part_num, union_feats.shape[1], crop_size, crop_size)).cuda(
            union_feats.get_device())
        part_feats = RoIAlignFunction(crop_size, crop_size, self.spatial_scale, cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)(
            union_feats, part_boxes.view(-1, part_boxes.shape[-1])).view(ret.shape)

        valid_n, valid_p = np.where(flag > 0)
        if len(valid_n) > 0:
            ret[valid_n, valid_p] = part_feats[valid_n, valid_p]
        # return ret.reshape(triplets_num, part_num, -1)
        return ret

    @staticmethod
    def loss(hoi_blob):

        interaction_action_score = hoi_blob['interaction_action_score']
        device_id = interaction_action_score.get_device()

        ''' for fine_grained action loss'''
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(device_id)
        interaction_action_loss = F.binary_cross_entropy_with_logits(
            interaction_action_score, interaction_action_labels)
        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.VCOCO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        return interaction_action_loss, interaction_action_accuray_cls