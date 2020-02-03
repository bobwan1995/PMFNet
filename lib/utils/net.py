import logging
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
import nn as mynn

logger = logging.getLogger(__name__)


def weighted_binary_cross_entropy_interaction(output, target, weights=None):
    '''
     weights: (A, 2), 0 for negative 1 for positive
     output: (N, A)
     target: (N, A)
     A is action number
    '''
    output = F.sigmoid(output)
    if weights is not None:
        assert len(weights.shape) == 2
        loss = weights[:, 1].unsqueeze(dim=0) * (target * torch.log(output+1e-8)) + \
               weights[:, 0].unsqueeze(dim=0) * ((1 - target) * torch.log(1 - output+1e-8))
    else:
        loss = target * torch.log(output+1e-8) + (1 - target) * torch.log(1 - output+1e-8)

    return torch.neg(torch.mean(loss))


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, relu=True, same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) * dilation / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualConv, self).__init__()

        self.residual_block = nn.Sequential(*[
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                   stride=1, relu=True, same_padding=True, bn=True, dilation=1),
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                   stride=1, relu=True, same_padding=True, bn=True, dilation=1),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                   stride=1, relu=True, same_padding=True, bn=True, dilation=1),
            ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x+self.residual_block(x))


# class SpacialConv(nn.Module):
#     def __init__(self, pooling_size=32, d_g=64):
#         super(SpacialConv, self).__init__()
#         self.pooling_size = pooling_size
#         self.conv1 = Conv2d(in_channels=2, out_channels=96, kernel_size=5, stride=2, relu=True, same_padding=True)
#         self.conv2 = Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=2, relu=False, same_padding=True)
#         self.conv3 = Conv2d(in_channels=128, out_channels=d_g, kernel_size=8, stride=1, relu=True, same_padding=False)
#
#     def forward(self, human_rois, object_rois, human_inds, object_inds, im_info, device_id):
#         human_mask, object_mask = self.generate_mask(human_rois, object_rois, im_info)
#         x = np.zeros((human_inds.size, 2, self.pooling_size, self.pooling_size))
#         x[:, 0, :, :] = human_mask[human_inds]
#         x[:, 1, :, :] = object_mask[object_inds]
#
#         x = torch.from_numpy(x).float().cuda(device_id)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.squeeze(-1).squeeze(-1)
#         return x
#
#     def generate_mask(self, human_rois, object_rois, im_info):
#         human_location_mask = np.zeros((human_rois.shape[0], self.pooling_size, self.pooling_size))
#         object_location_mask = np.zeros((object_rois.shape[0], self.pooling_size, self.pooling_size))
#
#         human_count = 0
#         object_count = 0
#         for i in range(im_info.shape[0]):
#             # h, w, scale
#             weights_t = self.pooling_size / np.tile(np.array([im_info[i, 1], im_info[i, 0]]), 2)
#             human_rois_t = (human_rois[human_rois[:, 0] == i][:, 1:] * weights_t).astype(np.int32)
#             object_rois_t = (object_rois[object_rois[:, 0] == i][:, 1:] * weights_t).astype(np.int32)
#             for h_roi in human_rois_t:
#                 human_location_mask[human_count, h_roi[1]:h_roi[3], h_roi[0]:h_roi[2]] = 1
#                 human_count += 1
#             for obj_roi in object_rois_t:
#                 object_location_mask[object_count, obj_roi[1]:obj_roi[3], obj_roi[0]:obj_roi[2]] = 1
#                 object_count += 1
#
#         return human_location_mask, object_location_mask

class SpacialConv(nn.Module):
    def __init__(self, out_dim=64):
        super(SpacialConv, self).__init__()
        self.conv1 = Conv2d(in_channels=2, out_channels=96, kernel_size=5, stride=2, relu=True, same_padding=True)
        self.conv2 = Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=2, relu=False, same_padding=True)
        self.conv3 = Conv2d(in_channels=128, out_channels=out_dim, kernel_size=8, stride=1, relu=True, same_padding=False)

    def forward(self, x, device_id):
        x = torch.from_numpy(x).float().cuda(device_id)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(-1).squeeze(-1)
        return x


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, beta=1.0):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    1 / N * sum_i alpha_out[i] * SmoothL1(alpha_in[i] * (y_hat[i] - y[i])).
    N is the number of batch elements in the input predictions
    """
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < beta).detach().float()
    in_loss_box = smoothL1_sign * 0.5 * torch.pow(in_box_diff, 2) / beta + \
                  (1 - smoothL1_sign) * (abs_in_box_diff - (0.5 * beta))
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    N = loss_box.size(0)  # batch size
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def decay_learning_rate(optimizer, cur_lr, decay_rate):
    """Decay learning rate"""
    new_lr = cur_lr * decay_rate
    # ratio = _get_lr_change_ratio(cur_lr, new_lr)
    ratio = 1 / decay_rate
    if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
        logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
    # Update learning rate, note that different parameter may have different learning rate
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
        new_lr = decay_rate * param_group['lr']
        param_group['lr'] = new_lr
        if cfg.SOLVER.TYPE in ['SGD']:
            if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                    ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
                _CorrectMomentum(optimizer, param_group['params'], new_lr / cur_lr)

def update_learning_rate(optimizer, cur_lr, new_lr):
    """Update learning rate"""
    if cur_lr != new_lr:
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
            logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
        # Update learning rate, note that different parameter may have different learning rate
        param_keys = []
        for ind, param_group in enumerate(optimizer.param_groups):
            if ind == 0 or ind == 2:  # param hoi
                param_group['lr'] = new_lr
            elif ind == 1 or ind == 3:  # param faster
                param_group['lr'] = new_lr * cfg.SOLVER.FASTER_RCNN_WEIGHT

            if cfg.SOLVER.BIAS_DOUBLE_LR and (ind == 2 or ind == 3): # param bias
                param_group['lr'] = param_group['lr'] * 2.

            param_keys += param_group['params']
        if cfg.SOLVER.TYPE in ['SGD'] and cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            _CorrectMomentum(optimizer, param_keys, new_lr / cur_lr)


def _CorrectMomentum(optimizer, param_keys, correction):
    """The MomentumSGDUpdate op implements the update V as

        V := mu * V + lr * grad,

    where mu is the momentum factor, lr is the learning rate, and grad is
    the stochastic gradient. Since V is not defined independently of the
    learning rate (as it should ideally be), when the learning rate is
    changed we should scale the update history V in order to make it
    compatible in scale with lr * grad.
    """
    logger.info('Scaling update history by %.6f (new lr / old lr)', correction)
    for p_key in param_keys:
        optimizer.state[p_key]['momentum_buffer'] *= correction


def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio


def affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid


def save_ckpt(output_dir, args, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_{}_{}.pth'.format(args.epoch, args.step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    # TODO: (maybe) Do not save redundant shared params
    # model_state_dict = model.state_dict()
    torch.save({
        'epoch': args.epoch,
        'step': args.step,
        'iters_per_epoch': args.iters_per_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


# def load_ckpt(model, ckpt):
#     """Load checkpoint"""
#     state_dict = ckpt
#     # mapping, _ = model.detectron_weight_mapping
#     # state_dict = {}
#     # for name in ckpt:
#     #     if mapping[name]:
#     #         state_dict[name] = ckpt[name]

#     model.load_state_dict(state_dict, strict=False)

def load_ckpt(model, ckpt):
    """Load checkpoint"""
    # mapping, _ = model.detectron_weight_mapping
    state_dict = model.state_dict()
    for name in state_dict.keys():
        value = ckpt.get(name)
        if value is not None:
            if value.shape == state_dict[name].shape:
                state_dict[name] = ckpt[name]
            else:
                print('don\'t load {} because size mismatch'.format(name))
        else:
            print('don\'t load {} because not exist in ckpt'.format(name))
    model.load_state_dict(state_dict, strict=False)
    

def get_group_gn(dim):
    """
    get number of groups used by GroupNorm, based on number of channels
    """
    dim_per_gp = cfg.GROUP_NORM.DIM_PER_GP
    num_groups = cfg.GROUP_NORM.NUM_GROUPS

    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0
        group_gn = num_groups
    return group_gn


def load_krcnn_from_faster(model, ckpt):
    # Try initialize keypoint-rcnn from faster-rcnn
    mapping, _ = model.detectron_weight_mapping
    state_dict = {}
    for name in ckpt:
        if mapping[name]:
            if name.startswith('Box_Outs.cls_score'):
                state_dict[name] = ckpt[name][:2]
            elif name.startswith('Box_Outs.bbox_pred'):
                state_dict[name] = ckpt[name][:8]
            else:
                state_dict[name] = ckpt[name]
    model.load_state_dict(state_dict, strict=False)
