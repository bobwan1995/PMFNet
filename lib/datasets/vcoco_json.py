# AUTORIGHTS
# ---------------------------------------------------------
# Copyright (c) 2017, Saurabh Gupta
#
# This file is part of the VCOCO dataset hooks and is available
# under the terms of the Simplified BSD License provided in
# LICENSE. Please retain this notice and LICENSE if you use
# this file (or any portion of it) in your project.
# ---------------------------------------------------------

# vsrl_data is a dictionary for each action class:
# image_id       - Nx1
# ann_id         - Nx1
# label          - Nx1
# action_name    - string
# role_name      - ['agent', 'obj', 'instr']
# role_object_id - N x K matrix, obviously [:,0] is same as ann_id

import numpy as np
import json
import pdb
import os
import pickle
import matplotlib.pyplot as plt

from core.config import cfg
import ipdb

class VCOCO(object):

    def __init__(self, vsrl_annot_file):
        """Input:
        vslr_annot_file: path to the vcoco annotations
        coco_annot_file: path to the coco annotations
        split_file: image ids for split
        """
        self.vcoco_data = self._load_vcoco(vsrl_annot_file)
        # self.image_ids = np.loadtxt(split_file)
        # simple check
        # negative iamges is removed, so it's unnecessary
        # assert np.all(np.equal(np.sort(np.unique(self.vcoco_data[0]['image_id'])),
        #                        np.sort(self.image_ids.astype(int))))

        self._init_vcoco()

    def _init_vcoco(self):
        actions = [x['action_name'] for x in self.vcoco_data]
        assert len(actions) == cfg.VCOCO.NUM_ACTION_CLASSES
        # ToDo: excluding `point` just same as the paper
        roles = [['agent'] if x['action_name'] == 'point' else x['role_name'] for x in self.vcoco_data]
        for x in roles:
            assert len(x) <= cfg.VCOCO.NUM_TARGET_OBJECT_TYPES + 1
        self.actions = actions
        self.actions_to_id_map = {v: i for i, v in enumerate(self.actions)}
        self.num_actions = len(self.actions)
        self.roles = roles

    def _load_vcoco(self, vcoco_file):
        print('loading vcoco annotations...')
        with open(vcoco_file, 'r') as f:
            vsrl_data = json.load(f)
        for i in range(len(vsrl_data)):
            vsrl_data[i]['role_object_id'] = \
                np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']), -1)).T
            for j in ['ann_id', 'label', 'image_id']:
                vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1, 1))
        return vsrl_data

    def get_vsrl_data(self, ann_id, ann_ids, objs):
        """ Get VSRL data for ann_id."""
        action_id = -np.ones((self.num_actions), dtype=np.int32)
        # role id of anns mapping to action
        role_id = -np.ones((self.num_actions, cfg.VCOCO.NUM_TARGET_OBJECT_TYPES), dtype=np.int32)
        # check if ann_id in vcoco annotations
        in_vcoco = np.where(self.vcoco_data[0]['ann_id'] == ann_id)[0]
        if in_vcoco.size > 0:
            action_id[:] = 0
            role_id[:] = -1
        else:
            return action_id, role_id
        for i, x in enumerate(self.vcoco_data):
            assert x['action_name'] == self.actions[i]
            has_label = np.where(np.logical_and(x['ann_id'] == ann_id, x['label'] == 1))[0]
            if has_label.size > 0:
                action_id[i] = 1
                assert has_label.size == 1
                rids = x['role_object_id'][has_label]
                assert rids[0, 0] == ann_id
                for j in range(1, rids.shape[1]):
                    if rids[0, j] == 0:
                        # no role
                        continue
                    # get role object id of bbox in one image
                    aid = np.where(ann_ids == rids[0, j])[0]
                    assert aid.size > 0
                    role_id[i, j - 1] = aid
        return action_id, role_id

    @staticmethod
    def _collect_detections_for_image(dets, image_id):
        #print('image_id sss')
        #import pdb
        #pdb.set_trace()
        det = dets[image_id]
        return det['agents'], det['roles']

    def do_eval(self, dets, vcocodb, detections_file=None, ovr_thresh=0.5, plot_save_path=None):

        if dets is None and detections_file:
            with open(detections_file, 'rb') as f:
                dets = pickle.load(f)
        # Plot pr curve
        # if plot_save_path is not None:
        #     plot_save_path = os.path.join(plot_save_path, 'pr_curve')
        #     if not os.path.exists(plot_save_path):
        #         os.mkdir(plot_save_path)

        # self._do_agent_eval(vcocodb, dets, ovr_thresh=ovr_thresh, plot_save_path=plot_save_path)
        role_ap = self._do_role_eval(vcocodb, dets, ovr_thresh=ovr_thresh, eval_type='scenario_1', plot_save_path=plot_save_path)
        #self._do_role_eval(vcocodb, dets, ovr_thresh=ovr_thresh, eval_type='scenario_2', plot_save_path=plot_save_path)

        # self._visualize_error(vcocodb, dets, ovr_thresh=ovr_thresh, eval_type='scenario_1')
        # self._visualize_error(vcocodb, dets, ovr_thresh=ovr_thresh, eval_type='scenario_2')
        # self._visualize_agent_error_human_centric(vcocodb, dets, ovr_thresh=ovr_thresh)
        #self._visualize_error_human_centric(vcocodb, dets, ovr_thresh=ovr_thresh, eval_type='scenario_1')
        #self._visualize_error_human_centric(vcocodb, dets, ovr_thresh=ovr_thresh, eval_type='scenario_2')
        return role_ap

    def _do_role_eval(self, vcocodb, dets, ovr_thresh=0.5, eval_type='scenario_1', plot_save_path=None):
        if plot_save_path is not None:
            plot_save_path = os.path.join(plot_save_path, 'role_{}_eval'.format(eval_type))
            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)

        tp = [[[] for r in range(cfg.VCOCO.NUM_TARGET_OBJECT_TYPES)] for a in range(self.num_actions)]
        fp = [[[] for r in range(cfg.VCOCO.NUM_TARGET_OBJECT_TYPES)] for a in range(self.num_actions)]
        sc = [[[] for r in range(cfg.VCOCO.NUM_TARGET_OBJECT_TYPES)] for a in range(self.num_actions)]

        npos = np.zeros((self.num_actions), dtype=np.float32)
        for i in range(len(vcocodb)):
            image_id = vcocodb[i]['id']
            # if image_id not in dets.keys():
            #   continue
            
            gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]
            # person boxes
            gt_boxes = vcocodb[i]['boxes'][gt_inds]
            gt_actions = vcocodb[i]['gt_actions'][gt_inds]
            # some person instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)
            assert np.all(gt_actions[np.where(ignore == True)[0]] == -1)

            for aid in range(self.num_actions):
                npos[aid] += np.sum(gt_actions[:, aid] == 1)

            pred_agents, pred_roles = self._collect_detections_for_image(dets, image_id)
            #print('pred agents: ', pred_agents.shape)
            #print('pred roles: ', pred_roles.shape)

            for aid in range(self.num_actions):
                if len(self.roles[aid]) < 2:
                    # if action has no role, then no role AP computed
                    continue

                for rid in range(len(self.roles[aid]) - 1):

                    # keep track of detected instances for each action for each role
                    covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)

                    # get gt roles for action and role
                    gt_role_inds = vcocodb[i]['gt_role_id'][gt_inds, aid, rid]
                    gt_roles = -np.ones_like(gt_boxes)
                    for j in range(gt_boxes.shape[0]):
                        if gt_role_inds[j] > -1:
                            gt_roles[j] = vcocodb[i]['boxes'][gt_role_inds[j]]

                    agent_boxes = pred_agents[:, :4]
                    role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4, rid]
                    agent_scores = pred_roles[:, 5 * aid + 4, rid]

                    valid = np.where(np.isnan(agent_scores) == False)[0]
                    agent_scores = agent_scores[valid]
                    agent_boxes = agent_boxes[valid, :]
                    role_boxes = role_boxes[valid, :]

                    idx = agent_scores.argsort()[::-1]

                    for j in idx:
                        pred_box = agent_boxes[j, :]
                        #print('pred_box:', pred_box.shape)
                        #print('gt_boxes:', gt_boxes.shape)
                        #pdb.set_trace()
                        overlaps = get_overlap(gt_boxes, pred_box)

                        # matching happens based on the person
                        jmax = overlaps.argmax()
                        ovmax = overlaps.max()

                        # if matched with an instance with no annotations
                        # continue
                        if ignore[jmax]:
                            continue

                        # overlap between predicted role and gt role
                        if np.all(gt_roles[jmax, :] == -1):  # if no gt role
                            if eval_type == 'scenario_1':
                                if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                                    # if no role is predicted, mark it as correct role overlap
                                    ov_role = 1.0
                                else:
                                    # if a role is predicted, mark it as false
                                    ov_role = 0.0
                            elif eval_type == 'scenario_2':
                                # if no gt role, role prediction is always correct, irrespective of the actual predition
                                ov_role = 1.0
                            else:
                                raise ValueError('Unknown eval type')
                        else:
                            ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])

                        is_true_action = (gt_actions[jmax, aid] == 1)
                        sc[aid][rid].append(agent_scores[j])
                        if is_true_action and (ovmax >= ovr_thresh) and (ov_role >= ovr_thresh):
                            if covered[jmax]:
                                fp[aid][rid].append(1)
                                tp[aid][rid].append(0)
                            else:
                                fp[aid][rid].append(0)
                                tp[aid][rid].append(1)
                                covered[jmax] = True
                        else:
                            fp[aid][rid].append(1)
                            tp[aid][rid].append(0)

        # compute ap for each action
        role_ap = np.zeros((self.num_actions, cfg.VCOCO.NUM_TARGET_OBJECT_TYPES), dtype=np.float32)
        role_ap[:] = np.nan
        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2:
                continue
            for rid in range(len(self.roles[aid]) - 1):
                a_fp = np.array(fp[aid][rid], dtype=np.float32)
                a_tp = np.array(tp[aid][rid], dtype=np.float32)
                a_sc = np.array(sc[aid][rid], dtype=np.float32)
                # sort in descending score order
                idx = a_sc.argsort()[::-1]
                a_fp = a_fp[idx]
                a_tp = a_tp[idx]
                a_sc = a_sc[idx]

                a_fp = np.cumsum(a_fp)
                a_tp = np.cumsum(a_tp)
                if npos[aid]==0:
                    npos[aid]=10000.
                rec = a_tp / float(npos[aid])
                # check 
                assert (np.amax(rec) <= 1)
                prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
                if plot_save_path is not None:
                    plot_save_path_this_aid_rid = os.path.join(
                        plot_save_path, self.actions[aid] + '-' + self.roles[aid][rid + 1] + '.png')
                else:
                    plot_save_path_this_aid_rid = None
                role_ap[aid, rid] = voc_ap(rec, prec, plot_save_path_this_aid_rid)

        print('---------Reporting Role AP (%)------------------')
        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2: continue
            for rid in range(len(self.roles[aid]) - 1):
                print('{: >23}: AP = {:0.4f} (#pos = {:d})'.format(self.actions[aid] + '-' + self.roles[aid][rid + 1],
                                                                   role_ap[aid, rid], int(npos[aid])))

        ret = np.nanmean(role_ap)
        print('Average Role [%s] AP = %.4f' % (eval_type, ret))
        print('---------------------------------------------')
        return ret

    def _do_agent_eval(self, vcocodb, dets, ovr_thresh=0.5, plot_save_path=None):

        if plot_save_path is not None:
            plot_save_path = os.path.join(plot_save_path, 'agent_eval')
            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)

        tp = [[] for a in range(self.num_actions)]
        fp = [[] for a in range(self.num_actions)]
        sc = [[] for a in range(self.num_actions)]

        npos = np.zeros((self.num_actions), dtype=np.float32)

        for i in range(len(vcocodb)):
            image_id = vcocodb[i]['id']
            gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]
            # person boxes
            gt_boxes = vcocodb[i]['boxes'][gt_inds]
            gt_actions = vcocodb[i]['gt_actions'][gt_inds]
            # some peorson instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)

            for aid in range(self.num_actions):
                npos[aid] += np.sum(gt_actions[:, aid] == 1)

            pred_agents, _ = self._collect_detections_for_image(dets, image_id)

            for aid in range(self.num_actions):

                # keep track of detected instances for each action
                covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)

                agent_scores = pred_agents[:, 4 + aid]
                agent_boxes = pred_agents[:, :4]
                # remove NaNs
                valid = np.where(np.isnan(agent_scores) == False)[0]
                agent_scores = agent_scores[valid]
                agent_boxes = agent_boxes[valid, :]

                # sort in descending order
                idx = agent_scores.argsort()[::-1]

                for j in idx:
                    pred_box = agent_boxes[j, :]
                    overlaps = get_overlap(gt_boxes, pred_box)

                    jmax = overlaps.argmax()
                    ovmax = overlaps.max()

                    # if matched with an instance with no annotations
                    # continue
                    if ignore[jmax]:
                        continue

                    is_true_action = (gt_actions[jmax, aid] == 1)

                    sc[aid].append(agent_scores[j])
                    if is_true_action and (ovmax >= ovr_thresh):
                        if covered[jmax]:
                            fp[aid].append(1)
                            tp[aid].append(0)
                        else:
                            fp[aid].append(0)
                            tp[aid].append(1)
                            covered[jmax] = True
                    else:
                        fp[aid].append(1)
                        tp[aid].append(0)

        # compute ap for each action
        agent_ap = np.zeros((self.num_actions), dtype=np.float32)
        for aid in range(self.num_actions):
            a_fp = np.array(fp[aid], dtype=np.float32)
            a_tp = np.array(tp[aid], dtype=np.float32)
            a_sc = np.array(sc[aid], dtype=np.float32)
            # sort in descending score order
            idx = a_sc.argsort()[::-1]
            a_fp = a_fp[idx]
            a_tp = a_tp[idx]
            a_sc = a_sc[idx]

            a_fp = np.cumsum(a_fp)
            a_tp = np.cumsum(a_tp)
            rec = a_tp / float(npos[aid])
            # check
            assert (np.amax(rec) <= 1)
            prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
            if plot_save_path is not None:
                plot_save_path_this_aid = os.path.join(plot_save_path, self.actions[aid] + '.png')
            else:
                plot_save_path_this_aid = None
            agent_ap[aid] = voc_ap(rec, prec, plot_save_path_this_aid)

        print('---------Reporting Agent AP (%)------------------')
        for aid in range(self.num_actions):
            print(
                '{: >20}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid], agent_ap[aid] * 100.0, int(npos[aid])))
        print('Average Agent AP = %.2f' % (np.nansum(agent_ap) * 100.00 / self.num_actions))
        print('---------------------------------------------')

    def _visualize_error(self, vcocodb, dets, ovr_thresh=0.5, eval_type='scenario_1'):

        tp = [[[] for r in range(2)] for a in range(self.num_actions)]
        fp1 = [[[] for r in range(2)] for a in range(self.num_actions)]  # incorrect label
        fp2 = [[[] for r in range(2)] for a in range(self.num_actions)]  # bck
        fp3 = [[[] for r in range(2)] for a in range(self.num_actions)]  # person misloc
        fp4 = [[[] for r in range(2)] for a in range(self.num_actions)]  # obj misloc
        fp5 = [[[] for r in range(2)] for a in range(self.num_actions)]  # duplicate detection
        fp6 = [[[] for r in range(2)] for a in range(self.num_actions)]  # mis-grouping
        fp7 = [[[] for r in range(2)] for a in range(self.num_actions)]  # occlusion
        sc = [[[] for r in range(2)] for a in range(self.num_actions)]

        npos = np.zeros((self.num_actions), dtype=np.float32)  # A + B
        ndet = np.zeros((self.num_actions, 2), dtype=np.float32)  # B + C
        Test_occlusion = {}

        for i in range(len(vcocodb)):

            image_id = vcocodb[i][
                'id']  # img ID, not the full name (e.g. id= 165, 'file_name' = COCO_train2014_000000000165.jpg )
            gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]  # index of the person's box among all object boxes
            # person boxes
            gt_boxes = vcocodb[i]['boxes'][gt_inds]  # Nx4 all person's boxes in this image
            gt_actions = vcocodb[i]['gt_actions'][
                gt_inds]  # Nx26 binary array indicating the actions performed by this person

            # some peorson instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)
            assert np.all(gt_actions[np.where(ignore == True)[0]] == -1)

            for aid in range(self.num_actions):
                npos[aid] += np.sum(
                    gt_actions[:, aid] == 1)  # how many actions are involved in this image(for all the human)

            pred_agents, pred_roles = self._collect_detections_for_image(dets, image_id)
            # pred_agents Mx30
            # pred_roles Mx(5*26)x2
            for aid in range(self.num_actions):
                if len(self.roles[aid]) < 2:
                    # if action has no role, then no role AP computed
                    continue

                for rid in range(len(self.roles[aid]) - 1):  # rid = 0, instr; rid = 1, obj

                    # keep track of detected instances for each action for each role. Is this gt_human used or not.
                    covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)

                    # get gt roles for action and role
                    gt_role_inds = vcocodb[i]['gt_role_id'][
                        gt_inds, aid, rid]  # Nx1 index of the object among all detected objects related to this action. -1 means missing object.
                    gt_roles = -np.ones_like(gt_boxes)  # Nx4 [-1, -1, -1, -1] means gt missing object
                    for j in range(gt_boxes.shape[0]):  # loop all gt human instance
                        if gt_role_inds[j] > -1:  #
                            gt_roles[j] = vcocodb[i]['boxes'][gt_role_inds[j]]

                    agent_boxes = pred_agents[:, :4]  # Mx4 all detected human box
                    role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4,
                                 rid]  # Mx4 detected object(role) box for this human and action
                    agent_scores = pred_roles[:, 5 * aid + 4,
                                   rid]  # Mx1, action score for this human, object and action

                    if role_boxes.shape[0] == 0: continue

                    # ToDo: different with iCAN
                    # valid = np.where(np.isnan(role_boxes).any() == False)[0]
                    valid = np.where(np.isnan(agent_scores) == False)[0]

                    agent_scores = agent_scores[valid]
                    agent_boxes = agent_boxes[valid, :]
                    role_boxes = role_boxes[valid, :]

                    # ndet[aid][rid] += agent_boxes.shape[0]

                    # sort in descending order
                    idx = agent_scores.argsort()[::-1]  # A action can be done by multiple human.
                    for j in idx:  # in this image, this action with highest action score
                        pred_box = agent_boxes[j, :]
                        overlaps = get_overlap(gt_boxes, pred_box)  # gt_boxes: gt human box

                        jmax = overlaps.argmax()  # which gt_box best matches this detected box
                        ovmax = overlaps.max()

                        # if matched with an instance with no annotations
                        # continue
                        if ignore[jmax]:
                            continue

                        # overlap between predicted role and gt role
                        if np.all(gt_roles[jmax, :] == -1):  # if no gt role
                            if eval_type == 'scenario_1':
                                if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                                    # if no role is predicted, mark it as correct role overlap
                                    ov_role = 1.0
                                else:
                                    # if a role is predicted, mark it as false
                                    ov_role = -1.0
                            elif eval_type == 'scenario_2':
                                # if no gt role, role prediction is always correct, irrespective of the actual predition
                                ov_role = 1.0
                            else:
                                raise ValueError('Unknown eval type')
                        else:
                            ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])

                        is_true_action = (gt_actions[jmax, aid] == 1)  # Is this gt human actually doing this action?
                        sc[aid][rid].append(agent_scores[j])
                        ndet[aid][rid] += 1
                        if np.all(gt_actions[:,
                                  aid] == 0):  # All gt are not this action class. All detections are incorrect labels.
                            fp1[aid][rid].append(1)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                            continue
                        elif is_true_action == False:  # This detection j is a incorrect label
                            fp1[aid][rid].append(1)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        elif (ovmax < 0.1):  # bck
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(1)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        elif (ovmax < 0.5) & (ovmax >= 0.1):  # person misloc
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(1)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        elif (ovmax >= 0.5) & (ov_role == -1.0):  # occlusion
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(1)
                            tp[aid][rid].append(0)
                        elif (ovmax >= 0.5) & (0 <= ov_role <= 0.1):  # mis-grouping
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(1)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        elif (ovmax >= 0.5) & (0.1 <= ov_role < 0.5):  # obj misloc
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(1)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        elif (ovmax >= 0.5) & (ov_role >= 0.5):  # true positive
                            if not covered[jmax]:
                                fp1[aid][rid].append(0)
                                fp2[aid][rid].append(0)
                                fp3[aid][rid].append(0)
                                fp4[aid][rid].append(0)
                                fp5[aid][rid].append(0)
                                fp6[aid][rid].append(0)
                                fp7[aid][rid].append(0)
                                tp[aid][rid].append(1)
                                covered[jmax] = True
                            else:
                                fp1[aid][rid].append(0)
                                fp2[aid][rid].append(0)
                                fp3[aid][rid].append(0)
                                fp4[aid][rid].append(0)
                                fp5[aid][rid].append(1)
                                fp6[aid][rid].append(0)
                                fp7[aid][rid].append(0)
                                tp[aid][rid].append(0)

        fp_inc = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_bck = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_Hmis = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_Omis = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_dupl = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_misg = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_occl = np.zeros((self.num_actions, 2), dtype=np.float32)
        rec = np.zeros((self.num_actions, 2), dtype=np.float32)
        prec = np.zeros((self.num_actions, 2), dtype=np.float32)
        tp_ = np.zeros((self.num_actions, 2), dtype=np.float32)

        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2:
                continue

            for rid in range(len(self.roles[aid]) - 1):
                a_fp1 = np.array(fp1[aid][rid], dtype=np.float32)
                a_fp2 = np.array(fp2[aid][rid], dtype=np.float32)
                a_fp3 = np.array(fp3[aid][rid], dtype=np.float32)
                a_fp4 = np.array(fp4[aid][rid], dtype=np.float32)
                a_fp5 = np.array(fp5[aid][rid], dtype=np.float32)
                a_fp6 = np.array(fp6[aid][rid], dtype=np.float32)
                a_fp7 = np.array(fp7[aid][rid], dtype=np.float32)
                a_sc = np.array(sc[aid][rid], dtype=np.float32)
                a_tp = np.array(tp[aid][rid], dtype=np.float32)

                # sort in descending score order
                idx = a_sc.argsort()[::-1]
                a_fp1 = a_fp1[idx]
                a_fp2 = a_fp2[idx]
                a_fp3 = a_fp3[idx]
                a_fp4 = a_fp4[idx]
                a_fp5 = a_fp5[idx]
                a_fp6 = a_fp6[idx]
                a_fp7 = a_fp7[idx]
                a_tp = a_tp[idx]
                a_sc = a_sc[idx]

                # min(# GT, # not zero)
                num_inst = int(min(npos[aid], len(a_sc)))

                a_fp1 = a_fp1[:num_inst]
                a_fp2 = a_fp2[:num_inst]
                a_fp3 = a_fp3[:num_inst]
                a_fp4 = a_fp4[:num_inst]
                a_fp5 = a_fp5[:num_inst]
                a_fp6 = a_fp6[:num_inst]
                a_fp7 = a_fp7[:num_inst]
                a_tp = a_tp[:num_inst]
                a_sc = a_sc[:num_inst]

                frac_fp1 = np.sum(a_fp1) / (num_inst - np.sum(a_tp))
                frac_fp2 = np.sum(a_fp2) / (num_inst - np.sum(a_tp))
                frac_fp3 = np.sum(a_fp3) / (num_inst - np.sum(a_tp))
                frac_fp4 = np.sum(a_fp4) / (num_inst - np.sum(a_tp))
                frac_fp5 = np.sum(a_fp5) / (num_inst - np.sum(a_tp))
                frac_fp6 = np.sum(a_fp6) / (num_inst - np.sum(a_tp))
                frac_fp7 = np.sum(a_fp7) / (num_inst - np.sum(a_tp))

                tp_[aid, rid] = np.sum(a_tp)
                rec[aid, rid] = np.sum(a_tp) / float(npos[aid])
                prec[aid, rid] = np.sum(a_tp) / np.maximum(
                    np.sum(a_fp1) + np.sum(a_fp2) + np.sum(a_fp3) + np.sum(a_fp4) + np.sum(a_fp5) + np.sum(
                        a_fp6) + np.sum(a_fp7) + np.sum(a_tp), np.finfo(np.float64).eps)

                fp_inc[aid, rid] = frac_fp1
                fp_bck[aid, rid] = frac_fp2
                fp_Hmis[aid, rid] = frac_fp3
                fp_Omis[aid, rid] = frac_fp4
                fp_dupl[aid, rid] = frac_fp5
                fp_misg[aid, rid] = frac_fp6
                fp_occl[aid, rid] = frac_fp7

        print(
            '--------------------------------------------Reporting Error Analysis (%)-----------------------------------------------')
        print('{: >27} {:} {:} {:} {:} {:} {:}'.format(' ', 'inc', 'bck', 'H_mis', 'O_mis', 'mis-gr', 'occl'))
        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2: continue
            for rid in range(len(self.roles[aid]) - 1):
                print(
                    '{: >23}: {:6.2f} {:4.2f} {:4.2f} {:5.2f} {:5.2f} {:5.2f} (rec:{:5.2f} = #tp:{:4d}/#pos:{:4d}) (prec:{:5.2f} = #tp:{:4d}/#det:{:4d})'.format(
                        self.actions[aid] + '-' + self.roles[aid][rid + 1],
                        fp_inc[aid, rid] * 100.0,
                        fp_bck[aid, rid] * 100.0,
                        fp_Hmis[aid, rid] * 100.0,
                        fp_Omis[aid, rid] * 100.0,
                        fp_misg[aid, rid] * 100.0,
                        fp_occl[aid, rid] * 100.0,
                        rec[aid, rid] * 100.0,
                        int(tp_[aid, rid]),
                        int(npos[aid]),
                        prec[aid, rid] * 100.0,
                        int(tp_[aid, rid]),
                        int(ndet[aid, rid])))

    def _visualize_error_human_centric(self, vcocodb, dets, ovr_thresh=0.5, eval_type='scenario_1'):
        """
        Visualize role error human centric.
        Human centric means we count human location miss before interaction mistake.
        If human location is correct;
        elif action is correct;
        elif target location is correct;
        elif occlusion.
        :param vcocodb:
        :param dets:
        :param ovr_thresh:
        :param eval_type:
        :return:
        """
        tp = [[[] for r in range(2)] for a in range(self.num_actions)]
        fp1 = [[[] for r in range(2)] for a in range(self.num_actions)]  # incorrect label
        fp2 = [[[] for r in range(2)] for a in range(self.num_actions)]  # bck
        fp3 = [[[] for r in range(2)] for a in range(self.num_actions)]  # person misloc
        fp4 = [[[] for r in range(2)] for a in range(self.num_actions)]  # obj misloc
        fp5 = [[[] for r in range(2)] for a in range(self.num_actions)]  # duplicate detection
        fp6 = [[[] for r in range(2)] for a in range(self.num_actions)]  # mis-grouping
        fp7 = [[[] for r in range(2)] for a in range(self.num_actions)]  # occlusion
        sc = [[[] for r in range(2)] for a in range(self.num_actions)]

        npos = np.zeros((self.num_actions), dtype=np.float32)  # A + B
        ndet = np.zeros((self.num_actions, 2), dtype=np.float32)  # B + C
        Test_occlusion = {}

        for i in range(len(vcocodb)):

            image_id = vcocodb[i][
                'id']  # img ID, not the full name (e.g. id= 165, 'file_name' = COCO_train2014_000000000165.jpg )
            gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]  # index of the person's box among all object boxes
            # person boxes
            gt_boxes = vcocodb[i]['boxes'][gt_inds]  # Nx4 all person's boxes in this image
            gt_actions = vcocodb[i]['gt_actions'][
                gt_inds]  # Nx26 binary array indicating the actions performed by this person

            # some peorson instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)
            assert np.all(gt_actions[np.where(ignore == True)[0]] == -1)

            for aid in range(self.num_actions):
                npos[aid] += np.sum(
                    gt_actions[:, aid] == 1)  # how many actions are involved in this image(for all the human)

            pred_agents, pred_roles = self._collect_detections_for_image(dets, image_id)
            # pred_agents Mx30
            # pred_roles Mx(5*26)x2
            for aid in range(self.num_actions):
                if len(self.roles[aid]) < 2:
                    # if action has no role, then no role AP computed
                    continue

                for rid in range(len(self.roles[aid]) - 1):  # rid = 0, instr; rid = 1, obj

                    # keep track of detected instances for each action for each role. Is this gt_human used or not.
                    covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)

                    # get gt roles for action and role
                    gt_role_inds = vcocodb[i]['gt_role_id'][
                        gt_inds, aid, rid]  # Nx1 index of the object among all detected objects related to this action. -1 means missing object.
                    gt_roles = -np.ones_like(gt_boxes)  # Nx4 [-1, -1, -1, -1] means gt missing object
                    for j in range(gt_boxes.shape[0]):  # loop all gt human instance
                        if gt_role_inds[j] > -1:  #
                            gt_roles[j] = vcocodb[i]['boxes'][gt_role_inds[j]]

                    agent_boxes = pred_agents[:, :4]  # Mx4 all detected human box
                    role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4,
                                 rid]  # Mx4 detected object(role) box for this human and action
                    agent_scores = pred_roles[:, 5 * aid + 4,
                                   rid]  # Mx1, action score for this human, object and action

                    if role_boxes.shape[0] == 0: continue

                    # ToDo: different with iCAN
                    # ToDo: find wrong objects or bg
                    # valid = np.where(np.isnan(role_boxes).any() == False)[0]
                    valid = np.where(np.isnan(agent_scores) == False)[0]

                    agent_scores = agent_scores[valid]
                    agent_boxes = agent_boxes[valid, :]
                    role_boxes = role_boxes[valid, :]

                    # ndet[aid][rid] += agent_boxes.shape[0]

                    # sort in descending order
                    idx = agent_scores.argsort()[::-1]  # A action can be done by multiple human.
                    for j in idx:  # in this image, this action with highest action score
                        pred_box = agent_boxes[j, :]
                        overlaps = get_overlap(gt_boxes, pred_box)  # gt_boxes: gt human box

                        jmax = overlaps.argmax()  # which gt_box best matches this detected box
                        ovmax = overlaps.max()

                        # if matched with an instance with no annotations
                        # continue
                        if ignore[jmax]:
                            continue

                        # overlap between predicted role and gt role
                        if np.all(gt_roles[jmax, :] == -1):  # if no gt role
                            if eval_type == 'scenario_1':
                                if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                                    # if no role is predicted, mark it as correct role overlap
                                    ov_role = 1.0
                                else:
                                    # if a role is predicted, mark it as false
                                    ov_role = -1.0
                            elif eval_type == 'scenario_2':
                                # if no gt role, role prediction is always correct, irrespective of the actual predition
                                ov_role = 1.0
                            else:
                                raise ValueError('Unknown eval type')
                        else:
                            ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])

                        is_true_action = (gt_actions[jmax, aid] == 1)  # Is this gt human actually doing this action?
                        sc[aid][rid].append(agent_scores[j])
                        ndet[aid][rid] += 1
                        if (ovmax < 0.1):  # bck
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(1)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        elif (ovmax < 0.5) & (ovmax >= 0.1):  # person misloc
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(1)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        # elif np.all(gt_actions[:,
                        #           aid] == 0):  # All gt are not this action class. All detections are incorrect labels.
                        #     fp1[aid][rid].append(1)
                        #     fp2[aid][rid].append(0)
                        #     fp3[aid][rid].append(0)
                        #     fp4[aid][rid].append(0)
                        #     fp5[aid][rid].append(0)
                        #     fp6[aid][rid].append(0)
                        #     fp7[aid][rid].append(0)
                        #     tp[aid][rid].append(0)
                        #     continue
                        elif is_true_action == False:  # This detection j is a incorrect label
                            fp1[aid][rid].append(1)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        elif (ovmax >= 0.5) & (ov_role == -1.0):  # occlusion
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(1)
                            tp[aid][rid].append(0)
                        elif (ovmax >= 0.5) & (0 <= ov_role <= 0.1):  # mis-grouping
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(0)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(1)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        elif (ovmax >= 0.5) & (0.1 <= ov_role < 0.5):  # obj misloc
                            fp1[aid][rid].append(0)
                            fp2[aid][rid].append(0)
                            fp3[aid][rid].append(0)
                            fp4[aid][rid].append(1)
                            fp5[aid][rid].append(0)
                            fp6[aid][rid].append(0)
                            fp7[aid][rid].append(0)
                            tp[aid][rid].append(0)
                        elif (ovmax >= 0.5) & (ov_role >= 0.5):  # true positive
                            if not covered[jmax]:
                                fp1[aid][rid].append(0)
                                fp2[aid][rid].append(0)
                                fp3[aid][rid].append(0)
                                fp4[aid][rid].append(0)
                                fp5[aid][rid].append(0)
                                fp6[aid][rid].append(0)
                                fp7[aid][rid].append(0)
                                tp[aid][rid].append(1)
                                covered[jmax] = True
                            else:
                                fp1[aid][rid].append(0)
                                fp2[aid][rid].append(0)
                                fp3[aid][rid].append(0)
                                fp4[aid][rid].append(0)
                                fp5[aid][rid].append(1)
                                fp6[aid][rid].append(0)
                                fp7[aid][rid].append(0)
                                tp[aid][rid].append(0)

        fp_inc = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_bck = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_Hmis = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_Omis = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_dupl = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_misg = np.zeros((self.num_actions, 2), dtype=np.float32)
        fp_occl = np.zeros((self.num_actions, 2), dtype=np.float32)
        rec = np.zeros((self.num_actions, 2), dtype=np.float32)
        prec = np.zeros((self.num_actions, 2), dtype=np.float32)
        tp_ = np.zeros((self.num_actions, 2), dtype=np.float32)

        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2:
                continue

            for rid in range(len(self.roles[aid]) - 1):
                a_fp1 = np.array(fp1[aid][rid], dtype=np.float32)
                a_fp2 = np.array(fp2[aid][rid], dtype=np.float32)
                a_fp3 = np.array(fp3[aid][rid], dtype=np.float32)
                a_fp4 = np.array(fp4[aid][rid], dtype=np.float32)
                a_fp5 = np.array(fp5[aid][rid], dtype=np.float32)
                a_fp6 = np.array(fp6[aid][rid], dtype=np.float32)
                a_fp7 = np.array(fp7[aid][rid], dtype=np.float32)
                a_sc = np.array(sc[aid][rid], dtype=np.float32)
                a_tp = np.array(tp[aid][rid], dtype=np.float32)

                # sort in descending score order
                idx = a_sc.argsort()[::-1]
                a_fp1 = a_fp1[idx]
                a_fp2 = a_fp2[idx]
                a_fp3 = a_fp3[idx]
                a_fp4 = a_fp4[idx]
                a_fp5 = a_fp5[idx]
                a_fp6 = a_fp6[idx]
                a_fp7 = a_fp7[idx]
                a_tp = a_tp[idx]
                a_sc = a_sc[idx]

                # min(# GT, # not zero)
                num_inst = int(min(npos[aid], len(a_sc)))

                a_fp1 = a_fp1[:num_inst]
                a_fp2 = a_fp2[:num_inst]
                a_fp3 = a_fp3[:num_inst]
                a_fp4 = a_fp4[:num_inst]
                a_fp5 = a_fp5[:num_inst]
                a_fp6 = a_fp6[:num_inst]
                a_fp7 = a_fp7[:num_inst]
                a_tp = a_tp[:num_inst]
                a_sc = a_sc[:num_inst]

                frac_fp1 = np.sum(a_fp1) / (num_inst - np.sum(a_tp))
                frac_fp2 = np.sum(a_fp2) / (num_inst - np.sum(a_tp))
                frac_fp3 = np.sum(a_fp3) / (num_inst - np.sum(a_tp))
                frac_fp4 = np.sum(a_fp4) / (num_inst - np.sum(a_tp))
                frac_fp5 = np.sum(a_fp5) / (num_inst - np.sum(a_tp))
                frac_fp6 = np.sum(a_fp6) / (num_inst - np.sum(a_tp))
                frac_fp7 = np.sum(a_fp7) / (num_inst - np.sum(a_tp))

                tp_[aid, rid] = np.sum(a_tp)
                rec[aid, rid] = np.sum(a_tp) / float(npos[aid])
                prec[aid, rid] = np.sum(a_tp) / np.maximum(
                    np.sum(a_fp1) + np.sum(a_fp2) + np.sum(a_fp3) + np.sum(a_fp4) + np.sum(a_fp5) + np.sum(
                        a_fp6) + np.sum(a_fp7) + np.sum(a_tp), np.finfo(np.float64).eps)

                fp_inc[aid, rid] = frac_fp1
                fp_bck[aid, rid] = frac_fp2
                fp_Hmis[aid, rid] = frac_fp3
                fp_Omis[aid, rid] = frac_fp4
                fp_dupl[aid, rid] = frac_fp5
                fp_misg[aid, rid] = frac_fp6
                fp_occl[aid, rid] = frac_fp7

        print(
            '--------------------------------------------Reporting Error Analysis (%)-----------------------------------------------')
        print('{: >27} {:} {:} {:} {:} {:} {:}'.format(' ', 'inc', 'bck', 'H_mis', 'O_mis', 'mis-gr', 'occl'))
        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2: continue
            for rid in range(len(self.roles[aid]) - 1):
                print(
                    '{: >23}: {:6.2f} {:4.2f} {:4.2f} {:5.2f} {:5.2f} {:5.2f} (rec:{:5.2f} = #tp:{:4d}/#pos:{:4d}) (prec:{:5.2f} = #tp:{:4d}/#det:{:4d})'.format(
                        self.actions[aid] + '-' + self.roles[aid][rid + 1],
                        fp_inc[aid, rid] * 100.0,
                        fp_bck[aid, rid] * 100.0,
                        fp_Hmis[aid, rid] * 100.0,
                        fp_Omis[aid, rid] * 100.0,
                        fp_misg[aid, rid] * 100.0,
                        fp_occl[aid, rid] * 100.0,
                        rec[aid, rid] * 100.0,
                        int(tp_[aid, rid]),
                        int(npos[aid]),
                        prec[aid, rid] * 100.0,
                        int(tp_[aid, rid]),
                        int(ndet[aid, rid])))

    def _visualize_agent_error_human_centric(self, vcocodb, dets, ovr_thresh=0.5):
        # visualize agent error
        tp = [[] for a in range(self.num_actions)]
        fp1 = [[] for a in range(self.num_actions)]  # incorrect label
        fp2 = [[] for a in range(self.num_actions)]  # bck
        fp3 = [[] for a in range(self.num_actions)]  # person misloc
        fp5 = [[] for a in range(self.num_actions)]  # duplicate detection
        sc = [[] for a in range(self.num_actions)]

        npos = np.zeros((self.num_actions), dtype=np.float32)  # A + B
        ndet = np.zeros((self.num_actions), dtype=np.float32)  # B + C
        Test_occlusion = {}

        for i in range(len(vcocodb)):

            image_id = vcocodb[i][
                'id']  # img ID, not the full name (e.g. id= 165, 'file_name' = COCO_train2014_000000000165.jpg )
            gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]  # index of the person's box among all object boxes
            # person boxes
            gt_boxes = vcocodb[i]['boxes'][gt_inds]  # Nx4 all person's boxes in this image
            gt_actions = vcocodb[i]['gt_actions'][
                gt_inds]  # Nx26 binary array indicating the actions performed by this person

            # some peorson instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)
            assert np.all(gt_actions[np.where(ignore == True)[0]] == -1)

            for aid in range(self.num_actions):
                npos[aid] += np.sum(
                    gt_actions[:, aid] == 1)  # how many actions are involved in this image(for all the human)

            pred_agents, _ = self._collect_detections_for_image(dets, image_id)
            # pred_agents Mx30
            # pred_roles Mx(5*26)x2
            for aid in range(self.num_actions):

                # keep track of detected instances for each action for each role. Is this gt_human used or not.
                covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)

                agent_scores = pred_agents[:, 4 + aid]
                agent_boxes = pred_agents[:, :4]

                # ToDo: different with iCAN
                # valid = np.where(np.isnan(role_boxes).any() == False)[0]
                valid = np.where(np.isnan(agent_scores) == False)[0]
                agent_scores = agent_scores[valid]
                agent_boxes = agent_boxes[valid, :]

                # ndet[aid][rid] += agent_boxes.shape[0]
                # sort in descending order
                idx = agent_scores.argsort()[::-1]  # A action can be done by multiple human.
                for j in idx:  # in this image, this action with highest action score
                    pred_box = agent_boxes[j, :]
                    overlaps = get_overlap(gt_boxes, pred_box)  # gt_boxes: gt human box

                    jmax = overlaps.argmax()  # which gt_box best matches this detected box
                    ovmax = overlaps.max()
                    # if matched with an instance with no annotations
                    # continue
                    if ignore[jmax]:
                        continue

                    is_true_action = (gt_actions[jmax, aid] == 1)  # Is this gt human actually doing this action?
                    sc[aid].append(agent_scores[j])
                    ndet[aid] += 1
                    if (ovmax < 0.1):  # bck
                        fp1[aid].append(0)
                        fp2[aid].append(1)
                        fp3[aid].append(0)
                        fp5[aid].append(0)
                        tp[aid].append(0)
                    elif (ovmax < 0.5) & (ovmax >= 0.1):  # person misloc
                        fp1[aid].append(0)
                        fp2[aid].append(0)
                        fp3[aid].append(1)
                        fp5[aid].append(0)
                        tp[aid].append(0)
                    # elif np.all(gt_actions[:,
                    #           aid] == 0):  # All gt are not this action class. All detections are incorrect labels.
                    #     fp1[aid][rid].append(1)
                    #     fp2[aid][rid].append(0)
                    #     fp3[aid][rid].append(0)
                    #     fp4[aid][rid].append(0)
                    #     fp5[aid][rid].append(0)
                    #     fp6[aid][rid].append(0)
                    #     fp7[aid][rid].append(0)
                    #     tp[aid][rid].append(0)
                    #     continue
                    elif is_true_action == False:  # This detection j is a incorrect label
                        fp1[aid].append(1)
                        fp2[aid].append(0)
                        fp3[aid].append(0)
                        fp5[aid].append(0)
                        tp[aid].append(0)
                    elif (ovmax >= 0.5):  # true positive
                        if not covered[jmax]:
                            fp1[aid].append(0)
                            fp2[aid].append(0)
                            fp3[aid].append(0)
                            fp5[aid].append(0)
                            tp[aid].append(1)
                            covered[jmax] = True
                        else:
                            fp1[aid].append(0)
                            fp2[aid].append(0)
                            fp3[aid].append(0)
                            fp5[aid].append(1)
                            tp[aid].append(0)

        fp_inc = np.zeros(self.num_actions, dtype=np.float32)
        fp_bck = np.zeros(self.num_actions, dtype=np.float32)
        fp_Hmis = np.zeros(self.num_actions, dtype=np.float32)
        fp_dupl = np.zeros(self.num_actions, dtype=np.float32)
        rec = np.zeros(self.num_actions, dtype=np.float32)
        prec = np.zeros(self.num_actions, dtype=np.float32)
        tp_ = np.zeros(self.num_actions, dtype=np.float32)

        for aid in range(self.num_actions):
            a_fp1 = np.array(fp1[aid], dtype=np.float32)
            a_fp2 = np.array(fp2[aid], dtype=np.float32)
            a_fp3 = np.array(fp3[aid], dtype=np.float32)
            a_fp5 = np.array(fp5[aid], dtype=np.float32)
            a_sc = np.array(sc[aid], dtype=np.float32)
            a_tp = np.array(tp[aid], dtype=np.float32)
            # sort in descending score order
            idx = a_sc.argsort()[::-1]
            a_fp1 = a_fp1[idx]
            a_fp2 = a_fp2[idx]
            a_fp3 = a_fp3[idx]
            a_fp5 = a_fp5[idx]
            a_tp = a_tp[idx]
            a_sc = a_sc[idx]
            # min(# GT, # not zero)
            num_inst = int(min(npos[aid], len(a_sc)))
            a_fp1 = a_fp1[:num_inst]
            a_fp2 = a_fp2[:num_inst]
            a_fp3 = a_fp3[:num_inst]
            a_fp5 = a_fp5[:num_inst]
            a_tp = a_tp[:num_inst]
            a_sc = a_sc[:num_inst]
            frac_fp1 = np.sum(a_fp1) / (num_inst - np.sum(a_tp))
            frac_fp2 = np.sum(a_fp2) / (num_inst - np.sum(a_tp))
            frac_fp3 = np.sum(a_fp3) / (num_inst - np.sum(a_tp))
            frac_fp5 = np.sum(a_fp5) / (num_inst - np.sum(a_tp))
            tp_[aid] = np.sum(a_tp)
            rec[aid] = np.sum(a_tp) / float(npos[aid])
            prec[aid] = np.sum(a_tp) / np.maximum(
                np.sum(a_fp1) + np.sum(a_fp2) + np.sum(a_fp3) + np.sum(a_fp5) + np.sum(a_tp), np.finfo(np.float64).eps)
            fp_inc[aid] = frac_fp1
            fp_bck[aid] = frac_fp2
            fp_Hmis[aid] = frac_fp3
            fp_dupl[aid] = frac_fp5

        print(
            '--------------------------------------------Reporting Error Analysis (%)-----------------------------------------------')
        print('{: >27} {:} {:} {:}'.format(' ', 'inc', 'bck', 'H_mis'))
        for aid in range(self.num_actions):
            print(
                '{: >23}: {:6.2f} {:4.2f} {:4.2f} (rec:{:5.2f} = #tp:{:4d}/#pos:{:4d}) (prec:{:5.2f} = #tp:{:4d}/#det:{:4d})'.format(
                    self.actions[aid],
                    fp_inc[aid] * 100.0,
                    fp_bck[aid] * 100.0,
                    fp_Hmis[aid] * 100.0,
                    rec[aid] * 100.0,
                    int(tp_[aid]),
                    int(npos[aid]),
                    prec[aid] * 100.0,
                    int(tp_[aid]),
                    int(ndet[aid])))


    def _save_error(self, vcocodb, dets, ovr_thresh=0.5, eval_type='scenario_1', save_file=None):
        all_res = []
        set_zero_ret = {}
        fp1 = fp2 = fp3 = fp4 = fp5 = fp6 = tp = count =0

        for i in range(len(vcocodb)):
            sub_count = 0
            res = []
            image_id = vcocodb[i]['id']  # img ID, not the full name (e.g. id= 165, 'file_name' = COCO_train2014_000000000165.jpg )

            pred_agents, pred_roles = self._collect_detections_for_image(dets, image_id)
            gt_human_inds, gt_act_inds, gt_r_ids = np.where(vcocodb[i]['gt_role_id'] > -1)
            gt_obj_inds = vcocodb[i]['gt_role_id'][gt_human_inds, gt_act_inds, gt_r_ids]
            if len(gt_obj_inds) == 0:
                set_zero_ret[image_id] = {'agents': pred_agents, 'roles': pred_roles}
                continue
            # ipdb.set_trace()
            gt_all_relboxes = vcocodb[i]['boxes'][np.unique(gt_obj_inds)]


            gt_h_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]  # index of the person's box among all object boxes
            # person boxes
            gt_h_boxes = vcocodb[i]['boxes'][gt_h_inds]  # Nx4 all person's boxes in this image
            gt_actions = vcocodb[i]['gt_actions'][gt_h_inds]  # Nx26 binary array indicating the actions performed by this person

            # some peorson instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)
            assert np.all(gt_actions[np.where(ignore == True)[0]] == -1)

            # pred_agents Mx30
            # pred_roles Mx(5*26)x2
            for aid in range(self.num_actions):
                if len(self.roles[aid]) < 2:
                    # if action has no role, then no role AP computed
                    continue

                for rid in range(len(self.roles[aid]) - 1):  # rid = 0, instr; rid = 1, obj

                    # keep track of detected instances for each action for each role. Is this gt_human used or not.
                    covered = np.zeros((gt_h_boxes.shape[0]), dtype=np.bool)

                    # get gt roles for action and role
                    gt_role_inds = vcocodb[i]['gt_role_id'][gt_h_inds, aid, rid]  # Nx1 index of the object among all detected objects related to this action. -1 means missing object.
                    gt_roles = -np.ones_like(gt_h_boxes)  # Nx4 [-1, -1, -1, -1] means gt missing object
                    for j in range(gt_h_boxes.shape[0]):  # loop all gt human instance
                        if gt_role_inds[j] > -1:  #
                            gt_roles[j] = vcocodb[i]['boxes'][gt_role_inds[j]]

                    agent_boxes = pred_agents[:, :4]  # Mx4 all detected human box
                    role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4, rid]  # Mx4 detected object(role) box for this human and action
                    agent_scores = pred_roles[:, 5 * aid + 4, rid]  # Mx1, action score for this human, object and action

                    if role_boxes.shape[0] == 0: continue

                    # ToDo: different with iCAN
                    # valid = np.where(np.isnan(role_boxes).any() == False)[0]
                    valid = np.where(np.isnan(agent_scores) == False)[0]

                    agent_scores = agent_scores[valid]
                    agent_boxes = agent_boxes[valid, :]
                    role_boxes = role_boxes[valid, :]

                    # sort in descending order
                    idx = agent_scores.argsort()[::-1]  # A action can be done by multiple human.
                    for j in idx:  # in this image, this action with highest action score
                        pred_box = agent_boxes[j, :]
                        overlaps = get_overlap(gt_h_boxes, pred_box)  # gt_h_boxes: gt human box

                        jmax = overlaps.argmax()  # which gt_box best matches this detected box
                        ovmax = overlaps.max()

                        # if matched with an instance with no annotations
                        # continue
                        if ignore[jmax]:
                            continue
                        sub_count += 1

                        # ov_all_relboxes = get_overlap(gt_all_relboxes, role_boxes[j, :])
                        ov_all_relboxes = get_overlap(vcocodb[i]['boxes'], role_boxes[j, :])
                        ov_max_all_relboxes = ov_all_relboxes.max()

                        gt_role_j_human = vcocodb[i]['gt_role_id'][gt_h_inds[jmax]]
                        gt_box_j_objs = vcocodb[i]['boxes'][np.unique(gt_role_j_human[np.where(gt_role_j_human > -1)])]
                        if len(gt_box_j_objs) == 0:
                            ov_max_j_objs = -1
                        else:
                            ov_j_objs = get_overlap(gt_box_j_objs, role_boxes[j, :])
                            ov_max_j_objs = ov_j_objs.max()

                        # overlap between predicted role and gt role
                        if np.all(gt_roles[jmax, :] == -1):  # if no gt role
                            if eval_type == 'scenario_1':
                                if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                                    # if no role is predicted, mark it as correct role overlap
                                    ov_role = 1.0
                                else:
                                    # if a role is predicted, mark it as false
                                    ov_role = -1.0
                            elif eval_type == 'scenario_2':
                                # if no gt role, role prediction is always correct, irrespective of the actual predition
                                ov_role = 1.0
                            else:
                                raise ValueError('Unknown eval type')
                        else:
                            ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])

                        is_true_action = (gt_actions[jmax, aid] == 1)  # Is this gt human actually doing this action?

                        # if np.all(gt_actions[:, aid] == 0):
                        #     # All gt are not this action class. All detections are incorrect labels.
                        #     res.append(np.concatenate([pred_box, role_boxes[j, :], [aid, rid, agent_scores[j]], [1]]))
                        #     continue
                        if ovmax < 0.5:  # person mis-loc
                            fp1 += 1

                            '''**************  set fp score to 0 **********************'''
                            # pred_roles[valid[j], 5 * aid + 4, rid] = 0

                            res.append(np.concatenate([pred_box, role_boxes[j, :], [aid, rid, agent_scores[j]], [1]]))
                        elif 0 <= ov_max_all_relboxes < 0.5:
                            if is_true_action == True and ov_max_j_objs == -1.0: # occlusion
                                fp2 += 1

                                '''**************  set fp score to 0 **********************'''
                                # pred_roles[valid[j], 5 * aid + 4, rid] = 0

                                res.append(np.concatenate([pred_box, role_boxes[j, :], [aid, rid, agent_scores[j]], [5]]))
                            else: # no relationship relevant, obj misloc
                                fp3 += 1
                                '''**************  set fp score to 0 **********************'''
                                # pred_roles[valid[j], 5 * aid + 4, rid] = 0

                                res.append(np.concatenate([pred_box, role_boxes[j, :], [aid, rid, agent_scores[j]], [2]]))
                        elif ov_max_j_objs < 0.5: # the target object fg but not in target triplet; misgrouping
                            fp4 += 1
                            '''**************  set fp score to 0 **********************'''
                            # pred_roles[valid[j], 5 * aid + 4, rid] = 0

                            res.append(np.concatenate([pred_box, role_boxes[j, :], [aid, rid, agent_scores[j]], [3]]))
                        elif ov_role < 0.5: # person and obj loc is right but predict action wrong; prediction wrong
                            fp5 += 1
                            '''**************  set fp score to 0 **********************'''
                            # pred_roles[valid[j], 5 * aid + 4, rid] = 0

                            res.append(np.concatenate([pred_box, role_boxes[j, :], [aid, rid, agent_scores[j]], [4]]))
                        # elif is_true_action == False: # person and obj loc is right but predict action wrong; prediction wrong
                        #     ipdb.set_trace()
                        #     res.append(np.concatenate([pred_box, role_boxes[j, :], [aid, rid, agent_scores[j]], [4]]))
                        elif (ovmax >= 0.5) & (ov_role >= 0.5):  # true positive
                            if not covered[jmax]:
                                tp += 1
                                res.append(np.concatenate([pred_box, role_boxes[j, :], [aid, rid, agent_scores[j]], [0]]))
                                covered[jmax] = True
                            else:
                                fp6 += 1
                                res.append(np.concatenate([pred_box, role_boxes[j, :], [aid, rid, agent_scores[j]], [6]])) # triplet duplicate

            assert len(res) == sub_count
            count += sub_count

            set_zero_ret[image_id] = {'agents': pred_agents, 'roles':pred_roles}

            ret = {}
            ret['im_id'] = image_id
            ret['gt_classes'] = vcocodb[i]['gt_classes']
            ret['gt_boxes'] = vcocodb[i]['boxes']
            ret['gt_actions'] = vcocodb[i]['gt_actions']
            ret['gt_role_inds'] = vcocodb[i]['gt_role_id']
            ret['pred_triplets'] = np.array(res)

            if count != fp1+fp2+fp3+fp4+fp5+fp6+tp:
                ipdb.set_trace()

            all_res.append(ret)
        print(fp1,fp2,fp3,fp4,fp5,fp6,tp,count)
        with open(save_file, 'wb') as f:
            pickle.dump(all_res, f)
            # pickle.dump(set_zero_ret, f)


def get_overlap(boxes, ref_box):
    ixmin = np.maximum(boxes[:, 0], ref_box[0])
    iymin = np.maximum(boxes[:, 1], ref_box[1])
    ixmax = np.minimum(boxes[:, 2], ref_box[2])
    iymax = np.minimum(boxes[:, 3], ref_box[3])
    # maximum zero
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) +
           (boxes[:, 2] - boxes[:, 0] + 1.) *
           (boxes[:, 3] - boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


def voc_ap(rec, prec, plot_save_path=None):
    """ ap = voc_ap(rec, prec)
    Compute VOC AP given precision and recall.
    [as defined in PASCAL VOC]
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    if plot_save_path is not None:
        decreasing_max_precision = np.maximum.accumulate(mpre[::-1])[::-1]
        fig, ax = plt.subplots(1, 1)
        ax.plot(mrec, mpre, '--b')
        ax.step(mrec, decreasing_max_precision, '-r')
        plt.savefig(plot_save_path)

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

