#!/usr/bin/env bash

model_name="e2e_pmf_net_R-50-FPN_1x"
EXP="final_trainval"
mkdir -p ./Outputs/e2e_pmfnet_R-50-FPN_1x/${EXP}

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u tools/train_net_step.py --dataset vcoco_trainval \
       --cfg configs/baselines/$model_name.yaml \
       --use_precomp_box \
       --vcoco_use_union_feat --lr 4e-2 \
       --bs 4 --nw 8 --disp_interval 200 \
       --freeze_at 5 --mlp_head_dim 256 \
       --part_crop_size 5 --use_kps17 \
       --max_iter 12000 --solver_steps 0 6000 8000 \
       --net_name PMFNet_Final \
       --triplets_num_per_im 32 \
       --expID ${EXP} \
       --load_ckpt data/pretrained_model/e2e_faster_rcnn_R-50-FPN_1x_step119999.pth \
       --use_tfboard |tee ./Outputs/e2e_pmfnet_R-50-FPN_1x/${EXP}/train-log.out

 
# --no_save \
# --use_precomp_box \
# --vcoco_use_spatial --vcoco_kp_on --vcoco_use_union_feat \
# --load_ckpt data/pretrained_model/e2e_faster_rcnn_R-50-FPN_1x_step119999.pth \
# --load_detectron data/pretrained_model/e2e_faster_rcnn_R-50-FPN_1x.pkl \
