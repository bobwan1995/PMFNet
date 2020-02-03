#!/usr/bin/env bash

model_name="e2e_pmf_net_R-50-FPN_1x"
EXP="final_trainval"
mkdir -p ./Outputs/e2e_pmfnet_R-50-FPN_1x/${EXP}

 CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset vcoco_test \
        --cfg configs/baselines/$model_name.yaml \
        --use_precomp_box \
        --mlp_head_dim 256 \
        --part_crop_size 5 --use_kps17 \
        --net_name PMFNet_Final \
        --load_ckpt ./Outputs/model_test_best.pth
