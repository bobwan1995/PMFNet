
import subprocess


#ckpt_file = os.path.join(output_dir, 'ckpt/model_step{}.pth'.format(step))
ckpt_file = './Outputs/fine_net/fine_net_baseline_freeze5/ckpt/model_step31999.pth'
cmd = 'python tools/test_net.py --dataset vcoco_all \
--cfg configs/baselines/e2e_interact_net_R-50-FPN_1x.yaml \
--vcoco_use_union_feat \
--use_precomp_box \
--net_name {net_name} \
--load_ckpt {ckpt_file} >/dev/null'.format(
    net_name = 'InteractNetUnion_Baseline',
    ckpt_file = ckpt_file
    )

import pdb
#pdb.set_trace()
subprocess.run(cmd, shell=True)