import _init_paths  # pylint: disable=unused-import
import torch
from datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg


if __name__ == '__main__':
    datasets = ('vcoco_trainval',)
    batch_size = 2
    num_classes = 81
    cfg.RPN.RPN_ON = True
    # assert_and_infer_cfg()

    roidb, ratio_list, ratio_index = combined_roidb_for_training(datasets, ())
    # print(roidb[0])
    print(roidb[0].keys())
    '''
    batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index),
        batch_size=batch_size,
        drop_last=True
    )
    dataset = RoiDataLoader(
        roidb,
        num_classes,
        training=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batchSampler,
        num_workers=1,
        collate_fn=collate_minibatch
    )

    for i, data in enumerate(dataloader):
        if i > 1:
            break
        # print(data)
        print(data.keys())
        for k in data.keys():
            print(k, type(data[k], len(data[k])))
    '''
