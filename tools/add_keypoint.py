import json
import os


def add_keypoint_to_vcoco_anns(keypoint_anns, vcoco_anns_file_path):
    vcoco_anns = get_data(vcoco_anns_file_path)

    vcoco_image_ids = []
    for image in vcoco_anns['images']:
        vcoco_image_ids.append(image['id'])
    print('v-coco image num', len(vcoco_image_ids))

    keypoint_anns_of_vcoco = {}
    for ann in keypoint_anns['annotations']:
        if ann['image_id'] in vcoco_image_ids:
            keypoint_anns_of_vcoco[ann['id']] = ann
    print('key-point annotation num', len(keypoint_anns_of_vcoco))

    print('vcoco anns num', len(vcoco_anns['annotations']))
    count = 0
    vcoco_annotations_with_keypoints = []
    for ann in vcoco_anns['annotations']:
        if keypoint_anns_of_vcoco.get(ann['id'], None):
            vcoco_annotations_with_keypoints.append(keypoint_anns_of_vcoco[ann['id']])
        else:
            ann['num_keypoints'] = 0
            vcoco_annotations_with_keypoints.append(ann)

    print('new vcoco anns num', len(vcoco_annotations_with_keypoints))
    vcoco_anns['annotations'] = vcoco_annotations_with_keypoints

    # add keypoints info in categories
    vcoco_anns['categories'][0] = keypoint_anns['categories'][0]

    vcoco_with_keypoints_save_path = vcoco_anns_file_path.replace(
        'annotations', 'annotations_with_keypoints').replace('instances', 'instances_with_keypoints')
    print(vcoco_with_keypoints_save_path)
    with open(vcoco_with_keypoints_save_path, 'w') as f:
        json.dump(vcoco_anns, f)


def get_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def main():
    keypoint_train_path = '../data/coco/annotations_kp/person_keypoints_train2014.json'
    keypoint_val_path = '../data/coco/annotations_kp/person_keypoints_val2014.json'
    vcoco_train_path = '../data/coco/vcoco/annotations/instances_vcoco_train_2014.json'
    vcoco_val_path = '../data/coco/vcoco/annotations/instances_vcoco_val_2014.json'
    vcoco_trainval_path = '../data/coco/vcoco/annotations/instances_vcoco_trainval_2014.json'
    vcoco_test_path = '../data/coco/vcoco/annotations/instances_vcoco_test_2014.json'

    keypoint_train = get_data(keypoint_train_path)
    keypoint_val = get_data(keypoint_val_path)

    add_keypoint_to_vcoco_anns(keypoint_train, vcoco_train_path)
    add_keypoint_to_vcoco_anns(keypoint_train, vcoco_val_path)
    add_keypoint_to_vcoco_anns(keypoint_train, vcoco_trainval_path)

    add_keypoint_to_vcoco_anns(keypoint_val, vcoco_test_path)


if __name__ == '__main__':
    main()