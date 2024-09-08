import mmengine
import os
from tqdm.auto import tqdm
import numpy as np
import argparse

OBJ2SEM = {'wall': 0,
 'ceiling': 1,
 'floor': 2,
 'table': 3,
 'door': 4,
 'ceiling lamp': 5,
 'cabinet': 6,
 'blinds': 7,
 'curtain': 8,
 'chair': 9,
 'storage cabinet': 10,
 'office chair': 11,
 'bookshelf': 12,
 'whiteboard': 13,
 'window': 14,
 'box': 15,
 'window frame': 16,
 'monitor': 17,
 'shelf': 18,
 'doorframe': 19,
 'pipe': 20,
 'heater': 21,
 'kitchen cabinet': 22,
 'sofa': 23,
 'windowsill': 24,
 'bed': 25,
 'shower wall': 26,
 'trash can': 27,
 'book': 28,
 'plant': 29,
 'blanket': 30,
 'tv': 31,
 'computer tower': 32,
 'kitchen counter': 33,
 'refrigerator': 34,
 'jacket': 35,
 'electrical duct': 36,
 'sink': 37,
 'bag': 38,
 'picture': 39,
 'pillow': 40,
 'towel': 41,
 'suitcase': 42,
 'backpack': 43,
 'crate': 44,
 'keyboard': 45,
 'rack': 46,
 'toilet': 47,
 'paper': 48,
 'printer': 49,
 'poster': 50,
 'painting': 51,
 'microwave': 52,
 'board': 53,
 'shoes': 54,
 'socket': 55,
 'bottle': 56,
 'bucket': 57,
 'cushion': 58,
 'basket': 59,
 'shoe rack': 60,
 'telephone': 61,
 'file folder': 62,
 'cloth': 63,
 'blind rail': 64,
 'laptop': 65,
 'plant pot': 66,
 'exhaust fan': 67,
 'cup': 68,
 'coat hanger': 69,
 'light switch': 70,
 'speaker': 71,
 'table lamp': 72,
 'air vent': 73,
 'clothes hanger': 74,
 'kettle': 75,
 'smoke detector': 76,
 'container': 77,
 'power strip': 78,
 'slippers': 79,
 'paper bag': 80,
 'mouse': 81,
 'cutting board': 82,
 'toilet paper': 83,
 'paper towel': 84,
 'pot': 85,
 'clock': 86,
 'pan': 87,
 'tap': 88,
 'jar': 89,
 'soap dispenser': 90,
 'binder': 91,
 'bowl': 92,
 'tissue box': 93,
 'whiteboard eraser': 94,
 'toilet brush': 95,
 'spray bottle': 96,
 'headphones': 97,
 'stapler': 98,
 'marker': 99}

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_txt(path):
    res = []

    with open(path) as f:
        for line in tqdm(f):
            res.append(line.strip())

    return res

def create_dirs(path):
    points = os.path.join(path, 'points')
    create_dir(points)
    
    semantic_mask = os.path.join(path, 'semantic_mask')
    create_dir(semantic_mask)
    
    instance_mask = os.path.join(path, 'instance_mask')
    create_dir(instance_mask)
    
    bboxs = os.path.join(path, 'bboxs')
    create_dir(bboxs)

    superpoints = os.path.join(path, 'superpoints')
    create_dir(superpoints)
    return {
        'points': points,
        'semantic_mask': semantic_mask,
        'instance_mask': instance_mask,
        'bboxs': bboxs,
        'superpoints': superpoints
    }

def create_metainfo():

    return {
        'categories': OBJ2SEM,
        'dataset': 'scannetpp',
        'info_version': '1.0'
    }

def create_data_list(split, splits, bins_path):
    
    scenes = splits[split]
    final_list = []
    for scene in tqdm(scenes):
        lidar_points = {
            'num_pts_feats': 6,
            'lidar_path': f'{scene}.bin'
        }
        raw_bboxs = np.load(os.path.join(bins_path['bboxs'], f'{scene}.npy'))
        instances = []
        for rb in raw_bboxs:
            if len(rb) == 0:
                instances = []       
            else:
                instances.append({
                    'bbox_3d': rb[:6].tolist(),
                    'bbox_label_3d': int(rb[-1])
                })
        final_list.append({
            'lidar_points': lidar_points,
            'instances': instances,
            'pts_semantic_mask_path': f'{scene}.bin',
            'pts_instance_mask_path': f'{scene}.bin',
            'axis_align_matrix': np.eye(4)
        })

    return final_list

def create_pkl_file(path_to_save, split, splits, 
                    bins_path, pkl_prefix = 'scannetpp'):
    metainfo = create_metainfo()
    data_list = create_data_list(split, splits, bins_path)
    anno = {
        'metainfo': metainfo,
        'data_list': data_list
    }
    filename = os.path.join(path_to_save, f'{pkl_prefix}_infos_{split}.pkl')
    mmengine.dump(anno, filename, 'pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        required=True,
        help='Path to preprocessed raw data',
        type=str,
    )

    parser.add_argument(
        '--path_to_save_bins',
        required=True,
        help='Enter here the path where to save bins and pkls',
        type=str,
    )

    args = parser.parse_args()
    print(args)

    path_to_raw_data = args.path_to_data
    path_to_save_data = args.path_to_save_bins
    create_dir(path_to_save_data)
    bins_path = create_dirs(path_to_save_data)

    path_to_train_ids = os.path.join(path_to_raw_data, 'nvs_sem_train.txt')
    train_scenes = load_txt(path_to_train_ids)
    path_to_val_ids = os.path.join(path_to_raw_data, 'nvs_sem_val.txt')
    val_scenes = load_txt(path_to_val_ids)
    path_to_sem_test_ids = os.path.join(path_to_raw_data, 'sem_test.txt')
    test_scenes = load_txt(path_to_sem_test_ids)

    splits = {
        'train': train_scenes,
        'val': val_scenes,
        'test': test_scenes
    }

    path_to_raw_data = os.path.join(path_to_raw_data, 'data')
    scene_ids = os.listdir(path_to_raw_data)
    
    for si in tqdm(scene_ids):
        temp_path = os.path.join(path_to_raw_data, si)
        point_cloud = np.load(temp_path + f'/{si}_point_cloud.npy')
        sem_label = np.load(temp_path + f'/{si}_semantic.npy')
        ins_label = np.load(temp_path + f'/{si}_instance.npy')
        bboxs = np.load(temp_path + f'/{si}_bboxs.npy')
        superpoints = np.load(temp_path + f'/{si}_superpoints.npy')
    
        point_cloud.astype(np.float32).tofile(
            os.path.join(bins_path['points'], f'{si}.bin'))
        sem_label.astype(np.int64).tofile(
            os.path.join(bins_path['semantic_mask'], f'{si}.bin'))
        ins_label.astype(np.int64).tofile(
            os.path.join(bins_path['instance_mask'], f'{si}.bin'))
        superpoints.astype(np.int64).tofile(
            os.path.join(bins_path['superpoints'], f'{si}.bin'))
        np.save(os.path.join(bins_path['bboxs'], f'{si}.npy'), bboxs)

    create_pkl_file(path_to_save_data, 'train', splits, bins_path)
    create_pkl_file(path_to_save_data, 'val', splits, bins_path)
    create_pkl_file(path_to_save_data, 'test', splits, bins_path)
























