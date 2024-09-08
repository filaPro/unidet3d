import mmengine
import os
from tqdm.auto import tqdm
import numpy as np
import argparse

COLOR_TO_LABEL = {
    (0, 0, 0): 'unknown',
    (174, 199, 232): 'wall',
    (152, 223, 138): 'floor',
    (31, 119, 180): 'cabinet',
    (255, 187, 120): 'bed',
    (188, 189, 34): 'chair',
    (140, 86, 75): 'sofa',
    (255, 152, 150): 'table',
    (214, 39, 40): 'door',
    (197, 176, 213): 'window',
    (148, 103, 189): 'bookshelf',
    (196, 156, 148): 'picture',
    (23, 190, 207): 'counter',
    (178, 76, 76): 'blinds',
    (247, 182, 210): 'desk',
    (66, 188, 102): 'shelves',
    (219, 219, 141): 'curtain',
    (140, 57, 197): 'dresser',
    (202, 185, 52): 'pillow',
    (51, 176, 203): 'mirror',
    (200, 54, 131): 'floor mat',
    (92, 193, 61): 'clothes',
    (78, 71, 183): 'ceiling',
    (172, 114, 82): 'books',
    (255, 127, 14): 'fridge',
    (91, 163, 138): 'television',
    (153, 98, 156): 'paper',
    (140, 153, 101): 'towel',
    (158, 218, 229): 'shower curtain',
    (100, 125, 154): 'box',
    (178, 127, 135): 'whiteboard',
    (120, 185, 128): 'person',
    (146, 111, 194): 'night stand',
    (44, 160, 44): 'toilet',
    (112, 128, 144): 'sink',
    (96, 207, 209): 'lamp',
    (227, 119, 194): 'bathtub',
    (213, 92, 176): 'bag',
    (94, 106, 211): 'structure',
    (82, 84, 163): 'furniture',
    (100, 85, 144): 'prop'
}

OBJ2SEM = {v: idx for idx, (k, v) in enumerate(COLOR_TO_LABEL.items())}
OBJ2SEM['unknown'] = -1
REMAIN_BB_LABELS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39] 

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def _filter_bb(bb):
    final  = []
    for i in bb:
        if i[-1] in REMAIN_BB_LABELS:
            final.append(i)

    if len(final) == 0:
        return np.zeros((0,7))

    return np.stack(final)

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



def rearrange_sup(sup):
    sup = sup.copy()
    unique_super = np.unique(sup)

    for idx, un in enumerate(unique_super):
        ind  = np.where(sup == un)[0]
        sup[ind] = idx

    return sup


def create_metainfo():

    return {
        'categories': OBJ2SEM,
        'dataset': '3RScan',
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

def create_pkl_file(path_to_save, split, splits, bins_path, pkl_prefix = '3rscan'):
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

    parser.add_argument(
        '--path_to_splits',
        default='meta_data/split/',
        help='Path to train/val/test splits',
        type=str,
    )

    args = parser.parse_args()
    print(args)

    path_to_splits = args.path_to_splits
    path_to_raw_data = args.path_to_data

    path_to_save_data = args.path_to_save_bins
    create_dir(path_to_save_data)
    bins_path = create_dirs(path_to_save_data)
    
    with open(path_to_splits + '/train.txt') as train_file:
        train_scenes = train_file.read().splitlines()
    with open(path_to_splits + '/val.txt') as val_file:
        val_scenes = val_file.read().splitlines()
    with open(path_to_splits + '/test.txt') as test_file:
        test_scenes = test_file.read().splitlines()

    splits = {
        'train': train_scenes,
        'val': val_scenes,
        'test': test_scenes
    }

    scene_ids = os.listdir(path_to_raw_data)

    for si in tqdm(scene_ids):
        temp_path = os.path.join(path_to_raw_data, si)
        point_cloud = np.load(temp_path + f'/{si}_aligned_vert.npy')
        sem_label = np.load(temp_path + f'/{si}_sem_label.npy')[:, 0]
        ins_label = np.load(temp_path + f'/{si}_ins_label.npy')[:, 0]
        bboxs = np.load(temp_path + f'/{si}_aligned_bbox.npy')
        superpoints = np.load(temp_path + f'/{si}_superpoints.npy')
        superpoints = rearrange_sup(superpoints)
        bboxs = _filter_bb(bboxs)
    
        superpoints = np.load(temp_path + f'/{si}_superpoints.npy')
        superpoints = rearrange_sup(superpoints)
    
        point_cloud.astype(np.float32).tofile(os.path.join(bins_path['points'], 
                                                           f'{si}.bin'))
        sem_label.astype(np.int64).tofile(os.path.join(bins_path['semantic_mask'], 
                                                       f'{si}.bin'))
        ins_label.astype(np.int64).tofile(os.path.join(bins_path['instance_mask'], 
                                                       f'{si}.bin'))
        superpoints.astype(np.int64).tofile(os.path.join(bins_path['superpoints'], 
                                                         f'{si}.bin'))
        np.save(os.path.join(bins_path['bboxs'], f'{si}.npy'), bboxs)


    create_pkl_file(path_to_save_data, 'train', splits, bins_path)
    create_pkl_file(path_to_save_data, 'val', splits, bins_path)
    create_pkl_file(path_to_save_data, 'test', splits, bins_path)
























