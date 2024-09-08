import os
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
import mmengine
import segmentator


remove_labels = ['floor', 'ceiling', 'wall']
rem_lab2sem = {'floor': 0, 'ceiling': 1, 'wall': 2}

obj_name2obj_sem_name = {'door': 'door',
                         'sliding_door': 'door',
                         'glass_door': 'door',
                         'bifold_door': 'door',
                         'adjustable_desk': 'table',
                         'computer_table': 'table',
                         'table': 'table',
                         'desk': 'table',
                         'computer_desk': 'table',
                         'bar_table': 'table',
                         'chair': 'chair',
                         'stacked_chairs': 'chair',
                         'wine_cabinet': 'cabinet',
                         'sink_cabinet': 'cabinet',
                         'cabinet': 'cabinet',
                         'wardrobe': 'cabinet',
                         'nightstand': 'cabinet',
                         'shoe_cabinet': 'cabinet',
                         'wall_cabinet': 'cabinet',
                         'tv_cabinet': 'cabinet',
                         'drawer_unit': 'cabinet',
                         'cabinet_otherroom': 'cabinet',
                         'window': 'window',
                         'sofa': 'sofa',
                         'microwave': 'microwave',
                         'sofa_cushion': 'pillow',
                         'thow_pillow': 'pillow',
                         'chair_cushion': 'pillow',
                         'back_cushion': 'pillow',
                         'cushion': 'pillow',
                         'pillow': 'pillow',
                         'tv': 'tv_monitor',
                         'monitor': 'tv_monitor',
                         'curtain': 'curtain',
                         'door_curtain': 'curtain',
                         'shower_curtain': 'curtain',
                         'trashbin': 'trash_can',
                         'trash_bin': 'trash_can',
                         'waste_container': 'trash_can',
                         'suitcase': 'suitcase',
                         'sink': 'sink',
                         'backpack': 'backpack',
                         'bed': 'bed',
                         'refrigerator': 'refrigerator',
                         'fridge': 'refrigerator',
                         'toilet': 'toilet',
                         'pit_toilet': 'toilet'}



obj2sem = {'floor': 0,
           'ceiling': 1,
           'wall': 2,
           'door': 3,
           'table': 4,
           'chair': 5,
           'cabinet': 6,
           'window': 7,
           'sofa': 8,
           'microwave': 9,
           'pillow': 10,
           'tv_monitor': 11,
           'curtain': 12,
           'trash_can': 13,
           'suitcase': 14,
           'sink': 15,
           'backpack': 16,
           'bed': 17,
           'refrigerator': 18,
           'toilet': 19,
           'no_target': -1}


_obj_name2obj_sem_name = lambda x: obj_name2obj_sem_name[x]
sem2obj = {v:k for k,v in obj2sem.items()}
_sem2obj = lambda x: sem2obj[x]
sem2rem_lab = {v:k for k,v in rem_lab2sem.items()}

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def create_dirs(path):
    points = os.path.join(path, 'points')
    create_dir(points)
    
    semantic_mask = os.path.join(path, 'semantic_mask')
    create_dir(semantic_mask)
    
    instance_mask = os.path.join(path, 'instance_mask')
    create_dir(instance_mask)
    
    super_points = os.path.join(path, 'super_points')
    create_dir(super_points)
    
    bboxs = os.path.join(path, 'bboxs')
    create_dir(bboxs)
    return {
        'points': points,
        'semantic_mask': semantic_mask,
        'instance_mask': instance_mask,
        'super_points': super_points,
        'bboxs': bboxs
    }

def get_data(path, file):
    scene = torch.load(os.path.join(path, file))
    coords = scene['xyz']
    colors = scene['rgb']
    faces = scene['faces']
    instance_ids = scene['instance_ids']
    sem_labels = scene['sem_labels']
    sem_labels_neg_1_indexes = sem_labels == -1
    instance_ids[sem_labels_neg_1_indexes] = -1
    
    instance_unique = np.unique(scene['instance_ids'])
    inst2obj = scene['inst2obj']
    _inst2obj = lambda x: inst2obj[x]
    scene_id = file.split('.')[0]

    return scene, coords, colors, faces, instance_ids, \
           sem_labels, instance_unique, inst2obj, \
           _inst2obj, scene_id



def prepare_data(path, files):
    final_res = []
    for file in tqdm(files):
        scene, coords, rgb, faces, instance_ids, \
        sem_labels, instance_unique, inst2obj, \
        _inst2obj, scene_id = get_data(path, file)

        vertices = torch.from_numpy(coords.astype(np.float32))
        faces = torch.from_numpy(faces.astype(np.int64))
        super_points = segmentator.segment_mesh(vertices, faces).numpy()
        
        for k,v in rem_lab2sem.items():
            indexes = sem_labels == v
            assert np.all(instance_ids[indexes] == -1)
            
        sem_labels_neg_1_indexes = sem_labels == -1
        instance_ids[sem_labels_neg_1_indexes] = -1
        assert np.all(instance_ids[sem_labels == -1] == -1)
        
        temp_bb = []
        for idx, inst_id in enumerate(instance_unique[1:]):
            indexes = instance_ids == inst_id
            current_points = coords[indexes]
            current_points_min = current_points.min(0)
            current_points_max = current_points.max(0)
            current_points_avg = (current_points_max +
                                  current_points_min) / 2
        
            lwh = (current_points_max - current_points_min).copy()
            vals, occurs = np.unique(
                sem_labels[indexes], return_counts=True)
            bbox_labels = vals[occurs.argmax()].copy()
            inst_label = _inst2obj(inst_id).split('.')[0]
            sem_label = _sem2obj(bbox_labels)
        
            if inst_label in obj_name2obj_sem_name:
                inst_label = _obj_name2obj_sem_name(inst_label)
           
            if inst_label in obj2sem:
                assert inst_label == sem_label
                
            temp_bb.append(
                np.hstack([current_points_avg, lwh, bbox_labels]))
   
        point_cloud = np.hstack([coords, rgb], dtype=np.float32)
        sem_labels = sem_labels.astype(np.int64)
        instance_ids = instance_ids.astype(np.int64)
        final_res.append({
            'scene_id': scene_id,
            'point_cloud': point_cloud,
            'sem_labels': sem_labels,
            'instance_ids': instance_ids,
            'super_points': super_points,
            'bboxs': temp_bb
        })

    return final_res

def create_bins(split, splits, path):
    data = splits[split]
    for idx, d in enumerate(tqdm(data)):
        scene_id = d['scene_id']
        d['point_cloud'].astype(
            np.float32).tofile(os.path.join(path['points'], 
                                            f'{scene_id}.bin'))
        d['sem_labels'].astype(
            np.int64).tofile(os.path.join(path['semantic_mask'], 
                                          f'{scene_id}.bin'))
        d['instance_ids'].astype(
            np.int64).tofile(os.path.join(path['instance_mask'], 
                                          f'{scene_id}.bin'))
        d['super_points'].astype(
            np.int64).tofile(os.path.join(path['super_points'], 
                                          f'{scene_id}.bin'))
        bbxs = d['bboxs']
        np.save(os.path.join(path['bboxs'], f'{scene_id}.npy'), bbxs)
        
def create_metainfo():

    return {
        'categories': obj2sem,
        'dataset': 'MultiScan',
        'info_version': '1.0'
    }

def create_data_list(split, splits, bins_path):
    num_files = len(splits[split])
    final_list = []

    for i in tqdm(range(num_files)):
        scene_id = splits[split][i]['scene_id']
        lidar_points = {
            'num_pts_feats': 6,
            'lidar_path': f'{scene_id}.bin'
        }

        raw_bboxs = np.load(os.path.join(bins_path['bboxs'], 
                                        f'{scene_id}.npy'))

        instances = []
        for rb in raw_bboxs:
            instances.append({
                'bbox_3d': rb[:6].tolist(),
                'bbox_label_3d': int(rb[-1])
            })


        final_list.append({
            'lidar_points': lidar_points,
            'instances': instances,
            'pts_semantic_mask_path': f'{scene_id}.bin',
            'pts_instance_mask_path': f'{scene_id}.bin',
            'axis_align_matrix': np.eye(4)
        })

    return final_list


def create_pkl_file(path_to_save, split, splits, 
                    bins_path, pkl_prefix = 'multiscan'):
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
        '--path_to_pths',
        required=True,
        help='Path to object instance segmentation folder',
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
    
    path_to_pths = args.path_to_pths
    create_dir(args.path_to_save_bins)
    bins_root = os.path.join(args.path_to_save_bins, 'bins')
    create_dir(bins_root)
    
    bins_path = create_dirs(bins_root)

    train_path = os.path.join(path_to_pths, 'train')
    test_path = os.path.join(path_to_pths, 'test')
    val_path = os.path.join(path_to_pths, 'val')
    
    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)
    val_files = os.listdir(val_path)
    
    final_res_train = prepare_data(train_path, train_files)
    final_res_test = prepare_data(test_path, test_files)
    final_res_val = prepare_data(val_path, val_files)

    for i in final_res_train:
        assert len([j[-1] for j in i['bboxs'] if j[-1]
                     in list(rem_lab2sem.values()) + [-1]]) == 0

    for i in final_res_test:
        assert len([j[-1] for j in i['bboxs'] if j[-1] 
                    in list(rem_lab2sem.values()) + [-1]]) == 0
    
    for i in final_res_val:
        assert len([j[-1] for j in i['bboxs'] if j[-1] 
                    in list(rem_lab2sem.values()) + [-1]]) == 0
    
    splits = {
        'train': final_res_train,
        'val': final_res_val,
        'test': final_res_test
    }

    
    create_bins(split = 'train', splits = splits, path = bins_path)
    create_bins(split = 'val', splits = splits, path = bins_path)
    create_bins(split = 'test', splits = splits, path = bins_path)
    
    create_pkl_file(bins_root, 'train', splits, bins_path)
    create_pkl_file(bins_root, 'val', splits, bins_path)
    create_pkl_file(bins_root, 'test', splits, bins_path)

    
    

    

















