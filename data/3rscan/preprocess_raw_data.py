import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle
import sys
import os
import argparse
import glob
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from utils import read_objmesh, point_indices_from_group

#CLOUD_FILE_PFIX = 'mesh.refined.v2.color'
CLOUD_FILE_PFIX = 'mesh.refined.v2'
AGGREGATIONS_FILE_PFIX = 'semseg.v2.json'
SEGMENTS_FILE_PFIX = 'mesh.refined.0.010000.segs.v2.json'


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def read_transform_matrix(Scan3RJson_PATH):
    rescan2ref = {}
    with open(Scan3RJson_PATH , "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = \
                        np.array(scans["transform"]).reshape(4,4).T
    return rescan2ref

def get_reference_dic(Scan3RJson_PATH):
    meta_data = json.load(open(Scan3RJson_PATH))
    reference_dic = {}
    for record in meta_data:
        reference = record['reference'] 
        reference_dic[reference] = reference
        if 'scans' not in record:
            continue
        for scan in record['scans']:
            reference_dic[scan['reference']] = reference
    return reference_dic

def handle_process(scene_path, output_path, labels_pd, 
                   train_scenes, val_scenes, test_scenes):
    scene_id = scene_path.split('/')[-1]
    obj_path = os.path.join(scene_path, f'{CLOUD_FILE_PFIX}.obj')    
    aggregations_file = os.path.join(scene_path, f'{AGGREGATIONS_FILE_PFIX}')
    segments_file = os.path.join(scene_path, f'{SEGMENTS_FILE_PFIX}')
    # Rotating the mesh to axis aligned
    rot_matrix = rescan2ref.get(scene_id, np.identity(4))
    
    ref_scene_id = reference_dic[scene_id]
    ref_rot_matrix = reference_axis_align_matrix_dic[ref_scene_id]
    
    if scene_id in train_scenes:
        split_name = 'train'
    elif scene_id in val_scenes:
        split_name = 'val'
    elif scene_id in test_scenes:
        split_name = 'test'
    else:
        print('*', scene_id, 
              'does not exist in [train, val, test] that have seg files')
        return 

    print('Processing: ', scene_id, 'in', split_name)
    
    pointcloud, faces_array = read_objmesh(obj_path)
    points = pointcloud[:, :3]
    colors = pointcloud[:, 3:6]

    # Rotate PC to axis aligned
    r_points = pointcloud[:, :3].transpose()
    r_points = np.append(r_points, np.ones((1, 
                                            r_points.shape[1])), axis=0)
    # reference align
    r_points = np.dot(rot_matrix, r_points)
    # reference axis align
    r_points = np.dot(ref_rot_matrix, r_points)
    ##### !
    aligned_pointcloud = np.append(r_points.transpose()[:, :3], 
                                   pointcloud[:, 3:], axis=1)

    # Generate new labels
    labelled_pc = np.zeros((pointcloud.shape[0], 1)) - 1 # -1: unannotated
    instance_ids = np.zeros((pointcloud.shape[0], 1)) - 1 # -1: unannotated
        
    if os.path.isfile(aggregations_file):
        # Load segments file
        with open(segments_file) as f:
            segments = json.load(f)
            seg_indices = np.array(segments['segIndices'])        
        # Load Aggregations file
        with open(aggregations_file) as f:
            aggregation = json.load(f)
            seg_groups = np.array(aggregation['segGroups'])

        num_instances = len(seg_groups)        
        instance_bboxes = np.zeros((num_instances, 7))
        aligned_instance_bboxes = np.zeros((num_instances, 7))
            
        for obj_idx, group in enumerate(seg_groups):
            segment_points, aligned_segment_points, p_inds, label_id = \
                point_indices_from_group(pointcloud, aligned_pointcloud, 
                                         seg_indices, group, labels_pd)
            labelled_pc[p_inds] = label_id
            
            if len(segment_points) == 0: continue
                
            xmin = np.min(segment_points[:,0])
            ymin = np.min(segment_points[:,1])
            zmin = np.min(segment_points[:,2])
            xmax = np.max(segment_points[:,0])
            ymax = np.max(segment_points[:,1])
            zmax = np.max(segment_points[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, 
                             (zmin+zmax)/2, xmax-xmin, 
                             ymax-ymin, zmax-zmin, label_id]) # also include object id
            instance_bboxes[obj_idx,:] = bbox 
            
            if len(aligned_segment_points) == 0: continue
                
            instance_ids[p_inds] = obj_idx
            xmin = np.min(aligned_segment_points[:,0])
            ymin = np.min(aligned_segment_points[:,1])
            zmin = np.min(aligned_segment_points[:,2])
            xmax = np.max(aligned_segment_points[:,0])
            ymax = np.max(aligned_segment_points[:,1])
            zmax = np.max(aligned_segment_points[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, 
                             (zmin+zmax)/2, xmax-xmin, ymax-ymin, 
                             zmax-zmin, label_id]) # also include object id
            aligned_instance_bboxes[obj_idx,:] = bbox 
    else:
        # use zero as placeholders for the test scene
        #print("use placeholders")
        instance_bboxes = np.zeros((1, 7)) 
        aligned_instance_bboxes = np.zeros((1, 7))

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)
    assert np.all(instance_ids[np.where(labelled_pc == -1)[0]] == -1)
    if -1 in np.unique(instance_ids):
        assert len(instance_bboxes) == len(np.unique(instance_ids)[1:])
        
    else:
        assert len(instance_bboxes) == len(np.unique(instance_ids))   
    
    if (np.any(np.isnan(pointcloud)) or not np.all(np.isfinite(pointcloud))):
        raise ValueError('nan')   
    
    output_path = os.path.join(output_path, f'{scene_id}')
    create_dir(os.path.join(output_path))
    output_prefix = os.path.join(output_path, f'{scene_id}')
    np.save(output_prefix+'_aligned_vert.npy', aligned_pointcloud[:, :6])
    np.save(output_prefix+'_sem_label.npy', labelled_pc)
    np.save(output_prefix+'_ins_label.npy', instance_ids)
    np.save(output_prefix+'_aligned_bbox.npy', aligned_instance_bboxes)
    np.save(output_prefix+'_superpoints.npy', seg_indices)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='../data/3rscan/', 
                        help='Path to the 3RScan dataset containing scene folders')
    parser.add_argument('--output_root', default='preprocessed_raw_data', 
                        help='Output path where processed data will be located')
    parser.add_argument('--label_map_file', 
                        default='meta_data/3RScan.v2_Semantic-Classes-Mapping.csv', 
                        help='path to scannetv2-labels.combined.tsv')
    parser.add_argument('--num_workers', default=12, 
                        type=int, help='The number of parallel workers')
    parser.add_argument('--splits_path', default='meta_data/split', 
                        help='Where the txt files with the train/val splits live')
    config = parser.parse_args()

    # Load label map
    labels_pd = pd.read_csv(config.label_map_file, sep=',', header=1)

    # Load train/val splits
    with open(config.splits_path + '/train.txt') as train_file:
        train_scenes = train_file.read().splitlines()
    with open(config.splits_path + '/val.txt') as val_file:
        val_scenes = val_file.read().splitlines()
    with open(config.splits_path + '/test.txt') as test_file:
        test_scenes = test_file.read().splitlines()
        
    META_FILE = 'meta_data/3RScan.json'
    rescan2ref = read_transform_matrix(META_FILE)
    reference_dic = get_reference_dic(META_FILE)
    
    with open('./meta_data/reference_axis_align_matrix.pkl', 'rb') as f:
        reference_axis_align_matrix_dic = pickle.load(f)

    os.makedirs(config.output_root, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + '/*'))
    
    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    print('Processing scenes...')
    _ = list(pool.map(handle_process, scene_paths, 
                      repeat(config.output_root), repeat(labels_pd), 
                      repeat(train_scenes), repeat(val_scenes), 
                      repeat(test_scenes)))
