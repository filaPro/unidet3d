import os
import argparse 
import json
import numpy as np
from plyfile import PlyData
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import shutil
import segmentator
import torch
import trimesh

POINT_CLOUD_PFX = "mesh_aligned_0.05.ply"
SEGMENTS_ANNO_PFX = "segments_anno.json"

def _handle_id(scene_id):
    print(f'Processing: {scene_id}')
    if not os.path.isdir(os.path.join(PATH_TO_IDS, scene_id, 'scans')):
        return
    
    point_cloud, _ = read_plymesh(os.path.join(PATH_TO_IDS, scene_id, 
                                               'scans', POINT_CLOUD_PFX))

    mesh = trimesh.load_mesh(os.path.join(PATH_TO_IDS, scene_id, 
                                          'scans', POINT_CLOUD_PFX))
    vertices = mesh.vertices
    faces = mesh.faces
        
    vertices = torch.from_numpy(vertices.astype(np.float32))
    faces = torch.from_numpy(faces.astype(np.int64))
    super_points = segmentator.segment_mesh(vertices, faces).numpy()

    mapping_superpoints = {tuple(i.tolist()): 
                           super_points[idx] for idx, i in enumerate(vertices)}
    super_points = np.array([mapping_superpoints[tuple(i.tolist())] 
                             for i in point_cloud[:, :3]])

    assert point_cloud.shape[1] == 6
    assert point_cloud.shape[0] == super_points.shape[0]

    semantic = np.zeros((point_cloud.shape[0], 1)) - 1 # -1: unannotated
    instance = np.zeros((point_cloud.shape[0], 1)) - 1 # -1: unannotated
    if scene_id in TRAIN_IDS or scene_id in VAL_IDS:
        seg_anno = load_json(os.path.join(PATH_TO_IDS, scene_id, 
                                          'scans', SEGMENTS_ANNO_PFX))
        seg_groups = seg_anno['segGroups']
        obj_idx = 0
        bboxs = []
        for idx, group in enumerate(seg_groups):
            label = group['label']
            segments = np.array(group['segments'])

            if label in TOP100SEM2ID:
                new_label = label
                
            elif label in SEMANTIC_MAP_TO and label not in TOP100SEM2ID:
                if SEMANTIC_MAP_TO[label] in TOP100SEM2ID:
                    new_label = SEMANTIC_MAP_TO[label]
                else:
                    continue
            else:
                continue

            label_id = TOP100SEM2ID[new_label]
            
            point_segments = point_cloud[segments]
            instance[segments] = obj_idx
            semantic[segments] = label_id
            xmin = np.min(point_segments[:,0])
            ymin = np.min(point_segments[:,1])
            zmin = np.min(point_segments[:,2])
            xmax = np.max(point_segments[:,0])
            ymax = np.max(point_segments[:,1])
            zmax = np.max(point_segments[:,2])
            
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, 
                             xmax-xmin, ymax-ymin, zmax-zmin, label_id])
            
            bboxs.append(bbox)
            obj_idx += 1
            
        bboxs = np.stack(bboxs)
        data = {
                'point_cloud': point_cloud,
                'semantic': semantic[:, 0].astype(int),
                'instance': instance[:, 0].astype(int),
                'bboxs': bboxs,
                'super_points': super_points
                }
        
    elif scene_id in SEM_TEST_IDS:

        data =  {
                'point_cloud': point_cloud,
                'semantic': semantic[:, 0].astype(int),
                'instance': instance[:, 0].astype(int),
                'bboxs': np.zeros((0,7)),
                'super_points': super_points
                }

    output_path = os.path.join(OUTPUT_DIR_DATA, f'{scene_id}')
    create_dir(os.path.join(output_path))
    output_prefix = os.path.join(output_path, f'{scene_id}')
    np.save(output_prefix+'_point_cloud.npy', data['point_cloud'])
    np.save(output_prefix+'_semantic.npy', data['semantic'])
    np.save(output_prefix+'_instance.npy', data['instance'])
    np.save(output_prefix+'_bboxs.npy', data['bboxs'])
    np.save(output_prefix+'_superpoints.npy', data['super_points'])

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_json(path):
    with open(path) as jd:
        return json.load(jd)

def load_txt(path):
    res = []

    with open(path) as f:
        for line in tqdm(f):
            res.append(line.strip())

    return res

def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata['vertex'].data).values
        faces = np.array([f[0] for f in plydata["face"].data])
        return vertices, faces
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_data',
        required=True,
        help='Path to raw data',
        type=str,
    )

    parser.add_argument(
        '--output_dir',
        required=True,
        help='Path to save preprocessed raw data',
        type=str,
    )

    parser.add_argument('--num_workers', default=20, type=int, 
                        help='The number of parallel workers')

    args = parser.parse_args()
    print(args)
    PATH_TO_DATA = args.path_to_data
    PATH_TO_IDS = os.path.join(PATH_TO_DATA, 'data')
    OUTPUT_DIR = args.output_dir
    create_dir(OUTPUT_DIR)

    OUTPUT_DIR_DATA = os.path.join(OUTPUT_DIR, 'data')
    create_dir(OUTPUT_DIR_DATA)

    TOP100SEM2ID = {}
    with open(os.path.join(PATH_TO_DATA , 
                           'metadata/semantic_benchmark/top100.txt')) as f:
        # check = f.read()
        for idx, line in enumerate(f):
            line = line.strip()
            TOP100SEM2ID[line] = idx

    TOPINST2ID = {}
    with open(os.path.join(PATH_TO_DATA, 
            'metadata/semantic_benchmark/top100_instance.txt')) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            TOPINST2ID[line] = TOP100SEM2ID[line]

    MAPPING_BENCH = pd.read_csv(os.path.join(PATH_TO_DATA, 
                    'metadata/semantic_benchmark/map_benchmark.csv'))
    SEMANTIC_MAP_TO = MAPPING_BENCH[~MAPPING_BENCH['semantic_map_to'].isna()]
    INSTANCE_MAP_TO = MAPPING_BENCH[~MAPPING_BENCH['instance_map_to'].isna()]

    SEMANTIC_MAP_TO = SEMANTIC_MAP_TO[['class','semantic_map_to']].values
    SEMANTIC_MAP_TO = dict(zip(SEMANTIC_MAP_TO[:, 0], SEMANTIC_MAP_TO[:, 1]))
    print(len(SEMANTIC_MAP_TO))

    INSTANCE_MAP_TO = INSTANCE_MAP_TO[['class','instance_map_to']].values
    INSTANCE_MAP_TO = dict(zip(INSTANCE_MAP_TO[:, 0], INSTANCE_MAP_TO[:, 1]))
    print(len(INSTANCE_MAP_TO))

    SCENE_IDS = os.listdir(os.path.join(PATH_TO_DATA, 'data'))
    SCENE_IDS.remove('.ipynb_checkpoints')
    
    assert len(SCENE_IDS) == 380

    path_to_train_ids = os.path.join(PATH_TO_DATA, 'splits', 'nvs_sem_train.txt')
    TRAIN_IDS = load_txt(path_to_train_ids)
    path_to_val_ids = os.path.join(PATH_TO_DATA, 'splits', 'nvs_sem_val.txt')
    VAL_IDS = load_txt(path_to_val_ids)
    path_to_sem_test_ids = os.path.join(PATH_TO_DATA, 'splits', 'sem_test.txt')
    SEM_TEST_IDS = load_txt(path_to_sem_test_ids)

    shutil.copytree(os.path.join(PATH_TO_DATA, 'splits'), 
                    OUTPUT_DIR, dirs_exist_ok=True)

    pool = ProcessPoolExecutor(max_workers=args.num_workers)
    print('Processing scenes...')
    _ = list(pool.map(_handle_id, SCENE_IDS))
















