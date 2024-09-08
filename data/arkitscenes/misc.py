import os
import numpy as np
import pandas as pd
from collections import defaultdict
from plyfile import PlyData
from tqdm import tqdm

import mmengine
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.apis import inference_detector, init_model
from projects.TR3D.tr3d.local_visualizer import TR3DLocalVisualizer
from utils.box_utils import boxes_to_corners_3d
from utils.pc_utils import down_sample


def verify_corners():
    a = np.random.rand(100, 7)
    mmdet3d_corners = DepthInstance3DBoxes(a, origin=(.5, .5, .5)).corners.numpy()
    a[:, -1] = -a[:, -1]
    arkiscenes_corners = boxes_to_corners_3d(a)[:, [2, 6, 7, 3, 1, 5, 4, 0]]
    assert np.abs(arkiscenes_corners - mmdet3d_corners).max() < 1e-5


def print_object_statistics(path):
    print(path)
    infos = mmengine.load(path)
    categories = infos['metainfo']['categories']
    inverse_categories = {v: k for k, v in categories.items()}
    data = {c: defaultdict(list) for c in categories}
    for d in infos['data_list']:
        for instance in d['instances']:
            category_data = data[inverse_categories[instance['bbox_label_3d']]]
            box = instance['bbox_3d']
            category_data['xy_min'].append(min(box[3], box[4]))
            category_data['xy_max'].append(max(box[3], box[4]))
            category_data['z'].append(box[5])

    quantiles = (0, .75, 1)
    columns = ['category', 'N']
    df_data = []
    for key in category_data.keys():
        for q in quantiles:
            columns.append(f'{key}.{q}')
    for category, category_data in data.items():
        table_row = [category, len(category_data['z'])]
        for key in category_data.keys():
            for q in quantiles:
                value = np.quantile(category_data[key], q)
                table_row.append(value)  # f'{value:.4f}'
        df_data.append(table_row)
    df = pd.DataFrame(data=df_data, columns=columns)
    pd.set_option('display.precision', 3)
    target = df[['xy_max.0.75', 'z.0.75']].to_numpy().max(axis=1)
    target = target > np.median(target)
    df['target'] = target
    print(df)
    print('target:', target.astype(int).tolist())


def aggregate_multiple_ply(path, grid_size=0.05):
    world_pc, world_rgb = np.zeros((0, 3)), np.zeros((0, 3))
    for file_name in tqdm(os.listdir(path)):
        data = PlyData.read(os.path.join(path, file_name))
        pc = np.stack((
            data['vertex']['x'],
            data['vertex']['y'],
            data['vertex']['z']), axis=1)
        rgb = np.stack((
            data['vertex']['red'],
            data['vertex']['green'],
            data['vertex']['blue']), axis=1)
        world_pc = np.concatenate((world_pc, pc))
        world_rgb = np.concatenate((world_rgb, rgb))
        choices = down_sample(world_pc, grid_size)
        world_pc = world_pc[choices]
        world_rgb = world_rgb[choices]
    points = np.concatenate((world_pc, world_rgb), axis=1).astype(np.float32)
    file_name = f'{os.path.basename(os.path.dirname(path))}.bin'
    points.tofile(os.path.join('work_dirs/tmp/tmp', file_name))


def predict(pcd_path, config_path, checkpoint_path):
    model = init_model(config_path, checkpoint_path, device='cuda:0',
                       cfg_options=dict(test_dataloader=dict(dataset=dict(box_type_3d='depth'))))
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 6)
    points = np.concatenate((points[:, :3], points[:, 3:] / 255), axis=1)
    result = inference_detector(model, points)
    TR3DLocalVisualizer().add_datasample(
        name='',
        data_input=dict(points=points),
        data_sample=result[0],
        draw_gt=False,
        out_file=pcd_path,
        vis_task='lidar_det')

if __name__ == '__main__':
    # verify_corners()
    # print_object_statistics('/opt/project/data/arkitscenes/arkitscenes_offline_infos_train.pkl')
    # print_object_statistics('/opt/project/data/arkitscenes/arkitscenes_offline_infos_val.pkl')
    aggregate_multiple_ply('data/tmp/230621_sr_room_samples/Jun18at10-18PM-poly/pcds')
    predict(
        'work_dirs/tmp/tmp/Jun18at10-18PM-poly.bin',
        'projects/arkitscenes/configs/tr3d_1xb16_arkitscenes-offline-3d-4class.py',
        'work_dirs/tmp/tr3d_arkitscenes_epoch10.pth')
