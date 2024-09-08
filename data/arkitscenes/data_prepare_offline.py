# adapted from https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/data_prepare_offline.py
import argparse
import numpy as np
import os
import pandas as pd
from functools import partial
from tqdm.contrib.concurrent import process_map

import utils.box_utils as box_utils
import utils.pc_utils as pc_utils
import utils.taxonomy as taxonomy
from utils.tenFpsDataLoader import TenFpsDataLoader, extract_gt

# we keep this rough grid_size=0.05 from the original benchmark,
# however accuracy might be better with smaller grid_size
def accumulate_wrapper(loader, grid_size=0.05):
    """
    Args:
        loader: TenFpsDataLoader
    Returns:
        world_pc: (N, 3)
            xyz in world coordinate system
        world_sem: (N, d)
            semantic for each point
        grid_size: float
            keep only one point in each (g_size, g_size, g_size) grid
    """
    world_pc, world_rgb, poses = np.zeros((0, 3)), np.zeros((0, 3)), []
    for i in range(len(loader)):
        frame = loader[i]
        image_path = frame["image_path"]
        pcd = frame["pcd"]  # in world coordinate
        pose = frame["pose"]
        rgb = frame["color"]

        world_pc = np.concatenate((world_pc, pcd), axis=0)
        world_rgb = np.concatenate((world_rgb, rgb), axis=0)

        choices = pc_utils.down_sample(world_pc, grid_size)
        world_pc = world_pc[choices]
        world_rgb = world_rgb[choices]

    return world_pc, world_rgb, poses


def main(scene_id, split, data_root, output_dir):
    # step 0.0: output folder, make dir
    os.makedirs(output_dir, exist_ok=True)
    point_output_path = os.path.join(output_dir, f"{scene_id}_point.npy")
    bbox_output_path = os.path.join(output_dir, f"{scene_id}_bbox.npy")
    label_output_path = os.path.join(output_dir, f"{scene_id}_label.npy")
    # skip already processed scenes
    if os.path.exists(point_output_path) \
        and os.path.exists(bbox_output_path) \
        and os.path.exists(label_output_path):
        return

    # step 0.1: get annotation first,
    # if skipped or no gt boxes, we will not bother calling further steps
    gt_path = os.path.join(data_root, split, scene_id, f"{scene_id}_3dod_annotation.json")
    skipped, boxes_corners, _, _, labels, _ = extract_gt(gt_path)
    if skipped or boxes_corners.shape[0] == 0:
        return

    # step 0.2: data
    data_path = os.path.join(data_root, split, scene_id, f"{scene_id}_frames")
    loader = TenFpsDataLoader(
        dataset_cfg=None,
        class_names=taxonomy.class_names,
        root_path=data_path)

    # step 1: accumulate points and save points
    world_pc, world_rgb, _ = accumulate_wrapper(loader)
    # despite original benchmark script ignores rgb here, we save it 
    # to allow user to use or skip it for trainig / testing / visualization
    points = np.concatenate((world_pc, world_rgb), axis=1).astype(np.float32)
    points.tofile(point_output_path)

    # step 2: save labels and boxes
    # not sure if we need uids, but keep them followinig original benchmark
    boxes = box_utils.corners_to_boxes(boxes_corners)
    np.save(bbox_output_path, boxes)
    np.save(label_output_path, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="./3dod",
        help="input folder with ./Training/{scene_id}, ./Validation/{scene_id}"
             "and metadata.json"
    )
    parser.add_argument(
        "--output-dir",
        default="./offline_prepared_data",
        help="directory to save the data and annoation"
    )
    parser.add_argument(
        "--max-workers",
        default=1,
        type=int,
        help="number of parallel processes"
    )

    args = parser.parse_args()
    df = pd.read_csv(os.path.join(args.data_root, "metadata.csv"))
    scene_ids = list(map(str, df["video_id"].to_list()))
    splits = list(map(str, df["fold"].to_list()))
    process_map(
        partial(main, data_root=args.data_root, output_dir=args.output_dir),
        scene_ids, splits, max_workers=args.max_workers)
