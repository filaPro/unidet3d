from pathlib import Path
import numpy as np
from sklearn.neighbors import KDTree
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Remap superpoints from source to destination point cloud data')
    parser.add_argument('--src', type=str, required=True, help='Path to source data')
    parser.add_argument('--dst', type=str, required=True, help='Path to destination data')
    return parser.parse_args()

args = parse_args()

src_folder = Path(args.src)
dst_folder = Path(args.dst)

for src_file in tqdm(src_folder.glob('points/*.bin'), desc="Processing files"):
    pcds_src = np.fromfile(src_file, dtype=np.float32).reshape(-1, 6)[:, :3]
    sp_src = np.fromfile(src_file.parent.parent / 'super_points' / src_file.name, dtype=np.int64)
    
    dst_file = dst_folder / 'points' / src_file.name
    if dst_file.exists():
        pcds_dst = np.fromfile(dst_file, dtype=np.float32).reshape(-1, 6)[:, :3]
        
        tree = KDTree(pcds_src)
        _, indices = tree.query(pcds_dst, k=1)
        sp_dst = sp_src[indices.flatten()]
        
        dst_sp_file = dst_file.parent.parent / 'super_points' / dst_file.name
        dst_sp_file.parent.mkdir(parents=True, exist_ok=True)
        sp_dst.astype(np.int64).tofile(dst_sp_file)        
    else:
        print(f"Corresponding file not found in destination folder: {dst_file}")