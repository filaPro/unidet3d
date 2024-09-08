## Prepare MultiScan Data for Indoor 3D Detection

1. Download and unzip data from the official [MultiScan](https://github.com/smartscenes/multiscan?tab=readme-ov-file).

2. Generate bins and pkls data by running:

```bash
python prepare_bins_pkls.py --path_to_pths path_to_unzipped_folder --path_to_save_bins path_to_save_bins
```

Overall you achieve the following file structure in `bins` directory:
```
bins
├── bboxs
│   ├── xxxxx_xx.npy
├── instance_mask
│   ├── xxxxx_xx.bin
├── points
│   ├── xxxxx_xx.bin
├── semantic_mask
│   ├── xxxxx_xx.bin
├── super_points
│   ├── xxxxx_xx.bin
├── multiscan_infos_train.pkl
├── multiscan_infos_val.pkl
├── multiscan_infos_test.pkl
```
