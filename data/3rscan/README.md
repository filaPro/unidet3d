## Prepare 3RScan Data for Indoor 3D Detection

1. Download data from the official [3RScan](https://waldjohannau.github.io/RIO/).

2. Preprocess raw data by running:

```bash
python preprocess_raw_data.py --dataset_root path_to_dataset --output_root path_to_save_preprocessed_raw_data
```

3. Generate bins and pkls data by running:

```bash
python prepare_bins_pkls.py --path_to_data path_to_preprocessed_raw_data --path_to_save_bins path_to_save_bins
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
├── superpoints
│   ├── xxxxx_xx.bin
├── 3rscan_infos_train.pkl
├── 3rscan_infos_val.pkl
├── 3rscan_infos_test.pkl
```
