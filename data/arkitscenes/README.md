## Prepare ARKitScenes Data for Indoor 3D Detection

For now we only support offline benchmark with a single reconstructed point clound for each scene. Online benchmark for single RGB-D frame detection can be supported in the future. The `utils` directory is used unchanged from [ARKitScenes](https://github.com/apple/ARKitScenes/tree/main/threedod/benchmark_scripts/utils), except fixing a single [issue](https://github.com/apple/ARKitScenes/issues/53).

1. Download data from the official [ARKitScenes](https://github.com/apple/ARKitScenes). From their repo you may run:
```
python download_data.py 3dod --video-id-csv threedod/3dod_train_val_splits.csv
```

After this step you have the following file structure here:
```
3dod
├── metadata.csv
├── Training
│   ├── xxxxxxxx
│   │   ├── xxxxxxxx_3dod_annotation.json
│   │   ├── xxxxxxxx_3dod_mesh.ply
│   │   ├── xxxxxxxx_frames
├── Validation
│   ├── xxxxxxxx
│   │   ├── xxxxxxxx_3dod_annotation.json
│   │   ├── xxxxxxxx_3dod_mesh.ply
│   │   ├── xxxxxxxx_frames
```

2. Preprocess data for offline benchmark with our adapted script:
```
python data_prepare_offline.py
```
After this step you have the following file structure here:
```
offline_prepared_data
├── xxxxxxxx_point.npy
├── xxxxxxxx_bbox.npy
├── xxxxxxxx_label.npy
```

3. Enter the project root directory, generate training and validation data by running:
```
python tools/create_data.py arkitscenes --root-path ./data/arkitscenes --out-dir ./data/arkitscenes --extra-tag arkitscenes-offline
```
Overall you achieve the following file structure in `data` directory:
```
arkitscenes
├── offline_prepared_data
│   ├── xxxxxxxx_point.bin
├── arkitscenes_offline_train_infos.pkl
├── arkitscenes_offline_val_infos.pkl

```
