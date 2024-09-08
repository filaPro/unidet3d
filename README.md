## UniDet3D: Multi-dataset Indoor 3D Object Detection

**News**:
 * :fire: September, 2024. UniDet3D is state-of-the-art in 6 indoor benchmarks: ScanNet ?paperswithcode?, ARKitScenes ?paperswithcode?, S3DIS ?paperswithcode?, MultiScan ?paperswithcode?, 3RScan ?paperswithcode?, and ScanNet++ ?paperswithcode?.  

This repository contains an implementation of UniDet3D, a multi-dataset indoor 3D object detection method introduced in our paper:

> **UniDet3D: Multi-dataset Indoor 3D Object Detection**<br>
> [Maksim Kolodiazhnyi](https://github.com/col14m),
> [Anna Vorontsova](https://github.com/highrut),
> [Matvey Skripkin](https://scholar.google.com/citations?user=hAlwb4wAAAAJ),
> [Danila Rukhovich](https://github.com/filaPro),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Artificial Intelligence Research Institute<br>
> https://arxiv.org/abs/2409.?????

### Installation

For convenience, we provide a [Dockerfile](Dockerfile).
This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework `v1.1.0`. If not using Docker, please follow [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/get_started.md) for the installation instructions.


### Getting Started

Please see [test_train.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/user_guides/train_test.md) for some basic usage examples.

#### Data Preprocessing

UniDet3D is trained and tested using 6 datasets: [ScanNet](data/scannet), [ARKitScenes](data/arkitscenes), [S3DIS](data/s3dis), [MultiScan](data/multiscan), [3RScan](data/3rscan), and [ScanNet++](data/scannetpp).
Preprocessed data can be found at our [Hugging Face](https://huggingface.co/datasets/maksimko123/UniDet3D). Download each archive, unpack, and move into the corresponding directory in [data](data). Please comply with the license agreement before downloading the data.

Alternatively, you can preprocess the data by youself. 
Training data for 3D object detection methods that do not requires superpoints, e.g. [TR3D](https://github.com/SamsungLabs/tr3d) or [FCAF3D](https://github.com/SamsungLabs/tr3d), can be prepared according to the [instructions](data).

Superpoints for ScanNet and MultiScan are provided as a part of the original annotation. For the rest datasets, you can either download pre-computed superpoints at our [Hugging Face](https://huggingface.co/datasets/maksimko123/UniDet3D), or compute them using [superpoint_transformer](https://github.com/drprojects/superpoint_transformer).

#### Training

Before training, please download the backbone [checkpoint](https://github.com/filapro/oneformer3d/releases/download/v1.0/oneformer3d_1xb4_scannet.pth) and save it under `work_dirs/tmp`.

To train UniDet3D on 6 datasets jointly, simply run the [training](tools/train.py) script:

```bash
python tools/train.py configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py
```

UniDet3D can also be trained on individual datasets, e.g., we provide a [config](configs/unidet3d_1xb8_scannet.py) for training using ScanNet solely.


#### Testing

To test a trained model, you can run the [testing](tools/test.py) script:

```bash
python tools/test.py configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py \
    work_dirs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes/epoch_1024.pth
```

UniDet3D can also be tested on individual datasets. To this end, simply remove the unwanted datasets from `val_dataloader.dataset.datasets` in the config file.

#### Visualization

To visualize ground truth and predicted boxes, run the [testing](tools/test.py) script with additional arguments:

```bash
python tools/test.py configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py \
    work_dirs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes/latest.pth --show \
    --show-dir work_dirs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes
```
You can also set `score_thr` in configs to `0.3` for better visualizations.

### Trained Model

Please refer to the UniDet3D [checkpoint](https://github.com/filapro/unidet3d/releases/download/v1.0/unidet3d.pth) and [log file](https://github.com/filapro/unidet3d/releases/download/v1.0/log.txt). The corresponding metrics are given below (they might slightly deviate from the values reported in the paper due to the randomized training/testing procedure).

| Dataset     | mAP<sub>25</sub>  | mAP<sub>50</sub>  |
|:-----------:|:-----------------:|:-----------------:|
| ScanNet     | 77.0              | 65.9              |
| ARKitScenes | 60.1              | 47.2              |
| S3DIS       | 76.7              | 65.3              |
| MultiScan   | 62.6              | 52.3              |
| 3RScan      | 63.6              | 44.9              |
| ScanNet++   | 24.0              | 16.8              |

### Predictions Example

<p align="center">
  <img src="???" alt="UniDet3D predictions"/>
</p>

### Citation

If you find this work useful for your research, please cite our paper:

```
@article{kolodiazhnyi2024unidet3d,
  title={UniDet3D: Multi-dataset Indoor 3D Object Detection},
  author={Kolodiazhnyi, Maxim and Vorontsova, Anna and Skripkin, Matvey and Rukhovich, Danila and Konushin, Anton},
  journal={arXiv preprint arXiv:2409.?????},
  year={2024}
}
```
