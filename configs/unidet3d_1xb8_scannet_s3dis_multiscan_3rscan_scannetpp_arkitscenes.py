_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(imports=['unidet3d'])


classes_scannet = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
                    'toilet', 'sink', 'bathtub', 'otherfurniture']
classes_s3dis = ['table', 'chair', 'sofa', 'bookcase', 'board']

classes_multiscan = ['door', 'table',  'chair',  'cabinet',  'window',  'sofa',  'microwave',  'pillow',  
         'tv_monitor',  'curtain',  'trash_can',  'suitcase',  'sink',  'backpack',  'bed',  
         'refrigerator',  'toilet']

classes_3rscan = classes_scannet

classes_scannetpp = ['table', 'door', 'ceiling lamp', 'cabinet', 'blinds', 'curtain', 'chair', 'storage cabinet', 'office chair', 'bookshelf', 'whiteboard', 'window', 'box', 
                     'monitor', 'shelf', 'heater', 'kitchen cabinet', 'sofa', 'bed', 'trash can', 'book', 'plant', 'blanket', 'tv', 'computer tower', 'refrigerator', 'jacket', 
                     'sink', 'bag', 'picture', 'pillow', 'towel', 'suitcase', 'backpack', 'crate', 'keyboard', 'rack', 'toilet', 'printer', 'poster', 'painting', 'microwave', 'shoes', 
                     'socket', 'bottle', 'bucket', 'cushion', 'basket', 'shoe rack', 'telephone', 'file folder', 'laptop', 'plant pot', 'exhaust fan', 'cup', 'coat hanger', 'light switch', 
                     'speaker', 'table lamp', 'kettle', 'smoke detector', 'container', 'power strip', 'slippers', 'paper bag', 'mouse', 'cutting board', 'toilet paper', 'paper towel', 
                     'pot', 'clock', 'pan', 'tap', 'jar', 'soap dispenser', 'binder', 'bowl', 'tissue box', 'whiteboard eraser', 'toilet brush', 'spray bottle', 'headphones', 'stapler', 'marker']

classes_arkitscenes = ['cabinet', 'refrigerator', 'shelf', 'stove', 'bed',
                        'sink', 'washer', 'toilet', 'bathtub', 'oven',
                        'dishwasher', 'fireplace', 'stool', 'chair', 'table',
                        'tv_monitor', 'sofa']

# model settings
num_channels=32
voxel_size=0.02

model = dict(
    type='UniDet3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=voxel_size,
    min_spatial_shape=128,
    query_thr=3000,
    bbox_by_mask=[True, True, False, False, False, False],
    target_by_distance=[False, False, True, True, True, True],
    use_superpoints=[True, True, True, False, False, False],
    fast_nms=[True, False, True, True, True, None],
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_scannet, classes_s3dis, 
                          classes_multiscan, classes_3rscan,
                          classes_scannetpp, classes_arkitscenes],
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        datasets=['scannet', 's3dis', 'multiscan', '3rscan', 
                  'scannetpp', 'arkitscenes'],
        angles=[False, False, False, False, False, True]),
    criterion=dict(
        type='UniDet3DCriterion',
            datasets=['scannet', 's3dis', 'multiscan', '3rscan', 
                      'scannetpp', 'arkitscenes'],
            datasets_weights=[1, 1, 1, 1, 1, 1],
            bbox_loss_simple=dict(
                type='UniDet3DAxisAlignedIoULoss',
                mode='diou',
                reduction='none'),
            bbox_loss_rotated=dict(
                type='UniDet3DRotatedIoU3DLoss',
                mode='diou',
                reduction='none'),
            matcher=dict(
                type='UniMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='BboxCostJointTraining', 
                            weight=2.0,
                            loss_simple=dict(
                                type='UniDet3DAxisAlignedIoULoss',
                                mode='diou',
                                reduction='none'),
                            loss_rotated=dict(
                                type='UniDet3DRotatedIoU3DLoss',
                                mode='diou',
                                reduction='none'))]),
            loss_weight=[0.5, 1.0],
            non_object_weight=0.1,
            topk=[6, 6, 3, 3, 3, 3],
            iter_matcher=True),
    train_cfg=dict(topk=6),
    test_cfg=dict(
        low_sp_thr=0.18,
        up_sp_thr=0.81,
        topk_insts=1000,
        score_thr=0,
        iou_thr=[0.5, 0.55, 0.55, 0.55, 0.55, 0.55]))

# scannet dataset settings

metainfo_scannet = dict(classes=classes_scannet)
data_root_scannet = 'data/scannet/'

max_class_scannet = 20
dataset_type_scannet = 'ScanNetDetDataset'
data_prefix_scannet = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

train_pipeline_scannet = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='PointDetClassMappingScanNet',
        num_classes=max_class_scannet,
        stuff_classes=[0, 1]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=0.5),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'sp_pts_mask', 'gt_sp_masks', 'elastic_coords'
        ])
]
test_pipeline_scannet = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]


# s3dis dataset settings
dataset_type_s3dis = 'S3DISSegDetDataset'
data_root_s3dis = 'data/s3dis/'
data_prefix_s3dis = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

train_area = [1, 2, 3, 4, 6]
test_area = 5

train_pipeline_s3dis = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_label_3d=False,
        with_bbox_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(
        type='PointSample_',
        num_points=180000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='PointDetClassMappingS3DIS',
        classes=[7, 8, 9, 10, 11]),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=-1),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'elastic_coords', 'gt_labels_3d', 
            'sp_pts_mask', 'gt_sp_masks',
            'pts_semantic_mask', 'pts_instance_mask'
        ])
]
test_pipeline_s3dis = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PointSample_',
                num_points=180000),
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]

# multiscan dataset settings
data_root_multiscan = 'data/multiscan/bins'
dataset_type_multiscan = 'MultiScan_'
data_prefix_multiscan = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

train_pipeline_multiscan = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=True,
         with_bbox_3d=True,
         with_sp_mask_3d=True),
    dict(type='PointSample_', 
         num_points=100000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=-1),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'elastic_coords', 'gt_bboxes_3d', 
              'gt_labels_3d', 'sp_pts_mask'])
]
test_pipeline_multiscan = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=False,
         with_bbox_3d=False,
         with_sp_mask_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample_', num_points=100000),
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]

# 3rscan dataset settings
data_root_3rscan = 'data/3rscan'
dataset_type_3rscan = 'ThreeRScan_'
data_prefix_3rscan = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points_spt')

train_pipeline_3rscan = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=True,
         with_bbox_3d=True,
         with_sp_mask_3d=True),
    dict(type='PointSample_', 
         num_points=100000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=-1),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'elastic_coords', 'gt_bboxes_3d', 
              'gt_labels_3d', 'sp_pts_mask'])
]
test_pipeline_3rscan = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=False,
         with_bbox_3d=False,
         with_sp_mask_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample_', num_points=100000),
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]


# scannetpp dataset settings
data_root_scannetpp = 'data/scannetpp/bins'
dataset_type_scannetpp = 'Scannetpp_'
data_prefix_scannetpp = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points_spt')

train_pipeline_scannetpp = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=True,
         with_bbox_3d=True,
         with_sp_mask_3d=True),
    dict(type='PointSample_', 
         num_points=200000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=-1),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'elastic_coords', 'gt_bboxes_3d', 
              'gt_labels_3d', 'sp_pts_mask'])
]
test_pipeline_scannetpp = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=False,
         with_bbox_3d=False,
         with_sp_mask_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample_', num_points=200000),
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]

# arkitscenes dataset settings
dataset_type_arkitscenes = 'ARKitScenesOfflineDataset'
data_root_arkitscenes = 'data/arkitscenes'
data_prefix_arkitscenes = dict(
    pts='offline_prepared_data',
    sp_pts_mask='super_points')

train_pipeline_arkitscenes = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=True,
         with_bbox_3d=True,
         with_sp_mask_3d=True),
    dict(type='PointSample_', num_points=100000),
    dict(
        type='DenormalizePointsColor',
        color_mean=[0, 0, 0],
        color_std=[255, 255, 255]),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.5, 0.5],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=-1),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'elastic_coords', 'gt_bboxes_3d', 
              'gt_labels_3d', 'sp_pts_mask'])
]
test_pipeline_arkitscenes = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=False,
         with_bbox_3d=False,
         with_sp_mask_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample_', num_points=100000),
            dict(
                type='DenormalizePointsColor',
                color_mean=[0, 0, 0],
                color_std=[255, 255, 255]),
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]


# run settings
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset_',
        datasets=[dict(
                    type=dataset_type_scannet,
                    ann_file='scannet_infos_train.pkl',
                    data_prefix=data_prefix_scannet,
                    data_root=data_root_scannet,
                    metainfo=metainfo_scannet,
                    pipeline=train_pipeline_scannet,
                    ignore_index=max_class_scannet,
                    scene_idxs=None,
                    test_mode=False)] + \
                 [dict(
                    type=dataset_type_s3dis,
                    data_root=data_root_s3dis,
                    ann_file=f's3dis_sp_infos_Area_{i}.pkl',
                    partition=0.33,
                    pipeline=train_pipeline_s3dis,
                    filter_empty_gt=True,
                    data_prefix=data_prefix_s3dis,
                    box_type_3d='Depth',
                    backend_args=None) for i in train_area] + \
                [dict(
                    type=dataset_type_multiscan,
                    ann_file='multiscan_infos_train.pkl',
                    partition=0.25,
                    data_prefix=data_prefix_multiscan,
                    data_root=data_root_multiscan,
                    pipeline=train_pipeline_multiscan,
                    test_mode=False)] + \
                [dict(
                    type=dataset_type_3rscan,
                    ann_file='3rscan_infos_train.pkl',
                    partition=0.15,
                    data_prefix=data_prefix_3rscan,
                    data_root=data_root_3rscan,
                    pipeline=train_pipeline_3rscan,
                    test_mode=False)] + \
                [dict(
                    type=dataset_type_scannetpp,
                    ann_file='scannetpp_infos_train.pkl',
                    partition=0.33,
                    data_prefix=data_prefix_scannetpp,
                    data_root=data_root_scannetpp,
                    pipeline=train_pipeline_scannetpp,
                    test_mode=False)] + \
                [dict(
                    type=dataset_type_arkitscenes,
                    ann_file='arkitscenes_offline_infos_train.pkl',
                    partition=0.08,
                    data_prefix=data_prefix_arkitscenes,
                    data_root=data_root_arkitscenes,
                    pipeline=train_pipeline_arkitscenes,
                    test_mode=False)] 
                    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset_',
        datasets= \
                [dict(
                    type=dataset_type_scannet,
                    ann_file='scannet_infos_val.pkl',
                    data_prefix=data_prefix_scannet,
                    data_root=data_root_scannet,
                    metainfo=metainfo_scannet,
                    pipeline=test_pipeline_scannet,
                    ignore_index=max_class_scannet,
                    test_mode=True)] + \
                [dict(
                    type=dataset_type_s3dis,
                    data_root=data_root_s3dis,
                    ann_file=f's3dis_sp_infos_Area_{test_area}.pkl',
                    pipeline=test_pipeline_s3dis,
                    test_mode=True,
                    data_prefix=data_prefix_s3dis,
                    box_type_3d='Depth',
                    backend_args=None)] + \
                [dict(
                    type=dataset_type_multiscan,
                    ann_file='multiscan_infos_val.pkl',
                    data_prefix=data_prefix_multiscan,
                    data_root=data_root_multiscan,
                    pipeline=test_pipeline_multiscan,
                    test_mode=True)] + \
                [dict(
                    type=dataset_type_3rscan,
                    ann_file='3rscan_infos_val.pkl',
                    data_prefix=data_prefix_3rscan,
                    data_root=data_root_3rscan,
                    pipeline=test_pipeline_3rscan,
                    test_mode=True)] + \
                [dict(
                    type=dataset_type_scannetpp,
                    ann_file='scannetpp_infos_val.pkl',
                    data_prefix=data_prefix_scannetpp,
                    data_root=data_root_scannetpp,
                    pipeline=test_pipeline_scannetpp,
                    test_mode=True)] + \
                [dict(
                    type=dataset_type_arkitscenes,
                    ann_file='arkitscenes_offline_infos_val.pkl',
                    data_prefix=data_prefix_arkitscenes,
                    data_root=data_root_arkitscenes,
                    pipeline=test_pipeline_arkitscenes,
                    test_mode=True)] 
                    ))

test_dataloader = val_dataloader

load_from = 'work_dirs/tmp/oneformer3d_1xb4_scannet.pth'

test_evaluator = dict(type='IndoorMetric_', 
                      datasets=['scannet', 's3dis', 'multiscan', '3rscan', 'scannetpp', 'arkitscenes'],
                      datasets_classes=[classes_scannet, classes_s3dis, 
                                        classes_multiscan, classes_3rscan,
                                        classes_scannetpp, classes_arkitscenes])

val_evaluator = test_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001 * 2, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=1024, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=16))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1024,
    dynamic_intervals=[(1, 16), (1024 - 16, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
