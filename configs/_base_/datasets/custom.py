# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/custom/'
class_names = [ 'Bollard',
                #'Building',
                #'Bus Stop',
                'ControlBox',
                #'Ground',
                'LampPost',
                # 'Pole',
                # 'Railing',
                # 'Road',
                # 'Shrub',
                'Sign',
                'SolarPanel',
                'Tree']  
point_cloud_range = [28063,31615,0, 28351, 31834, 32], #[-20,-6, 0, 20, 6, 20] #[28063,31615,0, 28351, 31834, 32] #[0, 0, 0, 656, 840, 249]  # adjust according to your dataset
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

train_pipeline = [
    dict(
        type='LoadPointsFromLas',
        coord_type='LIDAR',
        load_dim=4,  # replace with your point cloud data dimension
        use_dim=4),  # replace with the actual dimension used in training and inference
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromLas',
        coord_type='LIDAR',
        load_dim=4,  # replace with your point cloud data dimension
        use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
eval_pipeline = [
    dict(type='LoadPointsFromLas', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points']),
]
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='custom_infos_train.pkl',  # specify your training pkl info
            data_prefix=dict(pts='training/velodyne'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='custom_infos_val.pkl',  # specify your validation pkl info
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'custom_infos_val.pkl',  # specify your validation pkl info
    metric='bbox')
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='testing/velodyne'),
        ann_file='custom_infos_test.pkl',  
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
test_evaluator = val_evaluator