_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_custom.py',
    '../_base_/datasets/custom.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# point_cloud_range = [28063, 31615, 0, 28351, 31834, 32]  # [-20,-6, 0, 20, 6, 20] #  #[-20,-20,-20, 656, 840, 249] #[-20, -6, 0, 20, 6, 20]
point_cloud_range = [28063.999207891196, 31615.99992779688, 0.402, 28351.999207891196, 31834.89992779688, 30.702]

# dataset settings
# data_root = 'data/custom/'
data_root = '../data/orchard_road/'
class_names = ['Bollard',
                # 'Building',
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
metainfo = dict(classes=class_names)
backend_args = None

# PointPillars adopted a different sampling strategies among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'custom_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        # filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)
        filter_by_min_points = {
                'Bollard': 0,
                # 'Building': 15, #5,
                # 'Bus Stop': 5,
                'ControlBox': 5, #5,
                # 'Ground': 5,
                'LampPost': 5, #5,
                # 'Pole': 5,
                # 'Railing': 5,
                # 'Road': 5,
                # 'Shrub': 5,
                'Sign': 5,
                'SolarPanel': 5,
                'Tree': 5,
                }
        ),
    classes=class_names,
    # sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    sample_groups = {
        'Bollard': 15,
        # 'Building': 1,
        # 'Bus Stop': 15,
        'ControlBox': 15,
        # 'Ground': 15,
        'LampPost': 15,
        # 'Pole': 15,
        # 'Railing': 15,
        # 'Road': 15,
        # 'Shrub': 15,
        'Sign': 15,
        'SolarPanel': 15,
        'Tree': 15
    },
    points_loader=dict(
        type='LoadPointsFromLas',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromLas',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromLas',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1., 1.],
    #             translation_std=[0, 0, 0]),
    #         dict(type='RandomFlip3D'),
    #         dict(
    #             type='PointsRangeFilter', point_cloud_range=point_cloud_range)
    #     ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# train_dataloader = dict(
#     dataset=dict(dataset=dict(pipeline=train_pipeline, metainfo=metainfo)))
# test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
# val_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
epoch_num = 80
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=4)
val_cfg = dict()
test_cfg = dict()

train_dataloader = dict(
    batch_size=2, #6
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='CustomDataset',
            data_root=data_root,
            ann_file='custom_infos_train.pkl',  # specify your training pkl info
            data_prefix=dict(pts='training/velodyne'),
            pipeline=train_pipeline,
            modality=dict(use_lidar=True, use_camera=False),
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
        type='CustomDataset',
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='custom_infos_val.pkl',  # specify your validation pkl info
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
val_evaluator = dict(
    type='CustomMetric',
    ann_file=data_root + 'custom_infos_val.pkl',  # specify your validation pkl info
    metric='bbox')
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        data_prefix=dict(pts='testing/velodyne'),
        ann_file='custom_infos_test.pkl',  
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
test_evaluator = val_evaluator
# test_evaluator = dict(
#     type='CustomMetric',
#     ann_file=data_root + 'custom_infos_val.pkl',  # specify your validation pkl info
#     metric='bbox')
