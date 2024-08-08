_base_ = [
    '../_base_/datasets/custom.py',
    '../_base_/models/centerpoint_pillar02_second_secfpn_custom.py',
    '../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [30264,29862, 1.7, 34040, 41990, 25.7]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-51.2, -52, -5.0, 51.2, 50.4, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
            'Bollard',
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
            'Tree'
]
# data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

dataset_type = 'CustomDataset'
data_root = 'data/custom/'
backend_args = None
# dataset_type = 'NuScenesDataset'
# data_root = 'data/nuscenes/'
# backend_args = None

db_sampler = dict(
    data_root=data_root,
    # info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    info_path=data_root + 'custom_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points = {
                'Bollard': 0,
                # 'Building': 15, #5,
                # 'Bus Stop': 5,
                'ControlBox': 5,
                # 'Ground': 5,
                'LampPost': 5,
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
    sample_groups={
        'Bollard': 1,
        # 'Building': 1,
        # 'Bus Stop': 15,
        'ControlBox': 1,
        # 'Ground': 15,
        'LampPost': 1,
        # 'Pole': 15,
        # 'Railing': 15,
        # 'Road': 15,
        # 'Shrub': 15,
        'Sign': 1,
        'SolarPanel': 1,
        'Tree': 1
    },
    points_loader=dict(
        type='LoadPointsFromLas', #'LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromLas', #'LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     use_dim=[0, 1, 2, 3, 4],
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromLas', #'LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     use_dim=[0, 1, 2, 3, 4],
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     backend_args=backend_args),
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
    #         dict(type='RandomFlip3D')
    #     ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# train_dataloader = dict(
#     _delete_=True,
#     batch_size=4,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type='CBGSDataset',
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             ann_file='nuscenes_infos_train.pkl',
#             pipeline=train_pipeline,
#             metainfo=dict(classes=class_names),
#             test_mode=False,
#             data_prefix=data_prefix,
#             use_valid_flag=True,
#             # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#             # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#             box_type_3d='LiDAR',
#             backend_args=backend_args)))


# optimizer
lr = 0.001
epoch_num = 20
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

# dataloader

# train_dataloader = dict(
#     dataset=dict(dataset=dict(pipeline=train_pipeline, metainfo=dict(classes=class_names))))
# test_dataloader = dict(
#     dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))
# val_dataloader = dict(
#     dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))

# train_cfg = dict(val_interval=20)

metainfo = dict(classes=class_names)

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
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

val_evaluator = dict(
    type='CustomMetric',
    ann_file=data_root + 'custom_infos_val.pkl',  # specify your validation pkl info
    metric='bbox')
test_evaluator = val_evaluator