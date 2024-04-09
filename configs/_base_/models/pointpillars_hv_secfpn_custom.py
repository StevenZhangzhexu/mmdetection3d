voxel_size = [0.3, 0.3, 32]  # adjust according to your dataset
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range= [28063,31615,0, 28351, 31855, 32], #[-20,-6, 0, 20, 6, 20], #[-20,-20,-20, 656, 840, 249],
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range= [28063,31615,0, 28351, 31834, 32], #[-20,-6, 0, 20, 6, 20] #[-20,-20,-20, 656, 840, 249]
        ), 
    # the `output_shape` should be adjusted according to `point_cloud_range`
    # and `voxel_size`
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, 
        output_shape=[800, 960] #[730, 960]
        ), 
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=6,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        # adjust the `ranges` and `sizes` according to your dataset
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                # [-20,-6, 0, 20, 6, 20],
                # [-20,-6, 0, 20, 6, 20],
                # [-20,-6, 0, 20, 6, 20],
                # [-20,-6, 0, 20, 6, 20],
                # [-20,-6, 0, 20, 6, 20],
                # [-20,-6, 0, 20, 6, 20]
                [28063,31615, 0, 28351, 31834, 32],
                [28063,31615, 0, 28351, 31834, 32],
                [28063,31615, 0, 28351, 31834, 32],
                [28063,31615, 0, 28351, 31834, 32],
                [28063,31615, 0, 28351, 31834, 32],
                [28063,31615, 0, 28351, 31834, 32]
            ],
            sizes=[[0.16, 0.16, 0.7], 
                   [1.2, 0.5, 4.7],
                   [2.3, 0.3, 9.7],
                   [1.3 ,0.4, 3.4],
                   [1, 0.7, 3.14],
                   [1.6, 0.2, 9.7]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            # for Bollard
            # dict(
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.5,
            #     neg_iou_thr=0.35,
            #     min_pos_iou=0.35,
            #     ignore_iof_thr=-1
            # ),
            # for Building
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1
            ),
            # for Bus Stop
            # dict(
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.5,
            #     neg_iou_thr=0.35,
            #     min_pos_iou=0.35,
            #     ignore_iof_thr=-1
            # ),
            # for Control Box
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1
            ),
            # for Ground
            # dict(
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.5,
            #     neg_iou_thr=0.35,
            #     min_pos_iou=0.35,
            #     ignore_iof_thr=-1
            # ),
            # for Lamp Post
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1
            ),
            # for Pole
            # dict(
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.5,
            #     neg_iou_thr=0.35,
            #     min_pos_iou=0.35,
            #     ignore_iof_thr=-1
            # ),
            # # for Railing
            # dict(
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.5,
            #     neg_iou_thr=0.35,
            #     min_pos_iou=0.35,
            #     ignore_iof_thr=-1
            # ),
            # # for Road
            # dict(
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.5,
            #     neg_iou_thr=0.35,
            #     min_pos_iou=0.35,
            #     ignore_iof_thr=-1
            # ),
            # # for Shrub
            # dict(
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.5,
            #     neg_iou_thr=0.35,
            #     min_pos_iou=0.35,
            #     ignore_iof_thr=-1
            # ),
            # for Sign
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1
            ),
            # for Solar Panel
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1
            ),
            # for Tree
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1
            )
            # # for dontcare
            # dict(
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.5,
            #     neg_iou_thr=0.35,
            #     min_pos_iou=0.35,
            #     ignore_iof_thr=-1
            # )
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))


    