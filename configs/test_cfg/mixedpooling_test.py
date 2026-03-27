_base_ = [
    "mmdet3d::_base_/default_runtime.py",
]
custom_imports = dict(imports=["oneformer3d"])

# model settings
num_channels = 64
num_instance_classes = 36
num_semantic_classes = 36

model = dict(
    type="FloorplanOneFormer3D_PTv3_Onlycoord_Metric",
    data_preprocessor=dict(type="Det3DDataPreprocessor"),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.168,
    grid_size=0.168,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    pooling=dict(type="MixedPooling", compose=(dict(type="SegmentMeanPooling"), dict(type="SegmentMaxPooling"))),
    decoder=dict(
        type="QueryDecoder",
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=220,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="gelu",
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True,
    ),
    criterion=dict(
        type="S3DISUnifiedCriterion",
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type="FloorplanSemanticCriterion",
            loss_weight=5,
            seg_loss=dict(
                type="mmdet.CrossEntropyLoss",
                use_sigmoid=True,
                class_weight=[1.0] * 35 + [0.1],
            ),
        ),
        inst_criterion=dict(
            type="FloorplanInstanceCriterion",
            matcher=dict(
                type="HungarianMatcher",
                costs=[
                    dict(type="QueryClassificationCost", weight=0.5),
                    dict(type="MaskBCECost", weight=1.0),
                    dict(type="MaskDiceCost", weight=1.0),
                ],
            ),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True,
            class_weight=[1] * 35 + [0.1, 0.05],
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=220,
        inst_score_thr=0.0,
        pan_score_thr=0.2,
        npoint_thr=1,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel="gaussian",
        num_sem_cls=num_semantic_classes,
        thing_cls=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
        ],
        stuff_cls=[30, 31, 32, 33, 34, 35],
    ),
)

# dataset settings
dataset_type = "S3DFloorplanSegDataset_"
data_root = "data/newfloorplan/"
data_prefix = dict(
    pts="points",
    pts_instance_mask="instance_mask",
    pts_semantic_mask="semantic_mask",
    pts_reverse_map="reverse_map",
)

train_area = [1]
test_area = [2]

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
    ),
    dict(
        type="LoadFloorplanAnnotations3D",
        with_label_3d=False,
        with_bbox_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=False,
        with_reverse_map=True,
    ),
    dict(type="FloorplanPointSample_PTv3", num_points=180000),
    dict(type="PointInstClassMapping_", num_classes=num_instance_classes),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0,
        flip_box3d=False,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[0.0, 0.0],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0],
        shift_height=False,
    ),
    dict(type="NormalizePointsColor_", color_mean=[127.5, 127.5, 127.5]),
    # dict(type="NormalizePointsCoord"),
    dict(
        type="Pack3DDetInputs_",
        keys=[
            "points",
            "gt_labels_3d",
            "pts_semantic_mask",
            "pts_instance_mask",
            "pts_reverse_map",
        ],
    ),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
    ),
    dict(
        type="LoadFloorplanAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=False,
        with_reverse_map=True,
    ),
    # dict(
    #     type="MultiScaleFlipAug3D",
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(type="NormalizePointsColor_", color_mean=[127.5, 127.5, 127.5])
    #     ],
    # ),
    dict(type="NormalizePointsColor_", color_mean=[127.5, 127.5, 127.5]),
    dict(type="Pack3DDetInputs_", keys=["points", "pts_reverse_map"]),
]

# run settings
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="ConcatDataset",
        datasets=(
            [
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=f"s3dfloorplan_infos_Area_{i}.pkl",
                    pipeline=train_pipeline,
                    filter_empty_gt=True,
                    data_prefix=data_prefix,
                    box_type_3d="Depth",
                    backend_args=None,
                )
                for i in train_area
            ]
        ),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    # dataset=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     ann_file=f"s3dfloorplan_infos_Area_{test_area}.pkl",
    #     pipeline=test_pipeline,
    #     test_mode=True,
    #     data_prefix=data_prefix,
    #     box_type_3d="Depth",
    #     backend_args=None,
    # ),
    dataset=dict(
        type="ConcatDataset",
        datasets=(
            [
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=f"s3dfloorplan_infos_Area_{i}.pkl",
                    pipeline=test_pipeline,
                    test_mode=True,
                    data_prefix=data_prefix,
                    box_type_3d="Depth",
                    backend_args=None,
                )
                for i in test_area
            ]
        ),
    ),
)
test_dataloader = val_dataloader
find_unused_parameters = True
class_names = [
    "single door",
    "double door",
    "sliding door",
    "folding door",
    "revolving door",
    "rolling door",
    "window",
    "bay window",
    "blind window",
    "opening symbol",
    "sofa",
    "bed",
    "chair",
    "table",
    "TV cabinet",
    "Wardrobe",
    "cabinet",
    "gas stove",
    "sink",
    "refrigerator",
    "airconditioner",
    "bath",
    "bath tub",
    "washing machine",
    "squat toilet",
    "urinal",
    "toilet",
    "stairs",
    "elevator",
    "escalator",
    "row chairs",
    "parking spot",
    "wall",
    "curtain wall",
    "railing",
    "bg",
    "unlabeled",
]
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names,
    dataset_name="Floorplan",
)
sem_mapping = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
]

val_evaluator = dict(
    type="UnifiedSegMetric_Save_Primitive",
    stuff_class_inds=[30, 31, 32, 33, 34, 35],
    thing_class_inds=[
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
    ],
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=sem_mapping,
    submission_prefix_semantic=None,
    submission_prefix_instance=None,
    metric_meta=metric_meta,
    ignore_bg=True,
    ifeval=True,
    saved_dir="./work_dirs/aaai_nocolor_1024_test_mixedpooling/pred",
    npz_path="data/floorplan_color/test_color/npz_gt",
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2),
)
param_scheduler = dict(type="PolyLR", begin=0, end=1024, power=0.9)

custom_hooks = [dict(type="EmptyCacheHook", after_iter=True)]
default_hooks = dict(checkpoint=dict(interval=8, max_keep_ckpts=1024, save_best=["pq"], rule="greater"))

load_from = None

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1024, val_interval=8)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
