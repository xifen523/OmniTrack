# ================ base config ===================
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

total_batch_size = 16
num_gpus = 8
batch_size = total_batch_size // num_gpus
num_workers = 6   # 0 means single-process data loading
# num_iters_per_epoch = int(27947 // (num_gpus * batch_size))
num_iters_per_epoch = int(21744 // (num_gpus * batch_size))
num_epochs = 100
checkpoint_epoch_interval = 5

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
    
)
log_config = dict(
    # interval=51,
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)
load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
# stitched images(3760,480)
# extend stitched images to (4160,480)
input_shape = (4160, 480)
output_dim = 4 # x1, y1, w, h

tracking_test = True
tracking_threshold = 0.2


class_names = [
    "pedestrian",
]

num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = False  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
decouple_attn = True
with_quality_estimation = True

model = dict(
    type="JRDB2DOMNIDETR",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="CircularStatE",
        in_channels=[512, 1024, 2048],
        output_layer=[14, 17, 20],
    ),
    head=dict(
        type="OmniETRDecoder",
        nc=len(class_names),
        ch=(256, 256, 256),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        loss = dict(
            type="OmniDETRDetectionLoss",
            nc = len(class_names),
            
        ),
        post_conf = 0.1,
        classes = None,
        sampler=dict(
            type="OmniDETRBox2DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            dn_noise_scale=[1.0] * 2 + [0.5] * 2,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[1.0] * 2 + [0.5] * 2,
            cls_wise_reg_weights={class_names.index("pedestrian"): [1.0,],},
        ),
        instance_bank=dict(
            type="InstanceBackOMNIDETR",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="data/JRDB2019_2d_stitched_anno_pkls/JRDB2D_kmeans900v1_1.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        temp_group_queries = 3,
        id_noise_ratio= 0.1,
        motion_noise_scale = 0.5, 
        temp_query = True,
        is_track = True,
    ),
)

# ================== data ========================
dataset_type = "JRDB2DDetTrackDataset"
data_root = "data/JRDB2019/"
anno_root = "data/JRDB2019_2d_stitched_anno_pkls/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    # dict(type="ResizeCropFlipImageJRDB2D"),
    dict(type="ExtendStitchedImageJRDB2D"),
    dict(type="BBoxExtendJRDB2DDETR"),
    # dict(type="BBoxRotationJRDB2D"),
    # dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    # dict(
    #     type="CircleObjectRangeFilterJRDB2D",
    #     class_arce_thred=[55] * len(class_names),
    # ),
    # dict(type="InstanceNameFilterJRDB2D", classes=class_names),
    dict(type="JRDBSparse2DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            # "focal",
            "gt_bboxes_2d",
            "gt_labels_2d",
            "ori_shape",
        ],
        meta_keys=["timestamp", "instance_id"],
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ExtendStitchedImageJRDB2D"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="JRDBSparse2DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "ori_shape",
        ],
        meta_keys=["timestamp"],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

data_aug_conf = {
    # "resize_lim": (0.40, 0.47),
    "resize_lim": (1, 1),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0, 0),
    "H": 480,
    "W": 4160,
    "rand_flip": False,
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=num_workers,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "JRDB_infos_train_v1.2.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_frame_num=20,
        keep_consistent_seq_aug=True,
        with_velocity = False,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "JRDB_infos_val_v1.2.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_test,
        tracking_threshold=tracking_threshold,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "JRDB_infos_val_v1.2.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_test,
        tracking_threshold=tracking_threshold,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=6e-6,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
vis_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=["timestamp", "lidar2img"],
    ),
]
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval * 100_000_000,
    # interval=1,
    metric =["detection", "tracking"],
    pipeline=vis_pipeline,
    # out_dir="./vis",  # for visualization
)
