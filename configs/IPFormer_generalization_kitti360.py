data_root = './dataset/semantickitti'
ann_file = './dataset/semantickitti/labels'
stereo_depth_root =  './dataset/semantickitti/depth'
preprocess_root = './dataset/pasco_preprocess/kitti'
camera_used = ['left']
max_epochs = 30
dataset_type = 'KITTI360DatasetGen'
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]

kitti360_class_frequencies = [
        2264087502, 20098728, 104972, 96297, 1149426, 
        4051087, 125103, 105540713, 16292249, 45297267,
        14454132, 110397082, 6766219, 295883213, 50037503,
        1561069, 406330, 30516166, 1950115,
]

# 20 classes with unlabeled
class_names = [
    'unlabeled',   # 0
    'car',         # 1
    'bicycle',     # 2
    'motorcycle',  # 3
    'truck',       # 4
    'other-vehicle',  # 5
    'person',      # 6
    'ignore',      # 7
    'ignore',      # 8
    'road',        # 9
    'parking',     # 10
    'sidewalk',    # 11
    'other-ground',# 12
    'building',    # 13
    'fence',       # 14
    'vegetation',  # 15
    'ignore',      # 16
    'terrain',     # 17
    'pole',        # 18
    'traffic-sign' # 19
]

num_class = 20
thing_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # IDs for "thing" classes (instances)
stuff_ids = [9, 10, 11, 12, 13, 14, 15, 16,17,18,19]  # IDs for "stuff" classes
remap_lut =  "False" # Mapping labels
valid_class_ids = list(range(20)) # Ensure all class IDs are valid
remap_lut = [
        0,  # 0 unlabeled
        1,  # 1 car
        2,  # 2 bicycle
        3,  # 3 motorcycle
        4,  # 4 truck
        5,  # 5 other-vehicle
        6,  # 6 person
        9,  # 7 road
        10, # 8 parking
        11, # 9 sidewalk
        12, # 10 other-ground
        13, # 11 building
        14, # 12 fence
        15, # 13 vegetation
        17, # 14 terrain
        18, # 15 pole
        19, # 16 traffic-sign
        0,  # 17 other-structure
        0   # 18 other-object
    ]

# dataset config #
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0
)

data_config={
    'input_size': (384, 1280),
    'resize': (0., 0.),
    'rot': (0.0, 0.0 ),
    'flip': (0.0, 0.0 ),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=True, color_jitter=(0.4, 0.4, 0.4)),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, lidar_root=lidar_root, dataset='kitti360', load_seg=False),
    dict(type='LoadAnnotationOccGen', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ','semantic_label', 'instance_label', 'mask_label'],
            meta_keys=['pc_range', 'occ_size','raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape', 'gt_depths']),
]
evaluator = dict( # PSCMetrics
  num_classes = 20,
  with_logits = True, # Set to False when evaluating on pkl files
  thing_classes = [1, 2, 3, 4, 5, 6, 7, 8], 
  stuff_classes = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  class_names = [
    'unlabeled',   # 0
    'car',         # 1
    'bicycle',     # 2
    'motorcycle',  # 3
    'truck',       # 4
    'other-vehicle',  # 5
    'person',      # 6
    'ignore',      # 7
    'ignore',      # 8
    'road',        # 9
    'parking',     # 10
    'sidewalk',    # 11
    'other-ground',# 12
    'building',    # 13
    'fence',       # 14
    'vegetation',  # 15
    'ignore',      # 16
    'terrain',     # 17
    'pole',        # 18
    'traffic-sign' # 19
        ], # 0 to 19
  print_out = True, # Print the per class result directly after compute().
  debug = True, # Print the PQStat state.
  overlap_flag=True,
  stuff_from_ssc=False, 
  dual_head=False, 
)
trainset_config=dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=train_pipeline,
    split='train',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=False,
    thing_ids=thing_ids,
    preprocess_root=preprocess_root
)

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=False, color_jitter=None),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, lidar_root=lidar_root, dataset='kitti360'),
    dict(type='LoadAnnotationOccGen', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ','semantic_label', 'instance_label', 'mask_label'],  
            meta_keys=['pc_range', 'occ_size', 'raw_img', 'frame_id','stereo_depth', 'focal_length', 'baseline', 'img_shape', 'gt_depths'])
]

testset_config=dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    split='test', # 'train','val', 'test' or 'all'
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    thing_ids=thing_ids,
    preprocess_root=preprocess_root
)

data = dict(
    train=trainset_config,
    val=testset_config,
    test=testset_config
)

train_dataloader_config = dict(
    batch_size=1,
    num_workers=4)

test_dataloader_config = dict(
    batch_size=1,
    num_workers=4)

# model
numC_Trans = 128
lss_downsample = [2, 2, 2]
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1
_dim_ = 128
_pos_dim_ = _dim_//2

_num_layers_self_ = 2
_num_points_self_ = 8
# Panoptic Transformer Settings
_panoptic_num_layers_ = 3  # Number of panoptic transformer layers
_panoptic_cross_attention_ = "standard"  
_panoptic_self_attention_ = "standard"  
_panoptic_query_init_ = "query_from_context_plus_random_init"  # Options: "query_random_init" or "query_from_context" or "query_from_context_plus_random_init"
dustbin_class=False
_use_aux_losses=False
num_classes= num_class

# Two stage training
two_stage_training = True
first_stage_epochs = 25
second_stage_epochs = 30
#for psc_only set ssc_only and dual_head to False
ssc_only = False
dual_head = False

freeze_ssc = False


# model config
model = dict(
    type='IPFormerDualHead',
    thing_ids=thing_ids,
    criterions=['loss_ce_inst', 'loss_mask_and_dice_inst', 'depth'], #'ce_ssc', 'geo_scal', 'sem_scal', (use when infereing ssc from psc only!)
    aux_losses=['loss_ce_inst', 'loss_mask_and_dice_inst'],
    ssc_factor=1, #ssc_losses contibution factor 
    panoptic_query_init=_panoptic_query_init_,
    num_classes=num_classes,
    dustbin_class=dustbin_class,
    instance_cross_attention=_panoptic_cross_attention_,
    use_aux_losses=_use_aux_losses, # set true for intermediate layers supervision 
    ssc_only=ssc_only,
    dual_head=dual_head,
    freeze_ssc=freeze_ssc,
    remap_lut = remap_lut,
    class_freq = kitti360_class_frequencies,
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(type='Pretrained', prefix='backbone', 
        checkpoint='./ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4], 
        out_channels=[128, 128, 128, 128, 128]),
    depth_net=dict(
        type='GeometryDepth_Net',
        downsample=8,
        numC_input=640,
        numC_Trans=numC_Trans,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_depth_weight=0.0001,
    ),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        downsample=8,
        grid_config=grid_config,
        data_config=data_config,
    ),
    proposal_layer=dict(
        type='VoxelProposalLayer',
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        input_dimensions=[128, 128, 16],
        data_config=data_config,
        init_cfg=None,
    ),
    VoxFormer_head=dict(
        type='VoxFormerHead',
        volume_h=128,
        volume_w=128,
        volume_z=16,
        data_config=data_config,
        point_cloud_range=point_cloud_range,
        embed_dims=_dim_,
        panoptic_query_init =_panoptic_query_init_,
        instance_cross_attention_type =_panoptic_cross_attention_,  # Options: "query_random_init" or "query_from_context" or "query_from_context_plus_random_init"
        cross_transformer=dict(
           type='PerceptionTransformer_DFA3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           encoder=dict(
               type='VoxFormerEncoder_DFA3D',
               num_layers=_num_layers_cross_,
               pc_range=point_cloud_range,
               data_config=data_config,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer',
                   attn_cfgs=[
                       dict(
                           type='DeformCrossAttention_DFA3D',
                           pc_range=point_cloud_range,
                           num_cams=_num_cams_,
                           deformable_attention=dict(
                               type='MSDeformableAttention3D_DFA3D',
                               embed_dims=_dim_,
                               num_points=_num_points_cross_,
                               num_levels=_num_levels_),
                           embed_dims=_dim_,
                       )
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_dim_ * 2,
                   ffn_dropout=0.1,
                   operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
        self_transformer=dict(
           type='PerceptionTransformer_DFA3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           use_level_embeds = False,
           use_cams_embeds = False,
           encoder=dict(
               type='VoxFormerEncoder',
               num_layers=_num_layers_self_,
               pc_range=point_cloud_range,
               data_config=data_config,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer',
                   attn_cfgs=[
                       dict(
                           type='DeformSelfAttention',
                           embed_dims=_dim_,
                           num_levels=1,
                           num_points=_num_points_self_)
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_dim_ * 2,
                   ffn_dropout=0.1,
                   operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=512,
            col_num_embed=512,
           ),
        mlp_prior=True
    ),
    
    occ_encoder_backbone=dict(
        type='Fuser',
        embed_dims=128,
        global_aggregator=dict(
            type='TPVGlobalAggregator',
            embed_dims=_dim_,
            split=[8,8,8],
            grid_size=[128,128,16],
            global_encoder_backbone=dict(
                type='Swin',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                in_channels=128,
                patch_size=4,
                strides=[1,2,2,2],
                frozen_stages=-1,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=[1,2,3],
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='./ckpts/swin_tiny_patch4_window7_224.pth'),
                    ),
            global_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[192, 384, 768],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(
                type='BN2d',
                requires_grad=True,
                track_running_stats=False),
                act_cfg=dict(
                type='ReLU',
                inplace=True),
                upsample_cfg=dict(
                mode='bilinear',
                align_corners=False),
            ),
        ),
        local_aggregator=dict(
            type='LocalAggregator',
            local_encoder_backbone=dict(
                type='CustomResNet3D',
                numC_input=128,
                num_layer=[2, 2, 2],
                num_channels=[128, 128, 128],
                stride=[1, 2, 2]
            ),
            local_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[128, 128, 128],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=norm_cfg,
                conv_cfg=dict(type='Conv3d'),
                act_cfg=dict(
                    type='ReLU',
                    inplace=True),
                upsample_cfg=dict(
                    mode='trilinear',
                    align_corners=False
                )
            )
        )
    ),
        pts_bbox_head=dict(
        type='OccHead',
        in_channels=[sum(voxel_out_channels)],
        out_channel=num_class,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=occ_size,
        loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
        },
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=kitti360_class_frequencies
    ),


    instance_transformer=dict(
        type='InstanceTransformer',
        panoptic_query_init=_panoptic_query_init_,
        num_classes=num_classes,
        dustbin_class=dustbin_class,
        scene_size=[128, 128, 16],
        in_channels=128,
        hidden_dim=384,
        num_queries=100,
        nheads=4,
        dec_layers=_panoptic_num_layers_,
        cross_attention_type=_panoptic_cross_attention_,
        self_attention_type=_panoptic_self_attention_
        
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='./ckpts/C2_ep28_pq_dagger=0.1445.ckpt'
        )
)

"""Training params."""
learning_rate=1e-4 # 3e-4
training_steps= 30 * 15000 #epochs*(iteration/epoch)

optimizer = dict(
    type="AdamW",
    lr=learning_rate,
    weight_decay=0.01
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=(2*4649)/training_steps, #(epochs*it/epochs) / max_epochs
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    frequency=1
)


load_from = './pretrain/pretrain_geodepth.pth'
output_dir ='./outputs/IPFormer/'
save_path = './outputs/IPFormer/'
check_val_every_n_epoch = 10
log_folder = 'logs/IPFormer/'