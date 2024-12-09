import torch
from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from omegaconf import OmegaConf
import detectron2.data.transforms as T
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
import albumentations as A

# from exp.evaluators.yolo_evalutor import YoloEvalutor
from util import CLASS_TEN_NAMES

from dataset_mapper import DatasetMapper
from coco import dataloader
# from ..common.models.retinanet import model

# from dataset_label import abp_wsi_shishi_patch_w_more_categories_nc20, abp_tct_wsi_patch_mt
import dataset_label
import dataset_unlabel
from wsi_data import build_wsi_train_dataloader, build_wsi_test_dataloader
from retinanet import model

########## model ############
# GN head
model.head.norm = 'GN'
# giou loss
model.box_reg_loss_type = 'giou'
# num classes
model.num_classes = 20
model.backbone.bottom_up.freeze_at = 2
model.test_nms_thresh = 0.3


######### optimizer #########
lr_multiplier = L(WarmupParamScheduler)(
        scheduler=L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
        #    # note that scheduler is scale-invariant. This is equivalent to
        #    # milestones=[22, 28, 30]
            milestones=[22000, 28000, 30000],
        ),
        #scheduler=L(CosineParamScheduler)(
        #    start_value=1.0, end_value=0.001
        #),
        warmup_length=10000,
        warmup_method="linear",
        warmup_factor=0.001,
        rescale_interval=True,
    )

optimizer = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    # lr=5e-3,#8gpu
    lr=5e-3, # small lr to finetune the model
    momentum=0.9,
    weight_decay=1e-4,
)

########## data #############
dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
#     dataset=L(get_detection_dataset_dicts)(names="abp_shishi_train", filter_empty=False),
    dataset=L(get_detection_dataset_dicts)(names="abp_wsi_shishi_patch_train", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(512, 544, 576, 608, 640, 672, 704, 736, 768),
                sample_style="choice",
                max_size=1536,
            ),
            L(T.RandomFlip)(horizontal=True, vertical=False),
            L(T.RandomFlip)(horizontal=False, vertical=True),
            L(T.RandomContrast)(intensity_min=0.8, intensity_max=1.5),
            L(T.RandomBrightness)(intensity_min=0.8, intensity_max=1.5),
            L(T.RandomSaturation)(intensity_min=0.8, intensity_max=1.5),
        ],
        image_format="BGR",
        albu_augmentations=[
            A.CLAHE(p=0.65),
            A.OneOf([
                A.GaussNoise(p=0.8),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.8) 
            ], p=0.65),
            A.Blur(blur_limit=[5, 9], p=0.4),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
                A.ToGray(p=0.7),
            ], p=0.65),
            A.CoarseDropout(max_holes=64, max_height=8, max_width=16, min_height=8, min_width=16, p=0.5),
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.4),
        ],
        use_instance_mask=False,
    ),
    total_batch_size=1,
    num_workers=16,
)

dataloader.train_unlabeled = L(build_detection_train_loader)(
#     dataset=L(get_detection_dataset_dicts)(names="abp_shishi_train", filter_empty=False),
    dataset=L(get_detection_dataset_dicts)(names="abp_tct_wsi_patch_mt_train", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(512, 544, 576, 608, 640, 672, 704, 736, 768),
                sample_style="choice",
                max_size=1536,
            ),
            L(T.RandomFlip)(horizontal=True, vertical=False),
            L(T.RandomFlip)(horizontal=False, vertical=True),
            L(T.RandomContrast)(intensity_min=0.8, intensity_max=1.5),
            L(T.RandomBrightness)(intensity_min=0.8, intensity_max=1.5),
            L(T.RandomSaturation)(intensity_min=0.8, intensity_max=1.5),
        ],
        image_format="BGR",
        albu_augmentations=[
            A.CLAHE(p=0.65),
            A.OneOf([
                A.GaussNoise(p=0.8),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.8) 
            ], p=0.65),
            A.Blur(blur_limit=[5, 9], p=0.4),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
                A.ToGray(p=0.7),
            ], p=0.65),
            A.CoarseDropout(max_holes=64, max_height=8, max_width=16, min_height=8, min_width=16, p=0.5),
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.4),
        ],
        use_instance_mask=False,
    ),
    total_batch_size=1,
    num_workers=16,
)

dataloader.test = L(build_detection_test_loader)(
#     dataset=L(get_detection_dataset_dicts)(names="abp_shishi_test", filter_empty=False),
    dataset=L(get_detection_dataset_dicts)(names="abp_wsi_shishi_patch_test", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=720, max_size=1280),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=16,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
# dataloader.evaluator = L(YoloEvalutor)(
#     dataset_name="${..test.dataset.names}",
#     names=CLASS_TEN_NAMES
# )


########## train ############
train = dict(
    output_dir="./output/abp_tct_wsi_patch",
    init_checkpoint="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
    max_iter=30,
    amp=dict(enabled=True),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=10000, max_to_keep=40),  # options for PeriodicCheckpointer
    eval_period=10000,
    log_period=20,
    device="cuda"
    # ...
)

