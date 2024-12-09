from fvcore.common.file_io import PathManager
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

# from datasets.our_datasets import WSI_PATCH, WSI_SHISHI_MIX_RAW, WSI_SHISHI_MIX_V2_RAW
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000358703.jpg ASC-US
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000224100.jpg LSIL
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000043040.jpg ASC-H
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000043040.jpg ASC-H
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000043040.jpg HSIL
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000043040.jpg HSIL
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000150026.jpg AGC
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000150026.jpg AGC
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000003647.jpg SCC
# /nasdata/dataset/wsi_shishi_mix_data_v2/multi_label_v3/yolo_images/shishi/images/ss_000003647.jpg SCC



CLASS_NAMES= [
    "ASCUS_S",
    "LSIL_S",
    "ASCH_S",
    "HSIL_S",
    "ASCUS_M",
    "LSIL_M",
    "ASCH_M",
    "HSIL_M",
    "TRI",
    "AGC",
    "EC",
    "FUNGI",
    "CC",
    "ACTINO",
    "HSV",
    "MP_RC",
    "ECC",
    "SCC",
    "AGC_NOS",
    "AGC_FN"
]

def get_tct_dicts(anno_file):
    dicts = []
    idx = 0
    for l in PathManager.open(anno_file).readlines():
        data = l.split()
        r = {
                "file_name": data[0],
                "image_id": idx,
                "height": 2160,
                "width": 3840,
                "image_label": [0] * len(CLASS_NAMES)
        }

        instances = []
        for obj in data[1:]:
            ymin, xmin, ymax, xmax, label_id = [int(v) for v in obj.split(',')]
            label = label_id
            if label == -1:
                continue
            r['image_label'][label] = 1

            bbox = [xmin, ymin, xmax, ymax]
            instances.append(
                    {"category_id": label, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )

        r["annotations"] = instances
        dicts.append(r)
        idx += 1

    return dicts


def register_tct(name, anno_file):
    DatasetCatalog.register(name, lambda: get_tct_dicts(anno_file))
    MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES)
    MetadataCatalog.get(name).evaluator_type = "coco"


def register_all_tct(name, train_anno_file, test_anno_file):
    SPLITS = [
        (name+"_train", train_anno_file),
        (name+"_test", test_anno_file),
    ]
    for n, a in SPLITS:
        register_tct(n, a)

# train_txt = WSI_SHISHI_MIX_V2_RAW.at(WSI_SHISHI_MIX_V2_RAW.default_train_file_path)
# test_txt = WSI_SHISHI_MIX_V2_RAW.at(WSI_SHISHI_MIX_V2_RAW.default_test_file_path)

# train_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_11/train.txt"
# test_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_11/test.txt"
# test_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_anno_10_remove_he/test.txt"
# test_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_11/test_tct.txt"

# train_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_12/train_shuffle.txt"
# test_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_12/test_tct.txt"

# mix 14
# train_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_14/train_shuffle.txt"
# test_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_14/test_tct.txt"

# 
# train_txt = "/nasdata/private/zwlu/Now/ai_trainer/my_exps/竞赛数据集准备/标注评估/train_files/others_7.txt"
# test_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_14/test_tct.txt"

# train_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_15/train_shuffle.txt"
# test_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/mix_annos_15/test_tct.txt"

# train_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/finetune_datasets/OOD/moshi_train.txt"
# test_txt = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pseudo_label/annos/finetune_datasets/noise/ID/wsi_all_noise_nature.txt"

# testset
train_txt = "/nasdata/dataset/wsi_shishi_mix_data_v2/mix_datasets/mix_v2/detectron2/nc20/train.txt"
test_txt = "/nasdata/dataset/wsi_shishi_mix_data_v2/mix_datasets/mix_v2/detectron2/nc20/val.txt"
register_all_tct('abp_wsi_shishi_patch', train_txt, test_txt)
