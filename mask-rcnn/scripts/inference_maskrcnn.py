import os
import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from tqdm import tqdm
import argparse

# 1. 推理主类
def setup_cfg(config_file, model_weights, score_thresh, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def batch_inference(cfg, input_dir, output_dir, thing_classes=None):
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    predictor = DefaultPredictor(cfg)
    # 新增自定义 metadata
    if thing_classes:
        from detectron2.data import MetadataCatalog
        custom_metadata = MetadataCatalog.get("_custom_watermark_")
        custom_metadata.thing_classes = thing_classes
    else:
        custom_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if len(cfg.DATASETS.TRAIN) > 0 else MetadataCatalog.get("_default_")
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(tuple(image_exts))]
    for img_path in tqdm(image_paths, desc="推理中"):
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取: {img_path}")
            continue
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
        base = os.path.splitext(os.path.basename(img_path))[0]
        for i, mask in enumerate(masks):
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, "masks", f"{base}_mask_{i}.png"), mask_img)
        v = Visualizer(image[:, :, ::-1], metadata=custom_metadata, scale=1.0)
        out = v.draw_instance_predictions(instances)
        cv2.imwrite(os.path.join(output_dir, "visualizations", f"{base}.jpg"), out.get_image()[:, :, ::-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="水印检测批量推理脚本")
    parser.add_argument("--config-file", required=True, help="模型配置文件路径")
    parser.add_argument("--model-weights", required=True, help="模型权重文件路径")
    parser.add_argument("--input-dir", required=True, help="待检测图片目录")
    parser.add_argument("--output-dir", required=True, help="输出结果目录")
    parser.add_argument("--score-thresh", type=float, default=0.7, help="检测阈值")
    parser.add_argument("--num-classes", type=int, default=1, help="类别数")
    parser.add_argument("--thing-classes", nargs="*", default=["watermark"], help="类别标签名")
    args = parser.parse_args()
    cfg = setup_cfg(args.config_file, args.model_weights, args.score_thresh, args.num_classes)
    batch_inference(cfg, args.input_dir, args.output_dir, args.thing_classes)

'''
python inference_maskrcnn.py \
    --config-file ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --model-weights ./models/model_final.pth \
    --input-dir /Users/hyx/Pictures/image/input100 \
    --output-dir /Users/hyx/Pictures/image/output100 \
    --score-thresh 0.7 --num-classes 1 --thing-classes watermark
'''