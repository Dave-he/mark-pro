import os
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
import cv2
import random
import matplotlib.pyplot as plt

# 1. 注册数据集
def register_dataset(dataset_name, image_dir, annotation_file):
    """注册COCO格式的数据集"""
    register_coco_instances(dataset_name, {}, annotation_file, image_dir)
    metadata = MetadataCatalog.get(dataset_name)
    return metadata

# 2. 配置模型
def setup_config(output_dir, train_dataset, val_dataset, num_classes, pretrained_model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    """配置Mask R-CNN模型参数"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(pretrained_model))
    cfg.INPUT.MASK_FORMAT = "bitmask"
    # 数据集设置
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)
    cfg.DATALOADER.NUM_WORKERS = 4  # 根据CPU核心数调整
    
    # 模型设置
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pretrained_model)
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 水印类别数
    
    # 优化器设置
    cfg.SOLVER.IMS_PER_BATCH = 2  # 根据GPU内存调整
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000    # 训练迭代次数
    cfg.SOLVER.STEPS = (3000, 4000)  # 学习率衰减点
    cfg.SOLVER.GAMMA = 0.1
    
    # 评估设置
    cfg.TEST.EVAL_PERIOD = 500    # 每500次迭代评估一次
    
    # 输出目录
    cfg.OUTPUT_DIR = output_dir
    
    return cfg

# 3. 自定义训练器（可选）
class WatermarkTrainer(DefaultTrainer):
    """自定义训练器，添加评估功能"""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# 4. 训练模型
def train_model(cfg):
    """训练Mask R-CNN模型"""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = WatermarkTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer

# 5. 评估模型
def evaluate_model(cfg, model_path=None):
    """评估模型性能"""
    if model_path:
        cfg.MODEL.WEIGHTS = model_path
    
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    return results

# 6. 可视化预测结果
def visualize_predictions(cfg, dataset_name, model_path, num_samples=5):
    """可视化模型预测结果"""
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置检测阈值
    predictor = DefaultPredictor(cfg)
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, num_samples):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(dataset_name), 
                       scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        plt.figure(figsize=(10, 8))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.show()

# 主函数
if __name__ == "__main__":
    # 数据集路径
    DATASET_DIR = "dataset"  # 修改为你的数据集路径
    OUTPUT_DIR = "models"    # 修改为输出路径
    
    # 注册数据集
    train_dataset = "watermark_train"
    val_dataset = "watermark_val"
    register_dataset(
        train_dataset, 
        os.path.join(DATASET_DIR, "images", "train"),
        os.path.join(DATASET_DIR, "annotations", "instances_train.json")
    )
    register_dataset(
        val_dataset, 
        os.path.join(DATASET_DIR, "images", "val"),
        os.path.join(DATASET_DIR, "annotations", "instances_val.json")
    )
    
    # 配置模型
    cfg = setup_config(OUTPUT_DIR, train_dataset, val_dataset, num_classes=1)
    
    # 训练模型
    trainer = train_model(cfg)
    
    # 评估模型
    model_path = os.path.join(OUTPUT_DIR, "model_final.pth")
    evaluate_model(cfg, model_path)
    
    # 可视化预测结果
    visualize_predictions(cfg, val_dataset, model_path)    