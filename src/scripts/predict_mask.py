import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode as CN
from models.unetpp.unet_plus_plus import UNetPlusPlus
from common.data.dataset_mask import get_val_transform
from configs.unetmask import cfg

def predict_batch(model, image_paths, device, cfg):
    """批量预测水印掩码"""
    model.eval()
    
    # 创建输出目录
    os.makedirs(cfg.PREDICT.OUTPUT_DIR, exist_ok=True)
    
    # 预处理转换
    transform = get_val_transform(cfg.DATA.IMG_SIZE)
    
    # 分批次处理图像
    for i in tqdm(range(0, len(image_paths), cfg.PREDICT.BATCH_SIZE), desc="Predicting"):
        batch_paths = image_paths[i:i+cfg.PREDICT.BATCH_SIZE]
        batch_images = []
        original_sizes = []
        
        # 加载和预处理批次中的图像
        for path in batch_paths:
            image = cv2.imread(path)
            if image is None:
                print(f"警告: 无法加载图像 {path}")
                continue
                
            original_sizes.append((image.shape[0], image.shape[1]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = transform(image=image)
            batch_images.append(augmented['image'])
        
        if not batch_images:
            continue
            
        # 转换为批次张量
        batch_tensor = torch.stack(batch_images).to(device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.sigmoid(outputs)
        
        # 处理每个预测结果
        for j, (prob, path, size) in enumerate(zip(probs, batch_paths, original_sizes)):
            # 获取预测掩码
            mask = prob.cpu().numpy()[0]
            
            # 调整回原始尺寸
            mask = cv2.resize(mask, (size[1], size[0]))
            
            # 二值化
            mask = (mask > cfg.PREDICT.THRESHOLD).astype(np.uint8) * 255
            
            # 后处理（可选）
            if cfg.PREDICT.POST_PROCESS:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 保存掩码
            filename = os.path.basename(path)
            output_path = os.path.join(cfg.PREDICT.OUTPUT_DIR, filename)
            cv2.imwrite(output_path, mask)

def main():
    
    # 检查输入目录或文件列表
    input_path = cfg.PREDICT.INPUT_PATH if hasattr(cfg.PREDICT, 'INPUT_PATH') else None
    if not input_path:
        print("请在配置文件中设置PREDICT.INPUT_PATH为输入图像目录或文件列表")
        return
    
    # 获取所有输入图像路径
    image_paths = []
    if os.path.isdir(input_path):
        # 处理目录中的所有图像
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(input_path, filename))
    elif os.path.isfile(input_path):
        # 处理单个图像文件
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(input_path)
        else:
            # 假设是文件列表
            with open(input_path, 'r') as f:
                image_paths = [line.strip() for line in f if line.strip()]
    else:
        print(f"错误: 输入路径不存在 {input_path}")
        return
    
    if not image_paths:
        print(f"错误: 在 {input_path} 中未找到图像")
        return
    
    # 初始化模型
    device = torch.device(cfg.DEVICE)
    model = UNetPlusPlus(
        in_channels=cfg.MODEL.INPUT_CHANNELS,
        num_classes=cfg.MODEL.NUM_CLASSES,
        deep_supervision=cfg.MODEL.DEEP_SUPERVISION
    ).to(device)
    
    # 加载模型权重
    if os.path.exists(cfg.TRAIN.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(cfg.TRAIN.MODEL_SAVE_PATH, map_location=device))
        print(f"加载模型权重: {cfg.TRAIN.MODEL_SAVE_PATH}")
    else:
        print(f"错误: 模型权重文件不存在 {cfg.TRAIN.MODEL_SAVE_PATH}")
        return
    
    # 批量预测
    predict_batch(model, image_paths, device, cfg)
    print(f"预测完成! 掩码已保存到 {cfg.PREDICT.OUTPUT_DIR}")

if __name__ == "__main__":
    main()