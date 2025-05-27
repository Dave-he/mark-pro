import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.unetpp.seg_unetpp import SegGuidedUnetPP
from common.utils.visualize import overlay_mask, visualize_results
from configs.default import cfg

def predict_single_image(model, image_path, output_dir, visualize=True):
    # 加载图像并保存原始尺寸
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    w, h = original_size
    
    # 根据配置选择预处理方式
    if cfg.data.resize_mode == 'fixed':
        # 直接调整到固定尺寸
        input_tensor = transforms.Compose([
            transforms.Resize(cfg.data.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])(image).unsqueeze(0).to(cfg.device)
        
        padding = None
        
    elif cfg.data.resize_mode == 'scale':
        # 等比缩放
        max_size = max(cfg.data.image_size)
        image_scaled = transforms.Resize(max_size)(image)
        
        input_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])(image_scaled).unsqueeze(0).to(cfg.device)
        
        padding = None
        
    elif cfg.data.resize_mode == 'pad':
        # 填充到固定尺寸
        target_w, target_h = cfg.data.image_size
        
        # 计算填充量
        pad_w = max(0, target_w - w)
        pad_h = max(0, target_h - h)
        padding = (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2)
        
        image_padded = transforms.Pad(padding, fill=cfg.data.pad_value)(image)
        
        input_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])(image_padded).unsqueeze(0).to(cfg.device)
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        restored_img, seg_mask = model(input_tensor)
    
    # 后处理
    restored_img = restored_img.squeeze().cpu()
    restored_img = restored_img.permute(1, 2, 0).numpy()
    restored_img = (restored_img * 0.5 + 0.5) * 255  # 反归一化
    restored_img = Image.fromarray(restored_img.astype(np.uint8))
    
    seg_mask = seg_mask.squeeze().cpu().numpy()
    seg_mask = (seg_mask > cfg.predict.threshold).astype(np.uint8) * 255
    seg_mask = Image.fromarray(seg_mask).convert('L')
    
    # 根据预处理方式恢复原始尺寸
    if cfg.data.resize_mode == 'fixed':
        # 从固定尺寸恢复
        restored_img = restored_img.resize(original_size, Image.BICUBIC)
        seg_mask = seg_mask.resize(original_size, Image.NEAREST)
        
    elif cfg.data.resize_mode == 'scale':
        # 从等比缩放恢复
        restored_img = restored_img.resize(original_size, Image.BICUBIC)
        seg_mask = seg_mask.resize(original_size, Image.NEAREST)
        
    elif cfg.data.resize_mode == 'pad':
        # 从填充恢复（裁剪回原始尺寸）
        restored_img = restored_img.crop((padding[0], padding[1], 
                                         padding[0] + w, padding[1] + h))
        seg_mask = seg_mask.crop((padding[0], padding[1], 
                                 padding[0] + w, padding[1] + h))
    
    # 保存结果
    filename = os.path.basename(image_path)
    os.makedirs(os.path.join(output_dir, 'restored'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)
    
    restored_img_path = os.path.join(output_dir, 'restored', filename)
    mask_path = os.path.join(output_dir, 'masks', filename)
    overlay_path = os.path.join(output_dir, 'overlays', filename)
    
    restored_img.save(restored_img_path)
    seg_mask.save(mask_path)
    
    # 创建并保存叠加图
    overlay = overlay_mask(image, seg_mask)
    overlay.save(overlay_path)
    
    if visualize:
        visualize_results(image, seg_mask, restored_img)
    
    return restored_img, seg_mask

def batch_predict():
    # 加载模型
    model = SegGuidedUnetPP().to(cfg.predict.device)
    # 生成模型名（与train.py一致）
    imgsize = f"{cfg.data.image_size[0]}x{cfg.data.image_size[1]}"
    segw = cfg.train.seg_loss_weight
    model_name = f"unetpp_img{imgsize}_sw{segw}.pth"
    model_path = os.path.join(cfg.model.save_dir, model_name)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    
    # 创建输出目录
    os.makedirs(cfg.predict.output_dir, exist_ok=True)
    
    # 获取所有输入图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(cfg.predict.input_dir) 
                  if os.path.isfile(os.path.join(cfg.predict.input_dir, f)) 
                  and os.path.splitext(f)[1].lower() in image_extensions]
    
    # 批量预测
    for i, filename in enumerate(image_files):
        image_path = os.path.join(cfg.predict.input_dir, filename)
        print(f"Processing {i+1}/{len(image_files)}: {filename}")
        
        try:
            predict_single_image(model, image_path, cfg.predict.output_dir, visualize=False)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"All images processed. Results saved to {cfg.predict.output_dir}")

if __name__ == "__main__":
    batch_predict()