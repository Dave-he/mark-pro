import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入模型
from models.seg_unet import SegEnhancedUNet

def predict_single_image(model, image_path, output_dir, image_size=256, mask_threshold=0.5, device='cuda'):
    """预测单张图像并保存结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # 预处理
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        restoration, seg_logits = model(input_tensor)
    
    # 处理去水印结果
    restored_img = restoration.squeeze().cpu().detach()
    restored_img = (restored_img * 0.5 + 0.5) * 255  # 反归一化
    restored_img = restored_img.permute(1, 2, 0).numpy().astype(np.uint8)
    restored_img = Image.fromarray(restored_img).resize(original_size)
    
    # 处理分割掩码
    seg_probs = torch.softmax(seg_logits, dim=1)[:, 1, :, :]  # 获取水印类别的概率
    mask = (seg_probs >= mask_threshold).float().squeeze().cpu().numpy() * 255
    mask = Image.fromarray(mask.astype(np.uint8)).resize(original_size)
    
    # 保存结果
    base_name = os.path.basename(image_path).split('.')[0]
    restored_path = os.path.join(output_dir, f'{base_name}_restored.png')
    mask_path = os.path.join(output_dir, f'{base_name}_mask.png')
    
    restored_img.save(restored_path)
    mask.save(mask_path)
    
    return restored_path, mask_path

def predict_folder(model, input_folder, output_folder, image_size=256, mask_threshold=0.5, device='cuda'):
    """批量预测文件夹中的所有图像"""
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_folder) 
                  if os.path.isfile(os.path.join(input_folder, f)) 
                  and os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"发现 {len(image_files)} 张图像")
    
    # 为每张图像执行预测
    results = []
    for image_file in tqdm(image_files, desc="预测进度"):
        image_path = os.path.join(input_folder, image_file)
        result = predict_single_image(
            model, image_path, output_folder, 
            image_size=image_size, 
            mask_threshold=mask_threshold,
            device=device
        )
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Watermark Removal Prediction')
    parser.add_argument('--model-path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像或文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--image-size', type=int, default=256, help='输入图像尺寸')
    parser.add_argument('--mask-threshold', type=float, default=0.5, help='掩码二值化阈值')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='使用的设备')
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误：输入路径 {args.input} 不存在")
        return
    
    # 初始化模型
    device = torch.device(args.device)
    model = SegEnhancedUNet(in_channels=3, out_channels=3, seg_classes=2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 执行预测
    if os.path.isfile(args.input):
        predict_single_image(
            model, args.input, args.output, 
            image_size=args.image_size, 
            mask_threshold=args.mask_threshold,
            device=device
        )
        print(f"预测完成，结果已保存到 {args.output}")
    else:
        predict_folder(
            model, args.input, args.output, 
            image_size=args.image_size, 
            mask_threshold=args.mask_threshold,
            device=device
        )
        print(f"批量预测完成，所有结果已保存到 {args.output}")

if __name__ == '__main__':
    main()