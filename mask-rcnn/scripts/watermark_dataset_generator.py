import os
import shutil
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm
import json
import random
from PIL import Image

class WatermarkDatasetGenerator:
    def __init__(self, original_dir, watermarked_dir, output_dir, threshold=30):
        """
        初始化水印数据集生成器
        
        参数:
            original_dir: 原始图像目录
            watermarked_dir: 带水印图像目录
            output_dir: 输出数据集目录
            threshold: 像素差异阈值，用于检测水印
        """
        self.original_dir = original_dir
        self.watermarked_dir = watermarked_dir
        self.output_dir = output_dir
        self.threshold = threshold
        # 清空输出目录
        shutil.rmtree(self.output_dir, ignore_errors=True)
        # 创建输出目录
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
        
        # COCO 格式模板
        self.coco_template = {
            "info": {"year": 2023, "version": "1.0", "description": "Watermark Dataset"},
            "licenses": [{"id": 1, "name": "MIT License"}],
            "categories": [{"id": 1, "name": "watermark", "supercategory": "object"}]
        }
    
    def generate_mask(self, original_img, watermarked_img):
        """生成水印掩码和边界框"""
        # Always resize watermarked to match original dimensions
        watermarked_img = cv2.resize(
            watermarked_img, 
            (original_img.shape[1], original_img.shape[0]),  # (width, height)
            interpolation=cv2.INTER_AREA
        )
        
        # Convert to grayscale
        original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        watermarked_gray = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2GRAY)
        
        # 计算绝对差异
        diff = cv2.absdiff(original_gray, watermarked_gray)
        
        # 二值化
        _, binary = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作优化掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 生成掩码和边界框
        masks = []
        bboxes = []
        
        for contour in contours:
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 100:  # 过滤小区域
                continue
                
            bboxes.append([x, y, w, h])
            
            # 生成掩码
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            masks.append(mask)
        
        return masks, bboxes
    
    def convert_to_coco(self, image_id, file_name, width, height, masks, bboxes):
        """将掩码和边界框转换为 COCO 格式"""
        annotations = []
        
        for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
            # 计算面积
            area = bbox[2] * bbox[3]
            
            # Convert to RLE format
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('ascii')
            rle['size'] = [mask.shape[0], mask.shape[1]]  # Ensure correct order: [height, width]
            
            annotation = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": 1,  # watermark category
                "bbox": bbox,
                "area": area,
                "segmentation": rle,
                "iscrowd": 0
            }
            annotations.append(annotation)
        
        return annotations
    
    def generate_dataset(self, split_ratio=0.8):
        """生成训练集和验证集"""
        # 获取图像对
        image_pairs = []
        for filename in os.listdir(self.original_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                original_path = os.path.join(self.original_dir, filename)
                watermarked_path = os.path.join(self.watermarked_dir, filename)
                
                if os.path.exists(watermarked_path):
                    image_pairs.append((original_path, watermarked_path, filename))
        
        # 随机分割训练集和验证集
        random.shuffle(image_pairs)
        split_idx = int(len(image_pairs) * split_ratio)
        train_pairs = image_pairs[:split_idx]
        val_pairs = image_pairs[split_idx:]
        
        # 生成训练集
        self._generate_split_dataset(train_pairs, "train")
        # 生成验证集
        self._generate_split_dataset(val_pairs, "val")
    
    def _generate_split_dataset(self, image_pairs, split_name):
        """生成单个分割的数据集"""
        images = []
        annotations = []
        annotation_id = 1
        
        # 创建输出图像目录
        split_image_dir = os.path.join(self.output_dir, "images", split_name)
        os.makedirs(split_image_dir, exist_ok=True)
        
        # 处理每对图像
        for i, (original_path, watermarked_path, filename) in enumerate(tqdm(image_pairs, desc=f"Processing {split_name}")):
            original_img = cv2.imread(original_path)
            watermarked_img = cv2.imread(watermarked_path)
            if original_img is None or watermarked_img is None:
                print(f"读取图片失败: {filename}")
                continue

            # Resize watermarked image to match original
            watermarked_img_resized = cv2.resize(
                watermarked_img,
                (original_img.shape[1], original_img.shape[0]),
                interpolation=cv2.INTER_AREA
            )

            masks, bboxes = self.generate_mask(original_img, watermarked_img)

            # If no masks, skip
            if not masks:
                continue

            # Save the resized watermarked image
            output_image_path = os.path.join(split_image_dir, filename)
            cv2.imwrite(output_image_path, watermarked_img_resized)

            # Use the resized image's shape for annotation
            height, width = watermarked_img_resized.shape[:2]

            image_info = {
                "id": i + 1,
                "file_name": filename,
                "width": width,
                "height": height,
                "license": 1
            }
            images.append(image_info)

            img_annotations = self.convert_to_coco(i + 1, filename, width, height, masks, bboxes)

            for ann in img_annotations:
                rle_size = ann["segmentation"]["size"]
                if rle_size != [height, width]:
                    print(f"Invalid RLE size {rle_size} vs image size [{height}, {width}] in {filename}")
                    continue
                ann["id"] = annotation_id
                annotation_id += 1

            annotations.extend(img_annotations)
        
        # 创建 COCO 数据集
        coco_data = self.coco_template.copy()
        coco_data["images"] = images
        coco_data["annotations"] = annotations
        
        # 保存注释文件
        output_json_path = os.path.join(self.output_dir, "annotations", f"instances_{split_name}.json")
        with open(output_json_path, "w") as f:
            json.dump(coco_data, f)
        
        print(f"Generated {len(images)} {split_name} images with {len(annotations)} annotations")

# 使用示例
if __name__ == "__main__":
    generator = WatermarkDatasetGenerator(
        original_dir="data/train/clean",
        watermarked_dir="data/train/watermarked",
        output_dir="data/dataset/"
    )
    generator.generate_dataset(split_ratio=0.8)