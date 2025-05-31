import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import subprocess
import tempfile
import argparse
# 修复导入路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_unet_mask_config
from models.unetpp.unet_plus_plus import UNetPlusPlus
from common.data.dataset_mask import get_val_transform

class WatermarkRemovalPipeline:
    """Complete watermark removal pipeline using UNet + IOPaint"""
    
    def __init__(self, model_path, device='cuda'):
        # 使用新的配置系统
        self.config = get_unet_mask_config()
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        # 修复配置键名：从 img_size 改为 image_size
        self.transform = get_val_transform(self.config['data']['image_size'][0])
        
    def _load_model(self, model_path):
        """Load UNet model for mask prediction"""
        model = UNetPlusPlus(
            in_channels=self.config['model']['input_channels'],
            num_classes=self.config['model']['num_classes'],
            deep_supervision=self.config['model']['deep_supervision']
        ).to(self.device)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights: {model_path}")
        else:
            raise FileNotFoundError(f"Model weights not found: {model_path}")
            
        model.eval()
        return model
    
    def predict_mask(self, image_path):
        """Predict watermark mask for a single image"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
            
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        augmented = self.transform(image=image_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Model inference
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output)
            
        # Process mask
        mask = prob.cpu().numpy()[0, 0]  # Remove batch and channel dimensions
        mask = cv2.resize(mask, original_size)
        
        # Binarize mask
        # 使用配置中的阈值
        binary_mask = (mask > self.config['predict']['threshold']).astype(np.uint8) * 255
        
        # 后处理
        if self.config['predict']['post_process']:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
        return binary_mask
    
    def remove_watermark_with_iopaint(self, image_path, mask, output_path):
        """Use iopaint to remove watermark based on predicted mask"""
        try:
            # Create temporary files for iopaint
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_mask:
                cv2.imwrite(temp_mask.name, mask)
                temp_mask_path = temp_mask.name
            
            # Prepare iopaint command
            cmd = [
                'iopaint', 'run',
                '--model', 'lama',  # You can change to other models like 'ldm', 'zits', 'mat'
                '--device', self.device.type,
                '--input', image_path,
                '--mask', temp_mask_path,
                '--output', output_path
            ]
            
            # Run iopaint
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Clean up temporary file
            os.unlink(temp_mask_path)
            
            return True, "Success"
            
        except subprocess.CalledProcessError as e:
            return False, f"IOPaint error: {e.stderr}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def process_single_image(self, image_path, output_dir, save_mask=True):
        """Complete pipeline for single image"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Step 1: Predict mask
            print(f"Predicting mask for {image_path}...")
            mask = self.predict_mask(image_path)
            
            # Save mask if requested
            if save_mask:
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, mask)
                print(f"Mask saved: {mask_path}")
            
            # Step 2: Remove watermark using iopaint
            output_path = os.path.join(output_dir, f"{base_name}_restored.png")
            print(f"Removing watermark using iopaint...")
            
            success, message = self.remove_watermark_with_iopaint(image_path, mask, output_path)
            
            if success:
                print(f"Watermark removed successfully: {output_path}")
                return {
                    'status': 'success',
                    'input': image_path,
                    'mask': mask_path if save_mask else None,
                    'output': output_path
                }
            else:
                print(f"Failed to remove watermark: {message}")
                return {
                    'status': 'failed',
                    'input': image_path,
                    'error': message
                }
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {
                'status': 'error',
                'input': image_path,
                'error': str(e)
            }
    
    def process_batch(self, input_path, output_dir):
        """Process multiple images"""
        # Get image paths
        image_paths = []
        if os.path.isdir(input_path):
            for filename in os.listdir(input_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(input_path, filename))
        elif os.path.isfile(input_path):
            if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(input_path)
        
        if not image_paths:
            print(f"No images found in {input_path}")
            return []
        
        print(f"Processing {len(image_paths)} images...")
        
        results = []
        for image_path in tqdm(image_paths, desc="Processing images"):
            result = self.process_single_image(image_path, output_dir)
            results.append(result)
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        print(f"\nProcessing complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        return results

# 在 main() 函数中修改参数解析部分
def main():
    # 使用新的配置系统
    config = get_unet_mask_config()
    
    parser = argparse.ArgumentParser(description='Watermark Removal Pipeline')
    # 将 required=True 改为 required=False，并提供默认值
    parser.add_argument('--input', type=str, required=False, 
                       default=config.get('predict', {}).get('input_dir', './input'), 
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default=config['predict']['output_dir'], help='Output directory')
    parser.add_argument('--model', type=str, default=config['model']['save_path'], help='Model weights path')
    parser.add_argument('--device', type=str, default=config['runtime']['device'], help='Device (cuda/cpu)')
    parser.add_argument('--save-mask', action='store_true', help='Save predicted masks')
    parser.add_argument('--iopaint-model', type=str, default=config['iopaint']['model'], 
                       choices=['lama', 'ldm', 'zits', 'mat'], help='IOPaint model')
    
    args = parser.parse_args()
    
    # 检查输入路径是否存在
    if not os.path.exists(args.input):
        print(f"错误：输入路径不存在: {args.input}")
        print("请使用 --input 参数指定有效的输入路径")
        return
    
    # Check if iopaint is installed
    try:
        subprocess.run(['iopaint', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: iopaint is not installed. Please install it with:")
        print("pip install iopaint")
        return
    
    # Initialize pipeline
    try:
        pipeline = WatermarkRemovalPipeline(args.model, args.device)
        
        # Process images
        results = pipeline.process_batch(args.input, args.output)
        
        # Save results summary
        import json
        summary_path = os.path.join(args.output, 'processing_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Processing summary saved: {summary_path}")
        
    except Exception as e:
        print(f"Pipeline initialization failed: {str(e)}")

if __name__ == '__main__':
    main()