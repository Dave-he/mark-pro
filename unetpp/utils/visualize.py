import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def overlay_mask(image, mask, color=(255, 0, 0, 64), alpha=0.5):
    """在原图上叠加半透明掩码"""
    image = image.copy()
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
    
    # 创建彩色掩码
    color_mask = Image.new('RGBA', image.size, color)
    
    # 应用掩码
    image.paste(color_mask, mask=mask)
    return image

def visualize_results(input_img, mask, output_img, save_path=None):
    """可视化输入图像、预测掩码和输出图像"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 输入图像
    axes[0].imshow(input_img)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    # 掩码
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # 输出图像
    axes[2].imshow(output_img)
    axes[2].set_title('Restored')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()