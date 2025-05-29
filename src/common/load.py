import logger

# 添加更健壮的错误处理
def safe_load_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"无法加载图像 {image_path}: {str(e)}")
        # 返回一个空白图像作为替代
        return Image.new('RGB', (cfg.data.image_size[0], cfg.data.image_size[1]), color='black')