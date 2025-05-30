import logger

# 添加更健壮的错误处理
def safe_load_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"无法加载图像 {image_path}: {str(e)}")
        # 修改函数签名，接收配置参数
        def create_black_image(config):
            """创建黑色图像，接收配置参数"""
            return Image.new('RGB', (config['data']['image_size'][0], config['data']['image_size'][1]), color='black')