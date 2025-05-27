# 添加结构化日志
import logging
import time
from pathlib import Path

def setup_logger(log_dir='logs'):
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/run_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('watermark_removal')

# 在主脚本中使用
logger = setup_logger()
logger.info(f"开始训练，配置: {cfg}")
