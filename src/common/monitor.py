# 添加性能监控
import time

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        logger.info(f"{self.name} 耗时: {elapsed:.4f}秒")

# 在代码中使用
with Timer("数据加载"):
    train_loader, val_loader = create_dataloaders()