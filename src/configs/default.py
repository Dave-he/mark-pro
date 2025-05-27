import yaml
from yacs.config import CfgNode as CN

# 创建配置节点
cfg = CN()

# 数据配置
cfg.data = CN()
cfg.data.watermarked_dir = ""
cfg.data.clean_dir = ""
cfg.data.image_size = [256, 256]
cfg.data.mask_threshold = 10
cfg.data.train_ratio = 0.8
cfg.data.resize_mode = "scale"
cfg.data.keep_aspect_ratio = True
cfg.data.pad_value = 0

# 训练配置
cfg.train = CN()
cfg.train.batch_size = 8
cfg.train.epochs = 100
cfg.train.lr = 5e-5
cfg.train.seg_loss_weight = 0.3
cfg.train.validation_split = 0.2
cfg.train.save_interval = 10
cfg.train.device = "cuda"
cfg.train.num_workers = 8
cfg.train.pin_memory = True
cfg.train.prefetch_factor = 2
cfg.train.persistent_workers = True
cfg.train.log_dir = "logs/"

# 模型配置
cfg.model = CN()
cfg.model.use_seg_branch = True
cfg.model.save_dir = "models/"
cfg.model.checkpoint = "seg_unetpp.pth"

# 预测配置
cfg.predict = CN()
cfg.predict.input_dir = "input/"
cfg.predict.output_dir = "output/"
cfg.predict.threshold = 0.5
cfg.predict.device = "cuda"



# 从YAML文件加载配置
cfg.merge_from_file('configs/unetpp.yaml')