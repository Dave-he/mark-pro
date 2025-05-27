
from yacs.config import CfgNode as CN

cfg = CN()
    
# 模型配置
cfg.MODEL = CN()
cfg.MODEL.NAME = "UNetPlusPlus"
cfg.MODEL.INPUT_CHANNELS = 3
cfg.MODEL.NUM_CLASSES = 1
cfg.MODEL.DEEP_SUPERVISION = False

# 数据配置
cfg.DATA = CN()
cfg.DATA.ROOT_DIR = "data"
cfg.DATA.IMG_SIZE = 512
cfg.DATA.GENERATE_MASK_THRESHOLD = 30
cfg.DATA.TRAIN_RATIO = 0.8
cfg.DATA.VAL_RATIO = 0.2
cfg.DATA.SHUFFLE = True
cfg.DATA.SEED = 42

# 训练配置
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.EPOCHS = 50
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.DEVICE = "cuda"
cfg.TRAIN.OUTPUT_DIR = "./../log/output"
cfg.TRAIN.MODEL_SAVE_PATH = "./../models/unet_mask_watermark.pth"
cfg.TRAIN.LOG_INTERVAL = 10
cfg.TRAIN.SAVE_INTERVAL = 5

# 损失函数配置
cfg.LOSS = CN()
cfg.LOSS.BCE_WEIGHT = 0.5
cfg.LOSS.DICE_SMOOTH = 1e-5

# 预测配置
cfg.PREDICT = CN()
cfg.PREDICT.INPUT_PATH = "data/test/watermarked"
cfg.PREDICT.OUTPUT_DIR = "results/masks"
cfg.PREDICT.BATCH_SIZE = 8
cfg.PREDICT.THRESHOLD = 0.5
cfg.PREDICT.POST_PROCESS = True


 
# 合并配置文件
cfg.merge_from_file('configs/unetmask.yaml')