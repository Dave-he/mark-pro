# 图片去水印



          
# 项目目录结构

```
mark-pro/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── run.sh                      # 项目运行脚本
├── .gitignore                  # Git忽略文件配置
│
├── data/                       # 数据集目录
│   ├── train/                  # 训练数据
│   │   ├── watermarked/        # 带水印图像
│   │   └── clean/              # 无水印图像
│   ├── val/                    # 验证数据
│   │   ├── watermarked/        # 验证集带水印图像
│   │   └── clean/              # 验证集无水印图像
│   └── test/                   # 测试数据
│       └── watermarked/        # 测试用带水印图像
│
├── models/                     # 训练好的模型权重
│   ├── unet_mask_watermark.pth # UNet掩码检测模型
│   └── checkpoints/            # 训练检查点
│
├── results/                    # 输出结果目录
│   ├── masks/                  # 生成的掩码图像
│   ├── restored/               # 去水印后的图像
│   └── logs/                   # 训练日志
│
├── src/                        # 源代码目录
│   ├── __init__.py
│   ├── main.py                 # 主程序入口
│   ├── run.py                  # 运行脚本
│   │
│   ├── common/                 # 公共模块
│   │   ├── __init__.py
│   │   ├── load.py             # 模型加载工具
│   │   ├── loggger.py          # 日志工具
│   │   ├── monitor.py          # 训练监控
│   │   ├── data/               # 数据处理模块
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py      # 通用数据集类
│   │   │   ├── dataset_mask.py # 掩码数据集类
│   │   │   └── transforms.py   # 数据变换
│   │   └── utils/              # 工具函数
│   │       ├── __init__.py
│   │       ├── losses.py       # 损失函数
│   │       ├── metrics.py      # 评估指标
│   │       └── visualize.py    # 可视化工具
│   │
│   ├── configs/                # 配置文件
│   │   ├── base.yaml           # 基础配置
│   │   ├── default.py          # 默认配置
│   │   ├── unetmask.py         # UNet掩码配置
│   │   ├── unetmask.yaml       # UNet掩码YAML配置
│   │   ├── unetpp.yaml         # UNet++配置
│   │   ├── rmvl.yaml           # 去水印配置
│   │   └── mask_rcnn.yaml      # Mask R-CNN配置
│   │
│   ├── models/                 # 模型定义
│   │   ├── unetpp/             # UNet++模型
│   │   │   ├── unet_plus_plus.py    # UNet++核心实现
│   │   │   ├── unetpp.py            # UNet++变体
│   │   │   └── seg_unetpp.py        # 分割引导UNet++
│   │   ├── rmvl/               # 去水印模型
│   │   │   ├── unet.py         # 基础UNet
│   │   │   └── seg_unet.py     # 分割增强UNet
│   │   └── mask_rcnn/          # Mask R-CNN模型
│   │
│   ├── scripts/                # 脚本文件
│   │   ├── train.py            # 通用训练脚本
│   │   ├── train_mask.py       # 掩码训练脚本
│   │   ├── train_rmvl.py       # 去水印训练脚本
│   │   ├── predict.py          # 通用预测脚本
│   │   ├── predict_mask.py     # 掩码预测脚本（优化版）
│   │   └── predict_rmvl.py     # 去水印预测脚本
│   │
│   └── other/                  # 其他工具
│       ├── check.py            # 数据集检查工具
│       ├── generate.py         # 数据生成工具
│       ├── down_data.py        # 数据下载工具
│       └── zip_current_dir.py  # 项目打包工具
│
└── mask-rcnn/                  # Mask R-CNN专用目录
    ├── README.md               # Mask R-CNN说明
    ├── configs/                # Detectron2配置
    │   ├── Base-RCNN-FPN.yaml
    │   ├── default.yaml
    │   └── mask_rcnn_R_50_FPN_3x.yaml
    └── scripts/                # Mask R-CNN脚本
        ├── train_maskrcnn.py   # 训练脚本
        ├── inference_maskrcnn.py # 推理脚本
        ├── remwm.py            # 水印移除
        └── watermark_dataset_generator.py # 数据集生成
```

## 目录功能说明

### 核心模块

1. **src/scripts/predict_mask.py** - 主要的水印去除流水线
   - 使用UNet++进行水印区域检测
   - 生成二值化掩码
   - 集成iopaint进行图像修复
   - 支持批量处理和单张图像处理

2. **src/models/** - 深度学习模型定义
   - `unetpp/`: UNet++系列模型，用于精确的水印检测
   - `rmvl/`: 专门的去水印模型
   - `mask_rcnn/`: 基于Detectron2的实例分割模型

3. **src/common/data/** - 数据处理模块
   - `dataset_mask.py`: 专门处理水印掩码数据集
   - `dataset.py`: 通用数据集处理
   - `transforms.py`: 数据增强和预处理

### 配置管理

4. **src/configs/** - 配置文件管理
   - 使用YACS进行配置管理
   - 支持YAML和Python配置文件
   - 模块化配置，便于实验管理

### 工具脚本

5. **src/other/** - 辅助工具
   - `check.py`: 数据集完整性检查
   - `generate.py`: 训练/验证集划分
   - 数据下载和项目打包工具

### 数据组织

6. **data/** - 数据集目录结构
   - 按训练/验证/测试划分
   - 水印图像和干净图像配对存储
   - 支持自动掩码生成

## 使用流程

### 1. 数据准备
```bash
# 检查数据集完整性
python src/other/check.py

# 生成训练/验证集划分
python src/other/generate.py
```

### 2. 模型训练
```bash
# 训练UNet掩码检测模型
python src/scripts/train_mask.py

# 训练去水印模型
python src/scripts/train_rmvl.py
```

### 3. 水印去除
```bash
# 单张图像处理
python src/scripts/predict_mask.py --input image.jpg --output results/ --save-mask

# 批量处理
python src/scripts/predict_mask.py --input data/test/ --output results/ --iopaint-model lama
```

## 技术特点

- **模块化设计**: 清晰的代码组织结构，便于维护和扩展
- **配置驱动**: 使用YACS进行统一配置管理
- **多模型支持**: 支持UNet++、Mask R-CNN等多种模型
- **完整流水线**: 从数据预处理到模型训练再到推理的完整流程
- **工具齐全**: 包含数据检查、可视化、评估等辅助工具
- **生产就绪**: 支持批量处理、错误处理、进度跟踪等生产环境需求

这个项目结构为水印检测和去除提供了一个完整、可扩展的解决方案。
        
## 安装
```bash
#本代码python 3.9-3.13都可以, （iopaint最大支持到3.11，最好用3.11以下版本）
virtualenv venv -p python3.10
source venv/bin/activate
# ./venv/Scripts/activate.ps1 #windows

#conda
conda create -n env10 python=3.10
conda activate env10


pip install torch torchvision transformers huggingface_hub requests
pip install opencv-python
pip install pillow numpy matplotlib tqdm pyyaml 
pip install yacs
pip install tensorboard
pip install albumentations

## 如果是3.11以下版本 可以安装iopaint
pip install iopaint
pip install -r requirements.txt

```

## 运行
```bash
python main.py
```

## 数据集

- 训练所需
在data/train/ 目录下创建两个子目录：
  watermarked/  存放带水印的图像
  clean/：存放对应的无水印图像
  图像应按文件名匹配（例如：watermarked/1.jpg对应clean/1.jpg）

- 预测所需
  data/inputs/   目录下存放待修复的图像。
  data/outputs/ 目录将用于保存修复后的图像、水印掩码和叠加图。

- 目录结构
```bash
data/
├── train/                  # 训练图片目录，0.8作为训练集，0.2作为验证集
│   ├── watermarked/        # 带水印图像
│   └── clean/              # 无水印图像
├── inputs/                 # 待修复的图像
└── outputs/                # 修复后的图像、水印掩码和叠加图
    ├── restored/           # 修复后的图像  
    ├── masks/              # 水印掩码
    └── overlays/           # 叠加图
```

# 代码说明

## 目录结构

```bash
mark-pro/
├── README.md
├── src/                    # RMVL模型
  ├── common/                  # 所有模型共享的通用组件
  │   ├── __init__.py
  │   ├── data/                # 数据处理相关
  │   │   ├── __init__.py
  │   │   ├── dataset.py       # 基础数据集类
  │   │   └── transforms.py    # 数据变换
  │   ├── utils/               # 通用工具
  │   │   ├── __init__.py
  │   │   ├── metrics.py       # 评估指标
  │   │   ├── losses.py        # 损失函数
  │   │   └── visualize.py     # 可视化工具
  │   └── config/              # 配置管理
  │       ├── __init__.py
  │       └── base_config.py   # 基础配置类
  ├── models/                  # 各种模型实现
  │   ├── __init__.py
  │   ├── unetpp/              # UNet++模型
  │   ├── mask_rcnn/           # Mask-RCNN模型
  │   └── rmvl/                # RMVL模型
  ├── scripts/                 # 运行脚本
  │   ├── train.py             # 统一训练入口
  │   └── predict.py           # 统一预测入口
  └── configs/                 # 配置文件
      ├── unetpp.yaml
      ├── mask_rcnn.yaml
      └── rmvl.yaml
``` 

2. 数据集检测 
```bash
# 检测模式（默认）
python check.py

# 删除模式
python check.py --delete

# 指定自定义目录
python check.py --base-dir /path/to/data --delete

# 指定自定义子目录名
python check.py --watermarked-dir images_with_wm --clean-dir images_clean
```


3. 模型训练
```bash
cd src
python main.py
```

1. 模型预测
```bash
cd src
python run.py



# 如果有iopaint,可以基于这个擦除
cd ..

iopaint run --model=runwayml/stable-diffusion-inpainting \
  --device=cuda --image=data/input --mask=data/output/masks \
  --output=data/out1 --model-dir=~/.cache

# iopaint模型预下载地址,可解压放到.cache目录下
https://s1-12864.ap4r.com/oversea-game/ai-model/model.7z
```





## 使用
```bash

# Process single image
python src/scripts/predict_mask.py --input image.jpg --output results/ --save-mask

# Process directory
python src/scripts/predict_mask.py --input data/test/ --output results/ --iopaint-model lama

# Use specific model and device
python src/scripts/predict_mask.py --input data/test/ --output results/ --model models/best_model.pth --device cpu

```

```bash
from scripts.predict_mask import WatermarkRemovalPipeline

# Initialize pipeline
pipeline = WatermarkRemovalPipeline('models/unet_mask_watermark.pth')

# Process single image
result = pipeline.process_single_image('input.jpg', 'output/')

# Process batch
results = pipeline.process_batch('input_folder/', 'output_folder/')
```




# 上传数据集

huggingface-cli upload heyongxian/watermark_images ./data --repo-type=dataset


```
7z a -v500m data.7z ./data

-mx9 最高压缩级别，-mmt=8 使用 8 线程加速
7z a -t7z -mx9 -mmt=8 final.7z ./

aws s3 --no-sign-request \
  --endpoint http://bs3-sgp.internal \
  cp mark-pro.zip s3://oversea-game/ai-model/

```