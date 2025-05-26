# 图片去水印

## 安装
```bash
#python 3.9-3.13都可以
virtualenv venv -p python3.12
source venv/bin/activate
# ./venv/Scripts/activate.ps1 #windows

#conda
conda create -n env12 python=3.12
conda activate env12


pip install torch torchvision
pip install opencv-python
pip install pillow numpy matplotlib tqdm pyyaml 
pip install yacs
pip install tensorboard

pip install -r requirements.txt
```

## 运行
```bash
python main.py
```

## 数据集

在data/目录下创建两个子目录：
watermarked/：存放带水印的图像
clean/：存放对应的无水印图像
图像应按文件名匹配（例如：watermarked/1.jpg对应clean/1.jpg）

1. 数据集
```bash
RMVL/
├── train/                  # 训练集
│   ├── watermarked/        # 带水印图像
│   └── clean/              # 无水印图像
└── val/                    # 验证集
    ├── watermarked/        # 带水印图像
    └── clean/              # 无水印图像
```

# 代码说明

## RMVL处理水印

1. 代码结构

```bash
watermark-removal/
├── data/                   # 数据集目录
│   ├── train/              # 训练集
│   │   ├── watermarked/    # 带水印图像
│   │   └── clean/          # 无水印图像
│   └── val/                # 验证集
├── models/                 # 模型定义
│   ├── unet.py             # 基础U-Net模型
│   └── seg_unet.py         # 带分割分支的增强U-Net
├── utils/                  # 工具函数
│   ├── dataset.py          # 数据集加载
│   ├── transforms.py       # 图像变换
│   ├── losses.py           # 损失函数
│   └── metrics.py          # 评估指标
├── scripts/                # 脚本目录
│   ├── train.py            # 训练脚本
│   ├── predict.py          # 预测脚本
│   └── evaluate.py         # 评估脚本
├── configs/                # 配置文件
│   └── default.yaml        # 默认配置
├── logs/                   # 训练日志
├── checkpoints/            # 模型检查点
└── results/                # 预测结果

``` 

2. 模型训练
```bash
PYTHONPATH=./rmvl python rmvl/scripts/train.py --config rmvl/configs/default.yaml
```

3. 模型预测
```bash
cd rmvl

python scripts/predict.py \
  --model-path checkpoints/model_best.pth \
  --input test_images/watermarked_001.jpg \
  --output results/

python scripts/predict.py \
  --model-path checkpoints/model_best.pth \
  --input test_images/ \
  --output results/
```

## UNet++

1. 项目结构
   
```bash
watermark_remover/
├── configs/
│   └── default.yaml          # 配置文件
├── dataset.py                # 数据集处理
├── models/
│   ├── __init__.py
│   └── seg_unetpp.py         # 分割引导Unet++模型
├── train.py                  # 训练脚本
├── predict.py                # 预测脚本
├── utils/
│   ├── __init__.py
│   ├── losses.py             # 损失函数
│   ├── metrics.py            # 评估指标
│   └── visualize.py          # 可视化工具
├── requirements.txt          # 依赖项
└── README.md                 # 使用说明
```
2. 训练
```bash
cd unetpp
python unetpp/train.py 
```

3. 预测
```bash
cd unetpp
python predict.py --config configs/default.yaml --input inputs/ --output results/
```

结果将保存在outputs/目录下的三个子目录：
- restored/：修复后的图像
- masks/：预测的水印掩码
- overlays/：带水印区域标记的叠加图



# 上传数据集

huggingface-cli upload heyongxian/watermark_images ./data --repo-type=dataset


```
-mx9 最高压缩级别，-mmt=8 使用 8 线程加速

7z a -t7z -mx9 -mmt=8 final.7z ./

aws s3 --no-sign-request \
  --endpoint http://bs3-sgp.internal \
  cp mark-pro.zip s3://oversea-game/ai-model/
```