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

2. 模型训练
```bash
PYTHONPATH=./rmvl python rmvl/scripts/train.py --config rmvl/configs/default.yaml

cd unetpp
python unetpp/train.py 
```

3. 模型预测
```bash
python scripts/predict.py \
  --model-path checkpoints/model_best.pth \
  --input test_images/watermarked_001.jpg \
  --output results/

python scripts/predict.py \
  --model-path checkpoints/model_best.pth \
  --input test_images/ \
  --output results/


```bash
cd unetpp
python predict.py --config configs/default.yaml --input inputs/ --output results/
```





# 上传数据集

huggingface-cli upload heyongxian/watermark_images ./data --repo-type=dataset


```
-mx9 最高压缩级别，-mmt=8 使用 8 线程加速

7z a -t7z -mx9 -mmt=8 final.7z ./

aws s3 --no-sign-request \
  --endpoint http://bs3-sgp.internal \
  cp mark-pro.zip s3://oversea-game/ai-model/
```