# MASK-RCNN 识别水印区域

## 安装
```bash
pip install detectron2
pip install iopaint
```

## 使用方法

### 数据集


### 训练
```bash
python train_maskrcnn.py \
    --config-file./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --output-dir./models \
    --data-dir./data \
    --num-classes 1 --thing-classes watermark
``` 

### 推理
```bash
python inference_maskrcnn.py \
    --config-file ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --model-weights ./models/model_final.pth \
    --input-dir /Users/hyx/Pictures/image/input100 \
    --output-dir /Users/hyx/Pictures/image/output100 \
    --score-thresh 0.7 --num-classes 1 --thing-classes watermark



 iopaint run --model=runwayml/stable-diffusion-inpainting --device=cpu \
    --image=/Users/hyx/Pictures/image/input100 \
    --mask=/Users/hyx/Pictures/image/mask \
    --output=output
```



### 后台
```bash



nohup iopaint run --model=Sanster/PowerPaint-V1-stable-diffusion-inpainting         --device=cuda --image=input --mask=mask --output=output --model-dir=.cache >> output.log 2>&1 &

nohup iopaint run --model=Sanster/AnyText --device=cuda --image=input --mask=mask --output=output --model-dir=.cache >> output.log 2>&1 &

nohup iopaint run --model=lama --device=cuda --image=input --mask=mask --output=output --model-dir=.cache >> output.log 2>&1 &

nohup python remwm.py input out1 \
 --iopaint-model-dir .cache --iopaint-model-name lama  >> output.log 2>&1 &
```