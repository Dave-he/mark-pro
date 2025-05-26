#!/bin/bash
python3 unetpp/train.py 
if [ $? -eq 0 ]; then
    python3 unetpp/predict.py
else
    echo "训练失败，未执行预测。"
fi


PYTHONPATH=./rmvl python3 rmvl/scripts/train.py --config rmvl/configs/default.yaml
if [ $? -eq 0 ]; then
    python3 rmvl/scripts/predict.py --model-path outputs/best_model.pth --input data/test/watermarked --output outputs/predict_results --image-size 256 --mask-threshold 0.5 --device cuda
else
    echo "训练失败，未执行预测。"
fi