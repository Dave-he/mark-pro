#!/bin/bash
cd src 

#训练 UNet++
python3 main.py --config configs/unetpp.yaml
if [ $? -eq 0 ]; then
    python3 main.py --config configs/unetpp.yaml
else
    echo "训练失败，未执行预测。"
fi


#训练 rmvl
python3 main.py --config configs/rmvl.yaml
if [ $? -eq 0 ]; then
    python3 main.py --config configs/rmvl.yaml
else
    echo "训练失败，未执行预测。"
fi

#训练 mask-rcnn
python3 main.py --config configs/mask-rcnn.yaml
if [ $? -eq 0 ]; then
    python3 main.py --config configs/mask-rcnn.yaml
else
    echo "训练失败，未执行预测。"
fi
