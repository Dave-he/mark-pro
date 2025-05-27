#!/bin/bash
cd src 

#训练 UNet++
python3 main.py --mode default
if [ $? -eq 0 ]; then
    python3 run.py --mode default
else
    echo "训练失败，未执行预测。"
fi


#训练 rmvl
python3 main.py --mode mask
if [ $? -eq 0 ]; then
    python3 run.py --mode mask
else
    echo "训练失败，未执行预测。"
fi

#训练 mask-rcnn
python3 main.py --mode rmvl
if [ $? -eq 0 ]; then
    python3 run.py --mode rmvl
else
    echo "训练失败，未执行预测。"
fi
