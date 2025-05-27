#!/bin/bash
cd src 

#训练 UNet++
python3 main.py --model default
if [ $? -eq 0 ]; then
    python3 main.py --model default --predict true
else
    echo "训练失败，未执行预测。"
fi


#训练 rmvl
python3 main.py --model mask
if [ $? -eq 0 ]; then
    python3 main.py --model mask --predict true
else
    echo "训练失败，未执行预测。"
fi

#训练 mask-rcnn
python3 main.py --model rmvl
if [ $? -eq 0 ]; then
    python3 main.py --model rmvl --predict true
else
    echo "训练失败，未执行预测。"
fi
