
"""
训练主函数

指定训练的配置文件，开始训练对应的模型
"""
import argparse

#from scripts.train import train
from scripts.train_mask import train

def main():
    train()
    


if __name__ == '__main__':
    main()