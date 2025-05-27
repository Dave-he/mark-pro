
"""
训练主函数

指定训练的配置文件，开始训练对应的模型
"""
import argparse

# 导入不同的train脚本
from scripts import train, train_mask, train_rmvl, predict, predict_mask, predict_rmvl

def main():
    parser = argparse.ArgumentParser(description="选择训练脚本")
    parser.add_argument('--mode', type=str, default='default', choices=['default', 'mask', 'rmvl'], help='选择训练模式')
    parser.add_argument('--predict', type=bool, default=False, help='是否预测', required=False)
    args = parser.parse_args()
    if args.predict:
        if args.mode == 'default':
            predict.batch_predict()
        elif args.mode =='mask':
            predict_mask.main()
        elif args.mode == 'rmvl':
            predict_rmvl.main()
        else:
            print('未知模式，请选择 default、mask 或 rmvl')
    else :
        if args.mode == 'default':
            train.train()
        elif args.mode == 'mask':
            train_mask.train()
        elif args.mode == 'rmvl':
            train_rmvl.main()
        else:
            print('未知模式，请选择 default、mask 或 rmvl')

if __name__ == '__main__':
    main()