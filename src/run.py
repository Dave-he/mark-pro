
#!/usr/bin/env python3
"""
训练主函数

指定训练的配置文件，开始训练对应的模型
"""
import argparse
import sys

# 导入不同的train脚本
from scripts import predict, predict_mask, predict_rmvl

def main():
    parser = argparse.ArgumentParser(description="选择预测脚本")
    parser.add_argument('--mode', type=str, default='default', choices=['default', 'mask', 'rmvl'], help='选择预测模式')
    
    # 为 mask 模式添加必要的参数
    parser.add_argument('--input', type=str, help='输入图像或目录 (mask 模式必需)')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--model', type=str, help='模型权重路径')
    parser.add_argument('--device', type=str, help='设备 (cuda/cpu)')
    parser.add_argument('--save-mask', action='store_true', help='保存预测的 mask')
    parser.add_argument('--iopaint-model', type=str, choices=['lama', 'ldm', 'zits', 'mat'], help='IOPaint 模型')
    
    args = parser.parse_args()

    if args.mode == 'default':
        predict.batch_predict()
    elif args.mode == 'mask':
        if not args.input:
            print('错误：mask 模式需要 --input 参数')
            print('使用方法：python run.py --mode mask --input <图像路径>')
            return
        
        # 构建 predict_mask 需要的参数
        mask_args = ['--input', args.input]
        if args.output:
            mask_args.extend(['--output', args.output])
        if args.model:
            mask_args.extend(['--model', args.model])
        if args.device:
            mask_args.extend(['--device', args.device])
        if args.save_mask:
            mask_args.append('--save-mask')
        if args.iopaint_model:
            mask_args.extend(['--iopaint-model', args.iopaint_model])
        
        # 临时修改 sys.argv 来传递参数
        original_argv = sys.argv
        sys.argv = ['predict_mask.py'] + mask_args
        try:
            predict_mask.main()
        finally:
            sys.argv = original_argv
            
    elif args.mode == 'rmvl':
        predict_rmvl.main()
    else:
        print('未知模式，请选择 default、mask 或 rmvl')

if __name__ == '__main__':
    main()