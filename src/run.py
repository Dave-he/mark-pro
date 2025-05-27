
"""
运行脚本

读取对应流程的配置文件，读取文件夹下的图片逐个处理，将结果保存到指定文件夹
1.各个模型输出mask
2.使用iopaint里的模型消除水印(lama)
3.使用unet同时生成mask和消除水印

"""
from scripts.predict import batch_predict

def main():
    batch_predict()
    


if __name__ == '__main__':
    main()
