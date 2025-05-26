import os
import shutil
import random

DATA_ROOT = './data'
SPLIT_RATIO = 0.2  # 20%作为验证集

train_watermarked = os.path.join(DATA_ROOT, 'train', 'watermarked')
train_clean = os.path.join(DATA_ROOT, 'train', 'clean')
val_watermarked = os.path.join(DATA_ROOT, 'val', 'watermarked')
val_clean = os.path.join(DATA_ROOT, 'val', 'clean')
os.makedirs(val_watermarked, exist_ok=True)
os.makedirs(val_clean, exist_ok=True)

# 只选择两个目录下都存在的同名文件
files_w = set(os.listdir(train_watermarked))
files_c = set(os.listdir(train_clean))
common_files = list(files_w & files_c)
random.shuffle(common_files)
n_val = int(len(common_files) * SPLIT_RATIO)
val_files = common_files[:n_val]

for fname in val_files:
    src_w = os.path.join(train_watermarked, fname)
    dst_w = os.path.join(val_watermarked, fname)
    src_c = os.path.join(train_clean, fname)
    dst_c = os.path.join(val_clean, fname)
    shutil.move(src_w, dst_w)
    shutil.move(src_c, dst_c)
print(f"已将{n_val}对图片从train移动到val，确保一一对应")