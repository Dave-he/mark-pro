import os
import filecmp
from pathlib import Path

def compare_and_clean_directories(watermarked_dir, clean_dir):
    # 确保目录存在
    for dir_path in [watermarked_dir, clean_dir]:
        if not os.path.exists(dir_path):
            print(f"目录不存在: {dir_path}")
            return

    # 获取两个目录中的文件列表
    files_watermarked = set(os.listdir(watermarked_dir))
    files_clean = set(os.listdir(clean_dir))

    # 处理仅在单个目录中存在的文件（存在性不一致）
    only_in_watermarked = files_watermarked - files_clean
    only_in_clean = files_clean - files_watermarked

    for filename in only_in_watermarked:
        file_path = Path(watermarked_dir) / filename
        print(f"删除仅在 watermarked 存在的文件: {file_path}")
        os.remove(file_path)

    for filename in only_in_clean:
        file_path = Path(clean_dir) / filename
        print(f"删除仅在 clean 存在的文件: {file_path}")
        os.remove(file_path)

    # 处理同名但内容不同的文件（内容不一致）
    common_files = files_watermarked & files_clean
    for filename in common_files:
        path_watermarked = Path(watermarked_dir) / filename
        path_clean = Path(clean_dir) / filename
        if not filecmp.cmp(str(path_watermarked), str(path_clean)):
            print(f"删除内容不一致的文件: {filename}")
            os.remove(path_watermarked)
            os.remove(path_clean)

if __name__ == "__main__":
    base_dir = "data/train"
    watermarked_dir = os.path.join(base_dir, "watermarked")
    clean_dir = os.path.join(base_dir, "clean")
    
    print("开始对比目录...")
    compare_and_clean_directories(watermarked_dir, clean_dir)
    print("处理完成。")
