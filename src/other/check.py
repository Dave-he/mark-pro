import os
import argparse
from pathlib import Path

def compare_and_clean_directories(watermarked_dir, clean_dir, dry_run=False):
    """
    对比两个目录，删除仅在单个目录中存在的文件
    只按文件名处理，不对比文件内容
    
    Args:
        watermarked_dir: 水印图片目录
        clean_dir: 干净图片目录
        dry_run: 如果为True，只检测不删除；如果为False，真正删除文件
    """
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

    # 显示模式信息
    mode_text = "检测模式" if dry_run else "删除模式"
    print(f"\n当前运行模式: {mode_text}")
    print("=" * 50)

    # 处理仅在 watermarked 目录存在的文件
    if only_in_watermarked:
        print(f"\n发现 {len(only_in_watermarked)} 个仅在 watermarked 目录存在的文件:")
        for filename in only_in_watermarked:
            file_path = Path(watermarked_dir) / filename
            if dry_run:
                print(f"  [检测] 将删除: {file_path}")
            else:
                print(f"  [删除] {file_path}")
                os.remove(file_path)
    else:
        print("\n未发现仅在 watermarked 目录存在的文件")

    # 处理仅在 clean 目录存在的文件
    if only_in_clean:
        print(f"\n发现 {len(only_in_clean)} 个仅在 clean 目录存在的文件:")
        for filename in only_in_clean:
            file_path = Path(clean_dir) / filename
            if dry_run:
                print(f"  [检测] 将删除: {file_path}")
            else:
                print(f"  [删除] {file_path}")
                os.remove(file_path)
    else:
        print("\n未发现仅在 clean 目录存在的文件")

    # 显示保留的文件数量
    remaining_files = files_watermarked & files_clean
    print(f"\n同名文件数量: {len(remaining_files)} 个（将保留）")
    
    # 输出处理结果统计
    print("\n=" * 50)
    if dry_run:
        print(f"检测结果: 发现 {len(only_in_watermarked) + len(only_in_clean)} 个需要删除的文件")
        print("提示: 使用 --delete 参数可真正执行删除操作")
    else:
        print(f"删除完成: 共删除 {len(only_in_watermarked) + len(only_in_clean)} 个文件")
        print(f"保留文件: {len(remaining_files)} 个同名文件")

def main():
    parser = argparse.ArgumentParser(
        description="对比两个目录中的文件，处理仅在单个目录中存在的文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例:
  python check.py                    # 检测模式，只显示将要删除的文件
  python check.py --delete           # 删除模式，真正删除文件
  python check.py --base-dir custom  # 指定自定义基础目录
        """
    )
    
    parser.add_argument(
        "--delete", 
        action="store_true", 
        help="真正删除文件（默认为检测模式，只显示将要删除的文件）"
    )
    
    parser.add_argument(
        "--base-dir", 
        default="data/train",
        help="基础目录路径（默认: data/train）"
    )
    
    parser.add_argument(
        "--watermarked-dir", 
        help="水印图片目录名（默认: watermarked）"
    )
    
    parser.add_argument(
        "--clean-dir", 
        help="干净图片目录名（默认: clean）"
    )
    
    args = parser.parse_args()
    
    # 构建目录路径
    base_dir = args.base_dir
    watermarked_dir = os.path.join(base_dir, args.watermarked_dir or "watermarked")
    clean_dir = os.path.join(base_dir, args.clean_dir or "clean")
    
    print(f"基础目录: {base_dir}")
    print(f"水印目录: {watermarked_dir}")
    print(f"干净目录: {clean_dir}")
    
    # 执行对比和清理
    compare_and_clean_directories(watermarked_dir, clean_dir, dry_run=not args.delete)

if __name__ == "__main__":
    main()
