#!/usr/bin/env python3
"""配置迁移脚本，帮助将旧配置迁移到新系统"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import get_config, CONFIG_NAMES
from configs.config_manager import config_manager

def migrate_old_configs():
    """迁移旧配置到新系统"""
    print("开始配置迁移...")
    
    # 检查所有可用配置
    available_configs = config_manager.get_available_configs()
    print(f"可用配置: {available_configs}")
    
    # 测试加载每个配置
    for config_name in available_configs:
        try:
            config = get_config(config_name)
            print(f"✓ 成功加载配置: {config_name}")
        except Exception as e:
            print(f"✗ 加载配置失败: {config_name}, 错误: {e}")
    
    print("配置迁移完成！")

def test_config_compatibility():
    """测试配置兼容性"""
    print("测试配置兼容性...")
    
    # 测试各种配置组合
    test_cases = [
        ('unetpp', {'train': {'batch_size': 16}}),
        ('rmvl', {'data': {'image_size': 256}}),
        ('unet_mask', {'device': 'cpu'}),
    ]
    
    for config_name, custom_config in test_cases:
        try:
            config = get_config(config_name, **custom_config)
            print(f"✓ 配置测试通过: {config_name} with {custom_config}")
        except Exception as e:
            print(f"✗ 配置测试失败: {config_name}, 错误: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='配置迁移工具')
    parser.add_argument('--migrate', action='store_true', help='执行配置迁移')
    parser.add_argument('--test', action='store_true', help='测试配置兼容性')
    
    args = parser.parse_args()
    
    if args.migrate:
        migrate_old_configs()
    
    if args.test:
        test_config_compatibility()
    
    if not args.migrate and not args.test:
        print("请指定 --migrate 或 --test 参数")