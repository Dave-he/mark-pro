"""统一配置入口"""

from .config_manager import config_manager, load_config, get_config
from typing import Dict, Any
import os

# 预定义的配置名称
CONFIG_NAMES = {
    'unetpp': 'unetpp',
    'unet_mask': 'unet_mask', 
    'rmvl': 'rmvl',
    'mask_rcnn': 'mask_rcnn'
}

def get_unetpp_config(**kwargs) -> Dict[str, Any]:
    """获取UNet++配置"""
    return get_config('unetpp', **kwargs)

def get_unet_mask_config(**kwargs) -> Dict[str, Any]:
    """获取UNet掩码配置"""
    return get_config('unet_mask', **kwargs)

def get_rmvl_config(**kwargs) -> Dict[str, Any]:
    """获取水印去除配置"""
    return get_config('rmvl', **kwargs)

def get_mask_rcnn_config(**kwargs) -> Dict[str, Any]:
    """获取Mask R-CNN配置"""
    return get_config('mask_rcnn', **kwargs)

# 向后兼容的配置对象
class LegacyConfig:
    """向后兼容的配置类"""
    
    def __init__(self, config_name: str = 'unetpp'):
        self._config = load_config(config_name)
        self._setup_attributes()
    
    def _setup_attributes(self):
        """设置属性以支持点号访问"""
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, self._dict_to_obj(value))
            else:
                setattr(self, key, value)
    
    def _dict_to_obj(self, d):
        """将字典转换为对象"""
        class DictObj:
            def __init__(self, dictionary):
                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        setattr(self, key, DictObj(value))
                    else:
                        setattr(self, key, value)
        return DictObj(d)
    
    def update(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """转换为字典"""
        return self._config

# 默认配置实例（向后兼容）
cfg = LegacyConfig('unetpp')