import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """统一的配置管理器"""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self.common_config_path = self.config_dir / 'common.yaml'
        self._common_config = None
    
    def load_common_config(self) -> Dict[str, Any]:
        """加载公共配置"""
        if self._common_config is None:
            with open(self.common_config_path, 'r', encoding='utf-8') as f:
                self._common_config = yaml.safe_load(f)
        return self._common_config.copy()
    
    def load_config(self, config_name: str, custom_config: Optional[Dict] = None) -> Dict[str, Any]:
        """加载指定配置文件并与公共配置合并
        
        Args:
            config_name: 配置文件名（不含扩展名）
            custom_config: 自定义配置字典，会覆盖文件配置
        
        Returns:
            合并后的完整配置
        """
        # 加载公共配置
        config = self.load_common_config()
        
        # 加载特定配置文件
        config_file = self.config_dir / f'{config_name}.yaml'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                specific_config = yaml.safe_load(f)
            if specific_config:
                config = self._deep_merge(config, specific_config)
        
        # 应用自定义配置
        if custom_config:
            config = self._deep_merge(config, custom_config)
        
        # 处理路径
        config = self._resolve_paths(config)
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _resolve_paths(self, config: Dict) -> Dict:
        """解析相对路径为绝对路径"""
        def resolve_path_recursive(obj, base_path):
            if isinstance(obj, dict):
                return {k: resolve_path_recursive(v, base_path) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_path_recursive(item, base_path) for item in obj]
            elif isinstance(obj, str) and obj.startswith('./'):
                return str(Path(base_path) / obj[2:])
            return obj
        
        base_path = self.config_dir.parent  # src目录
        return resolve_path_recursive(config, base_path)
    
    def save_config(self, config: Dict, filename: str):
        """保存配置到文件"""
        config_file = self.config_dir / f'{filename}.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def get_available_configs(self) -> list:
        """获取所有可用的配置文件"""
        yaml_files = list(self.config_dir.glob('*.yaml'))
        return [f.stem for f in yaml_files if f.stem != 'common']

# 全局配置管理器实例
config_manager = ConfigManager()

# 便捷函数
def load_config(config_name: str, custom_config: Optional[Dict] = None) -> Dict[str, Any]:
    """加载配置的便捷函数"""
    return config_manager.load_config(config_name, custom_config)

def get_config(config_name: str = 'default', **kwargs) -> Dict[str, Any]:
    """获取配置的便捷函数，支持关键字参数覆盖"""
    return config_manager.load_config(config_name, kwargs if kwargs else None)