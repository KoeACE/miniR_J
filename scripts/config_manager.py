"""
配置管理模块
功能：支持从外置 JSON 文件加载路径配置，支持运行时重载
"""

import os
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)


class ConfigManager:
    """配置管理器"""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_dir: str = None):
        if ConfigManager._initialized:
            return

        if config_dir is None:
            config_dir = os.path.join(WORKSPACE_ROOT, "config")
        self.config_dir = config_dir

        self.paths_path = os.path.join(config_dir, "paths.json")

        self._paths_config: Dict = {}

        self._load_paths_config()

        ConfigManager._initialized = True

    def _load_paths_config(self):
        default_config = {
            "project_root": WORKSPACE_ROOT,
            "md_directories": [os.path.join(WORKSPACE_ROOT, "doc")],
            "md_directory": os.path.join(WORKSPACE_ROOT, "doc"),
            "faiss_index_path": os.path.join(WORKSPACE_ROOT, "faiss_index"),
            "model_path": os.path.join(WORKSPACE_ROOT, "modelscope_models", "bge-m3"),
            "reranker_path": os.path.join(WORKSPACE_ROOT, "modelscope_models", "bge-reranker-v2-m3"),
            "db_path": os.path.join(WORKSPACE_ROOT, "rag_data.db")
        }

        if os.path.exists(self.paths_path):
            try:
                with open(self.paths_path, 'r', encoding='utf-8-sig') as f:
                    self._paths_config = json.load(f)
                self._paths_config = self._normalize_paths_config(self._paths_config)
                print(f"已加载路径配置: {self.paths_path}")
            except Exception as e:
                print(f"加载路径配置失败: {e}，使用默认配置")
                self._paths_config = default_config
        else:
            print(f"路径配置文件不存在: {self.paths_path}，使用默认配置")
            self._paths_config = default_config
            self._create_default_paths_config()

    def _get_default_config(self) -> Dict:
        return {
            "project_root": "${PROJECT_ROOT}",
            "md_directories": ["doc"],
            "faiss_index_path": "faiss_index",
            "model_path": "modelscope_models/bge-m3",
            "reranker_path": "modelscope_models/bge-reranker-v2-m3",
            "db_path": "rag_data.db"
        }

    def _resolve_path_value(self, value: Any) -> Any:
        if isinstance(value, str):
            value = value.replace("${PROJECT_ROOT}", WORKSPACE_ROOT)
            if not os.path.isabs(value):
                value = os.path.join(WORKSPACE_ROOT, value)
            return os.path.normpath(value)
        if isinstance(value, list):
            return [self._resolve_path_value(item) for item in value]
        return value

    def _normalize_paths_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(config)
        for key in ["project_root", "md_directory", "md_directories", "faiss_index_path", "model_path", "reranker_path", "db_path"]:
            if key in normalized:
                normalized[key] = self._resolve_path_value(normalized[key])
        if "md_directories" not in normalized:
            md_directory = normalized.get("md_directory", os.path.join(WORKSPACE_ROOT, "doc"))
            normalized["md_directories"] = [md_directory]
        return normalized

    def _create_default_paths_config(self):
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            default_config = self._get_default_config()
            with open(self.paths_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print(f"已创建默认路径配置文件: {self.paths_path}")
        except Exception as e:
            print(f"创建默认路径配置文件失败: {e}")

    def reload(self):
        print("正在重新加载配置...")
        self._load_paths_config()
        print("配置重新加载完成")

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            pass
        cls._initialized = False
        cls._instance = None

    def reload_paths(self):
        print("正在重新加载路径配置...")
        self._load_paths_config()
        print("路径配置重新加载完成")

    @property
    def project_root(self) -> str:
        return self._paths_config.get("project_root", WORKSPACE_ROOT)

    @property
    def md_directories(self) -> list:
        dirs = self._paths_config.get("md_directories", [os.path.join(WORKSPACE_ROOT, "doc")])
        return [
            os.path.join(WORKSPACE_ROOT, d) if not os.path.isabs(d) else d
            for d in dirs
        ]

    @property
    def md_directory(self) -> str:
        dirs = self.md_directories
        return dirs[0] if dirs else os.path.join(WORKSPACE_ROOT, "doc")

    @property
    def faiss_index_path(self) -> str:
        return self._paths_config.get("faiss_index_path", os.path.join(WORKSPACE_ROOT, "faiss_index"))

    @property
    def model_path(self) -> str:
        return self._paths_config.get("model_path", os.path.join(WORKSPACE_ROOT, "modelscope_models", "bge-m3"))

    @property
    def reranker_path(self) -> str:
        return self._paths_config.get("reranker_path", os.path.join(WORKSPACE_ROOT, "modelscope_models", "bge-reranker-v2-m3"))

    @property
    def db_path(self) -> str:
        return self._paths_config.get("db_path", os.path.join(WORKSPACE_ROOT, "rag_data.db"))

    def get_path(self, key: str, default: Any = None) -> Any:
        return self._paths_config.get(key, default)

    @staticmethod
    def is_external_path(path: str) -> bool:
        if not isinstance(path, str):
            return False
        parsed = urlparse(path.strip())
        return parsed.scheme in {"http", "https", "data"}

    def to_relative_path(self, abs_path: str) -> str:
        if self.is_external_path(abs_path):
            return abs_path
        try:
            return os.path.relpath(abs_path, self.project_root)
        except ValueError:
            return abs_path

    def to_absolute_path(self, rel_path: str) -> str:
        if self.is_external_path(rel_path):
            return rel_path
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(self.project_root, rel_path))

    def print_config(self):
        print("=" * 60)
        print("当前配置")
        print("=" * 60)
        print("\n路径配置:")
        print(f"  项目根目录: {self.project_root}")
        print(f"  Markdown目录: {self.md_directory}")
        print(f"  FAISS索引路径: {self.faiss_index_path}")
        print(f"  模型路径: {self.model_path}")
        print(f"  数据库路径: {self.db_path}")
        print("=" * 60)


def get_config() -> ConfigManager:
    return ConfigManager()


if __name__ == "__main__":
    config = get_config()
    config.print_config()
