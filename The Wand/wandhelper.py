import logging
import json
from pathlib import Path
from typing import Dict, Optional

def initialize_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

def create_default_config() -> Dict:
    """Create default configuration dictionary"""
    return {
        'build_directory': 'builds',
        'plugins_directory': 'plugins',
        'temp_directory': 'temp',
        'log_directory': 'logs',
        'enable_ai_learning': False,
        'learning_rate': 0.001,
        'cpu_limit': 0.8,
        'memory_limit': 0.7,
        'max_retries': 3,
        'recovery_timeout': 300,
        'monitoring_interval': 1.0
    }

class WandHelper:
    def __init__(self, wand_instance):
        self.wand = wand_instance
        self.logger = logging.getLogger('WandHelper')

    def ensure_directory(self, path: str) -> bool:
        """Ensure directory exists, create if not"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create directory {path}: {e}")
            return False

    def get_project_path(self) -> str:
        """Get project directory path"""
        return str(Path(self.wand.config.config.get('project_directory', 'projects')))

    def validate_path(self, path: str) -> bool:
        """Validate path exists and is accessible"""
        try:
            p = Path(path)
            return p.exists() and p.is_dir()
        except Exception:
            return False

    def load_json_file(self, path: str) -> Optional[Dict]:
        """Load and parse JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {path}: {e}")
            return None

    def save_json_file(self, path: str, data: Dict) -> bool:
        """Save data to JSON file"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save JSON file {path}: {e}")
            return False
