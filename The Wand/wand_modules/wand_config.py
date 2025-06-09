import json
from pathlib import Path
from typing import Dict, Optional

class WandConfig:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        # ...existing code...
        
    def save_config(self):
        """Save current configuration"""
        # ...existing code...
