import logging
import json
import os
import shutil
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

def log_message(message: str):
    print(f"[LOG] {message}")

def validate_input(data):
    return bool(data)

def load_config(file_path: str):
    print(f"Loading config from {file_path}...")
    return {"config": "stub_config"}

class WandHelper:
    def __init__(self):
        self.log_path = Path('logs')
        self.log_path.mkdir(exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        """Configure system-wide logging"""
        log_file = self.log_path / f'aios_io_{datetime.now().strftime("%Y%m%d")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu': {
                'cores': psutil.cpu_count(),
                'usage_per_core': cpu_percent,
                'average_usage': sum(cpu_percent) / len(cpu_percent)
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent_used': disk.percent
            }
        }
        
    def backup_config(self, config_path: Path, max_backups: int = 5):
        """Create timestamped backup of configuration"""
        if not config_path.exists():
            logging.error(f"Config file not found: {config_path}")
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.parent / 'backups'
        backup_path.mkdir(exist_ok=True)
        
        backup_file = backup_path / f"{config_path.stem}_{timestamp}{config_path.suffix}"
        
        try:
            shutil.copy2(config_path, backup_file)
            logging.info(f"Config backup created: {backup_file}")
            
            # Cleanup old backups
            self._cleanup_old_backups(backup_path, max_backups)
            return True
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            return False
            
    def _cleanup_old_backups(self, backup_path: Path, max_backups: int):
        """Remove oldest backups exceeding max_backups"""
        backups = sorted(backup_path.glob('*'), key=os.path.getctime)
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            oldest.unlink()
            logging.info(f"Removed old backup: {oldest}")
            
    def validate_config(self, config: Dict) -> List[str]:
        """Validate configuration structure and values"""
        errors = []
        required_sections = ['system', 'ai', 'hpc', 'security']
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
                
        # Validate system section
        if 'system' in config:
            if 'version' not in config['system']:
                errors.append("Missing system version")
                
        # Validate HPC section
        if 'hpc' in config:
            hpc = config['hpc']
            if 'max_nodes' in hpc and not isinstance(hpc['max_nodes'], int):
                errors.append("HPC max_nodes must be an integer")
                
        # Validate security section
        if 'security' in config:
            security = config['security']
            if 'encryption_level' in security and security['encryption_level'] not in ['high', 'medium', 'low']:
                errors.append("Invalid encryption_level (must be high, medium, or low)")
                
        return errors
        
    def load_json_safe(self, path: Path) -> Optional[Dict]:
        """Safely load and validate JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")
        return None
        
    def save_json_safe(self, path: Path, data: Dict) -> bool:
        """Safely save dictionary to JSON file with backup"""
        if path.exists():
            self.backup_config(path)
            
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving {path}: {e}")
            return False