import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import json

def setup_dependencies():
    # Auto-install dependencies if needed and configure OS/HPC settings.
    # Example: Check and install 'psutil' if missing.
    try:
        import psutil
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    # ...additional setup...
    print("Dependencies and system setup complete.")

def ensure_system_ready() -> bool:
    """Ensure all required directories and configurations exist"""
    try:
        # Create essential directories
        required_dirs = [
            'logs',
            'plugins',
            'builds',
            'temp',
            'configs'
        ]
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize required services
        _initialize_services()
        
        # Verify system dependencies
        _verify_dependencies()
        
        return True
        
    except Exception as e:
        logging.error(f"System setup failed: {e}")
        return False

def _initialize_services():
    """Initialize required background services"""
    try:
        # Set up logging
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Add logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    except Exception as e:
        raise RuntimeError(f"Service initialization failed: {e}")

def _verify_dependencies():
    """Verify all required dependencies are available"""
    required_packages = [
        'flask',
        'requests',
        'waitress',
        'numpy',
        'psutil'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
            
    if missing:
        raise RuntimeError(f"Missing required packages: {', '.join(missing)}")

def get_default_config() -> Dict:
    """Get default system configuration"""
    return {
        'host': '0.0.0.0',
        'port': 6000,
        'log_level': 'INFO',
        'enable_ai': True,
        'max_retries': 3,
        'timeout': 30,
        'directories': {
            'logs': 'logs',
            'plugins': 'plugins',
            'builds': 'builds',
            'temp': 'temp',
            'configs': 'configs'
        }
    }

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from file or return default"""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            
    return get_default_config()
