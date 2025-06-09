import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import sys
import codecs

class WandLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.build_log_path = log_dir / 'builds.json'
        self.error_log_path = log_dir / 'errors.json'
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging system with structured output"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure main logger with proper encoding handling
        console_handler = logging.StreamHandler(sys.stdout)  # Use stdout instead of stderr
        console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        
        file_handler = logging.FileHandler(
            self.log_dir / 'wand.log',
            encoding='utf-8',
            mode='a'
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )

        # Initialize log history files
        if not self.build_log_path.exists():
            self._save_json(self.build_log_path, {'builds': []})
        if not self.error_log_path.exists():
            self._save_json(self.error_log_path, {'errors': []})

    def log_info(self, message: str):
        """Log an info message."""
        logging.info(message)
        self._append_to_log('info', message)

    def log_error(self, message: str, error: Exception = None):
        """Log an error message with optional exception."""
        if error:
            message = f"{message}: {str(error)}"
        logging.error(message)
        self._append_to_log('error', message)

    def log_warning(self, message: str):
        """Log a warning message."""
        logging.warning(message)
        self._append_to_log('warning', message)

    def log_debug(self, message: str):
        """Log a debug message."""
        logging.debug(message)
        self._append_to_log('debug', message)

    def _append_to_log(self, level: str, message: str):
        """Append message to appropriate log file with timestamp."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        
        # Determine target log file based on level
        target_path = self.error_log_path if level == 'error' else self.build_log_path
        log_data = self._load_json(target_path)
        
        if level == 'error':
            log_data['errors'].append(log_entry)
        else:
            if 'general_logs' not in log_data:
                log_data['general_logs'] = []
            log_data['general_logs'].append(log_entry)
            
        self._save_json(target_path, log_data)
        
    def log_build_step(self, step: Dict, success: bool, error: str = None):
        """Log build step with detailed information"""
        timestamp = datetime.now().isoformat()
        build_entry = {
            'timestamp': timestamp,
            'step_number': step.get('step_number'),
            'success': success,
            'tasks_count': len(step.get('tasks', [])),
            'error': error
        }
        
        # Update build history
        builds = self._load_json(self.build_log_path)
        builds['builds'].append(build_entry)
        self._save_json(self.build_log_path, builds)
        
        # Log to main log file
        level = logging.INFO if success else logging.ERROR
        self._log(level, f"Build step {step.get('step_number')}: {'Success' if success else 'Failed'}")
        
    def log_error(self, component: str, error: str, context: Dict = None):
        """Log error with component context"""
        timestamp = datetime.now().isoformat()
        error_entry = {
            'timestamp': timestamp,
            'component': component,
            'error': str(error),
            'context': context or {}
        }
        
        # Update error history
        errors = self._load_json(self.error_log_path)
        errors['errors'].append(error_entry)
        self._save_json(self.error_log_path, errors)
        
        # Log to main log file
        self._log(logging.ERROR, f"[{component}] {error}")
        
    def _log(self, level: int, message: str):
        """Internal logging helper"""
        logging.log(level, message)
        
    def _load_json(self, path: Path) -> Dict:
        """Load JSON file with error handling"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'builds': [], 'errors': [], 'general_logs': []}
        except json.JSONDecodeError:
            self._log(logging.ERROR, f"Corrupted log file: {path}")
            return {'builds': [], 'errors': [], 'general_logs': []}
            
    def _save_json(self, path: Path, data: Dict):
        """Save JSON file with error handling"""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._log(logging.ERROR, f"Failed to save log file {path}: {e}")
