from typing import Dict, Optional
import json
import os
import logging
from pathlib import Path
import time
import shutil

class WandBuilder:
    def __init__(self, config: Dict):
        self.config = config
        self.build_history = []
        self.current_step = None
        self.logger = logging.getLogger('WandBuilder')
        self.rollback_history = {}
        self.task_results = {}
        
    def build_from_json(self, json_data: Dict) -> bool:
        """Execute build instructions from JSON data"""
        try:
            # 1. Validate JSON schema
            if not self._validate_json_schema(json_data):
                raise ValueError("Invalid build step schema")
                
            # 2. Prepare build environment
            build_dir = Path(self.config['build_directory'])
            build_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. Execute build steps
            success = self._execute_build_steps(json_data)
            
            # 4. Record build history
            self.build_history.append({
                'step': json_data.get('step_number'),
                'success': success,
                'timestamp': time.time()
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Build failed: {e}")
            return False
            
    def _validate_json_schema(self, json_data: Dict):
        """Validate JSON against required schema"""
        # ...existing code...
        
    def _execute_build_steps(self, json_data: Dict) -> bool:
        """Execute each build step with rollback capability"""
        tasks = json_data.get('tasks', [])
        step_number = json_data.get('step_number', 'unknown')
        
        # Initialize tracking for this build
        build_id = f"build_{step_number}_{int(time.time())}"
        self.rollback_history[build_id] = []
        self.task_results[build_id] = []
        
        try:
            for task_index, task in enumerate(tasks):
                task_id = f"{build_id}_task_{task_index}"
                success = self._execute_single_task(task, task_id)
                
                if not success:
                    self.logger.error(f"Task {task_id} failed, initiating rollback")
                    self._rollback_failed_build(build_id)
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Build failed: {e}")
            self._rollback_failed_build(build_id)
            return False
            
    def _execute_single_task(self, task: Dict, task_id: str) -> bool:
        """Execute a single task with backup creation"""
        try:
            # Extract task details
            action = task.get('action', '').lower()
            path = task.get('path', '')
            content = task.get('content', '')
            
            # Create backup if modifying existing file
            if action in ['update_file', 'delete_file'] and os.path.exists(path):
                backup_path = self._create_backup(path)
                self.rollback_history[task_id] = {
                    'original_path': path,
                    'backup_path': backup_path,
                    'action': action
                }
            
            # Execute task
            success = self._execute_task_action(action, path, content)
            
            # Log result
            self.task_results[task_id] = {
                'success': success,
                'timestamp': time.time(),
                'task': task
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return False
            
    def _create_backup(self, file_path: str) -> str:
        """Create backup of existing file"""
        backup_dir = Path(self.config['temp_directory']) / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        backup_path = backup_dir / f"{Path(file_path).name}.{timestamp}.bak"
        
        try:
            shutil.copy2(file_path, backup_path)
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return None
            
    def _rollback_failed_build(self, build_id: str):
        """Rollback all changes from failed build"""
        if build_id not in self.rollback_history:
            return
            
        self.logger.info(f"Rolling back build {build_id}")
        
        for task_id, backup_info in reversed(self.rollback_history[build_id]):
            try:
                original_path = backup_info['original_path']
                backup_path = backup_info['backup_path']
                
                if backup_path and os.path.exists(backup_path):
                    shutil.copy2(backup_path, original_path)
                    os.remove(backup_path)
                    
                self.logger.info(f"Rolled back {original_path}")
                
            except Exception as e:
                self.logger.error(f"Rollback failed for {task_id}: {e}")
