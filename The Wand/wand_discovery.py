import os
import importlib.util
import inspect
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def discover_nodes():
    print("Scanning network for AIOS IO nodes...")
    return ["Node_A", "Node_B", "Node_C"]

def get_node_status(node: str):
    print(f"Retrieving status for {node}...")
    return {"status": "active", "info": "stub_data"}

class ScriptDiscovery:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_path = Path(config.get('system', {}).get('base_path', '.'))
        self.discovered_scripts = {}
        self.observer = Observer()
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for script discovery"""
        log_path = self.base_path / 'logs' / 'discovery.log'
        log_path.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def start_monitoring(self):
        """Start monitoring for new scripts"""
        event_handler = ScriptEventHandler(self)
        self.observer.schedule(event_handler, str(self.base_path), recursive=True)
        self.observer.start()
        logging.info("Started script discovery monitoring")
        
    def stop_monitoring(self):
        """Stop monitoring for new scripts"""
        self.observer.stop()
        self.observer.join()
        logging.info("Stopped script discovery monitoring")
        
    def scan_for_scripts(self) -> Dict[str, Any]:
        """Scan system for Python scripts and analyze their capabilities"""
        discovered = {}
        
        for path in self.base_path.rglob("*.py"):
            if self._should_ignore_file(path):
                continue
                
            try:
                script_info = self._analyze_script(path)
                if script_info:
                    discovered[str(path)] = script_info
                    logging.info(f"Discovered script: {path}")
            except Exception as e:
                logging.error(f"Error analyzing {path}: {e}")
                
        self.discovered_scripts.update(discovered)
        return discovered
        
    def _should_ignore_file(self, path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = [
            '__pycache__',
            '.git',
            'venv',
            'env',
            'test_'
        ]
        return any(pattern in str(path) for pattern in ignore_patterns)
        
    def _analyze_script(self, path: Path) -> Dict[str, Any]:
        """Analyze a Python script for capabilities and dependencies"""
        try:
            spec = importlib.util.spec_from_file_location("module", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            capabilities = []
            dependencies = []
            
            # Analyze classes and functions
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    capabilities.extend(self._analyze_class(obj))
                elif inspect.isfunction(obj):
                    capabilities.append({
                        'type': 'function',
                        'name': name,
                        'doc': inspect.getdoc(obj)
                    })
                    
            # Extract import statements
            with open(path, 'r') as f:
                content = f.read()
                dependencies = self._extract_dependencies(content)
                
            return {
                'path': str(path),
                'last_modified': os.path.getmtime(path),
                'capabilities': capabilities,
                'dependencies': dependencies
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze {path}: {e}")
            return None
            
    def _analyze_class(self, cls) -> List[Dict[str, Any]]:
        """Analyze a class for its methods and properties"""
        capabilities = []
        
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('_'):  # Skip private methods
                capabilities.append({
                    'type': 'method',
                    'class': cls.__name__,
                    'name': name,
                    'doc': inspect.getdoc(method)
                })
                
        return capabilities
        
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract import statements from code"""
        import_lines = [
            line.strip() for line in content.split('\n')
            if line.strip().startswith(('import ', 'from '))
        ]
        return import_lines
        
    def integrate_script(self, path: Path) -> bool:
        """Attempt to integrate a discovered script into AIOS IO"""
        if not path.exists():
            logging.error(f"Script not found: {path}")
            return False
            
        try:
            # Add to registry
            registry_path = self.base_path / 'wand_registry.json'
            registry = self._load_registry(registry_path)
            
            script_info = self._analyze_script(path)
            if script_info:
                registry[str(path)] = script_info
                self._save_registry(registry_path, registry)
                logging.info(f"Successfully integrated {path}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to integrate {path}: {e}")
            
        return False
        
    def _load_registry(self, path: Path) -> Dict:
        """Load the script registry"""
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_registry(self, path: Path, registry: Dict):
        """Save the script registry"""
        with open(path, 'w') as f:
            json.dump(registry, f, indent=2)

class ScriptEventHandler(FileSystemEventHandler):
    def __init__(self, discovery):
        self.discovery = discovery
        
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.py'):
            return
            
        path = Path(event.src_path)
        if not self.discovery._should_ignore_file(path):
            logging.info(f"New script detected: {path}")
            self.discovery.integrate_script(path)
            
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.py'):
            return
            
        path = Path(event.src_path)
        if not self.discovery._should_ignore_file(path):
            logging.info(f"Script modified: {path}")
            self.discovery.integrate_script(path)