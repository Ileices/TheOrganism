import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class WandPluginManager:
    def __init__(self, plugins_dir: Path):
        self.plugins_dir = plugins_dir
        self.loaded_plugins: Dict[str, Any] = {}
        self.logger = logging.getLogger('WandPluginManager')

    def load_plugins(self) -> Dict[str, Any]:
        """Load all plugins from the plugins directory"""
        if not self.plugins_dir.exists():
            self.plugins_dir.mkdir(parents=True)
            return self.loaded_plugins

        for plugin_file in self.plugins_dir.glob('*.py'):
            if plugin_file.stem.startswith('__'):
                continue
                
            try:
                self._load_plugin(plugin_file)
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_file.name}: {e}")

        return self.loaded_plugins

    def _load_plugin(self, plugin_path: Path):
        """Load a single plugin"""
        try:
            spec = importlib.util.spec_from_file_location(
                plugin_path.stem, 
                str(plugin_path)
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'register_plugin'):
                    plugin_instance = module.register_plugin()
                    self.loaded_plugins[plugin_path.stem] = plugin_instance
                    self.logger.info(f"Successfully loaded plugin: {plugin_path.name}")
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_path.name}: {e}")
            raise

    def get_plugin(self, name: str) -> Optional[Any]:
        """Get a loaded plugin by name"""
        return self.loaded_plugins.get(name)

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin by name"""
        if name in self.loaded_plugins:
            try:
                plugin = self.loaded_plugins[name]
                if hasattr(plugin, 'cleanup'):
                    plugin.cleanup()
                del self.loaded_plugins[name]
                return True
            except Exception as e:
                self.logger.error(f"Error unloading plugin {name}: {e}")
        return False
