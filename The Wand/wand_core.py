import tkinter as tk
from wand_setup import setup_dependencies
from wand_hpc_manager import initialize_hpc_manager
from .wand_helper import WandHelper
from .wand_network_manager import NetworkManager
from .wand_hpc_manager import HPCManager

class WandCore:
    def __init__(self, config):
        self.config = config
        self.helper = WandHelper()
        self.network = NetworkManager(config)
        self.hpc = HPCManager(config)
    
    def initialize(self):
        """Initialize core AIOS IO systems"""
        self.helper.setup_logging()
        self.network.initialize()
        self.hpc.initialize()
    
    def start_services(self):
        """Start all AIOS IO services"""
        self.network.start()
        self.hpc.start()

def run_system(config):
    # Initialize high-level system settings from the configuration
    print("Running wand_core with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Delegate further initialization and processing
    # e.g., loading UI, ML pipelines, dynamic module registration, etc.
    # ...existing logic moved from TheWand.py...
    print("System initialized. Launching core functionalities...")
    
    # Placeholder: launch additional functionalities
    # This is where wand_helper, wand_dashboard, and other modules would be integrated.

def main():
    # Initialize system dependencies and HPC manager.
    setup_dependencies()
    initialize_hpc_manager()
    
    # Create and configure main window.
    root = tk.Tk()
    root.title("AIOS IO - Modularized")
    root.geometry("1200x800")
    # ...additional GUI code...
    root.mainloop()

if __name__ == "__main__":
    main()
