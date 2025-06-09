# Big Link Controller for AIOS IO
import logging
from wand_setup import configure_environment, install_dependencies
from wand_core import WandCore
from wand_hpc_manager import initialize_hpc_manager
# Optional: extend with network manager integration
# from wand_network_manager import initialize_network_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("AIOS IO Big Link Controller starting...")
    configure_environment(silent=False)
    install_dependencies(silent=True)
    initialize_hpc_manager()
    # Optional: initialize global network manager if available
    # initialize_network_manager()
    core = WandCore()
    # ...existing wait loop if required...

if __name__ == "__main__":
    main()
