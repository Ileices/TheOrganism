import logging
import threading
import time

global_node_registry = {}

def register_global_node(node_id, metrics):
    global_node_registry[node_id] = {"metrics": metrics, "last_seen": time.time()}
    logging.info(f"Registered global node: {node_id}")

def exchange_models():
    logging.info("Exchanging AI models between global nodes...")
    # Placeholder: Implement merging of models, self-optimization logic.
    
def monitor_global_network():
    while True:
        for node, info in list(global_node_registry.items()):
            if time.time() - info["last_seen"] > 120:
                del global_node_registry[node]
                logging.warning(f"Global node {node} removed as inactive")
        exchange_models()
        time.sleep(30)

def initialize_network_manager():
    logging.info("Initializing Global Network Manager...")
    threading.Thread(target=monitor_global_network, daemon=True).start()
