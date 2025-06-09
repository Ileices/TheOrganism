"""
Merged AIOS IO implementation incorporating:
 - wand_helper
 - wand_federated
 - wand_discovery
 - wand_dashboard
 - wand_ai_model
 - Two NodeRegistryService variants merged into one (Flask endpoints and a custom class)
 - HPC_Scheduler
 - AIOS_DistributedLoadBalancer
 - wand_core (main execution)
"""

# ---------- wand_helper ----------
def log_message(message: str):
    print(f"[LOG] {message}")

def validate_input(data):
    return bool(data)

def load_config(file_path: str):
    print(f"Loading config from {file_path}...")
    return {"config": "stub_config"}

# ---------- wand_federated ----------
def initiate_federated_learning(node_list: list):
    print("Initiating federated learning across nodes:")
    for node in node_list:
        print(f" - Node: {node}")
    return "Federated learning started."

def aggregate_models(models: list):
    print("Aggregating models from nodes...")
    return {"aggregated_model": "stub_aggregated"}

# ---------- wand_discovery ----------
def discover_nodes():
    print("Scanning network for AIOS IO nodes...")
    return ["Node_A", "Node_B", "Node_C"]

def get_node_status(node: str):
    print(f"Retrieving status for {node}...")
    return {"status": "active", "info": "stub_data"}

# ---------- wand_dashboard ----------
import tkinter as tk
from tkinter import ttk
def create_dashboard():
    root = tk.Tk()
    root.title("AIOS IO Dashboard")
    root.geometry("800x600")
    
    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)
    
    label = ttk.Label(frame, text="Dashboard - Monitoring AIOS IO Activity")
    label.pack(pady=10)
    
    # ...existing UI widget code...
    print("Dashboard created.")
    root.mainloop()

# ---------- wand_ai_model ----------
def load_model(model_path: str):
    print(f"Loading AI model from {model_path}...")
    return {"model": "stub_model"}

def train_model(model, data):
    print("Training model with provided data...")
    return "Training completed."

def update_model(model, updates):
    print("Updating model with new parameters...")
    return "Model updated."

# ---------- Merged NodeRegistryService ----------
from flask import Flask, request, jsonify
import threading
import time
from typing import Dict, Any, List

# Flask-based registry (for API access)
app = Flask(__name__)
node_registry: Dict[str, Dict[str, Any]] = {}

@app.route("/register", methods=["POST"])
def register_node():
    data = request.get_json()
    node_id = data.get("node_id")
    node_info = data.get("info", {})
    node_info["last_seen"] = time.time()
    node_registry[node_id] = node_info
    return jsonify({"status": "registered", "node_id": node_id})

@app.route("/nodes", methods=["GET"])
def get_nodes():
    cutoff = time.time() - 60
    active_nodes = {nid: info for nid, info in node_registry.items() if info["last_seen"] >= cutoff}
    return jsonify(active_nodes)

def run_registry_service():
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=6000), daemon=True).start()

# Custom class-based registry (for internal use)
class NodeRegistryService:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def register_node(self, node_id: str, info: Dict[str, Any]):
        with self.lock:
            self.nodes[node_id] = {"info": info, "last_seen": time.time()}

    def update_node(self, node_id: str, info: Dict[str, Any]):
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id]["info"].update(info)
                self.nodes[node_id]["last_seen"] = time.time()

    def deregister_node(self, node_id: str):
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]

    def get_active_nodes(self, timeout: int = 60) -> List[str]:
        current_time = time.time()
        with self.lock:
            return [node_id for node_id, data in self.nodes.items() if current_time - data["last_seen"] <= timeout]

# ---------- HPC_Scheduler ----------
import queue
class HPCScheduler:
    def __init__(self):
        self.job_queue = queue.PriorityQueue()
        self.lock = threading.Lock()

    def schedule_job(self, priority: int, job: callable, *args, **kwargs):
        self.job_queue.put((priority, (job, args, kwargs)))
        print(f"Job scheduled with priority {priority}")

    def run(self):
        while not self.job_queue.empty():
            priority, (job, args, kwargs) = self.job_queue.get()
            print(f"Executing job with priority {priority}")
            job(*args, **kwargs)
            time.sleep(0.1)  # simulate processing delay

# ---------- AIOS_DistributedLoadBalancer ----------
class DistributedLoadBalancer:
    def __init__(self, nodes: list):
        self.nodes = nodes
        self.index = 0
        self.lock = threading.Lock()

    def get_next_node(self):
        with self.lock:
            if not self.nodes:
                raise Exception("No available nodes.")
            node = self.nodes[self.index]
            self.index = (self.index + 1) % len(self.nodes)
            return node

    def assign_task(self, task):
        node = self.get_next_node()
        print(f"Assigning task '{task}' to {node}")
        return node

# ---------- wand_core (Main Execution Engine) ----------
def main():
    # ...existing initialization code...
    config = load_config("/c:/Users/lokee/Documents/AIOS IO/config.yaml")
    log_message("Configuration loaded.")

    # Start the Node Registry Service (Flask API)
    run_registry_service()
    log_message("Node registry service started.")

    # Discover nodes and initiate federated learning
    nodes = discover_nodes()
    log_message(f"Discovered nodes: {nodes}")
    initiate_federated_learning(nodes)

    # Initialize Distributed Load Balancer
    lb = DistributedLoadBalancer(nodes)
    lb.assign_task("Sample Task")

    # Schedule a sample job with HPC Scheduler
    scheduler = HPCScheduler()
    scheduler.schedule_job(1, log_message, "Executing scheduled job")
    scheduler.run()

    # Optionally start dashboard (blocking; comment out if not needed)
    # create_dashboard()

    # ...existing code...

if __name__ == "__main__":
    main()
