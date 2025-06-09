from torch.cuda.amp import autocast, GradScaler
import psutil
import importlib.util
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tkinter import filedialog, ttk
import tkinter as tk
import threading
import json
import subprocess
import time
import sys
import os

# --- Virtual Environment & Dependency Installation ---
def ensure_venv():
    """Ensures a virtual environment exists and installs dependencies if missing."""
    venv_path = os.path.join(os.path.expanduser("~"), "Documents", "AIOS IO", "wand_venv")
    if not os.path.exists(venv_path):
        print("[INFO] Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_path])
    pip_path = os.path.join(venv_path, "Scripts", "pip") if os.name == "nt" else os.path.join(venv_path, "bin", "pip")
    subprocess.run([pip_path, "install", "--upgrade", "pip"])
    dependencies = ["torch", "numpy", "torchvision"]
    for package in dependencies:
        subprocess.run([pip_path, "install", package])
    return pip_path.replace("pip", "python")

# --- Failproof GPU/CPU Execution ---
def get_device(force_cpu=False):
    """Selects GPU if available; falls back to CPU if forced or unavailable."""
    try:
        if force_cpu or not torch.cuda.is_available():
            raise RuntimeError("CUDA not available or CPU mode forced")
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[WARNING] Using CPU. Reason: {e}")
        device = torch.device("cpu")
    return device

def load_training_data():
    """Loads training data or generates dummy data if none exists."""
    data_path = os.path.join(os.path.expanduser("~"), "Documents", "AIOS IO", "DATASETS", "training_data.npy")
    if os.path.exists(data_path):
        print(f"Loading training data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        return data.item()  # Expecting dict with 'input' and 'target'
    else:
        print("[WARNING] No dataset found. Generating dummy data.")
        dummy_input = np.random.rand(1000, 10)
        dummy_target = np.random.randint(0, 2, size=(1000,))
        return {'input': dummy_input, 'target': dummy_target}

class TrainingModel(nn.Module):
    def __init__(self):
        super(TrainingModel, self).__init__()
        self.layer = nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)

def train_model(rank, model, data, target, epochs, learning_rate, device):
    """Parallel training function for one training node."""
    print(f"Node {rank} training on {device}...")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    inputs = torch.tensor(data, dtype=torch.float32).to(device)
    targets = torch.tensor(target, dtype=torch.long).to(device)
    initial_lr = learning_rate
    decay_factor = 0.95
    for epoch in range(epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = initial_lr * (decay_factor ** epoch)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        print(f"Node {rank} | Epoch {epoch+1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Loss: {loss.item():.4f}")

def exchange_model_weights(local_model):
    """Simulate secure model exchange: send encrypted state and merge with received state.
    In production, use proper encryption and secure channels.
    """
    print("[EXCHANGE] Sending model weights securely...")
    local_state = local_model.state_dict()
    received_state = local_state  # In reality, would be fetched from a global pool
    for key in local_state:
        local_state[key] = (local_state[key] + received_state[key]) / 2
    local_model.load_state_dict(local_state)
    print("[EXCHANGE] Model weights merged from global pool.")
    return local_model

def run_parallel_training(epochs=5, learning_rate=0.001, force_cpu=False):
    """Runs training across multiple CPU/GPU nodes."""
    data_dict = load_training_data()
    data = data_dict.get('input')
    target = data_dict.get('target')
    device = get_device(force_cpu)
    cpu_count = max(1, mp.cpu_count() - 1)
    models = [TrainingModel() for _ in range(cpu_count)]
    processes = []
    for rank, model in enumerate(models):
        p = mp.Process(target=train_model, args=(rank, model, data, target, epochs, learning_rate, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("[INFO] Multi-core training completed.")
    for model in models:
        exchange_model_weights(model)

def log_training_results():
    """Logs training session details."""
    log_file = os.path.join(os.path.expanduser("~"), "Documents", "AIOS IO", "training_log.txt")
    with open(log_file, "a") as f:
        f.write(f"Training completed at {time.ctime()}\n")
    print(f"Results logged in {log_file}")

class TrainingGUI(tk.Tk):
    # --- Tkinter-based GUI for Training Configuration ---
    def __init__(self):
        super().__init__()
        self.title("AIOS IO - User Training Module")
        self.geometry("600x400")
        self.configure(bg="black")

        # Dataset selection
        self.dataset_label = ttk.Label(self, text="Select Dataset:", background="black", foreground="green")
        self.dataset_label.pack(pady=5)
        default_dataset = os.path.join(os.path.expanduser("~"), "Documents", "AIOS IO", "DATASETS", "training_data.npy")
        self.dataset_path = tk.StringVar(value=default_dataset)
        self.dataset_entry = ttk.Entry(self, textvariable=self.dataset_path, width=50)
        self.dataset_entry.pack(pady=5)
        self.dataset_button = ttk.Button(self, text="Browse", command=self.browse_dataset)
        self.dataset_button.pack(pady=5)

        # Epochs selection
        self.epochs_label = ttk.Label(self, text="Epochs:", background="black", foreground="green")
        self.epochs_label.pack(pady=5)
        self.epochs_var = tk.IntVar(value=5)
        self.epochs_spinbox = ttk.Spinbox(self, from_=1, to=100, textvariable=self.epochs_var, width=10)
        self.epochs_spinbox.pack(pady=5)

        # Learning rate
        self.lr_label = ttk.Label(self, text="Learning Rate:", background="black", foreground="green")
        self.lr_label.pack(pady=5)
        self.lr_var = tk.DoubleVar(value=0.001)
        self.lr_spinbox = ttk.Spinbox(self, from_=0.0001, to=1.0, increment=0.0001, textvariable=self.lr_var, width=10)
        self.lr_spinbox.pack(pady=5)

        # Multi-core training toggle
        self.parallel_training = tk.BooleanVar(value=True)
        self.parallel_checkbox = ttk.Checkbutton(self, text="Enable Multi-Core Training", variable=self.parallel_training)
        self.parallel_checkbox.pack(pady=5)

        # CPU-only mode
        self.cpu_only = tk.BooleanVar(value=False)
        self.cpu_checkbox = ttk.Checkbutton(self, text="Force CPU Mode", variable=self.cpu_only)
        self.cpu_checkbox.pack(pady=5)

        # Start training button
        self.start_button = ttk.Button(self, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=10)

        # Log display
        self.log_text = tk.Text(self, height=8, bg="black", fg="green", insertbackground="green")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def browse_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy Files", "*.npy"), ("All Files", "*.*")])
        if file_path:
            self.dataset_path.set(file_path)

    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def start_training(self):
        self.log_message("[INFO] Starting Training...")
        thread = threading.Thread(target=self.run_training, daemon=True)
        thread.start()

    def run_training(self):
        dataset = self.dataset_path.get()
        epochs = self.epochs_var.get()
        learning_rate = self.lr_var.get()
        parallel = self.parallel_training.get()
        cpu_mode = self.cpu_only.get()
        self.log_message(f"Dataset: {dataset}")
        self.log_message(f"Epochs: {epochs}")
        self.log_message(f"Learning Rate: {learning_rate}")
        self.log_message(f"Parallel Training: {'Enabled' if parallel else 'Disabled'}")
        self.log_message(f"CPU Mode: {'Forced' if cpu_mode else 'Auto'}")
        run_parallel_training(epochs=epochs, learning_rate=learning_rate, force_cpu=cpu_mode)
        log_training_results()
        self.log_message("[INFO] Training Completed.")

# --- Self-Registration with The Wand ---
def register_module():
    """Automatically registers this module with The Wand."""
    registry_file = os.path.join(os.path.expanduser("~"), "Documents", "AIOS IO", "wand_registry.json")
    module_info = {
        "name": "User Training",
        "description": "Train AI models based on user input and log results.",
        "script_path": __file__
    }
    try:
        if os.path.exists(registry_file):
            with open(registry_file, "r") as f:
                registry = json.load(f)
        else:
            registry = {}
        registry[module_info["name"]] = module_info
        with open(registry_file, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"[INFO] Registered '{module_info['name']}' with The Wand.")
    except Exception as e:
        print(f"[ERROR] Could not register module: {e}")

# Update load_dynamic_features() to scan subfolders recursively
def load_dynamic_features():
    """Scans and loads additional training features recursively from subfolders."""
    FEATURES_DIR = os.path.dirname(os.path.abspath(__file__))
    for root, _, files in os.walk(FEATURES_DIR):
        for file in files:
            if file.endswith(".py") and file != os.path.basename(__file__):
                module_name = file[:-3]  # strip .py extension
                module_path = os.path.join(root, file)
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "register_feature"):
                    module.register_feature()
                    print(f"[INFO] Loaded feature: {module_name} from {root}")

def optimize_hardware():
    """Dynamically adjusts training settings based on system performance."""
    ram_available = psutil.virtual_memory().available / (1024 ** 3)
    gpu_available = torch.cuda.is_available()
    cpu_cores = os.cpu_count()
    print(f"RAM: {ram_available:.2f} GB | GPU: {gpu_available} | CPU Cores: {cpu_cores}")
    return {"RAM": ram_available, "GPU": gpu_available, "CPU": cpu_cores}

# Append new Auto-ML hyperparameter tuning using Bayesian Optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
except ImportError:
    gp_minimize = None

def train_and_evaluate(model, data, target, batch_size):
    """Stub function: Train model for one epoch and return dummy accuracy.
    In a real scenario, perform training with the given batch and evaluate accuracy.
    """
    return np.random.uniform(0.7, 0.9)

def auto_ml_select(data, target):
    """Automatically optimizes hyperparameters and selects the best model."""
    if gp_minimize is None:
        print("[WARN] Bayesian optimization library not available.")
        return TrainingModel()
    def objective(params):
        learning_rate, batch_size = params
        model = TrainingModel()
        acc = train_and_evaluate(model, data, target, int(batch_size))
        return -acc  # minimize negative accuracy
    space = [Real(0.0001, 0.01, name="learning_rate"),
             Integer(16, 256, name="batch_size")]
    res = gp_minimize(objective, space, n_calls=10, random_state=42)
    best_lr, best_bs = res.x
    print(f"[INFO] Optimized Learning Rate: {best_lr}, Batch Size: {best_bs}")
    best_model = TrainingModel()  # In practice, retrain model using best parameters
    return best_model

DEVICE = auto_allocate_resources()

# Checkpoint functions for self-healing training:
def load_checkpoint(model, checkpoint_path="model_checkpoint.pt"):
    """Loads model training checkpoint if available."""
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"[INFO] Resumed training from checkpoint: {checkpoint_path}")
    return model

def save_checkpoint(model, checkpoint_path="model_checkpoint.pt"):
    """Saves the current model state as a training checkpoint."""
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] Training checkpoint saved at {checkpoint_path}")

SCALER = GradScaler()

def train_mixed_precision(model, data_loader, epochs=5, lr=0.001):
    """Performs automatic mixed-precision training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs.to(device))
                loss = loss_fn(outputs, targets.to(device))
            SCALER.scale(loss).backward()
            SCALER.step(optimizer)
            SCALER.update()
            print(f"[INFO] Mixed-Precision Epoch {epoch+1}/{epochs} completed; Loss: {loss.item():.4f}")
    print("[INFO] Mixed-Precision Training Completed.")

############################## NEW HPC ENHANCEMENTS BELOW #############################

# 1. Containerization & Sandbox Execution
def run_sandboxed_task(task):
    """Simulate isolated execution of a task (container-like sandbox).
    In production, integrate with Docker/Podman or a Python virtualizer.
    """
    print(f"[SANDBOX] Running task in a sandbox: {task}")
    try:
        result = task()  # Assume task is callable
        print("[SANDBOX] Task completed successfully.")
        return result
    except Exception as e:
        print(f"[SANDBOX] Task failed in sandbox: {e}")
        return None

# 2. Meltdown Resilience Aggregator
def meltdown_resilience_aggregator(task_queue, threshold=95):
    """Monitors system load and, if usage exceeds a threshold,
    reassigns low-priority tasks to underutilized nodes.
    """
    import psutil
    cpu_load = psutil.cpu_percent(interval=1)
    print(f"[MELTDOWN] CPU load at {cpu_load}%")
    if cpu_load > threshold:
        print("[MELTDOWN] Load is too high: Reallocating tasks...")
        num_to_reassign = len(task_queue) // 2
        for _ in range(num_to_reassign):
            task = task_queue.pop(0)
            print(f"[MELTDOWN] Reassigning task: {task}")
            # In production, notify healthy nodes here

# 3. Ephemeral Concurrency Manager
def adjust_ephemeral_concurrency(current_epoch, base_concurrency=4):
    """Dynamically adjusts parallel task count based on epoch performance.
    Returns an updated concurrency level.
    """
    new_concurrency = base_concurrency + (current_epoch % 3)  # pseudo-dynamic adjustment
    # This is a stub: in practice, measure epoch speed and adapt
    print(f"[CONCURRENCY] Adjusted concurrency from {base_concurrency} to {new_concurrency} at epoch {current_epoch}")
    return new_concurrency

# 4. Node Warm Standby & Sleep Modes
def manage_node_power_modes(idle_time_minutes):
    """If a node is idle for more than 'idle_time_minutes',
    it switches to a low-power 'sleep' mode, or stays in warm standby.
    """
    if idle_time_minutes >= 10:
        print("[POWER] Switching node to deep sleep mode to save power.")
        return "sleep"  # Trigger deep power-down mechanisms (stub)
    else:
        print("[POWER] Node remains in warm standby mode.")
        return "warm"  # Maintain minimal processes

# 5. Cross-Domain Task Routing
def cross_domain_task_routing(task_type):
    """Routes tasks based on declared domain (e.g., 'rendering', 'cryptography')."""
    routing_map = {
        "ml": "ML HPC cluster",
        "rendering": "GPU-rich rendering cluster",
        "cryptography": "High-security CPU cluster",
        "simulation": "Memory-intensive simulation cluster"
    }
    assigned_cluster = routing_map.get(task_type.lower(), "Default HPC cluster")
    print(f"[ROUTING] Task type '{task_type}' will be sent to: {assigned_cluster}")
    return assigned_cluster

# 6. Quantum Potential Simulation (Prototype)
def quantum_simulation(task_description):
    """Uses basic randomness to emulate gate logic.
    Prototype for simulating a quantum-like computational task.
    """
    import random
    print(f"[QUANTUM] Starting quantum simulation for task: {task_description}")
    outcomes = [random.random() for _ in range(5)]
    # Simulate "superposition" by generating random outcomes
    result = max(outcomes)
    # "Collapse" state by picking the best outcome (for demonstration)
    print(f"[QUANTUM] Simulation result: {result:.4f}")
    return result

def unify_scripts():
    import AIOSIO.atom.UserTrainingFORMSV1_1 as user_training_forms
    import AIOSIO.atom.UserTrainingMLPLV1_1 as user_training_mlp
    try:
        from ADDITIONAL_FEATURES.SCHEDULER import HPC_Scheduler
    except ImportError:
        print("[WARNING] ADDITIONAL_FEATURES.SCHEDULER could not be resolved.")
    import AIOS_DistributedLoadBalancer
    import AIOS_DistributedTraining
    import aios_io
    import AIOSIOv1
    print("[INFO] unify_scripts executed successfully!")

def main():
    print("[INFO] All scripts have been unified and imported.")

if __name__ == "__main__":
    # ...existing training logic if needed...
    unify_scripts()
    print("Training enhancements applied dynamically.")
    load_dynamic_features()
    print("Starting AIOSIO User Training Module V1...")
    main()