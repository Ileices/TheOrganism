import threading
import numpy as np
from pathlib import Path

class FederatedLearning:
    def __init__(self, config):
        self.config = config
        self.model_updates = []
        self.aggregation_lock = threading.Lock()
        
    def initialize_model(self, model_architecture):
        """Initialize the federated learning model"""
        self.model_architecture = model_architecture
        self.global_model = self._create_model(model_architecture)
        
    def receive_update(self, client_update):
        """Receive and store model updates from clients"""
        with self.aggregation_lock:
            self.model_updates.append(client_update)
            if len(self.model_updates) >= self.config.get('min_updates_for_aggregation', 3):
                self._aggregate_updates()
    
    def _create_model(self, architecture):
        """Create a new model instance based on architecture"""
        try:
            import torch
            import torch.nn as nn
            # Create PyTorch model from architecture
            return nn.Sequential(*architecture)
        except ImportError:
            print("PyTorch not available. Using simplified model structure.")
            return {}
    
    def _aggregate_updates(self):
        """Aggregate model updates using federated averaging"""
        try:
            updates = np.array(self.model_updates)
            aggregated = np.mean(updates, axis=0)
            self._update_global_model(aggregated)
            self.model_updates.clear()
        except Exception as e:
            print(f"Aggregation error: {e}")
    
    def _update_global_model(self, aggregated_update):
        """Update the global model with aggregated parameters"""
        try:
            for param, update in zip(self.global_model.parameters(), aggregated_update):
                param.data = update.clone()
        except Exception as e:
            print(f"Model update error: {e}")

def initiate_federated_learning(node_list: list):
    print("Initiating federated learning across nodes:")
    for node in node_list:
        print(f" - Node: {node}")
    return "Federated learning started."

def aggregate_models(models: list):
    print("Aggregating models from nodes...")
    return {"aggregated_model": "stub_aggregated"}