import os
import h5py
import torch
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from queue import Queue
import threading
import time

class MLFileManager:
    """Manages ML file creation, loading, and synchronization"""
    def __init__(self, config: Dict):
        self.config = config
        self.ml_dir = Path(config.get('ml_directory', 'ml_files'))
        self.ml_dir.mkdir(parents=True, exist_ok=True)
        
        # ML file paths
        self.perception_path = self.ml_dir / "ileices_perception.pt"
        self.processing_path = self.ml_dir / "ileices_processing.h5"
        self.generation_path = self.ml_dir / "ileices_generation.pkl"
        
        # Knowledge state
        self.current_knowledge = self._initialize_knowledge()
        self.knowledge_queue = Queue()
        
        # Monitoring
        self.logger = logging.getLogger('MLFileManager')
        self.update_interval = 300  # 5 minutes
        self._start_monitoring()
        
    def create_ml_files(self, knowledge: Dict):
        """Generate ML files from current AI knowledge"""
        structured = self._structure_knowledge(knowledge)
        
        try:
            # Save perception model
            torch.save(structured['perception'], self.perception_path)
            
            # Save processing model
            with h5py.File(self.processing_path, 'w') as f:
                for key, value in structured['processing'].items():
                    f.create_dataset(key, data=value)
                    
            # Save generation model
            with open(self.generation_path, 'wb') as f:
                pickle.dump(structured['generation'], f)
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to create ML files: {e}")
            return False
            
    def load_ml_files(self) -> Dict:
        """Load and integrate existing ML files"""
        knowledge = {
            'perception': {},
            'processing': {},
            'generation': {}
        }
        
        try:
            # Load perception model if exists
            if self.perception_path.exists():
                knowledge['perception'] = torch.load(self.perception_path)
                
            # Load processing model if exists
            if self.processing_path.exists():
                with h5py.File(self.processing_path, 'r') as f:
                    knowledge['processing'] = {k: f[k][:] for k in f.keys()}
                    
            # Load generation model if exists
            if self.generation_path.exists():
                with open(self.generation_path, 'rb') as f:
                    knowledge['generation'] = pickle.load(f)
                    
            return knowledge
        except Exception as e:
            self.logger.error(f"Failed to load ML files: {e}")
            return knowledge
            
    def _structure_knowledge(self, knowledge: Dict) -> Dict:
        """Structure knowledge following Law of Three"""
        return {
            "perception": {
                "sensory": knowledge.get("perception", {}).get("raw", {}),
                "patterning": knowledge.get("perception", {}).get("processed", {}),
                "structuring": knowledge.get("perception", {}).get("final", {})
            },
            "processing": {
                "error_refinement": knowledge.get("processing", {}).get("errors", {}),
                "optimization": knowledge.get("processing", {}).get("improvements", {}),
                "recursive_compression": knowledge.get("processing", {}).get("finalized", {})
            },
            "generation": {
                "idea_expansion": knowledge.get("generation", {}).get("concepts", {}),
                "model_evolution": knowledge.get("generation", {}).get("enhancements", {}),
                "intelligence_synthesis": knowledge.get("generation", {}).get("finalized", {})
            }
        }
        
    def _initialize_knowledge(self) -> Dict:
        """Initialize empty knowledge structure"""
        return {
            'perception': {
                'raw': {},
                'processed': {},
                'final': {}
            },
            'processing': {
                'errors': {},
                'improvements': {},
                'finalized': {}
            },
            'generation': {
                'concepts': {},
                'enhancements': {},
                'finalized': {}
            }
        }
        
    def _start_monitoring(self):
        """Start ML file monitoring thread"""
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        
    def _monitor_loop(self):
        """Monitor and update ML files periodically"""
        while True:
            try:
                # Process any new knowledge updates
                while not self.knowledge_queue.empty():
                    knowledge = self.knowledge_queue.get()
                    self._update_knowledge(knowledge)
                    
                # Save current state to ML files
                if self.current_knowledge:
                    self.create_ml_files(self.current_knowledge)
                    
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                
            time.sleep(self.update_interval)
            
    def _update_knowledge(self, new_knowledge: Dict):
        """Update knowledge state following Law of Three"""
        for domain in ['perception', 'processing', 'generation']:
            if domain in new_knowledge:
                self._recursive_update(
                    self.current_knowledge[domain],
                    new_knowledge[domain]
                )
                
    def _recursive_update(self, existing: Dict, new_data: Dict):
        """Recursively update knowledge while maintaining structure"""
        for key, value in new_data.items():
            if key in existing:
                if isinstance(value, dict):
                    self._recursive_update(existing[key], value)
                else:
                    existing[key] = self._merge_values(existing[key], value)
                    
    def _merge_values(self, old_val, new_val):
        """Merge values following Law of Three"""
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            return (old_val * 2 + new_val) / 3  # Weighted average
        return new_val  # Default to new value for non-numeric types

class MLFileControlPanel:
    """User interface for ML file management"""
    def __init__(self, master, ml_manager: MLFileManager):
        self.window = tk.Toplevel(master)
        self.window.title("ML File Control Panel")
        self.window.geometry("400x300")
        self.ml_manager = ml_manager
        
        # Create controls
        self._create_controls()
        
    def _create_controls(self):
        """Create control panel elements"""
        # Use existing files toggle
        self.use_existing = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.window,
            text="Use Existing ML Files",
            variable=self.use_existing,
            command=self._toggle_ml_files
        ).pack(pady=5)
        
        # Generate new files button
        tk.Button(
            self.window,
            text="Generate New ML Files",
            command=self._generate_new_files
        ).pack(pady=5)
        
        # Load existing files button
        tk.Button(
            self.window,
            text="Load Existing ML Files",
            command=self._load_existing_files
        ).pack(pady=5)
        
        # Status display
        self.status_label = tk.Label(self.window, text="Status: Ready")
        self.status_label.pack(pady=10)
        
    def _toggle_ml_files(self):
        """Toggle use of ML files"""
        if not self.use_existing.get():
            self.status_label.config(text="Status: Using fresh learning only")
        else:
            self._load_existing_files()
            
    def _generate_new_files(self):
        """Generate new ML files"""
        success = self.ml_manager.create_ml_files(
            self.ml_manager.current_knowledge
        )
        status = "Generated" if success else "Failed to generate"
        self.status_label.config(text=f"Status: {status} new ML files")
        
    def _load_existing_files(self):
        """Load existing ML files"""
        knowledge = self.ml_manager.load_ml_files()
        has_knowledge = any(knowledge.values())
        status = "Loaded" if has_knowledge else "No"
        self.status_label.config(text=f"Status: {status} existing ML files")
