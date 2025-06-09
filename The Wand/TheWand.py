#!/usr/bin/env python3
"""
autoGPTBuilderGUI.py

An all-in-one Tkinter GUI application that:
1. Lets you paste or load a text file containing multiple JSON objects (defining build steps).
2. Extracts these JSON blocks, displays them, and allows building (files & folders) from their instructions.
3. Provides a machine learning "pipeline" placeholder with adjustable CPU usage, learning rate, etc.
4. Offers a theme toggle (dark theme vs. basic/default).
5. Allows 24/7 "learning" mode and repeatedly building the project to "improve" over time (stub demonstration).
6. Contains a minimal "chat" interface stub.
7. **NEW:** Provides a button to open a window with a built‑in prompt that instructs ChatGPT to output in the exact JSON schema required for auto‑build continuity.

Color and style requirements:
- Background: black
- Panel borders: dark red
- Text: green
- Buttons: dark red background with white text (active state: dark orange)
- A gold footer label with soft red glow at the bottom links to YouTube.

Menus:
- File: "Save Project", "Load Project", "Exit"
- Edit: "Switch Theme", "Machine Learning Config Form"
- Help: "How to Use", "About"

Copyright:
"The God Factory | Project Ileices | ASIO IO | ProPro | Roswan Miller"
  (gold text, soft red glow, clickable → opens "https://youtube.com/thegodfactory")

Author: ChatGPT (Demonstration)
"""

import os
import re
import json
import ast
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import webbrowser
import threading
import time
import random
import psutil  # New dependency: pip install psutil
from tqdm import tqdm
import numpy as np
from queue import Queue
import subprocess  # <-- New import for running external scripts
import glob  # New import for recursive file scan
import importlib.util

# Update module imports to point to the wand_modules package
import wand_modules.wand_chatbot as wand_chatbot
import wand_modules.wand_about as wand_about
import wand_modules.wand_howto as wand_howto

# Optional: Import ML libraries for demonstration purposes
try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    SGDClassifier = None
    CountVectorizer = None

# =============================================================================
# Built-In Prompt Text
# =============================================================================
BUILT_IN_PROMPT = r'''{
  "step_number": 1,
  "description": "Initialize basic project structure",
  "tasks": [
    {
      "action": "create_file",
      "path": "wand_modules/module_template.py",
      "content": """import logging
from typing import Dict, Optional

class WandModule:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
    def initialize(self) -> bool:
        \"\"\"Initialize module\"\"\"
        try:
            self.logger.info('Initializing module')
            return True
        except Exception as e:
            self.logger.error(f'Failed to initialize: {e}')
            return False
"""
    },
    {
      "action": "create_file", 
      "path": "README.md",
      "content": """# The Wand Automation System

A Python-based automation system with:
- JSON-based build steps
- AI-assisted optimization
- Plugin architecture
- Real-time monitoring
- Dark-themed GUI interface

## Usage
1. Paste or load JSON build instructions
2. Select and execute build steps
3. Monitor progress in logs
4. Use AI chat for assistance

## Build Step Format
```json
{
  "step_number": 1,
  "description": "Step description",
  "tasks": [
    {
      "action": "create_file|update_file|install_dependency",
      "path": "relative/path/to/file",
      "content": "file content"
    }
  ]
}
```
"""
    }
  ],
  "dependencies": [
    "Python 3.9+",
    "tkinter",
    "logging"
  ],
  "context_tracking": {
    "supported_actions": [
      "create_file",
      "update_file", 
      "install_dependency"
    ],
    "next_steps": [
      "Add module functionality",
      "Create build pipeline",
      "Set up monitoring"
    ]
  }
}'''

# =============================================================================
# Main Application Class
# =============================================================================
class AutoGPTBuilderGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # ----------------------------------
        # Basic State
        # ----------------------------------
        self.title("Auto GPT Builder GUI")
        self.geometry("1200x800")

        # The dark theme is default; user can switch to basic
        self.is_dark_theme = True

        self.pasted_text_content = ""  # For storing user-pasted text
        self.json_objects = []         # The extracted JSON steps
        self.combined_json = {}
        self.base_directory = os.getcwd()

        # New: Dynamically find AIOS IO directory and set additional features path
        self.aios_io_path = self._find_aios_io_directory()
        self.additional_features_path = os.path.join(self.aios_io_path, "ADDITIONAL FEATURES")

        # ML & "Chat" placeholders
        self.is_24_7_learning = False
        self.cpu_usage_limit = 1.0   # Placeholder scale 0..1
        self.learning_rate = 0.001
        self.epochs = 5
        self.model_objective = "Improve auto-building and NLP chat"
        self.auto_build_enabled = False

        # "Chat" logs
        self.chat_log = []

        # Action handlers dispatch dictionary
        self.action_handlers = {
            'update_file': self._handle_update_file,
            'install_dependency': self._handle_install_dependency,
            'update_instructions': self._handle_update_instructions,
            'create_file': self._handle_create_file,
            'enhance context awareness': self._handle_enhance_context_awareness,
            'embed cosmic metadata': self._handle_embed_cosmic_metadata,
            'expand modular system design': self._handle_expand_modular_system_design,
        }

        # ----------------------------------
        # Create UI
        # ----------------------------------
        self._create_menu()
        self._create_main_layout()
        self._apply_dark_theme()  # Default theme

        # Call to update the plugins menu at startup
        self._scan_and_update_scripts_menu()

        # Start logs
        self._log("Welcome to the Auto GPT Builder GUI (Dark Theme).")

    # =============================================================================
    # MENU BAR
    # =============================================================================
    def _create_menu(self):
        menubar = tk.Menu(self, bg="black", fg="green", tearoff=False)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=False, bg="black", fg="green")
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_command(label="Load Project", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=False, bg="black", fg="green")
        edit_menu.add_command(label="Switch Theme", command=self.switch_theme)
        edit_menu.add_command(label="Machine Learning Config Form", command=self._open_ml_config_form)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=False, bg="black", fg="green")
        help_menu.add_command(label="How to Use", command=self._show_how_to_use)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        # New: Modular Features Menu
        self.features_menu = tk.Menu(menubar, tearoff=False)
        # First commands to refresh list and change path
        self.features_menu.add_command(label="Refresh Scripts", command=self._scan_and_update_scripts_menu)
        self.features_menu.add_command(label="Change Directory", command=self._change_features_directory)
        self.features_menu.add_separator()
        menubar.add_cascade(label="Modular Features", menu=self.features_menu)

        self.config(menu=menubar)

    # =============================================================================
    # MAIN LAYOUT
    # =============================================================================
    def _create_main_layout(self):
        container = ttk.Frame(self, padding=5)
        container.pack(fill=tk.BOTH, expand=True)

        # Left frame for inputs, buttons, and steps list
        left_frame = ttk.Frame(container, borderwidth=3, relief="groove")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # NEW: Built-In Prompt Button
        built_in_prompt_btn = ttk.Button(left_frame, text="Show Built-In GPT Prompt", command=self.show_built_in_prompt_window)
        built_in_prompt_btn.pack(pady=5, anchor="center")

        # 1) A scrolled text for "Paste JSON Text"
        label_paste = ttk.Label(left_frame, text="Paste JSON Text Here:")
        label_paste.pack(anchor="w")

        self.txt_paste = scrolledtext.ScrolledText(left_frame, height=10, wrap=tk.WORD, bg="black", fg="green", insertbackground="green")
        self.txt_paste.pack(fill=tk.BOTH, expand=False)
        parse_btn = ttk.Button(left_frame, text="Parse Pasted JSON", command=self.parse_pasted_json)
        parse_btn.pack(pady=5, anchor="center")

        # 2) A "Load from file" area
        file_btn_frame = ttk.Frame(left_frame)
        file_btn_frame.pack(fill=tk.X, expand=False, pady=5)
        load_file_btn = ttk.Button(file_btn_frame, text="Load .txt File", command=self.load_file_dialog)
        load_file_btn.pack(side=tk.LEFT, padx=5)
        self.lbl_loaded_file = ttk.Label(file_btn_frame, text="No file loaded.")
        self.lbl_loaded_file.pack(side=tk.LEFT)

        # 3) Steps list
        steps_frame = ttk.Frame(left_frame)
        steps_frame.pack(fill=tk.BOTH, expand=True)
        steps_label = ttk.Label(steps_frame, text="Extracted Steps:")
        steps_label.pack(anchor="w")
        self.steps_listbox = tk.Listbox(steps_frame, selectmode=tk.SINGLE, bg="black", fg="green", selectbackground="#800000", selectforeground="white")
        self.steps_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(steps_frame, command=self.steps_listbox.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.steps_listbox.config(yscrollcommand=sb.set)

        # 4) Buttons to build
        action_btn_frame = ttk.Frame(left_frame)
        action_btn_frame.pack(fill=tk.X, pady=5)
        btn_build_one = ttk.Button(action_btn_frame, text="Build Selected Step", command=self.build_selected_step)
        btn_build_one.pack(side=tk.LEFT, padx=5)
        btn_build_all = ttk.Button(action_btn_frame, text="Build All Steps", command=self.build_all_steps)
        btn_build_all.pack(side=tk.LEFT, padx=5)

        # 5) Combine / Save combined
        combine_btn_frame = ttk.Frame(left_frame)
        combine_btn_frame.pack(fill=tk.X, pady=5)
        btn_combine = ttk.Button(combine_btn_frame, text="Combine All JSON", command=self.combine_all_json)
        btn_combine.pack(side=tk.LEFT, padx=5)
        btn_save_combined = ttk.Button(combine_btn_frame, text="Save Combined", command=self.save_combined_json)
        btn_save_combined.pack(side=tk.LEFT, padx=5)

        # Right frame: Notebook with JSON Preview, Logs, and Chat tabs
        right_frame = ttk.Frame(container, borderwidth=3, relief="groove")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab: JSON Preview
        self.tab_preview = ttk.Frame(self.notebook, borderwidth=2, relief="groove")
        self.notebook.add(self.tab_preview, text="JSON Preview")
        self.txt_preview = scrolledtext.ScrolledText(self.tab_preview, wrap=tk.NONE, bg="black", fg="green", insertbackground="green")
        self.txt_preview.pack(fill=tk.BOTH, expand=True)

        # Tab: Logs
        self.tab_logs = ttk.Frame(self.notebook, borderwidth=2, relief="groove")
        self.notebook.add(self.tab_logs, text="Logs")
        self.txt_logs = scrolledtext.ScrolledText(self.tab_logs, wrap=tk.NONE, bg="black", fg="green", insertbackground="green")
        self.txt_logs.pack(fill=tk.BOTH, expand=True)

        # Tab: Chat (Placeholder)
        self.tab_chat = ttk.Frame(self.notebook, borderwidth=2, relief="groove")
        self.notebook.add(self.tab_chat, text="AI Chat")
        self._create_chat_tab(self.tab_chat)

        # Footer: Copyright
        self.footer_frame = ttk.Frame(self, borderwidth=3, relief="groove")
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self._create_copyright_label()

        # New: HPC Dashboard Tab
        self._create_hpc_dashboard_tab()

        # New: Global Network Visualization Tab
        self._create_global_network_tab()

    # New function: Show built-in prompt window
    def show_built_in_prompt_window(self):
        win = tk.Toplevel(self)
        win.title("Built-In GPT Prompt")
        win.geometry("800x400")
        win.configure(bg="black")
        lbl = tk.Label(win, text="Copy the prompt below and paste it into your ChatGPT thread.", bg="black", fg="green", font=("Courier", 12))
        lbl.pack(padx=10, pady=10)
        text_area = scrolledtext.ScrolledText(win, wrap=tk.WORD, bg="black", fg="green", insertbackground="green", font=("Courier", 10))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_area.insert(tk.END, BUILT_IN_PROMPT)
        # Make text_area read-only
        text_area.config(state=tk.NORMAL)
        copy_btn = ttk.Button(win, text="Copy to Clipboard", command=lambda: self.copy_to_clipboard(BUILT_IN_PROMPT))
        copy_btn.pack(pady=5)
        close_btn = ttk.Button(win, text="Close", command=win.destroy)
        close_btn.pack(pady=5)

    def copy_to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        self._log("Built-In GPT Prompt copied to clipboard.")

    import wand_chatbot  # Original chat module import restored exactly as it appeared in the GUI before refactoring

    # =============================================================================
    # CHAT TAB CREATION (Same as before)
    # =============================================================================
    def _create_chat_tab(self, parent):
        # Delegate creation of chat tab to wand_chatbot module
        wand_chatbot.create_chat_tab(self, parent)

    def _open_prompt_tuner(self):
        # Delegate opening of prompt tuner to wand_chatbot module
        wand_chatbot.open_prompt_tuner(self)

    # =============================================================================
    # COPYRIGHT LABEL
    # =============================================================================
    def _create_copyright_label(self):
        def open_link(event):
            webbrowser.open("https://youtube.com/thegodfactory")
        lbl = tk.Label(
            self.footer_frame,
            text="The God Factory | Project Ileices | ASIO IO | ProPro | Roswan Miller",
            fg="gold",
            bg="black",
            cursor="hand2",
            font=("Arial", 10, "bold")
        )
        lbl.pack(side=tk.LEFT, padx=10)
        lbl.bind("<Button-1>", open_link)

    # =============================================================================
    # THEME MANAGEMENT (Same as before)
    # =============================================================================
    def switch_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        if self.is_dark_theme:
            self._apply_dark_theme()
            self._log("Switched to dark theme.")
        else:
            self._apply_basic_theme()
            self._log("Switched to basic (default) theme.")

    def _apply_dark_theme(self):
        self.configure(bg="black")
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure(".", background="black", foreground="green")
        style.configure("TFrame", background="black")
        style.configure("TLabel", background="black", foreground="green")
        style.configure("TButton", background="dark red", foreground="white")
        style.configure("TCheckbutton", background="black", foreground="green")
        style.configure("TRadiobutton", background="black", foreground="green")
        style.map("TButton", foreground=[("active", "white"), ("disabled", "gray")],
                  background=[("active", "#800000")])
        self._update_tk_widgets_bg_fg("black", "green")

    def _apply_basic_theme(self):
        style = ttk.Style(self)
        style.theme_use("default")
        self.configure(bg=None)
        self._update_tk_widgets_bg_fg(None, None)

    def _update_tk_widgets_bg_fg(self, bg, fg):
        for widget in self.winfo_children():
            self._recursive_update_tk(widget, bg, fg)

    def _recursive_update_tk(self, widget, bg, fg):
        if isinstance(widget, (tk.Frame, tk.LabelFrame)):
            if bg:
                widget.configure(bg=bg)
        elif isinstance(widget, tk.Label):
            if bg:
                widget.configure(bg=bg)
            if fg:
                widget.configure(fg=fg)
        elif isinstance(widget, tk.Button):
            if bg:
                widget.configure(bg="dark red", fg="white")
        elif isinstance(widget, scrolledtext.ScrolledText):
            if bg:
                widget.configure(bg="black", fg="green", insertbackground="green")
        for child in widget.winfo_children():
            self._recursive_update_tk(child, bg, fg)

    # =============================================================================
    # MENU ACTIONS: SAVE/LOAD/EXIT (Same as before)
    # =============================================================================
    def save_project(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                              filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        data = {
            "json_objects": self.json_objects,
            "combined_json": self.combined_json,
            "ml_config": {
                "cpu_usage_limit": self.cpu_usage_limit,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "model_objective": self.model_objective
            },
            "auto_build_enabled": self.auto_build_enabled
        }
        try:
            with open(path, "w", encoding="utf-8") as f:  # Removed extra ')'
                f.write(json.dumps(data, indent=2))
            self._log(f"Project saved to {path}")
        except Exception as e:
            self._log(f"[ERROR] Failed to save project: {e}")

    def load_project(self):
        path = filedialog.askopenfilename(defaultextension=".json",
                                          filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.json_objects = data.get("json_objects", [])
            self.combined_json = data.get("combined_json", {})
            ml_config = data.get("ml_config", {})
            self.cpu_usage_limit = ml_config.get("cpu_usage_limit", 1.0)
            self.learning_rate = ml_config.get("learning_rate", 0.001)
            self.epochs = ml_config.get("epochs", 5)
            self.model_objective = ml_config.get("model_objective", "Improve auto-building and NLP chat")
            self.auto_build_enabled = data.get("auto_build_enabled", False)
            self._refresh_steps_listbox()
            self._log(f"Project loaded from {path}")
        except Exception as e:
            self._log(f"[ERROR] Could not load project: {e}")

    def _on_exit(self):
        self.destroy()

    # =============================================================================
    # MENU ACTIONS: EDIT (ML Config Form and Theme switch - same as before)
    # =============================================================================
    def _open_ml_config_form(self):
        form = tk.Toplevel(self)
        form.title("Machine Learning Configuration")
        form.geometry("400x550")
        form.configure(bg="black")
        label_info = tk.Label(form, text=(
            "Use this form to configure the ML pipeline.\n"
            "Adjust the parameters and see live ML status during training."
        ), fg="green", bg="black", wraplength=380, justify="left")
        label_info.pack(padx=10, pady=10)
        tk.Label(form, text="CPU Usage (0.0-1.0):", fg="green", bg="black").pack(anchor="w", padx=10)
        cpu_var = tk.DoubleVar(value=self.cpu_usage_limit)
        tk.Entry(form, textvariable=cpu_var, bg="black", fg="green").pack(fill=tk.X, padx=10, pady=5)
        tk.Label(form, text="Learning Rate:", fg="green", bg="black").pack(anchor="w", padx=10)
        lr_var = tk.DoubleVar(value=self.learning_rate)
        tk.Entry(form, textvariable=lr_var, bg="black", fg="green").pack(fill=tk.X, padx=10, pady=5)
        tk.Label(form, text="Epochs:", fg="green", bg="black").pack(anchor="w", padx=10)
        epochs_var = tk.IntVar(value=self.epochs)
        tk.Entry(form, textvariable=epochs_var, bg="black", fg="green").pack(fill=tk.X, padx=10, pady=5)
        tk.Label(form, text="Model Objective (what to learn):", fg="green", bg="black").pack(anchor="w", padx=10)
        obj_var = tk.StringVar(value=self.model_objective)
        tk.Entry(form, textvariable=obj_var, bg="black", fg="green").pack(fill=tk.X, padx=10, pady=5)
        t7_var = tk.BooleanVar(value=self.is_24_7_learning)
        tk.Checkbutton(form, text="Enable 24/7 Learning", variable=t7_var, bg="black", fg="green").pack(anchor="w", padx=10)
        ab_var = tk.BooleanVar(value=self.auto_build_enabled)
        tk.Checkbutton(form, text="Enable Auto-Build", variable=ab_var, bg="black", fg="green").pack(anchor="w", padx=10)
        # New: Live ML status label
        self.ml_status_label = tk.Label(form, text="ML Status: Idle", fg="gold", bg="black")
        self.ml_status_label.pack(pady=5)
        def save_config():
            try:
                cpu_val = cpu_var.get()
                if not (0.0 <= cpu_val <= 1.0):
                    raise ValueError("CPU Usage must be between 0.0 and 1.0")
                self.cpu_usage_limit = cpu_val
            except Exception as e:
                messagebox.showerror("Invalid Input", f"Invalid CPU Usage: {e}")
                return
            try:
                lr_val = lr_var.get()
                if lr_val <= 0:
                    raise ValueError("Learning Rate must be positive")
                self.learning_rate = lr_val
            except Exception as e:
                messagebox.showerror("Invalid Input", f"Invalid Learning Rate: {e}")
                return
            try:
                epochs_val = epochs_var.get()
                if epochs_val <= 0:
                    raise ValueError("Epochs must be positive")
                self.epochs = epochs_val
            except Exception as e:
                messagebox.showerror("Invalid Input", f"Invalid Epochs: {e}")
                return
            self.model_objective = obj_var.get()
            self.is_24_7_learning = t7_var.get()
            self.auto_build_enabled = ab_var.get()
            self._log(f"ML config updated:\n   CPU={self.cpu_usage_limit}, LR={self.learning_rate}, Epochs={self.epochs}, 24/7={self.is_24_7_learning}, AutoBuild={self.auto_build_enabled}")
            if self.is_24_7_learning:
                self._start_24_7_learning()
            self._update_ml_status("Config Saved")
            form.destroy()
        tk.Button(form, text="Save Config", bg="dark red", fg="white", command=save_config).pack(pady=10)

    def _show_how_to_use(self):
        def _show_how_to_use(self):
            try:
                wand_howto.show_how_to_use()
            except ImportError:
                messagebox.showerror("Error", "The how-to module (wand_howto.py) could not be found.")

    def _show_about(self):
        wand_about.show_about()

    # =============================================================================
    # LOADING / PARSING JSON (Same as before)
    # =============================================================================
    def load_file_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not path:
            return
        self.lbl_loaded_file.config(text=os.path.basename(path))
        self._extract_json_from_file(path)

    def _extract_json_from_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            self._log(f"[ERROR] Could not read {path}: {e}")
            return
        new_objs = self._extract_json_blocks(text)
        self.json_objects.extend(new_objs)
        self._log(f"Loaded {len(new_objs)} new JSON object(s) from {path}.")
        self._refresh_steps_listbox()

    def parse_pasted_json(self):
        raw_text = self.txt_paste.get("1.0", tk.END)
        new_objs = self._extract_json_blocks(raw_text)
        self.json_objects.extend(new_objs)
        self._log(f"Parsed {len(new_objs)} new JSON object(s) from pasted text.")
        self._refresh_steps_listbox()

    def _extract_json_blocks(self, text):
        blocks = []
        brace_stack = []
        start_index = None
        for i, ch in enumerate(text):
            if ch == '{':
                brace_stack.append(i)
                if len(brace_stack) == 1:
                    start_index = i
            elif ch == '}':
                if brace_stack:
                    brace_stack.pop()
                    if len(brace_stack) == 0 and start_index is not None:
                        block = text[start_index: i+1]
                        blocks.append(block)
                        start_index = None
        results = []
        for blk in blocks:
            blk = blk.strip()
            try:
                parsed = json.loads(blk)
                results.append(parsed)
            except json.JSONDecodeError:
                self._log(f"[WARN] Failed to parse JSON block: {blk[:50]}...")
        return results

    def _refresh_steps_listbox(self):
        self.steps_listbox.delete(0, tk.END)
        for i, obj in enumerate(self.json_objects):
            step_num = obj.get("step_number", "??")
            desc = obj.get("description", "")
            short_desc = (desc[:50] + "...") if len(desc) > 50 else desc
            label = f"{i}) step={step_num}, {short_desc}"
            self.steps_listbox.insert(tk.END, label)

    # =============================================================================
    # BUILD ACTIONS (Same as before)
    # =============================================================================
    def build_selected_step(self):
        sel = self.steps_listbox.curselection()
        if not sel:
            self._log("[WARN] No step selected.")
            return
        index = sel[0]
        if 0 <= index < len(self.json_objects):
            step_data = self.json_objects[index]
            self._build_from_json(step_data)

    def build_all_steps(self):
        if not self.json_objects:
            self._log("[WARN] No JSON objects to build.")
            return
        self.task_queue = Queue()
        for step in self.json_objects:
            self.task_queue.put(step)
        self._log(f"[AI] Build queue initialized with {self.task_queue.qsize()} steps.")
        worker_thread = threading.Thread(target=self._execute_task_queue, daemon=True)
        worker_thread.start()

    def _execute_task_queue(self):
        while not self.task_queue.empty():
            step_data = self.task_queue.get()
            try:
                self._build_from_json(step_data)
                self._log(f"[SUCCESS] Built step: {step_data.get('step_number', 'Unknown')}")
            except Exception as e:
                self._log(f"[ERROR] Failed to build step: {e}")
            self.task_queue.task_done()

    def _build_from_json(self, json_data):
        step_number = json_data.get("step_number", "??")
        desc = json_data.get("description", "")
        tasks = json_data.get("tasks", [])
        self._log(f"\n=== Build Step {step_number}: {desc} ===")
        for i, task in enumerate(tasks, start=1):
            action = task.get("action", "").lower()
            path = task.get("path", "")
            content = task.get("content", "")
            self._log(f" -> Task #{i}: {action} - {path}")
            handler = self.action_handlers.get(action)
            if handler:
                handler(path, content)
            else:
                self._log(f"[WARN] Unknown action '{action}'. Skipping...")
        self._log(f"-> Building step #{step_number}")

    # =============================================================================
    # ACTION HANDLERS (Same as before)
    # =============================================================================
    def _handle_update_file(self, path, content):
        full_path = os.path.join(self.base_directory, path)
        dir_part = os.path.dirname(full_path)
        if dir_part and not os.path.exists(dir_part):
            try:
                os.makedirs(dir_part, exist_ok=True)
                self._log(f"    Created directory: {dir_part}")
            except Exception as e:
                self._log(f"[ERROR] Could not create directory {dir_part}: {e}")
                return
        if full_path.endswith(".py"):
            new_code, success = self._auto_fix_python_content(content)
            if not success:
                self._log("    [WARN] Auto-fix failed. Skipping .py update.")
                return
            try:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(new_code)
                self._log(f"    Updated Python file: {full_path}")
            except Exception as e:
                self._log(f"[ERROR] Failed to write {full_path}: {e}")
        else:
            try:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self._log(f"    Updated file: {full_path}")
            except Exception as e:
                self._log(f"[ERROR] Failed to write {full_path}: {e}")

    def _handle_install_dependency(self, path, content):
        full_path = os.path.join(self.base_directory, path)
        dir_part = os.path.dirname(full_path)
        if dir_part and not os.path.exists(dir_part):
            try:
                os.makedirs(dir_part, exist_ok=True)
                self._log(f"    Created directory: {dir_part}")
            except Exception as e:
                self._log(f"[ERROR] Could not create directory {dir_part}: {e}")
                return
        try:
            dependencies = content.strip().split('\n')
            with open(full_path, "a", encoding="utf-8") as f:
                for dep in dependencies:
                    dep_clean = dep.strip()
                    if dep_clean:
                        f.write(dep_clean + "\n")
                        self._log(f"    Appended dependency '{dep_clean}' to {full_path}")
        except Exception as e:
            self._log(f"[ERROR] Failed to append dependency to {full_path}: {e}")

    def _handle_update_instructions(self, path, content):
        self._handle_update_file(path, content)

    def _handle_create_file(self, path, content):
        full_path = os.path.join(self.base_directory, path)
        dir_part = os.path.dirname(full_path)
        if dir_part and not os.path.exists(dir_part):
            try:
                os.makedirs(dir_part, exist_ok=True)
                self._log(f"    Created directory: {dir_part}")
            except Exception as e:
                self._log(f"[ERROR] Could not create directory {dir_part}: {e}")
                return
        if not os.path.exists(full_path):
            try:
                with open(full_path, "w", encoding="utf-8") as (f):
                    f.write(content)
                self._log(f"    Created file: {full_path}")
            except Exception as e:
                self._log(f"[ERROR] Failed to create file {full_path}: {e}")
        else:
            self._log(f"    File already exists: {full_path}")

    def _handle_enhance_context_awareness(self, path, content):
        self._log("    [INFO] Enhancing context awareness (stub implementation).")
        self._handle_update_file(path, content)

    def _handle_embed_cosmic_metadata(self, path, content):
        self._log("    [INFO] Embedding cosmic metadata (stub implementation).")
        self._handle_update_file(path, content)

    def _handle_expand_modular_system_design(self, path, content):
        self._log("    [INFO] Expanding modular system design (stub implementation).")
        self._handle_update_file(path, content)

    # =============================================================================
    # COMBINE / SAVE COMBINED (Same as before)
    # =============================================================================
    def combine_all_json(self):
        self.combined_json = {"all_steps": self.json_objects}
        self._log("All JSON combined into 'combined_json'.")
        self._preview_combined_json()

    def save_combined_json(self):
        if not self.combined_json:
            self._log("[WARN] No combined JSON to save. Please combine first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                              filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.combined_json, f, indent=2)
            self._log(f"Combined JSON saved to {path}")
        except Exception as e:
            self._log(f"[ERROR] Failed to save combined JSON: {e}")

    def _preview_combined_json(self):
        self.txt_preview.config(state=tk.NORMAL)
        self.txt_preview.delete("1.0", tk.END)
        try:
            text = json.dumps(self.combined_json, indent=2)
            self.txt_preview.insert(tk.END, text)
        except Exception as e:
            self.txt_preview.insert(tk.END, f"[ERROR] Could not serialize combined JSON: {e}")
        self.txt_preview.config(state=tk.DISABLED)

    # =============================================================================
    # PYTHON CODE AUTO-FIX (Same as before)
    # =============================================================================
    def _auto_fix_python_content(self, code_str):
        def can_parse(c):
            try:
                ast.parse(c)
                return True
            except SyntaxError:
                return False
            except Exception:
                return False
        def fix_indentation(c):
            lines = c.splitlines()
            new_lines = []
            for line in lines:
                line = line.replace("\t", "    ")
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 20:
                    line = line.lstrip()
                new_lines.append(line)
            return "\n".join(new_lines)
        def remove_nonlocal(c):
            lines = c.splitlines()
            out = []
            for ln in lines:
                if "nonlocal " in ln:
                    out.append(f"# (commented) {ln}")
                else:
                    out.append(ln)
            return "\n".join(out)
        working = code_str
        for _ in range(5):
            if can_parse(working):
                return (working, True)
            working = fix_indentation(working)
            working = remove_nonlocal(working)
        if can_parse(working):
            return (working, True)
        return (code_str, False)

    # =============================================================================
    # LOGGING (Same as before)
    # =============================================================================
    def _log(self, message):
        self.txt_logs.config(state=tk.NORMAL)
        self.txt_logs.insert(tk.END, message + "\n")
        self.txt_logs.see(tk.END)
        self.txt_logs.config(state=tk.DISABLED)
        print(message)

    # =============================================================================
    # AI CHAT HANDLING (Same as before)
    # =============================================================================
    def _chat_send_message(self):
        user_msg = self.chat_input.get().strip()
        if not user_msg:
            return
        self.chat_input.set("")
        self._append_chat("You", user_msg)
        response = self._generate_ai_response(user_msg)
        self._append_chat("AI", response)

    def _append_chat(self, speaker, text):
        self.txt_chat_log.config(state=tk.NORMAL)
        self.txt_chat_log.insert(tk.END, f"{speaker}: {text}\n")
        self.txt_chat_log.see(tk.END)
        self.txt_chat_log.config(state=tk.DISABLED)

    def _generate_ai_response(self, prompt):
        placeholders = [
            "Interesting question! Let me think about it.",
            "Based on my current learning, I'd suggest you try building again.",
            "I'm still training, please ask me later!",
            "Sure, let's talk more about your project."
        ]
        return random.choice(placeholders)

    # =============================================================================
    # 24/7 LEARNING (Stub - Same as before)
    # =============================================================================
    def _start_24_7_learning(self):
        if self.is_24_7_learning:
            self._log("[AI] 24/7 Learning is already running.")
            return
        self.is_24_7_learning = True
        t = threading.Thread(target=self._background_learn_loop, daemon=True)
        t.start()
        self._log("[AI] Started 24/7 Learning.")

    def _background_learn_loop(self):
        while self.is_24_7_learning:
            cpu_load = psutil.cpu_percent(interval=1)
            if cpu_load > 80:  # Throttle learning when CPU is heavily loaded
                self._log(f"[AI] CPU Load High ({cpu_load}%). Pausing learning cycle...")
                time.sleep(30)
                continue
            self._log(f"[AI] CPU Load Normal ({cpu_load}%). Running learning cycle...")
            self._do_learning_cycle()
            if self.auto_build_enabled:
                self._log("[AI] Auto-building project after learning cycle...")
                self.build_all_steps()
            time.sleep(10)  # Delay configurable by the user
        self._log("[AI] 24/7 learning stopped or never started properly.")

    def _do_learning_cycle(self):
        self._update_ml_status("Training...")
        if SGDClassifier is None or CountVectorizer is None:
            self._log("[AI] ML libraries not available (sklearn). Skipping training.")
            self._update_ml_status("Idle")
            return

        # Prepare training data from JSON steps (mock example)
        texts, labels = [], []
        for i, step in enumerate(self.json_objects):
            desc = step.get("description", "")
            if desc:
                texts.append(desc)
                labels.append(i % 2)
        if not texts:
            self._log("[AI] No training data available.")
            self._update_ml_status("Idle")
            return
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        y = np.array(labels)
        model = SGDClassifier(alpha=self.learning_rate, max_iter=1, warm_start=True)
        
        self._log("[AI] Training model...")
        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            try:
                model.partial_fit(X, y, classes=np.unique(y))
                self._log(f"[AI] Epoch {epoch+1}/{self.epochs} completed. LR={self.learning_rate}")
            except Exception as e:
                self._log(f"[ERROR] Training failure: {e}")
                self._update_ml_status("Training Failed")
                return
        self._log("[AI] Training completed successfully!")
        self._update_ml_status("Training Complete")

    def _update_ml_status(self, status):
        if hasattr(self, "ml_status_label"):
            self.ml_status_label.config(text=f"ML Status: {status}")

    # =============================================================================
    # EXTENSIBILITY (Same as before)
    # =============================================================================
    def add_action_handler(self, action_name, handler_function):
        action_key = action_name.lower()
        self.action_handlers[action_key] = handler_function
        self._log(f"[INFO] Added new action handler for '{action_name}'.")

    # =============================================================================
    # CLEAN UP ON EXIT (Same as before)
    # =============================================================================
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.is_24_7_learning = False
            self.destroy()

    # New method: Find AIOS IO directory dynamically
    def _find_aios_io_directory(self):
        home = os.path.expanduser("~")
        potential_paths = [os.path.join(home, "Documents", "AIOS IO"), home]
        for path in potential_paths:
            if os.path.exists(os.path.join(path, "ADDITIONAL FEATURES")):
                return path
        # If not found, prompt user to select the AIOS IO directory
        messagebox.showinfo("Set Directory", "AIOS IO directory not auto-detected. Please select it manually.")
        return filedialog.askdirectory(title="Select AIOS IO Directory")

    # Replace _scan_and_update_scripts_menu() with the following:
    def _scan_and_update_scripts_menu(self):
        """Auto-detects and dynamically loads all feature scripts in AIOS IO without requiring manual edits."""
        self.features_menu.delete(3, tk.END)  # Keep first 3 menu items

        registry_file = os.path.join(os.path.expanduser("~"), "Documents", "AIOS IO", "wand_registry.json")
        additional_features_path = os.path.join(os.path.expanduser("~"), "Documents", "AIOS IO", "ADDITIONAL FEATURES")
        if not os.path.exists(additional_features_path):
            os.makedirs(additional_features_path)
        
        # Step 1: Scan through and auto-register unregistered scripts
        script_registry = {}
        if os.path.exists(registry_file):
            with open(registry_file, "r") as f:
                script_registry = json.load(f)
        for root, _, files in os.walk(additional_features_path):
            for file in files:
                if file.lower().endswith((".py", ".bat", ".sh")):
                    script_path = os.path.join(root, file)
                    if file not in script_registry:
                        try:
                            subprocess.Popen([sys.executable, script_path])
                            self._log(f"[INFO] Auto-registered new module: {file}")
                        except Exception as e:
                            self._log(f"[ERROR] Failed to auto-register {file}: {e}")
        # Step 2: Load modules from registry
        if os.path.exists(registry_file):
            try:
                with open(registry_file, "r") as f:
                    registry = json.load(f)
                for module_name, module_info in registry.items():
                    script_path = module_info["script_path"]
                    self.features_menu.add_command(
                        label=module_name,
                        command=lambda s=script_path: self._run_script(s)
                    )
                    self._log(f"Loaded module: {module_name}")
            except Exception as e:
                self._log(f"[ERROR] Failed to load modules: {e}")

    # Replace _run_script() with the following:
    def _run_script(self, script_path):
        """Runs a feature script dynamically without preloading it into memory."""
        try:
            if script_path.lower().endswith(".py"):
                spec = importlib.util.spec_from_file_location("module.name", script_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["module.name"] = module
                spec.loader.exec_module(module)
            elif script_path.lower().endswith(".bat"):
                subprocess.Popen(["cmd.exe", "/c", script_path],
                                 creationflags=subprocess.CREATE_NEW_CONSOLE)
            elif script_path.lower().endswith(".sh"):
                subprocess.Popen(["bash", script_path])
            else:
                messagebox.showerror("Error", "Unsupported file type.")
            self._log(f"[INFO] Executed feature: {script_path}")
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self._log(f"[ERROR] Failed to execute script: {e}")

    # New method: Allow the user to manually set the ADDITIONAL FEATURES directory
    def _change_features_directory(self):
        new_dir = filedialog.askdirectory(title="Select ADDITIONAL FEATURES Directory")
        if new_dir:
            self.additional_features_path = new_dir
            self._scan_and_update_scripts_menu()
            self._log(f"Updated ADDITIONAL FEATURES path to {new_dir}")

    # New method: Create HPC Dashboard Tab
    def _create_hpc_dashboard_tab(self):
        dashboard = ttk.Frame(self.notebook, borderwidth=2, relief="groove")
        self.notebook.add(dashboard, text="HPC Dashboard")
        node_list = tk.Listbox(dashboard, bg="black", fg="green", selectbackground="dark red")
        node_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Use real info if available; here we use a placeholder:
        cpu_count = psutil.cpu_count()
        mem = psutil.virtual_memory().total / (1024**3)  # in GB
        node_list.insert(tk.END, f"Local Node: {cpu_count} cores, {mem:.2f} GB RAM")
        stats_text = tk.Text(dashboard, height=10, bg="black", fg="green", insertbackground="green")
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        node_statuses = []  # Added placeholder for active node statuses
        stats_text.insert(tk.END, f"Global Stats:\nActive Nodes: {len(node_statuses)}\n")
        stats_text.insert(tk.END, f"Total RAM: {mem:.2f} GB\n")
        stats_text.config(state=tk.DISABLED)
        return dashboard

    # New method: Create Global Network Visualization Tab
    def _create_global_network_tab(self):
        network_tab = ttk.Frame(self.notebook, borderwidth=2, relief="groove")
        self.notebook.add(network_tab, text="Global Network")
        # Dummy visualization using a Listbox and Text widget
        device_list = tk.Listbox(network_tab, bg="black", fg="green", selectbackground="dark red")
        device_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        for device in ["Node1: 16GB, GPU: 3090", "Node2: 12GB, GPU: 3070", "Node3: Idle (CPU only)"]:
            device_list.insert(tk.END, device)
        stats_text = tk.Text(network_tab, height=10, bg="black", fg="green", insertbackground="green")
        stats_text.pack(fill=tk.BOTH, padx=5, pady=5)
        stats_text.insert(tk.END, "Global HPC Stats:\nDevices: 3\nTotal GPU Memory: 28GB\nTotal RAM: 48GB\n")
        stats_text.config(state=tk.DISABLED)
        return network_tab

# =============================================================================
# Main Entry Point
# =============================================================================
import json
import sys
from pathlib import Path
from wand_modules.wand_core import WandCore
from wand_modules.wand_setup import ensure_system_ready
from wandhelper import WandHelper, initialize_logging, create_default_config
from wand_modules.wand_ai import WandAI
from wand_modules.wand_builder import WandBuilder
from wand_modules.wand_monitor import WandMonitor
from wand_modules.wand_logger import WandLogger
from wand_modules.wand_config import WandConfig
from wand_modules.wand_plugin_manager import WandPluginManager

class TheWand:
    def __init__(self):
        """Initialize The Wand system with all required components."""
        # 1. Basic initialization and logging
        initialize_logging()
        
        # 2. Load or create default configuration
        config_path = Path(__file__).parent / 'wand_config.json'
        if not config_path.exists():
            print("Configuration not found. Creating default configuration...")
            config_dict = create_default_config()
            self.config = WandConfig(config_path)
            self.config.config = config_dict  # Set the config dict directly
        else:
            self.config = WandConfig(config_path)
            if not self.config.config:  # If config failed to load
                print("Configuration invalid. Creating default configuration...")
                config_dict = create_default_config()
                self.config.config = config_dict  # Set the config dict directly

        # Ensure we have valid configuration before continuing
        if not self.config.config:
            raise RuntimeError("Failed to initialize configuration")

        # 3. Initialize logging system
        log_dir = Path(self.config.config.get('log_directory', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = WandLogger(log_dir)
        self.logger.log_info("Initializing The Wand system...")

        # 4. Set up helper and monitoring
        self.helper = WandHelper(self)
        self.monitor = WandMonitor()
        
        # 5. Initialize builder and AI components
        self.builder = WandBuilder(self.config.config)
        self.ai = WandAI(self.config.config)
        
        # 6. Set up plugin system
        plugins_dir = Path(self.config.config.get('plugins_directory', 'plugins'))
        self.plugin_manager = WandPluginManager(plugins_dir)
        
        # 7. Initialize core system (handles orchestration)
        self.core = WandCore(self.config.config)
        
        # 8. Register core components with monitoring
        self._register_components()
        
        # 9. Start monitoring service
        self.monitor.start_monitoring()
        self.logger.log_info("System initialization complete.")

    def _register_components(self):
        """Register all components for monitoring."""
        components = {
            'builder': self.builder,
            'ai': self.ai,
            'plugin_manager': self.plugin_manager,
            'core': self.core
        }
        for name, component in components.items():
            self.monitor.register_component(name, component)

    def start(self):
        """Start The Wand system."""
        try:
            self.logger.log_info("[LAUNCH] Starting AIOS IO Global HPC System")  # Remove emoji
            # Ensure project directory exists
            self.helper.ensure_directory(self.helper.get_project_path())
            
            # Initialize core systems
            self.core.initialize()
            
            # Start core services
            self.core.start_services()
            
            # Enable AI learning if configured
            if self.config.config.get('enable_ai_learning', False):
                self.ai.start_learning_loop()
                
            return True
        except Exception as e:
            self.logger.log_error("Failed to start system", e)  # Fix error logging format
            return False

if __name__ == "__main__":
    # First initialize the backend
    wand = TheWand()
    wand.start()
    
    # Then start the GUI
    root = AutoGPTBuilderGUI()
    root.protocol("WM_DELETE_WINDOW", root.on_closing)  # Handle window close
    root.mainloop()  # Start the GUI event loop

# existing code continues below

def load_dynamic_features():
    """Scans and loads modules; logs errors without crashing."""
    FEATURES_DIR = os.path.dirname(os.path.abspath(__file__))
    error_log_path = os.path.join(os.path.expanduser("~"), "Documents", "AIOS IO", "wand_errors.log")
    for root, _, files in os.walk(FEATURES_DIR):
        for file in files:
            if file.endswith(".py") and file != os.path.basename(__file__):
                module_name = file[:-3]
                module_path = os.path.join(root, file)
                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Cannot load module {module_name}")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "register_feature"):
                        module.register_feature()
                        print(f"[INFO] Loaded feature: {module_name} from {root}")
                except Exception as e:
                    with open(error_log_path, "a") as err_log:
                        err_log.write(f"{time.ctime()} ERROR in {module_path}: {e}\n")
                    print(f"[ERROR] Skipped module {module_name}: {e}")
# Update imports to use absolute paths
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from wandhelper import WandHelper, initialize_logging, create_default_config
# ...existing code...



