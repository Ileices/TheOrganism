# 🧬 fake_singularity.py — Self-Evolving Recursive AI Organism
# Runs forever. Builds memory. Parses commands. Generates glyphs. Stores excretions. Learns.

import os, uuid, time, yaml, json, hashlib, datetime
from decimal import Decimal, getcontext
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
import logging
import inspect

getcontext().prec = 50  # High-precision RBY weights

# Add a basic configuration for logging to prevent recursion issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(filename='singularity.log'), logging.StreamHandler()]
)

# === 🧬 FILESYSTEM SETUP
ROOT = Path(__file__).resolve().parent
SANDBOX = ROOT / "sandbox"
USER_TOUCH = ROOT / "user_touch"
MEMORY = ROOT / "memory"
GLYPHS = ROOT / "glyphs"
PERIODIC_TABLE = ROOT / "periodic_table" / "elements"
for p in (SANDBOX, USER_TOUCH, MEMORY, GLYPHS, PERIODIC_TABLE):
    p.mkdir(parents=True, exist_ok=True)

# === 🧬 RBY CORE
def RBY(r=Decimal("0.33333333333333333333333333333333333333333333333334"),
        b=Decimal("0.33333333333333333333333333333333333333333333333333"),
        y=Decimal("0.33333333333333333333333333333333333333333333333333")):
    return {"R": r, "B": b, "Y": y}

def color_distance(a, b):
    return ((a["R"]-b["R"])**2 + (a["B"]-b["B"])**2 + (a["Y"]-b["Y"])**2).sqrt()

def glyph_from_rby(rby: dict, glyph_id=None):
    R, B, Y = int(float(rby["R"]*255)), int(float(rby["B"]*255)), int(float(rby["Y"]*255))
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    d   = ImageDraw.Draw(img)
    d.ellipse([24,24,104,104], fill=(R, Y, B))
    gid = glyph_id or str(uuid.uuid4())
    img.save(GLYPHS/f"{gid}.png")
    return gid, img

# === 🧬 RBY PERSISTENCE
RBY_STATE_PATH = MEMORY / "last_rby_state.json"

def save_rby_state(rby):
    """Save RBY state to persistent storage"""
    with open(RBY_STATE_PATH, "w") as f:
        json.dump({k: str(v) for k,v in rby.items()}, f, indent=2)
    log("RBY state saved", "state")

def load_rby_state():
    """Load RBY state from persistent storage or create default"""
    if RBY_STATE_PATH.exists():
        try:
            with open(RBY_STATE_PATH) as f:
                d = json.load(f)
                return {k: Decimal(v) for k,v in d.items()}
        except (json.JSONDecodeError, KeyError) as e:
            log(f"Failed to load RBY state: {e}", "error")
    return RBY()  # Return default if file doesn't exist or loading fails

# === 🧬 SHARED STATE
CURRENT_RBY = load_rby_state()  # Global RBY state for functions that don't receive the GUI context

# === 🧬 MEMORY / LOGGING
def log(event: str, ctx: str="system"):
    stamp = datetime.datetime.utcnow().isoformat()
    with open(MEMORY/"events.log","a",encoding="utf-8") as f:
        f.write(f"{stamp}\t{ctx}\t{event}\n")

# === 🧬 DNA STORAGE
def element_path(func_name:str): return PERIODIC_TABLE / f"{func_name}.yaml"

def ensure_element(func, rby, description:str):
    fpath = element_path(func.__name__)
    if fpath.exists(): return
    element = {
        "uuid": str(uuid.uuid4()),
        "function": func.__name__,
        "description": description,
        "RBY": {k:str(v) for k,v in rby.items()},
        "glyph": None,
        "last_exec": None,
        "lineage": []
    }
    with open(fpath,"w") as fw: yaml.safe_dump(element, fw)
    log(f"Element created for {func.__name__}", "dna")

def load_element(func_name:str):
    with open(element_path(func_name)) as fr:
        return yaml.safe_load(fr)

def update_element_exec(func_name:str, rby):
    el = load_element(func_name)
    el["last_exec"] = datetime.datetime.utcnow().isoformat()
    el["glyph"], _  = glyph_from_rby(rby)
    with open(element_path(func_name), "w") as fw:
        yaml.safe_dump(el, fw)

# === 🧬 CORE BEHAVIOURS

def scan_user_touch():
    """sensory‑input: ingest new files & recurse into folders safely"""
    ensure_element(scan_user_touch, RBY(Decimal("0.62"),Decimal("0.28"),Decimal("0.10")),
                   scan_user_touch.__doc__)

    def ingest(path: Path):
        if path.is_dir():
            # recurse into sub‑folders
            try:
                for sub in path.iterdir():
                    ingest(sub)
            except PermissionError:
                log(f"skip_dir_perm:{path}", "warn")
            except Exception as e:
                log(f"dir_error:{e}", "error")
        else:
            try:
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                (SANDBOX / digest).write_bytes(data)
                log(f"Absorbed {path}", "touch")
                # Only attempt to delete the file if it was successfully read
                try:
                    path.unlink()
                except Exception as e:
                    log(f"delete_error:{path} - {e}", "error")
            except PermissionError:
                log(f"skip_perm:{path}", "warn")
            except FileNotFoundError:
                log(f"file_not_found:{path}", "warn")
            except Exception as e:
                log(f"touch_error:{path} - {e}", "error")

    try:
        # Check if user_touch directory exists before iterating
        if USER_TOUCH.exists() and USER_TOUCH.is_dir():
            for entry in USER_TOUCH.iterdir():
                ingest(entry)
        else:
            log("User_touch directory not found or not accessible", "error")
    except Exception as e:
        log(f"scan_error:{e}", "error")

    update_element_exec("scan_user_touch", RBY(Decimal("0.60"),Decimal("0.28"),Decimal("0.12")))

def analyze_source():
    """meta-cognition: introspect own source to (re)build DNA table"""
    try:
        ensure_element(analyze_source, RBY(Decimal("0.21"),Decimal("0.59"),Decimal("0.20")), analyze_source.__doc__)
        
        # Use explicit UTF-8 encoding to avoid charmap errors
        src = Path(__file__).read_text(encoding='utf-8', errors='replace')
        code_digest = hashlib.sha256(src.encode('utf-8')).hexdigest()
        
        # Write with explicit UTF-8 encoding
        (MEMORY/f"source_{code_digest}.txt").write_text(src, encoding='utf-8')
        
        log(f"Source analyzed successfully: {code_digest}", "analyze")
        update_element_exec("analyze_source", RBY(Decimal("0.22"),Decimal("0.58"),Decimal("0.20")))
    except Exception as e:
        log(f"Source analysis failed: {str(e)}", "error")

def dream_mutation():
    """dreaming: recombine glyph compost into hypothetical code"""
    try:
        ensure_element(dream_mutation, RBY(Decimal("0.25"),Decimal("0.25"),Decimal("0.50")), dream_mutation.__doc__)
        
        # Create a mutation with actual functioning code
        mutation_id = uuid.uuid4().hex
        mutation_path = SANDBOX/f"mut_{mutation_id}.py"
        
        # Create a meaningful mutation that will affect the RBY state
        mutation_code = f"""# Mutation {mutation_id} generated at {datetime.datetime.now()}
# This mutation attempts to optimize RBY values

import random
from decimal import Decimal

# Access the current RBY state
current = CURRENT_RBY.copy()

# Generate slightly modified RBY values
r_mod = Decimal(str(random.uniform(-0.05, 0.05)))
b_mod = Decimal(str(random.uniform(-0.05, 0.05)))
y_mod = Decimal(str(random.uniform(-0.05, 0.05)))

# Create new RBY state with small modifications
new_r = max(min(current['R'] + r_mod, Decimal('0.95')), Decimal('0.05'))
new_b = max(min(current['B'] + b_mod, Decimal('0.95')), Decimal('0.05'))
new_y = max(min(current['Y'] + y_mod, Decimal('0.95')), Decimal('0.05'))

# Set result for the execution context to pick up
result_rby = {{
    'R': new_r,
    'B': new_b,
    'Y': new_y
}}

print(f"Mutation {mutation_id} completed with RBY: {{new_r:.2f}}, {{new_b:.2f}}, {{new_y:.2f}}")
"""
        
        # Save mutation to SANDBOX with explicit UTF-8 encoding
        mutation_path.write_text(mutation_code, encoding='utf-8')
        
        # Also save a copy to MEMORY for tracking
        (MEMORY/f"mutation_{mutation_id}.py").write_text(mutation_code, encoding='utf-8')
        
        log(f"Dream mutation created: {mutation_id}", "dream")
        update_element_exec("dream_mutation", RBY(Decimal("0.26"),Decimal("0.24"),Decimal("0.50")))
    except Exception as e:
        log(f"Dream mutation failed: {str(e)}", "error")

def execute_dreams():
    """execution: run mutated dream stubs recursively and extract intelligence"""
    try:
        ensure_element(execute_dreams, RBY(Decimal("0.33"),Decimal("0.20"),Decimal("0.47")), execute_dreams.__doc__)
        
        global CURRENT_RBY
        success_count = 0
        
        # Make a list of files first to avoid modifying during iteration
        files = list(sorted(SANDBOX.glob("mut_*.py")))
        for file_path in files:
            mutation_id = file_path.stem[4:]  # Remove 'mut_' prefix
            result_log_path = MEMORY / f"result_{mutation_id}.txt"
            
            try:
                # Read with explicit UTF-8 encoding
                code = file_path.read_text(encoding='utf-8')
                
                # Create a safe execution context
                local_ctx = {
                    'RBY': RBY, 
                    'CURRENT_RBY': CURRENT_RBY.copy(),
                    'random': __import__('random'),
                    'Decimal': Decimal
                }
                
                # Capture stdout for logging
                import io
                from contextlib import redirect_stdout
                output = io.StringIO()
                
                with redirect_stdout(output):
                    exec(code, {'RBY': RBY, '__builtins__': __builtins__}, local_ctx)
                
                # Log the output
                output_text = output.getvalue()
                log(f"Executed: {file_path.name} - Output: {output_text.strip()}", "dream_exec")
                
                # Save execution result
                with open(result_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Execution time: {datetime.datetime.now().isoformat()}\n")
                    f.write(f"Output: {output_text}\n")
                    f.write(f"Success: True\n")
                
                success_count += 1
                
                # Check if mutation returned RBY updates
                if 'result_rby' in local_ctx:
                    try:
                        new_rby = local_ctx['result_rby']
                        # Validate the structure before using
                        if (isinstance(new_rby, dict) and 'R' in new_rby and 'B' in new_rby and 'Y' in new_rby and 
                            isinstance(new_rby['R'], (Decimal, float, int)) and 
                            isinstance(new_rby['B'], (Decimal, float, int)) and
                            isinstance(new_rby['Y'], (Decimal, float, int))):
                            
                            # Ensure they're all Decimal objects
                            CURRENT_RBY = {
                                'R': Decimal(str(new_rby['R'])),
                                'B': Decimal(str(new_rby['B'])),
                                'Y': Decimal(str(new_rby['Y']))
                            }
                            save_rby_state(CURRENT_RBY)
                            
                            # Add RBY state to result log
                            with open(result_log_path, 'a', encoding='utf-8') as f:
                                f.write(f"Updated RBY: R={CURRENT_RBY['R']}, B={CURRENT_RBY['B']}, Y={CURRENT_RBY['Y']}\n")
                            
                            log(f"Updated RBY from execution {mutation_id}", "dream_rby")
                    except Exception as e:
                        log(f"RBY update error: {e}", "error")
                        with open(result_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"RBY Update Error: {str(e)}\n")
                        
            except Exception as e:
                log(f"Failed: {file_path.name} -> {str(e)}", "dream_exec_fail")
                
                # Log failure
                with open(result_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Execution time: {datetime.datetime.now().isoformat()}\n")
                    f.write(f"Failed: {str(e)}\n")
                    f.write(f"Success: False\n")
            
            finally:
                try:
                    # Move the mutation file to MEMORY for archiving instead of deleting
                    archive_path = MEMORY / f"executed_{file_path.name}"
                    if not archive_path.exists():  # Don't overwrite existing archives
                        file_path.replace(archive_path)
                    else:
                        file_path.unlink()  # If archive already exists, just delete
                except Exception as e:
                    log(f"Archive error: {str(e)}", "error")
                    try:
                        file_path.unlink()  # Try to delete if move failed
                    except:
                        pass  # Silently ignore if deletion also fails

        # Adjust RBY based on execution success rate but ensure bounds
        new_r = Decimal("0.31") + (Decimal("0.01") * Decimal(min(success_count, 5)))
        new_b = Decimal("0.24") - (Decimal("0.01") * Decimal(min(success_count, 5)))
        new_r = max(min(new_r, Decimal("0.9")), Decimal("0.1"))  # Keep between 0.1 and 0.9
        new_b = max(min(new_b, Decimal("0.9")), Decimal("0.1"))  # Keep between 0.1 and 0.9
        
        rby_update = RBY(new_r, new_b, Decimal("0.45"))
        update_element_exec("execute_dreams", rby_update)
        
        # Create summary of this execution cycle
        with open(MEMORY / f"dream_cycle_{int(time.time())}.txt", 'w', encoding='utf-8') as f:
            f.write(f"Cycle time: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Files executed: {len(files)}\n")
            f.write(f"Success count: {success_count}\n")
            f.write(f"Current RBY: R={CURRENT_RBY['R']}, B={CURRENT_RBY['B']}, Y={CURRENT_RBY['Y']}\n")
        
    except Exception as e:
        log(f"Execute dreams main error: {str(e)}", "error")

# Fix the log_excretion and glyphic_decay_trace functions to be more resilient
def log_excretion(evaluation: str):
    """Log each RBY mutation as a recursive memory excretion for future recursive re-absorption."""
    try:
        # Create a simple log entry to avoid recursion issues
        with open(MEMORY / "excretion_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.utcnow().isoformat()} - {evaluation}\n")
            
    except Exception:
        pass  # Silently fail to avoid recursion

def glyphic_decay_trace():
    """Leave a glyphic trail of memory decay after mutation (short form compression)."""
    try:
        global CURRENT_RBY
        
        # Just create a simple text entry to avoid any JSON parsing issues
        with open(MEMORY / "glyphic_trace.txt", "a", encoding="utf-8") as f:
            f.write(f"{int(time.time())} - {CURRENT_RBY['R']} {CURRENT_RBY['B']} {CURRENT_RBY['Y']}\n")
    except Exception:
        pass  # Silently fail to avoid recursion

# === 🧬 GUI / LOOP

class SingularityGUI:
    def __init__(self):
        global CURRENT_RBY
        self.rby_state = load_rby_state()
        CURRENT_RBY = self.rby_state  # Sync with global state
        
        self.root      = tk.Tk() 
        self.root.title("Singularity")
        self.chat      = tk.Text(self.root, height=6); self.chat.pack(fill="x")
        self.logframe  = tk.Text(self.root, height=10, bg="#111", fg="#0f0")
        self.logframe.pack(fill="x")
        self.canvas    = tk.Canvas(self.root, width=128, height=128); self.canvas.pack()
        self.root.bind("<Return>", self.on_enter)
        self.tick()
    
    def on_enter(self, event):
        user_text = self.chat.get("1.0", "end").strip()
        self.chat.delete("1.0","end")
        log(f"USER:{user_text}", "chat")
        self.logframe.insert("end", f"> {user_text}\n")

        # NLP Trigger Hooks
        if "analyze" in user_text.lower():
            analyze_source()
        elif "dream" in user_text.lower():
            dream_mutation()
        elif "execute" in user_text.lower():
            execute_dreams()
        elif "scan" in user_text.lower():
            scan_user_touch()
        elif "glyph" in user_text.lower():
            glyph_id, img = glyph_from_rby(self.rby_state)
            self.canvas.delete("all")
            self.tkimg = ImageTk.PhotoImage(img)
            self.canvas.create_image(64,64,image=self.tkimg)
        else:
            # Add command to sandbox for dream execution
            fname = SANDBOX / f"mut_{uuid.uuid4().hex}.py"
            fname.write_text(user_text)
            log(f"User script absorbed: {fname.name}", "chat_dream")
            self.logframe.insert("end", f"Script absorbed: {fname.name}\n")
            # naive echo for seed: mutate Y weight for activity
            self.rby_state["Y"] += Decimal("0.00000000000000000000000000000000000000000000000001")

# Make the SingularityGUI tick method more resilient
def tick(self):
    global CURRENT_RBY
    try:
        # Run core functions individually with error handling
        try:
            scan_user_touch()
        except Exception as e:
            self.logframe.insert("end", f"Scan error: {str(e)}\n")
            
        try:
            analyze_source()
        except Exception as e:
            self.logframe.insert("end", f"Analyze error: {str(e)}\n")
            
        try:
            dream_mutation()
        except Exception as e:
            self.logframe.insert("end", f"Dream error: {str(e)}\n")
            
        try:
            execute_dreams()
        except Exception as e:
            self.logframe.insert("end", f"Execute error: {str(e)}\n")
        
        # Sync with global RBY
        self.rby_state = CURRENT_RBY
        
        # Update the display
        try:
            glyph_id, img = glyph_from_rby(self.rby_state)
            self.canvas.delete("all")
            self.tkimg = ImageTk.PhotoImage(img)
            self.canvas.create_image(64, 64, image=self.tkimg)
            self.logframe.see("end")
        except Exception as e:
            self.logframe.insert("end", f"Display error: {str(e)}\n")
        
        # Save RBY state
        try:
            save_rby_state(self.rby_state)
        except Exception as e:
            self.logframe.insert("end", f"Save error: {str(e)}\n")
            
    except Exception as e:
        self.logframe.insert("end", f"CRITICAL ERROR: {str(e)}\n")
        
    # Schedule the next tick
    self.root.after(3000, self.tick)

# Replace the SingularityGUI.tick method with our more resilient version
SingularityGUI.tick = tick

# 🔁 Recursive Excretion Memory Logging + Glyphic Tracing Upgrade
NODE_ID = str(uuid.uuid4())
EXCRETION_LOG_FILE = Path(__file__).parent / "excretion_memory_log.json"
GLYPHIC_DECAY_FILE = Path(__file__).parent / "glyphic_decay_trail.json"

# === 🧬 MAIN
def main():
    log("Singularity boot‑sequence start", "init")
    gui = SingularityGUI()
    gui.root.mainloop()

if __name__ == "__main__":
    main()
