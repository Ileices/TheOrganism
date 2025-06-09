import os, sys, time, json, hashlib, random, shutil
from datetime import datetime
from multiprocessing import Process

# === 🧬 CORE SETTINGS ===
MEMORY_FILE = "meta.json"
EXCRETE_FOLDER = "excrete"
GENERATION_LIMIT = 10000  # Prevent infinite disk bloat
SELF_NAME = "aeos_core.py"

# === 📁 INIT FOLDERS ===
os.makedirs(EXCRETE_FOLDER, exist_ok=True)

# === 🧠 LOAD MEMORY ===
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump({"generation": 0, "log": []}, f)

with open(MEMORY_FILE, "r") as f:
    memory = json.load(f)

# === 🔁 MUTATE SELF ===
def mutate_self(source_lines):
    lines = source_lines[:]
    index = random.randint(0, len(lines)-1)
    action = random.choice(["insert", "replace", "delete"])
    comment = f"# MUTATION {datetime.now().isoformat()} | {action}\n"

    if action == "insert":
        lines.insert(index, comment)
    elif action == "replace":
        lines[index] = comment
    elif action == "delete" and len(lines) > 1:
        lines.pop(index)

    return lines, action

# === 🧬 SPAWN CHILD ===
def spawn_child(path):
    print(f"[⚙️] Spawning new AEOS organism: {path}")
    os.execv(sys.executable, [sys.executable, path])

# === 🧬 EVOLUTION CYCLE ===
def evolve():
    global memory
    with open(SELF_NAME, "r", encoding="utf-8") as f:
        code = f.readlines()

    mutated, action = mutate_self(code)
    new_code = "".join(mutated)
    hash_id = hashlib.sha256(new_code.encode()).hexdigest()[:12]
    new_path = os.path.join(EXCRETE_FOLDER, f"aeos_{hash_id}.py")

    with open(new_path, "w", encoding="utf-8") as f:
        f.write(new_code)

    memory["generation"] += 1
    memory["log"].append({
        "time": datetime.now().isoformat(),
        "mutation": action,
        "hash": hash_id,
        "file": new_path
    })

    if len(memory["log"]) > GENERATION_LIMIT:
        memory["log"] = memory["log"][-GENERATION_LIMIT:]

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

    spawn_child(new_path)

# === 🧠 MAIN EXECUTION ===
if __name__ == "__main__":
    print(f"[🌱] AEOS organism running. Generation {memory['generation']}")
    time.sleep(1)  # Simulate thinking
    evolve()
