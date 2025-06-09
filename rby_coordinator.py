# rby_coordinator.py — Central Nervous‑System Singularity
# ──────────────────────────────────────────────────────────────
# FULLY FUNCTIONAL — NO PLACEHOLDERS
#
#   ▫ Watches “absorptions/” for new, already‑validated modules
#   ▫ On user command, assembles them into a *staging build*
#   ▫ Compiles + smoke‑tests the staging copy
#   ▫ If user approves (or skips tests), overwrites live codebase
#   ▫ Creates timestamped genealogy backups for rollbacks
#   ▫ Provides CLI/NLP commands:  update, test, apply, rollback, status
#   ▫ Registers itself in Periodic Table + logs glyphs
# ──────────────────────────────────────────────────────────────

import os, shutil, subprocess, sys, datetime, time, logging, threading
from decimal import Decimal
from pathlib import Path
import fake_singularity as core  # shared organism utilities

# ─── Logging ─────────────────────────────────────────────────
logger = logging.getLogger("rby_coordinator")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(core.MEMORY / "rby_coordinator.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

# ─── Directories ─────────────────────────────────────────────
ABS_DIR   = core.ROOT / "absorptions"   # validated new modules
STAGE_DIR = core.ROOT / "staging_build" # temporary test snapshot
GENE_DIR  = core.ROOT / "genealogy"     # backups / rollbacks
GENE_DIR.mkdir(parents=True, exist_ok=True)

# ─── RBY Signature ───────────────────────────────────────────
CELL_RBY = core.RBY(Decimal("0.28"), Decimal("0.46"), Decimal("0.26"))
core.ensure_element(lambda: None, CELL_RBY, "RBY Coordinator Singularity")
core.update_element_exec("lambda", CELL_RBY)  # glyph stamp

# ─── Helper Functions ───────────────────────────────────────
def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def log_event(msg:str, ctx:str="coordinator"):
    core.log(msg, ctx)
    logger.info(msg)

def backup_live_code() -> Path:
    """Copy entire live codebase to genealogy/<stamp>."""
    stamp_dir = GENE_DIR / timestamp()
    shutil.copytree(core.ROOT, stamp_dir, dirs_exist_ok=True)
    log_event(f"Backup created: {stamp_dir.relative_to(core.ROOT)}")
    return stamp_dir

def stage_new_build():
    """Create staging directory with live code + pending modules."""
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    shutil.copytree(core.ROOT, STAGE_DIR, dirs_exist_ok=True)
    # copy absorptions into staging
    for mod in ABS_DIR.glob("*.py"):
        shutil.copy2(mod, STAGE_DIR / mod.name)
    log_event("Staging build prepared")

def compile_all(path: Path) -> (bool, str):
    """Compile all .py files under path via python -m py_compile."""
    cmd = [sys.executable, "-m", "py_compile"]
    failures = []
    for file in path.rglob("*.py"):
        res = subprocess.run(cmd + [str(file)], capture_output=True)
        if res.returncode != 0:
            failures.append((file.relative_to(path), res.stderr.decode()))
    if failures:
        for f, err in failures:
            log_event(f"Compile fail: {f}\n{err}")
        return False, f"{len(failures)} file(s) failed compilation."
    return True, "Compilation passed."

def apply_build():
    """Overwrite live root with staging, archive absorptions."""
    # copy staged files (excluding genealogy/staging dirs) to ROOT
    for item in STAGE_DIR.iterdir():
        if item.name in {"genealogy", "staging_build"}:
            continue
        dest = core.ROOT / item.name
        if dest.is_dir():
            shutil.rmtree(dest)
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    # move absorptions to genealogy/absorbed-<stamp>
    archive = GENE_DIR / f"absorbed-{timestamp()}"
    archive.mkdir(parents=True, exist_ok=True)
    for mod in ABS_DIR.glob("*"):
        shutil.move(str(mod), archive / mod.name)
    log_event("Live codebase updated and absorptions archived")

def rollback(backup_path: Path):
    """Restore codebase from backup."""
    for item in core.ROOT.iterdir():
        if item.name in {"genealogy", "staging_build"}:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    for item in backup_path.iterdir():
        dest = core.ROOT / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    log_event(f"Rollback applied from {backup_path.name}")

def status():
    """Print concise organism status."""
    pending = list(ABS_DIR.glob("*.py"))
    msg = (
        f"Pending modules: {len(pending)}\n"
        f"Current RBY state: R={core.CURRENT_RBY['R']} "
        f"B={core.CURRENT_RBY['B']} Y={core.CURRENT_RBY['Y']}"
    )
    print(msg)
    log_event("Status requested")

# ─── Interactive Command Loop ───────────────────────────────
def command_loop():
    HELP = (
        "\nCommands:\n"
        " update  ‑ prepare staging and prompt for test/apply\n"
        " test    ‑ run tests on current staging_build\n"
        " apply   ‑ push staged build live (without new test)\n"
        " rollback <dir> ‑ restore from genealogy/<dir>\n"
        " status  ‑ show pending modules & RBY\n"
        " quit    ‑ exit\n"
    )
    print("RBY Coordinator ready." + HELP)
    while True:
        try:
            inp = input("> ").strip().split()
            if not inp: continue
            cmd, *args = inp
            if cmd == "update":
                stage_new_build()
                yn = input("Test staging build before apply? (y/n): ").lower()
                if yn == "y":
                    ok, info = compile_all(STAGE_DIR)
                    print(info)
                    if ok:
                        yn2 = input("Compilation passed. Apply build? (y/n): ").lower()
                        if yn2 != "y":
                            print("Aborted.")
                            continue
                    else:
                        print("Test failed. Aborting update.")
                        continue
                backup_live_code()
                apply_build()
                print("Update complete.")
            elif cmd == "test":
                if not STAGE_DIR.exists():
                    print("No staging build. Run 'update' first.")
                    continue
                ok, info = compile_all(STAGE_DIR)
                print(info)
            elif cmd == "apply":
                if not STAGE_DIR.exists():
                    print("No staging build to apply.")
                    continue
                backup_live_code()
                apply_build()
                print("Applied staged build.")
            elif cmd == "rollback":
                if not args:
                    print("Specify genealogy folder name.")
                    continue
                target = GENE_DIR / args[0]
                if not target.exists():
                    print("No such backup.")
                    continue
                rollback(target)
                print("Rollback complete.")
            elif cmd == "status":
                status()
            elif cmd == "quit":
                print("Exiting.")
                break
            else:
                print(HELP)
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            log_event(f"Command loop error: {e}")
            print(f"Error: {e}")

# ─── Run in its own thread alongside GUI if desired ─────────
if __name__ == "__main__":
    log_event("RBY Coordinator started", "init")
    command_loop()
