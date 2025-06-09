# gene_splicer_absorber.py — Recursive AI Code Splicer & Integrator Feature Module

import os
import re
import ast
import csv
import json
import yaml
import uuid
import time
import logging
import subprocess
import datetime
from decimal import Decimal
from pathlib import Path

import fake_singularity as core  # assumes this module is in same directory and exposes RBY, ensure_element, update_element_exec, PERIODIC_TABLE, USER_TOUCH, MEMORY, log

# ─── Logging Setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("gene_splicer_absorber")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename=core.MEMORY / "gene_splicer_absorber.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

# ─── Constants ──────────────────────────────────────────────────────────────────
ABSORPTIONS_DIR = core.ROOT / "absorptions"
ABSORPTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ─── RBY CLASSIFICATION ─────────────────────────────────────────────────────────
def classify_rby(text: str) -> dict:
    """
    Very basic RBY scorer: equal weights, can be replaced by NLP analysis later.
    """
    return {"R": Decimal("0.33"), "B": Decimal("0.33"), "Y": Decimal("0.34")}

# ─── METADATA EXTRACTION ─────────────────────────────────────────────────────────
def extract_metadata(path: Path) -> dict:
    """
    Parse docstrings & comments from Python and comment‐based languages.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    docstrings, comments = [], []
    if path.suffix.lower() == ".py":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    doc = ast.get_docstring(node)
                    if doc:
                        docstrings.append(doc.strip())
            comments = re.findall(r"#\s?(.*)", text)
        except Exception as e:
            logger.error(f"AST parse failed for {path.name}: {e}")
    else:
        # fallback: C‐style and Python‐style comment regex
        comments = re.findall(r"//\s?(.*)", text) + re.findall(r"#\s?(.*)", text)
    description = docstrings[0] if docstrings else (comments[0] if comments else path.name)
    return {"description": description.strip(), "docstrings": docstrings, "comments": comments}

# ─── INTEGRATION TESTING ────────────────────────────────────────────────────────
def integration_test(path: Path) -> (bool, str):
    """
    For Python: py_compile
    JSON/YAML/CSV: parse
    Others: marked unsupported
    """
    try:
        ext = path.suffix.lower()
        if ext == ".py":
            res = subprocess.run(
                ["python", "-m", "py_compile", str(path)],
                capture_output=True
            )
            return res.returncode == 0, res.stderr.decode().strip()
        elif ext == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            return True, ""
        elif ext in (".yaml", ".yml"):
            yaml.safe_load(path.read_text(encoding="utf-8"))
            return True, ""
        elif ext == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                _ = next(csv.reader(f), None)
            return True, ""
        else:
            return False, "Unsupported file type"
    except Exception as e:
        return False, str(e)

# ─── PERIODIC TABLE WRITING ────────────────────────────────────────────────────
def write_periodic_element(name: str, description: str, rby: dict):
    """
    Create a YAML entry under periodic_table/elements for NLP triggers.
    """
    fpath = core.PERIODIC_TABLE / f"{name}.yaml"
    element = {
        "uuid": str(uuid.uuid4()),
        "function": name,
        "description": description,
        "RBY": {k: str(v) for k, v in rby.items()},
        "glyph": None,
        "last_exec": None,
        "lineage": []
    }
    with open(fpath, "w", encoding="utf-8") as fw:
        yaml.safe_dump(element, fw)
    core.log(f"Element written: {name}", "gene_splicer")

# ─── ABSORB & INTEGRATE ────────────────────────────────────────────────────────
def absorb_code_file(path: Path) -> bool:
    """
    Full absorption pipeline for one file:
     1. Extract metadata
     2. Classify RBY
     3. Write periodic element
     4. Run integration test
     5. Update element exec + glyph
     6. Archive file on success/failure
    """
    name = path.stem
    core.log(f"Absorbing {name}", "gene_splicer")
    md = extract_metadata(path)
    rby = classify_rby(md["description"])
    write_periodic_element(name, md["description"], rby)
    success, output = integration_test(path)
    core.log(f"Test {name}: success={success}, output={output}", "gene_splicer")
    # update last_exec and generate glyph via core
    core.update_element_exec(name, core.RBY(rby["R"], rby["B"], rby["Y"]))
    # move file to absorptions archive
    archive = ABSORPTIONS_DIR / path.name
    path.replace(archive)
    return success

def process_absorptions():
    """
    Scan USER_TOUCH for any code files and absorb them.
    """
    for entry in core.USER_TOUCH.iterdir():
        if entry.is_file():
            try:
                absorb_code_file(entry)
            except Exception as e:
                logger.error(f"Absorb failure {entry.name}: {e}")
    core.log("Absorption cycle complete", "gene_splicer")

# ─── INTEGRATION LOOP ──────────────────────────────────────────────────────────
def integration_loop(interval_seconds: int = 5):
    """
    Run absorption cycles forever at fixed interval.
    """
    while True:
        process_absorptions()
        time.sleep(interval_seconds)

# ─── MODULE ENTRYPOINT ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    core.log("GeneSplicer absorber starting", "init")
    integration_loop()
