# src/common/paths.py
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
KG_DIR = os.path.join(DATA_DIR, "knowledge_graph")
OUT_DIR = os.path.join(DATA_DIR, "output")

# Make sure these exist
for d in [RAW_DIR, PROC_DIR, KG_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)