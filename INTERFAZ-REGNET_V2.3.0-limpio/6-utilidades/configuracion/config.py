"""Configuration helper for the project.

Simple loader that reads YAML config files. Adjust as needed.
"""
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def load_yaml(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
