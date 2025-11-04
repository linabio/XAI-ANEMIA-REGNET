import logging
from logging.config import dictConfig
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]

def setup_logging(cfg_path: str = '6-utilidades/configuracion/logger.yaml'):
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = ROOT / cfg_path
    if cfg_file.exists():
        with open(cfg_file, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        dictConfig(cfg)
    else:
        logging.basicConfig(level=logging.INFO)
