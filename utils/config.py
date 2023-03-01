from pathlib import Path
import logging
import os

def ensure_dir(dir_):
    os.makedirs(dir_, exist_ok=True)
    return dir_

HOME_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ensure_dir(HOME_DIR / "data")
PROCESSED_DATA_DIR = ensure_dir(HOME_DIR / "processed")
OUT_DIR = ensure_dir(HOME_DIR / "outputs")

def get_logger(name, *, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger

__all__ = [
    "HOME_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "OUT_DIR",
]
