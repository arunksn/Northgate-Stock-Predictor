"""Shared helpers: config, seeding, paths, logging."""
from __future__ import annotations
import logging, os, random
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
MODELS, FIGS, RES = ROOT / "models", ROOT / "reports" / "figures", ROOT / "reports" / "results"
DISCLAIMER = ("Educational research and decision-support tool only. Outputs are not financial advice.")


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:  # tensorflow optional for non-DL stages
        pass


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def ensure_dirs() -> None:
    for d in (RAW, PROC, MODELS, FIGS, RES):
        d.mkdir(parents=True, exist_ok=True)
