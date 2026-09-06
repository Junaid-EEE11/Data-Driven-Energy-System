"""Reproducibility utilities: random seed management, environment logging, and metadata."""

from __future__ import annotations
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds deterministically across standard libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str = "dsse",
    level: str = "INFO",
    log_file: Optional[Path | str] = None,
) -> logging.Logger:
    """Configures a structured console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def collect_environment_info() -> Dict[str, Any]:
    """Collects comprehensive hardware, OS, and package version metadata for reproducibility."""
    import opendssdirect as dss

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "numpy_version": np.__version__,
        "opendssdirect_version": getattr(dss, "__version__", "unknown"),
    }
    return info


def save_metadata(metadata: Dict[str, Any], filepath: Path | str) -> None:
    """Saves structured experiment metadata to JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
