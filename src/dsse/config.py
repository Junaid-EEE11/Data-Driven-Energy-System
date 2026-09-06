"""Configuration loader and schema definitions for DSSE."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dataclasses import dataclass, field


def get_project_root() -> Path:
    """Returns absolute path to project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Loads a YAML configuration file safely."""
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class PathConfig:
    feeder_dir: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_splits: Path
    results_raw: Path
    results_aggregated: Path
    results_tables: Path
    results_figures: Path

    @property
    def data_dir(self) -> Path:
        return self.data_raw.parent

    @property
    def results_dir(self) -> Path:
        return self.results_raw.parent

    @classmethod
    def from_dict(cls, data: Dict[str, str], root: Path) -> PathConfig:
        return cls(
            feeder_dir=root / data.get("feeder_dir", "feeder_models/ieee123"),
            data_raw=root / data.get("data_raw", "data/raw"),
            data_interim=root / data.get("data_interim", "data/interim"),
            data_processed=root / data.get("data_processed", "data/processed"),
            data_splits=root / data.get("data_splits", "data/splits"),
            results_raw=root / data.get("results_raw", "results/raw"),
            results_aggregated=root / data.get("results_aggregated", "results/aggregated"),
            results_tables=root / data.get("results_tables", "results/tables"),
            results_figures=root / data.get("results_figures", "results/figures"),
        )

    def make_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for p in [
            self.feeder_dir,
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.data_splits,
            self.results_raw,
            self.results_aggregated,
            self.results_tables,
            self.results_figures,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    project_name: str
    random_seed: int
    seeds: List[int]
    log_level: str
    device: str
    paths: PathConfig
    data_cfg: Dict[str, Any] = field(default_factory=dict)
    model_cfg: Dict[str, Any] = field(default_factory=dict)
    experiments_cfg: Dict[str, Any] = field(default_factory=dict)
    robustness_cfg: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_all(cls, root: Optional[Path] = None) -> AppConfig:
        if root is None:
            root = get_project_root()
        base = load_yaml(root / "configs/base.yaml")
        data = load_yaml(root / "configs/data.yaml")
        model = load_yaml(root / "configs/model.yaml")
        exps = load_yaml(root / "configs/experiments.yaml")
        robust = load_yaml(root / "configs/robustness.yaml")

        paths_cfg = PathConfig.from_dict(base.get("paths", {}), root)
        paths_cfg.make_dirs()

        return cls(
            project_name=base.get("project_name", "dsse-physics-conformal"),
            random_seed=base.get("random_seed", 42),
            seeds=base.get("seeds", [42, 123, 456, 789, 1011]),
            log_level=base.get("log_level", "INFO"),
            device=base.get("device", "cpu"),
            paths=paths_cfg,
            data_cfg=data,
            model_cfg=model,
            experiments_cfg=exps,
            robustness_cfg=robust,
        )
