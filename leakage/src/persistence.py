"""
persistence.py
--------------
Save and load all trained model artefacts using joblib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib

from src.config import MODELS_DIR

if TYPE_CHECKING:
    from .models import EnsembleModels
    from .data_loader import DataSplits

logger   = logging.getLogger(__name__)
_NAMES   = ("RF", "ET", "XGB")


def save(models: "EnsembleModels", splits: "DataSplits",
         prefix: str = "leakage_model", out_dir: Path = MODELS_DIR) -> None:
    """
    Write scaler, six model files, and metadata to *out_dir*.

    Files produced
    --------------
    <prefix>_scaler.pkl
    <prefix>_classifier_{RF,ET,XGB}.pkl
    <prefix>_regressor_{RF,ET,XGB}.pkl
    <prefix>_metadata.pkl
    """
    if not models.is_fitted:
        raise RuntimeError("Models have not been trained — nothing to save.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(splits.scaler, out_dir / f"{prefix}_scaler.pkl")

    for name in _NAMES:
        joblib.dump(models.classifiers[name], out_dir / f"{prefix}_classifier_{name}.pkl")
        joblib.dump(models.regressors[name],  out_dir / f"{prefix}_regressor_{name}.pkl")

    joblib.dump(
        {"n_nodes": splits.n_nodes, "n_pipes": splits.n_pipes,
         "feature_names": splits.feature_names},
        out_dir / f"{prefix}_metadata.pkl",
    )

    logger.info("Models saved → %s/  (prefix='%s')", out_dir, prefix)


def load(models: "EnsembleModels", splits: "DataSplits",
         prefix: str = "leakage_model", src_dir: Path = MODELS_DIR) -> None:
    """Restore all artefacts in-place into *models* and *splits*."""
    src_dir = Path(src_dir)

    scaler_path = src_dir / f"{prefix}_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"\nNo saved models found at: {src_dir / prefix}*\n"
            "Run  python train.py  first to train and save the models."
        )

    splits.scaler = joblib.load(scaler_path)

    for name in _NAMES:
        models.classifiers[name] = joblib.load(src_dir / f"{prefix}_classifier_{name}.pkl")
        models.regressors[name]  = joblib.load(src_dir / f"{prefix}_regressor_{name}.pkl")

    meta = joblib.load(src_dir / f"{prefix}_metadata.pkl")
    splits.n_nodes       = meta["n_nodes"]
    splits.n_pipes        = meta["n_pipes"]
    splits.feature_names  = meta["feature_names"]

    models.is_fitted = True
    logger.info("Models loaded ← %s/  (prefix='%s')", src_dir, prefix)
