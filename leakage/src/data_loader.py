"""
data_loader.py
--------------
Loads and pre-processes the GCN-estimated hydraulic data from Excel files.

Converts hydraulic heads [m] to pressures [kPa], merges pressure and flow
features, and performs a robust train/val/test split with automatic
stratification fallback for small datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FLOW_FILE,
    FLOW_PIPE_PREFIX,
    N_NODES,
    N_PIPES,
    PRESSURE_FILE,
    PRESSURE_LABEL_LEVEL,
    PRESSURE_LABEL_NODE,
    PRESSURE_NODE_PREFIX,
    PRESSURE_SCALE,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class DataSplits:
    """Container for scaled train / val / test splits plus the fitted scaler."""
    X_train: np.ndarray
    X_val:   np.ndarray
    X_test:  np.ndarray

    y_node_train:  np.ndarray
    y_node_val:    np.ndarray
    y_node_test:   np.ndarray

    y_level_train: np.ndarray
    y_level_val:   np.ndarray
    y_level_test:  np.ndarray

    scaler:        StandardScaler = field(repr=False)
    feature_names: list[str]      = field(repr=False)
    n_nodes:       int            = N_NODES
    n_pipes:       int            = N_PIPES


# ── Helpers ──────────────────────────────────────────────────────────────────

def _select_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return columns whose name starts with *prefix*, sorted numerically."""
    cols = [c for c in df.columns if c.upper().startswith(prefix.upper())]
    if not cols:
        raise ValueError(
            f"No columns with prefix '{prefix}' found. "
            f"Available columns: {list(df.columns)}"
        )
    cols.sort(key=lambda c: int(c.split()[-1]))
    return df[cols]


def _stratify_ok(min_cnt: int, frac: float) -> bool:
    """Return True if sklearn's stratified split is feasible."""
    needed = int(np.ceil(1.0 / min(frac, 1.0 - frac))) + 1
    return min_cnt >= needed


def build_feature_names(n_nodes: int, n_pipes: int) -> list[str]:
    return (
        [f"Node_{i+1}_Pressure_kPa" for i in range(n_nodes)] +
        [f"Pipe_{i+1}_Flow"          for i in range(n_pipes)]
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_raw_data(
    pressure_file: Path | str = PRESSURE_FILE,
    flow_file:     Path | str = FLOW_FILE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the two Excel files and return (X, y_node, y_level).

    X has shape (n_samples, n_nodes + n_pipes).
    Pressure columns are converted from hydraulic head [m] to kPa.
    Labels (y_node, y_level) come from the pressure file.
    """
    pressure_file = Path(pressure_file)
    flow_file     = Path(flow_file)

    for f in (pressure_file, flow_file):
        if not f.exists():
            raise FileNotFoundError(
                f"\nData file not found: {f}"
                f"\nExpected location: {f.resolve()}"
                f"\nMake sure the Excel files are in the same folder as train.py."
            )

    logger.info("Loading: %s", pressure_file.name)
    pressure_df = pd.read_excel(pressure_file)

    logger.info("Loading: %s", flow_file.name)
    flow_df = pd.read_excel(flow_file)

    for col in (PRESSURE_LABEL_NODE, PRESSURE_LABEL_LEVEL):
        if col not in pressure_df.columns:
            raise ValueError(
                f"Column '{col}' not found in {pressure_file.name}. "
                f"Found: {list(pressure_df.columns)}"
            )

    y_node  = pressure_df[PRESSURE_LABEL_NODE].values.astype(int)
    y_level = pressure_df[PRESSURE_LABEL_LEVEL].values.astype(float)

    pressure_features = _select_columns(pressure_df, PRESSURE_NODE_PREFIX)
    flow_features     = _select_columns(flow_df,     FLOW_PIPE_PREFIX)

    if len(pressure_features) != len(flow_features):
        raise ValueError(
            f"Row count mismatch: pressure={len(pressure_features)}, "
            f"flow={len(flow_features)}"
        )

    pressure_kpa = pressure_features.values * PRESSURE_SCALE
    flow_values  = flow_features.values

    X = np.hstack([pressure_kpa, flow_values])
    n_nodes = pressure_kpa.shape[1]
    n_pipes = flow_values.shape[1]

    logger.info(
        "Loaded %d samples | %d nodes | %d pipes | %d unique leak nodes",
        len(X), n_nodes, n_pipes, len(np.unique(y_node)),
    )
    return X, y_node, y_level


def prepare_splits(
    X:       np.ndarray,
    y_node:  np.ndarray,
    y_level: np.ndarray,
    test_size:    float = TEST_SIZE,
    val_size:     float = VAL_SIZE,
    random_state: int   = RANDOM_STATE,
) -> DataSplits:
    """
    Stratified train/val/test split with automatic fallback for small datasets.
    Scaler is fit on training data only (no data leakage).
    """
    logger.info("Splitting data — test=%.0f%%, val=%.0f%%",
                test_size * 100, val_size * 100)

    min_cnt = int(np.bincount((y_node - y_node.min()).astype(int)).min())

    strat_1 = y_node if _stratify_ok(min_cnt, test_size) else None
    if strat_1 is None:
        logger.warning("Stratify disabled (first split): min class count=%d", min_cnt)

    X_tv, X_test, yn_tv, yn_test, yl_tv, yl_test = train_test_split(
        X, y_node, y_level,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_1,
    )

    val_ratio     = val_size / (1.0 - test_size)
    min_cnt_tv    = int(np.bincount((yn_tv - yn_tv.min()).astype(int)).min())
    strat_2       = yn_tv if _stratify_ok(min_cnt_tv, val_ratio) else None
    if strat_2 is None:
        logger.warning("Stratify disabled (second split): min class count=%d", min_cnt_tv)

    X_train, X_val, yn_train, yn_val, yl_train, yl_val = train_test_split(
        X_tv, yn_tv, yl_tv,
        test_size=val_ratio,
        random_state=random_state,
        stratify=strat_2,
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_val_sc    = scaler.transform(X_val)
    X_test_sc   = scaler.transform(X_test)

    logger.info("Split sizes — train:%d  val:%d  test:%d",
                len(X_train), len(X_val), len(X_test))

    n_nodes = X.shape[1] - N_PIPES
    return DataSplits(
        X_train=X_train_sc, X_val=X_val_sc, X_test=X_test_sc,
        y_node_train=yn_train, y_node_val=yn_val, y_node_test=yn_test,
        y_level_train=yl_train, y_level_val=yl_val, y_level_test=yl_test,
        scaler=scaler,
        feature_names=build_feature_names(n_nodes, N_PIPES),
        n_nodes=n_nodes,
        n_pipes=N_PIPES,
    )
