"""
config.py
---------
Central configuration for the WDS Leakage Detection Framework.

All paths, physical constants, and model hyper-parameters live here.
Change values in this one file.

Directory layout:

    leakage/
    ├── node_pressures_AS_8.xlsx   ← GCN pressure output
    ├── pipe_flows_AS_8.xlsx       ← GCN flow output
    ├── src/                       ← this package
    │   ├── config.py
    │   └── ...
    ├── train.py
    └── predict.py

Author : Soumyajit Banerjee, University of Calcutta
Project: Physics-Integrated GCNs for State Estimation and Leak Detection in WDS
"""

from pathlib import Path

# ── Project root = the folder that contains src/  (i.e. leakage/) ───────────
# src/config.py  →  parent = src/  →  parent.parent = leakage/
ROOT_DIR    : Path = Path(__file__).resolve().parent.parent

# ── Data files sit in leakage/data ──────────
PRESSURE_FILE : Path = ROOT_DIR / "data" / "node_pressures_gcn.xlsx"
FLOW_FILE     : Path = ROOT_DIR / "data" / "pipe_flows_gcn.xlsx"

# ── Output directories (auto-created on first run) ───────────────────────────
MODELS_DIR  : Path = ROOT_DIR / "models"
RESULTS_DIR : Path = ROOT_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Column name conventions (as they appear in the Excel files) ──────────────
# Pressure file : "NODE 1" … "NODE 51", "LEAKAGE NODE", "LEAKAGE LEVEL"
# Flow file     : "PIPE 1" … "PIPE 65", "NODE",         "LEAKAGE LEVEL"
PRESSURE_NODE_PREFIX : str = "NODE"
FLOW_PIPE_PREFIX     : str = "PIPE"

PRESSURE_LABEL_NODE  : str = "LEAKAGE NODE"
PRESSURE_LABEL_LEVEL : str = "LEAKAGE LEVEL"

# ── WDS topology (Alperovits-Shamir benchmark) ───────────────────────────────
N_NODES : int = 51
N_PIPES : int = 65

# ── Physical constants (Section 4.5.1 of the project report) ─────────────────
WATER_DENSITY  : float = 998.0          # kg/m³
GRAVITY        : float = 9.81           # m/s²
PRESSURE_SCALE : float = WATER_DENSITY * GRAVITY / 1000   # ≈ 9.79 kPa/m

# ── Train / validation / test split ratios ───────────────────────────────────
TEST_SIZE : float = 0.20
VAL_SIZE  : float = 0.10

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_STATE : int = 42

# ── Ensemble hyper-parameters ─────────────────────────────────────────────────
RF_PARAMS : dict = dict(
    n_estimators      = 200,
    max_depth         = 30,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
)

ET_PARAMS : dict = dict(
    n_estimators      = 200,
    max_depth         = 30,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
)

XGB_PARAMS : dict = dict(
    n_estimators  = 200,
    max_depth     = 10,
    learning_rate = 0.1,
    random_state  = RANDOM_STATE,
    n_jobs        = -1,
    eval_metric   = "mlogloss",
)

# ── Alert threshold ───────────────────────────────────────────────────────────
ALERT_CONFIDENCE_THRESHOLD : float = 0.80

# ── Plot settings ─────────────────────────────────────────────────────────────
FIGURE_DPI   : int  = 300
MODEL_COLORS : dict = {"RF": "#3498db", "ET": "#2ecc71", "XGB": "#e74c3c"}
