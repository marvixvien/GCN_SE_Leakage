import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import wntr
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    model_dir      = "results/semisupervised5",      # directory containing config.json + model_weights.weights.h5
    inp_file       = "generate_datasets/ASnet2.inp",             # EPANET network file
    out_dir        = "leakage/data",                      # where to write the two xlsx files
    monitor_nodes  = ["5","11","32","37","44"],# 10% sensor nodes (known heads)
    leakage_min    = 19.94,                    # L/s  — matches flow.py / pressure.py
    leakage_max    = 199.4,
    leakage_levels = 31,
    latent_dim     = 20,
    hidden_layers  = 2,
    correction_updates = 20,
    alpha          = 0.5,
    non_linearity  = "leaky_relu",
    batch_size     = 32,
)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Hazen-Williams helpers  (mirrors problem.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

HW_N = 1.0 / 1.852   # ≈ 0.54


def hw_coefficient(pipe) -> float:
    """Compute 1/c_ij from WNTR pipe object."""
    return 1.0 / (
        10.667
        * pipe.length
        / (pipe.roughness ** 1.852)
        / (pipe.diameter ** 4.871)
    )


def heads_to_flows(heads: np.ndarray, A_raw: np.ndarray) -> np.ndarray:
    """
    Compute pipe flows from nodal heads using Hazen-Williams.

    Parameters
    ----------
    heads : (n_nodes,)  hydraulic head [m]
    A_raw : (n_edges, 3)  columns = [start_node_id, end_node_id, 1/c_ij]

    Returns
    -------
    flows : (n_edges,)  flow [L/s], positive = from→to
    """
    i = A_raw[:, 0].astype(int)
    j = A_raw[:, 1].astype(int)
    c = A_raw[:, 2]                       # 1/c_ij

    dH   = heads[i] - heads[j]            # head difference
    Q    = np.sign(dH) * (np.maximum(np.abs(dH), 1e-9) * c) ** HW_N
    return Q                               # L/s  (c_ij already in consistent units)


# ══════════════════════════════════════════════════════════════════════════════
# Network topology builder
# ══════════════════════════════════════════════════════════════════════════════

def build_network_topology(wn, monitor_nodes):
    """
    Extract static network arrays from WNTR model.

    Returns
    -------
    node_ids  : list[str]   ordered junction IDs (sorted numerically)
    pipe_ids  : list[str]   ordered pipe IDs     (sorted numerically)
    A_raw     : (n_edges, 3)  [from_id, to_id, hw_coeff]  — integer node indices
    Nd        : (n_nodes,)  demand indicator  (1=junction, 0=reservoir/tank)
    Nh        : (n_nodes,)  head unknown flag (1=unknown,  0=monitored sensor)
    elev      : (n_nodes,)  node elevation [m]
    node_idx  : dict  str→int
    """
    # Assign sequential IDs to nodes in WNTR traversal order
    node_list = list(wn.nodes())           # (name, node) pairs
    node_idx  = {name: idx for idx, (name, _) in enumerate(node_list)}

    # Sorted junction and pipe lists for consistent column ordering
    junction_ids = sorted(wn.junction_name_list, key=lambda x: int(x))
    pipe_ids     = sorted(wn.pipe_name_list,     key=lambda x: int(x))

    n_nodes = len(node_list)

    # Demand indicator and elevation per node
    Nd   = np.zeros(n_nodes, dtype=np.float32)
    Nh   = np.zeros(n_nodes, dtype=np.float32)
    elev = np.zeros(n_nodes, dtype=np.float32)

    for name, node in node_list:
        idx = node_idx[name]
        elev[idx] = node.elevation if hasattr(node, "elevation") else 0.0
        if node.node_type == "Junction":
            Nd[idx] = 1.0
            Nh[idx] = 0.0 if name in monitor_nodes else 1.0
        # reservoirs/tanks: Nd=0, Nh=0 (head always known)

    # Edge matrix
    A_raw = []
    for _, link in wn.links():
        if link.link_type != "Pipe":
            continue
        fr = node_idx[link.start_node.name]
        to = node_idx[link.end_node.name]
        c  = hw_coefficient(link)
        A_raw.append([fr, to, c])

    A_raw = np.array(A_raw, dtype=np.float32)  # (n_edges, 3)

    return junction_ids, pipe_ids, A_raw, Nd, Nh, elev, node_idx


# ══════════════════════════════════════════════════════════════════════════════
# GCN model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_gcn_model(model_dir, cfg):
    """Load DeepStatisticalSolver from saved directory."""
    # Import from wherever models.py lives
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.models import DeepStatisticalSolver          # adjust if needed

    model = DeepStatisticalSolver(
        latent_dimension   = cfg.latent_dim,
        hidden_layers      = cfg.hidden_layers,
        correction_updates = cfg.correction_updates,
        alpha              = cfg.alpha,
        non_linearity      = cfg.non_linearity,
        batch_size         = cfg.batch_size,
        name               = "physics_gcn",
        directory          = model_dir,
        default_data_directory = cfg.data_directory,
        model_to_restore   = None,
        proxy              = True,
    )

    model.load_model(model_dir)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Per-scenario input construction
# ══════════════════════════════════════════════════════════════════════════════

def build_scenario_inputs(
    wn,
    leakage_node_str: str,
    leakage_lps: float,
    A_raw: np.ndarray,
    Nd: np.ndarray,
    Nh: np.ndarray,
    node_idx: dict,
    monitor_nodes: list,
    base_demands: dict,
):
    """
    Construct (A, B) tensors for one leakage scenario.

    For monitored nodes we need the true head — we run a lightweight WNTR
    simulation limited to those nodes only (full sim is fastest since EPANET
    is already loaded; we just extract the 5 values we need).

    Parameters
    ----------
    leakage_node_str : junction name where leak is injected
    leakage_lps      : leakage magnitude [L/s]
    base_demands     : {node_name: base_demand_m3s}  (pre-extracted, constant)

    Returns
    -------
    A_tf : (1, n_edges, 3)
    B_tf : (1, n_nodes, 4)
    """
    n_nodes = len(node_idx)
    n_edges = len(A_raw)

    # ── Apply leakage: temporarily modify demand ──────────────────────────────
    leakage_node = wn.get_node(leakage_node_str)
    original_demand = leakage_node.demand_timeseries_list[0].base_value
    leakage_m3s = leakage_lps / 1000.0
    leakage_node.demand_timeseries_list[0].base_value = original_demand + leakage_m3s

    # ── Run WNTR to get monitored-node heads ──────────────────────────────────
    try:
        results = wntr.sim.EpanetSimulator(wn).run_sim(version=2.0)
        head_sim = results.node["head"].iloc[0].to_dict()   # {node_name: head_m}
        demand_sim = results.node["demand"].iloc[0].to_dict()
    finally:
        # Always restore demand
        leakage_node.demand_timeseries_list[0].base_value = original_demand

    # ── Build B matrix [n_nodes, 4] ──────────────────────────────────────────
    B = np.zeros((n_nodes, 4), dtype=np.float32)
    for name, idx in node_idx.items():
        node = wn.get_node(name)
        B[idx, 0] = Nd[idx]                                  # demand indicator
        # demand: use simulated value (includes leakage for leakage node)
        B[idx, 1] = max(demand_sim.get(name, 0.0), 0.0)     # [m³/s] → model unit
        B[idx, 2] = Nh[idx]                                  # head unknown flag
        # known head: only for sources + monitored junctions
        if Nh[idx] == 0.0:                                   # known head node
            B[idx, 3] = head_sim.get(name, 0.0)

    # ── Build A matrix [n_edges, 3] ──────────────────────────────────────────
    A = A_raw.copy()   # [from_idx, to_idx, hw_coeff]

    # ── Add batch dimension ───────────────────────────────────────────────────
    A_tf = tf.constant(A[np.newaxis], dtype=tf.float32)   # (1, n_edges, 3)
    B_tf = tf.constant(B[np.newaxis], dtype=tf.float32)   # (1, n_nodes, 4)

    # Also return head_sim for reference (monitored nodes are ground-truth)
    return A_tf, B_tf, head_sim


# ══════════════════════════════════════════════════════════════════════════════
# Main inference loop
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(cfg):
    """Full pipeline: load model → iterate scenarios → write xlsx."""

    log.info("=" * 70)
    log.info("GCN Inference — recreating xlsx files")
    log.info("=" * 70)

    # ── Load network ──────────────────────────────────────────────────────────
    log.info(f"Loading network: {cfg.inp_file}")
    wn = wntr.network.WaterNetworkModel(cfg.inp_file)
    monitor_nodes = cfg.monitor_nodes

    junction_ids, pipe_ids, A_raw, Nd, Nh, elev, node_idx = build_network_topology(
        wn, monitor_nodes
    )

    n_nodes = len(node_idx)
    n_junctions = len(junction_ids)
    n_pipes = len(pipe_ids)
    log.info(f"  Nodes (total): {n_nodes}   Junctions: {n_junctions}   Pipes: {n_pipes}")
    log.info(f"  Monitored nodes (sensors): {monitor_nodes}")

    # Pre-extract base demands (before any leakage modification)
    base_demands = {}
    for name, node in wn.nodes():
        if node.node_type == "Junction":
            base_demands[name] = node.demand_timeseries_list[0].base_value

    # ── Load GCN model ────────────────────────────────────────────────────────
    log.info(f"\nLoading GCN model from: {cfg.model_dir}")
    model = load_gcn_model(cfg.model_dir, cfg)
    log.info("  Model loaded successfully.")

    # ── Leakage levels ────────────────────────────────────────────────────────
    leakage_values = np.linspace(cfg.leakage_min, cfg.leakage_max, cfg.leakage_levels)

    total_scenarios = n_junctions * cfg.leakage_levels
    log.info(f"\nRunning {n_junctions} nodes × {cfg.leakage_levels} levels = {total_scenarios} scenarios")

    # Output accumulators
    flow_rows     = []   # → pipe_flows xlsx
    pressure_rows = []   # → node_pressures xlsx

    # Reverse index: node_idx position → junction column name
    idx_to_name = {v: k for k, v in node_idx.items()}

    with tqdm(total=total_scenarios, desc="Inference") as pbar:
        for node_str in junction_ids:
            for level_i, leak_lps in enumerate(leakage_values, start=1):

                # ── Build inputs ──────────────────────────────────────────────
                try:
                    A_tf, B_tf, head_sim = build_scenario_inputs(
                        wn, node_str, leak_lps,
                        A_raw, Nd, Nh, node_idx,
                        monitor_nodes, base_demands,
                    )
                except Exception as e:
                    log.warning(f"  WNTR error at node {node_str} level {level_i}: {e}")
                    # Fill row with NaN so the index is preserved
                    flow_rows.append(
                        {**{f"PIPE {p}": np.nan for p in pipe_ids},
                         "NODE": int(node_str), "LEAKAGE LEVEL": level_i}
                    )
                    pressure_rows.append(
                        {**{f"NODE {n}": np.nan for n in junction_ids},
                         "LEAKAGE NODE": int(node_str), "LEAKAGE LEVEL": level_i}
                    )
                    pbar.update(1)
                    continue

                # ── GCN inference ─────────────────────────────────────────────
                outputs = model((A_tf, B_tf), training=False)
                U_pred  = outputs["final_prediction"].numpy()[0, :, 0]  # (n_nodes,)

                # Merge: use known heads for monitored/source nodes
                B_np = B_tf.numpy()[0]         # (n_nodes, 4)
                Nd_arr = B_np[:, 0]            # demand indicator
                Nh_arr = B_np[:, 2]            # head unknown flag

                # Actual head: GCN prediction for unknown nodes, true value for known
                H_actual = Nd_arr * U_pred + (1 - Nd_arr) * B_np[:, 3]
                # For monitored junctions (Nd=1, Nh=0): use known head
                for mon in monitor_nodes:
                    if mon in node_idx:
                        midx = node_idx[mon]
                        H_actual[midx] = head_sim.get(mon, H_actual[midx])

                # ── Pipe flows via Hazen-Williams ─────────────────────────────
                Q = heads_to_flows(H_actual, A_raw)   # (n_edges,) [L/s]

                # Map edge index → pipe_id (same ordering as A_raw construction)
                # We need to rebuild the pipe→edge index map once
                pipe_flow_map = {}
                edge_i = 0
                for _, link in wn.links():
                    if link.link_type != "Pipe":
                        continue
                    pipe_flow_map[link.name] = Q[edge_i]
                    edge_i += 1

                flow_row = {f"PIPE {p}": pipe_flow_map[p] for p in pipe_ids}
                flow_row["NODE"]          = int(node_str)
                flow_row["LEAKAGE LEVEL"] = level_i
                flow_rows.append(flow_row)

                # ── Node pressures (head − elevation) ─────────────────────────
                pressure_row = {}
                for junc_str in junction_ids:
                    idx = node_idx[junc_str]
                    pressure_row[f"NODE {junc_str}"] = float(H_actual[idx] - elev[idx])
                pressure_row["LEAKAGE NODE"]  = int(node_str)
                pressure_row["LEAKAGE LEVEL"] = level_i
                pressure_rows.append(pressure_row)

                pbar.update(1)

    # ── Build DataFrames ──────────────────────────────────────────────────────
    pipe_cols     = [f"PIPE {p}" for p in pipe_ids]
    pressure_cols = [f"NODE {n}" for n in junction_ids]

    df_flows     = pd.DataFrame(flow_rows)[pipe_cols + ["NODE", "LEAKAGE LEVEL"]]
    df_pressures = pd.DataFrame(pressure_rows)[pressure_cols + ["LEAKAGE NODE", "LEAKAGE LEVEL"]]

    # ── Save xlsx ─────────────────────────────────────────────────────────────
    os.makedirs(cfg.out_dir, exist_ok=True)

    flow_path = os.path.join(cfg.out_dir, "pipe_flows_gcn.xlsx")
    pres_path = os.path.join(cfg.out_dir, "node_pressures_gcn.xlsx")

    df_flows.to_excel(flow_path,     index=False)
    df_pressures.to_excel(pres_path, index=False)

    log.info(f"\n✅  pipe_flows written    → {flow_path}   ({df_flows.shape})")
    log.info(f"✅  node_pressures written → {pres_path}  ({df_pressures.shape})")

    return df_flows, df_pressures


# ══════════════════════════════════════════════════════════════════════════════
# Optional validation: compare GCN xlsx vs original WNTR xlsx
# ══════════════════════════════════════════════════════════════════════════════

def validate_against_wntr(gcn_flows_path, gcn_pres_path,
                           wntr_flows_path="leakage/data/pipe_flows_AS_31.xlsx",
                           wntr_pres_path="leakage/data/node_pressures_AS_31.xlsx"):
    """Print MAE / RMSE between GCN predictions and original WNTR results."""
    if not (os.path.exists(wntr_flows_path) and os.path.exists(wntr_pres_path)):
        log.info("Original WNTR xlsx not found — skipping validation.")
        return

    df_gcn_f  = pd.read_excel(gcn_flows_path)
    df_wnt_f  = pd.read_excel(wntr_flows_path)
    df_gcn_p  = pd.read_excel(gcn_pres_path)
    df_wnt_p  = pd.read_excel(wntr_pres_path)

    # Align on NODE/LEAKAGE LEVEL
    df_gcn_f  = df_gcn_f.sort_values(["NODE","LEAKAGE LEVEL"]).reset_index(drop=True)
    df_wnt_f  = df_wnt_f.sort_values(["NODE","LEAKAGE LEVEL"]).reset_index(drop=True)
    df_gcn_p  = df_gcn_p.sort_values(["LEAKAGE NODE","LEAKAGE LEVEL"]).reset_index(drop=True)
    df_wnt_p  = df_wnt_p.sort_values(["LEAKAGE NODE","LEAKAGE LEVEL"]).reset_index(drop=True)

    pipe_cols = [c for c in df_gcn_f.columns if c.startswith("PIPE")]
    node_cols = [c for c in df_gcn_p.columns if c.startswith("NODE ")]

    err_f = (df_gcn_f[pipe_cols].values - df_wnt_f[pipe_cols].values)
    err_p = (df_gcn_p[node_cols].values - df_wnt_p[node_cols].values)

    log.info("\n── Validation vs. WNTR ─────────────────────────────────────────")
    log.info(f"  Pipe flows  → MAE: {np.nanmean(np.abs(err_f)):.4f} L/s  |  RMSE: {np.sqrt(np.nanmean(err_f**2)):.4f} L/s")
    log.info(f"  Pressures   → MAE: {np.nanmean(np.abs(err_p)):.4f} m    |  RMSE: {np.sqrt(np.nanmean(err_p**2)):.4f} m")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir",      default=DEFAULTS["model_dir"])
    p.add_argument("--inp_file",       default=DEFAULTS["inp_file"])
    p.add_argument("--out_dir",        default=DEFAULTS["out_dir"])
    p.add_argument("--monitor_nodes",  nargs="+", default=DEFAULTS["monitor_nodes"],
                   help="Space-separated list of monitored junction IDs")
    p.add_argument("--leakage_min",    type=float, default=DEFAULTS["leakage_min"])
    p.add_argument("--leakage_max",    type=float, default=DEFAULTS["leakage_max"])
    p.add_argument("--leakage_levels", type=int,   default=DEFAULTS["leakage_levels"])

    p.add_argument('--data_directory', type=str, default='datasets/asnet2_1/', help='Directory containing training data')
    p.add_argument("--latent_dim",     type=int,   default=DEFAULTS["latent_dim"])
    p.add_argument("--hidden_layers",  type=int,   default=DEFAULTS["hidden_layers"])
    p.add_argument("--correction_updates", type=int,   default=DEFAULTS["correction_updates"])
    p.add_argument("--alpha",          type=float, default=DEFAULTS["alpha"])
    p.add_argument("--non_linearity",  default=DEFAULTS["non_linearity"])
    p.add_argument("--batch_size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--validate",       action="store_true",
                   help="Compare GCN outputs against original WNTR xlsx files")
    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()

    df_flows, df_pressures = run_inference(cfg)

    if cfg.validate:
        flow_out = os.path.join(cfg.out_dir, "pipe_flows_gcn.xlsx")
        pres_out = os.path.join(cfg.out_dir, "node_pressures_gcn.xlsx")
        validate_against_wntr(flow_out, pres_out)

'''
python gcn_inference_xlsx.py \
  --model_dir  results/semisupervised5 \
  --inp_file   generate_datasets/ASnet2.inp \
  --out_dir    leakage/data/ \
  --validate \
  --data_directory datasets/asnet2_5/ \
  --batch_size 500 \
  --latent_dim 20 \
  --correction_updates 20 \
  --hidden_layers 2 

  Requirements:
    pip install tensorflow wntr numpy pandas openpyxl tqdm
'''