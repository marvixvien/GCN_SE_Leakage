import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# ─────────────────────────────────────────────────────────────────────────────

import logging
import time
from typing import Dict

import numpy as np

from src.config import PRESSURE_SCALE
from src.detector import LeakageDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = _HERE / "results"


def generate_inference_report(
    idx: int,
    true_node: int,
    true_level: float,
    results: Dict[str, dict],
    ens_result: dict,
    inference_time: float,
    out_dir: Path = RESULTS_DIR,
    filename: str = "inference_demo_report.txt",
) -> None:
    """Generates a text report matching the style of the training evaluation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    # Safely convert true_node to a pure integer
    true_n_int = int(float(true_node))

    sep = "=" * 80; dash = "-" * 80
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\nLEAKAGE DETECTION FRAMEWORK — INFERENCE REPORT\n")
        f.write("Physics-Integrated GCNs · Soumyajit Banerjee · University of Calcutta\n")
        f.write(f"{sep}\n\n")

        f.write(f"SINGLE-SNAPSHOT DEMO (Test Sample #{idx})\n{dash}\n")
        f.write(f"  True Leak Node     : {true_node}\n")
        f.write(f"  True Leak Severity : {true_level:.4f}\n\n")

        f.write(f"  {'Model':<10} {'Predicted':<12} {'Severity':<12} {'Confidence':<12} {'Result'}\n")
        f.write(f"  {dash[:66]}\n")

        # Write individual model results
        for name in ("RF", "ET", "XGB"):
            res = results[name]
            
            # Bulletproof comparison
            pred_n_int = int(float(res["leak_node"]))+1
            is_correct = "CORRECT" if pred_n_int == true_n_int else "WRONG"
            
            f.write(
                f"  {name:<10} "
                f"{pred_n_int:<12} "
                f"{res['leak_severity']:<12.4f} "
                f"{res['confidence']*100:>7.1f}%      "
                f"{is_correct}\n"
            )

        # Write Ensemble result
        f.write(f"  {dash[:66]}\n")
        
        # Bulletproof comparison
        ens_n_int = int(float(ens_result["leak_node"]))+1
        ens_correct = "CORRECT" if ens_n_int == true_n_int else "WRONG"
        
        f.write(
            f"  {'ENSEMBLE':<10} "
            f"{pred_n_int:<12} "
            f"{ens_result['leak_severity']:<12.4f} "
            f"{ens_result['confidence']*100:>7.1f}%      "
            f"{ens_correct}\n"
        )

        f.write(f"\n  Inference time (3 models): {inference_time:.4f} s\n")
        f.write(f"\n{sep}\n")

    logger.info("Report → %s", path)


def main() -> None:
    sep = "=" * 70
    logger.info(sep)
    logger.info("WDS LEAKAGE DETECTION — INFERENCE DEMO")
    logger.info(sep)

    # ── Load saved models + rebuild test split for demonstration ──────────────
    logger.info("\n[1/3]  Loading trained models and data ...")
    det = LeakageDetector()
    det.load_data()           # rebuilds the same train/val/test split
    det.load(prefix="leakage_model")

    # ── Batch evaluation on the held-out test set ─────────────────────────────
    logger.info("\n[2/3]  Batch evaluation on held-out test set ...")
    det.evaluate(save_plots=False, save_report=False)

    # ── Single-snapshot demo ───────────────────────────────────────────────────
    logger.info("\n[3/3]  Single-snapshot inference demo ...")
    splits = det._splits
    rng    = np.random.default_rng(seed=0)
    idx    = rng.integers(0, len(splits.X_test))

    # Recover physical-unit values from the scaled test vector
    sample_unscaled = splits.scaler.inverse_transform(
        splits.X_test[idx].reshape(1, -1)
    ).ravel()

    n_nodes       = splits.n_nodes
    pressure_head = sample_unscaled[:n_nodes] / PRESSURE_SCALE  # kPa → metres
    flow_values   = sample_unscaled[n_nodes:]

    true_node  = splits.y_node_test[idx]
    true_level = splits.y_level_test[idx]
    
    # Pre-calculate pure integer for terminal logging
    true_n_int = int(float(true_node))

    logger.info("  True leak node  : %s", true_node)
    logger.info("  True leak level : %.4f", true_level)
    logger.info("  %s", "-" * 50)

    model_results = {}
    t_start = time.perf_counter()
    
    for model_type in ("RF", "ET", "XGB"):
        result  = det.predict(pressure_head, flow_values, model_type=model_type)
        model_results[model_type] = result
        
        # Bulletproof comparison for terminal
        pred_n_int = int(float(result["leak_node"]))+1
        correct = "CORRECT" if pred_n_int == true_n_int else "wrong  "
        
        logger.info(
            "  [%s]  %s  node=%3s  severity=%.4f  conf=%5.1f%%",
            model_type, correct,
            pred_n_int, result["leak_severity"],
            result["confidence"] * 100,
        )
        
    inference_time = time.perf_counter() - t_start
    logger.info("  Inference time (3 models): %.4f s", inference_time)

    # Ensemble
    logger.info("\n  Ensemble vote:")
    ens = det.predict(pressure_head, flow_values, use_ensemble=True)
    
    # Bulletproof comparison for terminal
    ens_n_int = int(float(ens["leak_node"]))+1
    ens_correct = "CORRECT" if ens_n_int == true_n_int else "wrong  "
    
    logger.info(
        "  [ENS]  %s  node=%3s  severity=%.4f  conf=%5.1f%%",
        ens_correct,
        ens_n_int, ens["leak_severity"], ens["confidence"] * 100,
    )

    # Save scenario visualisation
    det.visualise_scenario(
        pressure_head, flow_values, leak_node=true_node,
        title=f"Test Sample #{idx} — Leak at Node {true_node} (Level {true_level})",
    )
    
    # Save the text report
    generate_inference_report(
        idx=idx, 
        true_node=true_node, 
        true_level=true_level, 
        results=model_results, 
        ens_result=ens, 
        inference_time=inference_time
    )

    logger.info("\n%s", sep)
    logger.info("INFERENCE DEMO COMPLETE")
    logger.info(sep)


if __name__ == "__main__":
    main()