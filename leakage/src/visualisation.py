"""
visualisation.py
----------------
Plots and text report for the WDS Leakage Detection Framework.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import FIGURE_DPI, MODEL_COLORS, RESULTS_DIR

logger = logging.getLogger(__name__)


def _savefig(fig: plt.Figure, filename: str, out_dir: Path = RESULTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_performance_overview(
    cls_metrics: Dict[str, dict],
    reg_metrics: Dict[str, dict],
    y_node_true:  np.ndarray,
    y_level_true: np.ndarray,
    out_dir: Path = RESULTS_DIR,
) -> None:
    """Six-panel performance overview figure."""
    models = ["RF", "ET", "XGB"]
    colors = [MODEL_COLORS[m] for m in models]

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "Leakage Detection — Performance Overview\n"
        "(Physics-Integrated GCNs · Section 4.5.3)",
        fontsize=15, fontweight="bold", y=1.01,
    )

    def _bar(ax, values, title, ylabel, fmt):
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor="black")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    fmt.format(h), ha="center", va="bottom", fontsize=10)

    # 1 · Accuracy
    ax1 = fig.add_subplot(2, 3, 1)
    accs = [cls_metrics[m]["accuracy"] * 100 for m in models]
    ax1.set_ylim([max(0, min(accs) - 5), 100])
    _bar(ax1, accs, "Leak Node Classification Accuracy", "Accuracy (%)", "{:.2f}%")

    # 2 · MAE
    _bar(fig.add_subplot(2, 3, 2),
         [reg_metrics[m]["mae"] for m in models],
         "Leak Severity — MAE", "MAE", "{:.4f}")

    # 3 · RMSE
    _bar(fig.add_subplot(2, 3, 3),
         [reg_metrics[m]["rmse"] for m in models],
         "Leak Severity — RMSE", "RMSE", "{:.4f}")

    # 4 · Predicted vs True
    ax5 = fig.add_subplot(2, 3, 5)
    y_pred_level = reg_metrics["XGB"]["predictions"]
    ax5.scatter(y_level_true, y_pred_level, alpha=0.5, s=20, color=MODEL_COLORS["XGB"])
    mn, mx = min(y_level_true.min(), y_pred_level.min()), max(y_level_true.max(), y_pred_level.max())
    ax5.plot([mn, mx], [mn, mx], "k--", lw=2, label="Perfect Prediction")
    ax5.set_xlabel("True Severity"); ax5.set_ylabel("Predicted Severity")
    ax5.set_title("Severity: Predicted vs True — XGBoost", fontsize=13, fontweight="bold")
    ax5.legend(); ax5.grid(alpha=0.3)

    # 5 · Residuals
    ax6 = fig.add_subplot(2, 3, 6)
    residuals = y_level_true - y_pred_level
    ax6.scatter(y_pred_level, residuals, alpha=0.5, s=20, color=MODEL_COLORS["RF"])
    ax6.axhline(0, color="red", linestyle="--", lw=2)
    ax6.set_xlabel("Predicted Severity"); ax6.set_ylabel("Residuals")
    ax6.set_title("Residual Plot — XGBoost", fontsize=13, fontweight="bold")
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    _savefig(fig, "leakage_detection_results.png", out_dir)


def plot_feature_importance(
    classifiers: dict, feature_names: list[str],
    model_type: str = "XGB", top_n: int = 20,
    out_dir: Path = RESULTS_DIR,
) -> None:
    clf = classifiers[model_type]
    if not hasattr(clf, "feature_importances_") or clf.feature_importances_ is None:
        logger.warning("No feature_importances_ on %s — skipping.", model_type)
        return

    importance = clf.feature_importances_
    indices    = np.argsort(importance)[::-1][:top_n]
    top_imp    = importance[indices]
    top_feat   = [feature_names[i] for i in indices]
    colors     = [MODEL_COLORS["RF"] if "Pressure" in f else MODEL_COLORS["ET"]
                  for f in top_feat]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(top_n), top_imp, color=colors, alpha=0.8, edgecolor="black")
    ax.set_yticks(range(top_n)); ax.set_yticklabels(top_feat, fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Features — {model_type}", fontsize=14, fontweight="bold")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[
        mpatches.Patch(facecolor=MODEL_COLORS["RF"], alpha=0.8, label="Pressure Features"),
        mpatches.Patch(facecolor=MODEL_COLORS["ET"], alpha=0.8, label="Flow Features"),
    ], loc="lower right")

    plt.tight_layout()
    _savefig(fig, f"feature_importance_{model_type}.png", out_dir)


def plot_leak_scenario(
    pressure_head: np.ndarray, flow_values: np.ndarray,
    leak_node: int, title: str = "Leak Scenario",
    out_dir: Path = RESULTS_DIR,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(pressure_head, "b-", lw=2, label="Pressure Head")
    ax1.axvline(x=leak_node, color="r", ls="--", lw=2, label="Leak Location")
    ax1.scatter([leak_node], [pressure_head[leak_node]], color="r", s=200, zorder=5)
    ax1.set_xlabel("Node Index"); ax1.set_ylabel("Pressure Head (m)")
    ax1.set_title("Nodal Pressures", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(flow_values, "g-", lw=2, label="Flow")
    ax2.set_xlabel("Pipe Index"); ax2.set_ylabel("Flow Rate")
    ax2.set_title("Pipe Flows", fontsize=14, fontweight="bold")
    ax2.grid(alpha=0.3); ax2.legend()

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "leak_scenario.png", out_dir)


def generate_report(
    cls_metrics: Dict[str, dict], reg_metrics: Dict[str, dict],
    n_nodes: int, n_pipes: int,
    out_dir: Path = RESULTS_DIR,
    filename: str = "leakage_detection_report.txt",
) -> None:
    
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    sep = "=" * 80; dash = "-" * 80
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\nLEAKAGE DETECTION FRAMEWORK — EVALUATION REPORT\n")
        f.write("Physics-Integrated GCNs · Soumyajit Banerjee · University of Calcutta\n")
        f.write(f"{sep}\n\nSYSTEM: {n_nodes} nodes | {n_pipes} pipes | {n_nodes+n_pipes} features\n\n")

        f.write(f"CLASSIFICATION — Leak Node\n{dash}\n")
        f.write(f"  {'Model':<8} {'Accuracy':>12}\n  {dash[:30]}\n")
        for name in ("RF", "ET", "XGB"):
            f.write(f"  {name:<8} {cls_metrics[name]['accuracy']*100:>10.2f}%\n")

        f.write(f"\nREGRESSION — Leak Severity\n{dash}\n")
        f.write(f"  {'Model':<8} {'MAE':>10} {'RMSE':>10} {'R²':>10}\n  {dash[:42]}\n")
        for name in ("RF", "ET", "XGB"):
            m = reg_metrics[name]
            f.write(f"  {name:<8} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>10.4f}\n")
            
        f.write(f"\n{sep}\n")

    logger.info("Report → %s", path)

