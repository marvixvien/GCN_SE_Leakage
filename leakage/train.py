import logging
import time
import sys
from pathlib import Path

from src.detector import LeakageDetector

_HERE = Path(__file__).parent.resolve()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    sep = "=" * 70
    logger.info(sep)
    logger.info("WDS LEAKAGE DETECTION — TRAINING PIPELINE")
    logger.info("Physics-Integrated GCNs · Soumyajit Banerjee · Univ. of Calcutta")
    logger.info(sep)
    logger.info("Working directory : %s", Path.cwd())
    logger.info("Script directory  : %s", _HERE)
    logger.info(sep)

    t0 = time.perf_counter()

    # ── 1 · Load data ─────────────────────────────────────────────────────────
    logger.info("\n[1/4]  Loading data ...")
    det = LeakageDetector()
    det.load_data()

    # ── 2 · Train models ──────────────────────────────────────────────────────
    logger.info("\n[2/4]  Training ensemble models ...")
    det.train()

    # ── 3 · Evaluate ──────────────────────────────────────────────────────────
    logger.info("\n[3/4]  Evaluating on held-out test set ...")
    det.evaluate(save_plots=True, save_report=True)

    # ── 4 · Save ──────────────────────────────────────────────────────────────
    logger.info("\n[4/4]  Saving trained models ...")
    det.save(prefix="leakage_model")

    elapsed = time.perf_counter() - t0
    logger.info("\n%s", sep)
    logger.info("TRAINING COMPLETE  (%.1f s)", elapsed)
    logger.info(sep)
    logger.info("Outputs:")
    logger.info("  %s/models/   — model artefacts", _HERE)
    logger.info("  %s/results/  — plots and report", _HERE)
    logger.info("\nNext step:  python predict.py")


if __name__ == "__main__":
    main()
