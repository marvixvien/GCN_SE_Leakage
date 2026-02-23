from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from src import data_loader, persistence, visualisation
from src.config import (
    ALERT_CONFIDENCE_THRESHOLD,
    FLOW_FILE,
    MODELS_DIR,
    PRESSURE_FILE,
    PRESSURE_SCALE,
    RESULTS_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from .data_loader import DataSplits
from .models import EnsembleModels

logger = logging.getLogger(__name__)


class LeakageDetector:
    def __init__(
        self,
        pressure_file: Path | str = PRESSURE_FILE,
        flow_file:     Path | str = FLOW_FILE,
        models_dir:    Path | str = MODELS_DIR,
        results_dir:   Path | str = RESULTS_DIR,
    ) -> None:
        self.pressure_file = Path(pressure_file)
        self.flow_file     = Path(flow_file)
        self.models_dir    = Path(models_dir)
        self.results_dir   = Path(results_dir)

        self._models: EnsembleModels       = EnsembleModels()
        self._splits: Optional[DataSplits] = None

    # ── Read-only properties ─────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._models.is_fitted

    # ── Pipeline steps ───────────────────────────────────────────────────────

    def load_data(
        self,
        test_size: float = TEST_SIZE,
        val_size:  float = VAL_SIZE,
    ) -> "LeakageDetector":
        """Read Excel files and build train/val/test splits. Returns self."""
        X, y_node, y_level = data_loader.load_raw_data(
            pressure_file=self.pressure_file,
            flow_file=self.flow_file,
        )
        self._splits = data_loader.prepare_splits(
            X, y_node, y_level,
            test_size=test_size,
            val_size=val_size,
        )
        return self

    def train(self) -> "LeakageDetector":
        """Train all six ensemble models. Call load_data() first. Returns self."""
        self._require_splits()
        s = self._splits
        self._models.fit(
            s.X_train, s.y_node_train, s.y_level_train,
            s.X_val,   s.y_node_val,   s.y_level_val,
        )
        return self

    def evaluate(
        self,
        save_plots:  bool = True,
        save_report: bool = True,
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate on the held-out test set.
        Optionally saves plots and a text report to results/.
        Returns (cls_metrics, reg_metrics).
        """
        self._require_fitted()
        self._require_splits()
        s = self._splits

        cls_metrics, reg_metrics = self._models.evaluate(
            s.X_test, s.y_node_test, s.y_level_test,
        )

        if save_plots:
            visualisation.plot_performance_overview(
                cls_metrics, reg_metrics,
                s.y_node_test, s.y_level_test,
                out_dir=self.results_dir,
            )
            visualisation.plot_feature_importance(
                self._models.classifiers,
                s.feature_names,
                model_type="XGB",
                top_n=20,
                out_dir=self.results_dir,
            )

        if save_report:
            visualisation.generate_report(
                cls_metrics, reg_metrics,
                n_nodes=s.n_nodes,
                n_pipes=s.n_pipes,
                out_dir=self.results_dir,
            )

        return cls_metrics, reg_metrics

    def save(self, prefix: str = "leakage_model") -> "LeakageDetector":
        """Persist trained models to models/. Returns self."""
        self._require_fitted()
        self._require_splits()
        persistence.save(self._models, self._splits, prefix=prefix, out_dir=self.models_dir)
        return self

    def load(self, prefix: str = "leakage_model") -> "LeakageDetector":
        """
        Restore trained models from models/.
        load_data() is NOT required before calling this for inference.
        Returns self.
        """
        if self._splits is None:
            self._splits = DataSplits(
                X_train=np.empty((0,)), X_val=np.empty((0,)), X_test=np.empty((0,)),
                y_node_train=np.empty((0,)), y_node_val=np.empty((0,)), y_node_test=np.empty((0,)),
                y_level_train=np.empty((0,)), y_level_val=np.empty((0,)), y_level_test=np.empty((0,)),
                scaler=StandardScaler(),
                feature_names=[],
            )
        persistence.load(self._models, self._splits, prefix=prefix, src_dir=self.models_dir)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        pressure_head: np.ndarray,
        flow_values:   np.ndarray,
        model_type:    str  = "XGB",
        use_ensemble:  bool = False,
    ) -> Dict:
        """
        Predict leak node and severity for one network snapshot.

        Parameters
        ----------
        pressure_head : nodal hydraulic heads in metres (GCN output).
        flow_values   : pipe flow rates (any consistent unit matching training).
        model_type    : "RF", "ET", or "XGB".
        use_ensemble  : combine all three models via weighted voting.

        Returns
        -------
        dict with keys: leak_node (int), leak_severity (float),
        confidence (float), and optionally LEAK_ALERT (str).
        """
        self._require_fitted()
        self._require_splits()

        pressure_kpa = pressure_head * PRESSURE_SCALE
        x_raw    = np.hstack([pressure_kpa, flow_values])
        x_scaled = self._splits.scaler.transform(x_raw.reshape(1, -1)).ravel()

        result = (self._models.predict_ensemble(x_scaled)
                  if use_ensemble
                  else self._models.predict_single(x_scaled, model_type=model_type))

        if result["confidence"] >= ALERT_CONFIDENCE_THRESHOLD:
            result["LEAK_ALERT"] = (
                f"LEAK DETECTED at Node {int(float(result["leak_node"]))+1}  |  "
                f"Severity: {result['leak_severity']:.4f}  |  "
                f"Confidence: {result['confidence']*100:.1f}%"
            )
            logger.warning(result["LEAK_ALERT"])

        return result

    def visualise_scenario(
        self,
        pressure_head: np.ndarray,
        flow_values:   np.ndarray,
        leak_node:     int,
        title:         str = "Leak Scenario",
    ) -> None:
        """Save a pressure/flow side-by-side plot for one snapshot."""
        visualisation.plot_leak_scenario(
            pressure_head, flow_values, leak_node,
            title=title, out_dir=self.results_dir,
        )

    # ── Guards ───────────────────────────────────────────────────────────────

    def _require_splits(self) -> None:
        if self._splits is None:
            raise RuntimeError(
                "No data loaded.\n"
                "  For training:   call det.load_data() first.\n"
                "  For inference:  call det.load() to restore saved models."
            )

    def _require_fitted(self) -> None:
        if not self._models.is_fitted:
            raise RuntimeError(
                "Models are not trained.\n"
                "  Run  python train.py  to train, or call det.load() to restore."
            )
