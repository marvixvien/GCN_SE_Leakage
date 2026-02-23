"""
models.py
---------
Ensemble models for leak-node classification and leak-severity regression.

Three algorithms trained for each task:
  RF  : Random Forest
  ET  : Extra Trees
  XGB : XGBoost
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

from src.config import ET_PARAMS, RF_PARAMS, XGB_PARAMS

logger = logging.getLogger(__name__)


def _build_classifiers() -> dict:
    return {
        "RF" : RandomForestClassifier(**RF_PARAMS),
        "ET" : ExtraTreesClassifier(**ET_PARAMS),
        "XGB": XGBClassifier(**XGB_PARAMS),
    }


def _build_regressors() -> dict:
    xgb_reg = {k: v for k, v in XGB_PARAMS.items() if k != "eval_metric"}
    return {
        "RF" : RandomForestRegressor(**RF_PARAMS),
        "ET" : ExtraTreesRegressor(**ET_PARAMS),
        "XGB": XGBRegressor(**xgb_reg),
    }


class EnsembleModels:
    """Holds and coordinates three classifier + three regressor pairs."""

    def __init__(self) -> None:
        self.classifiers: dict = _build_classifiers()
        self.regressors:  dict = _build_regressors()
        self.is_fitted: bool   = False

    def fit(
        self,
        X_train: np.ndarray, y_node_train: np.ndarray, y_level_train: np.ndarray,
        X_val:   np.ndarray, y_node_val:   np.ndarray, y_level_val:   np.ndarray,
    ) -> "EnsembleModels":
        """Train all classifiers and regressors. Returns self for chaining."""
        
        # 1. Initialize and fit the LabelEncoder
        self.le = LabelEncoder()
        
        # 2. Transform train and validation labels (1-50 becomes 0-49)
        y_node_train_encoded = self.le.fit_transform(y_node_train)
        y_node_val_encoded = self.le.transform(y_node_val)
        
        logger.info("=" * 60)
        logger.info("TRAINING CLASSIFIERS (leak node)")
        logger.info("=" * 60)
        for name, clf in self.classifiers.items():
            logger.info("  Training %s ...", name)
            
            # 3. Fit using the ENCODED training labels
            clf.fit(X_train, y_node_train_encoded)
            
            # 4. Score using the ENCODED validation labels
            val_acc = accuracy_score(y_node_val_encoded, clf.predict(X_val))
            logger.info("    %s  Validation Accuracy: %.2f%%", name, val_acc * 100)

        logger.info("=" * 60)
        logger.info("TRAINING REGRESSORS (leak severity)")
        logger.info("=" * 60)
        for name, reg in self.regressors.items():
            logger.info("  Training %s ...", name)
            reg.fit(X_train, y_level_train)
            y_pred    = reg.predict(X_val)
            val_mae   = mean_absolute_error(y_level_val, y_pred)
            val_rmse  = float(np.sqrt(mean_squared_error(y_level_val, y_pred)))
            logger.info("    %s  Val MAE: %.4f  RMSE: %.4f", name, val_mae, val_rmse)

        self.is_fitted = True
        logger.info("Training complete.")
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_node_test:  np.ndarray,
        y_level_test: np.ndarray,
    ) -> Tuple[Dict, Dict]:
        """Return (cls_metrics, reg_metrics) dicts for the held-out test set."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .evaluate().")

        cls_metrics: Dict[str, dict] = {}
        reg_metrics: Dict[str, dict] = {}

        logger.info("=" * 60)
        logger.info("TEST-SET EVALUATION")
        logger.info("=" * 60)

        if not hasattr(self, 'le'):
            self.le = LabelEncoder()
            # Hardcode the classes it should know about (1 through 50)
            self.le.fit(np.arange(1, 51))

        logger.info("--- Classification (Leak Node) ---")
        for name, clf in self.classifiers.items():
            raw_pred = clf.predict(X_test)
            y_pred = self.le.inverse_transform(raw_pred)
            acc    = accuracy_score(y_node_test, y_pred)
            cls_metrics[name] = {"accuracy": acc, "predictions": y_pred}
            logger.info("  %s  Accuracy: %.2f%%", name, acc * 100)

        logger.info("--- Regression (Leak Severity) ---")
        for name, reg in self.regressors.items():
            y_pred = reg.predict(X_test)
            mae    = mean_absolute_error(y_level_test, y_pred)
            rmse   = float(np.sqrt(mean_squared_error(y_level_test, y_pred)))
            r2     = float(r2_score(y_level_test, y_pred))
            reg_metrics[name] = {"mae": mae, "rmse": rmse, "r2": r2, "predictions": y_pred}
            logger.info("  %s  MAE: %.4f  RMSE: %.4f  R²: %.4f", name, mae, rmse, r2)

        header = f"{'Model':<8} {'Accuracy':>10} {'MAE':>10} {'RMSE':>10}"
        logger.info("\nSUMMARY\n%s\n%s", header, "-" * len(header))
        for name in ("RF", "ET", "XGB"):
            logger.info(
                "  %-6s %8.2f%%  %10.4f  %10.4f",
                name,
                cls_metrics[name]["accuracy"] * 100,
                reg_metrics[name]["mae"],
                reg_metrics[name]["rmse"],
            )

        return cls_metrics, reg_metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_single(self, x_scaled: np.ndarray, model_type: str = "XGB") -> dict:
        """Predict for one already-scaled feature vector."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() or load models before predicting.")
        if model_type not in self.classifiers:
            raise ValueError(f"Unknown model '{model_type}'. Choose RF, ET, or XGB.")

        X   = x_scaled.reshape(1, -1)
        clf = self.classifiers[model_type]
        reg = self.regressors[model_type]

        leak_node     = int(clf.predict(X)[0])
        proba         = clf.predict_proba(X)[0]
        class_index   = list(clf.classes_).index(leak_node)
        confidence    = float(proba[class_index])
        leak_severity = float(reg.predict(X)[0])

        return {
            "leak_node":     leak_node,
            "leak_severity": leak_severity,
            "confidence":    confidence,
            "model_used":    model_type,
        }

    def predict_ensemble(self, x_scaled: np.ndarray) -> dict:
        """Weighted-vote ensemble across all three models (XGB weight=0.40)."""
        from scipy import stats
        preds      = {m: self.predict_single(x_scaled, m) for m in ("RF", "ET", "XGB")}
        leak_nodes = [p["leak_node"]     for p in preds.values()]
        severities = [p["leak_severity"] for p in preds.values()]
        confs      = [p["confidence"]    for p in preds.values()]

        return {
            "leak_node":              int(stats.mode(leak_nodes, keepdims=True)[0][0]),
            "leak_severity":          float(np.mean(severities)),
            "confidence":             float(np.average(confs, weights=[0.30, 0.30, 0.40])),
            "individual_predictions": preds,
        }
