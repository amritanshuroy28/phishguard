"""
Training PhishGuard model using the UCIMLR dataset (fetch_ucirepo)
===================================================================
This script demonstrates a rigorous training pipeline that:
- Pulls the Phishing Websites dataset (id=967) via the `ucimlrepo` Python package.
- Performs the same preprocessing as our original UCI ARFF loader (maps target -1 → 0, 1 → 1).
- Runs a stratified 5‑fold cross‑validation (default) and reports mean accuracy.
- Trains a final XGBoost model (same hyper‑parameters as the previous training script).
- Evaluates on a held‑out test split.
- Saves the trained model, feature‑importance plot, confusion matrix, and a JSON report.

The script can be invoked as:
    python ml_pipeline/train_uci_fetch.py [--optimize]
If `--optimize` is supplied, a quick GridSearchCV will be performed to tune a few
hyper‑parameters before the final model is trained.
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb

# Ensure the `ucimlrepo` package is available in the current environment.
# If it is not installed, we install it on‑the‑fly.
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
    from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "uci_fetch_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_ucimlr_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    # Load data and drop non‑numeric columns (URL, Domain, TLD, Title) which XGBoost cannot handle directly
    """Fetch the Phishing Websites dataset via ucimlrepo and return X, y.
    The repository returns:
        - `data.features`  : pandas DataFrame of 30 feature columns
        - `data.targets`   : pandas Series with values -1 (phishing) / 1 (legitimate)
    We map -1 → 0 (phishing) and 1 → 1 (legitimate) to match our XGBoost binary label.
    """
    logger.info("Fetching UCIMLR Phishing Websites dataset (id=967)…")
    dataset = fetch_ucirepo(id=967)
    # Dataset metadata (optional print for debugging)
    logger.debug("Metadata: %s", dataset.metadata)
    logger.debug("Variables: %s", dataset.variables)

    X = dataset.data.features.copy()
    y_raw = dataset.data.targets.copy()
    # Drop string/object columns (URL, Domain, TLD, Title) – XGBoost only accepts numeric types
    X = X.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"])
    # Map -1 → 0, 1 → 1
    y = y_raw.replace({-1: 0, 1: 1})
    logger.info(f"Loaded {X.shape[0]} rows × {X.shape[1]} features")
    logger.info(f"Target distribution: \n{y.value_counts()}" )
    return X, y

# ---------------------------------------------------------------------------
# MODEL TRAINING CLASS
# ---------------------------------------------------------------------------
class UCITrainer:
    def __init__(self, optimize: bool = False):
        self.optimize = optimize
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_params: Optional[Dict] = None
        self.history: Dict = {}
        self.metrics: Dict = {}

    def _default_params(self) -> Dict:
        return {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "use_label_encoder": False,
            "n_jobs": -1,
        }

    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        logger.info("Running quick GridSearchCV hyper‑parameter tuning…")
        param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.15],
            "n_estimators": [100, 200, 300],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.8, 0.9, 1.0],
        }
        base = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            n_jobs=-1,
        )
        gs = GridSearchCV(
            base,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X, y)
        logger.info(f"Best params from tuning: {gs.best_params_}")
        logger.info(f"Best CV accuracy: {gs.best_score_: .4f}")
        return gs.best_params_

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        params = self._default_params()
        if self.optimize:
            tuned = self._tune_hyperparameters(X_train, y_train)
            params.update(tuned)
        self.best_params = params
        self.model = xgb.XGBClassifier(**params)
        # Cross‑validation (reporting only)
        logger.info(f"Running {CV_FOLDS}-fold stratified CV on training data…")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="accuracy")
        self.history["cv_scores"] = cv_scores.tolist()
        logger.info(f"CV mean accuracy: {cv_scores.mean():.4f} (+‑ {cv_scores.std()*2:.4f})")
        # Fit final model
        logger.info("Fitting final model on full training set…")
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        logger.info("Test set metrics:\n" + json.dumps(metrics, indent=2))
        self.metrics = metrics
        return metrics

    def plot_feature_importance(self, feature_names: List[str]):
        importance = self.model.feature_importances_
        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        df_imp = df_imp.sort_values("Importance", ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_imp.head(20), y="Feature", x="Importance", palette="viridis")
        plt.title("UCI Phishing – Top 20 Feature Importances (XGBoost)")
        plt.tight_layout()
        out_path = OUTPUT_DIR / "feature_importance.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Feature‑importance plot saved to {out_path}")

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Phishing", "Legitimate"],
                    yticklabels=["Phishing", "Legitimate"])
        plt.title("Confusion Matrix – Test Set")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        out_path = OUTPUT_DIR / "confusion_matrix.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Confusion‑matrix plot saved to {out_path}")

    def save_artifacts(self):
        model_path = OUTPUT_DIR / "phishguard_uci_fetch.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "best_params": self.best_params,
                "metrics": self.metrics,
                "history": self.history,
            }, f)
        logger.info(f"Model serialized to {model_path}")
        # JSON report
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "UCIMLR Phishing Websites (id=967)",
            "features": list(self.model.get_booster().feature_names),
            "metrics": {k: v for k, v in self.metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": self.metrics.get("confusion_matrix"),
            "best_params": self.best_params,
            "cv_scores": self.history.get("cv_scores"),
        }
        report_path = OUTPUT_DIR / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Training report written to {report_path}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train PhishGuard on UCIMLR Phishing dataset")
    parser.add_argument("--optimize", action="store_true", help="Run quick hyper‑parameter tuning before final training")
    args = parser.parse_args()

    print("=" * 70)
    print("UCI Phishing Websites – Rigorous Training (fetch_ucirepo)")
    print("=" * 70)

    # Load data
    X, y = load_ucimlr_dataset()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train/Test split – train: {len(y_train)}, test: {len(y_test)}")

    # Train
    trainer = UCITrainer(optimize=args.optimize)
    trainer.train(X_train, y_train)

    # Evaluate
    trainer.evaluate(X_test, y_test)

    # Plots
    trainer.plot_feature_importance(list(X.columns))
    y_pred = trainer.model.predict(X_test)
    trainer.plot_confusion_matrix(y_test, y_pred)

    # Save artifacts
    trainer.save_artifacts()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final Accuracy: {trainer.metrics['accuracy']*100:.2f}%")
    print(f"Outputs stored in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
