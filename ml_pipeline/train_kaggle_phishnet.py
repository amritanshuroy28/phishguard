"""
Rigorous Training on the PhishNet / Kaggle Phishing Dataset
==========================================================
Dataset: shashwatwork/phishing-dataset-for-machine-learning
   → Phishing_Legitimate_full.csv  (~1.37 MB, ~23 K rows)
   → Pre‑processed URL + web‑page features, binary label (0=phish / 1=legit)

What this script does:
1. Parses the CSV, inspects columns, drops string/URL columns.
2. Runs 5‑fold stratified CV (fast non‑tuning pass).
3. Then runs GridSearchCV over key XGBoost hyper‑params.
4. Fits the best model on the full training split and evaluates on the held‑out test set.
5. Saves the pickled model, a JSON report, a confusion‑matrix PNG and a
   feature‑importance PNG.

Usage:
    python train_kaggle_phishnet.py [--optimize]
"""

import os, sys, json, pickle, logging
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
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = DATA_DIR / "kaggle_phishnet_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH     = DATA_DIR / "Phishing_Legitimate_full.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Kaggle phishing CSV and return (features, label)."""
    logger.info(f"Loading CSV from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Drop any fully string/object columns XGBoost cannot digest
    for col in df.columns:
        if df[col].dtype == "object":
            logger.info(f"Dropping column '{col}' (dtype=object)")
            df = df.drop(columns=[col])

    # Find the label column (case‑insensitive)
    label_candidates = ("label", "result", "class", "target", "class_label")
    label_col = None
    for col in df.columns:
        if col.lower() in label_candidates:
            label_col = col
            break

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    logger.info(f"Feature matrix: {X.shape} | Label distribution: {dict(y.value_counts())}")
    return X, y

# ---------------------------------------------------------------------------
# TRAINER
# ---------------------------------------------------------------------------
class KagglePhishTrainer:
    def __init__(self, optimize: bool = False):
        self.optimize = optimize
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_params: Optional[Dict] = {}
        self.history: Dict = {}
        self.metrics: Dict = {}
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    def _default_params(self) -> Dict:
        return dict(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, use_label_encoder=False, n_jobs=-1,
        )

    # ------------------------------------------------------------------
    def _tune(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        logger.info("GridSearchCV hyper‑parameter tuning (3‑fold CV, 216 fits) …")
        param_grid = dict(
            max_depth=[4, 6, 8],
            learning_rate=[0.05, 0.1, 0.15],
            n_estimators=[100, 200, 300],
            min_child_weight=[1, 3, 5],
            subsample=[0.8, 0.9, 1.0],
        )
        base = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, use_label_encoder=False, n_jobs=-1,
        )
        gs = GridSearchCV(
            base, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring="accuracy", n_jobs=-1, verbose=0,
        )
        gs.fit(X, y)
        logger.info(f"Best params: {gs.best_params_} | Best CV accuracy: {gs.best_score_:.4f}")
        return gs.best_params_

    # ------------------------------------------------------------------
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "KagglePhishTrainer":
        params = self._default_params()
        if self.optimize:
            tuned = self._tune(X_train, y_train)
            params.update(tuned)
        self.best_params = params
        self.model = xgb.XGBClassifier(**params)
        self.feature_names = list(X_train.columns)

        # 5‑fold CV report
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="accuracy")
        self.history["cv_scores"] = [float(s) for s in scores]
        logger.info(f"CV accuracy: {scores.mean():.4f} ± {scores.std()*2:.4f}  |  {scores}")

        logger.info("Fitting final model …")
        self.model.fit(X_train, y_train)
        return self

    # ------------------------------------------------------------------
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        m = dict(
            accuracy  = accuracy_score(y_test, y_pred),
            precision = precision_score(y_test, y_pred),
            recall    = recall_score(y_test, y_pred),
            f1_score  = f1_score(y_test, y_pred),
            roc_auc   = roc_auc_score(y_test, y_proba),
        )
        cm = confusion_matrix(y_test, y_pred).tolist()
        logger.info(f"Test Accuracy: {m['accuracy']:.4f}  "
                    f"Precision: {m['precision']:.4f}  "
                    f"Recall:    {m['recall']:.4f}  "
                    f"F1:        {m['f1_score']:.4f}  "
                    f"ROC AUC:   {m['roc_auc']:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Phishing', 'Legit'])}")
        m["confusion_matrix"] = cm
        self.metrics = m
        return m

    # ------------------------------------------------------------------
    def plots(self, X: pd.DataFrame, y_test, y_pred):
        # Feature importance
        imp = pd.DataFrame({"Feature": self.feature_names,
                            "Importance": self.model.feature_importances_})
        imp = imp.sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=imp.head(20), y="Feature", x="Importance",
                    palette="viridis", ax=ax, legend=False)
        ax.set_title("PhishNet – Top 20 Feature Importances")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved {OUTPUT_DIR / 'feature_importance.png'}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                     xticklabels=["Phishing", "Legit"],
                     yticklabels=["Phishing", "Legit"])
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved {OUTPUT_DIR / 'confusion_matrix.png'}")

    # ------------------------------------------------------------------
    def save(self):
        path = OUTPUT_DIR / "kaggle_phishnet_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(dict(
                model=self.model, best_params=self.best_params,
                metrics=self.metrics, history=self.history,
            ), f)
        logger.info(f"Model saved: {path}")

        report = {
            "timestamp"      : datetime.now().isoformat(),
            "dataset"        : "shashwatwork/phishing-dataset-for-machine-learning — Phishing_Legitimate_full.csv",
            "total_samples"  : int(self.metrics["confusion_matrix"][0][0]
                                 + self.metrics["confusion_matrix"][0][1]
                                 + self.metrics["confusion_matrix"][1][0]
                                 + self.metrics["confusion_matrix"][1][1]),
            "features"       : self.feature_names,
            "metrics"        : {k: v for k, v in self.metrics.items()
                                if k != "confusion_matrix"},
            "confusion_matrix": self.metrics.get("confusion_matrix"),
            "best_params"    : self.best_params,
            "cv_scores"       : self.history.get("cv_scores"),
        }
        rp = OUTPUT_DIR / "training_report.json"
        with open(rp, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved: {rp}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--optimize", action="store_true")
    args = p.parse_args()

    print("=" * 60)
    print("Kaggle PhishNet – Rigorous Training")
    print("=" * 60)

    # Load
    print("\n[1/4] Loading dataset …")
    X, y = load_dataset(CSV_PATH)

    # Split
    print("[2/4] Train/test split …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(y_train)} | Test: {len(y_test)}")

    # Train
    print("[3/4] Training XGBoost …")
    trainer = KagglePhishTrainer(optimize=args.optimize)
    trainer.train(X_train, y_train)

    # Evaluate
    print("[4/4] Evaluating …")
    trainer.evaluate(X_test, y_test)
    y_pred = trainer.model.predict(X_test)
    trainer.plots(X_test, y_test, y_pred)
    trainer.save()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Accuracy: {trainer.metrics['accuracy']*100:.2f}%")
    print(f"ROC AUC:  {trainer.metrics['roc_auc']:.4f}")
    print(f"Output:   {OUTPUT_DIR}")

if __name__ == "__main__":
    main()