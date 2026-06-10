"""
Rigorous Training: UCI Phishing Websites Dataset (11,055 samples)
===================================================================
Trains an XGBoost classifier on the UCI Phishing Websites dataset
following the approach from the Kaggle notebook, with enhancements:
- Hyperparameter tuning via GridSearchCV
- Stratified K-Fold cross-validation
- Multiple evaluation metrics
- Feature importance analysis
- Model serialization

Dataset: UCI Machine Learning Repository - Phishing Websites Dataset
Author notebook: akashkr/phishing-url-eda-and-modelling
Reference accuracy (basic XGB, 4-fold): ~97%
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

from scipy.io import arff

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "uci_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ARFF dataset path (downloaded from UCI)
ARFF_PATH = Path(__file__).parent / "data" / "phishing_web_extract" / "Training Dataset.arff"

# Training config
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ============================================================================
# DATA LOADING
# ============================================================================

def load_arff_dataset(path: Path) -> pd.DataFrame:
    """Load ARFF file and return a pandas DataFrame."""
    logger.info(f"Loading ARFF dataset from: {path}")
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    # Decode byte columns to strings then to numeric
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")
            df[col] = pd.to_numeric(df[col])
    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    return df


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the dataset.
    Maps Result: -1 (phishing) -> 0, 1 (legitimate) -> 1
    """
    logger.info("Preprocessing dataset...")
    # Separate features and target
    target = df["Result"].copy()
    features = df.drop(columns=["Result"])

    # Map target as in the notebook: -1 -> 0 (phishing), 1 -> 1 (legitimate)
    target = target.map({-1: 0, 1: 1})

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Target distribution:\n{target.value_counts()}")

    return features, target


# ============================================================================
# MODEL TRAINING
# ============================================================================

class UCIPhishingTrainer:
    """Trainer for the UCI Phishing Websites dataset."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.metrics = {}
        self.history = {}

    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Run GridSearchCV for XGBoost hyperparameters."""
        logger.info("Running hyperparameter tuning (GridSearchCV)...")

        param_grid = {
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [100, 200, 300],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
        }

        base = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            use_label_encoder=False,
        )

        grid = GridSearchCV(
            base,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X, y)

        logger.info(f"Best params: {grid.best_params_}")
        logger.info(f"Best CV accuracy: {grid.best_score_:.4f}")
        self.best_params = grid.best_params_
        return grid.best_params_

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              tune: bool = False) -> xgb.XGBClassifier:
        """Train the model."""
        if tune:
            best = self.hyperparameter_tuning(X_train, y_train)
            params = {**best, "objective": "binary:logistic",
                      "eval_metric": "logloss", "random_state": RANDOM_STATE,
                      "use_label_encoder": False, "n_jobs": -1}
        else:
            params = {
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
            self.best_params = params

        self.model = xgb.XGBClassifier(**params)

        # Cross-validation
        logger.info(f"Running {CV_FOLDS}-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="accuracy")
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        self.history["cv_scores"] = cv_scores.tolist()

        # Train final model on full training set
        logger.info("Training final model...")
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance."""
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

        logger.info(f"Test Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall:    {metrics['recall']:.4f}")
        logger.info(f"Test F1:        {metrics['f1_score']:.4f}")
        logger.info(f"Test ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Phishing', 'Legitimate'])}")

        self.metrics = metrics
        return metrics

    def plot_feature_importance(self, feature_names: List[str]):
        """Generate and save feature importance plot."""
        importance = self.model.feature_importances_
        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        df_imp = df_imp.sort_values("Importance", ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_imp.head(20), y="Feature", x="Importance", palette="viridis")
        plt.title("UCI Phishing Dataset: Top 20 Feature Importances (XGBoost)", fontsize=14)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        out_path = OUTPUT_DIR / "feature_importance_uci.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Feature importance plot saved to: {out_path}")

    def plot_confusion_matrix(self, y_test: pd.Series, y_pred):
        """Generate and save confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Phishing", "Legitimate"],
                    yticklabels=["Phishing", "Legitimate"])
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        out_path = OUTPUT_DIR / "confusion_matrix_uci.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix plot saved to: {out_path}")

    def save_model(self, filename: str = "uci_phishing_model.pkl"):
        """Save the trained model."""
        path = OUTPUT_DIR / filename
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "best_params": self.best_params,
                "metrics": self.metrics,
                "history": self.history,
            }, f)
        logger.info(f"Model saved to: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("UCI PHISHING WEBSITES DATASET - RIGOROUS TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # 1. Load data
    print("[1/5] Loading ARFF dataset...")
    df = load_arff_dataset(ARFF_PATH)

    # 2. Preprocess
    print("[2/5] Preprocessing...")
    features, target = preprocess(df)

    # 3. Split
    print("[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. Train
    print("[4/5] Training XGBoost...")
    trainer = UCIPhishingTrainer()
    trainer.train(X_train, y_train, tune=False)  # toggled off for speed; set True for tuning

    # 5. Evaluate
    print("[5/5] Evaluating...")
    metrics = trainer.evaluate(X_test, y_test)

    # Plots
    y_pred = trainer.model.predict(X_test)
    trainer.plot_feature_importance(list(features.columns))
    trainer.plot_confusion_matrix(y_test, y_pred)

    # Save
    trainer.save_model()

    # JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "UCI Phishing Websites (11,055 rows)",
        "features": list(features.columns),
        "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics.get("confusion_matrix"),
        "best_params": trainer.best_params,
        "cv_scores": trainer.history.get("cv_scores"),
    }
    with open(OUTPUT_DIR / "uci_training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"Final ROC AUC:      {metrics['roc_auc']:.4f}")
    print(f"Outputs saved to:   {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
