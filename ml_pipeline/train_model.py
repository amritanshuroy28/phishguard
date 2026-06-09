"""
PhishGuard Model Training Pipeline
===================================
Trains an XGBoost classifier for phishing URL detection.

This module handles:
- Dataset loading and feature extraction
- Model training with hyperparameter optimization
- Evaluation metrics calculation
- Model serialization for deployment

Target: >= 95% accuracy on balanced dataset
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Local imports
from features.feature_extraction import URLFeatureExtractor
from dataset_loader import DatasetLoader, FeatureExtractionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model output paths
MODEL_PATH = MODEL_DIR / "phishguard_model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.json"
EVALUATION_PATH = DATA_DIR / "evaluation_metrics.json"

# Training configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "phishing_samples": 2000,
    "legitimate_samples": 2000,
}

# XGBoost hyperparameters (will be optimized if optimization_enabled=True)
HYPERPARAMETERS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "scale_pos_weight": 1,  # Adjusted for class imbalance if any
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": 42,
}


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

FEATURE_NAMES = [
    "url_length", "path_length", "query_length", "fragment_length",
    "subdomain_count", "subdomain_length", "path_depth", "url_entropy",
    "domain_length", "is_free_domain_provider",
    "has_hex_encoding", "hex_encoded_chars", "has_ip_address",
    "has_at_symbol", "has_double_slash_redirect", "has_data_uri",
    "obfuscation_score", "suspicious_path_count", "has_suspicious_path",
    "has_login_keywords", "has_brand_in_subdomain", "suspicious_pattern_score",
    "special_char_count", "digit_count", "digit_ratio", "uppercase_count",
    "has_unicode", "punycode_detected", "domain_age_days", "is_recent_domain",
    "registrar_suspicious", "dns_record_exists", "typosquatting_score"
]


# ============================================================================
# MODEL TRAINING CLASS
# ============================================================================

class PhishGuardTrainer:
    """
    Complete training pipeline for PhishGuard ML model.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the trainer.

        Args:
            config: Optional configuration overrides
        """
        self.config = {**TRAINING_CONFIG, **(config or {})}
        self.model = None
        self.scaler = None
        self.feature_names = FEATURE_NAMES
        self.evaluation_metrics = {}
        self.training_history = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "hyperparameters": HYPERPARAMETERS,
        }

        logger.info("PhishGuard Trainer initialized")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load dataset and extract features.

        Returns:
            Tuple of (X, y, labels) where labels are URLs for reference
        """
        logger.info("Loading dataset...")

        # Initialize loader and feature extractor
        loader = DatasetLoader()
        pipeline = FeatureExtractionPipeline()

        # Load combined dataset
        phishing_count = self.config["phishing_samples"]
        legitimate_count = self.config["legitimate_samples"]

        dataset = loader.load_combined_dataset(
            phishing_count=phishing_count,
            legitimate_count=legitimate_count
        )

        logger.info(f"Loaded {dataset.size} URLs "
                   f"(Phishing: {dataset.phishing_count}, "
                   f"Legitimate: {dataset.legitimate_count})")

        # Extract features
        logger.info("Extracting features...")
        X, valid_urls = pipeline.extract_features(dataset.urls, show_progress=True)

        # Get corresponding labels
        url_to_label = dict(zip(dataset.urls, dataset.labels))
        y = np.array([url_to_label[url] for url in valid_urls], dtype=np.int32)

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Labels: {len(y)} (Phishing: {sum(y)}, Legitimate: {len(y) - sum(y)})")

        # Store class distribution
        self.training_history["dataset"] = {
            "total_samples": len(y),
            "phishing_samples": int(sum(y)),
            "legitimate_samples": int(len(y) - sum(y)),
            "feature_count": int(X.shape[1]),
        }

        return X, y, valid_urls

    def split_data(self, X: np.ndarray, y: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            X_train, X_test, y_train, y_test
        """
        test_size = self.config["test_size"]
        random_state = self.config["random_state"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )

        logger.info(f"Train/Test split: {len(y_train)}/{len(y_test)} samples")

        self.training_history["split"] = {
            "train_size": len(y_train),
            "test_size": len(y_test),
            "test_ratio": test_size,
        }

        return X_train, X_test, y_train, y_test

    def preprocess(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features (scaling, handling NaN/Inf).

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Preprocessed X_train, X_test
        """
        # Replace NaN and Inf with 0
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features (XGBoost doesn't strictly need scaling, but helps)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("Features preprocessed (NaN/Inf handled, scaled)")

        return X_train_scaled, X_test_scaled

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              optimize_hyperparameters: bool = False) -> xgb.XGBClassifier:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            optimize_hyperparameters: Whether to run hyperparameter optimization

        Returns:
            Trained model
        """
        logger.info("=" * 60)
        logger.info("TRAINING XGBOOST MODEL")
        logger.info("=" * 60)

        model_params = HYPERPARAMETERS.copy()

        if optimize_hyperparameters:
            logger.info("Running hyperparameter optimization (this may take a while)...")
            best_params = self._optimize_hyperparameters(X_train, y_train)
            model_params.update(best_params)
            self.training_history["hyperparameters"] = model_params

        # Create model
        self.model = xgb.XGBClassifier(**model_params)

        # Cross-validation on training set
        logger.info("Running cross-validation...")
        cv_folds = self.config["cv_folds"]
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='accuracy',
            verbose=1
        )

        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        self.training_history["cross_validation"] = {
            "folds": cv_folds,
            "scores": cv_scores.tolist(),
            "mean_accuracy": float(cv_scores.mean()),
            "std_accuracy": float(cv_scores.std()),
        }

        # Train final model
        logger.info("Training final model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Store training info
        self.training_history["model"] = {
            "type": "XGBoostClassifier",
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "learning_rate": model_params["learning_rate"],
        }

        logger.info("Model training complete!")

        return self.model

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Optimize hyperparameters using grid search.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Best hyperparameters dict
        """
        # Define parameter grid (reduced for faster training)
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [50, 100, 150],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
        }

        # Quick optimization with reduced CV folds
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            verbosity=0
        )

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,  # Reduced for speed
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV accuracy: {grid_search.best_score_:.4f}")

        return grid_search.best_params_

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("EVALUATING MODEL")
        logger.info("=" * 60)

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        # Detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["classification_report"] = report

        # Print results
        logger.info(f"\n{'='*50}")
        logger.info("EVALUATION RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn:4d}  FP: {fp:4d}")
        logger.info(f"  FN: {fn:4d}  TP: {tp:4d}")

        logger.info(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

        # Check target accuracy
        if metrics["accuracy"] >= 0.95:
            logger.info(f"\n✓ TARGET ACHIEVED: Accuracy >= 95% ({metrics['accuracy']*100:.2f}%)")
        else:
            logger.warning(f"\n✗ TARGET NOT MET: Accuracy < 95% ({metrics['accuracy']*100:.2f}%)")

        self.evaluation_metrics = metrics
        self.training_history["evaluation"] = metrics

        return metrics

    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                  top_n: int = 15) -> Dict[str, float]:
        """
        Analyze and visualize feature importance.

        Args:
            X: Features (all or training)
            y: Labels (all or training)
            top_n: Number of top features to display

        Returns:
            Dictionary of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        # Get feature importance
        importance = self.model.feature_importances_
        importance_dict = {
            name: float(imp) for name, imp in zip(self.feature_names, importance)
        }

        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        logger.info("\n" + "="*50)
        logger.info("TOP FEATURE IMPORTANCES")
        logger.info("="*50)
        for name, imp in sorted_importance[:top_n]:
            logger.info(f"  {name:30s}: {imp:.4f}")

        # Create importance DataFrame
        df_importance = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])

        # Save to CSV
        importance_csv_path = DATA_DIR / "feature_importance.csv"
        df_importance.to_csv(importance_csv_path, index=False)
        logger.info(f"\nFeature importance saved to: {importance_csv_path}")

        # Generate visualizations
        self._plot_feature_importance(df_importance.head(top_n))
        self._plot_confusion_matrix()

        self.training_history["feature_importance"] = dict(sorted_importance[:top_n])

        return importance_dict

    def _plot_feature_importance(self, df: pd.DataFrame):
        """Generate feature importance plot."""
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df)))[::-1]

        ax = sns.barplot(
            data=df,
            y='Feature',
            x='Importance',
            palette=colors,
            orient='h'
        )

        plt.title('PhishGuard: Feature Importance for Phishing Detection',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)

        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['Importance'] + 0.005, i,
                   f"{row['Importance']:.3f}",
                   va='center', fontsize=9)

        plt.tight_layout()

        # Save figure
        fig_path = DATA_DIR / "feature_importance.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to: {fig_path}")
        plt.close()

    def _plot_confusion_matrix(self):
        """Generate confusion matrix plot."""
        if not self.evaluation_metrics:
            return

        cm_data = self.evaluation_metrics["confusion_matrix"]
        cm = np.array([
            [cm_data["true_negatives"], cm_data["false_positives"]],
            [cm_data["false_negatives"], cm_data["true_positives"]]
        ])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'],
                   annot_kws={'size': 16})

        plt.title('PhishGuard: Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)

        plt.tight_layout()

        fig_path = DATA_DIR / "confusion_matrix.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to: {fig_path}")
        plt.close()

    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save the trained model and related objects.

        Args:
            path: Custom save path (default: MODEL_PATH)

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        path = path or MODEL_PATH

        model_package = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config,
            "training_history": self.training_history,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_package, f)

        logger.info(f"Model saved to: {path}")

        # Save evaluation metrics
        if self.evaluation_metrics:
            with open(EVALUATION_PATH, 'w') as f:
                json.dump(self.evaluation_metrics, f, indent=2)
            logger.info(f"Evaluation metrics saved to: {EVALUATION_PATH}")

        # Save feature names
        with open(FEATURES_PATH, 'w') as f:
            json.dump({"feature_names": self.feature_names}, f, indent=2)
        logger.info(f"Feature names saved to: {FEATURES_PATH}")

        return path

    def load_model(self, path: Optional[Path] = None) -> xgb.XGBClassifier:
        """
        Load a saved model.

        Args:
            path: Path to saved model file

        Returns:
            Loaded model
        """
        path = path or MODEL_PATH

        with open(path, 'rb') as f:
            model_package = pickle.load(f)

        self.model = model_package["model"]
        self.scaler = model_package.get("scaler")
        self.feature_names = model_package.get("feature_names", FEATURE_NAMES)
        self.config = model_package.get("config", TRAINING_CONFIG)
        self.training_history = model_package.get("training_history", {})

        logger.info(f"Model loaded from: {path}")

        return self.model

    def generate_training_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a summary report of the training run.

        Args:
            output_path: Path to save report (default: DATA_DIR/report.txt)

        Returns:
            Report text
        """
        output_path = output_path or (DATA_DIR / "training_report.txt")

        report_lines = [
            "=" * 70,
            "PHISHGUARD ML MODEL TRAINING REPORT",
            "=" * 70,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 70,
            "1. DATASET SUMMARY",
            "-" * 70,
        ]

        if "dataset" in self.training_history:
            ds = self.training_history["dataset"]
            report_lines.extend([
                f"   Total Samples: {ds['total_samples']}",
                f"   Phishing URLs:  {ds['phishing_samples']}",
                f"   Legitimate URLs:{ds['legitimate_samples']}",
                f"   Features:      {ds['feature_count']}",
            ])

        report_lines.extend(["", "-" * 70, "2. MODEL CONFIGURATION", "-" * 70])

        if "hyperparameters" in self.training_history:
            for key, value in self.training_history["hyperparameters"].items():
                report_lines.append(f"   {key}: {value}")

        report_lines.extend(["", "-" * 70, "3. CROSS-VALIDATION RESULTS", "-" * 70])

        if "cross_validation" in self.training_history:
            cv = self.training_history["cross_validation"]
            report_lines.extend([
                f"   Folds:         {cv['folds']}",
                f"   Mean Accuracy: {cv['mean_accuracy']:.4f}",
                f"   Std Dev:       {cv['std_accuracy']:.4f}",
                "",
                f"   Fold Scores:  {[f'{s:.4f}' for s in cv['scores']]}",
            ])

        report_lines.extend(["", "-" * 70, "4. TEST SET EVALUATION", "-" * 70])

        if self.evaluation_metrics:
            m = self.evaluation_metrics
            report_lines.extend([
                f"   Accuracy:   {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)",
                f"   Precision:  {m['precision']:.4f}",
                f"   Recall:     {m['recall']:.4f}",
                f"   F1 Score:   {m['f1_score']:.4f}",
                f"   ROC AUC:    {m['roc_auc']:.4f}",
                "",
                "   Confusion Matrix:",
                f"   True Negatives:  {m['confusion_matrix']['true_negatives']}",
                f"   False Positives: {m['confusion_matrix']['false_positives']}",
                f"   False Negatives: {m['confusion_matrix']['false_negatives']}",
                f"   True Positives:  {m['confusion_matrix']['true_positives']}",
            ])

            # Target assessment
            if m["accuracy"] >= 0.95:
                report_lines.append("")
                report_lines.append("   ✓ TARGET ACHIEVED: Model meets >= 95% accuracy requirement")
            else:
                report_lines.append("")
                report_lines.append(f"   ✗ TARGET NOT MET: Accuracy {m['accuracy']*100:.2f}% < 95%")

        report_lines.extend(["", "-" * 70, "5. TOP FEATURE IMPORTANCES", "-" * 70])

        if "feature_importance" in self.training_history:
            top_features = list(self.training_history["feature_importance"].items())[:10]
            for i, (name, imp) in enumerate(top_features, 1):
                report_lines.append(f"   {i:2d}. {name:30s}: {imp:.4f}")

        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])

        report_text = "\n".join(report_lines)

        with open(output_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Training report saved to: {output_path}")

        return report_text


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    Execute complete training pipeline.

    Usage:
        python train_model.py
        python train_model.py --optimize
    """
    import argparse

    parser = argparse.ArgumentParser(description="PhishGuard Model Training")
    parser.add_argument("--optimize", action="store_true",
                       help="Run hyperparameter optimization")
    parser.add_argument("--samples", type=int, default=2000,
                       help="Samples per class (default: 2000)")
    args = parser.parse_args()

    print("=" * 70)
    print("PHISHGUARD ML MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize trainer
    config = {
        "phishing_samples": args.samples,
        "legitimate_samples": args.samples,
    }
    trainer = PhishGuardTrainer(config)

    # Step 1: Load data
    print("\n[1/6] Loading dataset...")
    X, y, urls = trainer.load_and_prepare_data()

    # Step 2: Split data
    print("\n[2/6] Splitting data...")
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    # Step 3: Preprocess
    print("\n[3/6] Preprocessing features...")
    X_train_proc, X_test_proc = trainer.preprocess(X_train, X_test)

    # Step 4: Train
    print("\n[4/6] Training model...")
    trainer.train(X_train_proc, y_train, X_test_proc, y_test,
                optimize_hyperparameters=args.optimize)

    # Step 5: Evaluate
    print("\n[5/6] Evaluating model...")
    metrics = trainer.evaluate(X_test_proc, y_test)

    # Step 6: Analyze features
    print("\n[6/6] Analyzing feature importance...")
    trainer.analyze_feature_importance(X_train_proc, y_train)

    # Save model
    print("\n[Saving] Model...")
    model_path = trainer.save_model()

    # Generate report
    print("\n[Report] Generating training report...")
    report = trainer.generate_training_report()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nModel saved to: {model_path}")
    print(f"\nFinal Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Target (>=95%): {'✓ ACHIEVED' if metrics['accuracy'] >= 0.95 else '✗ NOT MET'}")

    return trainer


if __name__ == "__main__":
    trainer = main()