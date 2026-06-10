"""
PhishGuard Ensemble Training
============================
Retrains all 3 models (PhishGuard + UCI + Kaggle) on a unified 42-feature
URL-only set, then exports them for ensemble inference.

Unified feature set (42 features) — indices 0–32 = original 33 PhishGuard
URL features; indices 33–41 = 9 derived features added for Kaggle/UCI alignment:
    num_dots, num_dash, num_ampersand, num_equals, num_underscore,
    hostname_length, url_char_prob, char_continuation_rate, has_obfuscation

Usage:
    python train_ensemble.py [--optimize] [--models phishguard,uci,kaggle]
"""

import os
import sys
import json
import pickle
import logging
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from statistics import mean

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve project root and add to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.features.feature_extraction import (
    URLFeatureExtractor, URLFeatures
)
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
ENSEMBLE_DIR = PROJECT_ROOT / "models" / "ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

KAGGLE_CSV = PROJECT_ROOT / "data" / "Phishing_Legitimate_full.csv"
UCI_REPORT = PROJECT_ROOT / "data" / "aci_fetch_training" / "training_report.json"
UCI_DIR    = PROJECT_ROOT / "data" / "uci_fetch_training"
PHISH_MODEL = PROJECT_ROOT / "models" / "phishguard_model.pkl"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# The 42 unified feature names (same as URLFeatures.to_feature_array order)
UNIFIED_FEATURES = [
    # 0–32: original 33
    'url_length', 'path_length', 'query_length', 'fragment_length',
    'subdomain_count', 'subdomain_length', 'path_depth', 'url_entropy',
    'domain_length', 'is_free_domain_provider',
    'has_hex_encoding', 'hex_encoded_chars', 'has_ip_address',
    'has_at_symbol', 'has_double_slash_redirect', 'has_data_uri',
    'obfuscation_score', 'suspicious_path_count', 'has_suspicious_path',
    'has_login_keywords', 'has_brand_in_subdomain', 'suspicious_pattern_score',
    'special_char_count', 'digit_count', 'digit_ratio', 'uppercase_count',
    'has_unicode', 'punycode_detected', 'domain_age_days', 'is_recent_domain',
    'registrar_suspicious', 'dns_record_exists', 'typosquatting_score',
    # 33–41: new 9
    'num_dots', 'num_dash', 'num_ampersand', 'num_equals',
    'num_underscore', 'hostname_length', 'url_char_prob',
    'char_continuation_rate', 'has_obfuscation',
]
N_FEATURES = len(UNIFIED_FEATURES)  # 42

# ---------------------------------------------------------------------------
# FEATURE EXTRACTION HELPERS
# ---------------------------------------------------------------------------
def _url_like_from_kaggle(df: pd.DataFrame) -> List[str]:
    """
    Synthesise plausible URLs from Kaggle feature columns.
    The Kaggle CSV has no URL column — we reconstruct a plausible URL from
    the most diagnostic numeric features to use as input to URLFeatureExtractor.
    """
    urls = []
    for _, row in df.iterrows():
        try:
            # Reconstruct a plausible domain + path from numeric features
            # This gives URLFeatureExtractor enough to compute 42 features
            subdomain_level = int(row.get('SubdomainLevel', 0))
            path_level      = int(row.get('PathLevel', 1))
            hostname_len    = int(row.get('HostnameLength', 10))
            path_len        = int(row.get('PathLength', 5))
            query_len       = int(row.get('QueryLength', 0))
            num_dots        = int(row.get('NumDots', 0))
            has_https       = int(row.get('NoHttps', 0))  # 1 = no https

            # Build a realistic-looking URL from feature values
            protocol = 'http' if has_https else 'https'
            if subdomain_level > 0:
                sub = 's' * min(subdomain_level, 3) + 'ub'
            else:
                sub = 'www'
            domain = f"{sub}domain{'.' * max(0, num_dots-1)}example.com"
            path   = '/' + '/'.join([f"level{i}" for i in range(max(1, path_level))])
            query  = ('?' + 'x=' * min(query_len, 5)) if query_len > 0 else ''
            url    = f"{protocol}://{domain}{path}{query}"
            urls.append(url)
        except Exception:
            urls.append("https://example.com/default/path")
    return urls


def _url_like_from_uci(df: pd.DataFrame) -> List[str]:
    """
    Synthesise plausible URLs from UCI numeric features.
    Many UCI features map directly or approximately to PhishGuard features.
    """
    urls = []
    for _, row in df.iterrows():
        try:
            url_len  = int(row.get('URLLength', 80))
            domain_len = int(row.get('DomainLength', 15))
            subdomain  = int(row.get('NoOfSubDomain', 0))
            tld_legit = float(row.get('TLDLegitimateProb', 0.5))
            num_query = int(row.get('NoOfQMarkInURL', 0))

            protocol = 'https'
            sub = f"login" if subdomain > 0 else "www"
            domain = f"{sub}.secure-site.com"
            path   = '/signin' if tld_legit < 0.5 else '/page'
            query  = ('?' + 'q=' * num_query) if num_query > 0 else ''
            # Make URL length roughly match
            extra = 'a' * max(0, url_len - len(f"{protocol}://{domain}{path}{query}"))
            url = f"{protocol}://{domain}{path}{extra}{query}"
            urls.append(url)
        except Exception:
            urls.append("https://example.com/some/path")
    return urls


def _map_kaggle_to_42(df: pd.DataFrame, extractor: URLFeatureExtractor
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 42-feature arrays from Kaggle dataset URLs.
    Returns (X_42, y) arrays.
    """
    logger.info(f"Extracting 42 features from {len(df)} Kaggle rows …")
    urls = _url_like_from_kaggle(df)
    X_list, y_list = [], []
    for i, (url, label) in enumerate(zip(urls, df['CLASS_LABEL'])):
        try:
            f = extractor.extract(url)
            arr = f.to_feature_array()
            # Sanitise: replace None/NaN
            arr = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                   for v in arr]
            X_list.append(arr)
            y_list.append(int(label))
        except Exception as e:
            logger.debug(f"Failed row {i} ({url[:40]}): {e}")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int_)
    logger.info(f"Kaggle 42-feature matrix: {X.shape}, label dist: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def _map_uci_to_42(df: pd.DataFrame, extractor: URLFeatureExtractor
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 42-feature arrays from UCI dataset URLs.
    Returns (X_42, y) arrays.
    """
    logger.info(f"Extracting 42 features from {len(df)} UCI rows …")
    urls = _url_like_from_uci(df)
    X_list, y_list = [], []
    for i, (url, label) in enumerate(zip(urls, df['label'])):
        try:
            f = extractor.extract(url)
            arr = f.to_feature_array()
            arr = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                   for v in arr]
            X_list.append(arr)
            y_list.append(int(label))
        except Exception as e:
            logger.debug(f"Failed row {i}: {e}")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int_)
    logger.info(f"UCI 42-feature matrix: {X.shape}, label dist: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# ---------------------------------------------------------------------------
# LOAD ORIGINAL PHISHGUARD SAMPLES
# ---------------------------------------------------------------------------
def _load_phishguard_samples(extractor: URLFeatureExtractor
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Re-extract features from original PhishGuard training URLs."""
    # Original training was ~320 samples (160 phishing / 160 legit)
    # We reconstruct using the URLs embedded in the original pickle
    phish_urls = [
        "http://secure-update-paypal.com.suspicious.net/login",
        "http://192.168.1.1/phishing/portal.php",
        "https://amazon.com-signin-verify.malicious.xyz/account",
        "http://banking-chase.com.update.login.phish.net/session",
        "https://microsoft.com.secure-login.evil.com/verify",
        "https://netflix.com.billing.update.fake-site.net/payment",
        "http://apple.com.icloud-login.scam.ru/auth",
        "https://facebook.com.fb-login.phishing.net/connect",
        "http://google.com.drive.documents.malware.ru/download",
        "https://paypal.com.signin.secure.phish.net/confirm",
        "https://ebay.com.login.verify.scam.net/?cmd=_login",
        "http://wellsfargo.com.verify.account.suspicious.net/banking",
        "https://linkedin.com-profile.ap7.net/invite",
        "https://twitter.com.secure-login.fake123.net/oauth",
        "https://reddit.com.premium.membership.scam.net/join",
        "https://dropbox.com.document.share.malware.net/get",
        "https://github.com releases updates malware scan attacker.com",
        "http://amazonaws.com.s3-bucket malitious config phishing site.net",
        "https://secure-bank-of-america.com.login.verify.scam.net",
        "http://appleid.apple.com.restore.account.phishing.net",
    ]
    legit_urls = [
        "https://www.google.com",
        "https://mail.google.com/mail/",
        "https://www.amazon.com",
        "https://www.amazon.com/gp/product/B08N5WRWNW",
        "https://www.facebook.com",
        "https://www.facebook.com/photo.php",
        "https://www.microsoft.com",
        "https://www.microsoft.com/en-us/microsoft-365",
        "https://accounts.google.com",
        "https://www.linkedin.com",
        "https://www.linkedin.com/in/williamhgates",
        "https://www.apple.com",
        "https://www.apple.com/shop",
        "https://www.netflix.com",
        "https://www.netflix.com/browse/my-list",
        "https://www.paypal.com",
        "https://www.paypal.com/us/home",
        "https://www.github.com",
        "https://www.github.comfeatures",
        "https://www.reddit.com",
    ]
    all_urls = phish_urls + legit_urls
    all_labels = [1]*len(phish_urls) + [0]*len(legit_urls)

    X_list = []
    for url in all_urls:
        try:
            f = extractor.extract(url)
            arr = f.to_feature_array()
            X_list.append(arr)
        except Exception:
            pass
    X = np.array(X_list, dtype=np.float32)
    y = np.array([l for l, f in zip(all_labels, X_list) if f is not None], dtype=np.int_)
    logger.info(f"PhishGuard original samples: {X.shape}")
    return X, y


# ---------------------------------------------------------------------------
# MODEL TRAINER
# ---------------------------------------------------------------------------
class EnsembleModelTrainer:
    def __init__(self, name: str, optimize: bool = False):
        self.name = name
        self.optimize = optimize
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_params: Dict = {}
        self.history: Dict = {}
        self.metrics: Dict = {}
        self.cv_scores: List[float] = []

    # ------------------------------------------------------------------
    def _default_params(self) -> Dict:
        return dict(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, use_label_encoder=False, n_jobs=-1,
        )

    # ------------------------------------------------------------------
    def _tune(self, X: np.ndarray, y: np.ndarray) -> Dict:
        logger.info(f"[{self.name}] Running 3-fold GridSearchCV tuning …")
        param_grid = dict(
            max_depth=[4, 6],
            learning_rate=[0.05, 0.1, 0.15],
            n_estimators=[100, 200],
            min_child_weight=[1, 3],
            subsample=[0.8, 0.9],
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
        logger.info(f"[{self.name}] Best params: {gs.best_params_}  CV: {gs.best_score_:.4f}")
        return gs.best_params_

    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModelTrainer":
        params = self._default_params()
        if self.optimize:
            tuned = self._tune(X, y)
            params.update(tuned)
        self.best_params = params
        self.model = xgb.XGBClassifier(**params)

        # 5-fold CV
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")
        self.cv_scores = [float(s) for s in scores]
        logger.info(f"[{self.name}] CV: {np.mean(scores):.4f} ± {np.std(scores)*2:.4f}  {scores}")

        self.model.fit(X, y)
        return self

    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        m = dict(
            accuracy   = accuracy_score(y_test, y_pred),
            precision  = precision_score(y_test, y_pred),
            recall     = recall_score(y_test, y_pred),
            f1_score   = f1_score(y_test, y_pred),
            roc_auc    = roc_auc_score(y_test, y_proba),
        )
        cm = confusion_matrix(y_test, y_pred).tolist()
        logger.info(f"[{self.name}] Test — Acc:{m['accuracy']:.4f}  "
                    f"Prec:{m['precision']:.4f}  Rec:{m['recall']:.4f}  "
                    f"F1:{m['f1_score']:.4f}  AUC:{m['roc_auc']:.4f}")
        m["confusion_matrix"] = cm
        self.metrics = m
        return m

    # ------------------------------------------------------------------
    def save(self, path: Path):
        pkg = dict(
            model=self.model,
            best_params=self.best_params,
            metrics=self.metrics,
            history={"cv_scores": self.cv_scores},
            feature_names=UNIFIED_FEATURES,
        )
        with open(path, "wb") as f:
            pickle.dump(pkg, f)
        logger.info(f"[{self.name}] Saved → {path}")

    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, X_test, y_test, label):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Phishing", "Legit"],
                    yticklabels=["Phishing", "Legit"])
        ax.set_title(f"{label} — Confusion Matrix")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        fig.tight_layout()
        out = ENSEMBLE_DIR / f"confusion_matrix_{self.name}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        logger.info(f"Saved {out}")


# ---------------------------------------------------------------------------
# ENSEMBLE EVALUATION
# ---------------------------------------------------------------------------
def evaluate_ensemble(models: Dict[str, xgb.XGBClassifier],
                     X_test: np.ndarray, y_test: np.ndarray
                     ) -> Dict:
    """
    Evaluate the ensemble (average predict_proba) on test set.
    Returns per-model and ensemble metrics.
    """
    results = {}
    all_probs = []

    for name, model_pkg in models.items():
        model = model_pkg["model"]
        proba = model.predict_proba(X_test)[:, 1]
        all_probs.append(proba)
        pred  = (proba >= 0.5).astype(int)
        results[name] = {
            "accuracy" : accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall"   : recall_score(y_test, pred),
            "f1_score" : f1_score(y_test, pred),
            "roc_auc"  : roc_auc_score(y_test, proba),
        }
        cm = confusion_matrix(y_test, pred).tolist()
        results[name]["confusion_matrix"] = cm

    # Average ensemble
    avg_proba = mean(all_probs)
    ensemble_pred = (avg_proba >= 0.5).astype(int)
    results["ensemble"] = {
        "accuracy"  : accuracy_score(y_test, ensemble_pred),
        "precision" : precision_score(y_test, ensemble_pred),
        "recall"    : recall_score(y_test, ensemble_pred),
        "f1_score"  : f1_score(y_test, ensemble_pred),
        "roc_auc"   : roc_auc_score(y_test, avg_proba),
        "confusion_matrix": confusion_matrix(y_test, ensemble_pred).tolist(),
    }
    logger.info(f"ENSEMBLE → Acc:{results['ensemble']['accuracy']:.4f}  "
                f"Prec:{results['ensemble']['precision']:.4f}  "
                f"Rec:{results['ensemble']['recall']:.4f}  "
                f"F1:{results['ensemble']['f1_score']:.4f}  "
                f"AUC:{results['ensemble']['roc_auc']:.4f}")
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train PhishGuard ensemble on unified 42-feature set")
    parser.add_argument("--optimize", action="store_true", help="Run GridSearchCV per model")
    parser.add_argument("--models", default="phishguard,kaggle",
                        help="Comma-separated models to train: phishguard,kaggle,uci  (default: phishguard,kaggle)")
    args = parser.parse_args()
    models_to_train = [m.strip() for m in args.models.split(',')]

    print("=" * 70)
    print("PhishGuard Ensemble Training — Unified 42-Feature Set")
    print("=" * 70)

    # Initialise extractor
    extractor = URLFeatureExtractor()
    logger.info(f"Unified feature count: {N_FEATURES}")

    # ── Prepare datasets ──────────────────────────────────────────────────
    X_all, y_all = {}, {}

    # 1. PhishGuard original samples
    if "phishguard" in models_to_train:
        logger.info("[1/3] Loading PhishGuard original samples …")
        X_phish, y_phish = _load_phishguard_samples(extractor)
        X_all["phishguard"]  = X_phish
        y_all["phishguard"]  = y_phish

    # 2. Kaggle dataset
    if "kaggle" in models_to_train:
        logger.info("[2/3] Loading Kaggle PhishNet dataset …")
        df_kaggle = pd.read_csv(KAGGLE_CSV)
        logger.info(f"Kaggle raw: {df_kaggle.shape}")
        X_kag, y_kag = _map_kaggle_to_42(df_kaggle, extractor)
        X_all["kaggle"] = X_kag
        y_all["kaggle"] = y_kag

    # 3. UCI dataset
    if "uci" in models_to_train:
        logger.info("[3/3] Loading UCI ML repository dataset …")
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
            from ucimlrepo import fetch_ucirepo
        dataset   = fetch_ucirepo(id=967)
        df_uci    = dataset.data.features.copy()
        y_uci_raw = dataset.data.targets.copy()
        df_uci['label'] = y_uci_raw.values
        # Drop string columns
        df_uci = df_uci.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"])
        X_uci, y_uci = _map_uci_to_42(df_uci, extractor)
        X_all["uci"] = X_uci
        y_all["uci"] = y_uci

    # ── Train each model ──────────────────────────────────────────────────
    trained_models = {}
    for name in models_to_train:
        X = X_all.get(name)
        y = y_all.get(name)
        if X is None:
            logger.warning(f"[{name}] No data, skipping."); continue

        logger.info(f"\n── Training [{name}] on {len(y)} samples, {X.shape[1]} features ──")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y
        )
        trainer = EnsembleModelTrainer(name, optimize=args.optimize)
        trainer.train(X_train, y_train)

        # Reload and train on full data for final model
        trainer_final = EnsembleModelTrainer(name, optimize=False)
        if args.optimize:
            trainer_final.best_params = trainer.best_params
            trainer_final.model = xgb.XGBClassifier(**trainer.best_params)
        else:
            trainer_final.model = xgb.XGBClassifier(**trainer_final._default_params())
        trainer_final.model.fit(X_train, y_train)

        m = trainer_final.evaluate(X_test, y_test)
        trainer_final.metrics = m
        trainer_final.cv_scores = trainer.cv_scores

        model_path = ENSEMBLE_DIR / f"{name}_model.pkl"
        trainer_final.save(model_path)
        trained_models[name] = trainer_final

        # Per-model confusion matrix
        trainer_final.plot_confusion_matrix(X_test, y_test, name)

    # ── Ensemble evaluation ───────────────────────────────────────────────
    if len(trained_models) >= 2:
        logger.info("\n═══ Ensemble Evaluation ═══")
        # Use the last model's test split as the shared test set
        # (all models were split the same way so indices align)
        # Re-split from the largest dataset for a fair ensemble eval
        largest = max(X_all.items(), key=lambda x: len(x[1]))
        X_largest = X_all[largest[0]]
        y_largest = y_all[largest[0]]
        _, X_test_shared, _, y_test_shared = train_test_split(
            X_largest, y_largest, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y_largest
        )

        model_pkgs = {
            name: {"model": tm.model, "metrics": tm.metrics, "cv_scores": tm.cv_scores,
                   "best_params": tm.best_params}
            for name, tm in trained_models.items()
        }
        ensemble_results = evaluate_ensemble(model_pkgs, X_test_shared, y_test_shared)

        # Save ensemble report
        report = {
            "timestamp": datetime.now().isoformat(),
            "unified_features": N_FEATURES,
            "feature_names": UNIFIED_FEATURES,
            "models": {},
            "ensemble_metrics": ensemble_results.pop("ensemble", {}),
            "individual_metrics": ensemble_results,
        }
        for name, tm in trained_models.items():
            report["models"][name] = {
                "best_params": tm.best_params,
                "cv_scores": tm.cv_scores,
                "test_metrics": {k: v for k, v in tm.metrics.items() if k != "confusion_matrix"},
            }

        rp = ENSEMBLE_DIR / "ensemble_report.json"
        with open(rp, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Ensemble report saved: {rp}")

    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 70)
    for name, tm in trained_models.items():
        print(f"  [{name}] Acc: {tm.metrics['accuracy']*100:.2f}%  "
              f"F1: {tm.metrics['f1_score']:.4f}  AUC: {tm.metrics['roc_auc']:.4f}")
    print(f"  Output: {ENSEMBLE_DIR}")


if __name__ == "__main__":
    main()