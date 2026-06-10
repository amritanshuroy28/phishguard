"""
Ensemble ML Service
====================
Loads 3 PhishGuard models and averages their predict_proba() at inference time.

Models expected in: models/ensemble/{name}_model.pkl
Each pickle format: {"model": XGBClassifier, "best_params": {...},
                    "metrics": {...}, "history": {...}, "feature_names": [...]}

The ensemble runs 3 XGBoost predict_proba() calls (~1ms total) and averages
the probability outputs.  Target: <5ms inference overhead.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from statistics import mean

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "ensemble"
MODEL_CONFIGS = {
    "phishguard": _MODELS_DIR / "phishguard_model.pkl",
    "uci":        _MODELS_DIR / "uci_model.pkl",
    "kaggle":     _MODELS_DIR / "kaggle_model.pkl",
}

# 42 unified feature names (matching train_ensemble.py)
UNIFIED_FEATURES = [
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
    'num_dots', 'num_dash', 'num_ampersand', 'num_equals',
    'num_underscore', 'hostname_length', 'url_char_prob',
    'char_continuation_rate', 'has_obfuscation',
]


class EnsembleModelPackage:
    """Holds a loaded model and its metadata."""
    __slots__ = ("name", "model", "metrics", "cv_scores", "best_params", "loaded")

    def __init__(self, name: str):
        self.name = name
        self.model: Any = None
        self.metrics: Dict = {}
        self.cv_scores: List[float] = []
        self.best_params: Dict = {}
        self.loaded: bool = False


class EnsembleMLService:
    """
    Ensemble inference service — loads all 3 models at startup and averages
    their probability outputs at predict() time.

    Timing budget:
        - 3 × model.predict_proba():  ~1–2 ms
        - Ensemble averaging:         ~0.1 ms
        - Total inference overhead:    ~2–3 ms  (well within sub-500ms budget)
    """

    def __init__(self):
        self.models: Dict[str, EnsembleModelPackage] = {}
        self._feature_names = UNIFIED_FEATURES
        self._n_features = len(UNIFIED_FEATURES)  # 42
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        """Load all available ensemble models from disk."""
        for name, path in MODEL_CONFIGS.items():
            pkg = EnsembleModelPackage(name)
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    pkg.model       = data.get("model")
                    pkg.metrics     = data.get("metrics", {})
                    pkg.cv_scores   = (data.get("history") or {}).get("cv_scores", [])
                    pkg.best_params = data.get("best_params", {})
                    pkg.loaded      = True
                    logger.info(f"[ensemble] Loaded {name} from {path}  "
                                f"acc={pkg.metrics.get('accuracy', '?')}")
                except Exception as e:
                    logger.warning(f"[ensemble] Failed to load {name}: {e}")
            else:
                logger.warning(f"[ensemble] Model not found: {path}")
            self.models[name] = pkg

        loaded = [n for n, p in self.models.items() if p.loaded]
        logger.info(f"[ensemble] {len(loaded)}/{len(self.models)} models ready: {loaded}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, feature_array: Any) -> Tuple[bool, float]:
        """
        Run ensemble inference on a 42-feature array.

        Args:
            feature_array:  List or array of 42 floats
                           (from URLFeatures.to_feature_array())

        Returns:
            Tuple of (is_phishing: bool, confidence: float in [0, 1])
                    confidence = avg prob (use directly for phishing prob)
        """
        # Convert to numpy
        if not isinstance(feature_array, np.ndarray):
            arr = np.array(feature_array, dtype=np.float32)
        else:
            arr = feature_array.astype(np.float32)

        # Handle 1D → 2D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(arr.shape[0], -1)

        # Sanitise NaN/Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.shape[1] != self._n_features:
            logger.warning(
                f"[ensemble] Feature count mismatch: got {arr.shape[1]}, "
                f"expected {self._n_features}. Padding/truncating."
            )
            if arr.shape[1] < self._n_features:
                arr = np.pad(arr, ((0, 0), (0, self._n_features - arr.shape[1])),
                             constant_values=0.0)
            else:
                arr = arr[:, :self._n_features]

        # Collect per-model probabilities
        probs: List[float] = []
        fallback_pkg: Optional[EnsembleModelPackage] = None

        for name, pkg in self.models.items():
            if not pkg.loaded:
                continue
            try:
                prob = float(pkg.model.predict_proba(arr)[0][1])
                probs.append(prob)
            except Exception as e:
                logger.warning(f"[ensemble] {name} predict_proba error: {e}")

        # ── Rule-based fallback (when no models loaded) ─────────────────
        if not probs:
            logger.warning("[ensemble] No models loaded — using rule-based fallback")
            return self._rule_based_predict(arr[0])

        avg_prob = mean(probs)
        is_phishing = avg_prob >= 0.5
        # Confidence: for phishing, avg_prob is the phishing probability;
        # for legit, 1 - avg_prob is the "confidence it's legit"
        confidence = float(avg_prob) if is_phishing else float(1 - avg_prob)

        return is_phishing, confidence

    def predict_proba_only(self, feature_array: Any) -> Dict[str, float]:
        """
        Return per-model probabilities (useful for debugging / transparency).

        Returns:
            {"model_name": phishing_probability, ...}
        """
        if not isinstance(feature_array, np.ndarray):
            arr = np.array(feature_array, dtype=np.float32).reshape(1, -1)
        else:
            arr = feature_array.astype(np.float32).reshape(1, -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        result = {}
        for name, pkg in self.models.items():
            if not pkg.loaded:
                continue
            try:
                result[name] = float(pkg.model.predict_proba(arr)[0][1])
            except Exception:
                result[name] = 0.5
        return result

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------
    def _rule_based_predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Heuristic fallback when no ensemble models are available.
        Uses the same logic as MLService._rule_based_predict for consistency.
        """
        try:
            max_idx = min(len(features), self._n_features) - 1
            obfuscation = float(features[min(15, max_idx)])   # obfuscation_score
            suspicious  = float(features[min(21, max_idx)])   # suspicious_pattern_score

            typosquatting = 0.0
            if self._n_features > 32 and len(features) > 32:
                typosquatting = float(features[32])            # typosquatting_score

            risk = (obfuscation * 0.3 + suspicious * 0.4 + typosquatting * 0.3)
            confidence = min(risk / 5.0, 1.0)
            is_phishing = risk > 1.5
            return is_phishing, confidence
        except Exception:
            return False, 0.5

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_info(self) -> Dict[str, Any]:
        """Return ensemble status and per-model info."""
        return {
            "n_features"     : self._n_features,
            "feature_names"  : self._feature_names,
            "models"         : {
                name: {
                    "loaded"      : pkg.loaded,
                    "accuracy"    : pkg.metrics.get("accuracy"),
                    "f1_score"    : pkg.metrics.get("f1_score"),
                    "roc_auc"     : pkg.metrics.get("roc_auc"),
                    "cv_scores"   : pkg.cv_scores,
                    "best_params" : {k: str(v) for k, v in pkg.best_params.items()},
                }
                for name, pkg in self.models.items()
            },
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_ensemble_service: Optional[EnsembleMLService] = None


def get_ensemble_service() -> EnsembleMLService:
    global _ensemble_service
    if _ensemble_service is None:
        _ensemble_service = EnsembleMLService()
    return _ensemble_service


def reload_ensemble() -> bool:
    """Hot-reload the ensemble (e.g., after retraining)."""
    global _ensemble_service
    _ensemble_service = EnsembleMLService()
    return len([p for p in _ensemble_service.models.values() if p.loaded]) > 0