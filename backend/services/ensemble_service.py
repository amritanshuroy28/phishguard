"""
PhishGuard Ultimate ML Service
==============================
Single best XGBoost model trained on Phish.Database + PhiUSIIL legit URLs.

Inference latency: ~1-2 ms per request.
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
_MODELS_DIR = Path(__file__).parent.parent.parent / "models"
ULTIMATE_MODEL_PKL = _MODELS_DIR / "phishguard_ultimate.pkl"

# 42 unified feature names (matching train_ultimate.py)
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


class UltimateMLService:
    """
    PhishGuard Ultimate — single best XGBoost model.
    Trained on Phish.Database + PhiUSIIL + verified trusted domains.

    Target: >95% test accuracy (achieved 99.83% in training).
    Inference: ~1-2ms per request.
    """

    def __init__(self):
        self.model: Any = None
        self.metrics: Dict = {}
        self.cv_accuracy: float = 0.0
        self.cv_f1: float = 0.0
        self.training_data: Dict = {}
        self.trained_at: str = ""
        self._feature_names = UNIFIED_FEATURES
        self._n_features = len(UNIFIED_FEATURES)
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if not ULTIMATE_MODEL_PKL.exists():
            logger.error(f"[ultimate] Model file not found: {ULTIMATE_MODEL_PKL}")
            return

        try:
            with open(ULTIMATE_MODEL_PKL, "rb") as f:
                pkg = pickle.load(f)
            self.model         = pkg["model"]
            self.metrics       = pkg.get("metrics", {})
            self.cv_accuracy   = pkg.get("cv_accuracy", 0.0)
            self.cv_f1         = pkg.get("cv_f1", 0.0)
            self.training_data = pkg.get("training_data", {})
            self.trained_at    = pkg.get("trained_at", "unknown")
            self._loaded       = True
            logger.info(f"[ultimate] Loaded model from {ULTIMATE_MODEL_PKL}")
            logger.info(f"[ultimate] Test accuracy: {self.metrics.get('accuracy', '?')}")
            logger.info(f"[ultimate] CV accuracy:   {self.cv_accuracy}")
            logger.info(f"[ultimate] Training data:  {self.training_data}")
        except Exception as e:
            logger.error(f"[ultimate] Failed to load model: {e}", exc_info=True)

    @property
    def loaded(self) -> bool:
        return self._loaded and self.model is not None

    def predict(self, feature_array: Any) -> Tuple[bool, float]:
        """Run inference on a 42-feature array."""
        if not self.loaded:
            return False, 0.5

        if not isinstance(feature_array, np.ndarray):
            arr = np.array(feature_array, dtype=np.float32)
        else:
            arr = feature_array.astype(np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(arr.shape[0], -1)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.shape[1] != self._n_features:
            logger.warning(
                f"[ultimate] Feature count mismatch: got {arr.shape[1]}, "
                f"expected {self._n_features}. Adjusting."
            )
            if arr.shape[1] < self._n_features:
                arr = np.pad(arr, ((0, 0), (0, self._n_features - arr.shape[1])),
                             constant_values=0.0)
            else:
                arr = arr[:, :self._n_features]

        try:
            phishing_prob = float(self.model.predict_proba(arr)[0][1])
            is_phishing = phishing_prob >= 0.5
            confidence = phishing_prob if is_phishing else (1.0 - phishing_prob)
            return is_phishing, confidence
        except Exception as e:
            logger.error(f"[ultimate] predict error: {e}", exc_info=True)
            return False, 0.5

    def predict_proba_only(self, feature_array: Any) -> Dict[str, float]:
        """Return raw phishing probability (single model)."""
        if not self.loaded:
            return {}

        if not isinstance(feature_array, np.ndarray):
            arr = np.array(feature_array, dtype=np.float32).reshape(1, -1)
        else:
            arr = feature_array.astype(np.float32).reshape(1, -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            prob = float(self.model.predict_proba(arr)[0][1])
            return {"ultimate": prob}
        except Exception:
            return {"ultimate": 0.5}

    def get_info(self) -> Dict[str, Any]:
        """Return model status and metadata."""
        return {
            "model_name"   : "phishguard_ultimate",
            "model_version": "ultimate-v1.0",
            "loaded"       : self._loaded,
            "n_features"   : self._n_features,
            "feature_names": self._feature_names,
            "metrics"      : self.metrics,
            "cv_accuracy"  : self.cv_accuracy,
            "cv_f1"        : self.cv_f1,
            "training_data": self.training_data,
            "trained_at"   : self.trained_at,
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_ultimate_service: Optional[UltimateMLService] = None


def get_ultimate_service() -> UltimateMLService:
    """Get or create the global UltimateMLService instance."""
    global _ultimate_service
    if _ultimate_service is None:
        _ultimate_service = UltimateMLService()
    return _ultimate_service


# Backwards-compat alias
def get_ensemble_service():
    """Backwards-compat alias — returns the single ultimate model service."""
    return get_ultimate_service()


def reload_ultimate() -> bool:
    """Hot-reload the ultimate model (e.g., after retraining)."""
    global _ultimate_service
    _ultimate_service = UltimateMLService()
    return _ultimate_service.loaded
