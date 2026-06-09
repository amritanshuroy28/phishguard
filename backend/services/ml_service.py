"""
PhishGuard ML Service
=======================
Service for loading and running ML model inference.
Optimized for <500ms response time.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
import time

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML Pipeline path
ML_PIPELINE_PATH = Path(__file__).parent.parent.parent / "ml_pipeline"
MODEL_PATH = ML_PIPELINE_PATH / "models" / "phishguard_model.pkl"
FEATURES_PATH = ML_PIPELINE_PATH / "models" / "feature_names.json"

# Fallback model path (production)
PRODUCTION_MODEL_PATH = Path(__file__).parent.parent / "models" / "phishguard_model.pkl"


class MLService:
    """
    Service for ML model loading and inference.

    Loads the serialized XGBoost model and provides inference methods.
    Handles model versioning and feature extraction routing.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize ML service.

        Args:
            model_path: Optional custom path to model file
        """
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.version = "unknown"
        self.model_path = None
        self._load_timestamp = None
        self._feature_cache = {}

        # Try to load model
        self._load_model(model_path)

    def _load_model(self, custom_path: Optional[Path] = None) -> bool:
        """
        Load serialized model from disk.

        Args:
            custom_path: Override default model path

        Returns:
            True if load successful, False otherwise
        """
        # Try multiple paths
        paths_to_try = [
            custom_path,
            MODEL_PATH,
            PRODUCTION_MODEL_PATH,
            Path("ml_pipeline/models/phishguard_model.pkl"),
            Path("models/phishguard_model.pkl"),
        ]

        for path in paths_to_try:
            if path and path.exists():
                try:
                    logger.info(f"Loading model from: {path}")
                    with open(path, 'rb') as f:
                        model_package = pickle.load(f)

                    self.model = model_package.get("model")
                    self.scaler = model_package.get("scaler")
                    self.feature_names = model_package.get(
                        "feature_names",
                        ["url_length", "path_length", "query_length", "fragment_length",
                         "subdomain_count", "subdomain_length", "path_depth", "url_entropy",
                         "domain_length", "is_free_domain_provider",
                         "has_hex_encoding", "hex_encoded_chars", "has_ip_address",
                         "has_at_symbol", "has_double_slash_redirect", "has_data_uri",
                         "obfuscation_score", "suspicious_path_count", "has_suspicious_path",
                         "has_login_keywords", "has_brand_in_subdomain", "suspicious_pattern_score",
                         "special_char_count", "digit_count", "digit_ratio", "uppercase_count",
                         "has_unicode", "punycode_detected", "domain_age_days", "is_recent_domain",
                         "registrar_suspicious", "dns_record_exists", "typosquatting_score"]
                    )

                    # Extract model version from training history
                    training_history = model_package.get("training_history", {})
                    self.version = (
                        training_history.get("timestamp", "unknown").replace(":", "-").replace(".", "-")
                        if training_history else "v1.0.0"
                    )

                    self.model_path = path
                    self._load_timestamp = datetime.now()

                    logger.info(f"Model loaded successfully. Version: {self.version}")
                    logger.info(f"Feature count: {len(self.feature_names)}")
                    return True

                except Exception as e:
                    logger.warning(f"Failed to load model from {path}: {e}")
                    continue

        logger.warning("No model file found. Using rule-based fallback.")
        return False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def predict(self, feature_array: any) -> Tuple[bool, float]:
        """
        Run inference on feature array.

        Args:
            feature_array: List or NumPy array of features, shape (n_features,) or (n_samples, n_features)

        Returns:
            Tuple of (is_phishing: bool, confidence: float)
        """
        # Convert to numpy array if needed
        if not isinstance(feature_array, np.ndarray):
            feature_array = np.array(feature_array, dtype=np.float32)

        # Handle 1D array (single sample)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)

        # Handle NaN/Inf
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

        if self.model is None:
            # Fallback to rule-based prediction
            return self._rule_based_predict(feature_array[0])

        try:
            # Scale features if scaler available
            if self.scaler is not None:
                feature_array = self.scaler.transform(feature_array)

            # Get probability prediction
            probabilities = self.model.predict_proba(feature_array)

            # phishing probability (class 1)
            phishing_prob = probabilities[0][1] if probabilities.shape[1] > 1 else 0.5

            # Threshold at 0.5 for binary classification
            is_phishing = phishing_prob >= 0.5

            return is_phishing, float(phishing_prob)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._rule_based_predict(feature_array[0])

    def _rule_based_predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Rule-based fallback prediction using feature thresholds.

        Used when model is not available or prediction fails.
        Based on heuristic thresholds from known phishing patterns.
        """
        # Map feature indices to important features
        # Based on FEATURE_NAMES order
        try:
            obfuscation_score_idx = 15  # obfuscation_score
            suspicious_score_idx = 21    # suspicious_pattern_score
            typosquatting_score_idx = 32 # typosquatting_score - adjusted as some features may be missing

            # Adjust indices based on actual feature count
            # Ensure we don't go out of bounds
            max_idx = min(len(features), len(self.feature_names)) - 1

            obfuscation = float(features[min(obfuscation_idx := 15, max_idx)])
            suspicious = float(features[min(suspicious_idx := 21, max_idx)])

            # Get typosquatting if available
            typosquatting = 0.0
            if len(features) > 32:
                typosquatting = float(features[32])

            # Calculate risk score
            risk_score = (obfuscation * 0.3 +
                        suspicious * 0.4 +
                        typosquatting * 0.3)

            # Normalize to 0-1 range
            confidence = min(risk_score / 5.0, 1.0)  # Max theoretical score ~5.0

            is_phishing = risk_score > 1.5

            return is_phishing, confidence

        except Exception as e:
            logger.error(f"Rule-based prediction error: {e}")
            return False, 0.5  # Default to safe with 50% confidence

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "version": self.version,
            "loaded": self.is_loaded,
            "model_path": str(self.model_path) if self.model_path else None,
            "load_timestamp": self._load_timestamp.isoformat() if self._load_timestamp else None,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
        }


# Global singleton instance
_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """
    Get or create the global ML service instance.

    Returns:
        MLService singleton
    """
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service


def reload_model(model_path: Optional[Path] = None) -> bool:
    """
    Reload the ML model (e.g., after updating to new version).

    Args:
        model_path: Optional new model path

    Returns:
        True if reload successful
    """
    global _ml_service
    _ml_service = MLService(model_path)
    return _ml_service.is_loaded