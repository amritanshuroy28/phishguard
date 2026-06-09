"""
PhishGuard API Schemas
=======================
Pydantic models for request/response validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class RiskLevel(str, Enum):
    """Risk classification levels."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatCategory(str, Enum):
    """Categories of threats detected."""
    PHISHING = "phishing"
    MALWARE = "malware"
    CREDENTIAL_HARVEST = "credential_harvesting"
    TYPOSQUATTING = "typosquatting"
    OBFUSCATION = "obfuscation"
    RECENT_DOMAIN = "recent_domain"
    SUSPICIOUS_TLD = "suspicious_tld"
    NONE = "none"


class CTISource(str, Enum):
    """Cyber Threat Intelligence sources."""
    VIRUSTOTAL = "virustotal"
    URLHAUS = "urlhaus"
    OWN_MODEL = "own_model"


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class AnalyzeRequest(BaseModel):
    """
    Request schema for URL analysis.

    Attributes:
        url: The URL to analyze
        include_raw_features: Whether to include extracted features in response
        enable_cti: Whether to query external CTI sources (VirusTotal, URLhaus)
    """
    url: str = Field(
        ...,
        description="URL to analyze",
        min_length=5,
        max_length=2048,
        examples=["https://example.com/login"]
    )
    include_raw_features: bool = Field(
        default=False,
        description="Include extracted features in response"
    )
    enable_cti: bool = Field(
        default=True,
        description="Query external threat intelligence sources"
    )

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Basic URL validation."""
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        if len(v) < 5:
            raise ValueError("URL is too short")
        return v


class BatchAnalyzeRequest(BaseModel):
    """Request schema for batch URL analysis."""
    urls: List[str] = Field(
        ...,
        description="List of URLs to analyze",
        min_length=1,
        max_length=100
    )
    enable_cti: bool = Field(
        default=True,
        description="Query external threat intelligence sources"
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class CTIResult(BaseModel):
    """
    Cyber Threat Intelligence lookup result.

    Attributes:
        source: Name of the CTI source
        found: Whether the URL was found in the source
        malicious: Whether the source marks it as malicious
        positives: Number of vendors detecting as malicious (VirusTotal)
        total: Total number of vendors (VirusTotal)
        detection_rate: Percentage of detections
        metadata: Additional source-specific data
        error: Error message if lookup failed
    """
    source: CTISource
    found: bool = False
    malicious: bool = False
    positives: int = 0
    total: int = 0
    detection_rate: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    response_time_ms: Optional[float] = None


class ThreatDetail(BaseModel):
    """
    Detailed information about a detected threat.

    Attributes:
        category: Type of threat detected
        confidence: Confidence score (0-1)
        description: Human-readable description
        indicators: List of specific indicators found
    """
    category: ThreatCategory
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    indicators: List[str] = Field(default_factory=list)


class URLFeatures(BaseModel):
    """
    Extracted URL features for transparency/debugging.

    Attributes:
        url_length: Total character length of URL
        path_length: Length of the path component
        query_length: Length of query string
        subdomain_count: Number of subdomains
        url_entropy: Shannon entropy of URL characters
        obfuscation_score: Combined obfuscation indicators
        suspicious_pattern_score: Combined suspicious pattern score
        typosquatting_score: Typosquatting likelihood
        has_ip_address: URL contains IP instead of domain
        has_hex_encoding: URL contains hex-encoded characters
        punycode_detected: Domain uses punycode encoding
    """
    url_length: int = 0
    path_length: int = 0
    query_length: int = 0
    fragment_length: int = 0
    subdomain_count: int = 0
    subdomain_length: int = 0
    path_depth: int = 0
    url_entropy: float = 0.0
    domain_length: int = 0
    tld: str = ""
    is_free_domain_provider: bool = False
    has_hex_encoding: bool = False
    hex_encoded_chars: int = 0
    has_ip_address: bool = False
    has_at_symbol: bool = False
    obfuscation_score: float = 0.0
    suspicious_path_count: int = 0
    has_login_keywords: bool = False
    suspicious_pattern_score: float = 0.0
    special_char_count: int = 0
    digit_count: int = 0
    digit_ratio: float = 0.0
    uppercase_count: int = 0
    typosquatting_score: float = 0.0
    potential_brand: Optional[str] = None
    # DNS/WHOIS features (populated by domain_service)
    domain_age_days: Optional[int] = None
    is_recent_domain: Optional[bool] = None
    registrar_suspicious: Optional[bool] = None
    dns_record_exists: Optional[bool] = None


class AnalysisResult(BaseModel):
    """
    Complete analysis result for a single URL.

    This is the main response schema returned by /analyze endpoint.

    Attributes:
        id: Unique analysis ID
        url: Analyzed URL
        analyzed_at: Timestamp of analysis
        risk_level: Overall risk classification
        risk_score: Numeric risk score (0-100)
        is_malicious: Boolean indicator
        ml_confidence: ML model confidence
        ml_model_version: Version of the ML model
        ctis: List of CTI lookup results
        threats: List of detected threats
        features: Extracted features (if requested)
        processing_time_ms: Total analysis time
    """
    id: str
    url: str
    analyzed_at: datetime
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=100.0)
    is_malicious: bool
    ml_prediction: bool  # True = phishing, False = legitimate
    ml_confidence: float = Field(ge=0.0, le=1.0)
    ml_model_version: str
    ctis: List[CTIResult] = Field(default_factory=list)
    threats: List[ThreatDetail] = Field(default_factory=list)
    features: Optional[URLFeatures] = None
    processing_time_ms: float
    error: Optional[str] = None


class BatchAnalysisResult(BaseModel):
    """Response for batch analysis endpoint."""
    total: int
    analyzed: int
    threats_found: int
    results: List[AnalysisResult]
    processing_time_ms: float


# ============================================================================
# SCAN HISTORY & IoC SCHEMAS
# ============================================================================

class ScanRecord(BaseModel):
    """
    Record of a completed scan for history.
    """
    id: str
    url: str
    risk_level: RiskLevel
    risk_score: float
    is_malicious: bool
    analyzed_at: datetime
    threat_count: int


class IoCRecord(BaseModel):
    """
    Indicator of Compromise record for export/reporting.
    """
    type: str = "url"
    value: str
    threat_type: str
    risk_level: RiskLevel
    confidence: float
    first_seen: datetime
    last_seen: datetime
    tags: List[str] = Field(default_factory=list)
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# STATUS & HEALTH SCHEMAS
# ============================================================================

class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    api_version: str
    ml_model_loaded: bool
    ml_model_version: Optional[str]
    timestamp: datetime


class ModelInfo(BaseModel):
    """Information about the loaded ML model."""
    version: str
    feature_count: int
    training_date: str
    accuracy: float
    f1_score: float
    auc_score: float


# ============================================================================
# ERROR SCHEMAS
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    status_code: int


# ============================================================================
# CONFIGURATION SCHEMAS
# ============================================================================

class CTIConfig(BaseModel):
    """Configuration for CTI sources."""
    virustotal_enabled: bool = True
    virustotal_api_key: Optional[str] = None
    urlhaus_enabled: bool = True
    timeout_seconds: float = 2.0
    max_retries: int = 2


class PipelineConfig(BaseModel):
    """Configuration for analysis pipeline."""
    enable_ml: bool = True
    enable_cti: bool = True
    enable_dns_lookup: bool = True
    enable_whois: bool = True
    max_analysis_time_ms: float = 500.0