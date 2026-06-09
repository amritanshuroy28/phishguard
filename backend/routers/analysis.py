"""
PhishGuard Analysis Router
===========================
API endpoints for URL analysis.

POST /analyze - Analyze a single URL
POST /analyze/batch - Analyze multiple URLs
GET /health - Health check
GET /model/info - Model information
GET /history - Scan history
GET /iocs - Export IoCs
"""

import sys
import os
import time
import uuid
import logging
from dataclasses import asdict
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

# Import schemas
from schemas.schemas import (
    AnalyzeRequest, BatchAnalyzeRequest,
    AnalysisResult, BatchAnalysisResult,
    RiskLevel, ThreatCategory, ThreatDetail,
    CTIResult, URLFeatures,
    HealthStatus, ModelInfo,
    ScanRecord, IoCRecord
)

# Import services
from services.ml_service import get_ml_service
from services.cti_service import get_cti_service
from services.domain_service import get_domain_service

# For feature extraction
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml_pipeline"))
from features.feature_extraction import URLFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["analysis"])

# Global instances
ml_service = get_ml_service()
cti_service = get_cti_service()
domain_service = get_domain_service()
feature_extractor = URLFeatureExtractor()

# In-memory storage for demo (use database in production)
# Global list of scan history
_scan_history: List[AnalysisResult] = []
_history_max_size = 1000


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_risk_score(
    ml_confidence: float,
    is_ml_phishing: bool,
    cti_results: List[CTIResult],
    features: URLFeatures,
    threats: List[ThreatDetail]
) -> tuple[RiskLevel, float]:
    """
    Calculate overall risk score from all available signals.

    Combines:
    - ML model confidence
    - CTI hits (VirusTotal, URLhaus)
    - Feature-based heuristics
    - Threat details

    Returns:
        Tuple of (RiskLevel, numeric_score_0_100)
    """
    # Base score from ML
    if is_ml_phishing:
        base_score = ml_confidence * 60  # 0-60 based on ML
    else:
        base_score = (1 - ml_confidence) * 30  # 0-30 for legitimate

    # CTI contribution
    cti_score = 0.0
    max_detection_rate = 0.0

    for cti in cti_results:
        if cti.malicious:
            # High weight for confirmed malicious
            cti_score += 25 * cti.detection_rate
            max_detection_rate = max(max_detection_rate, cti.detection_rate)

    # Feature-based scoring
    feature_score = 0.0

    # Obfuscation indicators
    feature_score += min(features.obfuscation_score * 4, 8)

    # Suspicious patterns
    feature_score += min(features.suspicious_pattern_score * 5, 10)

    # Typosquatting
    feature_score += min(features.typosquatting_score * 6, 6)

    # Domain age (recent = suspicious)
    if features.domain_length > 0:
        # Brand in subdomain
        if features.has_login_keywords:
            feature_score += 5

        # IP address instead of domain
        if features.has_ip_address:
            feature_score += 8

        # Hex encoding
        if features.has_hex_encoding:
            feature_score += 3

    # Threat severity
    threat_score = 0.0
    for threat in threats:
        if threat.category in [ThreatCategory.PHISHING, ThreatCategory.CREDENTIAL_HARVEST]:
            threat_score += threat.confidence * 15
        else:
            threat_score += threat.confidence * 5

    # Combine all scores
    total_score = base_score + cti_score + feature_score + threat_score
    total_score = min(100.0, max(0.0, total_score))

    # Determine risk level
    if total_score >= 80:
        risk_level = RiskLevel.CRITICAL
    elif total_score >= 60:
        risk_level = RiskLevel.HIGH
    elif total_score >= 40:
        risk_level = RiskLevel.MEDIUM
    elif total_score >= 20:
        risk_level = RiskLevel.LOW
    else:
        risk_level = RiskLevel.SAFE

    return risk_level, total_score


def extract_threats(
    features: URLFeatures,
    cti_results: List[CTIResult]
) -> List[ThreatDetail]:
    """
    Extract detailed threat information from features and CTI results.
    """
    threats = []

    # Typosquatting
    if features.typosquatting_score > 1.0:
        threats.append(ThreatDetail(
            category=ThreatCategory.TYPOSQUATTING,
            confidence=min(features.typosquatting_score / 2.0, 1.0),
            description=f"Potential typosquatting of brand: {features.potential_brand}",
            indicators=[f"Edit distance match: {features.potential_brand}"]
        ))

    # Phishing
    if features.obfuscation_score > 1.5:
        threats.append(ThreatDetail(
            category=ThreatCategory.OBFUSCATION,
            confidence=min(features.obfuscation_score / 3.0, 1.0),
            description="URL contains obfuscation indicators",
            indicators=[
                f"Has hex encoding: {features.has_hex_encoding}",
                f"Contains IP address: {features.has_ip_address}",
                "Contains @ symbol redirect" if features.has_at_symbol else ""
            ]
        ))

    # Credential harvesting
    if features.has_login_keywords and features.suspicious_pattern_score > 0.5:
        threats.append(ThreatDetail(
            category=ThreatCategory.CREDENTIAL_HARVEST,
            confidence=min(features.suspicious_pattern_score / 2.0, 1.0),
            description="Page may be attempting to harvest credentials",
            indicators=["Contains login/authentication keywords"]
        ))

    # CTI-based threats
    for cti in cti_results:
        if cti.malicious:
            if cti.source.value == "virustotal":
                threats.append(ThreatDetail(
                    category=ThreatCategory.PHISHING,
                    confidence=cti.detection_rate,
                    description=f"Flagged by {cti.positives} VirusTotal vendors",
                    indicators=cti.metadata.get("malicious_vendors", [])[:5]
                ))
            elif cti.source.value == "urlhaus":
                threat_type = cti.metadata.get("threat_type", "unknown")
                threats.append(ThreatDetail(
                    category=ThreatCategory.MALWARE if "malware" in threat_type else ThreatCategory.PHISHING,
                    confidence=1.0,
                    description=f"Listed in URLhaus - {threat_type}",
                    indicators=[f"Threat type: {threat_type}"]
                ))

    # Recent domain (if domain_age_days available)
    if features.domain_length > 0:
        # If we only have features without WHOIS, use heuristics
        if features.url_entropy > 7.0 and features.digit_count > features.url_length * 0.3:
            threats.append(ThreatDetail(
                category=ThreatCategory.RECENT_DOMAIN,
                confidence=0.6,
                description="URL has high entropy (random-looking) - may indicate new domain",
                indicators=[f"URL entropy: {features.url_entropy:.2f}", "High digit ratio"]
            ))

    return threats


def features_to_schema(features_obj) -> URLFeatures:
    """Convert feature extraction result to schema."""
    return URLFeatures(
        url_length=features_obj.url_length,
        path_length=features_obj.path_length,
        query_length=features_obj.query_length,
        fragment_length=features_obj.fragment_length,
        subdomain_count=features_obj.subdomain_count,
        subdomain_length=features_obj.subdomain_length,
        path_depth=features_obj.path_depth,
        url_entropy=features_obj.url_entropy,
        domain_length=features_obj.domain_length,
        tld=features_obj.tld,
        is_free_domain_provider=features_obj.is_free_domain_provider,
        has_hex_encoding=features_obj.has_hex_encoding,
        hex_encoded_chars=features_obj.hex_encoded_chars,
        has_ip_address=features_obj.has_ip_address,
        has_at_symbol=features_obj.has_at_symbol,
        obfuscation_score=features_obj.obfuscation_score,
        suspicious_path_count=features_obj.suspicious_path_count,
        has_login_keywords=features_obj.has_login_keywords,
        suspicious_pattern_score=features_obj.suspicious_pattern_score,
        special_char_count=features_obj.special_char_count,
        digit_count=features_obj.digit_count,
        digit_ratio=features_obj.digit_ratio,
        uppercase_count=features_obj.uppercase_count,
        typosquatting_score=features_obj.typosquatting_score,
        potential_brand=features_obj.potential_brand,
        # DNS/WHOIS fields (populated by domain_service)
        domain_age_days=getattr(features_obj, 'domain_age_days', None),
        is_recent_domain=getattr(features_obj, 'is_recent_domain', None),
        registrar_suspicious=getattr(features_obj, 'registrar_suspicious', None),
        dns_record_exists=getattr(features_obj, 'dns_record_exists', None),
    )


async def analyze_url_internal(
    url: str,
    include_raw_features: bool = False,
    enable_cti: bool = True
) -> AnalysisResult:
    """
    Internal URL analysis - orchestrates feature extraction, ML inference, and CTI lookups.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    features_obj = None
    error_msg = None

    try:
        # 1. Domain intelligence: DNS/WHOIS lookups (async, runs concurrently with feature extraction)
        logger.info(f"[{analysis_id}] Gathering domain intelligence...")
        domain_lookups = await domain_service.lookup_all(url)

        # Separate WHOIS and DNS results
        whois_data = None
        dns_data = None
        for lookup in domain_lookups:
            lookup_dict = asdict(lookup)
            if lookup.source == "whois":
                # Build WHOIS dict without source/domain fields
                whois_data = {k: v for k, v in lookup_dict.items()
                              if k not in ['source', 'domain', 'response_time_ms', 'dns_error', 'whois_error']}
            elif lookup.source == "dns":
                # Build DNS dict without source/domain fields
                dns_data = {k: v for k, v in lookup_dict.items()
                            if k not in ['source', 'domain', 'response_time_ms', 'dns_error', 'whois_error']}

        # 2. Extract features (synchronous, fast) with DNS/WHOIS enrichment
        logger.info(f"[{analysis_id}] Extracting features for: {url[:60]}...")
        features_obj = feature_extractor.extract_with_dns(url, whois_result=whois_data, dns_result=dns_data)
        features = features_to_schema(features_obj)

        # 3. Prepare feature array for ML
        feature_array = features_obj.to_feature_array()

        # 4. ML inference (synchronous)
        logger.info(f"[{analysis_id}] Running ML inference...")
        is_ml_phishing, ml_confidence = ml_service.predict(feature_array)

        # 5. CTI lookups (asynchronous, concurrent)
        cti_results: List[CTIResult] = []
        if enable_cti:
            logger.info(f"[{analysis_id}] Querying CTI sources...")
            cti_lookup_results = await cti_service.lookup_all(url)

            # Convert to schema
            for cti in cti_lookup_results:
                cti_results.append(CTIResult(
                    source=cti.source,
                    found=cti.found,
                    malicious=cti.malicious,
                    positives=cti.positives,
                    total=cti.total,
                    detection_rate=cti.detection_rate,
                    metadata=cti.metadata,
                    error=cti.error,
                    response_time_ms=cti.response_time_ms
                ))

        # 5. Extract threats
        threats = extract_threats(features, cti_results)

        # 6. Calculate risk
        risk_level, risk_score = calculate_risk_score(
            ml_confidence, is_ml_phishing, cti_results, features, threats
        )

        # Determine final malicious flag
        is_malicious = (
            is_ml_phishing and ml_confidence > 0.7 or
            any(cti.malicious for cti in cti_results) or
            risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        )

        # 7. Build response
        processing_time = (time.time() - start_time) * 1000

        result = AnalysisResult(
            id=analysis_id,
            url=url,
            analyzed_at=datetime.now(timezone.utc),
            risk_level=risk_level,
            risk_score=round(risk_score, 2),
            is_malicious=is_malicious,
            ml_prediction=is_ml_phishing,
            ml_confidence=round(ml_confidence, 4),
            ml_model_version=ml_service.version,
            ctis=cti_results,
            threats=threats,
            features=features if include_raw_features else None,
            processing_time_ms=round(processing_time, 2),
            error=error_msg
        )

        # Add to history
        _scan_history.insert(0, result)
        if len(_scan_history) > _history_max_size:
            _scan_history.pop()

        logger.info(
            f"[{analysis_id}] Analysis complete: "
            f"risk={risk_level.value}, score={risk_score:.1f}, time={processing_time:.1f}ms"
        )

        return result

    except Exception as e:
        logger.error(f"[{analysis_id}] Analysis error: {e}")
        processing_time = (time.time() - start_time) * 1000

        return AnalysisResult(
            id=analysis_id,
            url=url,
            analyzed_at=datetime.now(timezone.utc),
            risk_level=RiskLevel.LOW,
            risk_score=0.0,
            is_malicious=False,
            ml_prediction=False,
            ml_confidence=0.0,
            ml_model_version=ml_service.version,
            ctis=[],
            threats=[],
            features=features if include_raw_features and features_obj else None,
            processing_time_ms=round(processing_time, 2),
            error=str(e)
        )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_url(request: AnalyzeRequest):
    """
    Analyze a single URL for phishing indicators.

    This endpoint orchestrates:
    1. Feature extraction (lexical, structural)
    2. ML model inference (XGBoost)
    3. Live CTI lookups (VirusTotal, URLhaus)

    Response time target: <500ms
    """
    result = await analyze_url_internal(
        url=request.url,
        include_raw_features=request.include_raw_features,
        enable_cti=request.enable_cti
    )
    return result


@router.post("/analyze/batch", response_model=BatchAnalysisResult)
async def batch_analyze(request: BatchAnalyzeRequest):
    """
    Analyze multiple URLs in batch.

    Each URL is processed through the full analysis pipeline.
    Results are returned as a list.
    """
    start_time = time.time()

    # Process all URLs concurrently with a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent analyses

    async def process_with_limit(url: str):
        async with semaphore:
            return await analyze_url_internal(
                url=url,
                include_raw_features=False,
                enable_cti=request.enable_cti
            )

    # Create tasks
    tasks = [process_with_limit(url) for url in request.urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    analysis_results = []
    threats_count = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle individual URL errors
            analysis_results.append(AnalysisResult(
                id=str(uuid.uuid4()),
                url=request.urls[i],
                analyzed_at=datetime.now(timezone.utc),
                risk_level=RiskLevel.LOW,
                risk_score=0.0,
                is_malicious=False,
                ml_prediction=False,
                ml_confidence=0.0,
                ml_model_version=ml_service.version,
                ctis=[],
                threats=[],
                processing_time_ms=0.0,
                error=str(result)
            ))
        else:
            analysis_results.append(result)
            if result.threats:
                threats_count += 1

    total_time = (time.time() - start_time) * 1000

    return BatchAnalysisResult(
        total=len(request.urls),
        analyzed=len(analysis_results),
        threats_found=threats_count,
        results=analysis_results,
        processing_time_ms=round(total_time, 2)
    )


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Health check endpoint.
    Returns API status and model loading information.
    """
    return HealthStatus(
        status="healthy",
        api_version="1.0.0",
        ml_model_loaded=ml_service.is_loaded,
        ml_model_version=ml_service.version if ml_service.is_loaded else None,
        timestamp=datetime.now(timezone.utc)
    )


@router.get("/model/info", response_model=ModelInfo)
async def model_info():
    """
    Get information about the loaded ML model.
    """
    info = ml_service.get_model_info()

    return ModelInfo(
        version=info["version"],
        feature_count=info["feature_count"],
        training_date=info.get("training_date", "unknown"),
        accuracy=0.0,  # Would be loaded from model metadata
        f1_score=0.0,
        auc_score=0.0
    )


@router.get("/history", response_model=List[AnalysisResult])
async def get_history(
    limit: int = Query(default=50, ge=1, le=500),
    risk_level: Optional[RiskLevel] = None
):
    """
    Get recent analysis history.

    Args:
        limit: Maximum number of records to return
        risk_level: Filter by risk level
    """
    history = _scan_history

    if risk_level:
        history = [r for r in history if r.risk_level == risk_level]

    return history[:limit]


@router.delete("/history")
async def clear_history():
    """Clear analysis history."""
    global _scan_history
    cleared = len(_scan_history)
    _scan_history = []
    return {"message": f"Cleared {cleared} records"}


@router.get("/iocs", response_model=List[IoCRecord])
async def get_iocs(
    limit: int = Query(default=100, ge=1, le=1000),
    min_risk_score: float = Query(default=50.0, ge=0, le=100)
):
    """
    Get Indicators of Compromise (IoCs).

    Exports URLs that were flagged as threats, suitable for
    integration with SIEM/SOC tools.

    Args:
        limit: Maximum number of IoCs to return
        min_risk_score: Minimum risk score threshold
    """
    iocs = []

    for record in _scan_history:
        if record.risk_score >= min_risk_score and record.is_malicious:
            # Determine threat type from threats
            threat_types = [t.category.value for t in record.threats]
            primary_threat = threat_types[0] if threat_types else "unknown"

            # Get tags from CTI results
            tags = []
            for cti in record.ctis:
                if cti.found:
                    tags.append(f"cti:{cti.source.value}")
            for threat in record.threats:
                tags.append(f"threat:{threat.category.value}")

            ioc = IoCRecord(
                type="url",
                value=record.url,
                threat_type=primary_threat,
                risk_level=record.risk_level,
                confidence=record.ml_confidence,
                first_seen=record.analyzed_at,
                last_seen=record.analyzed_at,
                tags=tags,
                source="phishguard",
                metadata={
                    "risk_score": record.risk_score,
                    "threats": [t.description for t in record.threats],
                    "cti_results": [
                        {"source": cti.source.value, "malicious": cti.malicious}
                        for cti in record.ctis if cti.found
                    ]
                }
            )
            iocs.append(ioc)

    return iocs[:limit]


@router.get("/iocs/export")
async def export_iocs(
    format: str = Query(default="json", regex="^(json|csv)$"),
    min_risk_score: float = Query(default=50.0, ge=0, le=100)
):
    """
    Export IoCs as JSON or CSV.

    Args:
        format: Output format (json or csv)
        min_risk_score: Minimum risk score threshold
    """
    iocs = await get_iocs(limit=10000, min_risk_score=min_risk_score)

    if format == "json":
        import json
        content = json.dumps([ioc.model_dump() for ioc in iocs], indent=2, default=str)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=iocs.json"}
        )

    elif format == "csv":
        import csv
        import io
        import json

        output = io.StringIO()
        if iocs:
            fieldnames = list(iocs[0].model_dump().keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for ioc in iocs:
                row = ioc.model_dump()
                # Convert datetime:
                for key, value in list(row.items()):
                    if value is None:
                        row[key] = ""
                    elif isinstance(value, datetime):
                        row[key] = value.isoformat()
                    elif isinstance(value, dict):
                        row[key] = json.dumps(value)
                    elif isinstance(value, list):
                        row[key] = ", ".join(str(v) for v in value)
                    else:
                        row[key] = str(value)
                writer.writerow(row)

        content = output.getvalue()
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=iocs.csv"}
        )


@router.get("/stats")
async def get_stats():
    """
    Get current API statistics.

    Returns aggregated stats from scan history.
    """
    # Aggregate stats
    total_scans = len(_scan_history)
    threats_detected = sum(1 for r in _scan_history if r.is_malicious)
    avg_risk_score = (sum(r.risk_score for r in _scan_history) / total_scans
                     if total_scans > 0 else 0.0)

    # Risk distribution
    risk_distribution = {}
    for level in ["safe", "low", "medium", "high", "critical"]:
        risk_distribution[level] = sum(
            1 for r in _scan_history if r.risk_level.value == level
        )

    return {
        "total_scans": total_scans,
        "threats_detected": threats_detected,
        "average_risk_score": round(avg_risk_score, 2),
        "threat_rate": round(threats_detected / total_scans * 100, 2) if total_scans > 0 else 0,
        "risk_distribution": risk_distribution,
        "ml_model_loaded": True,
        "uptime": "operational"
    }