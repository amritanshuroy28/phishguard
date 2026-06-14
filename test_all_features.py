#!/usr/bin/env python3
"""
PhishGuard Comprehensive Test Suite
====================================
Tests:
1. Ultimate ML Model accuracy and inference
2. URL obfuscation decoding
3. Typosquatting detection
4. Live DNS/WHOIS lookups
5. API response time (sub-500ms target)
6. End-to-end API endpoints
7. Stored model metrics validation (>95% accuracy)

All ML inference uses the single ultimate XGBoost model
(phishguard_ultimate.pkl) — no ensemble sub-models.
"""

import os
import sys
import time
import json
import asyncio
import pickle
import requests
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

API_BASE = 'http://localhost:8000/api/v1'
PASS = '\033[92m PASS\033[0m'
FAIL = '\033[91m FAIL\033[0m'
WARN = '\033[93m WARN\033[0m'

results = {'passed': 0, 'failed': 0, 'warned': 0}

def report(name, passed, detail=''):
    global results
    # Convert numpy.bool_ to Python bool to avoid `is True` comparison issues
    if passed is not None:
        passed = bool(passed)
    if passed is True:
        results['passed'] += 1
        print(f'  {PASS}  {name}  {detail}')
    elif passed is None:
        results['warned'] += 1
        print(f'  {WARN}  {name}  {detail}')
    else:
        results['failed'] += 1
        print(f'  {FAIL}  {name}  {detail}')


# ============================================================================
# 1. ML PIPELINE - Ultimate Model Accuracy and Performance
# ============================================================================
print('\n' + '='*70)
print('1. ML PIPELINE - Ultimate Model Accuracy and Performance')
print('='*70)

from backend.services.ensemble_service import get_ensemble_service
service = get_ensemble_service()
info = service.get_info()

print(f'\n  Model: {info["model_name"]}')
print(f'  Version: {info["model_version"]}')
print(f'  Loaded: {info["loaded"]}')
print(f'  Features: {info["n_features"]}')
print(f'  Trained at: {info.get("trained_at", "unknown")}')

report('Ultimate model loaded', info['loaded'])

if info['loaded']:
    metrics = info.get('metrics', {})
    acc = metrics.get('accuracy', 0)
    f1 = metrics.get('f1_score', 0)
    auc = metrics.get('roc_auc', 0)
    prec = metrics.get('precision', 0)
    rec = metrics.get('recall', 0)
    cv_acc = info.get('cv_accuracy', 0)
    cv_f1 = info.get('cv_f1', 0)

    print(f'\n  --- Stored Metrics ---')
    print(f'    Test Accuracy:  {acc:.4f}')
    print(f'    Test Precision: {prec:.4f}')
    print(f'    Test Recall:    {rec:.4f}')
    print(f'    Test F1 Score:  {f1:.4f}')
    print(f'    Test ROC AUC:   {auc:.4f}')
    print(f'    CV Accuracy:    {cv_acc:.4f}')
    print(f'    CV F1:          {cv_f1:.4f}')

    report('Test accuracy >= 0.95', acc >= 0.95, f'({acc:.4f})')
    report('Test F1 score >= 0.95', f1 >= 0.95, f'({f1:.4f})')
    report('Test ROC AUC >= 0.95', auc >= 0.95, f'({auc:.4f})')
    report('CV accuracy >= 0.95', cv_acc >= 0.95, f'({cv_acc:.4f})')
    report('Feature count == 42', info['n_features'] == 42, f'({info["n_features"]})')

    # Training data info
    td = info.get('training_data', {})
    if td:
        print(f'\n  --- Training Data ---')
        print(f'    PhishDB samples:  {td.get("phishdb_count", "?")}')
        print(f'    UCI legit count:  {td.get("uci_legit_count", "?")}')
        print(f'    Trusted domains:  {td.get("trusted_count", "?")}')
        print(f'    Total samples:    {td.get("total_samples", "?")}')
else:
    report('Model metrics', None, 'model not loaded — skipping metric checks')

# Test inference speed
print('\n  --- Inference Speed ---')
test_features = np.random.rand(42).astype(np.float32)
times = []
for _ in range(100):
    t0 = time.perf_counter()
    is_phish, conf = service.predict(test_features)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    times.append(elapsed_ms)

avg_ms = np.mean(times)
p99_ms = np.percentile(times, 99)
print(f'  Inference: avg={avg_ms:.2f}ms, p99={p99_ms:.2f}ms (100 runs)')
report('Inference < 50ms avg', avg_ms < 50, f'({avg_ms:.2f}ms)')
report('Inference < 100ms p99', p99_ms < 100, f'({p99_ms:.2f}ms)')


# ============================================================================
# 2. URL OBFUSCATION DECODING
# ============================================================================
print('\n' + '='*70)
print('2. URL OBFUSCATION DECODING')
print('='*70)

from ml_pipeline.features.feature_extraction import URLFeatureExtractor
extractor = URLFeatureExtractor()

obfuscated_urls = [
    # Hex encoding
    'http://exam%70le.com/%70hishing',
    # Double encoding
    'http://example.com/%252F%252Fhidden',
    # IP address instead of domain
    'http://192.168.1.1/login',
    # @ symbol redirect
    'http://google.com@evil.com/login',
    # Double slash redirect
    'http://example.com//evil.com/login',
    # Data URI
    'data:text/html,<script>alert(1)</script>',
    # Unicode / punycode
    'http://xn--pple-43d.com/login',
    # Base64 in URL
    'http://evil.com/redirect?url=aHR0cDovL3BoaXNoLmNvbQ==',
]

print()
for url in obfuscated_urls:
    try:
        features = extractor.extract(url)
        obs_score = getattr(features, 'obfuscation_score', 0)
        has_hex = getattr(features, 'has_hex_encoding', False)
        has_ip = getattr(features, 'has_ip_address', False)
        has_at = getattr(features, 'has_at_symbol', False)
        has_dbl = getattr(features, 'has_double_slash_redirect', False)
        has_data = getattr(features, 'has_data_uri', False)
        puny = getattr(features, 'punycode_detected', False)

        indicators = []
        if has_hex: indicators.append('hex')
        if has_ip: indicators.append('ip')
        if has_at: indicators.append('@')
        if has_dbl: indicators.append('//')
        if has_data: indicators.append('data:')
        if puny: indicators.append('punycode')

        detected = obs_score > 0 or len(indicators) > 0
        flag_str = ','.join(indicators)
        report(
            f'Obfuscation: {url[:50]}',
            detected,
            f'score={obs_score}, flags=[{flag_str}]'
        )
    except Exception as e:
        report(f'Obfuscation: {url[:50]}', False, f'ERROR: {e}')


# ============================================================================
# 3. TYPOSQUATTING DETECTION
# ============================================================================
print('\n' + '='*70)
print('3. TYPOSQUATTING DETECTION')
print('='*70)

typosquat_urls = [
    ('http://g00gle.com', 'Google typosquat (0->o)'),
    ('http://gooogle.com', 'Google typosquat (extra o)'),
    ('http://facebo0k.com', 'Facebook typosquat (0->o)'),
    ('http://paypa1.com', 'PayPal typosquat (1->l)'),
    ('http://amaz0n.com', 'Amazon typosquat (0->o)'),
    ('http://micros0ft.com', 'Microsoft typosquat (0->o)'),
    ('http://apple-support.com', 'Apple typosquat (hyphen)'),
    ('http://google.com', 'Legitimate Google (control)'),
]

print()
for url, desc in typosquat_urls:
    try:
        features = extractor.extract(url)
        typo_score = getattr(features, 'typosquatting_score', 0)
        brand_sub = getattr(features, 'has_brand_in_subdomain', False)
        sus_score = getattr(features, 'suspicious_pattern_score', 0)
        
        is_control = 'control' in desc.lower()
        if is_control:
            report(f'Typosquat control: {desc}', typo_score <= 1, f'score={typo_score}')
        else:
            report(f'Typosquat detect: {desc}', typo_score > 0, f'score={typo_score}')
    except Exception as e:
        report(f'Typosquat detect: {desc}', False, f'ERROR: {e}')


# ============================================================================
# 4. LIVE DNS/WHOIS LOOKUPS
# ============================================================================
print('\n' + '='*70)
print('4. LIVE DNS/WHOIS LOOKUPS')
print('='*70)

from backend.services.domain_service import get_domain_service
domain_svc = get_domain_service()

async def test_dns_whois():
    # DNS tests
    dns_tests = [
        ('google.com', True),
        ('this-domain-definitely-does-not-exist-xyz123.com', False),
        ('github.com', True),
    ]
    
    print('\n  --- DNS Resolution ---')
    for domain, should_resolve in dns_tests:
        try:
            result = await domain_svc.dns_lookup(domain)
            resolved = result.dns_resolves
            resp_time = result.response_time_ms
            ips = result.ip_addresses[:2] if result.ip_addresses else []
            
            if should_resolve:
                report(f'DNS resolve {domain}', resolved, f'{ips} ({resp_time:.0f}ms)')
            else:
                report(f'DNS non-resolve {domain}', not resolved, f'({resp_time:.0f}ms)')
        except Exception as e:
            report(f'DNS lookup {domain}', False, f'ERROR: {e}')
    
    # WHOIS tests
    print('\n  --- WHOIS Lookups ---')
    whois_tests = ['google.com', 'github.com']
    
    for domain in whois_tests:
        try:
            result = await domain_svc.whois_lookup(domain)
            found = result.found
            age = result.domain_age_days
            registrar = result.registrar
            resp_time = result.response_time_ms
            
            report(
                f'WHOIS lookup {domain}',
                found,
                f'age={age}d, registrar={registrar}, ({resp_time:.0f}ms)'
            )
        except Exception as e:
            report(f'WHOIS lookup {domain}', None, f'ERROR: {e}')

asyncio.run(test_dns_whois())


# ============================================================================
# 5. API ENDPOINT TESTS and RESPONSE TIME
# ============================================================================
print('\n' + '='*70)
print('5. API ENDPOINT TESTS and RESPONSE TIME')
print('='*70)

# Health check
print('\n  --- Health and Info ---')
try:
    t0 = time.perf_counter()
    r = requests.get(f'{API_BASE}/health', timeout=5)
    elapsed = (time.perf_counter() - t0) * 1000
    health = r.json()
    report('Health endpoint', r.status_code == 200, f'status={health.get("status")} ({elapsed:.0f}ms)')
    report('ML model loaded', health.get('ml_model_loaded', False))
    report('Model version', 'ultimate' in str(health.get('ml_model_version', '')),
           f'{health.get("ml_model_version")}')
except Exception as e:
    report('Health endpoint', False, str(e))

try:
    r = requests.get(f'{API_BASE}/model/info', timeout=5)
    model = r.json()
    report('Model info endpoint', r.status_code == 200, f'version={model.get("version")}')
    acc = model.get('accuracy', 0)
    report(f'Reported accuracy >= 0.95', acc >= 0.95, f'{acc:.4f}')
except Exception as e:
    report('Model info endpoint', False, str(e))

# Single URL analysis
print('\n  --- Single URL Analysis (sub-500ms target) ---')
test_urls = [
    ('https://www.google.com', 'legit'),
    ('https://www.github.com', 'legit'),
    ('http://paypa1-secure-login.suspicious-domain.tk/verify', 'phishing'),
    ('http://192.168.1.1/admin/login.php?redirect=http://evil.com', 'phishing'),
    ('http://g00gle.com/account/verify', 'phishing-typosquat'),
    ('http://secure-bankofamerica.tk/%70hishing', 'phishing-obfuscated'),
    ('http://xn--pple-43d.com/appleid/login', 'phishing-punycode'),
    ('http://login-microsoft.com@evil.com/verify', 'phishing-at-redirect'),
]

for url, expected_type in test_urls:
    try:
        t0 = time.perf_counter()
        r = requests.post(
            f'{API_BASE}/analyze',
            json={'url': url, 'include_raw_features': True, 'enable_cti': False, 'fast_mode': True},
            timeout=10
        )
        elapsed = (time.perf_counter() - t0) * 1000
        
        if r.status_code == 200:
            data = r.json()
            risk = data.get('risk_level', 'unknown')
            score = data.get('risk_score', 0)
            is_mal = data.get('is_malicious', False)
            ml_conf = data.get('ml_confidence', 0)
            proc_time = data.get('processing_time_ms', 0)
            
            report(
                f'Analyze {url[:55]}',
                True,
                f'risk={risk}, score={score:.1f}, mal={is_mal}, conf={ml_conf:.2f}, time={elapsed:.0f}ms'
            )
            report(f'  Response < 500ms', elapsed < 500, f'({elapsed:.0f}ms)')
        else:
            report(f'Analyze {url[:55]}', False, f'HTTP {r.status_code}: {r.text[:100]}')
    except Exception as e:
        report(f'Analyze {url[:55]}', False, f'ERROR: {e}')

# Batch analysis
print('\n  --- Batch Analysis ---')
try:
    batch_urls = [
        'https://google.com',
        'http://paypa1.com/login',
        'http://192.168.1.1/phish',
    ]
    t0 = time.perf_counter()
    r = requests.post(
        f'{API_BASE}/analyze/batch',
        json={'urls': batch_urls, 'enable_cti': False},
        timeout=30
    )
    elapsed = (time.perf_counter() - t0) * 1000
    if r.status_code == 200:
        batch = r.json()
        report(
            'Batch analysis',
            True,
            f'total={batch["total"]}, threats={batch["threats_found"]}, time={elapsed:.0f}ms'
        )
    else:
        report('Batch analysis', False, f'HTTP {r.status_code}')
except Exception as e:
    report('Batch analysis', False, str(e))

# History and stats
print('\n  --- History and Stats ---')
try:
    r = requests.get(f'{API_BASE}/history', timeout=5)
    report('History endpoint', r.status_code == 200, f'{len(r.json())} records')
except Exception as e:
    report('History endpoint', False, str(e))

try:
    r = requests.get(f'{API_BASE}/stats', timeout=5)
    if r.status_code == 200:
        stats = r.json()
        report('Stats endpoint', True, f'scans={stats["total_scans"]}, threats={stats["threats_detected"]}')
    else:
        report('Stats endpoint', False, f'HTTP {r.status_code}')
except Exception as e:
    report('Stats endpoint', False, str(e))


# ============================================================================
# 6. ULTIMATE MODEL — STORED METRICS VALIDATION
# ============================================================================
print('\n' + '='*70)
print('6. ULTIMATE MODEL — STORED METRICS VALIDATION (>95% target)')
print('='*70)

ultimate_pkl = PROJECT_ROOT / 'models' / 'phishguard_ultimate.pkl'
if ultimate_pkl.exists():
    with open(ultimate_pkl, 'rb') as f:
        pkg = pickle.load(f)

    metrics = pkg.get('metrics', {})
    cv_acc = pkg.get('cv_accuracy', 0)
    cv_f1 = pkg.get('cv_f1', 0)
    cm = pkg.get('confusion_matrix', [])
    td = pkg.get('training_data', {})
    params = pkg.get('params', {})

    acc = metrics.get('accuracy', 0)
    f1 = metrics.get('f1_score', 0)
    auc = metrics.get('roc_auc', 0)
    prec = metrics.get('precision', 0)
    rec = metrics.get('recall', 0)

    print(f'\n  Model file: {ultimate_pkl}')
    print(f'  File size: {ultimate_pkl.stat().st_size / 1024:.1f} KB')
    print(f'  Trained at: {pkg.get("trained_at", "unknown")}')
    print(f'\n  --- Test Set Metrics ---')
    print(f'    Accuracy:  {acc:.4f}')
    print(f'    Precision: {prec:.4f}')
    print(f'    Recall:    {rec:.4f}')
    print(f'    F1 Score:  {f1:.4f}')
    print(f'    ROC AUC:   {auc:.4f}')
    print(f'\n  --- Cross-Validation ---')
    print(f'    CV Accuracy: {cv_acc:.4f}')
    print(f'    CV F1:       {cv_f1:.4f}')
    print(f'\n  --- Confusion Matrix ---')
    if cm:
        print(f'    TN={cm[0][0]:,}  FP={cm[0][1]:,}')
        print(f'    FN={cm[1][0]:,}  TP={cm[1][1]:,}')
    print(f'\n  --- Training Data ---')
    for k, v in td.items():
        print(f'    {k}: {v:,}' if isinstance(v, int) else f'    {k}: {v}')
    print(f'\n  --- XGBoost Params ---')
    for k, v in params.items():
        print(f'    {k}: {v}')

    # Strict validation (>= 0.95)
    report('Accuracy >= 95%', acc >= 0.95, f'{acc:.4f}')
    report('Precision >= 95%', prec >= 0.95, f'{prec:.4f}')
    report('Recall >= 95%', rec >= 0.95, f'{rec:.4f}')
    report('F1 Score >= 95%', f1 >= 0.95, f'{f1:.4f}')
    report('ROC AUC >= 95%', auc >= 0.95, f'{auc:.4f}')
    report('CV Accuracy >= 95%', cv_acc >= 0.95, f'{cv_acc:.4f}')
else:
    report('Ultimate model file exists', False, f'{ultimate_pkl}')


# ============================================================================
# 7. CLASSIFICATION CORRECTNESS — Known URLs
# ============================================================================
print('\n' + '='*70)
print('7. CLASSIFICATION CORRECTNESS — Known URLs')
print('='*70)

if info['loaded']:
    known_phishing = [
        'http://paypa1-secure-login.suspicious-domain.tk/verify/account',
        'http://192.168.1.1/admin/login.php?redirect=http://evil.com',
        'http://g00gle.com/account/verify/reset?token=abc123',
        'http://secure-bankofamerica.tk/%70hishing/login',
        'http://xn--pple-43d.com/appleid/login/verify',
        'http://login-microsoft.com@evil.com/verify/credentials',
        'http://facebook-security-alert.tk/verify.php?id=12345',
        'http://amaz0n-order-update.ml/signin?ref=email',
    ]

    known_legit = [
        'https://www.google.com/',
        'https://github.com/features',
        'https://en.wikipedia.org/wiki/Phishing',
        'https://stackoverflow.com/questions',
        'https://www.amazon.com/dp/B08N5WRWNW',
        'https://www.microsoft.com/en-us/windows',
        'https://www.apple.com/iphone',
        'https://www.netflix.com/browse',
    ]

    print('\n  --- Phishing URLs (should be flagged) ---')
    phish_correct = 0
    for url in known_phishing:
        try:
            feats = extractor.extract(url)
            is_phish, conf = service.predict(feats.to_feature_array())
            phish_correct += int(is_phish)
            report(f'  Phish: {url[:55]}', is_phish, f'conf={conf:.3f}')
        except Exception as e:
            report(f'  Phish: {url[:55]}', False, f'ERROR: {e}')

    print('\n  --- Legitimate URLs (should NOT be flagged) ---')
    legit_correct = 0
    for url in known_legit:
        try:
            feats = extractor.extract(url)
            is_phish, conf = service.predict(feats.to_feature_array())
            legit_correct += int(not is_phish)
            report(f'  Legit: {url[:55]}', not is_phish, f'conf={conf:.3f}')
        except Exception as e:
            report(f'  Legit: {url[:55]}', False, f'ERROR: {e}')

    total_known = len(known_phishing) + len(known_legit)
    correct_known = phish_correct + legit_correct
    known_acc = correct_known / total_known if total_known else 0
    print(f'\n  Known URL accuracy: {correct_known}/{total_known} ({known_acc:.1%})')
    report('Known URL accuracy >= 87.5%', known_acc >= 0.875, f'{known_acc:.1%}')
else:
    report('Classification tests', None, 'model not loaded — skipping')


# ============================================================================
# SUMMARY
# ============================================================================
print('\n' + '='*70)
print('TEST SUMMARY')
print('='*70)
total = results['passed'] + results['failed'] + results['warned']
print(f'  Total:   {total}')
print(f'  Passed:  {results["passed"]}')
print(f'  Failed:  {results["failed"]}')
print(f'  Warned:  {results["warned"]}')
pass_rate = results['passed'] / total * 100 if total else 0
print(f'  Pass Rate: {pass_rate:.1f}%')
print('='*70)

# Ultimate model accuracy check
if ultimate_pkl.exists():
    with open(ultimate_pkl, 'rb') as f:
        d = pickle.load(f)
    acc = d.get('metrics', {}).get('accuracy', 0)
    print(f'\n  Ultimate model test accuracy: {acc:.4f}')
    print(f'  Target: >= 0.95 (strict)')
    if acc >= 0.95:
        print(f'  {PASS} TARGET MET ✅')
    else:
        print(f'  {FAIL} TARGET NOT MET ❌ — gap: {0.95 - acc:.4f}')
else:
    print(f'\n  {FAIL} Ultimate model file not found!')

print()

# Exit with error code if any test failed
if results['failed'] > 0:
    sys.exit(1)
