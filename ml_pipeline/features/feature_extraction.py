"""
PhishGuard Feature Extraction Module
=====================================
This module extracts comprehensive lexical, structural, and behavioral features
from URLs for ML-based phishing detection.

Feature Categories:
1. URL Length Features - Total length, path depth, entropy
2. Domain Features - Subdomain analysis, TLD patterns
3. Obfuscation Features - Hex encoding, URL encoding, special characters
4. Suspicious Patterns - Login forms, credential harvesting, IP addresses
5. DNS/WHOIS Anomalies - Domain age, registration patterns
6. Typosquatting Analysis - Levenshtein distance to legitimate brands
"""

import re
import math
import signal
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
from urllib.parse import urlparse, parse_qs, unquote
from difflib import SequenceMatcher
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & PATTERNS
# ============================================================================

# Trusted brands for typosquatting detection
KNOWN_BRANDS = {
    "google": ["google.com", "googledrive.com", "googlemail.com"],
    "facebook": ["facebook.com", "fb.com", "fbcdn.net"],
    "amazon": ["amazon.com", "amazonaws.com", "amazon.co.jp"],
    "apple": ["apple.com", "icloud.com", "apple.co"],
    "microsoft": ["microsoft.com", "live.com", "outlook.com", "office.com"],
    "paypal": ["paypal.com", "paypal.me"],
    "netflix": ["netflix.com", "netflix.net"],
    "github": ["github.com", "github.io", "ghcr.io"],
    "linkedin": ["linkedin.com"],
    "twitter": ["twitter.com", "x.com", "twttr.com"],
    "instagram": ["instagram.com"],
    "bank": ["chase.com", "bankofamerica.com", "wellsfargo.com", "citi.com"],
    "ebay": ["ebay.com", "ebay.co.uk"],
    "reddit": ["reddit.com", "redditmail.com"],
    "dropbox": ["dropbox.com", "dropboxapi.com"],
    "netbank": ["anz.com", "commbank.com.au", "nab.com.au", "westpac.com.au"],
}

# Suspicious path patterns (credential harvesting, phishing kits)
SUSPICIOUS_PATHS = [
    r"login", r"signin", r"auth", r"verify", r"secure", r"account",
    r"update", r"confirm", r"banking", r"wallet", r"password", r"credential",
    r"oauth", r"callback", r"redirect", r"return", r"token",
    r"admin", r"dashboard", r"control", r"panel", r"webscr",
    r"phish", r"hack", r"malware", r"malicious",
]

# Obfuscation patterns
OBFUSCATION_PATTERNS = {
    "hex_encoded": [
        r"%[0-9A-Fa-f]{2}",  # URL-encoded hex
        r"\\x[0-9A-Fa-f]{2}",  # CSS/JavaScript hex escape
        r"0x[0-9A-Fa-f]+",  # Numeric hex representation
    ],
    "ip_address": [
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IPv4
        r"\[[0-9a-fA-F:]+\]",  # IPv6
    ],
    "at_symbol": r"@",  # @ redirects
    "double_slash_redirect": r"^https?://[^/]{2,}",  # Protocol-relative
    "data_uri": r"^data:",  # Data URI scheme
    "javascript_uri": r"^javascript:",  # JavaScript execution
}

# Special TLDs associated with phishing
PHISHING_TLDS = {
    r"\.(xyz|tk|ml|ga|cf|gq|top|work|click|link|info|buzz|online|site|website|space|host|fun|pw|cc|ws|ms)$": 1.0,
    r"\.(ru|cn|ua|kz|by|su)$": 0.7,  # Eastern European/Asian TLDs
    r"\.(club|live|icu|buzz|fit|gym|art|inc|llc)$": 0.8,
}

# Common legitimate TLDs (score reduction)
LEGITIMATE_TLDS = {
    r".*\.(com|org|net|edu|gov|mil|co)$": -0.3,
    r".*\.(io|dev|app|ai|dev|cc)$": -0.2,  # Tech-focused, still legitimate
}

# Default timeout for DNS/WHOIS lookups
DEFAULT_TIMEOUT = 3.0


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class URLFeatures:
    """
    Container for all extracted URL features.
    Uses a dataclass for structured access and serialization.
    """
    # URL Length Features
    url_length: int
    path_length: int
    query_length: int
    fragment_length: int
    subdomain_count: int
    subdomain_length: int
    path_depth: int
    url_entropy: float

    # Domain Features
    domain_length: int
    tld: str
    is_free_domain_provider: bool  # github.io, herokuapp.com, etc.

    # Obfuscation Features
    has_hex_encoding: bool
    hex_encoded_chars: int
    has_ip_address: bool
    has_at_symbol: bool
    has_double_slash_redirect: bool
    has_data_uri: bool
    obfuscation_score: float

    # Suspicious Patterns
    suspicious_path_count: int
    has_suspicious_path: bool
    has_login_keywords: bool
    has_brand_in_subdomain: bool
    suspicious_pattern_score: float

    # Character Features
    special_char_count: int
    digit_count: int
    digit_ratio: float
    uppercase_count: int
    has_unicode: bool
    punycode_detected: bool

    # DNS/WHOIS Features (optional, requires external lookup)
    domain_age_days: Optional[int] = None
    is_recent_domain: Optional[bool] = None
    registrar_suspicious: Optional[bool] = None
    dns_record_exists: Optional[bool] = None

    # Typosquatting Features
    typosquatting_score: float = 0.0
    potential_brand: Optional[str] = None
    levenshtein_distance: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary for ML pipeline."""
        return {k: v for k, v in self.__dict__.items()}

    def to_feature_array(self) -> List[float]:
        """Convert to array for model inference."""
        # Define feature order for consistent array conversion
        feature_order = [
            'url_length', 'path_length', 'query_length', 'fragment_length',
            'subdomain_count', 'subdomain_length', 'path_depth', 'url_entropy',
            'domain_length', 'is_free_domain_provider',
            'has_hex_encoding', 'hex_encoded_chars', 'has_ip_address',
            'has_at_symbol', 'has_double_slash_redirect', 'has_data_uri',
            'obfuscation_score', 'suspicious_path_count', 'has_suspicious_path',
            'has_login_keywords', 'has_brand_in_subdomain', 'suspicious_pattern_score',
            'special_char_count', 'digit_count', 'digit_ratio', 'uppercase_count',
            'has_unicode', 'punycode_detected', 'domain_age_days', 'is_recent_domain',
            'registrar_suspicious', 'dns_record_exists', 'typosquatting_score'
        ]
        return [self.__dict__.get(k, 0.0) if self.__dict__.get(k) is not None else 0.0
                for k in feature_order]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class TimeoutException(Exception):
    """Exception raised when an operation exceeds the timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Operation timed out")


def calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.
    High entropy indicates random characters (common in malicious URLs).

    Args:
        text: Input string to analyze

    Returns:
        Entropy value (0-8 for ASCII, higher for mixed character sets)
    """
    if not text:
        return 0.0

    # Count character frequencies
    counter = Counter(text.lower())
    total_chars = len(text)

    # Calculate Shannon entropy
    entropy = 0.0
    for count in counter.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)

    return entropy


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    Used for typosquatting detection.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of single-character edits
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Use a single row for space efficiency
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent processing.
    Handles various encoding schemes and protocols.

    Args:
        url: Raw URL string

    Returns:
        Normalized URL string
    """
    # Decode common encodings
    normalized = unquote(url)
    normalized = unquote(normalized)  # Double decode

    # Handle punycode
    if normalized.startswith('xn--'):
        try:
            # This is a punycode domain
            import idna
            normalized = idna.decode(normalized)
        except (ImportError, UnicodeError):
            pass

    # Normalize protocol
    if not normalized.startswith(('http://', 'https://', '//')):
        normalized = 'https://' + normalized
    elif normalized.startswith('//'):
        normalized = 'https:' + normalized

    return normalized.strip()


def is_punycode(url: str) -> bool:
    """
    Check if URL contains punycode encoding (internationalized domain abuse).

    Args:
        url: URL to check

    Returns:
        True if punycode detected
    """
    # Check for punycode in domain part
    punycode_pattern = re.compile(r'xn--[a-z0-9]+')

    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        return bool(punycode_pattern.search(domain))
    except Exception:
        return False


# ============================================================================
# FEATURE EXTRACTION CLASS
# ============================================================================

class URLFeatureExtractor:
    """
    Main feature extraction engine for PhishGuard.
    Extracts comprehensive features from URLs for ML classification.

    Features are organized into categories:
    - Lexical: Length, entropy, character distribution
    - Structural: Subdomains, path depth, query parameters
    - Obfuscation: Encoding, redirects, suspicious protocols
    - Behavioral: Suspicious patterns, brand impersonation
    - Threat Intelligence: DNS/WHOIS, typosquatting
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize the feature extractor.

        Args:
            timeout: Maximum seconds for external lookups (DNS, WHOIS)
        """
        self.timeout = timeout
        self.known_brands = KNOWN_BRANDS
        self.suspicious_patterns = [re.compile(p, re.I) for p in SUSPICIOUS_PATHS]
        self.obfuscation_patterns = OBFUSCATION_PATTERNS

        logger.info("URLFeatureExtractor initialized")

    def extract(self, url: str) -> URLFeatures:
        """
        Extract all features from a URL.

        Args:
            url: Target URL to analyze

        Returns:
            URLFeatures object containing all extracted features
        """
        try:
            # Normalize URL for consistent processing
            normalized = normalize_url(url)
            parsed = urlparse(normalized)

        except Exception as e:
            logger.warning(f"Failed to parse URL '{url}': {e}. Using raw URL.")
            normalized = url
            parsed = None

        # Extract each feature category
        length_features = self._extract_length_features(normalized, parsed)
        domain_features = self._extract_domain_features(parsed)
        obfuscation_features = self._extract_obfuscation_features(normalized, parsed)
        suspicious_features = self._extract_suspicious_patterns(normalized, parsed)
        character_features = self._extract_character_features(normalized)

        # Typosquatting analysis
        typosquatting_features = self._analyze_typosquatting(parsed)

        # Combine all features
        features = URLFeatures(
            # Length features
            **length_features,

            # Domain features
            **domain_features,

            # Obfuscation features
            **obfuscation_features,

            # Suspicious pattern features
            **suspicious_features,

            # Character features
            **character_features,

            # Typosquatting features
            **typosquatting_features,

            # DNS/WHOIS features (set via separate lookup method)
            domain_age_days=None,
            is_recent_domain=None,
            registrar_suspicious=None,
            dns_record_exists=None,
        )

        return features

    def extract_with_dns(self, url: str, whois_result: Optional[Dict] = None,
                        dns_result: Optional[Dict] = None) -> URLFeatures:
        """
        Extract features with optional DNS/WHOIS data for enhanced detection.

        Args:
            url: Target URL to analyze
            whois_result: Optional WHOIS lookup result
            dns_result: Optional DNS lookup result

        Returns:
            URLFeatures object with DNS/WHOIS features populated
        """
        features = self.extract(url)

        # Populate WHOIS features
        if whois_result:
            try:
                # Extract creation date
                creation_date = whois_result.get('creation_date')
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]

                if creation_date:
                    # Parse date if it's a string (ISO format from WHOIS)
                    parsed_date = creation_date
                    if isinstance(creation_date, str):
                        try:
                            # Handle ISO format with timezone
                            parsed_date = datetime.fromisoformat(creation_date.replace('Z', '+00:00'))
                            # Remove timezone for age calculation (convert to local)
                            if parsed_date.tzinfo is not None:
                                parsed_date = parsed_date.replace(tzinfo=None) - parsed_date.utcoffset()
                        except ValueError:
                            parsed_date = None

                    if parsed_date and isinstance(parsed_date, datetime):
                        age_days = (datetime.now() - parsed_date).days
                        features.domain_age_days = age_days
                        features.is_recent_domain = age_days < 90  # Recent = less than 90 days
                    else:
                        features.domain_age_days = -1
                        features.is_recent_domain = None
                else:
                    features.domain_age_days = -1  # Unknown
                    features.is_recent_domain = None

                # Check for suspicious registrar
                registrar = whois_result.get('registrar', '').lower()
                suspicious_words = ['privacy', 'redemption', 'additional', 'premium']
                features.registrar_suspicious = any(word in registrar for word in suspicious_words)

            except Exception as e:
                logger.warning(f"Failed to process WHOIS data: {e}")

        # Populate DNS features
        if dns_result:
            # Support both 'has_a_record' and 'dns_resolves' field names
            features.dns_record_exists = dns_result.get('has_a_record', dns_result.get('dns_resolves', False))

        return features

    def _extract_length_features(self, url: str, parsed) -> Dict[str, Any]:
        """
        Extract length-based features from URL.

        Long URLs with short domains often indicate obfuscation.
        """
        # Get raw components
        path = parsed.path if parsed else url
        query = parsed.query if parsed else ""
        fragment = parsed.fragment if parsed else ""

        # Calculate lengths
        url_length = len(url)
        path_length = len(path)
        query_length = len(query)
        fragment_length = len(fragment)

        # Subdomain analysis
        if parsed and parsed.netloc:
            domain_parts = parsed.netloc.split('.')
            subdomain_count = max(0, len(domain_parts) - 2)  # Exclude TLD and main domain
            subdomain_length = sum(len(p) for p in domain_parts[:-2]) if subdomain_count > 0 else 0
        else:
            subdomain_count = 0
            subdomain_length = 0

        # Path depth (number of / in path)
        path_depth = path.count('/') if path else 0

        # URL entropy (randomness indicator)
        url_entropy = calculate_entropy(url)

        return {
            'url_length': url_length,
            'path_length': path_length,
            'query_length': query_length,
            'fragment_length': fragment_length,
            'subdomain_count': subdomain_count,
            'subdomain_length': subdomain_length,
            'path_depth': path_depth,
            'url_entropy': url_entropy,
        }

    def _extract_domain_features(self, parsed) -> Dict[str, Any]:
        """
        Extract domain-related features.
        """
        if not parsed or not parsed.netloc:
            return {
                'domain_length': 0,
                'tld': '',
                'is_free_domain_provider': False,
            }

        netloc = parsed.netloc.lower()

        # Remove port if present
        if ':' in netloc:
            netloc = netloc.split(':')[0]

        # Extract domain parts
        parts = netloc.split('.')
        domain_length = len(netloc)

        # Get TLD (last part)
        tld = parts[-1] if parts else ''

        # Check for free domain providers (often used in phishing)
        free_providers = [
            'github.io', 'herokuapp.com', 'azurewebsites.net', 'appspot.com',
            'wordpress.com', 'blogspot.com', 'squarespace.com', 'wix.com',
            'weebly.com', 'glitch.me', 'surge.sh', 'netlify.app', 'vercel.app',
            'firebaseapp.com', 'web.app', 'cloudfunctions.net',
        ]
        is_free_provider = any(provider in netloc for provider in free_providers)

        return {
            'domain_length': domain_length,
            'tld': tld,
            'is_free_domain_provider': is_free_provider,
        }

    def _extract_obfuscation_features(self, url: str, parsed) -> Dict[str, Any]:
        """
        Extract obfuscation indicators.

        Common obfuscation techniques:
        - Hex encoding: %41 = 'A'
        - IP addresses: 192.168.1.1 instead of domain
        - @ symbol: http://google.com@evil.com
        - Data URIs: data:text/html,<script>alert(1)</script>
        """
        obfuscation_score = 0.0

        # Hex encoding detection
        hex_pattern = re.compile(r'%[0-9A-Fa-f]{2}')
        hex_matches = hex_pattern.findall(url)
        has_hex = len(hex_matches) > 0
        hex_char_count = len(hex_matches)

        if has_hex:
            obfuscation_score += min(1.0, hex_char_count / 10)
            try:
                # Verify decoding produces readable text
                decoded = unquote(url)
                # If decoding changes the URL significantly, it's suspicious
                if decoded != url:
                    obfuscation_score += 0.5
            except Exception:
                pass

        # IP address detection
        ip_pattern = re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b|'
            r'\[(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\]'
        )
        has_ip = bool(ip_pattern.search(url))
        if has_ip:
            obfuscation_score += 2.0  # High weight for IP addresses

        # @ symbol (navigation bypass)
        # In URLs, @ redirects to the domain after @, e.g., http://legit.com@evil.com
        # Modern browsers block this but some phishing sites still use it
        has_at = '@' in url
        if has_at:
            obfuscation_score += 1.5

        # Double slash redirect (// at start of path)
        has_double_slash = '//' in url[url.find('://') + 3:] if '://' in url else False
        if has_double_slash and url.startswith('http:') and '//' in url[5:10]:
            obfuscation_score += 0.5

        # Data URI detection
        has_data_uri = url.lower().startswith('data:')
        if has_data_uri:
            obfuscation_score += 2.0

        # JavaScript URI
        has_js = url.lower().startswith('javascript:')
        if has_js:
            obfuscation_score += 2.0

        return {
            'has_hex_encoding': has_hex,
            'hex_encoded_chars': hex_char_count,
            'has_ip_address': has_ip,
            'has_at_symbol': has_at,
            'has_double_slash_redirect': has_double_slash,
            'has_data_uri': has_data_uri or has_js,
            'obfuscation_score': min(obfuscation_score, 5.0),  # Cap at 5.0
        }

    def _extract_suspicious_patterns(self, url: str, parsed) -> Dict[str, Any]:
        """
        Extract suspicious path and pattern indicators.

        Phishing sites often use paths resembling legitimate auth pages.
        """
        url_lower = url.lower()
        suspicious_count = 0

        # Check each suspicious pattern
        for pattern in self.suspicious_patterns:
            if pattern.search(url_lower):
                suspicious_count += 1

        has_suspicious = suspicious_count > 0

        # Login-related keywords
        login_keywords = ['login', 'signin', 'sign-in', 'log-in', 'authenticate']
        has_login = any(kw in url_lower for kw in login_keywords)

        # Brand impersonation in subdomain
        has_brand_subdomain = self._check_brand_in_subdomain(parsed)

        # Calculate suspicious score based on findings
        suspicious_score = 0.0
        if suspicious_count > 0:
            suspicious_score += min(1.0, suspicious_count * 0.3)
        if has_login:
            suspicious_score += 0.5
        if has_brand_subdomain:
            suspicious_score += 1.0

        return {
            'suspicious_path_count': suspicious_count,
            'has_suspicious_path': has_suspicious,
            'has_login_keywords': has_login,
            'has_brand_in_subdomain': has_brand_subdomain,
            'suspicious_pattern_score': min(suspicious_score, 3.0),
        }

    def _extract_character_features(self, url: str) -> Dict[str, Any]:
        """
        Extract character-based statistics.

        Phishing URLs often have:
        - High digit ratios (numeric spoofing)
        - Special characters for obfuscation
        - Mixed case for brand impersonation
        """
        special_chars = set('!$%^&*()_+-=[]{}|;:,<>?/~`')
        special_count = sum(1 for c in url if c in special_chars)
        digit_count = sum(1 for c in url if c.isdigit())
        upper_count = sum(1 for c in url if c.isupper())

        digit_ratio = digit_count / len(url) if len(url) > 0 else 0.0

        # Unicode detection
        has_unicode = any(ord(c) > 127 for c in url)
        has_punycode = is_punycode(url)

        return {
            'special_char_count': special_count,
            'digit_count': digit_count,
            'digit_ratio': digit_ratio,
            'uppercase_count': upper_count,
            'has_unicode': has_unicode,
            'punycode_detected': has_punycode,
        }

    def _check_brand_in_subdomain(self, parsed) -> bool:
        """
        Check if a known brand appears in the subdomain.

        E.g., "secure-paypal.com.malicious-site.com" would trigger this.
        """
        if not parsed or not parsed.netloc:
            return False

        # Extract subdomain portion
        parts = parsed.netloc.lower().split('.')
        if len(parts) < 2:
            return False

        # Subdomain is everything except domain and TLD
        subdomain = parts[:-2]
        subdomain_text = '.'.join(subdomain)

        # Check against known brands
        for brand, legitimate_domains in self.known_brands.items():
            for legitimate in legitimate_domains:
                legitimate_base = legitimate.split('.')[0]
                if legitimate_base in subdomain_text:
                    # Found brand in subdomain, but we need to verify it's NOT
                    # the legitimate domain before flagging as suspicious
                    legitimate_domain = legitimate
                    current_domain = parsed.netloc.lower()
                    if legitimate_domain not in current_domain:
                        return True  # Brand impersonation detected

        return False

    def _analyze_typosquatting(self, parsed) -> Dict[str, Any]:
        """
        Analyze potential typosquatting.

        Compares the domain against known brands using Levenshtein distance.
        """
        if not parsed or not parsed.netloc:
            return {
                'typosquatting_score': 0.0,
                'potential_brand': None,
                'levenshtein_distance': None,
            }

        # Extract base domain (without TLD and subdomain)
        try:
            netloc = parsed.netloc.lower()
            parts = netloc.split('.')

            # Remove port if present
            if ':' in parts[-1] and parts[-1].replace(':', '').isdigit():
                parts = parts[:-1]

            if len(parts) >= 2:
                base_domain = parts[-2]  # e.g., "g00gle" from "g00gle.com"
            else:
                base_domain = netloc

        except Exception:
            base_domain = ""

        # Compare against known brands
        best_match = None
        best_distance = float('inf')
        best_brand = None

        for brand, legitimate_domains in self.known_brands.items():
            for legitimate in legitimate_domains:
                legitimate_base = legitimate.split('.')[0]
                distance = levenshtein_distance(base_domain.lower(), legitimate_base)

                if distance < best_distance:
                    best_distance = distance
                    best_match = legitimate_base
                    best_brand = brand

        # Calculate typosquatting score
        # Lower distance = more suspicious (assuming intent to mislead)
        # Distance of 1-2 = highly suspicious (likely typo)
        # Distance of 3-4 = moderate suspicion
        # Distance >= 5 = likely not typosquatting

        typosquatting_score = 0.0
        if best_distance <= 2:
            typosquatting_score = 2.0  # High confidence typo
        elif best_distance <= 4:
            typosquatting_score = 1.0  # Suspicious
        elif best_distance <= 6:
            typosquatting_score = 0.5  # Low suspicion

        return {
            'typosquatting_score': typosquatting_score,
            'potential_brand': best_brand,
            'levenshtein_distance': int(best_distance) if best_distance != float('inf') else None,
        }

    def batch_extract(self, urls: List[str]) -> List[URLFeatures]:
        """
        Extract features from multiple URLs.

        Args:
            urls: List of URLs to analyze

        Returns:
            List of URLFeatures objects
        """
        results = []
        for url in urls:
            try:
                features = self.extract(url)
                results.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features from '{url}': {e}")
                # Return minimal features for failed extractions
                results.append(self._get_default_features())

        return results

    def _get_default_features(self) -> URLFeatures:
        """Return default features for URL extraction failures."""
        return URLFeatures(
            url_length=0, path_length=0, query_length=0, fragment_length=0,
            subdomain_count=0, subdomain_length=0, path_depth=0, url_entropy=0.0,
            domain_length=0, tld='', is_free_domain_provider=False,
            has_hex_encoding=False, hex_encoded_chars=0, has_ip_address=False,
            has_at_symbol=False, has_double_slash_redirect=False, has_data_uri=False,
            obfuscation_score=0.0, suspicious_path_count=0, has_suspicious_path=False,
            has_login_keywords=False, has_brand_in_subdomain=False, suspicious_pattern_score=0.0,
            special_char_count=0, digit_count=0, digit_ratio=0.0, uppercase_count=0,
            has_unicode=False, punycode_detected=False
        )


# ============================================================================
# DNS/WHOIS LOOKUP INTEGRATION
# ============================================================================

def whois_lookup(domain: str, timeout: float = DEFAULT_TIMEOUT) -> Optional[Dict]:
    """
    Perform WHOIS lookup with timeout protection.

    Args:
        domain: Domain to lookup
        timeout: Maximum seconds to wait

    Returns:
        WHOIS data dictionary or None on failure
    """
    try:
        import whois

        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            w = whois.whois(domain)
            signal.alarm(0)  # Cancel alarm
            return w
        except TimeoutException:
            logger.warning(f"WHOIS lookup timed out for {domain}")
            return None
        except Exception as e:
            logger.warning(f"WHOIS lookup failed for {domain}: {e}")
            signal.alarm(0)
            return None

    except ImportError:
        logger.warning("python-whois not installed. Install with: pip install python-whois")
        return None
    except Exception as e:
        logger.warning(f"WHOIS lookup error: {e}")
        return None


def dns_lookup(domain: str, timeout: float = DEFAULT_TIMEOUT) -> Optional[Dict]:
    """
    Perform DNS lookup with timeout protection.

    Args:
        domain: Domain to lookup
        timeout: Maximum seconds to wait

    Returns:
        DNS data dictionary or None on failure
    """
    try:
        import dns.resolver

        resolver = dns.resolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout

        result = {
            'has_a_record': False,
            'has_mx_record': False,
            'has_txt_record': False,
            'a_records': [],
            'mx_records': [],
        }

        # Check A record
        try:
            answers = resolver.resolve(domain, 'A')
            result['has_a_record'] = True
            result['a_records'] = [rdata.address for rdata in answers]
        except dns.resolver.NoAnswer:
            pass
        except dns.resolver.NXDOMAIN:
            result['has_a_record'] = False
        except dns.exception.Timeout:
            logger.warning(f"DNS A record lookup timed out for {domain}")

        # Check MX record
        try:
            answers = resolver.resolve(domain, 'MX')
            result['has_mx_record'] = True
            result['mx_records'] = [str(rdata.exchange) for rdata in answers]
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            pass
        except dns.exception.Timeout:
            pass

        # Check TXT record
        try:
            answers = resolver.resolve(domain, 'TXT')
            result['has_txt_record'] = True
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            pass
        except dns.exception.Timeout:
            pass

        return result

    except ImportError:
        logger.warning("dnspython not installed. Install with: pip install dnspython")
        return None
    except Exception as e:
        logger.warning(f"DNS lookup error: {e}")
        return None


# ============================================================================
# MAIN ENTRY POINT FOR TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the feature extractor with sample URLs
    from datetime import datetime

    test_urls = [
        "https://www.google.com",
        "https://secure-paypal.com.malicious-site.net/login",
        "http://192.168.1.1/phishing.html",
        "https://g00gle.com.auth.verify.account.update@example.com/secure",
        "https://legit-amazon.com.de/session/login?redirect=https://amazon.com",
        "https://google.com.xn--80adxhks/",
        "data:text/html,<script>alert(1)</script>",
        "https://1234567890abcdef.suspicious.xyz/login?user=admin&pass=%41%42%43",
    ]

    extractor = URLFeatureExtractor()

    print("=" * 80)
    print("PhishGuard Feature Extraction Test")
    print("=" * 80)

    for url in test_urls:
        print(f"\n[URL] {url}")
        print("-" * 60)

        try:
            features = extractor.extract(url)

            print(f"Length Features:")
            print(f"  URL Length: {features.url_length}")
            print(f"  Path Depth: {features.path_depth}")
            print(f"  Entropy: {features.url_entropy:.4f}")

            print(f"Obfuscation:")
            print(f"  Has Hex: {features.has_hex_encoding} ({features.hex_encoded_chars} chars)")
            print(f"  Has IP: {features.has_ip_address}")
            print(f"  Has @: {features.has_at_symbol}")
            print(f"  Score: {features.obfuscation_score:.2f}")

            print(f"Suspicious Patterns:")
            print(f"  Count: {features.suspicious_path_count}")
            print(f"  Has Login: {features.has_login_keywords}")
            print(f"  Score: {features.suspicious_pattern_score:.2f}")

            print(f"Typosquatting:")
            print(f"  Score: {features.typosquatting_score:.2f}")
            print(f"  Potential Brand: {features.potential_brand}")
            print(f"  Edit Distance: {features.levenshtein_distance}")

            print(f"\n[TOTAL RISK SIGNALS]")
            total = features.obfuscation_score + features.suspicious_pattern_score + features.typosquatting_score
            print(f"  Composite Score: {total:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
    print("Feature extraction test complete.")
    print("=" * 80)