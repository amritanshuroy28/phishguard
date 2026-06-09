"""
PhishGuard Domain Intelligence Service
=======================================
Async DNS and WHOIS lookups for phishing detection.

Provides real-time domain intelligence:
- DNS resolution checks (A, MX, TXT records)
- WHOIS registration data (age, registrar, suspension)
- Domain age as phishing indicator (new domains = more suspicious)

Usage:
    from services.domain_service import get_domain_service
    service = get_domain_service()
    results = await service.lookup_all("https://example.com/login")
"""

import os
import sys
import time
import socket
import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Union
from urllib.parse import urlparse
from datetime import datetime

from dateutil import parser as date_parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timeout for DNS/WHOIS lookups
DEFAULT_TIMEOUT = float(os.getenv("DNS_WHOIS_TIMEOUT", "6.0"))
DNS_TIMEOUT = float(os.getenv("DNS_TIMEOUT", "2.0"))

# Suspicious registrar keywords
SUSPICIOUS_REGISTRAR_WORDS = [
    "privacy", "redemption", "additional", "premium", "proxy",
    "anonymous", "mask", "guard", "protect", "private registration"
]

# Suspicious TLDs often used for phishing
SUSPICIOUS_TLDS = [
    "xyz", "info", "top", "click", "link", "work", "tk", "ml", "ga", "cf", "gq"
]


@dataclass
class DomainLookupResult:
    """
    Result of a DNS or WHOIS lookup.

    Attributes:
        source: "dns" or "whois"
        domain: The domain that was looked up
        found: Whether the lookup was successful
        dns_resolves: Whether DNS resolves (DNS lookups only)
        ip_addresses: Resolved IP addresses (DNS lookups only)
        has_mx: Whether MX records exist (DNS lookups only)
        has_txt: Whether TXT records exist (DNS lookups only)
        registrar: Domain registrar (WHOIS lookups only)
        creation_date: Domain creation date (WHOIS lookups only)
        domain_age_days: Age of domain in days (WHOIS lookups only)
        is_recent_domain: Whether domain is < 90 days old
        registrar_suspicious: Whether registrar suggests privacy/proxy
        dns_error: Error message if DNS lookup failed
        whois_error: Error message if WHOIS lookup failed
        response_time_ms: Time taken for the lookup
    """
    source: str
    domain: str
    found: bool = False
    dns_resolves: bool = False
    ip_addresses: List[str] = None
    has_mx: bool = False
    has_txt: bool = False
    registrar: Optional[str] = None
    creation_date: Optional[str] = None
    domain_age_days: Optional[int] = None
    is_recent_domain: bool = False
    registrar_suspicious: bool = False
    dns_error: Optional[str] = None
    whois_error: Optional[str] = None
    response_time_ms: float = 0.0

    def __post_init__(self):
        if self.ip_addresses is None:
            self.ip_addresses = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for feature extraction."""
        return {
            'found': self.found,
            'dns_resolves': self.dns_resolves,
            'ip_addresses': self.ip_addresses,
            'has_mx': self.has_mx,
            'has_txt': self.has_txt,
            'registrar': self.registrar,
            'creation_date': self.creation_date,
            'domain_age_days': self.domain_age_days,
            'is_recent_domain': self.is_recent_domain,
            'registrar_suspicious': self.registrar_suspicious,
        }


class DomainIntelService:
    """
    Async domain intelligence service for DNS and WHOIS lookups.

    Uses python-whois for WHOIS queries and socket/dnspython for DNS.
    All lookups run concurrently and have configurable timeouts.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize the domain intelligence service.

        Args:
            timeout: Maximum time to wait for each lookup (seconds)
        """
        self.timeout = timeout
        self.whois_enabled = True
        self.dns_enabled = True
        self._check_dependencies()

    def _check_dependencies(self):
        """Check which libraries are available."""
        try:
            import whois
            self.whois_enabled = True
        except ImportError:
            logger.warning("python-whois not available, WHOIS lookups disabled")
            self.whois_enabled = False

        # DNS: we use socket.getaddrinfo which is built-in
        self.dns_enabled = True

    @staticmethod
    def _extract_domain(url: str) -> str:
        """
        Extract domain from URL.

        Args:
            url: Full URL to parse

        Returns:
            Domain string (e.g., "example.com")
        """
        try:
            # Handle URLs without scheme
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            parsed = urlparse(url.lower())

            # Get netloc (domain:port)
            netloc = parsed.netloc

            # Remove port if present
            if ':' in netloc:
                netloc = netloc.split(':')[0]

            # Remove userinfo if present (e.g., user:pass@domain.com)
            if '@' in netloc:
                netloc = netloc.split('@')[1]

            return netloc
        except Exception as e:
            logger.warning(f"Failed to extract domain from {url}: {e}")
            return url.lower()

    async def lookup_all(self, url: str) -> List[DomainLookupResult]:
        """
        Run both DNS and WHOIS lookups concurrently.

        Args:
            url: URL to analyze

        Returns:
            List of DomainLookupResult objects (one per lookup type)
        """
        domain = self._extract_domain(url)
        if not domain:
            return []

        # Run both lookups concurrently
        tasks = []

        if self.dns_enabled:
            tasks.append(self.dns_lookup(domain))

        if self.whois_enabled:
            tasks.append(self.whois_lookup(domain))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        return [r for r in results if not isinstance(r, Exception)]

    async def dns_lookup(self, domain: str) -> DomainLookupResult:
        """
        Perform async DNS lookup for a domain.

        Checks:
        - DNS resolution (does domain resolve to an IP?)
        - A records (primary web address)
        - MX records (mail servers)
        - TXT records (SPF, domain verification)

        Args:
            domain: Domain to check

        Returns:
            DomainLookupResult with DNS data
        """
        start_time = time.time()
        result = DomainLookupResult(source="dns", domain=domain)

        try:
            loop = asyncio.get_event_loop()

            # Use socket.getaddrinfo for basic DNS resolution check
            try:
                addrs = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: socket.getaddrinfo(domain, 443)
                    ),
                    timeout=self.timeout
                )

                result.found = True
                result.dns_resolves = True

                # Extract unique IP addresses
                ips = set(addr[4][0] for addr in addrs if addr[4])
                result.ip_addresses = list(ips)

                logger.info(f"DNS resolved {domain} -> {result.ip_addresses[:3]}")

            except asyncio.TimeoutError:
                result.dns_error = "DNS lookup timed out"
                logger.warning(f"DNS timeout for {domain}")

            except socket.gaierror as e:
                result.dns_error = f"DNS resolution failed: {e}"
                logger.warning(f"DNS resolution failed for {domain}: {e}")

            # Try to get additional DNS records via dnspython if available
            try:
                import dns.resolver

                # MX record check
                try:
                    mx_answer = await loop.run_in_executor(
                        None,
                        lambda: dns.resolver.resolve(domain, 'MX', lifetime=self.timeout)
                    )
                    result.has_mx = len(mx_answer) > 0
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                    result.has_mx = False

                # TXT record check
                try:
                    txt_answer = await loop.run_in_executor(
                        None,
                        lambda: dns.resolver.resolve(domain, 'TXT', lifetime=self.timeout)
                    )
                    result.has_txt = len(txt_answer) > 0
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                    result.has_txt = False

            except ImportError:
                # dnspython not available, skip MX/TXT checks
                logger.debug("dnspython not available, MX/TXT checks skipped")
            except Exception as e:
                logger.debug(f"MX/TXT lookup failed: {e}")

        except Exception as e:
            result.dns_error = str(e)
            logger.error(f"DNS lookup error for {domain}: {e}")

        result.response_time_ms = (time.time() - start_time) * 1000
        return result

    async def whois_lookup(self, domain: str) -> DomainLookupResult:
        """
        Perform async WHOIS lookup for a domain.

        Extracts:
        - Creation date (for domain age calculation)
        - Registrar (for suspicious registrar detection)
        - Domain suspension status

        Args:
            domain: Domain to check

        Returns:
            DomainLookupResult with WHOIS data
        """
        start_time = time.time()
        result = DomainLookupResult(source="whois", domain=domain)

        if not self.whois_enabled:
            result.whois_error = "WHOIS disabled (python-whois not installed)"
            return result

        try:
            import whois
            loop = asyncio.get_event_loop()

            # Run WHOIS query in executor to not block
            w = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: whois.whois(domain)
                ),
                timeout=self.timeout
            )

            if not w:
                result.whois_error = "No WHOIS data found"
                return result

            result.found = True

            # Extract creation date
            creation_date = w.get('creation_date')
            if creation_date:
                # Handle list of dates (sometimes WHOIS returns multiple)
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]

                # Parse date if it's a string
                parsed_date = None
                if isinstance(creation_date, datetime):
                    parsed_date = creation_date
                elif isinstance(creation_date, str):
                    try:
                        parsed_date = date_parser.parse(creation_date)
                    except Exception:
                        pass

                if parsed_date:
                    result.creation_date = parsed_date.isoformat()

                    # Calculate domain age (handle timezone-aware dates)
                    now = datetime.now()
                    if parsed_date.tzinfo is not None:
                        # Convert to local timezone for comparison
                        parsed_date_local = parsed_date.replace(tzinfo=None) - parsed_date.utcoffset()
                        now_local = now
                    else:
                        parsed_date_local = parsed_date
                        now_local = now

                    age_days = (now_local - parsed_date_local).days
                    result.domain_age_days = age_days

                    # Recent domain = suspicious (less than 90 days old)
                    result.is_recent_domain = age_days < 90

                    logger.info(f"WHOIS: {domain} age={age_days} days, recent={result.is_recent_domain}")

            # Extract registrar
            registrar = w.get('registrar') or w.get('registrar_name')
            if registrar:
                result.registrar = registrar.lower()

                # Check for suspicious registrar
                registrar_lower = registrar.lower()
                if any(word in registrar_lower for word in SUSPICIOUS_REGISTRAR_WORDS):
                    result.registrar_suspicious = True
                    logger.info(f"Suspicious registrar for {domain}: {registrar}")

            # Check for domain status indicating suspension/cancellation
            domain_status = w.get('status') or []
            if isinstance(domain_status, str):
                domain_status = [domain_status]

            suspicious_statuses = ['clienthold', 'inactive', 'pending delete',
                                   'redemption', 'client renew prohibited']
            for status in domain_status:
                if status and any(s in status.lower() for s in suspicious_statuses):
                    logger.warning(f"Domain {domain} has suspicious status: {status}")

        except asyncio.TimeoutError:
            result.whois_error = "WHOIS lookup timed out"
            logger.warning(f"WHOIS timeout for {domain}")

        except Exception as e:
            result.whois_error = str(e)
            logger.error(f"WHOIS lookup error for {domain}: {e}")

        result.response_time_ms = (time.time() - start_time) * 1000
        return result

    def is_suspicious_tld(self, domain: str) -> bool:
        """
        Check if domain uses a suspicious TLD.

        Some TLDs are more commonly used for phishing due to low cost.

        Args:
            domain: Domain to check

        Returns:
            True if TLD is suspicious
        """
        parts = domain.lower().split('.')
        if len(parts) < 2:
            return False

        tld = parts[-1]
        return tld in SUSPICIOUS_TLDS


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_domain_service: Optional[DomainIntelService] = None


def get_domain_service() -> DomainIntelService:
    """
    Get or create the global DomainIntelService instance.

    Returns:
        DomainIntelService singleton
    """
    global _domain_service
    if _domain_service is None:
        _domain_service = DomainIntelService()
    return _domain_service


def reload_domain_service(timeout: Optional[float] = None) -> DomainIntelService:
    """
    Reload the domain intelligence service (e.g., after config change).

    Args:
        timeout: Optional new timeout value

    Returns:
        New DomainIntelService instance
    """
    global _domain_service
    _domain_service = DomainIntelService(timeout=timeout or DEFAULT_TIMEOUT)
    return _domain_service