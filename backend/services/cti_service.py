"""
PhishGuard CTI Service
=======================
Cyber Threat Intelligence integration for URL analysis.
Supports VirusTotal, URLhaus, and other threat feeds.

All external calls use configurable timeouts to ensure <500ms target.
"""

import os
import sys
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from urllib.parse import urlparse
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 2.0  # seconds
MAX_RETRIES = 2


@dataclass
class CTILookupResult:
    """Result of a CTI lookup."""
    source: str
    found: bool = False
    malicious: bool = False
    positives: int = 0
    total: int = 0
    detection_rate: float = 0.0
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    response_time_ms: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CTIService:
    """
    Service for querying external threat intelligence sources.

    Currently supported:
    - VirusTotal (requires API key)
    - URLhaus (public API)

    All methods are async for non-blocking execution.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize CTI service.

        Args:
            timeout: Request timeout in seconds for CTI lookups
        """
        self.timeout = timeout

        # VirusTotal configuration
        self.virustotal_api_key = os.getenv("VIRUSTOTAL_API_KEY", "").strip() or None
        self.virustotal_enabled = bool(self.virustotal_api_key)

        # URLhaus (public, no key required)
        self.urlhaus_enabled = True
        self.urlhaus_base_url = "https://urlhaus.abuse.ch"

        # Track API usage
        self._vt_requests_today = 0
        self._vt_daily_limit = 500  # Free tier limit

        logger.info(f"CTIService initialized. VirusTotal: {'enabled' if self.virustotal_enabled else 'disabled'}")

    @property
    def can_query_virustotal(self) -> bool:
        """Check if VirusTotal queries are allowed."""
        if not self.virustotal_enabled:
            return False
        if self._vt_requests_today >= self._vt_daily_limit:
            logger.warning("VirusTotal daily limit reached")
            return False
        return True

    def _get_domain_from_url(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove port
            if ':' in domain:
                domain = domain.split(':')[0]
            return domain
        except Exception:
            return url  # Return as-is if parsing fails

    def _get_hash(self, url: str) -> str:
        """Get MD5 hash of URL for certain lookups."""
        return hashlib.md5(url.encode()).hexdigest()

    # =========================================================================
    # VIRUSTOTAL LOOKUP
    # =========================================================================

    async def lookup_virustotal(self, url: str) -> CTILookupResult:
        """
        Look up URL in VirusTotal.

        Requires VIRUSTOTAL_API_KEY environment variable.

        Args:
            url: URL to check

        Returns:
            CTILookupResult with detection data
        """
        start_time = time.time()

        if not self.can_query_virustotal:
            return CTILookupResult(
                source="virustotal",
                found=False,
                malicious=False,
                error="VirusTotal API key not configured or daily limit reached",
                metadata={"rate_limited": True}
            )

        api_url = "https://www.virustotal.com/api/v3/urls"
        headers = {
            "x-apikey": self.virustotal_api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        try:
            # First, submit the URL for analysis (or get cached result)
            # Use URL ID (base64 encoded url_id)
            # For free tier, we can also use the /domains endpoint

            async with aiohttp.ClientSession() as session:
                # Get URL ID
                import base64
                url_id = base64.urlsafe_b64encode(url.encode()).decode().strip().rstrip('=')

                # Query the URL endpoint
                endpoint = f"{api_url}/{url_id}"

                async with session.get(endpoint, headers=headers, timeout=self.timeout) as resp:
                    response_time = (time.time() - start_time) * 1000

                    if resp.status == 404:
                        # URL not found in VT database
                        self._vt_requests_today += 1
                        return CTILookupResult(
                            source="virustotal",
                            found=False,
                            malicious=False,
                            response_time_ms=response_time
                        )

                    if resp.status == 429:
                        self._vt_requests_today = self._vt_daily_limit
                        return CTILookupResult(
                            source="virustotal",
                            found=False,
                            error="Rate limited",
                            response_time_ms=response_time,
                            metadata={"rate_limited": True}
                        )

                    if resp.status != 200:
                        self._vt_requests_today += 1
                        return CTILookupResult(
                            source="virustotal",
                            found=False,
                            error=f"HTTP {resp.status}",
                            response_time_ms=response_time
                        )

                    data = await resp.json()
                    self._vt_requests_today += 1

                    # Parse response
                    attributes = data.get("data", {}).get("attributes", {})
                    stats = attributes.get("last_analysis_stats", {})
                    results = attributes.get("last_analysis_results", {})

                    malicious = stats.get("malicious", 0)
                    suspicious = stats.get("suspicious", 0)
                    totals = stats.get("total", 0)
                    undetected = stats.get("undetected", 0)

                    total_vendors = malicious + suspicious + undetected
                    detection_rate = (malicious + suspicious) / total_vendors if total_vendors > 0 else 0.0

                    # Get vendor detection names
                    malicious_vendors = [
                        name for name, result in results.items()
                        if result.get("category") in ["malicious", "suspicious"]
                    ]

                    return CTILookupResult(
                        source="virustotal",
                        found=True,
                        malicious=malicious > 0,
                        positives=malicious,
                        total=total_vendors,
                        detection_rate=detection_rate,
                        metadata={
                            "suspicious_vendors": suspicious,
                            "malicious_vendors": malicious_vendors,
                            "undetected": undetected,
                            "reputation": attributes.get("reputation", 0),
                            "tags": attributes.get("tags", []),
                        },
                        response_time_ms=response_time
                    )

        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            logger.warning(f"VirusTotal lookup timed out for {url[:50]}...")
            return CTILookupResult(
                source="virustotal",
                found=False,
                error="Timeout",
                response_time_ms=response_time
            )
        except aiohttp.ClientError as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"VirusTotal lookup failed: {e}")
            return CTILookupResult(
                source="virustotal",
                found=False,
                error=str(e),
                response_time_ms=response_time
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"VirusTotal unexpected error: {e}")
            return CTILookupResult(
                source="virustotal",
                found=False,
                error=str(e),
                response_time_ms=response_time
            )

    # =========================================================================
    # URLHAUS LOOKUP
    # =========================================================================

    async def lookup_urlhaus(self, url: str) -> CTILookupResult:
        """
        Look up URL in URLhaus (abuse.ch).

        URLhaus provides a free, public API for malware URLs.
        No API key required.

        Args:
            url: URL to check

        Returns:
            CTILookupResult with detection data
        """
        start_time = time.time()

        try:
            # URLhaus POST endpoint
            api_url = f"{self.urlhaus_base_url}/api/endpoint.php"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    data={"url": url},
                    timeout=self.timeout
                ) as resp:

                    response_time = (time.time() - start_time) * 1000

                    if resp.status != 200:
                        return CTILookupResult(
                            source="urlhaus",
                            found=False,
                            error=f"HTTP {resp.status}",
                            response_time_ms=response_time
                        )

                    data = await resp.json()

                    # Parse URLhaus response
                    # Reference: https://urlhaus.abuse.ch/api/
                    if data.get("query_status") == "ok":
                        urlhaus_info = data.get("urlhaus_reference", {})
                        threat_variant = urlhaus_info.get("payloads", [{}])[0].get("variant", "unknown")
                        threat_type = data.get("threat", "unknown").lower()
                        status = data.get("url_status", "unknown")

                        is_malicious = status in ["malicious", "phishing"]

                        return CTILookupResult(
                            source="urlhaus",
                            found=True,
                            malicious=is_malicious,
                            positives=1 if is_malicious else 0,
                            total=1,
                            detection_rate=1.0 if is_malicious else 0.0,
                            metadata={
                                "threat_type": threat_type,
                                "status": status,
                                "date_added": urlhaus_info.get("date_added"),
                                "tags": data.get("tags", []),
                                "payloads": data.get("payloads", []),
                            },
                            response_time_ms=response_time
                        )

                    elif data.get("query_status") == "no_results":
                        return CTILookupResult(
                            source="urlhaus",
                            found=False,
                            malicious=False,
                            response_time_ms=response_time
                        )

                    else:
                        return CTILookupResult(
                            source="urlhaus",
                            found=False,
                            error=f"Query status: {data.get('query_status', 'unknown')}",
                            response_time_ms=response_time
                        )

        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            logger.warning(f"URLhaus lookup timed out for {url[:50]}...")
            return CTILookupResult(
                source="urlhaus",
                found=False,
                error="Timeout",
                response_time_ms=response_time
            )
        except aiohttp.ClientError as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"URLhaus lookup failed: {e}")
            return CTILookupResult(
                source="urlhaus",
                found=False,
                error=str(e),
                response_time_ms=response_time
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"URLhaus unexpected error: {e}")
            return CTILookupResult(
                source="urlhaus",
                found=False,
                error=str(e),
                response_time_ms=response_time
            )

    # =========================================================================
    # BATCH LOOKUP
    # =========================================================================

    async def lookup_all(self, url: str) -> List[CTILookupResult]:
        """
        Query all enabled CTI sources for a URL.

        Runs lookups concurrently and returns all results.

        Args:
            url: URL to check

        Returns:
            List of CTILookupResults from each source
        """
        tasks = []

        if self.virustotal_enabled:
            tasks.append(self.lookup_virustotal(url))

        if self.urlhaus_enabled:
            tasks.append(self.lookup_urlhaus(url))

        if not tasks:
            return []

        # Run all lookups concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                source_name = "unknown"
                if i == 0 and self.virustotal_enabled:
                    source_name = "virustotal"
                elif i == 1 and self.urlhaus_enabled:
                    source_name = "urlhaus"

                processed_results.append(CTILookupResult(
                    source=source_name,
                    found=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    # =========================================================================
    # DNS/WHOIS LOOKUP (BASIC)
    # =========================================================================

    async def basic_dns_check(self, url: str) -> Dict[str, Any]:
        """
        Perform basic DNS connectivity check.

        Checks if the domain resolves to an IP.
        This is a basic check - not a full WHOIS lookup.

        Args:
            url: URL to check

        Returns:
            Dictionary with DNS check results
        """
        start_time = time.time()
        domain = self._get_domain_from_url(url)

        result = {
            "domain": domain,
            "resolves": False,
            "ip_addresses": [],
            "error": None,
            "response_time_ms": 0.0
        }

        try:
            import socket

            # Use asyncio to not block
            loop = asyncio.get_event_loop()
            ip = await loop.run_in_executor(
                None,
                lambda: socket.gethostbyname(domain) if domain else None
            )

            response_time = (time.time() - start_time) * 1000
            result["resolves"] = True
            result["ip_addresses"] = [ip]
            result["response_time_ms"] = response_time

        except socket.gaierror:
            result["error"] = "Domain does not resolve"
        except Exception as e:
            result["error"] = str(e)

        return result


# Global singleton
_cti_service: Optional[CTIService] = None


def get_cti_service() -> CTIService:
    """Get or create global CTI service instance."""
    global _cti_service
    if _cti_service is None:
        _cti_service = CTIService()
    return _cti_service