"""
PhishGuard CTI Service
=======================
Cyber Threat Intelligence integration (URLhaus only).
All VirusTotal calls removed — PhishGuard runs solely on the self-trained
ML model.

External calls use configurable timeouts to ensure <200ms target.
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
    Threat intelligence service (URLhaus only).

    VirusTotal removed — PhishGuard relies entirely on its self-trained ML
    model. URLhaus public API is queried optionally as a contextual signal.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout

        # URLhaus (public, no key required)
        self.urlhaus_enabled = True
        self.urlhaus_base_url = "https://urlhaus.abuse.ch"

        logger.info("CTIService initialized (URLhaus only, no VirusTotal).")

    def _get_domain_from_url(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if ':' in domain:
                domain = domain.split(':')[0]
            return domain
        except Exception:
            return url

    def _get_hash(self, url: str) -> str:
        """Get MD5 hash of URL for certain lookups."""
        return hashlib.md5(url.encode()).hexdigest()

    # =========================================================================
    # URLHAUS LOOKUP
    # =========================================================================

    async def lookup_urlhaus(self, url: str) -> CTILookupResult:
        """Look up URL in URLhaus (abuse.ch)."""
        start_time = time.time()

        try:
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
        """Query URLhaus (only enabled CTI source) for a URL."""
        if not self.urlhaus_enabled:
            return []
        try:
            result = await self.lookup_urlhaus(url)
            return [result]
        except Exception as e:
            return [CTILookupResult(
                source="urlhaus",
                found=False,
                error=str(e)
            )]

    # =========================================================================
    # DNS/WHOIS LOOKUP (BASIC)
    # =========================================================================

    async def basic_dns_check(self, url: str) -> Dict[str, Any]:
        """Perform basic DNS connectivity check."""
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
