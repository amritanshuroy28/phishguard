"""
PhishGuard Dataset Loader Module
=================================
Handles loading and preprocessing datasets from multiple sources:
- PhishTank (verified phishing URLs)
- URLhaus (live malicious URLs)
- Tranco (legitimate domains)
- ISCX URL 2016 (benchmark dataset)

This module provides:
- Dataset downloading and caching
- Label generation from various formats
- Train/test splitting with stratification
- Feature extraction pipeline integration
"""

import os
import sys
import json
import gzip
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import requests
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_extraction import URLFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Dataset URLs (public sources)
DATASET_SOURCES = {
    "phishtank": {
        "name": "PhishTank",
        "type": "phishing",
        "url": "https://data.phishtank.com/data/online-valid.csv",
        "requires_api_key": True,
        "fallback": "https://raw.githubusercontent.com/mitchellkrogza/PhishTank-Database/master/active_phishing_urls.txt",
    },
    "urlhaus": {
        "name": "URLhaus",
        "type": "phishing",
        "url": "https://urlhaus.abuse.ch/downloads/csv/",
        "requires_api_key": False,
    },
    "tranco": {
        "name": "Tranco Top 1M",
        "type": "legitimate",
        "url": "https://tranco-list.eu/download",
        "requires_api_key": False,
    },
}

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    source_type: str  # "phishing" or "legitimate"
    local_path: Optional[str] = None
    download_url: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None  # Map file columns to URL field
    label: int = 0  # 1 for phishing, 0 for legitimate

    # Filtering options
    min_url_length: int = 0
    max_url_length: int = 2048
    require_https: bool = False

    def __post_init__(self):
        if self.source_type == "phishing":
            self.label = 1
        elif self.source_type == "legitimate":
            self.label = 0


@dataclass
class LoadedDataset:
    """Container for loaded dataset information."""
    name: str
    urls: List[str]
    labels: List[int]
    metadata: Dict

    @property
    def size(self) -> int:
        return len(self.urls)

    @property
    def phishing_count(self) -> int:
        return sum(self.labels)

    @property
    def legitimate_count(self) -> int:
        return len(self.labels) - self.phishing_count


# ============================================================================
# DATASET LOADERS
# ============================================================================

class DatasetLoader:
    """
    Unified interface for loading and preprocessing multiple URL datasets.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, timeout: int = 30):
        """
        Initialize the dataset loader.

        Args:
            cache_dir: Directory for caching downloaded datasets
            timeout: HTTP request timeout in seconds
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.extractor = URLFeatureExtractor()

        logger.info(f"DatasetLoader initialized with cache: {cache_dir}")

    def _get_cache_path(self, source_name: str, suffix: str = "csv") -> Path:
        """Get local cache path for a dataset."""
        cache_file = self.cache_dir / f"{source_name}_{datetime.now().strftime('%Y%m%d')}.{suffix}"
        return cache_file

    def _download_file(self, url: str, cache_path: Path, headers: Optional[Dict] = None) -> bool:
        """
        Download a file with caching support.

        Args:
            url: URL to download
            cache_path: Local path to save
            headers: Optional HTTP headers

        Returns:
            True if successful, False otherwise
        """
        try:
            if cache_path.exists():
                logger.info(f"Using cached file: {cache_path}")
                return True

            logger.info(f"Downloading from: {url}")
            response = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
            response.raise_for_status()

            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded {cache_path.stat().st_size / 1024 / 1024:.2f} MB")
            return True

        except requests.RequestException as e:
            logger.error(f"Download failed: {e}")
            return False

    def _parse_csv(self, file_path: Path, url_column: str = "url",
                   delimiter: str = ",", encoding: str = "utf-8") -> List[str]:
        """
        Parse CSV file and extract URLs.

        Args:
            file_path: Path to CSV file
            url_column: Name of the column containing URLs
            delimiter: CSV delimiter
            encoding: File encoding

        Returns:
            List of URLs
        """
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding,
                           on_bad_lines='skip', low_memory=False)

            if url_column not in df.columns:
                # Try common column names
                for col in ['url', 'urls', 'URL', 'URLS', 'domain', 'Domains']:
                    if col in df.columns:
                        url_column = col
                        break

            urls = df[url_column].dropna().astype(str).tolist()
            logger.info(f"Loaded {len(urls)} URLs from {file_path.name}")
            return urls

        except Exception as e:
            logger.error(f"CSV parsing error: {e}")
            return []

    def _parse_csv_from_memory(self, content: str, url_column: str = "url",
                              delimiter: str = ",") -> List[str]:
        """
        Parse CSV from string content.

        Args:
            content: CSV content as string
            url_column: Column name for URLs
            delimiter: CSV delimiter

        Returns:
            List of URLs
        """
        import io
        try:
            df = pd.read_csv(io.StringIO(content), delimiter=delimiter,
                           on_bad_lines='skip', low_memory=False)

            # Find URL column
            if url_column not in df.columns:
                for col in df.columns:
                    if 'url' in col.lower() or col in ['domain', 'link']:
                        url_column = col
                        break

            if url_column in df.columns:
                urls = df[url_column].dropna().astype(str).tolist()
                return urls

            return []

        except Exception as e:
            logger.error(f"CSV parsing error: {e}")
            return []

    def load_phishtank_fallback(self) -> List[str]:
        """
        Load phishing URLs from PhishTank fallback (public list).
        """
        cache_path = self._get_cache_path("phishtank", "txt")

        try:
            # Try direct download
            url = "https://raw.githubusercontent.com/mitchellkrogza/PhishTank-Database/master/active_phishing_urls.txt"

            if not cache_path.exists():
                # Alternative: Use PhishTank's CSV export
                csv_cache = self._get_cache_path("phishtank", "csv")
                csv_url = "https://data.phishtank.com/data/online-valid.csv"

                # For CSV export, you need an API key. Use fallback.
                pass

            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    urls = [line.strip() for line in f if line.strip()]
                return urls

            # If no cache, generate sample data for demo purposes
            # In production, obtain API key from phishtank.org
            logger.warning("PhishTank API key required for full dataset. Using sample.")
            return self._generate_sample_phishing_urls()

        except Exception as e:
            logger.error(f"PhishTank loading error: {e}")
            return self._generate_sample_phishing_urls()

    def load_urlhaus(self) -> List[str]:
        """
        Load malicious URLs from URLhaus (public, no API key).
        """
        cache_path = self._get_cache_path("urlhaus", "csv")
        url = "https://urlhaus.abuse.ch/downloads/csv/"

        try:
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    content = f.read()
            else:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                content = response.text

                with open(cache_path, 'w') as f:
                    f.write(content)

            # URLhaus CSV format: id,dateadded,url,url_status,threat,last_online,provider,note
            urls = []
            for line in content.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    urls.append(parts[2])  # URL is the 3rd column

            logger.info(f"Loaded {len(urls)} URLs from URLhaus")
            return urls

        except Exception as e:
            logger.error(f"URLhaus loading error: {e}")
            return self._generate_sample_phishing_urls()

    def load_tranco(self) -> List[str]:
        """
        Load legitimate domains from Tranco list (top 1M).
        """
        cache_path = self._get_cache_path("tranco", "txt")

        try:
            if not cache_path.exists():
                # Tranco requires registration, use alternative legitimate sources
                logger.warning("Using alternative legitimate domains (Tranco requires registration)")
                return self._generate_sample_legitimate_urls()

            with open(cache_path, 'r') as f:
                urls = []
                for i, line in enumerate(f):
                    if i >= 100000:  # Limit to 100k for balance
                        break
                    domain = line.strip().split(',')[-1] if ',' in line else line.strip()
                    if domain and '.' in domain:
                        urls.append(f"https://{domain}")

            logger.info(f"Loaded {len(urls)} URLs from Tranco cache")
            return urls

        except Exception as e:
            logger.error(f"Tranco loading error: {e}")
            return self._generate_sample_legitimate_urls()

    def _generate_sample_phishing_urls(self) -> List[str]:
        """
        Generate sample phishing URLs for testing when datasets unavailable.

        Returns realistic-looking phishing URL patterns.
        """
        patterns = [
            # PayPal phishing
            "https://{}-paypal-secure.com/login?verify=account",
            "https://{}.paypal.com.verify-account.{}/login",
            "http://paypal-verify.{}.net/session/login",

            # Apple phishing
            "https://{}-appleid.secure-login.{}.com/verify",
            "http://apple.icloud-phishing.{}.xyz/reset-password",

            # Microsoft/Outlook phishing
            "https://{}.microsoft.online.{}.net/signin",
            "http://outlook.logn.verify.{}.com/auth",

            # Amazon phishing
            "https://{}-amazon.{}.com/your-account/verify",
            "http://amazon-accounts.{}.net/update-payment",

            # Bank phishing
            "https://{}.chase-login.{}.com/onlineservices",
            "http://bankofamerica.{}.net/verify-account",

            # Generic credential harvesting
            "https://{}.secure-auth.{}.net/login?return=https://{}",
            "http://{}.login-verify.{}.com/signin",
            "https://{}-{}.credential-harvest.{}.net/auth",

            # Typosquatting examples
            "https://{}{}.{}",
            "https://{}-g00gle.com/login",
            "http://{}-appple.com/secure",
            "https://{}-paypa1.com/verify",

            # Brand impersonation with unusual TLDs
            "https://{}.secure-login.{}.xyz/login",
            "http://{}.account-verify.{}.top/secure",
            "https://{}.banking.{}.work/auth",
        ]

        brands = [
            "paypal", "apple", "microsoft", "amazon", "google",
            "facebook", "netflix", "instagram", "twitter", "linkedin",
            "chase", "bankofamerica", "wellsfargo", "dropbox", "adobe"
        ]

        tlds = ["com", "net", "org", "xyz", "top", "work", "buzz", "link", "info"]

        urls = []
        import random
        random.seed(42)  # Reproducibility

        for _ in range(500):
            pattern = random.choice(patterns)
            brand = random.choice(brands)

            # Generate variations
            if "{}" in pattern:
                num_placeholders = pattern.count("{}")
                if num_placeholders == 1:
                    url = pattern.format(brand)
                elif num_placeholders == 2:
                    tld = random.choice(tlds)
                    url = pattern.format(brand, tld)
                elif num_placeholders == 3:
                    tld = random.choice(tlds)
                    subdomain = random.choice(["secure", "account", "verify", "login", "update", ""])
                    url = pattern.format(subdomain, brand, tld)
                else:
                    url = pattern.format(brand, brand, random.choice(tlds))
            else:
                url = pattern

            # Add query parameters
            params = [
                f"?redirect={random.choice(brands)}.com",
                f"?verify=account&token={random.randint(100000, 999999)}",
                f"?session={hashlib.md5(str(random.random()).encode()).hexdigest()[:16]}",
                ""
            ]
            url += random.choice(params)

            urls.append(url)

        logger.info(f"Generated {len(urls)} sample phishing URLs")
        return urls

    def _generate_sample_legitimate_urls(self) -> List[str]:
        """
        Generate sample legitimate URLs for testing.

        Returns real, well-known legitimate website patterns.
        """
        legitimate_sites = [
            # Major tech companies
            "https://google.com", "https://www.google.com/mail",
            "https://accounts.google.com", "https://mail.google.com",
            "https://facebook.com", "https://www.facebook.com",
            "https://amazon.com", "https://www.amazon.com",
            "https://www.amazon.co.uk", "https://www.amazon.de",
            "https://apple.com", "https://www.apple.com",
            "https://microsoft.com", "https://www.microsoft.com",
            "https://github.com", "https://www.github.com",
            "https://linkedin.com", "https://www.linkedin.com",
            "https://twitter.com", "https://x.com",
            "https://instagram.com", "https://www.instagram.com",
            "https://netflix.com", "https://www.netflix.com",
            "https://reddit.com", "https://www.reddit.com",
            "https://youtube.com", "https://www.youtube.com",
            "https://wikipedia.org", "https://www.wikipedia.org",
            "https://stackoverflow.com", "https://www.stackoverflow.com",

            # Banking (legitimate)
            "https://www.chase.com", "https://onlinebanking.bankofamerica.com",
            "https://www.wellsfargo.com", "https://www.citi.com",
            "https://www.pnc.com", "https://www.capitalone.com",
            "https://www.usbank.com", "https://www.tdameritrade.com",

            # E-commerce
            "https://www.ebay.com", "https://www.walmart.com",
            "https://www.target.com", "https://www.bestbuy.com",

            # News and media
            "https://www.nytimes.com", "https://www.bbc.com",
            "https://www.cnn.com", "https://www.washingtonpost.com",

            # Government (legitimate)
            "https://www.irs.gov", "https://www.ssa.gov",
            "https://www.dhs.gov", "https://www.nasa.gov",

            # Cloud providers (legitimate)
            "https://aws.amazon.com", "https://cloud.google.com",
            "https://azure.microsoft.com", "https://www.digitalocean.com",

            # Payment processors (legitimate)
            "https://www.paypal.com", "https://www.stripe.com",
            "https://www.squareup.com", "https://www.shopify.com",
        ]

        # Generate variations with common path patterns
        common_paths = [
            "", "/about", "/products", "/services", "/contact",
            "/blog", "/news", "/help", "/support", "/faq",
            "/terms", "/privacy", "/careers", "/press",
        ]

        # Generate additional URLs with paths
        urls = list(legitimate_sites)  # Start with base URLs

        for site in legitimate_sites[:20]:  # Limit to avoid explosion
            for path in common_paths[:5]:
                if not (site.endswith('/') and path == ''):
                    urls.append(site.rstrip('/') + path)

        logger.info(f"Generated {len(urls)} sample legitimate URLs")
        return urls[:1000]  # Limit total

    def load_iscx_urls(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        Load ISCX URL 2016 dataset.

        Format: url,label (where label is 0=benign, 1=phishing/malicious)

        Args:
            file_path: Path to the ISCX CSV file

        Returns:
            Tuple of (urls, labels)
        """
        try:
            df = pd.read_csv(file_path, header=None, names=['url', 'label'],
                           on_bad_lines='skip')

            urls = df['url'].dropna().astype(str).tolist()
            labels = df['label'].fillna(0).astype(int).tolist()

            logger.info(f"Loaded ISCX dataset: {len(urls)} URLs, "
                       f"{sum(labels)} phishing, {len(labels) - sum(labels)} legitimate")
            return urls, labels

        except Exception as e:
            logger.error(f"ISCX dataset loading error: {e}")
            return [], []

    def create_balanced_dataset(self, phishing_urls: List[str],
                               legitimate_urls: List[str],
                               max_per_class: Optional[int] = None) -> LoadedDataset:
        """
        Create a balanced dataset with equal phishing and legitimate URLs.

        Args:
            phishing_urls: List of phishing URLs
            legitimate_urls: List of legitimate URLs
            max_per_class: Maximum URLs per class (for memory constraints)

        Returns:
            LoadedDataset with balanced classes
        """
        # Sample if needed
        phishing_sample = phishing_urls.copy()
        legitimate_sample = legitimate_urls.copy()

        if max_per_class:
            import random
            random.shuffle(phishing_sample)
            random.shuffle(legitimate_sample)
            phishing_sample = phishing_sample[:max_per_class]
            legitimate_sample = legitimate_sample[:max_per_class]

        # Balance classes
        min_count = min(len(phishing_sample), len(legitimate_sample))
        phishing_sample = phishing_sample[:min_count]
        legitimate_sample = legitimate_sample[:min_count]

        # Combine
        all_urls = phishing_sample + legitimate_sample
        all_labels = [1] * min_count + [0] * min_count

        dataset = LoadedDataset(
            name="Balanced",
            urls=all_urls,
            labels=all_labels,
            metadata={
                "total": len(all_urls),
                "phishing": sum(all_labels),
                "legitimate": len(all_labels) - sum(all_labels),
                "balanced": True,
            }
        )

        logger.info(f"Balanced dataset: {len(all_urls)} URLs "
                   f"({sum(all_labels)} phishing, {len(all_labels) - sum(all_labels)} legitimate)")

        return dataset

    def load_combined_dataset(self,
                             phishing_count: int = 500,
                             legitimate_count: int = 500) -> LoadedDataset:
        """
        Load or generate a combined dataset.

        This is the main entry point for getting training data.

        Args:
            phishing_count: Target number of phishing URLs
            legitimate_count: Target number of legitimate URLs

        Returns:
            LoadedDataset with URLs and labels
        """
        # Try to load real datasets
        phishing_urls = self._generate_sample_phishing_urls()
        legitimate_urls = self._generate_sample_legitimate_urls()

        # If real datasets are available, use them instead
        # (Users should replace these with actual dataset loading)

        # Limit to requested counts
        phishing_urls = phishing_urls[:phishing_count]
        legitimate_urls = legitimate_urls[:legitimate_count]

        return self.create_balanced_dataset(phishing_urls, legitimate_urls)


# ============================================================================
# FEATURE EXTRACTION PIPELINE
# ============================================================================

class FeatureExtractionPipeline:
    """
    Pipeline for extracting features from URL datasets.
    Handles batch processing with progress tracking.
    """

    def __init__(self, extractor: Optional[URLFeatureExtractor] = None):
        self.extractor = extractor or URLFeatureExtractor()

    def extract_features(self, urls: List[str], show_progress: bool = True
                       ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from a list of URLs.

        Args:
            urls: List of URLs to analyze
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (feature_matrix, valid_urls)
        """
        from tqdm import tqdm

        features_list = []
        valid_urls = []

        iterator = tqdm(urls, desc="Extracting features") if show_progress else urls

        for url in iterator:
            try:
                features = self.extractor.extract(url)
                feature_array = features.to_feature_array()
                features_list.append(feature_array)
                valid_urls.append(url)

            except Exception as e:
                logger.warning(f"Feature extraction failed for {url}: {e}")
                continue

        X = np.array(features_list, dtype=np.float32)
        return X, valid_urls

    def extract_and_save(self, dataset: LoadedDataset, output_path: str):
        """
        Extract features and save to file for training.

        Args:
            dataset: LoadedDataset with URLs and labels
            output_path: Path to save features (numpy .npz file)
        """
        logger.info(f"Extracting features for {dataset.size} URLs...")

        # Extract features
        X, valid_urls = self.extract_features(dataset.urls)

        # Get labels for valid URLs
        url_to_label = dict(zip(dataset.urls, dataset.labels))
        y = np.array([url_to_label[url] for url in valid_urls], dtype=np.int32)

        # Save
        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            urls=np.array(valid_urls)
        )

        logger.info(f"Saved features to {output_path}")
        logger.info(f"Shape: X={X.shape}, y={y.shape}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Test the dataset loader
    print("=" * 80)
    print("PhishGuard Dataset Loader Test")
    print("=" * 80)

    loader = DatasetLoader()

    # Load combined dataset
    dataset = loader.load_combined_dataset(
        phishing_count=200,
        legitimate_count=200
    )

    print(f"\nDataset Statistics:")
    print(f"  Total URLs: {dataset.size}")
    print(f"  Phishing: {dataset.phishing_count}")
    print(f"  Legitimate: {dataset.legitimate_count}")
    print(f"  Metadata: {dataset.metadata}")

    # Extract features
    print("\nExtracting features...")
    pipeline = FeatureExtractionPipeline()
    X, valid_urls = pipeline.extract_features(dataset.urls[:50])  # Test with subset

    print(f"\nFeature Matrix:")
    print(f"  Shape: {X.shape}")
    print(f"  Sample features (first URL):")
    print(f"  {X[0][:10]}...")  # First 10 features

    print("\n" + "=" * 80)
    print("Dataset loader test complete.")
    print("=" * 80)