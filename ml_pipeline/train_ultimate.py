"""
PhishGuard Ultimate Model Trainer — single best XGBoost trained on:
  - Phishing.Database (open-source threat intel)
  - PhiUSIIL Phishing URL Dataset (legitimate URLs from label=1)
  - Hardcoded trusted-domain list

Target: >95% test accuracy, F1, and AUC.

Saves to /home/amritanshu/new/phishguard/models/phishguard_ultimate.pkl
"""

import os, sys, json, pickle, random, logging, time
from pathlib import Path
from datetime import datetime
from statistics import mean
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

# Resolve project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PHISH_DB_PATH = Path("/tmp/phish_db")
UCI_CSV = PROJECT_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT_PKL = PROJECT_ROOT / "models" / "phishguard_ultimate.pkl"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("ultimate")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Configuration ─────────────────────────────────────────────────────────
PHISH_SAMPLE = 150_000        # from Phish.Database
LEGIT_UCI_SAMPLE = 100_000    # from PhiUSIIL label=1
LEGIT_HARDCODED_COUNT = 500   # trusted domains
TEST_SIZE = 0.2
N_FOLDS = 5


# ── Phish.Database loader ─────────────────────────────────────────────────
def load_phishdb(max_n: int = PHISH_SAMPLE) -> List[str]:
    """Pull real phishing URLs from Phishing.Database."""
    f = PHISH_DB_PATH / "phishing-links-ACTIVE.txt"
    urls, skipped = [], 0
    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            u = line.strip()
            if not u:
                continue
            low = u.lower()
            # Keep only http/https, drop credentials in URL, drop very short
            if not (low.startswith("http://") or low.startswith("https://")):
                skipped += 1
                continue
            if "@" in u:
                skipped += 1
                continue
            if len(u) < 20:
                skipped += 1
                continue
            urls.append(u)
            if len(urls) >= max_n:
                break
    log.info(f"PhishDB: {len(urls)} URLs loaded, {skipped} skipped")
    return urls


# ── PhiUSIIL legitimate URLs (label=1) ───────────────────────────────────
def load_uci_legit(max_n: int = LEGIT_UCI_SAMPLE) -> List[str]:
    """PhiUSIIL: actual files show label=1 = legitimate universities/companies,
    label=0 = suspicious/compromised.  Skip validating label semantics — extract
    URLs from rows whose URL has clean TLD and short hostname."""
    df = pd.read_csv(UCI_CSV, usecols=["URL", "label", "Domain", "TLD"])
    legit_rows = df[df["label"] == 1].copy()
    legit_rows = legit_rows[legit_rows["Domain"].str.len() < 30]  # sanity check
    legit_rows = legit_rows[legit_rows["Domain"].str.contains(r"\.(com|org|edu|net|gov|io|co\.uk|de|jp|fr)$",
                                                              case=False, regex=True, na=False)]
    urls = legit_rows["URL"].drop_duplicates().head(max_n).tolist()
    log.info(f"UCI legit (label=1, clean TLD): {len(urls)} URLs")
    return urls


# ── Hardcoded trusted domains ─────────────────────────────────────────────
TRUSTED_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "reddit.com",
    "amazon.com", "twitter.com", "instagram.com", "linkedin.com", "microsoft.com",
    "apple.com", "netflix.com", "paypal.com", "github.com", "dropbox.com",
    "salesforce.com", "wordpress.com", "bing.com", "yahoo.com", "ebay.com",
    "imdb.com", "stackexchange.com", "spotify.com", "twitch.tv", "tiktok.com",
    "pinterest.com", "tumblr.com", "flickr.com", "vimeo.com", "medium.com",
    "office.com", "live.com", "outlook.com", "gmail.com", "whatsapp.com",
    "cloudflare.com", "akamai.com", "fastly.com", "digitalocean.com",
    "heroku.com", "aws.amazon.com", "azure.microsoft.com", "cloud.google.com",
    "ibm.com", "oracle.com", "sap.com", "adobe.com", "autodesk.com",
    "walmart.com", "target.com", "bestbuy.com", "homedepot.com", "costco.com",
    "cnn.com", "bbc.com", "nytimes.com", "theguardian.com", "reuters.com",
    "bloomberg.com", "forbes.com", "wsj.com", "ft.com", "economist.com",
    "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu", "oxford.ac.uk",
    "cam.ac.uk", "yale.edu", "princeton.edu", "columbia.edu", "cornell.edu",
    "github.io", "gitlab.com", "bitbucket.org", "sourceforge.net",
    "stackoverflow.com", "quora.com", "duolingo.com", "coursera.org", "udemy.com",
    "khanacademy.org", "edx.org", "ted.com", "nasa.gov", "nih.gov",
    "cdc.gov", "who.int", "europa.eu", "gov.uk", "whitehouse.gov",
    "openai.com", "anthropic.com", "huggingface.co", "kaggle.com", "colab.research.google.com",
    "stripe.com", "squareup.com", "venmo.com", "wise.com", "revolut.com",
    "binance.com", "coinbase.com", "kraken.com", "bitstamp.net",
    "shopify.com", "squarespace.com", "wix.com", "webflow.com",
    "substack.com", "patreon.com", "kickstarter.com", "gofundme.com",
]


def load_trusted_urls() -> List[str]:
    """Build URLs from trusted domains with varied paths."""
    random.seed(RANDOM_STATE)
    paths = ["", "/", "/about", "/login", "/help", "/contact",
             "/products", "/api", "/docs", "/blog", "/home", "/index.html",
             "/accounts/login", "/en-us", "/?ref=home", "/users/sign_in"]
    urls = []
    for d in TRUSTED_DOMAINS:
        for p in paths:
            proto = random.choice(["https://", "http://"])
            sub = random.choice(["", "www.", "m.", "help."])
            urls.append(f"{proto}{sub}{d}{p}")
    random.shuffle(urls)
    log.info(f"Trusted domains: {len(urls)} URLs")
    return urls


# ── Feature extraction ────────────────────────────────────────────────────
def extract_features(urls: List[str], extractor) -> Tuple[np.ndarray, List[str]]:
    X, fail = [], []
    for u in urls:
        try:
            f = extractor.extract(u)
            arr = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                   for v in f.to_feature_array()]
            X.append(arr)
        except Exception:
            fail.append(u)
    return np.array(X, dtype=np.float32), fail


# ── Main training pipeline ───────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("PhishGuard ULTIMATE Model Trainer — single best XGBoost, target >95%")
    log.info("=" * 70)

    from ml_pipeline.features.feature_extraction import URLFeatureExtractor
    extractor = URLFeatureExtractor()

    # ── Load all URLs ────────────────────────────────────────────────────
    phish_urls = load_phishdb(PHISH_SAMPLE)
    legit_uci = load_uci_legit(LEGIT_UCI_SAMPLE)
    legit_dom = load_trusted_urls()

    # Combine legitimate URLs (de-dup with phish)
    legit_set = set(legit_uci + legit_dom) - set(phish_urls)
    legit_urls = list(legit_set)

    log.info(f"Final class sizes → phish={len(phish_urls)}, legit={len(legit_urls)}")

    log.info(f"Final class sizes → phish={len(phish_urls)}, legit={len(legit_urls)}")

    # ── Extract features (parallel possible, but we'll do sequential for simplicity) ──
    log.info("Extracting phish features …")
    t1 = time.time()
    X_p, fail_p = extract_features(phish_urls, extractor)
    log.info(f"  done — {X_p.shape}, {len(fail_p)} failures — {time.time()-t1:.1f}s")

    log.info("Extracting legit features …")
    t1 = time.time()
    X_l, fail_l = extract_features(legit_urls, extractor)
    log.info(f"  done — {X_l.shape}, {len(fail_l)} failures — {time.time()-t1:.1f}s")

    X = np.vstack([X_p, X_l])
    y = np.concatenate([np.ones(X_p.shape[0], dtype=np.int_),
                        np.zeros(X_l.shape[0], dtype=np.int_)])

    # Sanitize
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    log.info(f"Final dataset: X={X.shape}, y={y.shape}, phish_legit={(y==1).sum()}/{(y==0).sum()}")

    # ── Train/test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # ── XGBoost tuning ───────────────────────────────────────────────────
    log.info("Training final XGBoost …")
    t1 = time.time()

    params = dict(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    log.info(f"  trained in {time.time()-t1:.1f}s")

    # ── Test evaluation ──────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    m = dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1_score=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
    log.info(f"TEST — Acc:{m['accuracy']:.4f}  Prec:{m['precision']:.4f}  "
             f"Rec:{m['recall']:.4f}  F1:{m['f1_score']:.4f}  AUC:{m['roc_auc']:.4f}")
    log.info(f"Confusion matrix: {cm}")
    log.info("Classification report:")
    log.info(classification_report(y_test, y_pred, digits=4))

    # ── 5-fold cross-validation ──────────────────────────────────────────
    log.info("Running 5-fold cross-validation on training set …")
    t1 = time.time()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    log.info(f"CV accuracy: {cv_acc.mean():.4f} ± {cv_acc.std()*2:.4f}  [{cv_acc}]")
    log.info(f"CV f1:       {cv_f1.mean():.4f} ± {cv_f1.std()*2:.4f}  [{cv_f1}]")
    log.info(f"  CV in {time.time()-t1:.1f}s")

    # ── Refit on full data for the saved model ───────────────────────────
    log.info("Refitting on full dataset for final model …")
    t1 = time.time()
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)
    log.info(f"  full-data refit in {time.time()-t1:.1f}s")

    # ── Feature importance ───────────────────────────────────────────────
    feature_names = [
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
    importances = sorted(
        zip(feature_names, final_model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    log.info("Top 10 features:")
    for name, imp in importances[:10]:
        log.info(f"  {name:30} {imp:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    pkg = dict(
        model=final_model,
        features=feature_names,
        feature_names=feature_names,
        metrics=m,
        cv_accuracy=float(cv_acc.mean()),
        cv_accuracy_std=float(cv_acc.std()),
        cv_f1=float(cv_f1.mean()),
        cv_f1_std=float(cv_f1.std()),
        confusion_matrix=cm,
        params=params,
        training_data=dict(
            phishdb_count=len(phish_urls),
            uci_legit_count=len(legit_uci),
            trusted_count=len(legit_dom),
            total_samples=len(X),
        ),
        trained_at=datetime.now().isoformat(),
    )
    with open(OUTPUT_PKL, "wb") as fh:
        pickle.dump(pkg, fh)
    log.info(f"Saved → {OUTPUT_PKL}")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    target_met = m["accuracy"] >= 0.95
    log.info("=" * 70)
    log.info(f"FINAL — accuracy={m['accuracy']:.4f}  f1={m['f1_score']:.4f}  auc={m['roc_auc']:.4f}")
    log.info(f"Target >95% accuracy: {'✅ MET' if target_met else '❌ NOT MET'}")
    log.info(f"Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
