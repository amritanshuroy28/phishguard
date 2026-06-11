# PhishGuard

> ML-Powered Phishing URL Detection System with Cyber Threat Intelligence Integration

## Overview

PhishGuard is a comprehensive phishing detection system that combines statistical Machine Learning classification with active Cyber Threat Intelligence (CTI). It operates as a real-time Chrome browser extension, analyzing URLs on-the-fly and providing structured threat reports.

### Key Features

- **ML-Based Detection**: XGBoost classifier trained on 35+ URL features
- **Real-Time Protection**: Chrome extension with live URL monitoring
- **Threat Intelligence**: VirusTotal & URLhaus integration
- **Forensics**: DNS/WHOIS lookups, typosquatting detection
- **Analyst Dashboard**: React-based UI with visualizations and IoC export

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PhishGuard Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │    Chrome    │         │    React     │                     │
│  │  Extension   │────────▶│  Dashboard   │                     │
│  └──────┬───────┘         └──────┬───────┘                     │
│         │                       │                              │
│         │ REST API              │ REST API                     │
│         ▼                       ▼                              │
│  ┌─────────────────────────────────────────┐                   │
│  │           FastAPI Backend                 │                   │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  │                   │
│  │  │Feature  │  │   ML     │  │   CTI   │  │                   │
│  │  │Extractor│  │ Classifier│  │ Service │  │                   │
│  │  └─────────┘  └──────────┘  └─────────┘  │                   │
│  └─────────────────────────────────────────┘                   │
│         │                       │                              │
│         ▼                       ▼                              │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │   Local      │         │  External    │                     │
│  │   Model      │         │  APIs (VT,   │                     │
│  │  (XGBoost)   │         │   URLhaus)   │                     │
│  └──────────────┘         └──────────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### 1. ML Model Training (Optional - model is pre-trained)

```bash
cd ml_pipeline
pip install -r requirements.txt
python train_model.py
```

### 2. Start Backend API

```bash
cd backend
pip install -r requirements.txt

# Set environment variables (optional)
export VIRUSTOTAL_API_KEY="your-api-key"

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000
API docs: http://localhost:8000/docs

### 3. Start React Dashboard

```bash
cd frontend
npm install
npm run dev
```

Dashboard will be available at: http://localhost:5173

### 4. Load Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked"
4. Select the `chrome_extension` folder

## Project Structure

```
phishguard/
├── ml_pipeline/           # ML training pipeline
│   ├── features/          # Feature extraction code
│   ├── train_model.py     # Model training script
│   └── dataset_loader.py  # Data loading utilities
├── backend/               # FastAPI backend
│   ├── services/          # ML and CTI services
│   ├── routers/            # API endpoints
│   ├── schemas/            # Pydantic models
│   └── main.py             # Application entry point
├── chrome_extension/       # Browser extension
│   ├── background.js       # Service worker
│   ├── popup.html/js       # Popup UI
│   └── manifest.json       # Extension manifest
├── frontend/               # React dashboard
│   └── src/
│       ├── pages/          # Dashboard pages
│       ├── components/     # Shared components
│       └── utils/          # API client, helpers
└── docs/                   # Documentation
```

## API Reference

### POST /api/v1/analyze

Analyze a single URL.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**Response:**
```json
{
  "id": "uuid",
  "url": "https://example.com",
  "risk_level": "safe",
  "risk_score": 15.2,
  "is_malicious": false,
  "ml_confidence": 0.95,
  "threats": [],
  "ctis": [...],
  "processing_time_ms": 142.5
}
```

### GET /api/v1/history

Retrieve scan history.

```bash
curl http://localhost:8000/api/v1/history?limit=50
```

### GET /api/v1/iocs/export

Export IoCs as CSV or JSON.

```bash
curl -O http://localhost:8000/api/v1/iocs/export?format=csv
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIRUSTOTAL_API_KEY` | - | VirusTotal API key for threat intelligence |
| `VIRUSTOTAL_API_URL` | https://www.virustotal.com | VirusTotal API endpoint |
| `DEBUG` | false | Enable debug logging |
| `HOST` | 0.0.0.0 | Server bind host |
| `PORT` | 8000 | Server port |

### Feature Extraction

Features extracted include:
- URL length, path depth, entropy
- Obfuscation indicators (hex encoding, IP addresses)
- Suspicious path patterns
- Typosquatting detection (Levenshtein distance)
- Domain age indicators

## Testing

### Backend

```bash
cd backend
pytest
```

### Chrome Extension

1. Open `chrome://extensions/`
2. Find PhishGuard and click "Errors"
3. Alternatively, open DevTools console on the background page

## Performance

- API response time: < 500ms (target)
- Feature extraction: ~5ms per URL
- ML inference: < 10ms
- CTI lookups: Configurable timeout (default 2s)

## Deployment

### Backend (Production)

```bash
cd backend
pip install -r requirements.txt
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Frontend (Production)

```bash
cd frontend
npm run build
# Serve dist/ with nginx or similar
```

## Acknowledgments

- PhishTank for phishing URL data
- URLhaus for malware URL feeds
- Tranco for legitimate domain lists
