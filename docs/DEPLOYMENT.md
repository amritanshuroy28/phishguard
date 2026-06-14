# PhishGuard Deployment Guide

Complete deployment instructions for PhishGuard components (Backend API, React Dashboard, and Chrome Extension).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development Deployment](#development-deployment)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Environment Configuration](#environment-configuration)
- [Chrome Extension Deployment](#chrome-extension-deployment)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows (WSL2 recommended)
- **Python**: 3.9+
- **Node.js**: 18+
- **npm/yarn**: Latest version
- **Disk Space**: 2GB minimum (for dependencies)

### Required Accounts (Optional)

- **VirusTotal API Key**: Get at https://www.virustotal.com/gui/home/upload (for threat intelligence)
- **URLhaus API**: Free access at https://urlhaus.abuse.ch/api/ (no key required)

### Software to Install

```bash
# macOS
brew install python@3.11 node git

# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3-pip nodejs npm git

# Windows (via chocolatey)
choco install python nodejs git
```

---

## Development Deployment

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/phishguard.git
cd phishguard

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Set optional environment variables
export VIRUSTOTAL_API_KEY="your-api-key-here"
export DEBUG=true

# Start development server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API will be available at: http://localhost:8000/docs

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Dashboard will be available at: http://localhost:5173

### 4. Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked"
4. Select the `chrome_extension` folder from the project root

---

## Production Deployment

### Backend Production Setup

#### Option 1: Using Gunicorn + Uvicorn (Recommended for VPS)

```bash
cd backend
pip install gunicorn uvicorn

# Start with gunicorn
gunicorn main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

**Configuration for 4-core server:**
- Workers: 4
- Worker class: uvicorn.workers.UvicornWorker
- Bind: 0.0.0.0:8000

#### Option 2: Using Systemd Service

Create `/etc/systemd/system/phishguard-api.service`:

```ini
[Unit]
Description=PhishGuard API
After=network.target

[Service]
Type=notify
User=phishguard
WorkingDirectory=/home/phishguard/phishguard/backend
Environment="PATH=/home/phishguard/phishguard/venv/bin"
Environment="VIRUSTOTAL_API_KEY=your-key-here"
Environment="DEBUG=false"
ExecStart=/home/phishguard/phishguard/venv/bin/gunicorn \
  main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8000 \
  --access-logfile /var/log/phishguard/access.log \
  --error-logfile /var/log/phishguard/error.log

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable phishguard-api
sudo systemctl start phishguard-api
sudo systemctl status phishguard-api
```

### Frontend Production Build

```bash
cd frontend
npm install
npm run build

# Outputs to: frontend/dist/
```

Deploy `dist/` folder to a static hosting service:

- **Nginx** (recommended for custom domain)
- **Vercel** (easy deployment)
- **AWS S3 + CloudFront**
- **GitHub Pages**

#### Nginx Configuration Example

Create `/etc/nginx/sites-available/phishguard`:

```nginx
server {
    listen 80;
    server_name phishguard.example.com;

    root /var/www/phishguard/frontend/dist;
    index index.html;

    # SPA routing - always serve index.html for client-side routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # GZIP compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/phishguard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### SSL/HTTPS Setup (Let's Encrypt)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot certonly --nginx -d phishguard.example.com

# Auto-renew
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

---

## Docker Deployment

### Dockerfile for Backend

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEBUG=false

EXPOSE 8000

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

### Dockerfile for Frontend

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine

COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - VIRUSTOTAL_API_KEY=${VIRUSTOTAL_API_KEY}
      - DEBUG=false
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: always
```

Deploy:

```bash
docker-compose up -d
docker-compose logs -f backend
```

---

## Environment Configuration

### Backend Environment Variables

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `VIRUSTOTAL_API_KEY` | - | VirusTotal API key | `your-api-key` |
| `VIRUSTOTAL_API_URL` | https://www.virustotal.com | VirusTotal endpoint | - |
| `DEBUG` | false | Enable debug logging | `true` or `false` |
| `HOST` | 0.0.0.0 | Server bind address | `127.0.0.1` |
| `PORT` | 8000 | Server port | `8000` |
| `WORKERS` | 1 | Gunicorn workers | `4` |

### Set Environment Variables

**Linux/macOS:**

```bash
export VIRUSTOTAL_API_KEY="your-key"
export DEBUG=false
```

**Windows (PowerShell):**

```powershell
$env:VIRUSTOTAL_API_KEY="your-key"
$env:DEBUG="false"
```

**.env file (for Docker):**

```bash
# .env
VIRUSTOTAL_API_KEY=your-key-here
DEBUG=false
```

### Frontend Configuration

Update backend API URL in `frontend/src/utils/api.ts`:

```typescript
// Development
const API_BASE = 'http://localhost:8000/api/v1';

// Production
const API_BASE = 'https://api.phishguard.example.com/api/v1';
```

---

## Chrome Extension Deployment

### Development Version

1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `chrome_extension` folder

### Publishing to Chrome Web Store

1. **Create a Google account** for developer registration
2. **Pay $5 registration fee** (one-time)
3. **Upload extension package:**
   - Go to https://chrome.google.com/webstore/devconsole
   - Create new item
   - Upload `chrome_extension` folder as `.zip`
4. **Fill in details:**
   - Title: "PhishGuard"
   - Description: "ML-powered phishing URL detection"
   - Category: Security
   - Screenshots (at least 1)
   - Store icon (128x128 PNG)
5. **Wait for review** (usually 1-3 hours)

### Configure Extension for Production

Update `chrome_extension/manifest.json`:

```json
{
  "manifest_version": 3,
  "name": "PhishGuard",
  "version": "1.0.0",
  "description": "ML-powered phishing URL detection",
  "permissions": ["scripting", "tabs", "host_permissions"],
  "host_permissions": ["<all_urls>"],
  "action": {
    "default_popup": "popup.html",
    "default_title": "PhishGuard - Check this URL"
  },
  "background": {
    "service_worker": "background.js"
  },
  "icons": {
    "16": "images/icon-16.png",
    "48": "images/icon-48.png",
    "128": "images/icon-128.png"
  }
}
```

---

## Monitoring & Logging

### Backend Logging

Logs are configured in `backend/main.py`. Configure log level:

```bash
export LOG_LEVEL=info  # debug, info, warning, error
```

**Log file rotation (with systemd):**

Add to service file:

```ini
StandardOutput=journal
StandardError=journal
SyslogIdentifier=phishguard
```

View logs:

```bash
journalctl -u phishguard-api -f
```

### Health Checks

```bash
# Simple ping
curl http://localhost:8000/ping

# Full health check (with API docs)
curl http://localhost:8000/docs
```

### Performance Monitoring

Monitor with `/api/v1/analyze` endpoint response times:

```bash
# Using Apache Bench
ab -n 100 -c 10 http://localhost:8000/ping

# Using curl
time curl http://localhost:8000/api/v1/analyze -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

### Metrics to Monitor

- API response time (target: <500ms)
- Error rate (target: <1%)
- Uptime (target: 99.9%)
- CPU/Memory usage
- CTI API availability (VirusTotal, URLhaus)

---

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Use different port
uvicorn main:app --port 8001
```

**VirusTotal API key not working:**

```bash
# Test API key
curl -X GET "https://www.virustotal.com/api/v3/files" \
  -H "x-apikey: YOUR_API_KEY"
```

**Import errors in backend:**

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python path
echo $PYTHONPATH
```

### Frontend Issues

**API calls failing (CORS error):**

Check `backend/main.py` CORS configuration:

```python
allow_origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "https://phishguard.example.com"
]
```

**Build fails:**

```bash
# Clear cache and rebuild
rm -rf node_modules dist
npm install
npm run build
```

### Chrome Extension Issues

**Extension not loading:**

1. Check `manifest.json` syntax
2. Ensure all referenced files exist
3. Check browser console for errors (Ctrl+Shift+I)
4. Try reloading extension

**API calls not working:**

Check `chrome_extension/background.js`:

```javascript
const API_URL = 'http://localhost:8000/api/v1';  // Update for production
```

---

## Production Checklist

- [ ] Environment variables configured (VIRUSTOTAL_API_KEY, DEBUG=false)
- [ ] CORS origins configured for production domain
- [ ] Backend running with gunicorn/systemd
- [ ] Frontend built and deployed to static hosting
- [ ] HTTPS/SSL configured with valid certificate
- [ ] DNS records pointing to correct servers
- [ ] Monitoring and logging enabled
- [ ] Database backups configured (if applicable)
- [ ] Chrome extension published to Web Store
- [ ] Tested end-to-end flow (extension → backend → CTI APIs)

---

## Support

For deployment issues, check:

1. **Application Logs**: `journalctl -u phishguard-api -f`
2. **Browser Console**: F12 → Console tab
3. **Network Tab**: F12 → Network tab (check API requests)
4. **GitHub Issues**: Report bugs at the project repository

