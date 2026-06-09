"""
PhishGuard FastAPI Backend
===========================
Main application entry point for the PhishGuard API.

This API provides real-time URL analysis combining:
- ML-based phishing detection (XGBoost)
- Cyber Threat Intelligence (VirusTotal, URLhaus)
- DNS/WHOIS forensics

Running the server:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("PhishGuard API Starting Up")
    logger.info("=" * 60)

    # Log environment
    logger.info(f"Environment: {'development' if os.getenv('DEBUG') else 'production'}")
    logger.info(f"API Key (VT): {'set' if os.getenv('VIRUSTOTAL_API_KEY') else 'not set'}")

    # Initialize services
    from services.ml_service import get_ml_service
    from services.cti_service import get_cti_service

    ml_service = get_ml_service()
    cti_service = get_cti_service()

    logger.info(f"ML Service: {'loaded' if ml_service.is_loaded else 'using rule-based fallback'}")
    logger.info(f"ML Model Version: {ml_service.version}")
    logger.info(f"CTI Service: VirusTotal={'enabled' if cti_service.virustotal_enabled else 'disabled'}, "
               f"URLhaus={'enabled' if cti_service.urlhaus_enabled else 'disabled'}")

    logger.info("=" * 60)
    logger.info("PhishGuard API Ready")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("PhishGuard API Shutting Down")


# ============================================================================
# APPLICATION FACTORY
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="PhishGuard API",
        description="""
## PhishGuard - ML-Powered Phishing URL Detection

PhishGuard provides real-time analysis of URLs to detect phishing attempts.
It combines:

- **Machine Learning** - XGBoost classifier trained on 35+ URL features
- **Threat Intelligence** - VirusTotal and URLhaus integration
- **Feature Extraction** - Lexical, structural, and behavioral analysis

### Authentication

Currently operates without authentication for demo purposes.
In production, add authentication middleware.

### Rate Limits

- Analysis endpoint: 100 requests/minute
- Batch analysis: 10 requests/minute
- CTI lookups: Subject to upstream API limits
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # =========================================================================
    # MIDDLEWARE
    # =========================================================================

    # CORS - Allow frontend to access API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # React dev server
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            # In production, add your domain here
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # Trusted host middleware (prevent host header attacks)
    # Note: Enable in production with proper host configuration
    # app.add_middleware(TrustedHostMiddleware, allowed_hosts=["phishguard.example.com"])

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests."""
        start_time = datetime.now()
        request_id = id(request)

        logger.info(f"[{request_id}] {request.method} {request.url.path}")

        response = await call_next(request)

        duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"[{request_id}] Completed in {duration:.1f}ms - {response.status_code}")

        return response

    # =========================================================================
    # ERROR HANDLERS
    # =========================================================================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat()
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if os.getenv("DEBUG") else None,
                "status_code": 500,
                "timestamp": datetime.now().isoformat()
            }
        )

    # =========================================================================
    # ROOT ENDPOINTS
    # =========================================================================

    @app.get("/", tags=["root"])
    async def root():
        """
        Root endpoint - API information.
        """
        return {
            "name": "PhishGuard API",
            "version": "1.0.0",
            "description": "ML-Powered Phishing URL Detection with CTI Integration",
            "documentation": "/docs",
            "health": "/api/v1/health"
        }

    @app.get("/ping", tags=["health"])
    async def ping():
        """Simple ping endpoint for readiness checks."""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    # =========================================================================
    # INCLUDE ROUTERS
    # =========================================================================

    from routers.analysis import router as analysis_router

    app.include_router(analysis_router)

    return app


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================

app = create_app()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))

    print("=" * 60)
    print("PhishGuard API Server")
    print("=" * 60)
    print(f"Starting server on {host}:{port}")
    print(f"Documentation: http://localhost:{port}/docs")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )