"""Food AI Backend — FastAPI application entry point."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.middleware.audit_log import AuditLogMiddleware
from app.routers import admin, foods, health, insights, meals, reports, symptoms

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (global, keyed by client IP)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ---------------------------------------------------------------------------
# Tag metadata — drives ordering and descriptions in /docs and the generated SDK
# ---------------------------------------------------------------------------
tags_metadata = [
    {
        "name": "health",
        "description": "Service health and readiness checks. Used by load balancers and uptime monitors.",
    },
    {
        "name": "meals",
        "description": "Meal logging and food-item tracking. Core diary workflow for participants.",
    },
    {
        "name": "symptoms",
        "description": "Symptom logging with severity scores. Timestamped entries linked to meal windows.",
    },
    {
        "name": "foods",
        "description": "Food knowledge-base search with allergen and component information.",
    },
    {
        "name": "insights",
        "description": "AI-powered dietary trigger analysis — predicts foods correlated with symptom flares.",
    },
    {
        "name": "reports",
        "description": "Clinician-ready PDF report generation for patient/doctor review sessions.",
    },
    {
        "name": "admin",
        "description": "Internal admin operations (food ingestion, maintenance). Requires admin API key.",
    },
]

app = FastAPI(
    title="Food AI - Dietary Sensitivity Tracking API",
    description=(
        "Backend API for the Food AI mobile app. Tracks meals, logs symptoms, "
        "identifies food triggers through AI-powered pattern recognition, and "
        "generates clinician-ready reports."
    ),
    version="0.1.0",
    # Disable interactive API docs in production to reduce attack surface
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    contact={"name": "Food AI Team"},
    openapi_tags=tags_metadata,
)

# Attach rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Exception handlers — standardised error shapes for all consumers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 with a structured list of field-level validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Return the appropriate HTTP status with a consistent {detail} shape."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all for unexpected errors — never leak stack traces to clients."""
    _logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Middleware (outermost -> innermost; execute in reverse order)
# ---------------------------------------------------------------------------
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(AuditLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Admin-Key"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(meals.router)
app.include_router(symptoms.router)
app.include_router(foods.router)
app.include_router(insights.router)
app.include_router(reports.router)
app.include_router(admin.router)


@app.on_event("startup")
async def on_startup() -> None:
    _logger.info(
        "Food AI Backend starting — env=%s docs_enabled=%s",
        settings.app_env,
        not settings.is_production,
    )
