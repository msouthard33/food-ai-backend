"""Food AI Backend — FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.middleware.audit_log import AuditLogMiddleware
from app.routers import admin, foods, health, insights, meals, reports, symptoms

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="Food AI - Dietary Sensitivity Tracking API",
    description=(
        "Backend API for the Food AI mobile app. Tracks meals, logs symptoms, "
        "identifies food triggers through AI-powered pattern recognition, and "
        "generates clinician-ready reports."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Middleware ---
app.add_middleware(AuditLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(health.router)
app.include_router(meals.router)
app.include_router(symptoms.router)
app.include_router(foods.router)
app.include_router(insights.router)
app.include_router(reports.router)
app.include_router(admin.router)


@app.on_event("startup")
async def on_startup() -> None:
    logging.getLogger(__name__).info(
        "Food AI Backend starting — env=%s", settings.app_env
    )
