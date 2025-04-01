# app/api_setup.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import os
from core.config import settings
from utils.logging import get_logger

logger = get_logger("api_setup")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic here (if any)
    logger.info("Application startup")
    yield
    # Shutdown logic
    logger.info("Application shutting down")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Ollama LangChain API",
        description="A FastAPI application that provides an interface to Ollama LLM using LangChain",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware with better security
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        max_age=3600,
    )
    
    # Trusted hosts middleware
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"] if "*" in settings.CORS_ORIGINS else settings.CORS_ORIGINS
    )
    
    # Add a global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500, content={"detail": "An internal server error occurred."}
        )
    
    # Add request ID middleware for better tracking
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = f"{time.time()}-{os.urandom(4).hex()}"
        request.state.request_id = request_id
        logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed in {process_time:.3f}s")
        response.headers["X-Request-ID"] = request_id
        return response
    
    return app