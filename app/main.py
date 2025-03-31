# main.py
from typing import AsyncGenerator, Optional, Dict
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, ConfigDict
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import logging
from functools import lru_cache
import time
from datetime import datetime
import os
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)


# Configuration
class Settings(BaseSettings):
    CORS_ORIGINS: list = ["*"]
    RATE_LIMIT: int = 100
    DEFAULT_MODEL: str = "llama3.2"
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7
    CACHE_SIZE: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"  # Added for better encoding support


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic here (if any)
    logger.info("Application startup")
    yield
    # Shutdown logic
    logger.info("Application shutting down")
    # Add any cleanup operations here


# FastAPI app with metadata
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,
)

# Trusted hosts middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.CORS_ORIGINS)

# Rate limiting
rate_limit_dict: Dict[str, Dict] = {}


# Pydantic models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class StreamRequest(BaseModel):
    prompt: str
    messages: list[Message] = []  # Previous conversation history
    model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 1000


@lru_cache(maxsize=settings.CACHE_SIZE)
def get_llm(model: str, temperature: float, max_tokens: int) -> ChatOllama:
    """Cached LLM instance"""
    return ChatOllama(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        # format="json",  # Enable JSON mode for better response handling
    )


def create_chain(llm: ChatOllama) -> ChatPromptTemplate:
    """Create a LangChain chain with the given LLM"""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a helpful AI assistant. Provide clear and concise responses to user queries."
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    return prompt | llm | StrOutputParser()


async def check_rate_limit(request: Request) -> bool:
    """Rate limiting dependency"""
    client_ip = request.client.host
    current_time = time.time()

    if client_ip in rate_limit_dict:
        if current_time - rate_limit_dict[client_ip]["timestamp"] > 60:
            rate_limit_dict[client_ip] = {"count": 1, "timestamp": current_time}
        else:
            rate_limit_dict[client_ip]["count"] += 1
            if rate_limit_dict[client_ip]["count"] > settings.RATE_LIMIT:
                raise HTTPException(
                    status_code=429, detail="Too many requests. Please try again later."
                )
    else:
        rate_limit_dict[client_ip] = {"count": 1, "timestamp": current_time}

    return True


async def cleanup_rate_limit():
    """Background task to clean up old rate limit entries"""
    current_time = time.time()
    expired_ips = [
        ip
        for ip, data in rate_limit_dict.items()
        if current_time - data["timestamp"] > 60
    ]
    for ip in expired_ips:
        del rate_limit_dict[ip]
    logger.info(f"Cleaned up {len(expired_ips)} expired rate limit entries")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Ollama LangChain API is running",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "settings": {
            "default_model": settings.DEFAULT_MODEL,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
            "rate_limit": settings.RATE_LIMIT,
        },
    }


@app.get("/health", response_model=dict)
async def health():
    """Enhanced health check endpoint that verifies LLM availability"""
    try:
        # Try to create an LLM instance to verify Ollama is running
        llm = get_llm(
            model=settings.DEFAULT_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=10,  # Small value for quick check
        )
        return {
            "status": "healthy",
            "llm": "available",
            "model": settings.DEFAULT_MODEL,
        }
    except Exception as e:
        logger.warning(f"Health check failed: {str(e)}")
        return {"status": "degraded", "llm": "unavailable", "error": str(e)}


@app.post("/stream")
async def stream(request: StreamRequest,
                 background_tasks: BackgroundTasks,
                 _: bool = Depends(check_rate_limit),
                 ):
    """Stream response using Server-Sent Events (SSE)"""

    async def event_generator():
        try:
            start_time = time.time()

            llm = get_llm(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # Convert previous messages to LangChain message format
            messages = []

            # Always start with a system message
            messages.append(SystemMessage(content="You are a helpful assistant."))

            # Add conversation history
            for msg in request.messages:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))

            # Add the current prompt
            messages.append(HumanMessage(content=request.prompt))

            # Log the conversation for debugging
            logger.info(f"Processing conversation with {len(messages)} messages")

            # Format SSE events properly
            yield "data: {}\n\n".format(json.dumps({"type": "start"}))

            # Stream the response
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content'):
                    # Format as an SSE event
                    yield "data: {}\n\n".format(json.dumps({
                        "type": "chunk",
                        "data": chunk.content
                    }))

            # Send completion event
            yield "data: {}\n\n".format(json.dumps({
                "type": "end"
            }))

        except Exception as e:
            logger.exception(f"Error in streaming: {str(e)}")
            yield "data: {}\n\n".format(json.dumps({
                "type": "error",
                "data": str(e)
            }))
        finally:
            # Ensure cleanup task is added even in case of errors
            background_tasks.add_task(cleanup_rate_limit)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,  # For development
        log_level="info",
    )
