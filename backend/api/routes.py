# app/routes.py
from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import time
from core.config import settings
from core.models import StreamRequest
from utils.rate_limiter import RateLimiter
from core.llm import LLMManager
from utils.logging import get_logger

# Set up logger
logger = get_logger("routes")

# Initialize rate limiter and LLM manager
rate_limiter = RateLimiter(
    requests_per_minute=settings.RATE_LIMIT,
    cleanup_interval=settings.CLEANUP_INTERVAL
)
llm_manager = LLMManager(cache_size=settings.CACHE_SIZE)

# Create router
router = APIRouter()

@router.get("/", response_model=dict)
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

@router.get("/health", response_model=dict)
async def health():
    """Enhanced health check endpoint that verifies LLM availability"""
    try:
        # Try to create an LLM instance to verify Ollama is running
        llm = llm_manager.get_llm(
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

@router.post("/stream")
async def stream(
    request: StreamRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(rate_limiter.check_rate_limit),
):
    """Stream response using Server-Sent Events (SSE)"""

    async def event_generator():
        try:
            start_time = time.time()

            llm = llm_manager.get_llm(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # Convert previous messages to LangChain message format
            messages = [SystemMessage(content="You are a helpful assistant.")]

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

            # Format SSE events properly using f-strings
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # Stream the response using astream (async streaming)
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content'):
                    # Format as an SSE event using f-strings
                    yield f"data: {json.dumps({'type': 'chunk', 'data': chunk.content})}\n\n"

            # Send completion event using f-strings
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
            # Log completion time
            process_time = time.time() - start_time
            logger.info(f"Request completed in {process_time:.3f}s")

        except Exception as e:
            logger.exception(f"Error in streaming: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        finally:
            # Ensure cleanup task is added even in case of errors
            background_tasks.add_task(rate_limiter.cleanup_rate_limit)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )