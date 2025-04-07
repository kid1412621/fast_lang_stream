# app/routes.py
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from langchain.schema import AIMessage, SystemMessage
import json
import time

from core.config import settings
from core.models import ModelInfo, StreamRequest
from core.llm import LLMManagerDep
from core.ollama import OllamaAIPDep
from utils.logging import get_logger
from utils.version import is_version_at_least

# Set up logger
logger = get_logger("routes")

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
async def health(
    llm_manager: LLMManagerDep,
):
    """Enhanced health check endpoint that verifies LLM availability"""
    try:
        # Try to create an LLM instance to verify Ollama is running
        llm_manager.get_llm(
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


# seems langchain has no built-in support for model listing, use REST API instead
@router.get("/models", response_model=List[ModelInfo])
async def list_ollama_models(ollama: OllamaAIPDep):
    """
    Endpoint to list all available models from Ollama.
    Fetches the list of available models from the Ollama API.
    """
    try:
        models = await ollama.models()
        current_version = await ollama.version()
        detect_from_capabilities = is_version_at_least(current_version, "0.6.4")
        for model in models:
            # Check if model is multimodal based on name patterns
            name = model["name"].lower()
            model["details"] = model.get("details", {})

            capabilities = (
                await ollama.capabilities(name) if detect_from_capabilities else []
            )
            is_multimodal = await ollama.is_multimodal_model(
                model_name=name, capabilities=capabilities
            )

            model["multimodal"] = is_multimodal
            model["details"]["capabilities"] = capabilities

        return models
    except httpx.RequestError as e:
        # Handle request-related errors
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Ollama API unavailable: {str(e)}")
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors (e.g., 4xx or 5xx)
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        # Handle any other exceptions
        logger.exception(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/stream")
async def stream(
    request: StreamRequest,
    llm_manager: LLMManagerDep,
):
    """Stream response using Server-Sent Events (SSE)"""

    async def event_generator():
        try:
            start_time = time.time()

            # Get the LLM instance
            llm = llm_manager.get_llm(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # Convert previous messages to LangChain message format
            messages = []

            has_images = hasattr(request, "images") and request.images

            # Add conversation history
            for msg in request.messages:
                if msg.role == "user":
                    message_images = (
                        request.images
                        if (msg == request.messages[-1] and has_images)
                        else None
                    )

                    # Create and add the user message
                    messages.append(
                        llm_manager.create_user_message(
                            text_content=msg.content,
                            image_data=message_images,
                            model=request.model,
                        )
                    )
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))
                elif (
                    msg.role == "system" or msg.role == "developer"
                ):  # compatible with openAI
                    messages.append(SystemMessage(content=msg.content))

            # If no messages were added, add default system prompt
            if len(messages) == 0:
                messages.append(SystemMessage(content="You are a helpful assistant."))

            # Log the conversation for debugging
            logger.info(f"Processing conversation with {len(messages)} messages")
            if has_images:
                logger.info(f"Request includes {len(request.images)} images")
            if request.allow_image_generation:
                logger.info("Image generation is enabled for this request")

            # Format SSE events properly using f-strings
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # Stream the response using astream (async streaming)
            async for chunk in llm.astream(messages):
                if hasattr(chunk, "content"):
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

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
