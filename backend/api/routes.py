# app/routes.py
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import time

from core.config import settings
from core.models import ModelInfo, StreamRequest
from core.llm import LLMManagerDep
from utils.logging import get_logger

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
# ollama just updated its API to check model capability to check if is multimodal or not, wait for realease
# see: https://github.com/ollama/ollama/pull/10066
@router.get("/models", response_model=List[ModelInfo])
async def list_ollama_models(llm_manager: LLMManagerDep):
    """
    Endpoint to list all available models from Ollama.
    Fetches the list of available models from the Ollama API.
    """
    try:
        # Use httpx to fetch the list of models from the Ollama API
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_URL}/api/tags")

        # Raise an HTTP exception if the request was not successful
        response.raise_for_status()

        # Parse the JSON response from the Ollama API
        available_models = response.json()

        # Enhanced model info with multimodal capability detection
        models = available_models["models"]
        for model in models:
            # Check if model is multimodal based on name patterns
            name = model["name"].lower()
            model["details"] = model.get("details", {})
            
            # Add multimodal capability flag using the utility method
            is_multimodal = llm_manager.is_multimodal_model(name)
            
            # Add tags if not present
            if "tags" not in model["details"]:
                model["details"]["tags"] = []
                
            # Add multimodal tag if applicable
            if is_multimodal and "multimodal" not in model["details"]["tags"]:
                model["details"]["tags"].append("multimodal")

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
            messages = [SystemMessage(content="You are a helpful assistant.")]

            # Check if we're dealing with a multimodal model
            is_multimodal = llm_manager.is_multimodal_model(request.model)
            has_images = hasattr(request, 'images') and request.images

            # Add conversation history
            for msg in request.messages:
                if msg.role == "user":
                    # Check if this message has images and is the latest message
                    if is_multimodal and has_images and msg == request.messages[-1]:
                        # Create a multimodal message with text and images
                        multimodal_msg = llm_manager.create_multimodal_message(
                            text_content=msg.content,
                            image_data=request.images
                        )
                        messages.append(multimodal_msg)
                    else:
                        # Regular text message
                        messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))

            # If no messages were added but we have a prompt, add it
            if len(messages) == 1 and request.prompt:  # Only system message was added
                if is_multimodal and has_images:
                    # Create a multimodal message with the prompt and images
                    multimodal_msg = llm_manager.create_multimodal_message(
                        text_content=request.prompt,
                        image_data=request.images
                    )
                    messages.append(multimodal_msg)
                else:
                    # Regular text prompt
                    messages.append(HumanMessage(content=request.prompt))

            # Log the conversation for debugging
            logger.info(f"Processing conversation with {len(messages)} messages")
            if has_images:
                logger.info(f"Request includes {len(request.images)} images")

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