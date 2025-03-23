# main.py
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, ConfigDict
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import logging
import asyncio
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with metadata
app = FastAPI(
    title="Ollama LangChain API",
    description="A FastAPI application that provides an interface to Ollama LLM using LangChain",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware with better security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,
)

# Trusted hosts middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # In production, replace with specific hosts
)

# Rate limiting
RATE_LIMIT = 100  # requests per minute
rate_limit_dict = {}


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(default="llama3.2", pattern="^[a-zA-Z0-9._-]+$")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=1, le=4000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "What is the capital of France?",
                "model": "llama3.2",
                "temperature": 0.7,
                "max_tokens": 1000,
            }
        }
    )


class QueryResponse(BaseModel):
    response: str
    model: str
    processing_time: float


class StreamResponse(BaseModel):
    type: str
    data: Optional[str] = None


@lru_cache(maxsize=10)
def get_llm(model: str, temperature: float, max_tokens: int) -> ChatOllama:
    """Cached LLM instance"""
    return ChatOllama(
        model=model, temperature=temperature, max_tokens=max_tokens, streaming=True
    )


def create_chain(llm: ChatOllama) -> ChatPromptTemplate:
    """Create a LangChain chain with the given LLM"""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a helpful history professor."),
            HumanMessagePromptTemplate.from_template("{input}"),
        ],
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
            if rate_limit_dict[client_ip]["count"] > RATE_LIMIT:
                raise HTTPException(
                    status_code=429, detail="Too many requests. Please try again later."
                )
    else:
        rate_limit_dict[client_ip] = {"count": 1, "timestamp": current_time}

    return True


@app.get("/", response_model=dict)
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Ollama LangChain API is running",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, _: bool = Depends(check_rate_limit)):
    """Process a single query and return the response"""
    try:
        start_time = time.time()

        llm = get_llm(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        chain = create_chain(llm)
        response = chain.invoke({"input": request.prompt})

        processing_time = time.time() - start_time

        return QueryResponse(
            response=response, model=request.model, processing_time=processing_time
        )
    except Exception as e:
        logger.exception("Error in query endpoint")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        )


@app.post("/stream")
async def stream(request: QueryRequest, _: bool = Depends(check_rate_limit)):
    """Stream response using Server-Sent Events (SSE)"""

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            start_time = time.time()

            llm = get_llm(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            chain = create_chain(llm)

            yield f"data: {json.dumps({'type': 'start', 'model': request.model})}\n\n"

            async for chunk in chain.astream({"input": request.prompt}):
                yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"
                await asyncio.sleep(0)

            processing_time = time.time() - start_time
            yield f"data: {json.dumps({'type': 'end', 'processing_time': processing_time})}\n\n"

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
