# utils/models.py
from pydantic import BaseModel, Field
from typing import List, Optional
from core.config import settings

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class StreamRequest(BaseModel):
    prompt: str
    messages: List[Message] = Field(default_factory=list)  # Previous conversation history
    model: str = settings.DEFAULT_MODEL
    temperature: float = settings.TEMPERATURE
    max_tokens: int = settings.MAX_TOKENS