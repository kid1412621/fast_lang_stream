# utils/models.py
from datetime import datetime
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

# Define the details structure
class ModelDetails(BaseModel):
    parent_model: Optional[str]
    format: str
    family: str
    families: List[str]
    parameter_size: str
    quantization_level: str

# Define the structure for each model
class ModelInfo(BaseModel):
    name: str
    model: str
    modified_at: datetime
    size: int
    digest: str
    details: ModelDetails