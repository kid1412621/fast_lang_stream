# utils/models.py
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    """Message model for chat conversations"""

    role: str
    content: str


class StreamRequest(BaseModel):
    """Request model for streaming responses"""

    messages: List[Message] = []
    model: str = Field(..., description="The model to use for generation")
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    images: Optional[List[str]] = None  # Add support for base64 encoded images
    allow_image_generation: Optional[bool] = False  # Flag to allow image generation


# Define the details structure
class ModelDetails(BaseModel):
    parent_model: Optional[str]
    format: str
    family: str
    families: List[str]
    parameter_size: str
    quantization_level: str
    # TODO
    tags: List[str]


# Define the structure for each model
class ModelInfo(BaseModel):
    name: str
    model: str
    modified_at: datetime
    size: int
    digest: str
    details: ModelDetails
