# config.py
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings with simple environment variable loading"""
    
    # Server settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
    RATE_LIMIT = int(os.getenv("RATE_LIMIT", "100"))
    CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "300"))
    
    # LLM settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "10"))

# Initialize settings once
settings = Settings()