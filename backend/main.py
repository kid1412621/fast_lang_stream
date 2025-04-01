# main.py
import uvicorn
from api.setup import create_app
from api.routes import router
from utils.logging import setup_logging

# Set up logging
logger = setup_logging()

# Create FastAPI app
app = create_app()

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,  # For development
        log_level="info",
    )