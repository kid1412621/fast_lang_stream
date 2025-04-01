# utils/logging.py
import logging
import os
from datetime import datetime

def setup_logging():
    """Configure logging with detailed format"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{log_dir}/app_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ],
    )
    
    # Return logger for the main application
    return logging.getLogger("api")

def get_logger(name):
    """Get a logger with the given name"""
    return logging.getLogger(name)