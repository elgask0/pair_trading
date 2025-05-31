from loguru import logger
import sys
import os
from config.settings import settings

def setup_logger():
    """Setup structured logging with loguru"""
    
    # Remove default handler
    logger.remove()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler
    logger.add(
        "logs/trading.log",
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    return logger

# Initialize logger
log = setup_logger()