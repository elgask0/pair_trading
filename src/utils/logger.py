from loguru import logger
import sys
import os
from pathlib import Path
from config.settings import settings

def setup_logger(script_name: str = None):
    """Setup structured logging with loguru for specific scripts"""
    
    # Remove default handler
    logger.remove()
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Determine log file name based on script
    if script_name:
        log_file = logs_dir / f"{script_name}.log"
    else:
        log_file = logs_dir / "trading.log"
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler for specific script
    logger.add(
        str(log_file),
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # Also keep a general trading.log for overall system
    if script_name and script_name != "trading":
        logger.add(
            str(logs_dir / "trading.log"),
            level="INFO",  # Only important messages go to general log
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | [{extra[script]}] {name}:{function} - {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    # Add script name to context
    if script_name:
        logger = logger.bind(script=script_name)
    
    return logger

# Convenience functions for different scripts
def get_setup_logger():
    """Logger for database setup operations"""
    return setup_logger("setup_database")

def get_ingestion_logger():
    """Logger for data ingestion operations"""
    return setup_logger("data_ingestion")

def get_backtest_logger():
    """Logger for backtesting operations"""
    return setup_logger("backtest")

def get_trading_logger():
    """Logger for live trading operations"""
    return setup_logger("live_trading")

def get_validation_logger():
    """Logger for data validation operations"""
    return setup_logger("validation")

def get_test_logger():
    """Logger for test operations"""
    return setup_logger("test_ingestion")

# Default logger (backwards compatibility)
log = setup_logger()