from loguru import logger as loguru_logger
import sys
import os
from pathlib import Path
from config.settings import settings

def setup_logger(script_name: str = None):
    """Setup structured logging with loguru for specific scripts"""
    
    # Remove default handler
    loguru_logger.remove()
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Determine log file name based on script
    if script_name:
        log_file = logs_dir / f"{script_name}.log"
    else:
        log_file = logs_dir / "trading.log"
    
    # Console handler with colors
    loguru_logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler for specific script
    loguru_logger.add(
        str(log_file),
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # Always add to general trading.log with simple format
    if script_name != "trading":  # Evitar duplicados
        loguru_logger.add(
            str(logs_dir / "trading.log"),
            level="INFO",  # Only important messages go to general log
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    return loguru_logger

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