from loguru import logger as loguru_logger
import sys
import os
from pathlib import Path
from config.settings import settings
import inspect

def get_script_name():
    """Detectar automáticamente el nombre del script que llama al logger"""
    # Obtener el frame del script que llama (saltando logger.py)
    frame = inspect.currentframe()
    try:
        # Ir hacia atrás en el stack hasta encontrar un script fuera de logger.py
        caller_frame = frame.f_back.f_back  # Saltar setup_logger() y get_*_logger()
        while caller_frame:
            filename = caller_frame.f_code.co_filename
            if not filename.endswith('logger.py'):
                script_path = Path(filename)
                return script_path.stem  # Nombre sin extensión
            caller_frame = caller_frame.f_back
        return "unknown_script"
    finally:
        del frame

def setup_logger(script_name: str = None):
    """Setup structured logging with loguru for specific scripts"""
    
    # Remove default handler
    loguru_logger.remove()
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Auto-detect script name if not provided
    if script_name is None:
        script_name = get_script_name()
    
    # Determine log file name based on script
    log_file = logs_dir / f"{script_name}.log"
    
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
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {script_name} - {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    return loguru_logger

# Nueva función principal que auto-detecta
def get_logger():
    """Get logger with automatic script name detection"""
    return setup_logger()

# Mantener funciones específicas para backwards compatibility pero que usen auto-detección
def get_setup_logger():
    """Logger for database setup operations"""
    return get_logger()

def get_ingestion_logger():
    """Logger for data ingestion operations"""  
    return get_logger()

def get_backtest_logger():
    """Logger for backtesting operations"""
    return get_logger()

def get_trading_logger():
    """Logger for live trading operations"""
    return get_logger()

def get_validation_logger():
    """Logger for data validation operations"""
    return get_logger()

def get_test_logger():
    """Logger for test operations"""
    return get_logger()

# Default logger (backwards compatibility)
log = get_logger()