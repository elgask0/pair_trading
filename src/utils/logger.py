# src/utils/logger.py
from loguru import logger
import sys
import os
from pathlib import Path
from datetime import datetime

def get_script_name():
    """Detectar nombre del script de forma simple"""
    try:
        if len(sys.argv) > 0:
            script_path = Path(sys.argv[0])
            script_name = script_path.stem
            if script_name not in ['python', 'python3', '__main__', '-c']:
                return script_name
        return "unknown_script"
    except:
        return "unknown_script"

def setup_logger(script_name: str = None):
    """Setup simple y funcional del logger con archivo timestamped"""
    
    # Remove existing handlers
    logger.remove()
    
    # Crear directorio logs
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Auto-detect script name
    if script_name is None:
        script_name = get_script_name()
    
    # Get log level with fallback
    try:
        from config.settings import settings
        log_level = getattr(settings, 'LOG_LEVEL', 'INFO')
    except:
        log_level = 'INFO'
    
    # Console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # 🔧 NUEVO: Archivo específico con timestamp para cada ejecución
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_log = logs_dir / f"{script_name}_{timestamp}.log"
    logger.add(
        str(timestamped_log),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # File handler específico del script (histórico)
    log_file = logs_dir / f"{script_name}.log"
    logger.add(
        str(log_file),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # General trading.log
    if script_name != "trading":
        logger.add(
            str(logs_dir / "trading.log"),
            level="INFO",
            format=f"{script_name} | {{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {{name}}:{{function}} - {{message}}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    logger.info(f"📝 Logging to: {timestamped_log}")
    return logger

def get_logger(script_name: str = None):
    """Get logger with automatic script name detection"""
    return setup_logger(script_name)

def get_logger(script_name: str = None):
    """Get logger with automatic script name detection"""
    return setup_logger(script_name)

# Funciones específicas para backwards compatibility
def get_setup_logger():
    """Logger for database setup operations"""
    return get_logger("setup_database")

def get_ingestion_logger():
    """Logger for data ingestion operations"""  
    return get_logger("data_ingestion")

def get_backtest_logger():
    """Logger for backtesting operations"""
    return get_logger("backtest")

def get_trading_logger():
    """Logger for live trading operations"""
    return get_logger("live_trading")

def get_validation_logger():
    """Logger for data validation operations"""
    return get_logger("data_validation")

def get_test_logger():
    """Logger for test operations"""
    return get_logger("test_operations")

def get_diagnose_logger():
    """Logger for diagnosis operations"""
    return get_logger("diagnose_data")

def get_analysis_logger():
    """Logger for analysis operations"""
    return get_logger("analysis")

def get_cleanup_logger():
    """Logger for cleanup operations"""
    return get_logger("data_cleanup")

# Default logger (backwards compatibility)
log = get_logger()