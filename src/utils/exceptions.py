"""Custom exceptions for the trading system"""

class TradingSystemError(Exception):
    """Base exception for trading system"""
    pass

class DatabaseError(TradingSystemError):
    """Database related errors"""
    pass

class APIError(TradingSystemError):
    """API related errors"""
    pass

class DataValidationError(TradingSystemError):
    """Data validation errors"""
    pass

class TradingError(TradingSystemError):
    """Trading logic errors"""
    pass

class RiskManagementError(TradingSystemError):
    """Risk management violations"""
    pass

class InsufficientDataError(TradingSystemError):
    """Not enough data for analysis"""
    pass

class ConfigurationError(TradingSystemError):
    """Configuration related errors"""
    pass

class StrategyError(TradingSystemError):
    """Strategy execution errors"""
    pass

class OrderError(TradingSystemError):
    """Order management errors"""
    pass