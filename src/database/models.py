from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class SymbolInfo(Base):
    __tablename__ = "symbol_info"
    
    id = Column(Integer, primary_key=True)
    symbol_id = Column(String(50), unique=True, nullable=False)
    exchange_id = Column(String(20))
    symbol_type = Column(String(20))
    asset_id_base = Column(String(20))
    asset_id_quote = Column(String(20))
    data_start = Column(DateTime)
    data_end = Column(DateTime)
    data_quote_start = Column(DateTime)
    data_quote_end = Column(DateTime)
    data_orderbook_start = Column(DateTime)
    data_orderbook_end = Column(DateTime)
    data_trade_start = Column(DateTime)
    data_trade_end = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class Symbol(Base):
    __tablename__ = "symbols"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), unique=True, nullable=False)
    base_asset = Column(String(20), nullable=False)
    quote_asset = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class OHLCV(Base):
    __tablename__ = "ohlcv"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp', unique=True),
        Index('idx_timestamp', 'timestamp'),
    )

class Orderbook(Base):
    __tablename__ = "orderbook"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # 10 niveles de bid
    bid1_price = Column(Float)
    bid1_size = Column(Float)
    bid2_price = Column(Float)
    bid2_size = Column(Float)
    bid3_price = Column(Float)
    bid3_size = Column(Float)
    bid4_price = Column(Float)
    bid4_size = Column(Float)
    bid5_price = Column(Float)
    bid5_size = Column(Float)
    bid6_price = Column(Float)
    bid6_size = Column(Float)
    bid7_price = Column(Float)
    bid7_size = Column(Float)
    bid8_price = Column(Float)
    bid8_size = Column(Float)
    bid9_price = Column(Float)
    bid9_size = Column(Float)
    bid10_price = Column(Float)
    bid10_size = Column(Float)
    
    # 10 niveles de ask
    ask1_price = Column(Float)
    ask1_size = Column(Float)
    ask2_price = Column(Float)
    ask2_size = Column(Float)
    ask3_price = Column(Float)
    ask3_size = Column(Float)
    ask4_price = Column(Float)
    ask4_size = Column(Float)
    ask5_price = Column(Float)
    ask5_size = Column(Float)
    ask6_price = Column(Float)
    ask6_size = Column(Float)
    ask7_price = Column(Float)
    ask7_size = Column(Float)
    ask8_price = Column(Float)
    ask8_size = Column(Float)
    ask9_price = Column(Float)
    ask9_size = Column(Float)
    ask10_price = Column(Float)
    ask10_size = Column(Float)
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_orderbook_symbol_timestamp', 'symbol', 'timestamp', unique=True),
    )

class PairConfiguration(Base):
    __tablename__ = "pair_configurations"
    
    id = Column(Integer, primary_key=True)
    pair_name = Column(String(100), unique=True, nullable=False)
    symbol1 = Column(String(50), nullable=False)
    symbol2 = Column(String(50), nullable=False)
    
    # Trading signals
    entry_zscore = Column(Float, nullable=False, default=2.0)
    exit_zscore = Column(Float, nullable=False, default=0.5)
    stop_loss_zscore = Column(Float, nullable=False, default=3.0)
    
    # Position sizing
    position_size_pct = Column(Float, nullable=False, default=0.1)
    
    # Technical analysis windows (in minutes)
    correlation_window = Column(Integer, nullable=False, default=60)
    cointegration_window = Column(Integer, nullable=False, default=1440)
    zscore_window = Column(Integer, nullable=False, default=60)
    
    # Validation criteria
    min_correlation = Column(Float, nullable=False, default=0.7)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    notes = Column(Text)
    
    __table_args__ = (
        Index('idx_pair_active', 'pair_name', 'is_active'),
    )

class PairMetrics(Base):
    __tablename__ = "pair_metrics"
    
    id = Column(Integer, primary_key=True)
    pair_name = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # Spread metrics
    spread = Column(Float)
    zscore = Column(Float)
    
    # Statistical metrics
    correlation = Column(Float)
    cointegration_pvalue = Column(Float)
    beta = Column(Float)
    alpha = Column(Float)
    r_squared = Column(Float)
    
    # Additional metrics
    volatility_ratio = Column(Float)
    half_life = Column(Float)
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_pair_timestamp', 'pair_name', 'timestamp'),
    )

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False, default="pair_trading")
    pair_name = Column(String(100), nullable=False)
    
    # Trade details
    side = Column(String(10), nullable=False)  # "long_spread", "short_spread"
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    
    # Entry data
    entry_spread = Column(Float, nullable=False)
    entry_zscore = Column(Float, nullable=False)
    entry_price_1 = Column(Float, nullable=False)
    entry_price_2 = Column(Float, nullable=False)
    
    # Exit data
    exit_spread = Column(Float)
    exit_zscore = Column(Float)
    exit_price_1 = Column(Float)
    exit_price_2 = Column(Float)
    
    # Position sizing
    quantity_1 = Column(Float, nullable=False)
    quantity_2 = Column(Float, nullable=False)
    
    # P&L
    pnl = Column(Float)
    fees = Column(Float, default=0.0)
    
    # Status and reason
    status = Column(String(20), default="open")  # open, closed, cancelled
    exit_reason = Column(String(50))  # signal, stop_loss, take_profit, manual
    
    # Metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_strategy_status', 'strategy_name', 'status'),
        Index('idx_entry_time', 'entry_time'),
    )

class SystemState(Base):
    __tablename__ = "system_state"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(50), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())