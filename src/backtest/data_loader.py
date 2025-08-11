#!/usr/bin/env python3
"""
Data Loader - Carga y preparación de datos para backtest
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from typing import Optional

from src.database.connection import db_manager
from src.utils.logger import get_logger

log = get_logger()

class DataLoader:
    """Carga datos históricos desde la base de datos"""
    
    def load_ohlcv(self, symbol: str, start_date: datetime, end_date: datetime,
                   resample_minutes: int = 1) -> pd.DataFrame:
        """
        Cargar datos OHLCV y resamplear si es necesario
        """
        log.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        
        with db_manager.get_session() as session:
            query = text("""
                SELECT 
                    timestamp,
                    open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = :symbol
                  AND timestamp >= :start_date
                  AND timestamp <= :end_date
                ORDER BY timestamp
            """)
            
            df = pd.read_sql(
                query,
                session.bind,
                params={
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                },
                index_col='timestamp'
            )
            
            if df.empty:
                log.warning(f"No data found for {symbol}")
                return df
            
            # Resamplear si es necesario
            if resample_minutes > 1:
                df = self._resample_ohlcv(df, resample_minutes)
            
            log.info(f"Loaded {len(df)} rows for {symbol}")
            
            return df
    
    def _resample_ohlcv(self, df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """Resamplear datos OHLCV a frecuencia especificada"""
        
        rule = f'{minutes}min'
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        log.info(f"Resampled from {len(df)} to {len(resampled)} rows ({minutes}min)")
        
        return resampled
    
    def load_orderbook_snapshots(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, sample_rate: float = 0.1) -> pd.DataFrame:
        """
        Cargar snapshots de orderbook para análisis de liquidez
        """
        with db_manager.get_session() as session:
            sample_clause = f"TABLESAMPLE SYSTEM ({sample_rate * 100:.1f})"
            
            query = text(f"""
                SELECT 
                    timestamp,
                    bid1_price, bid1_size,
                    ask1_price, ask1_size
                FROM orderbook {sample_clause}
                WHERE symbol = :symbol
                  AND timestamp >= :start_date
                  AND timestamp <= :end_date
                ORDER BY timestamp
            """)
            
            df = pd.read_sql(
                query,
                session.bind,
                params={
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                },
                index_col='timestamp'
            )
            
            return df