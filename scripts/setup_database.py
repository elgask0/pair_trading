#!/usr/bin/env python3
"""
Database setup script
Creates tables and initializes the database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.database.migrations import create_all_tables
from src.database.models import Symbol, PairConfiguration
from src.utils.logger import log
from config.settings import settings

def setup_symbols():
    """Initialize symbols table with trading pairs"""
    with db_manager.get_session() as session:
        # Get all unique symbols from pair configurations
        symbols_to_add = set()
        for pair_config in settings.get_trading_pairs():
            symbols_to_add.add(pair_config.symbol1)
            symbols_to_add.add(pair_config.symbol2)
        
        for symbol in symbols_to_add:
            # Check if symbol already exists
            existing = session.query(Symbol).filter(Symbol.symbol == symbol).first()
            if not existing:
                # Parse symbol (assuming format like "GIGA_USDT")
                parts = symbol.split('_')
                base_asset = parts[0] if len(parts) > 1 else symbol
                quote_asset = parts[1] if len(parts) > 1 else "USDT"
                
                new_symbol = Symbol(
                    symbol=symbol,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    is_active=True
                )
                session.add(new_symbol)
                log.info(f"Added symbol: {symbol}")
            else:
                log.info(f"Symbol already exists: {symbol}")

def setup_pair_configurations():
    """Initialize pair configurations table"""
    with db_manager.get_session() as session:
        for pair_config in settings.get_trading_pairs():
            # Check if configuration already exists
            existing = session.query(PairConfiguration).filter(
                PairConfiguration.pair_name == pair_config.pair_name
            ).first()
            
            if not existing:
                new_config = PairConfiguration(
                    pair_name=pair_config.pair_name,
                    symbol1=pair_config.symbol1,
                    symbol2=pair_config.symbol2,
                    entry_zscore=pair_config.entry_zscore,
                    exit_zscore=pair_config.exit_zscore,
                    stop_loss_zscore=pair_config.stop_loss_zscore,
                    position_size_pct=pair_config.position_size_pct,
                    correlation_window=pair_config.correlation_window,
                    cointegration_window=pair_config.cointegration_window,
                    zscore_window=pair_config.zscore_window,
                    min_correlation=pair_config.min_correlation,
                    is_active=pair_config.is_active
                )
                session.add(new_config)
                log.info(f"Added pair configuration: {pair_config.pair_name}")
            else:
                log.info(f"Pair configuration already exists: {pair_config.pair_name}")

def main():
    """Main setup function"""
    log.info("Starting database setup...")
    
    # Test connection
    if not db_manager.test_connection():
        log.error("Database connection failed. Check your configuration.")
        return False
    
    try:
        # Create tables
        create_all_tables()
        
        # Setup initial data
        setup_symbols()
        setup_pair_configurations()
        
        log.info("Database setup completed successfully!")
        return True
        
    except Exception as e:
        log.error(f"Database setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)