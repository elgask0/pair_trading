#!/usr/bin/env python3
"""
Database setup script
Creates tables and initializes the database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.database.migrations import (
    create_all_tables, 
    add_data_quality_columns, 
    check_data_quality_schema, 
    create_funding_rates_table, 
    check_funding_rates_schema,
    run_all_migrations
)
from src.database.models import Symbol, PairConfiguration
from config.settings import settings
from src.utils.logger import get_setup_logger

log = get_setup_logger()

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
                # Parse symbol (assuming format like "MEXCFTS_PERP_SPX_USDT")
                if "PERP_" in symbol:
                    parts = symbol.split('_')
                    if len(parts) >= 4:  # MEXCFTS_PERP_SPX_USDT
                        base_asset = parts[2]
                        quote_asset = parts[3]
                    else:
                        base_asset = symbol.split('_')[0]
                        quote_asset = "USDT"
                else:
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

def setup_database_schema():
    """Setup complete database schema including all tables"""
    log.info("Setting up database schema...")
    
    try:
        # Run all migrations (this includes base tables, data quality, and funding rates)
        success = run_all_migrations()
        
        if success:
            log.info("Database schema setup completed successfully")
            return True
        else:
            log.error("Database schema setup failed")
            return False
        
    except Exception as e:
        log.error(f"Database schema setup failed: {e}")
        return False

def verify_schema():
    """Verify that all required tables and columns exist"""
    log.info("Verifying database schema...")
    
    try:
        # Check funding rates schema
        funding_ok = check_funding_rates_schema()
        
        # Check data quality schema
        quality_ok = check_data_quality_schema()
        
        if funding_ok and quality_ok:
            log.info("✅ All schema components verified successfully")
            return True
        else:
            log.warning("⚠️ Some schema components are missing")
            if not funding_ok:
                log.warning("  - Funding rates table/schema missing")
            if not quality_ok:
                log.warning("  - Data quality columns missing")
            return False
        
    except Exception as e:
        log.error(f"Schema verification failed: {e}")
        return False

def main():
    """Main setup function"""
    log.info("Starting database setup...")
    
    # Test connection
    if not db_manager.test_connection():
        log.error("Database connection failed. Check your configuration.")
        log.error("Make sure PostgreSQL is running and .env file is configured correctly")
        return False
    
    try:
        # Setup database schema
        if not setup_database_schema():
            log.error("Database schema setup failed")
            return False
        
        # Verify schema
        if not verify_schema():
            log.error("Schema verification failed")
            return False
        
        # Setup initial data
        log.info("Setting up initial data...")
        setup_symbols()
        setup_pair_configurations()
        
        log.info("✅ Database setup completed successfully!")
        log.info("\nNext steps:")
        log.info("  1. Run 'python scripts/ingest_data.py --funding-only' to ingest funding rates")
        log.info("  2. Run 'python scripts/validate_data.py' to validate data quality")
        log.info("  3. Run 'python scripts/clean_data.py' to clean data")
        
        return True
        
    except Exception as e:
        log.error(f"Database setup failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Database setup completed successfully!")
    else:
        print("\n❌ Database setup failed. Check the logs above.")
    
    sys.exit(0 if success else 1)