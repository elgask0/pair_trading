#!/usr/bin/env python3
"""
Data ingestion script - WITH SELECTIVE OPTIONS AND FORCE OVERWRITE
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.ingestion import data_ingestion
from src.utils.logger import get_ingestion_logger
from config.settings import settings

log = get_ingestion_logger()

def _delete_orderbook_data(symbol: str):
    """Delete existing orderbook data for symbol"""
    from src.database.connection import db_manager
    from sqlalchemy import text
    
    with db_manager.get_session() as session:
        result = session.execute(text("DELETE FROM orderbook WHERE symbol = :symbol"), 
                               {'symbol': symbol})
        log.info(f"   🗑️ Deleted {result.rowcount:,} orderbook records")
        return result.rowcount

def _delete_ohlcv_data(symbol: str):
    """Delete existing OHLCV data for symbol"""
    from src.database.connection import db_manager
    from sqlalchemy import text
    
    with db_manager.get_session() as session:
        result = session.execute(text("DELETE FROM ohlcv WHERE symbol = :symbol"), 
                               {'symbol': symbol})
        log.info(f"   🗑️ Deleted {result.rowcount:,} OHLCV records")
        return result.rowcount

def _delete_funding_data(symbol: str):
    """Delete existing funding rate data for symbol"""
    from src.database.connection import db_manager
    from sqlalchemy import text
    
    with db_manager.get_session() as session:
        result = session.execute(text("DELETE FROM funding_rates WHERE symbol = :symbol"), 
                               {'symbol': symbol})
        log.info(f"   🗑️ Deleted {result.rowcount:,} funding rate records")
        return result.rowcount

def main():
    parser = argparse.ArgumentParser(description="Data ingestion with selective options and force overwrite")
    parser.add_argument("--symbol", type=str, help="Specific symbol to process")
    parser.add_argument("--funding-only", action="store_true", help="Only funding rates")
    parser.add_argument("--orderbook-only", action="store_true", help="Only orderbook data")
    parser.add_argument("--ohlcv-only", action="store_true", help="Only OHLCV data")
    parser.add_argument("--force-overwrite", action="store_true", help="Delete existing data first then re-ingest")
    
    args = parser.parse_args()
    
    # Determine data types to process
    if args.funding_only:
        data_types = ["funding"]
        log.info("🚀 Starting FUNDING-ONLY ingestion...")
    elif args.orderbook_only:
        data_types = ["orderbook"]
        log.info("🚀 Starting ORDERBOOK-ONLY ingestion...")
    elif args.ohlcv_only:
        data_types = ["ohlcv"]
        log.info("🚀 Starting OHLCV-ONLY ingestion...")
    else:
        data_types = ["ohlcv", "orderbook", "funding"]
        log.info("🚀 Starting COMPLETE data ingestion...")
    
    if args.force_overwrite:
        log.warning("⚠️ FORCE OVERWRITE MODE: Will delete existing data first!")
        log.warning("   This action cannot be undone!")
        
        # Ask for confirmation unless single symbol
        if not args.symbol:
            response = input("\nAre you sure you want to delete existing data for ALL symbols? (yes/no): ")
            if response.lower() != 'yes':
                log.info("Operation cancelled by user")
                return False
    
    try:
        # Get symbols to process
        if args.symbol:
            symbols = [args.symbol]
        else:
            active_pairs = settings.get_active_pairs()
            symbols = list(set([pair.symbol1 for pair in active_pairs] + [pair.symbol2 for pair in active_pairs]))
        
        log.info(f"Processing {len(symbols)} symbols: {[s.split('_')[-2] for s in symbols]}")
        
        success_count = 0
        total_deleted = {"ohlcv": 0, "orderbook": 0, "funding": 0}
        
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"PROCESSING {symbol}")
            log.info(f"{'='*60}")
            
            symbol_success = True
            
            # STEP 1: Delete existing data if force overwrite
            if args.force_overwrite:
                log.info(f"🗑️ Deleting existing data for {symbol}...")
                
                if "ohlcv" in data_types:
                    deleted = _delete_ohlcv_data(symbol)
                    total_deleted["ohlcv"] += deleted
                
                if "orderbook" in data_types:
                    deleted = _delete_orderbook_data(symbol)
                    total_deleted["orderbook"] += deleted
                
                if "funding" in data_types and "PERP_" in symbol:
                    deleted = _delete_funding_data(symbol)
                    total_deleted["funding"] += deleted
            
            # STEP 2: Update symbol info (lightweight operation)
            log.info(f"📋 Updating symbol info for {symbol}...")
            data_ingestion.update_symbol_info(symbol)
            
            # STEP 3: Ingest data based on selected types
            if "funding" in data_types:
                if "PERP_" in symbol:
                    log.info(f"💰 Ingesting funding rates for {symbol}...")
                    funding_results = data_ingestion.ingest_funding_rates([symbol])
                    if not funding_results.get(symbol, False):
                        log.error(f"❌ Funding rates failed for {symbol}")
                        symbol_success = False
                    else:
                        log.info(f"✅ Funding rates completed for {symbol}")
                else:
                    log.info(f"⏭️ Skipping funding rates for {symbol} (not a perpetual contract)")
            
            if "ohlcv" in data_types:
                log.info(f"📈 Ingesting OHLCV data for {symbol}...")
                if not data_ingestion.ingest_ohlcv_data(symbol):
                    log.error(f"❌ OHLCV failed for {symbol}")
                    symbol_success = False
                else:
                    log.info(f"✅ OHLCV completed for {symbol}")
            
            if "orderbook" in data_types:
                log.info(f"📊 Ingesting orderbook data for {symbol}...")
                if not data_ingestion.ingest_orderbook_data(symbol):
                    log.error(f"❌ Orderbook failed for {symbol}")
                    symbol_success = False
                else:
                    log.info(f"✅ Orderbook completed for {symbol}")
            
            # STEP 4: Summary for this symbol
            if symbol_success:
                success_count += 1
                log.info(f"🎉 {symbol} completed successfully!")
                
                # Quick stats
                from src.database.connection import db_manager
                from sqlalchemy import text
                with db_manager.get_session() as session:
                    if "ohlcv" in data_types:
                        result = session.execute(text("""
                            SELECT COUNT(*) as count 
                            FROM ohlcv WHERE symbol = :symbol
                        """), {'symbol': symbol}).fetchone()
                        log.info(f"   📈 OHLCV records: {result.count:,}")
                    
                    if "orderbook" in data_types:
                        result = session.execute(text("""
                            SELECT COUNT(*) as count 
                            FROM orderbook WHERE symbol = :symbol
                        """), {'symbol': symbol}).fetchone()
                        log.info(f"   📊 Orderbook records: {result.count:,}")
                    
                    if "funding" in data_types and "PERP_" in symbol:
                        result = session.execute(text("""
                            SELECT COUNT(*) as count 
                            FROM funding_rates WHERE symbol = :symbol
                        """), {'symbol': symbol}).fetchone()
                        log.info(f"   💰 Funding records: {result.count:,}")
            else:
                log.error(f"💥 {symbol} had some failures")
        
        # FINAL SUMMARY
        data_type_names = {
            "funding": "funding rates",
            "orderbook": "orderbook",
            "ohlcv": "OHLCV"
        }
        selected_types = [data_type_names[dt] for dt in data_types]
        
        log.info(f"\n🎉 Data ingestion completed!")
        log.info(f"📊 Data types processed: {', '.join(selected_types)}")
        log.info(f"✅ Successful symbols: {success_count}/{len(symbols)}")
        
        if args.force_overwrite and any(total_deleted.values()):
            log.info(f"🗑️ Total records deleted:")
            for data_type, count in total_deleted.items():
                if count > 0:
                    log.info(f"   {data_type_names[data_type]}: {count:,}")
        
        log.info(f"\nNext steps:")
        if "ohlcv" in data_types or "orderbook" in data_types:
            log.info(f"  - Run 'python scripts/validate_data.py' to validate data quality")
            log.info(f"  - Run 'python scripts/clean_data.py' to clean and mark quality")
        if "orderbook" in data_types:
            log.info(f"  - Run 'python scripts/calculate_markprices.py' to calculate mark prices")
        
        return success_count == len(symbols)
        
    except Exception as e:
        log.error(f"Data ingestion failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)