#!/usr/bin/env python3
"""
Data ingestion script - CON PARÁMETROS SELECTIVOS Y SOBREESCRITURA
FIXED: days_back como None por defecto para usar todos los datos en overwrite mode
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

def main():
    parser = argparse.ArgumentParser(description="Data ingestion with selective options and overwrite")
    parser.add_argument("--symbol", type=str, help="Specific symbol to process")
    parser.add_argument("--funding-only", action="store_true", help="Only funding rates")
    parser.add_argument("--orderbook-only", action="store_true", help="Only orderbook data")
    parser.add_argument("--ohlcv-only", action="store_true", help="Only OHLCV data")
    parser.add_argument("--force-overwrite", action="store_true", help="Overwrite existing data")
    # FIXED: days_back default None para usar todos los datos disponibles en overwrite mode
    parser.add_argument("--days-back", type=int, default=None, 
                       help="Days back to fetch. If not specified in overwrite mode, fetches ALL available data")
    
    args = parser.parse_args()
    
    # Determine data types to process
    data_types = None  # None = all types
    
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
        data_types = None  # All types
        log.info("🚀 Starting COMPLETE data ingestion...")
    
    # Overwrite mode warning
    if args.force_overwrite:
        if args.days_back is None:
            log.warning("⚠️ FORCE OVERWRITE MODE: Will fetch ALL AVAILABLE DATA and overwrite!")
        else:
            log.warning(f"⚠️ FORCE OVERWRITE MODE: Will fetch last {args.days_back} days and overwrite!")
        log.warning("   This action cannot be undone!")
        
        # Ask for confirmation unless single symbol
        if not args.symbol:
            if args.days_back is None:
                response = input("\nAre you sure you want to fetch ALL available data and overwrite for ALL symbols? (yes/no): ")
            else:
                response = input(f"\nAre you sure you want to overwrite last {args.days_back} days for ALL symbols? (yes/no): ")
            
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
        
        if not symbols:
            log.error("No symbols to process")
            return False
        
        symbol_names = [s.split('_')[-2] if '_' in s else s for s in symbols]
        log.info(f"Processing {len(symbols)} symbols: {symbol_names}")
        
        # FIXED: Pasar days_back como None si no se especifica para usar todos los datos
        run_ingestion_args = {
            'symbols': symbols,
            'data_types': data_types,
            'overwrite': args.force_overwrite,
            'days_back': args.days_back  # Esto será None si no se especifica
        }
        
        log.info(f"Ingestion parameters: overwrite={args.force_overwrite}, days_back={args.days_back}")
        
        # Run ingestion with new unified function
        results = data_ingestion.ingest_data(**run_ingestion_args)
        
        # Analyze results
        total_symbols = len(results)
        successful_symbols = 0
        
        for symbol, symbol_results in results.items():
            symbol_success = all(symbol_results.values())
            if symbol_success:
                successful_symbols += 1
        
        # Final summary
        log.info(f"\n🎉 Data ingestion completed!")
        log.info(f"✅ Successful symbols: {successful_symbols}/{total_symbols}")
        
        if data_types:
            log.info(f"📊 Data types processed: {data_types}")
        else:
            log.info(f"📊 Data types processed: all (ohlcv, orderbook, funding)")
        
        if args.force_overwrite:
            if args.days_back is None:
                log.info(f"🔄 Mode: OVERWRITE (ALL AVAILABLE DATA)")
            else:
                log.info(f"🔄 Mode: OVERWRITE ({args.days_back} days back)")
        else:
            log.info(f"🔄 Mode: INCREMENTAL")
        
        log.info(f"\nNext steps:")
        if not data_types or 'ohlcv' in data_types or 'orderbook' in data_types:
            log.info(f"  - Run 'python scripts/validate_data.py' to validate data quality")
            log.info(f"  - Run 'python scripts/clean_data.py' to clean and mark quality")
        if not data_types or 'orderbook' in data_types:
            log.info(f"  - Run 'python scripts/calculate_markprices.py' to calculate mark prices")
        
        return successful_symbols == total_symbols
        
    except Exception as e:
        log.error(f"Data ingestion failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)