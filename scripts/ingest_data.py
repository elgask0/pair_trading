#!/usr/bin/env python3
"""
Data ingestion script
Downloads historical data from multiple sources for configured trading pairs
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ingestion import data_ingestion
from config.settings import settings
from src.utils.logger import get_ingestion_logger

log = get_ingestion_logger()

def main():
    parser = argparse.ArgumentParser(description="Ingest historical data from multiple sources")
    parser.add_argument("--symbol", type=str, help="Specific symbol to ingest")
    parser.add_argument("--funding-only", action="store_true", help="Ingest only funding rates for perpetual symbols")
    parser.add_argument("--show-summary", action="store_true", help="Show data summary after ingestion")
    
    args = parser.parse_args()
    
    log.info("Starting data ingestion process...")
    
    try:
        if args.symbol:
            # Ingest specific symbol
            if args.funding_only and "PERP_" in args.symbol:
                # Ingest only funding rates for the perpetual symbol
                results = data_ingestion.ingest_funding_rates([args.symbol])
                success = results.get(args.symbol, False)
                
                if success:
                    log.info(f"Successfully ingested funding rates for {args.symbol}")
                else:
                    log.error(f"Failed to ingest funding rates for {args.symbol}")
                    return False
            elif args.funding_only:
                log.warning(f"Symbol {args.symbol} is not a perpetual contract (no PERP_ in name)")
                log.info("Use a symbol like MEXCFTS_PERP_SPX_USDT for funding rates")
                return False
            else:
                # Complete ingestion for the symbol
                success = data_ingestion.ingest_symbol_data(args.symbol, funding_only=False)
                
                if success:
                    log.info(f"Successfully ingested all data types for {args.symbol}")
                else:
                    log.error(f"Failed to ingest data for {args.symbol}")
                    return False
        else:
            # Ingest all symbols
            if args.funding_only:
                # Get all perpetual symbols for funding rates
                all_symbols = settings.get_all_symbols()
                perp_symbols = [s for s in all_symbols if "PERP_" in s]
                
                if not perp_symbols:
                    log.warning("No perpetual symbols found for funding rates ingestion")
                    log.info("Check your config/symbols.yaml file for symbols with PERP_ in the name")
                    return True
                
                log.info(f"Ingesting funding rates for {len(perp_symbols)} perpetual symbols: {perp_symbols}")
                success = data_ingestion.ingest_all_symbols(funding_only=True)
                
                if success:
                    log.info("Successfully ingested funding rates for all perpetual symbols")
                else:
                    log.warning("Some perpetual symbols failed during funding rates ingestion")
            else:
                # Complete ingestion for all symbols
                log.info("Starting complete data ingestion (OHLCV, orderbook, funding rates)")
                success = data_ingestion.ingest_all_symbols(funding_only=False)
                
                if success:
                    log.info("Successfully ingested all data types for all symbols")
                else:
                    log.warning("Some symbols failed during complete ingestion")
        
        # Show data summary
        if args.show_summary or args.funding_only:
            log.info("\n=== Data Summary ===")
            
            if args.funding_only:
                # Show funding rates summary
                symbols_to_check = [args.symbol] if args.symbol else [s for s in settings.get_all_symbols() if "PERP_" in s]
                
                log.info("Funding Rates Data:")
                for symbol in symbols_to_check:
                    min_date, max_date = data_ingestion.get_funding_data_range(symbol)
                    if min_date and max_date:
                        log.info(f"  {symbol}: {min_date} to {max_date}")
                    else:
                        log.warning(f"  {symbol}: No funding rate data found")
            else:
                # Show OHLCV summary
                symbols_to_check = [args.symbol] if args.symbol else settings.get_all_symbols()
                
                log.info("OHLCV Data:")
                for symbol in symbols_to_check:
                    min_date, max_date = data_ingestion.get_symbol_data_range(symbol)
                    if min_date and max_date:
                        log.info(f"  {symbol}: {min_date} to {max_date}")
                    else:
                        log.warning(f"  {symbol}: No OHLCV data found")
                
                # Also show funding rates for perpetual symbols
                perp_symbols = [s for s in symbols_to_check if "PERP_" in s]
                if perp_symbols:
                    log.info("Funding Rates Data:")
                    for symbol in perp_symbols:
                        min_date, max_date = data_ingestion.get_funding_data_range(symbol)
                        if min_date and max_date:
                            log.info(f"  {symbol}: {min_date} to {max_date}")
                        else:
                            log.warning(f"  {symbol}: No funding rate data found")
        
        # Final success message
        log.info("\n=== Ingestion Complete ===")
        if args.funding_only:
            log.info("✅ Funding rates ingestion completed")
            log.info("Next steps:")
            log.info("  - Run 'python scripts/clean_data.py' to clean and validate data")
            log.info("  - Run 'python scripts/analyze_data.py' to analyze funding patterns")
        else:
            log.info("✅ Complete data ingestion finished")
            log.info("Next steps:")
            log.info("  - Run 'python scripts/validate_data.py' to validate data quality")
            log.info("  - Run 'python scripts/clean_data.py' to clean data")
        
        return True
        
    except Exception as e:
        log.error(f"Data ingestion failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)