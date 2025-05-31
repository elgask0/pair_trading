#!/usr/bin/env python3
"""
Data ingestion script
Downloads historical data from CoinAPI for configured trading pairs
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ingestion import data_ingestion
from src.utils.logger import log
from config.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Ingest historical data from CoinAPI")
    parser.add_argument("--symbol", type=str, help="Specific symbol to ingest")
    
    args = parser.parse_args()
    
    log.info("Starting data ingestion process...")
    
    try:
        if args.symbol:
            # Ingest specific symbol
            success = data_ingestion.ingest_symbol_data(args.symbol)
            if success:
                log.info(f"Successfully ingested data for {args.symbol}")
            else:
                log.error(f"Failed to ingest data for {args.symbol}")
                return False
        else:
            # Ingest all symbols
            success = data_ingestion.ingest_all_symbols()
            if success:
                log.info("Successfully ingested data for all symbols")
            else:
                log.error("Some symbols failed during ingestion")
                return False
        
        # Show data summary
        log.info("\n=== Data Summary ===")
        for pair_config in settings.get_active_pairs():
            for symbol in [pair_config.symbol1, pair_config.symbol2]:
                min_date, max_date = data_ingestion.get_symbol_data_range(symbol)
                if min_date and max_date:
                    log.info(f"{symbol}: {min_date} to {max_date}")
                else:
                    log.warning(f"{symbol}: No data found")
        
        return True
        
    except Exception as e:
        log.error(f"Data ingestion failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)