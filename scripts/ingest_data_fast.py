#!/usr/bin/env python3
"""
Script de ingesta ULTRA-RÁPIDO
10-20x más rápido que la versión secuencial
"""

import sys
import os
import asyncio
import argparse
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.ingestion_async import run_ultra_fast_ingestion
from config.settings import settings
from src.utils.logger import get_ingestion_logger

log = get_ingestion_logger()

async def main():
    parser = argparse.ArgumentParser(description="Ultra-fast data ingestion")
    parser.add_argument("--symbol", type=str, help="Specific symbol")
    parser.add_argument("--days", type=int, default=30, help="Days back to fetch")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent API calls")
    
    args = parser.parse_args()
    
    # Get symbols
    if args.symbol:
        symbols = [args.symbol]
    else:
        active_pairs = settings.get_active_pairs()
        symbols = list(set([p.symbol1 for p in active_pairs] + [p.symbol2 for p in active_pairs]))
    
    log.info(f"⚡ ULTRA-FAST INGESTION MODE ⚡")
    log.info(f"Symbols: {len(symbols)}")
    log.info(f"Days: {args.days}")
    log.info(f"Max concurrent: {args.max_concurrent}")
    
    # Run async ingestion
    start_time = datetime.now()
    
    result = await run_ultra_fast_ingestion(symbols, args.days)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if result['success']:
        log.info(f"\n🎉 INGESTION COMPLETED!")
        log.info(f"⏱️ Total time: {elapsed:.1f} seconds")
        log.info(f"📊 Records: {result['total_records']:,}")
        log.info(f"⚡ Speed: {result['total_records']/elapsed:.0f} records/second")
    else:
        log.error("Ingestion failed!")
    
    return result['success']

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)