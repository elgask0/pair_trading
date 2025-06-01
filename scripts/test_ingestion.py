#!/usr/bin/env python3
"""
Test script para probar la ingesta de un día específico
"""

import sys
import os
import argparse
from datetime import datetime, date
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.coinapi_client import coinapi_client
from src.data.ingestion import data_ingestion
from config.settings import settings
from src.utils.logger import get_setup_logger

log = get_setup_logger()

def test_single_day_ohlcv(symbol: str, test_date: str):
    """Test OHLCV ingestion for a single day"""
    log.info(f"Testing OHLCV ingestion for {symbol} on {test_date}")
    
    try:
        df = coinapi_client.get_ohlcv_for_date(symbol, test_date)
        
        if df.empty:
            log.warning(f"No OHLCV data for {symbol} on {test_date}")
            return False
        
        log.info(f"SUCCESS: Retrieved {len(df)} OHLCV records for {symbol} on {test_date}")
        log.info(f"Time range: {df.index.min()} to {df.index.max()}")
        log.info(f"Columns: {list(df.columns)}")
        
        # Show sample data
        if len(df) > 0:
            log.info(f"Sample data:\n{df.head(3)}")
        
        return True
        
    except Exception as e:
        log.error(f"FAILED: Error testing OHLCV for {symbol} on {test_date}: {e}")
        return False

def test_single_day_orderbook(symbol: str, test_date: str):
    """Test orderbook ingestion for a single day"""
    log.info(f"Testing orderbook ingestion for {symbol} on {test_date}")
    
    try:
        df = coinapi_client.get_orderbook_for_date(symbol, test_date)
        
        if df.empty:
            log.warning(f"No orderbook data for {symbol} on {test_date}")
            return False
        
        log.info(f"SUCCESS: Retrieved {len(df)} orderbook snapshots for {symbol} on {test_date}")
        log.info(f"Time range: {df.index.min()} to {df.index.max()}")
        log.info(f"Columns: {list(df.columns)}")
        
        # Show sample data
        if len(df) > 0:
            log.info(f"Sample data:\n{df.head(2)}")
        
        return True
        
    except Exception as e:
        log.error(f"FAILED: Error testing orderbook for {symbol} on {test_date}: {e}")
        return False

def test_symbol_info(symbol: str):
    """Test symbol info retrieval"""
    log.info(f"Testing symbol info for {symbol}")
    
    try:
        info = coinapi_client.get_symbol_info(symbol)
        log.info(f"SUCCESS: Retrieved symbol info for {symbol}")
        log.info(f"Exchange: {info.get('exchange_id')}")
        log.info(f"Type: {info.get('symbol_type')}")
        log.info(f"Data start: {info.get('data_start')}")
        log.info(f"Data end: {info.get('data_end')}")
        
        return True
        
    except Exception as e:
        log.error(f"FAILED: Error getting symbol info for {symbol}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test data ingestion for specific symbol/date")
    parser.add_argument("--symbol", type=str, help="Symbol to test (default: first active symbol)")
    parser.add_argument("--date", type=str, help="Date to test (YYYY-MM-DD, default: yesterday)")
    parser.add_argument("--test-ohlcv", action="store_true", help="Test OHLCV ingestion")
    parser.add_argument("--test-orderbook", action="store_true", help="Test orderbook ingestion")
    parser.add_argument("--test-symbol-info", action="store_true", help="Test symbol info")
    parser.add_argument("--test-all", action="store_true", help="Test everything")
    
    args = parser.parse_args()
    
    # Default symbol
    if not args.symbol:
        active_pairs = settings.get_active_pairs()
        if active_pairs:
            args.symbol = active_pairs[0].symbol1
        else:
            log.error("No active pairs configured")
            return False
    
    # Default date (yesterday)
    if not args.date:
        yesterday = date.today() - timedelta(days=1)
        args.date = yesterday.isoformat()
    
    # Default to test all if nothing specified
    if not any([args.test_ohlcv, args.test_orderbook, args.test_symbol_info]):
        args.test_all = True
    
    log.info(f"Testing ingestion for symbol: {args.symbol}, date: {args.date}")
    
    success_count = 0
    total_tests = 0
    
    # Test symbol info
    if args.test_symbol_info or args.test_all:
        total_tests += 1
        if test_symbol_info(args.symbol):
            success_count += 1
    
    # Test OHLCV
    if args.test_ohlcv or args.test_all:
        total_tests += 1
        if test_single_day_ohlcv(args.symbol, args.date):
            success_count += 1
    
    # Test orderbook
    if args.test_orderbook or args.test_all:
        total_tests += 1
        if test_single_day_orderbook(args.symbol, args.date):
            success_count += 1
    
    log.info(f"\n=== Test Results ===")
    log.info(f"Passed: {success_count}/{total_tests}")
    log.info(f"Symbol: {args.symbol}")
    log.info(f"Date: {args.date}")
    
    return success_count == total_tests

if __name__ == "__main__":
    from datetime import timedelta
    success = main()
    sys.exit(0 if success else 1)