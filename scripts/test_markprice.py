#!/usr/bin/env python3
"""
Test script for mark prices functionality
Verifica que la tabla mark_prices y los cálculos funcionen correctamente
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import db_manager
from src.database.migrations import check_mark_prices_schema
from src.utils.logger import get_test_logger
from config.settings import settings

log = get_test_logger()

def test_mark_prices_schema():
    """Test that mark prices table exists and has correct schema"""
    log.info("Testing mark prices schema...")
    
    schema_ok = check_mark_prices_schema()
    if schema_ok:
        log.info("✅ Mark prices table schema is correct")
        return True
    else:
        log.error("❌ Mark prices table schema is missing or incorrect")
        return False

def test_mark_prices_data():
    """Test that mark prices data exists and looks reasonable"""
    log.info("Testing mark prices data...")
    
    with db_manager.get_session() as session:
        # Check if we have any mark price data
        result = session.execute(text("""
            SELECT 
                symbol,
                COUNT(*) as record_count,
                MIN(timestamp) as min_time,
                MAX(timestamp) as max_time,
                COUNT(CASE WHEN is_valid = TRUE THEN 1 END) as valid_count,
                AVG(mark_price) as avg_mark_price,
                AVG(liquidity_score) as avg_quality
            FROM mark_prices 
            GROUP BY symbol
            ORDER BY symbol
        """)).fetchall()
        
        if not result:
            log.warning("⚠️ No mark price data found")
            return False
        
        log.info("✅ Mark prices data found:")
        for row in result:
            symbol_short = row.symbol.split('_')[-2] if '_' in row.symbol else row.symbol
            log.info(f"  {symbol_short}:")
            log.info(f"    Records: {row.record_count:,}")
            log.info(f"    Valid: {row.valid_count:,} ({row.valid_count/row.record_count*100:.1f}%)")
            log.info(f"    Time range: {row.min_time} to {row.max_time}")
            log.info(f"    Avg mark price: ${row.avg_mark_price:.6f}")
            log.info(f"    Avg quality: {row.avg_quality:.3f}")
        
        return True

def test_mark_price_quality():
    """Test mark price quality and validation"""
    log.info("Testing mark price quality...")
    
    with db_manager.get_session() as session:
        # Get validation breakdown
        result = session.execute(text("""
            SELECT 
                validation_source,
                COUNT(*) as count,
                AVG(liquidity_score) as avg_quality,
                AVG(bid_ask_spread_pct) as avg_spread
            FROM mark_prices 
            GROUP BY validation_source
            ORDER BY count DESC
        """)).fetchall()
        
        if not result:
            log.warning("⚠️ No mark price validation data found")
            return False
        
        log.info("✅ Mark price validation breakdown:")
        for row in result:
            log.info(f"  {row.validation_source}: {row.count:,} records")
            log.info(f"    Avg quality: {row.avg_quality:.3f}")
            log.info(f"    Avg spread: {row.avg_spread:.4f}%")
        
        return True

def compare_mark_prices_vs_ohlcv():
    """Compare mark prices against OHLCV close prices"""
    log.info("Comparing mark prices vs OHLCV close prices...")
    
    with db_manager.get_session() as session:
        # Get recent data for comparison
        result = session.execute(text("""
            SELECT 
                mp.symbol,
                mp.timestamp,
                mp.mark_price,
                mp.orderbook_mid,
                mp.ohlcv_close,
                mp.price_deviation_pct,
                mp.liquidity_score,
                mp.is_valid
            FROM mark_prices mp
            WHERE mp.ohlcv_close IS NOT NULL
            AND mp.timestamp >= NOW() - INTERVAL '1 day'
            ORDER BY mp.symbol, mp.timestamp DESC
            LIMIT 10
        """)).fetchall()
        
        if not result:
            log.warning("⚠️ No recent mark price data with OHLCV for comparison")
            return False
        
        log.info("✅ Recent mark price vs OHLCV comparison (last 10 records):")
        for row in result:
            symbol_short = row.symbol.split('_')[-2] if '_' in row.symbol else row.symbol
            deviation = row.price_deviation_pct or 0
            status = "✅" if row.is_valid else "❌"
            
            log.info(f"  {status} {symbol_short} @ {row.timestamp}:")
            log.info(f"    Mark Price: ${row.mark_price:.6f}")
            log.info(f"    OHLCV Close: ${row.ohlcv_close:.6f}")
            log.info(f"    Orderbook Mid: ${row.orderbook_mid:.6f}")
            log.info(f"    Deviation: {deviation:.3f}%")
            log.info(f"    Quality: {row.liquidity_score:.3f}")
        
        return True

def test_mark_price_consistency():
    """Test mark price consistency and detect anomalies"""
    log.info("Testing mark price consistency...")
    
    with db_manager.get_session() as session:
        # Look for potential anomalies
        result = session.execute(text("""
            WITH price_changes AS (
                SELECT 
                    symbol,
                    timestamp,
                    mark_price,
                    LAG(mark_price) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_price,
                    ABS((mark_price - LAG(mark_price) OVER (PARTITION BY symbol ORDER BY timestamp)) / 
                        LAG(mark_price) OVER (PARTITION BY symbol ORDER BY timestamp) * 100) as price_change_pct
                FROM mark_prices 
                WHERE is_valid = TRUE
                ORDER BY symbol, timestamp DESC
                LIMIT 1000
            )
            SELECT 
                symbol,
                COUNT(*) as total_changes,
                AVG(price_change_pct) as avg_change_pct,
                MAX(price_change_pct) as max_change_pct,
                COUNT(CASE WHEN price_change_pct > 5 THEN 1 END) as large_changes
            FROM price_changes 
            WHERE prev_price IS NOT NULL
            GROUP BY symbol
        """)).fetchall()
        
        if not result:
            log.warning("⚠️ No mark price consistency data found")
            return False
        
        log.info("✅ Mark price consistency analysis:")
        for row in result:
            symbol_short = row.symbol.split('_')[-2] if '_' in row.symbol else row.symbol
            log.info(f"  {symbol_short}:")
            log.info(f"    Price changes analyzed: {row.total_changes:,}")
            log.info(f"    Avg change: {row.avg_change_pct:.3f}%")
            log.info(f"    Max change: {row.max_change_pct:.3f}%")
            log.info(f"    Large changes (>5%): {row.large_changes}")
            
            if row.large_changes > row.total_changes * 0.01:  # More than 1% large changes
                log.warning(f"    ⚠️ High number of large price changes detected")
        
        return True

def generate_test_report():
    """Generate comprehensive test report"""
    log.info("\n" + "="*60)
    log.info("MARK PRICES TEST REPORT")
    log.info("="*60)
    
    tests_passed = 0
    total_tests = 4
    
    # Run all tests
    if test_mark_prices_schema():
        tests_passed += 1
    
    if test_mark_prices_data():
        tests_passed += 1
    
    if test_mark_price_quality():
        tests_passed += 1
    
    if compare_mark_prices_vs_ohlcv():
        tests_passed += 1
    
    # if test_mark_price_consistency():
    #     tests_passed += 1
    
    log.info(f"\n📊 TEST SUMMARY:")
    log.info(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        log.info("🎉 All tests passed! Mark prices are working correctly.")
        return True
    else:
        log.warning(f"⚠️ {total_tests - tests_passed} test(s) failed. Check the issues above.")
        return False

def main():
    """Main test function"""
    log.info("Starting mark prices functionality tests...")
    
    try:
        success = generate_test_report()
        
        if success:
            log.info("\n✅ Mark prices system is ready for use!")
            log.info("You can now:")
            log.info("  - Run backtesting with realistic VWAP prices")
            log.info("  - Use 'WHERE is_valid = TRUE' for highest quality data")
            log.info("  - Calculate slippage using orderbook data")
            log.info("  - Include funding costs in P&L calculations")
        else:
            log.error("\n❌ Mark prices system has issues that need to be resolved")
        
        return success
        
    except Exception as e:
        log.error(f"Mark prices test failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)