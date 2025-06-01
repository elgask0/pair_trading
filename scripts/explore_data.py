#!/usr/bin/env python3
"""
Simple data exploration script
Shows what data we actually have in the database
"""

import sys
import os
import pandas as pd
from sqlalchemy import text
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.utils.logger import get_validation_logger

log = get_validation_logger()

def explore_tables():
    """Explore all tables and show basic info"""
    log.info("Starting database exploration")
    
    with db_manager.get_session() as session:
        
        # 1. Show all tables
        log.info("Checking available tables")
        tables_query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = session.execute(tables_query).fetchall()
        for table in tables:
            log.info(f"Found table: {table[0]}")
        
        # 2. OHLCV Analysis
        log.info("Analyzing OHLCV table")
        
        # Basic counts
        ohlcv_query = text("""
            SELECT 
                symbol,
                COUNT(*) as total_records,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date,
                COUNT(DISTINCT DATE(timestamp)) as unique_days
            FROM ohlcv 
            GROUP BY symbol
            ORDER BY symbol
        """)
        
        ohlcv_results = session.execute(ohlcv_query).fetchall()
        
        if ohlcv_results:
            for row in ohlcv_results:
                log.info(f"OHLCV {row.symbol}: {row.total_records:,} records")
                log.info(f"  Date range: {row.first_date} to {row.last_date}")
                log.info(f"  Unique days: {row.unique_days}")
                
                # Calculate expected vs actual
                if row.first_date and row.last_date:
                    total_minutes = int((row.last_date - row.first_date).total_seconds() / 60) + 1
                    coverage = (row.total_records / total_minutes) * 100 if total_minutes > 0 else 0
                    log.info(f"  Coverage: {coverage:.1f}% ({row.total_records:,}/{total_minutes:,} minutes)")
        else:
            log.warning("No OHLCV data found")
        
        # 3. Sample OHLCV data
        log.info("Showing sample OHLCV data")
        sample_ohlcv = text("""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM ohlcv 
            ORDER BY symbol, timestamp
            LIMIT 5
        """)
        
        sample_results = session.execute(sample_ohlcv).fetchall()
        if sample_results:
            for row in sample_results:
                log.info(f"Sample OHLCV: {row.symbol} | {row.timestamp} | O:{row.open} H:{row.high} L:{row.low} C:{row.close} V:{row.volume}")
        
        # 4. OHLCV Data Quality Quick Check
        log.info("Running OHLCV quality checks")
        
        quality_query = text("""
            SELECT 
                symbol,
                COUNT(*) as total,
                SUM(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL THEN 1 ELSE 0 END) as null_values,
                SUM(CASE WHEN open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 THEN 1 ELSE 0 END) as negative_prices,
                SUM(CASE WHEN volume < 0 THEN 1 ELSE 0 END) as negative_volume,
                SUM(CASE WHEN high < GREATEST(open, close) OR low > LEAST(open, close) THEN 1 ELSE 0 END) as invalid_ohlc
            FROM ohlcv 
            GROUP BY symbol
        """)
        
        quality_results = session.execute(quality_query).fetchall()
        for row in quality_results:
            log.info(f"Quality check {row.symbol}:")
            log.info(f"  Null values: {row.null_values}")
            log.info(f"  Negative prices: {row.negative_prices}")
            log.info(f"  Negative volume: {row.negative_volume}")
            log.info(f"  Invalid OHLC: {row.invalid_ohlc}")
        
        # 5. Orderbook Analysis
        log.info("Analyzing orderbook table")
        
        orderbook_query = text("""
            SELECT 
                symbol,
                COUNT(*) as total_snapshots,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date,
                COUNT(DISTINCT DATE(timestamp)) as unique_days
            FROM orderbook 
            GROUP BY symbol
            ORDER BY symbol
        """)
        
        orderbook_results = session.execute(orderbook_query).fetchall()
        
        if orderbook_results:
            for row in orderbook_results:
                log.info(f"Orderbook {row.symbol}: {row.total_snapshots:,} snapshots")
                log.info(f"  Date range: {row.first_date} to {row.last_date}")
                log.info(f"  Unique days: {row.unique_days}")
                
                # Snapshots per day average
                if row.unique_days > 0:
                    avg_per_day = row.total_snapshots / row.unique_days
                    log.info(f"  Avg snapshots/day: {avg_per_day:,.0f}")
        else:
            log.warning("No orderbook data found")
        
        # 6. Sample Orderbook data
        log.info("Showing sample orderbook data")
        sample_orderbook = text("""
            SELECT symbol, timestamp, bid1_price, bid1_size, ask1_price, ask1_size
            FROM orderbook 
            ORDER BY symbol, timestamp
            LIMIT 5
        """)
        
        sample_ob_results = session.execute(sample_orderbook).fetchall()
        if sample_ob_results:
            for row in sample_ob_results:
                spread = row.ask1_price - row.bid1_price if row.ask1_price and row.bid1_price else None
                log.info(f"Sample orderbook: {row.symbol} | {row.timestamp} | Bid:{row.bid1_price}@{row.bid1_size} Ask:{row.ask1_price}@{row.ask1_size} Spread:{spread}")
        
        # 7. Orderbook Quality Check
        log.info("Running orderbook quality checks")
        
        ob_quality_query = text("""
            SELECT 
                symbol,
                COUNT(*) as total,
                SUM(CASE WHEN bid1_price IS NULL OR ask1_price IS NULL THEN 1 ELSE 0 END) as missing_best_quotes,
                SUM(CASE WHEN bid1_price >= ask1_price THEN 1 ELSE 0 END) as crossed_books
            FROM orderbook 
            GROUP BY symbol
        """)
        
        ob_quality_results = session.execute(ob_quality_query).fetchall()
        for row in ob_quality_results:
            log.info(f"Orderbook quality {row.symbol}:")
            log.info(f"  Missing best quotes: {row.missing_best_quotes}")
            log.info(f"  Crossed books: {row.crossed_books}")
        
        # 8. Gaps Analysis
        log.info("Analyzing temporal gaps")
        
        for symbol_row in ohlcv_results:
            symbol = symbol_row.symbol
            log.info(f"Checking gaps for {symbol}")
            
            gaps_query = text("""
                WITH time_diffs AS (
                    SELECT 
                        timestamp,
                        LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                        EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)))/60 as gap_minutes
                    FROM ohlcv 
                    WHERE symbol = :symbol
                    ORDER BY timestamp
                )
                SELECT 
                    COUNT(*) as total_gaps,
                    AVG(gap_minutes) as avg_gap,
                    MAX(gap_minutes) as max_gap
                FROM time_diffs 
                WHERE gap_minutes > 5
            """)
            
            gaps_result = session.execute(gaps_query, {"symbol": symbol}).fetchone()
            if gaps_result and gaps_result.total_gaps > 0:
                log.warning(f"  Found {gaps_result.total_gaps} gaps > 5min")
                log.warning(f"  Average gap: {gaps_result.avg_gap:.1f} minutes")
                log.warning(f"  Largest gap: {gaps_result.max_gap:.1f} minutes")
            else:
                log.info(f"  No significant gaps found")
        
        # 9. Temporal Alignment Check
        log.info("Checking temporal alignment between symbols")
        
        if len(ohlcv_results) >= 2:
            symbol1 = ohlcv_results[0].symbol
            symbol2 = ohlcv_results[1].symbol
            
            alignment_query = text("""
                SELECT COUNT(*) as synchronized_points
                FROM ohlcv o1
                INNER JOIN ohlcv o2 ON o1.timestamp = o2.timestamp
                WHERE o1.symbol = :symbol1 AND o2.symbol = :symbol2
            """)
            
            alignment_result = session.execute(alignment_query, {"symbol1": symbol1, "symbol2": symbol2}).fetchone()
            
            total1 = ohlcv_results[0].total_records
            total2 = ohlcv_results[1].total_records
            sync_points = alignment_result.synchronized_points
            
            log.info(f"Alignment between {symbol1} and {symbol2}:")
            log.info(f"  {symbol1}: {total1:,} records")
            log.info(f"  {symbol2}: {total2:,} records")
            log.info(f"  Synchronized points: {sync_points:,}")
            sync_pct = (sync_points/min(total1, total2)*100) if min(total1, total2) > 0 else 0
            log.info(f"  Sync percentage: {sync_pct:.1f}%")

def show_data_structure():
    """Show the structure of our tables"""
    log.info("Showing table structures")
    
    with db_manager.get_session() as session:
        
        # OHLCV structure
        ohlcv_struct = text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'ohlcv'
            ORDER BY ordinal_position
        """)
        
        log.info("OHLCV table structure:")
        ohlcv_columns = session.execute(ohlcv_struct).fetchall()
        for col in ohlcv_columns:
            nullable = 'NULL' if col.is_nullable == 'YES' else 'NOT NULL'
            log.info(f"  {col.column_name}: {col.data_type} {nullable}")
        
        # Orderbook structure  
        orderbook_struct = text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'orderbook'
            ORDER BY ordinal_position
        """)
        
        log.info("Orderbook table structure:")
        orderbook_columns = session.execute(orderbook_struct).fetchall()
        for col in orderbook_columns:
            nullable = 'NULL' if col.is_nullable == 'YES' else 'NOT NULL'
            log.info(f"  {col.column_name}: {col.data_type} {nullable}")

def main():
    log.info("Starting data exploration script")
    
    try:
        show_data_structure()
        explore_tables()
        
        log.info("Data exploration completed successfully")
        return True
        
    except Exception as e:
        log.error(f"Data exploration failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)