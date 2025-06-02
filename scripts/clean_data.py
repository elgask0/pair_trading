#!/usr/bin/env python3
"""
Data cleaning script - FIXED VERSION
Calculates P80 thresholds and marks data quality without deleting any data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import db_manager
from src.database.migrations import add_data_quality_columns, check_data_quality_schema
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

def ensure_data_quality_schema():
    """Ensure data quality columns exist"""
    log.info("Checking data quality schema...")
    
    if not check_data_quality_schema():
        log.info("Adding missing data quality columns...")
        add_data_quality_columns()
    else:
        log.info("Data quality schema is up to date")

def check_if_already_processed(symbol: str) -> bool:
    """Check if symbol has already been processed for quality"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT COUNT(*) as processed_count
            FROM orderbook 
            WHERE symbol = :symbol 
            AND liquidity_quality IS NOT NULL
        """), {'symbol': symbol}).fetchone()
        
        if result.processed_count > 0:
            log.info(f"{symbol} already processed ({result.processed_count:,} records with quality marks)")
            return True
        return False

def calculate_p80_threshold(symbol: str) -> float:
    """Calculate P80 threshold for spread quality - FIXED VERSION"""
    log.info(f"Calculating P80 threshold for {symbol}...")
    
    with db_manager.get_session() as session:
        # Get ALL valid spread data - not ordered to save memory
        result = session.execute(text("""
            SELECT (ask1_price - bid1_price) / bid1_price * 100 as spread_pct
            FROM orderbook 
            WHERE symbol = :symbol 
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL 
            AND bid1_price > 0 
            AND ask1_price > 0
            AND bid1_price < ask1_price
            AND ask1_price > bid1_price * 1.00001
        """), {'symbol': symbol}).fetchall()
        
        if not result:
            log.warning(f"No valid spread data for {symbol}")
            return 0.1  # Default fallback
        
        spreads = [row.spread_pct for row in result if row.spread_pct is not None]
        
        if len(spreads) == 0:
            log.warning(f"No valid spread calculations for {symbol}")
            return 0.1
        
        spreads_array = np.array(spreads)
        
        # Remove potential outliers (>10% spread is unrealistic)
        clean_spreads = spreads_array[spreads_array <= 10.0]
        
        if len(clean_spreads) == 0:
            log.warning(f"No reasonable spreads for {symbol} after filtering")
            return 0.1
        
        # Calculate P80 (80th percentile)
        p80_threshold = np.percentile(clean_spreads, 80)
        
        # Ensure minimum threshold of 0.001% to avoid zero
        p80_threshold = max(p80_threshold, 0.001)
        
        log.info(f"P80 threshold for {symbol}: {p80_threshold:.4f}%")
        log.info(f"  Based on {len(clean_spreads):,} valid spread observations")
        log.info(f"  Min spread: {clean_spreads.min():.4f}%")
        log.info(f"  Median spread: {np.median(clean_spreads):.4f}%")
        log.info(f"  P90 spread: {np.percentile(clean_spreads, 90):.4f}%")
        log.info(f"  Max reasonable spread: {clean_spreads.max():.4f}%")
        
        return p80_threshold

def mark_data_quality(symbol: str, threshold: float):
    """Mark data quality based on P80 threshold - IMPROVED VERSION"""
    log.info(f"Marking data quality for {symbol} with P80 threshold {threshold:.4f}%...")
    
    with db_manager.get_session() as session:
        # 1. Reset all quality fields first
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = NULL,
                valid_for_trading = FALSE,
                spread_pct = NULL,
                threshold_p80 = :threshold
            WHERE symbol = :symbol
        """), {'symbol': symbol, 'threshold': threshold})
        
        # 2. Calculate spread_pct for valid quotes only
        session.execute(text("""
            UPDATE orderbook 
            SET spread_pct = ((ask1_price - bid1_price) / bid1_price * 100)
            WHERE symbol = :symbol 
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            AND bid1_price > 0 
            AND ask1_price > 0
            AND bid1_price < ask1_price
        """), {'symbol': symbol})
        
        # 3. Mark records with invalid spreads
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = 'Invalid',
                valid_for_trading = FALSE
            WHERE symbol = :symbol 
            AND (bid1_price IS NULL 
                 OR ask1_price IS NULL 
                 OR bid1_price <= 0 
                 OR ask1_price <= 0
                 OR bid1_price >= ask1_price
                 OR spread_pct IS NULL
                 OR spread_pct > 10.0)
        """), {'symbol': symbol})
        
        # 4. Classify valid spreads based on P80 thresholds
        excellent_threshold = threshold * 0.5  # 50% of P80
        fair_threshold = threshold * 1.5       # 150% of P80
        
        # Excellent (≤50% of P80)
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = 'Excellent', valid_for_trading = TRUE
            WHERE symbol = :symbol 
            AND spread_pct IS NOT NULL
            AND spread_pct <= :excellent_threshold
            AND liquidity_quality IS NULL
        """), {'symbol': symbol, 'excellent_threshold': excellent_threshold})
        
        # Good (50% < spread ≤ P80)
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = 'Good', valid_for_trading = TRUE
            WHERE symbol = :symbol 
            AND spread_pct IS NOT NULL
            AND spread_pct > :excellent_threshold 
            AND spread_pct <= :threshold
            AND liquidity_quality IS NULL
        """), {'symbol': symbol, 'excellent_threshold': excellent_threshold, 'threshold': threshold})
        
        # Fair (P80 < spread ≤ 150% of P80)
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = 'Fair', valid_for_trading = FALSE
            WHERE symbol = :symbol 
            AND spread_pct IS NOT NULL
            AND spread_pct > :threshold 
            AND spread_pct <= :fair_threshold
            AND liquidity_quality IS NULL
        """), {'symbol': symbol, 'threshold': threshold, 'fair_threshold': fair_threshold})
        
        # Poor (>150% of P80)
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = 'Poor', valid_for_trading = FALSE
            WHERE symbol = :symbol 
            AND spread_pct IS NOT NULL
            AND spread_pct > :fair_threshold
            AND liquidity_quality IS NULL
        """), {'symbol': symbol, 'fair_threshold': fair_threshold})
        
        # 5. Verify results and log summary
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as trading_ready,
                COUNT(CASE WHEN liquidity_quality = 'Excellent' THEN 1 END) as excellent,
                COUNT(CASE WHEN liquidity_quality = 'Good' THEN 1 END) as good,
                COUNT(CASE WHEN liquidity_quality = 'Fair' THEN 1 END) as fair,
                COUNT(CASE WHEN liquidity_quality = 'Poor' THEN 1 END) as poor,
                COUNT(CASE WHEN liquidity_quality = 'Invalid' THEN 1 END) as invalid,
                COUNT(CASE WHEN liquidity_quality IS NULL THEN 1 END) as unprocessed,
                AVG(CASE WHEN spread_pct IS NOT NULL AND spread_pct <= 10.0 THEN spread_pct END) as avg_valid_spread
            FROM orderbook 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        log.info(f"Quality marking completed for {symbol}:")
        log.info(f"  Total records: {result.total:,}")
        log.info(f"  Trading ready (≤P80): {result.trading_ready:,} ({result.trading_ready/result.total*100:.1f}%)")
        log.info(f"  Quality breakdown:")
        log.info(f"    Excellent: {result.excellent:,} ({result.excellent/result.total*100:.1f}%)")
        log.info(f"    Good: {result.good:,} ({result.good/result.total*100:.1f}%)")
        log.info(f"    Fair: {result.fair:,} ({result.fair/result.total*100:.1f}%)")
        log.info(f"    Poor: {result.poor:,} ({result.poor/result.total*100:.1f}%)")
        log.info(f"    Invalid: {result.invalid:,} ({result.invalid/result.total*100:.1f}%)")
        if result.unprocessed > 0:
            log.warning(f"    Unprocessed: {result.unprocessed:,} ({result.unprocessed/result.total*100:.1f}%)")
        log.info(f"  Average valid spread: {result.avg_valid_spread:.4f}%")
        
        return {
            'symbol': symbol,
            'threshold_p80': threshold,
            'total_records': result.total,
            'trading_ready': result.trading_ready,
            'trading_ready_pct': result.trading_ready/result.total*100 if result.total > 0 else 0,
            'quality_counts': {
                'excellent': result.excellent,
                'good': result.good,
                'fair': result.fair,
                'poor': result.poor,
                'invalid': result.invalid,
                'unprocessed': result.unprocessed
            },
            'avg_valid_spread': result.avg_valid_spread or 0
        }

def clean_symbol_data(symbol: str, force_recalculate: bool = False) -> Dict:
    """Clean and mark quality for a single symbol"""
    
    # Check if already processed
    if not force_recalculate and check_if_already_processed(symbol):
        log.info(f"Skipping {symbol} - already processed (use --force to recalculate)")
        return None
    
    # Calculate P80 threshold
    p80_threshold = calculate_p80_threshold(symbol)
    
    if p80_threshold <= 0:
        log.error(f"Invalid threshold calculated for {symbol}")
        return None
    
    # Mark data quality
    result = mark_data_quality(symbol, p80_threshold)
    
    return result

def generate_cleaning_summary(cleaning_results: Dict):
    """Generate summary of cleaning process"""
    log.info("\n" + "="*80)
    log.info("DATA CLEANING SUMMARY")
    log.info("="*80)
    
    total_symbols = len(cleaning_results)
    total_records = sum(r['total_records'] for r in cleaning_results.values() if r)
    total_trading_ready = sum(r['trading_ready'] for r in cleaning_results.values() if r)
    
    log.info(f"Processed {total_symbols} symbols")
    log.info(f"Total records processed: {total_records:,}")
    log.info(f"Total trading-ready records: {total_trading_ready:,} ({total_trading_ready/total_records*100:.1f}%)")
    
    log.info(f"\nSymbol Details:")
    log.info("-" * 50)
    
    for symbol, result in cleaning_results.items():
        if not result:
            continue
            
        symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
        log.info(f"{symbol_short}:")
        log.info(f"  P80 Threshold: {result['threshold_p80']:.4f}%")
        log.info(f"  Trading Ready: {result['trading_ready']:,} ({result['trading_ready_pct']:.1f}%)")
        log.info(f"  Avg Valid Spread: {result['avg_valid_spread']:.4f}%")
        
        # Quality assessment
        if result['trading_ready_pct'] >= 75:
            assessment = "EXCELLENT for trading"
        elif result['trading_ready_pct'] >= 60:
            assessment = "GOOD for trading"
        elif result['trading_ready_pct'] >= 40:
            assessment = "FAIR for trading"
        else:
            assessment = "NEEDS REVIEW"
        
        log.info(f"  Assessment: {assessment}")
    
    log.info(f"\nNext Steps:")
    log.info(f"- Use 'WHERE valid_for_trading = TRUE' for optimal trading data")
    log.info(f"- Run 'python scripts/analyze_data.py' for detailed analysis")
    log.info(f"- Quality data is preserved - no records were deleted")

def main():
    """Main cleaning function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean and mark data quality")
    parser.add_argument("--symbol", type=str, help="Specific symbol to clean")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if already processed")
    
    args = parser.parse_args()
    
    log.info("Starting data cleaning process...")
    
    # Ensure schema is ready
    ensure_data_quality_schema()
    
    # Get symbols to clean
    if args.symbol:
        symbols = [args.symbol]
    else:
        try:
            active_pairs = settings.get_active_pairs()
            symbols = []
            for pair in active_pairs:
                symbols.extend([pair.symbol1, pair.symbol2])
            symbols = list(set(symbols))  # Remove duplicates
        except Exception as e:
            log.error(f"Could not load symbols from config: {e}")
            symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']  # Fallback
    
    if not symbols:
        log.error("No symbols to clean")
        return False
    
    log.info(f"Cleaning {len(symbols)} symbols: {', '.join([s.split('_')[-2] for s in symbols])}")
    
    cleaning_results = {}
    
    try:
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"CLEANING {symbol}")
            log.info(f"{'='*60}")
            
            result = clean_symbol_data(symbol, args.force)
            cleaning_results[symbol] = result
        
        # Generate summary
        valid_results = {k: v for k, v in cleaning_results.items() if v is not None}
        if valid_results:
            generate_cleaning_summary(valid_results)
        
        log.info(f"\nData cleaning completed successfully!")
        log.info(f"Processed: {len(valid_results)}/{len(symbols)} symbols")
        
        return True
        
    except Exception as e:
        log.error(f"Data cleaning failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)