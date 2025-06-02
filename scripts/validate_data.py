#!/usr/bin/env python3
"""
Data validation script
Validates raw data integrity before cleaning process
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

def get_symbol_data_summary(symbol: str) -> Dict:
    """Get basic data summary for a symbol"""
    log.info(f"Getting data summary for {symbol}...")
    
    with db_manager.get_session() as session:
        # OHLCV summary
        ohlcv_result = session.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                COUNT(CASE WHEN volume > 0 THEN 1 END) as records_with_volume,
                AVG(volume) as avg_volume,
                COUNT(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL THEN 1 END) as null_ohlc
            FROM ohlcv 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        # Orderbook summary
        orderbook_result = session.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL THEN 1 END) as valid_quotes,
                COUNT(CASE WHEN bid1_price IS NULL OR ask1_price IS NULL THEN 1 END) as null_quotes
            FROM orderbook 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        return {
            'symbol': symbol,
            'ohlcv': {
                'total_records': ohlcv_result.total_records or 0,
                'min_date': ohlcv_result.min_date,
                'max_date': ohlcv_result.max_date,
                'records_with_volume': ohlcv_result.records_with_volume or 0,
                'avg_volume': ohlcv_result.avg_volume or 0,
                'null_ohlc': ohlcv_result.null_ohlc or 0
            },
            'orderbook': {
                'total_records': orderbook_result.total_records or 0,
                'min_date': orderbook_result.min_date,
                'max_date': orderbook_result.max_date,
                'valid_quotes': orderbook_result.valid_quotes or 0,
                'null_quotes': orderbook_result.null_quotes or 0
            }
        }

def validate_ohlcv_integrity(symbol: str) -> Dict:
    """Validate OHLCV data integrity"""
    log.info(f"Validating OHLCV integrity for {symbol}...")
    
    with db_manager.get_session() as session:
        # Get problematic records
        validation_query = text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN high < open OR high < close OR low > open OR low > close OR high < low THEN 1 END) as invalid_ohlc,
                COUNT(CASE WHEN volume < 0 THEN 1 END) as negative_volume,
                COUNT(CASE WHEN volume = 0 THEN 1 END) as zero_volume,
                COUNT(CASE WHEN open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 THEN 1 END) as non_positive_prices,
                COUNT(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL THEN 1 END) as null_values
            FROM ohlcv 
            WHERE symbol = :symbol
        """)
        
        result = session.execute(validation_query, {'symbol': symbol}).fetchone()
        
        # Calculate percentages
        total = result.total_records or 1  # Avoid division by zero
        
        validation_summary = {
            'symbol': symbol,
            'total_records': result.total_records or 0,
            'issues': {
                'invalid_ohlc': {
                    'count': result.invalid_ohlc or 0,
                    'percentage': (result.invalid_ohlc or 0) / total * 100
                },
                'negative_volume': {
                    'count': result.negative_volume or 0,
                    'percentage': (result.negative_volume or 0) / total * 100
                },
                'zero_volume': {
                    'count': result.zero_volume or 0,
                    'percentage': (result.zero_volume or 0) / total * 100
                },
                'non_positive_prices': {
                    'count': result.non_positive_prices or 0,
                    'percentage': (result.non_positive_prices or 0) / total * 100
                },
                'null_values': {
                    'count': result.null_values or 0,
                    'percentage': (result.null_values or 0) / total * 100
                }
            }
        }
        
        # Log results
        log.info(f"{symbol} OHLCV Validation Results:")
        log.info(f"  Total records: {validation_summary['total_records']:,}")
        for issue_type, issue_data in validation_summary['issues'].items():
            if issue_data['count'] > 0:
                log.warning(f"  {issue_type}: {issue_data['count']:,} ({issue_data['percentage']:.3f}%)")
            else:
                log.info(f"  {issue_type}: OK")
        
        return validation_summary

def validate_orderbook_integrity(symbol: str) -> Dict:
    """Validate orderbook data integrity"""
    log.info(f"Validating orderbook integrity for {symbol}...")
    
    with db_manager.get_session() as session:
        # Basic validation
        basic_validation = session.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN bid1_price IS NULL OR ask1_price IS NULL THEN 1 END) as missing_top_level,
                COUNT(CASE WHEN bid1_price <= 0 OR ask1_price <= 0 THEN 1 END) as non_positive_prices,
                COUNT(CASE WHEN bid1_size IS NULL OR ask1_size IS NULL THEN 1 END) as missing_top_size,
                COUNT(CASE WHEN bid1_size <= 0 OR ask1_size <= 0 THEN 1 END) as non_positive_sizes,
                COUNT(CASE WHEN bid1_price >= ask1_price THEN 1 END) as crossed_spread
            FROM orderbook 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        # Spread analysis (only for valid quotes)
        spread_analysis = session.execute(text("""
            SELECT 
                COUNT(*) as valid_quotes,
                AVG((ask1_price - bid1_price) / bid1_price * 100) as avg_spread_pct,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (ask1_price - bid1_price) / bid1_price * 100) as median_spread_pct,
                MIN((ask1_price - bid1_price) / bid1_price * 100) as min_spread_pct,
                MAX((ask1_price - bid1_price) / bid1_price * 100) as max_spread_pct
            FROM orderbook 
            WHERE symbol = :symbol 
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL 
            AND bid1_price > 0 
            AND ask1_price > 0
            AND bid1_price < ask1_price
        """), {'symbol': symbol}).fetchone()
        
        total = basic_validation.total_records or 1
        
        validation_summary = {
            'symbol': symbol,
            'total_records': basic_validation.total_records or 0,
            'issues': {
                'missing_top_level': {
                    'count': basic_validation.missing_top_level or 0,
                    'percentage': (basic_validation.missing_top_level or 0) / total * 100
                },
                'non_positive_prices': {
                    'count': basic_validation.non_positive_prices or 0,
                    'percentage': (basic_validation.non_positive_prices or 0) / total * 100
                },
                'missing_top_size': {
                    'count': basic_validation.missing_top_size or 0,
                    'percentage': (basic_validation.missing_top_size or 0) / total * 100
                },
                'non_positive_sizes': {
                    'count': basic_validation.non_positive_sizes or 0,
                    'percentage': (basic_validation.non_positive_sizes or 0) / total * 100
                },
                'crossed_spread': {
                    'count': basic_validation.crossed_spread or 0,
                    'percentage': (basic_validation.crossed_spread or 0) / total * 100
                }
            },
            'spread_stats': {
                'valid_quotes': spread_analysis.valid_quotes or 0,
                'avg_spread_pct': spread_analysis.avg_spread_pct or 0,
                'median_spread_pct': spread_analysis.median_spread_pct or 0,
                'min_spread_pct': spread_analysis.min_spread_pct or 0,
                'max_spread_pct': spread_analysis.max_spread_pct or 0
            } if spread_analysis.valid_quotes else None
        }
        
        # Log results
        log.info(f"{symbol} Orderbook Validation Results:")
        log.info(f"  Total records: {validation_summary['total_records']:,}")
        for issue_type, issue_data in validation_summary['issues'].items():
            if issue_data['count'] > 0:
                log.warning(f"  {issue_type}: {issue_data['count']:,} ({issue_data['percentage']:.3f}%)")
            else:
                log.info(f"  {issue_type}: OK")
        
        if validation_summary['spread_stats']:
            stats = validation_summary['spread_stats']
            log.info(f"  Valid quotes for spread analysis: {stats['valid_quotes']:,}")
            log.info(f"  Average spread: {stats['avg_spread_pct']:.4f}%")
            log.info(f"  Median spread: {stats['median_spread_pct']:.4f}%")
            log.info(f"  Spread range: {stats['min_spread_pct']:.4f}% to {stats['max_spread_pct']:.4f}%")
        
        return validation_summary

def check_data_completeness(symbol: str) -> Dict:
    """Check data completeness and gaps"""
    log.info(f"Checking data completeness for {symbol}...")
    
    with db_manager.get_session() as session:
        # Check for gaps in OHLCV data (expecting 1-minute intervals)
        ohlcv_gaps = session.execute(text("""
            WITH time_diffs AS (
                SELECT 
                    timestamp,
                    LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                    EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)))/60 as minutes_diff
                FROM ohlcv 
                WHERE symbol = :symbol
                ORDER BY timestamp
            )
            SELECT 
                COUNT(*) as total_intervals,
                COUNT(CASE WHEN minutes_diff > 1.1 THEN 1 END) as gaps_found,
                AVG(minutes_diff) as avg_interval_minutes,
                MAX(minutes_diff) as max_gap_minutes
            FROM time_diffs 
            WHERE prev_timestamp IS NOT NULL
        """), {'symbol': symbol}).fetchone()
        
        # Check for gaps in orderbook data
        orderbook_gaps = session.execute(text("""
            WITH time_diffs AS (
                SELECT 
                    timestamp,
                    LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                    EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)))/60 as minutes_diff
                FROM orderbook 
                WHERE symbol = :symbol
                ORDER BY timestamp
            )
            SELECT 
                COUNT(*) as total_intervals,
                COUNT(CASE WHEN minutes_diff > 1.1 THEN 1 END) as gaps_found,
                AVG(minutes_diff) as avg_interval_minutes,
                MAX(minutes_diff) as max_gap_minutes
            FROM time_diffs 
            WHERE prev_timestamp IS NOT NULL
        """), {'symbol': symbol}).fetchone()
        
        completeness_summary = {
            'symbol': symbol,
            'ohlcv_completeness': {
                'total_intervals': ohlcv_gaps.total_intervals or 0,
                'gaps_found': ohlcv_gaps.gaps_found or 0,
                'avg_interval_minutes': ohlcv_gaps.avg_interval_minutes or 0,
                'max_gap_minutes': ohlcv_gaps.max_gap_minutes or 0,
                'completeness_pct': ((ohlcv_gaps.total_intervals or 0) - (ohlcv_gaps.gaps_found or 0)) / max(1, ohlcv_gaps.total_intervals or 1) * 100
            },
            'orderbook_completeness': {
                'total_intervals': orderbook_gaps.total_intervals or 0,
                'gaps_found': orderbook_gaps.gaps_found or 0,
                'avg_interval_minutes': orderbook_gaps.avg_interval_minutes or 0,
                'max_gap_minutes': orderbook_gaps.max_gap_minutes or 0,
                'completeness_pct': ((orderbook_gaps.total_intervals or 0) - (orderbook_gaps.gaps_found or 0)) / max(1, orderbook_gaps.total_intervals or 1) * 100
            }
        }
        
        # Log results
        log.info(f"{symbol} Data Completeness:")
        log.info(f"  OHLCV: {completeness_summary['ohlcv_completeness']['completeness_pct']:.2f}% complete")
        log.info(f"    Gaps found: {completeness_summary['ohlcv_completeness']['gaps_found']:,}")
        log.info(f"    Max gap: {completeness_summary['ohlcv_completeness']['max_gap_minutes']:.1f} minutes")
        
        log.info(f"  Orderbook: {completeness_summary['orderbook_completeness']['completeness_pct']:.2f}% complete")
        log.info(f"    Gaps found: {completeness_summary['orderbook_completeness']['gaps_found']:,}")
        log.info(f"    Max gap: {completeness_summary['orderbook_completeness']['max_gap_minutes']:.1f} minutes")
        
        return completeness_summary

def generate_validation_report(validations: Dict):
    """Generate comprehensive validation report"""
    log.info("\n" + "="*80)
    log.info("COMPREHENSIVE DATA VALIDATION REPORT")
    log.info("="*80)
    
    for symbol, validation_data in validations.items():
        symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
        log.info(f"\n{symbol_short} - VALIDATION SUMMARY:")
        log.info("-" * 40)
        
        # Data summary
        summary = validation_data.get('summary', {})
        ohlcv = summary.get('ohlcv', {})
        orderbook = summary.get('orderbook', {})
        
        log.info(f"Data Period: {ohlcv.get('min_date', 'N/A')} to {ohlcv.get('max_date', 'N/A')}")
        log.info(f"OHLCV Records: {ohlcv.get('total_records', 0):,}")
        log.info(f"Orderbook Records: {orderbook.get('total_records', 0):,}")
        
        # Quality assessment
        ohlcv_validation = validation_data.get('ohlcv_validation', {})
        orderbook_validation = validation_data.get('orderbook_validation', {})
        completeness = validation_data.get('completeness', {})
        
        # Overall health score
        issues_count = 0
        total_checks = 0
        
        for validation in [ohlcv_validation, orderbook_validation]:
            for issue_type, issue_data in validation.get('issues', {}).items():
                total_checks += 1
                if issue_data.get('percentage', 0) > 1.0:  # More than 1% is concerning
                    issues_count += 1
        
        health_score = max(0, (total_checks - issues_count) / max(1, total_checks) * 100)
        
        if health_score >= 90:
            health_status = "EXCELLENT"
        elif health_score >= 75:
            health_status = "GOOD"
        elif health_score >= 50:
            health_status = "FAIR"
        else:
            health_status = "POOR"
        
        log.info(f"Data Health Score: {health_score:.1f}% - {health_status}")
        
        # Recommendations
        recommendations = []
        
        if ohlcv.get('total_records', 0) == 0:
            recommendations.append("NO OHLCV DATA - Check data ingestion")
        elif ohlcv_validation.get('issues', {}).get('invalid_ohlc', {}).get('percentage', 0) > 1:
            recommendations.append("High invalid OHLC rate - Review data source")
        
        if orderbook.get('total_records', 0) == 0:
            recommendations.append("NO ORDERBOOK DATA - Check data ingestion")
        elif orderbook_validation.get('issues', {}).get('crossed_spread', {}).get('percentage', 0) > 1:
            recommendations.append("High crossed spread rate - Review orderbook data")
        
        completeness_ohlcv = completeness.get('ohlcv_completeness', {}).get('completeness_pct', 0)
        completeness_orderbook = completeness.get('orderbook_completeness', {}).get('completeness_pct', 0)
        
        if completeness_ohlcv < 90:
            recommendations.append(f"OHLCV data has gaps ({completeness_ohlcv:.1f}% complete)")
        if completeness_orderbook < 90:
            recommendations.append(f"Orderbook data has gaps ({completeness_orderbook:.1f}% complete)")
        
        if not recommendations:
            recommendations.append("Data quality looks good - Ready for cleaning")
        
        log.info("Recommendations:")
        for rec in recommendations:
            log.info(f"  - {rec}")

def main():
    """Main validation function"""
    log.info("Starting comprehensive data validation...")
    
    # Get symbols to validate
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
        log.error("No symbols to validate")
        return False
    
    log.info(f"Validating {len(symbols)} symbols: {', '.join([s.split('_')[-2] for s in symbols])}")
    
    validations = {}
    
    try:
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"VALIDATING {symbol}")
            log.info(f"{'='*60}")
            
            # Collect all validation data
            validation_data = {}
            
            # 1. Basic data summary
            validation_data['summary'] = get_symbol_data_summary(symbol)
            
            # 2. OHLCV integrity
            validation_data['ohlcv_validation'] = validate_ohlcv_integrity(symbol)
            
            # 3. Orderbook integrity
            validation_data['orderbook_validation'] = validate_orderbook_integrity(symbol)
            
            # 4. Data completeness
            validation_data['completeness'] = check_data_completeness(symbol)
            
            validations[symbol] = validation_data
        
        # Generate final report
        generate_validation_report(validations)
        
        log.info(f"\nValidation completed successfully for {len(symbols)} symbols")
        log.info("Next step: Run 'python scripts/clean_data.py' to clean and mark data quality")
        
        return True
        
    except Exception as e:
        log.error(f"Validation failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)