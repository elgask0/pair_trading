#!/usr/bin/env python3
"""
Data cleaning script with intelligent forward fill and comprehensive validation
Fills gaps in OHLCV data and marks synthetic data in database
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text, MetaData, Table, Column, Boolean, Integer, String, DateTime, Float
from sqlalchemy.dialects.postgresql import insert
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger

log = get_validation_logger()

# Create plots directory
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

def add_synthetic_data_columns():
    """Add columns to track synthetic/filled data"""
    log.info("Adding synthetic data tracking columns...")
    
    with db_manager.get_session() as session:
        # Add is_synthetic column to OHLCV table if it doesn't exist
        try:
            session.execute(text("""
                ALTER TABLE ohlcv 
                ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS fill_method VARCHAR(50),
                ADD COLUMN IF NOT EXISTS fill_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """))
            log.info("Added synthetic tracking columns to OHLCV table")
        except Exception as e:
            log.warning(f"Columns may already exist in OHLCV: {e}")
        
        # Add similar columns to orderbook table
        try:
            session.execute(text("""
                ALTER TABLE orderbook 
                ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS fill_method VARCHAR(50),
                ADD COLUMN IF NOT EXISTS fill_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """))
            log.info("Added synthetic tracking columns to orderbook table")
        except Exception as e:
            log.warning(f"Columns may already exist in orderbook: {e}")

def load_and_analyze_ohlcv_data(symbol):
    """Load OHLCV data and analyze gaps"""
    log.info(f"Loading OHLCV data for {symbol}...")
    
    with db_manager.get_session() as session:
        query = text("""
            SELECT timestamp, open, high, low, close, volume, 
                   COALESCE(is_synthetic, FALSE) as is_synthetic,
                   fill_method
            FROM ohlcv 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params={"symbol": symbol}, index_col='timestamp')
        
        if len(df) == 0:
            log.warning(f"No OHLCV data found for {symbol}")
            return None
        
        log.info(f"Loaded {len(df):,} records for {symbol}")
        log.info(f"Existing synthetic records: {df['is_synthetic'].sum()}")
        
        return df

def load_and_analyze_orderbook_data(symbol):
    """Load orderbook data and analyze quality"""
    log.info(f"Loading orderbook data for {symbol}...")
    
    with db_manager.get_session() as session:
        query = text("""
            SELECT timestamp, 
                   bid1_price, bid1_size, ask1_price, ask1_size,
                   bid2_price, bid2_size, ask2_price, ask2_size,
                   bid3_price, bid3_size, ask3_price, ask3_size,
                   COALESCE(is_synthetic, FALSE) as is_synthetic
            FROM orderbook 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params={"symbol": symbol}, index_col='timestamp')
        
        if len(df) == 0:
            log.warning(f"No orderbook data found for {symbol}")
            return None
        
        log.info(f"Loaded {len(df):,} orderbook records for {symbol}")
        return df

def analyze_orderbook_quality(symbol, orderbook_df):
    """Comprehensive orderbook quality analysis"""
    log.info(f"\n=== ORDERBOOK QUALITY ANALYSIS FOR {symbol} ===")
    
    if orderbook_df is None or len(orderbook_df) == 0:
        log.warning("No orderbook data to analyze")
        return
    
    # Basic quality checks
    total_records = len(orderbook_df)
    
    # Check for missing best bid/ask
    missing_best_bid = orderbook_df['bid1_price'].isna().sum()
    missing_best_ask = orderbook_df['ask1_price'].isna().sum()
    
    log.info(f"Missing best bid: {missing_best_bid:,} ({missing_best_bid/total_records*100:.2f}%)")
    log.info(f"Missing best ask: {missing_best_ask:,} ({missing_best_ask/total_records*100:.2f}%)")
    
    # Calculate spreads where possible
    valid_quotes = orderbook_df.dropna(subset=['bid1_price', 'ask1_price'])
    
    if len(valid_quotes) > 0:
        spreads = valid_quotes['ask1_price'] - valid_quotes['bid1_price']
        spread_pct = spreads / valid_quotes['bid1_price'] * 100
        
        log.info(f"\n--- SPREAD ANALYSIS ---")
        log.info(f"Valid quotes: {len(valid_quotes):,} ({len(valid_quotes)/total_records*100:.1f}%)")
        log.info(f"Spread statistics (absolute):")
        log.info(f"  Mean: {spreads.mean():.6f}")
        log.info(f"  Median: {spreads.median():.6f}")
        log.info(f"  Std: {spreads.std():.6f}")
        log.info(f"  Min: {spreads.min():.6f}")
        log.info(f"  Max: {spreads.max():.6f}")
        
        log.info(f"Spread statistics (%):")
        log.info(f"  Mean: {spread_pct.mean():.4f}%")
        log.info(f"  Median: {spread_pct.median():.4f}%")
        log.info(f"  95th percentile: {spread_pct.quantile(0.95):.4f}%")
        log.info(f"  99th percentile: {spread_pct.quantile(0.99):.4f}%")
        
        # Check for crossed book
        crossed_book = (spreads < 0).sum()
        log.info(f"Crossed book instances: {crossed_book} ({crossed_book/len(valid_quotes)*100:.3f}%)")
        
        # Check for unrealistic spreads
        wide_spreads = (spread_pct > 5).sum()  # >5% spread
        log.info(f"Very wide spreads (>5%): {wide_spreads} ({wide_spreads/len(valid_quotes)*100:.3f}%)")
    
    # Analyze depth beyond level 1
    level2_coverage = (~orderbook_df[['bid2_price', 'ask2_price']].isna()).all(axis=1).sum()
    level3_coverage = (~orderbook_df[['bid3_price', 'ask3_price']].isna()).all(axis=1).sum()
    
    log.info(f"\n--- DEPTH ANALYSIS ---")
    log.info(f"Level 2 coverage: {level2_coverage:,} ({level2_coverage/total_records*100:.1f}%)")
    log.info(f"Level 3 coverage: {level3_coverage:,} ({level3_coverage/total_records*100:.1f}%)")
    
    # Analyze size distributions
    if 'bid1_size' in orderbook_df.columns:
        valid_sizes = orderbook_df.dropna(subset=['bid1_size', 'ask1_size'])
        if len(valid_sizes) > 0:
            log.info(f"\n--- SIZE ANALYSIS ---")
            log.info(f"Bid1 size stats:")
            log.info(f"  Mean: {valid_sizes['bid1_size'].mean():,.0f}")
            log.info(f"  Median: {valid_sizes['bid1_size'].median():,.0f}")
            log.info(f"  Min: {valid_sizes['bid1_size'].min():,.0f}")
            log.info(f"  Max: {valid_sizes['bid1_size'].max():,.0f}")
            
            log.info(f"Ask1 size stats:")
            log.info(f"  Mean: {valid_sizes['ask1_size'].mean():,.0f}")
            log.info(f"  Median: {valid_sizes['ask1_size'].median():,.0f}")
            log.info(f"  Min: {valid_sizes['ask1_size'].min():,.0f}")
            log.info(f"  Max: {valid_sizes['ask1_size'].max():,.0f}")

def validate_ohlcv_coherence(symbol, ohlcv_df):
    """Validate OHLCV data coherence and relationships"""
    log.info(f"\n=== OHLCV COHERENCE ANALYSIS FOR {symbol} ===")
    
    if ohlcv_df is None or len(ohlcv_df) == 0:
        log.warning("No OHLCV data to analyze")
        return
    
    # Basic OHLCV validation
    total_records = len(ohlcv_df)
    
    # Check for invalid OHLC relationships
    invalid_ohlc = (
        (ohlcv_df['high'] < ohlcv_df['open']) |
        (ohlcv_df['high'] < ohlcv_df['close']) |
        (ohlcv_df['low'] > ohlcv_df['open']) |
        (ohlcv_df['low'] > ohlcv_df['close']) |
        (ohlcv_df['high'] < ohlcv_df['low'])
    ).sum()
    
    log.info(f"Invalid OHLC relationships: {invalid_ohlc} ({invalid_ohlc/total_records*100:.3f}%)")
    
    # Check for zero/negative prices
    zero_prices = ((ohlcv_df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)).sum()
    log.info(f"Zero/negative prices: {zero_prices} ({zero_prices/total_records*100:.3f}%)")
    
    # Check for zero volume
    zero_volume = (ohlcv_df['volume'] == 0).sum()
    log.info(f"Zero volume records: {zero_volume} ({zero_volume/total_records*100:.3f}%)")
    
    # Check for price gaps/jumps
    price_changes = ohlcv_df['close'].pct_change().dropna()
    extreme_moves = (abs(price_changes) > 0.5).sum()  # >50% moves
    log.info(f"Extreme price moves (>50%): {extreme_moves} ({extreme_moves/len(price_changes)*100:.3f}%)")
    
    if extreme_moves > 0:
        log.info("Top 5 largest price moves:")
        top_moves = price_changes.abs().nlargest(5)
        for timestamp, move in top_moves.items():
            direction = "UP" if price_changes.loc[timestamp] > 0 else "DOWN"
            log.info(f"  {timestamp}: {direction} {abs(move)*100:.2f}%")
    
    # Analyze price-volume relationship
    if zero_volume < total_records:  # If we have volume data
        # Volume-weighted average price
        vwap_proxy = (ohlcv_df['close'] * ohlcv_df['volume']).rolling(window=60).sum() / ohlcv_df['volume'].rolling(window=60).sum()
        
        # Check for volume spikes
        volume_mean = ohlcv_df['volume'].mean()
        volume_std = ohlcv_df['volume'].std()
        volume_spikes = (ohlcv_df['volume'] > volume_mean + 5 * volume_std).sum()
        
        log.info(f"\n--- VOLUME ANALYSIS ---")
        log.info(f"Volume spikes (>5σ): {volume_spikes} ({volume_spikes/total_records*100:.3f}%)")
        log.info(f"Volume statistics:")
        log.info(f"  Mean: {volume_mean:,.0f}")
        log.info(f"  Std: {volume_std:,.0f}")
        log.info(f"  Min: {ohlcv_df['volume'].min():,.0f}")
        log.info(f"  Max: {ohlcv_df['volume'].max():,.0f}")

def cross_validate_ohlcv_orderbook(symbol, ohlcv_df, orderbook_df):
    """Cross-validate OHLCV data against orderbook data"""
    log.info(f"\n=== CROSS-VALIDATION: OHLCV vs ORDERBOOK FOR {symbol} ===")
    
    if ohlcv_df is None or orderbook_df is None:
        log.warning("Missing data for cross-validation")
        return
    
    # Find overlapping time periods
    ohlcv_start = ohlcv_df.index.min()
    ohlcv_end = ohlcv_df.index.max()
    ob_start = orderbook_df.index.min()
    ob_end = orderbook_df.index.max()
    
    overlap_start = max(ohlcv_start, ob_start)
    overlap_end = min(ohlcv_end, ob_end)
    
    log.info(f"OHLCV period: {ohlcv_start} to {ohlcv_end}")
    log.info(f"Orderbook period: {ob_start} to {ob_end}")
    log.info(f"Overlap period: {overlap_start} to {overlap_end}")
    
    if overlap_start >= overlap_end:
        log.warning("No overlapping time period found")
        return
    
    # Filter to overlap period
    ohlcv_overlap = ohlcv_df[(ohlcv_df.index >= overlap_start) & (ohlcv_df.index <= overlap_end)]
    
    # For each OHLCV minute, find closest orderbook snapshots
    validation_results = []
    
    # Sample every 60 minutes for performance (full validation would be too slow)
    sample_timestamps = ohlcv_overlap.index[::60][:100]  # Sample 100 points
    
    log.info(f"Sampling {len(sample_timestamps)} points for validation...")
    
    for timestamp in sample_timestamps:
        ohlcv_row = ohlcv_overlap.loc[timestamp]
        
        # Find orderbook snapshots within ±30 seconds
        window_start = timestamp - pd.Timedelta(seconds=30)
        window_end = timestamp + pd.Timedelta(seconds=30)
        
        ob_window = orderbook_df[(orderbook_df.index >= window_start) & 
                                (orderbook_df.index <= window_end)]
        
        if len(ob_window) > 0:
            # Get representative orderbook data (median of window)
            median_bid = ob_window['bid1_price'].median()
            median_ask = ob_window['ask1_price'].median()
            
            if pd.notna(median_bid) and pd.notna(median_ask):
                # Check if OHLCV prices are within reasonable bounds
                mid_price = (median_bid + median_ask) / 2
                
                # OHLCV prices should be within bid-ask spread or close to it
                within_spread_open = median_bid <= ohlcv_row['open'] <= median_ask
                within_spread_close = median_bid <= ohlcv_row['close'] <= median_ask
                
                # Or within reasonable distance (e.g., 2% of mid price)
                tolerance = mid_price * 0.02
                close_to_mid_open = abs(ohlcv_row['open'] - mid_price) <= tolerance
                close_to_mid_close = abs(ohlcv_row['close'] - mid_price) <= tolerance
                
                validation_results.append({
                    'timestamp': timestamp,
                    'ohlcv_open': ohlcv_row['open'],
                    'ohlcv_close': ohlcv_row['close'],
                    'ob_bid': median_bid,
                    'ob_ask': median_ask,
                    'ob_mid': mid_price,
                    'within_spread_open': within_spread_open,
                    'within_spread_close': within_spread_close,
                    'close_to_mid_open': close_to_mid_open,
                    'close_to_mid_close': close_to_mid_close,
                    'ob_samples': len(ob_window)
                })
    
    if validation_results:
        validation_df = pd.DataFrame(validation_results)
        
        within_spread_open_pct = validation_df['within_spread_open'].mean() * 100
        within_spread_close_pct = validation_df['within_spread_close'].mean() * 100
        close_to_mid_open_pct = validation_df['close_to_mid_open'].mean() * 100
        close_to_mid_close_pct = validation_df['close_to_mid_close'].mean() * 100
        
        log.info(f"Cross-validation results ({len(validation_df)} samples):")
        log.info(f"  Open prices within spread: {within_spread_open_pct:.1f}%")
        log.info(f"  Close prices within spread: {within_spread_close_pct:.1f}%")
        log.info(f"  Open prices close to mid (±2%): {close_to_mid_open_pct:.1f}%")
        log.info(f"  Close prices close to mid (±2%): {close_to_mid_close_pct:.1f}%")
        
        # Identify outliers
        outliers = validation_df[
            ~validation_df['within_spread_open'] & 
            ~validation_df['within_spread_close'] &
            ~validation_df['close_to_mid_open'] &
            ~validation_df['close_to_mid_close']
        ]
        
        if len(outliers) > 0:
            log.warning(f"Found {len(outliers)} potential data quality issues:")
            for _, outlier in outliers.head(5).iterrows():
                log.warning(f"  {outlier['timestamp']}: OHLCV {outlier['ohlcv_open']:.6f}-{outlier['ohlcv_close']:.6f}, "
                          f"OB {outlier['ob_bid']:.6f}-{outlier['ob_ask']:.6f}")

def identify_gaps_for_filling(symbol, df):
    """Identify gaps that should be filled vs gaps that should be left alone"""
    log.info(f"\n=== GAP IDENTIFICATION FOR {symbol} ===")
    
    if df is None or len(df) == 0:
        return [], []
    
    # Create expected timeline
    start_date = df.index.min()
    end_date = df.index.max()
    expected_timeline = pd.date_range(start=start_date, end=end_date, freq='1min')
    
    # Find gaps
    gaps_to_fill = []
    gaps_to_skip = []
    
    current_gap = []
    
    for expected_time in expected_timeline:
        if expected_time not in df.index:
            current_gap.append(expected_time)
        else:
            if current_gap:
                gap_duration = len(current_gap)
                gap_start = current_gap[0]
                gap_end = current_gap[-1]
                
                # Decision logic for filling gaps
                if gap_duration <= 5:
                    # Always fill very small gaps (≤5 minutes)
                    gaps_to_fill.append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration': gap_duration,
                        'method': 'ffill_small',
                        'reason': 'small_gap'
                    })
                elif gap_duration <= 60:
                    # Fill medium gaps (≤1 hour) during trading hours
                    gaps_to_fill.append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration': gap_duration,
                        'method': 'ffill_medium',
                        'reason': 'medium_gap'
                    })
                elif gap_duration <= 720:  # ≤12 hours
                    # Case-by-case for large gaps
                    gaps_to_skip.append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration': gap_duration,
                        'reason': 'large_gap_review'
                    })
                else:
                    # Never fill very large gaps (>12 hours)
                    gaps_to_skip.append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration': gap_duration,
                        'reason': 'maintenance_gap'
                    })
                
                current_gap = []
    
    # Handle final gap if exists
    if current_gap:
        gap_duration = len(current_gap)
        gap_start = current_gap[0]
        gap_end = current_gap[-1]
        
        if gap_duration <= 60:
            gaps_to_fill.append({
                'start': gap_start,
                'end': gap_end,
                'duration': gap_duration,
                'method': 'ffill_final',
                'reason': 'final_gap'
            })
        else:
            gaps_to_skip.append({
                'start': gap_start,
                'end': gap_end,
                'duration': gap_duration,
                'reason': 'final_large_gap'
            })
    
    log.info(f"Gaps to fill: {len(gaps_to_fill)}")
    log.info(f"Gaps to skip: {len(gaps_to_skip)}")
    
    if gaps_to_fill:
        total_fill_minutes = sum(gap['duration'] for gap in gaps_to_fill)
        log.info(f"Total minutes to fill: {total_fill_minutes:,}")
        
        # Show gap categories
        small_gaps = [g for g in gaps_to_fill if g['duration'] <= 5]
        medium_gaps = [g for g in gaps_to_fill if 5 < g['duration'] <= 60]
        
        log.info(f"  Small gaps (≤5min): {len(small_gaps)}")
        log.info(f"  Medium gaps (6-60min): {len(medium_gaps)}")
    
    if gaps_to_skip:
        total_skip_minutes = sum(gap['duration'] for gap in gaps_to_skip)
        log.info(f"Total minutes to skip: {total_skip_minutes:,}")
        
        # Show largest skipped gaps
        largest_skipped = sorted(gaps_to_skip, key=lambda x: x['duration'], reverse=True)[:5]
        log.info("Largest skipped gaps:")
        for gap in largest_skipped:
            log.info(f"  {gap['start']} to {gap['end']}: {gap['duration']/60:.1f}h ({gap['reason']})")
    
    return gaps_to_fill, gaps_to_skip

def perform_forward_fill(symbol, df, gaps_to_fill):
    """Perform forward fill operation and return filled data"""
    log.info(f"\n=== PERFORMING FORWARD FILL FOR {symbol} ===")
    
    if not gaps_to_fill:
        log.info("No gaps to fill")
        return [], df  # Return original data
    
    filled_records = []
    
    for gap in gaps_to_fill:
        # Find the last valid record before the gap
        pre_gap_data = df[df.index < gap['start']]
        
        if len(pre_gap_data) == 0:
            log.warning(f"No data before gap starting at {gap['start']}, skipping")
            continue
        
        last_valid = pre_gap_data.iloc[-1]
        
        # Create filled records for each minute in the gap
        gap_timeline = pd.date_range(start=gap['start'], end=gap['end'], freq='1min')
        
        for timestamp in gap_timeline:
            filled_record = {
                'symbol': symbol,
                'timestamp': timestamp,
                'open': float(last_valid['open']),  # Convert to Python float
                'high': float(last_valid['high']),  # Convert to Python float
                'low': float(last_valid['low']),    # Convert to Python float
                'close': float(last_valid['close']), # Convert to Python float
                'volume': 0,  # Synthetic data gets 0 volume
                'is_synthetic': True,
                'fill_method': gap['method'],
                'fill_timestamp': datetime.now()
            }
            filled_records.append(filled_record)
    
    log.info(f"Created {len(filled_records)} synthetic records")
    
    if filled_records:
        # Convert to DataFrame for return
        filled_df = pd.DataFrame(filled_records)
        filled_df.set_index('timestamp', inplace=True)
        
        # Combine with original data
        combined_df = pd.concat([df, filled_df]).sort_index()
        
        log.info(f"Combined dataset: {len(combined_df)} records ({len(filled_records)} synthetic)")
        
        return filled_records, combined_df
    
    return [], df

def insert_synthetic_data(symbol, filled_records):
    """Insert synthetic data into database"""
    if not filled_records:
        log.info("No synthetic data to insert")
        return
    
    log.info(f"Inserting {len(filled_records)} synthetic records for {symbol}...")
    
    with db_manager.get_session() as session:
        for record in filled_records:
            # Use PostgreSQL ON CONFLICT to avoid duplicates
            insert_stmt = text("""
                INSERT INTO ohlcv (symbol, timestamp, open, high, low, close, volume, 
                                 is_synthetic, fill_method, fill_timestamp, created_at)
                VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume,
                       :is_synthetic, :fill_method, :fill_timestamp, CURRENT_TIMESTAMP)
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    is_synthetic = EXCLUDED.is_synthetic,
                    fill_method = EXCLUDED.fill_method,
                    fill_timestamp = EXCLUDED.fill_timestamp
            """)
            
            session.execute(insert_stmt, record)
        
        log.info(f"Successfully inserted synthetic data for {symbol}")

def create_pre_fill_analysis(symbol, ohlcv_df, orderbook_df):
    """Create comprehensive pre-fill analysis and visualizations"""
    log.info(f"\n=== PRE-FILL ANALYSIS FOR {symbol} ===")
    
    # Create comprehensive pre-fill plots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    
    # 1. Coverage timeline
    start_date = ohlcv_df.index.min()
    end_date = ohlcv_df.index.max()
    expected_timeline = pd.date_range(start=start_date, end=end_date, freq='1min')
    
    # Daily coverage
    coverage_series = pd.Series(index=expected_timeline, data=False)
    coverage_series.loc[ohlcv_df.index] = True
    daily_coverage = coverage_series.resample('D').mean() * 100
    
    axes[0, 0].plot(daily_coverage.index, daily_coverage.values, linewidth=1, alpha=0.8)
    axes[0, 0].set_title(f'{symbol} - Pre-Fill Daily Coverage %')
    axes[0, 0].set_ylabel('Coverage %')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 105)
    
    # 2. Price and volume over time
    if len(ohlcv_df) > 0:
        daily_prices = ohlcv_df.groupby(ohlcv_df.index.date)['close'].last()
        daily_volume = ohlcv_df.groupby(ohlcv_df.index.date)['volume'].sum()
        
        axes[0, 1].plot(daily_prices.index, daily_prices.values, linewidth=1, alpha=0.8)
        axes[0, 1].set_title(f'{symbol} - Pre-Fill Daily Prices')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Volume distribution
    if len(ohlcv_df) > 0 and 'volume' in ohlcv_df.columns:
        volumes = ohlcv_df['volume'][ohlcv_df['volume'] > 0]
        if len(volumes) > 0:
            axes[1, 0].hist(volumes, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title(f'{symbol} - Pre-Fill Volume Distribution')
            axes[1, 0].set_xlabel('Volume')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_xscale('log')
    
    # 4. Hourly coverage pattern
    hourly_presence = pd.Series(index=expected_timeline, data=False)
    hourly_presence.loc[ohlcv_df.index] = True
    hourly_coverage = hourly_presence.groupby(hourly_presence.index.hour).mean() * 100
    
    axes[1, 1].bar(hourly_coverage.index, hourly_coverage.values, alpha=0.7)
    axes[1, 1].set_title(f'{symbol} - Pre-Fill Hourly Coverage %')
    axes[1, 1].set_xlabel('Hour (UTC)')
    axes[1, 1].set_ylabel('Coverage %')
    axes[1, 1].set_xticks(range(0, 24, 2))
    
    # 5. Orderbook spread analysis (if available)
    if orderbook_df is not None and len(orderbook_df) > 0:
        valid_quotes = orderbook_df.dropna(subset=['bid1_price', 'ask1_price'])
        if len(valid_quotes) > 100:
            spreads = valid_quotes['ask1_price'] - valid_quotes['bid1_price']
            spread_pct = spreads / valid_quotes['bid1_price'] * 100
            
            # Sample for performance
            sample_spreads = spread_pct.sample(min(10000, len(spread_pct)))
            
            axes[2, 0].hist(sample_spreads, bins=50, alpha=0.7, edgecolor='black')
            axes[2, 0].set_title(f'{symbol} - Orderbook Spread Distribution (%)')
            axes[2, 0].set_xlabel('Spread %')
            axes[2, 0].set_ylabel('Frequency')
            axes[2, 0].set_yscale('log')
        else:
            axes[2, 0].text(0.5, 0.5, 'Insufficient orderbook data', transform=axes[2, 0].transAxes, ha='center')
    else:
        axes[2, 0].text(0.5, 0.5, 'No orderbook data', transform=axes[2, 0].transAxes, ha='center')
    
    # 6. Price volatility analysis
    if len(ohlcv_df) > 0:
        returns = ohlcv_df['close'].pct_change().dropna()
        if len(returns) > 0:
            axes[2, 1].hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
            axes[2, 1].set_title(f'{symbol} - Pre-Fill Return Distribution (%)')
            axes[2, 1].set_xlabel('1-minute Returns %')
            axes[2, 1].set_ylabel('Frequency')
            axes[2, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'{symbol}_pre_fill_analysis.png', dpi=300, bbox_inches='tight')
    log.info(f"Saved pre-fill analysis to {plots_dir / f'{symbol}_pre_fill_analysis.png'}")
    plt.close()

def create_post_fill_analysis(symbol, original_df, filled_df, gaps_filled, gaps_skipped):
    """Create comprehensive post-fill analysis and visualizations"""
    log.info(f"\n=== POST-FILL ANALYSIS FOR {symbol} ===")
    
    # Create comprehensive post-fill plots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    
    # Timeline setup
    start_date = original_df.index.min()
    end_date = original_df.index.max()
    expected_timeline = pd.date_range(start=start_date, end=end_date, freq='1min')
    
    # 1. Coverage comparison
    original_coverage = pd.Series(index=expected_timeline, data=False)
    original_coverage.loc[original_df.index] = True
    filled_coverage = pd.Series(index=expected_timeline, data=False)
    filled_coverage.loc[filled_df.index] = True
    
    daily_original = original_coverage.resample('D').mean() * 100
    daily_filled = filled_coverage.resample('D').mean() * 100
    
    axes[0, 0].plot(daily_original.index, daily_original.values, label='Original', alpha=0.7, linewidth=1)
    axes[0, 0].plot(daily_filled.index, daily_filled.values, label='After Fill', alpha=0.7, linewidth=1)
    axes[0, 0].set_title(f'{symbol} - Coverage Comparison')
    axes[0, 0].set_ylabel('Coverage %')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 105)
    
    # 2. Gap analysis
    if gaps_filled or gaps_skipped:
        gap_sizes_filled = [gap['duration'] for gap in gaps_filled]
        gap_sizes_skipped = [gap['duration'] for gap in gaps_skipped]
        
        bins = [0, 5, 15, 60, 360, 720, float('inf')]
        labels = ['≤5min', '6-15min', '16-60min', '1-6h', '6-12h', '>12h']
        
        filled_counts = pd.cut(gap_sizes_filled, bins=bins, labels=labels).value_counts() if gap_sizes_filled else pd.Series(dtype=int)
        skipped_counts = pd.cut(gap_sizes_skipped, bins=bins, labels=labels).value_counts() if gap_sizes_skipped else pd.Series(dtype=int)
        
        gap_analysis = pd.DataFrame({
            'Filled': filled_counts,
            'Skipped': skipped_counts
        }).fillna(0)
        
        gap_analysis.plot(kind='bar', stacked=True, ax=axes[0, 1], alpha=0.7)
        axes[0, 1].set_title(f'{symbol} - Gap Treatment by Size')
        axes[0, 1].set_ylabel('Number of Gaps')
        axes[0, 1].set_xlabel('Gap Duration')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Volume comparison (synthetic vs real)
    real_data = filled_df[~filled_df.get('is_synthetic', False)]
    synthetic_data = filled_df[filled_df.get('is_synthetic', False)]
    
    if len(real_data) > 0:
        daily_volume_real = real_data.groupby(real_data.index.date)['volume'].sum()
        axes[1, 0].bar(daily_volume_real.index, daily_volume_real.values, alpha=0.7, label='Real Data')
        
        if len(synthetic_data) > 0:
            daily_volume_synthetic = synthetic_data.groupby(synthetic_data.index.date)['volume'].sum()
            axes[1, 0].bar(daily_volume_synthetic.index, daily_volume_synthetic.values, 
                          alpha=0.7, label='Synthetic Data', bottom=daily_volume_real.reindex(daily_volume_synthetic.index, fill_value=0))
        
        axes[1, 0].set_title(f'{symbol} - Daily Volume (Real vs Synthetic)')
        axes[1, 0].set_ylabel('Volume')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Synthetic data distribution over time
    if len(synthetic_data) > 0:
        synthetic_daily = synthetic_data.groupby(synthetic_data.index.date).size()
        axes[1, 1].bar(synthetic_daily.index, synthetic_daily.values, alpha=0.7, color='red')
        axes[1, 1].set_title(f'{symbol} - Synthetic Records per Day')
        axes[1, 1].set_ylabel('Synthetic Records')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No synthetic data', transform=axes[1, 1].transAxes, ha='center')
    
    # 5. Price continuity check
    price_jumps = filled_df['close'].pct_change().abs()
    large_jumps = price_jumps[price_jumps > 0.1]  # >10% jumps
    
    if len(large_jumps) > 0:
        axes[2, 0].scatter(large_jumps.index, large_jumps.values * 100, alpha=0.6, s=20)
        axes[2, 0].set_title(f'{symbol} - Large Price Jumps (>10%)')
        axes[2, 0].set_ylabel('Jump Size %')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # Highlight synthetic data points
        synthetic_jumps = large_jumps[filled_df.loc[large_jumps.index].get('is_synthetic', False)]
        if len(synthetic_jumps) > 0:
            axes[2, 0].scatter(synthetic_jumps.index, synthetic_jumps.values * 100, 
                             color='red', alpha=0.8, s=30, label='At synthetic data')
            axes[2, 0].legend()
    else:
        axes[2, 0].text(0.5, 0.5, 'No large price jumps', transform=axes[2, 0].transAxes, ha='center')
    
    # 6. Data quality metrics
    total_records = len(filled_df)
    synthetic_count = len(synthetic_data)
    real_count = len(real_data)
    
    quality_metrics = {
        'Real Data': real_count,
        'Synthetic Data': synthetic_count
    }
    
    colors = ['green', 'orange']
    axes[2, 1].pie(quality_metrics.values(), labels=quality_metrics.keys(), 
                   autopct='%1.1f%%', colors=colors, alpha=0.7)
    axes[2, 1].set_title(f'{symbol} - Data Composition')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'{symbol}_post_fill_analysis.png', dpi=300, bbox_inches='tight')
    log.info(f"Saved post-fill analysis to {plots_dir / f'{symbol}_post_fill_analysis.png'}")
    plt.close()

def comprehensive_orderbook_analysis(symbol, orderbook_df):
    """Comprehensive orderbook analysis with detailed visualizations"""
    log.info(f"\n=== COMPREHENSIVE ORDERBOOK ANALYSIS FOR {symbol} ===")
    
    if orderbook_df is None or len(orderbook_df) == 0:
        log.warning("No orderbook data for comprehensive analysis")
        return
    
    # Create orderbook analysis plots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    
    # Sample data for performance (use every 100th record)
    sample_df = orderbook_df.iloc[::100]
    log.info(f"Using {len(sample_df):,} samples for orderbook analysis")
    
    # 1. Spread over time
    valid_quotes = sample_df.dropna(subset=['bid1_price', 'ask1_price'])
    if len(valid_quotes) > 0:
        spreads = valid_quotes['ask1_price'] - valid_quotes['bid1_price']
        spread_pct = spreads / valid_quotes['bid1_price'] * 100
        
        # Daily average spreads
        daily_spreads = spread_pct.groupby(spread_pct.index.date).mean()
        
        axes[0, 0].plot(daily_spreads.index, daily_spreads.values, linewidth=1, alpha=0.8)
        axes[0, 0].set_title(f'{symbol} - Daily Average Spread %')
        axes[0, 0].set_ylabel('Spread %')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Spread distribution
        axes[0, 1].hist(spread_pct, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'{symbol} - Spread Distribution')
        axes[0, 1].set_xlabel('Spread %')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        
        # Add statistics
        axes[0, 1].axvline(spread_pct.mean(), color='red', linestyle='--', label=f'Mean: {spread_pct.mean():.4f}%')
        axes[0, 1].axvline(spread_pct.median(), color='orange', linestyle='--', label=f'Median: {spread_pct.median():.4f}%')
        axes[0, 1].legend()
    
    # 3. Bid-Ask sizes
    valid_sizes = sample_df.dropna(subset=['bid1_size', 'ask1_size'])
    if len(valid_sizes) > 0:
        axes[1, 0].scatter(valid_sizes['bid1_size'], valid_sizes['ask1_size'], alpha=0.5, s=1)
        axes[1, 0].set_title(f'{symbol} - Bid vs Ask Sizes')
        axes[1, 0].set_xlabel('Bid1 Size')
        axes[1, 0].set_ylabel('Ask1 Size')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        min_size = min(valid_sizes['bid1_size'].min(), valid_sizes['ask1_size'].min())
        max_size = max(valid_sizes['bid1_size'].max(), valid_sizes['ask1_size'].max())
        axes[1, 0].plot([min_size, max_size], [min_size, max_size], 'r--', alpha=0.5, label='Equal sizes')
        axes[1, 0].legend()
    
    # 4. Size imbalance over time
    if len(valid_sizes) > 0:
        size_imbalance = (valid_sizes['bid1_size'] - valid_sizes['ask1_size']) / (valid_sizes['bid1_size'] + valid_sizes['ask1_size'])
        daily_imbalance = size_imbalance.groupby(size_imbalance.index.date).mean()
        
        axes[1, 1].plot(daily_imbalance.index, daily_imbalance.values, linewidth=1, alpha=0.8)
        axes[1, 1].set_title(f'{symbol} - Daily Size Imbalance')
        axes[1, 1].set_ylabel('Imbalance (Bid-Ask)/(Bid+Ask)')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 5. Depth analysis (levels 1-3)
    depth_coverage = {}
    for level in [1, 2, 3]:
        bid_col = f'bid{level}_price'
        ask_col = f'ask{level}_price'
        if bid_col in sample_df.columns and ask_col in sample_df.columns:
            valid_level = (~sample_df[[bid_col, ask_col]].isna()).all(axis=1).sum()
            depth_coverage[f'Level {level}'] = valid_level / len(sample_df) * 100
    
    if depth_coverage:
        axes[2, 0].bar(depth_coverage.keys(), depth_coverage.values(), alpha=0.7)
        axes[2, 0].set_title(f'{symbol} - Orderbook Depth Coverage')
        axes[2, 0].set_ylabel('Coverage %')
        axes[2, 0].set_ylim(0, 105)
        
        # Add value labels on bars
        for i, (k, v) in enumerate(depth_coverage.items()):
            axes[2, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 6. Mid-price volatility
    if len(valid_quotes) > 0:
        mid_price = (valid_quotes['bid1_price'] + valid_quotes['ask1_price']) / 2
        mid_returns = mid_price.pct_change().dropna()
        
        if len(mid_returns) > 0:
            axes[2, 1].hist(mid_returns * 100, bins=50, alpha=0.7, edgecolor='black')
            axes[2, 1].set_title(f'{symbol} - Mid-Price Return Distribution')
            axes[2, 1].set_xlabel('Mid-Price Returns %')
            axes[2, 1].set_ylabel('Frequency')
            axes[2, 1].set_yscale('log')
            
            # Add statistics
            vol = mid_returns.std() * 100
            axes[2, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
            axes[2, 1].text(0.7, 0.9, f'Volatility: {vol:.4f}%', transform=axes[2, 1].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'{symbol}_orderbook_analysis.png', dpi=300, bbox_inches='tight')
    log.info(f"Saved orderbook analysis to {plots_dir / f'{symbol}_orderbook_analysis.png'}")
    plt.close()

def comprehensive_ohlcv_analysis(symbol, filled_df):
    """Comprehensive OHLCV analysis with detailed visualizations"""
    log.info(f"\n=== COMPREHENSIVE OHLCV ANALYSIS FOR {symbol} ===")
    
    if filled_df is None or len(filled_df) == 0:
        log.warning("No OHLCV data for comprehensive analysis")
        return
    
    # Create OHLCV analysis plots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    
    # Separate real and synthetic data
    is_synthetic = filled_df.get('is_synthetic', pd.Series(False, index=filled_df.index))
    real_data = filled_df[~is_synthetic]
    synthetic_data = filled_df[is_synthetic]
    
    log.info(f"Real records: {len(real_data):,}, Synthetic records: {len(synthetic_data):,}")
    
    # 1. Price evolution with synthetic markers
    daily_prices = filled_df.groupby(filled_df.index.date)['close'].last()
    axes[0, 0].plot(daily_prices.index, daily_prices.values, linewidth=1, alpha=0.8, label='Price')
    
    # Mark days with synthetic data
    if len(synthetic_data) > 0:
        synthetic_days = synthetic_data.groupby(synthetic_data.index.date).size()
        for day, count in synthetic_days.items():
            if day in daily_prices.index:
                axes[0, 0].scatter(day, daily_prices.loc[day], color='red', s=count/10, alpha=0.6)
    
    axes[0, 0].set_title(f'{symbol} - Price Evolution (Red dots = Synthetic data)')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Volume analysis
    real_volume = real_data['volume'][real_data['volume'] > 0]
    
    if len(real_volume) > 0:
        axes[0, 1].hist(real_volume, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'{symbol} - Volume Distribution (Real Data Only)')
        axes[0, 1].set_xlabel('Volume')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        
        # Add percentile lines
        percentiles = [25, 50, 75, 95]
        colors = ['green', 'orange', 'red', 'purple']
        for p, color in zip(percentiles, colors):
            val = real_volume.quantile(p/100)
            axes[0, 1].axvline(val, color=color, linestyle='--', alpha=0.7, label=f'P{p}: {val:.0f}')
        axes[0, 1].legend()
    
    # 3. Return analysis
    returns = filled_df['close'].pct_change().dropna()
    real_returns = real_data['close'].pct_change().dropna()
    
    if len(returns) > 0:
        axes[1, 0].hist(real_returns * 100, bins=50, alpha=0.7, edgecolor='black', label='Real data')
        
        # Overlay synthetic transitions if any
        synthetic_indices = synthetic_data.index
        if len(synthetic_indices) > 0:
            # Find returns at synthetic data points
            synthetic_returns = returns[returns.index.isin(synthetic_indices)]
            if len(synthetic_returns) > 0:
                axes[1, 0].hist(synthetic_returns * 100, bins=50, alpha=0.5, 
                               edgecolor='red', color='red', label='At synthetic points')
        
        axes[1, 0].set_title(f'{symbol} - Return Distribution')
        axes[1, 0].set_xlabel('1-minute Returns %')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        
        # Add statistics
        vol = real_returns.std() * 100
        skew = real_returns.skew()
        axes[1, 0].text(0.7, 0.9, f'Volatility: {vol:.3f}%\nSkewness: {skew:.3f}', 
                       transform=axes[1, 0].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 4. OHLC relationships
    if len(filled_df) > 0:
        # High-Low range as % of close
        hl_range = (filled_df['high'] - filled_df['low']) / filled_df['close'] * 100
        
        axes[1, 1].hist(hl_range, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title(f'{symbol} - High-Low Range Distribution')
        axes[1, 1].set_xlabel('Range % of Close')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_yscale('log')
        
        # Mark synthetic data ranges
        if len(synthetic_data) > 0:
            synthetic_hl = (synthetic_data['high'] - synthetic_data['low']) / synthetic_data['close'] * 100
            if len(synthetic_hl) > 0:
                axes[1, 1].axvline(synthetic_hl.mean(), color='red', linestyle='--', 
                                  label=f'Synthetic avg: {synthetic_hl.mean():.3f}%')
                axes[1, 1].legend()
    
    # 5. Trading patterns by hour
    hourly_volume = filled_df.groupby(filled_df.index.hour)['volume'].mean()
    hourly_count = filled_df.groupby(filled_df.index.hour).size()
    
    ax5_twin = axes[2, 0].twinx()
    
    bars1 = axes[2, 0].bar(hourly_volume.index, hourly_volume.values, alpha=0.7, color='blue', label='Avg Volume')
    bars2 = ax5_twin.bar(hourly_count.index + 0.3, hourly_count.values, alpha=0.7, color='orange', width=0.6, label='Record Count')
    
    axes[2, 0].set_title(f'{symbol} - Hourly Trading Patterns')
    axes[2, 0].set_xlabel('Hour (UTC)')
    axes[2, 0].set_ylabel('Average Volume', color='blue')
    ax5_twin.set_ylabel('Record Count', color='orange')
    axes[2, 0].set_xticks(range(0, 24, 2))
    
    # Add synthetic data count by hour
    if len(synthetic_data) > 0:
        hourly_synthetic = synthetic_data.groupby(synthetic_data.index.hour).size()
        for hour, count in hourly_synthetic.items():
            axes[2, 0].text(hour, hourly_volume.get(hour, 0) * 1.1, f'S:{count}', 
                           ha='center', va='bottom', color='red', fontsize=8)
    
    # 6. Data quality heatmap
    # Create a heatmap showing data availability by hour and day of week
    filled_df_copy = filled_df.copy()
    filled_df_copy['hour'] = filled_df_copy.index.hour
    filled_df_copy['weekday'] = filled_df_copy.index.day_name()
    
    # Count records by hour and weekday
    heatmap_data = filled_df_copy.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
    
    # Reorder weekdays
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(weekday_order)
    
    im = axes[2, 1].imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[2, 1].set_title(f'{symbol} - Data Availability Heatmap')
    axes[2, 1].set_xlabel('Hour (UTC)')
    axes[2, 1].set_ylabel('Day of Week')
    axes[2, 1].set_xticks(range(0, 24, 2))
    axes[2, 1].set_xticklabels(range(0, 24, 2))
    axes[2, 1].set_yticks(range(len(weekday_order)))
    axes[2, 1].set_yticklabels(weekday_order)
    
    # Add colorbar
    plt.colorbar(im, ax=axes[2, 1], label='Record Count')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'{symbol}_ohlcv_analysis.png', dpi=300, bbox_inches='tight')
    log.info(f"Saved OHLCV analysis to {plots_dir / f'{symbol}_ohlcv_analysis.png'}")
    plt.close()
    """Create visualizations showing before/after cleaning"""
    log.info("\n=== CREATING CLEANING VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(len(symbols_analysis), 3, figsize=(20, 6*len(symbols_analysis)))
    if len(symbols_analysis) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (symbol, analysis) in enumerate(symbols_analysis.items()):
        original_df = analysis['original_df']
        filled_df = analysis.get('filled_df', original_df)
        gaps_filled = analysis.get('gaps_filled', [])
        gaps_skipped = analysis.get('gaps_skipped', [])
        
        # Timeline coverage comparison
        start_date = original_df.index.min()
        end_date = original_df.index.max()
        expected_timeline = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        # Original coverage
        original_coverage = pd.Series(index=expected_timeline, data=False)
        original_coverage.loc[original_df.index] = True
        
        # Filled coverage
        filled_coverage = pd.Series(index=expected_timeline, data=False)
        filled_coverage.loc[filled_df.index] = True
        
        # Daily aggregation for visualization
        daily_original = original_coverage.resample('D').mean()
        daily_filled = filled_coverage.resample('D').mean()
        
        # Plot 1: Coverage comparison
        axes[i, 0].plot(daily_original.index, daily_original.values * 100, 
                       label='Original', alpha=0.7, linewidth=1)
        axes[i, 0].plot(daily_filled.index, daily_filled.values * 100, 
                       label='After Fill', alpha=0.7, linewidth=1)
        axes[i, 0].set_title(f'{symbol} - Daily Coverage %')
        axes[i, 0].set_ylabel('Coverage %')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylim(0, 105)
        
        # Plot 2: Gap analysis
        if gaps_filled or gaps_skipped:
            gap_sizes_filled = [gap['duration'] for gap in gaps_filled]
            gap_sizes_skipped = [gap['duration'] for gap in gaps_skipped]
            
            bins = [0, 5, 15, 60, 360, 720, float('inf')]
            labels = ['≤5min', '6-15min', '16-60min', '1-6h', '6-12h', '>12h']
            
            filled_counts = pd.cut(gap_sizes_filled, bins=bins, labels=labels).value_counts() if gap_sizes_filled else pd.Series(dtype=int)
            skipped_counts = pd.cut(gap_sizes_skipped, bins=bins, labels=labels).value_counts() if gap_sizes_skipped else pd.Series(dtype=int)
            
            # Combine for plotting
            gap_analysis = pd.DataFrame({
                'Filled': filled_counts,
                'Skipped': skipped_counts
            }).fillna(0)
            
            axes[i, 1].set_xlabel('Gap Duration')
            axes[i, 1].tick_params(axis='x', rotation=45)
        else:
            axes[i, 1].text(0.5, 0.5, 'No gaps to analyze', transform=axes[i, 1].transAxes, ha='center')
            axes[i, 1].set_title(f'{symbol} - No Gaps Found')
        
        # Plot 3: Volume impact
        if 'volume' in original_df.columns and 'volume' in filled_df.columns:
            # Daily volume comparison
            daily_volume_orig = original_df.groupby(original_df.index.date)['volume'].sum()
            daily_volume_filled = filled_df.groupby(filled_df.index.date)['volume'].sum()
            
            axes[i, 2].plot(daily_volume_orig.index, daily_volume_orig.values, 
                           label='Original', alpha=0.7, linewidth=1)
            axes[i, 2].plot(daily_volume_filled.index, daily_volume_filled.values, 
                           label='After Fill', alpha=0.7, linewidth=1)
            axes[i, 2].set_title(f'{symbol} - Daily Volume')
            axes[i, 2].set_ylabel('Volume')
            axes[i, 2].legend()
            axes[i, 2].tick_params(axis='x', rotation=45)
        else:
            axes[i, 2].text(0.5, 0.5, 'No volume data', transform=axes[i, 2].transAxes, ha='center')
            axes[i, 2].set_title(f'{symbol} - Volume Analysis')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'data_cleaning_results.png', dpi=300, bbox_inches='tight')
    log.info(f"Saved cleaning results to {plots_dir / 'data_cleaning_results.png'}")
    plt.close()

def generate_cleaning_summary(symbols_analysis):
    """Generate comprehensive cleaning summary"""
    log.info("\n" + "="*80)
    log.info("DATA CLEANING SUMMARY")
    log.info("="*80)
    
    for symbol, analysis in symbols_analysis.items():
        log.info(f"\n{symbol} CLEANING RESULTS:")
        
        original_df = analysis['original_df']
        filled_df = analysis.get('filled_df', original_df)
        gaps_filled = analysis.get('gaps_filled', [])
        gaps_skipped = analysis.get('gaps_skipped', [])
        filled_records = analysis.get('filled_records', [])
        
        # Original stats
        original_count = len(original_df)
        original_synthetic = original_df['is_synthetic'].sum() if 'is_synthetic' in original_df.columns else 0
        
        # After cleaning stats
        new_synthetic = len(filled_records)
        total_after = len(filled_df)
        total_synthetic_after = original_synthetic + new_synthetic
        
        log.info(f"  Original records: {original_count:,}")
        log.info(f"  Records after cleaning: {total_after:,}")
        log.info(f"  New synthetic records: {new_synthetic:,}")
        log.info(f"  Total synthetic records: {total_synthetic_after:,} ({total_synthetic_after/total_after*100:.1f}%)")
        
        # Gap analysis
        if gaps_filled:
            total_filled_minutes = sum(gap['duration'] for gap in gaps_filled)
            log.info(f"  Gaps filled: {len(gaps_filled)} ({total_filled_minutes:,} minutes)")
            
            # Break down by size
            small_gaps = [g for g in gaps_filled if g['duration'] <= 5]
            medium_gaps = [g for g in gaps_filled if 5 < g['duration'] <= 60]
            
            log.info(f"    Small gaps (≤5min): {len(small_gaps)}")
            log.info(f"    Medium gaps (6-60min): {len(medium_gaps)}")
        
        if gaps_skipped:
            total_skipped_minutes = sum(gap['duration'] for gap in gaps_skipped)
            log.info(f"  Gaps skipped: {len(gaps_skipped)} ({total_skipped_minutes:,} minutes)")
        
        # Coverage improvement
        start_date = original_df.index.min()
        end_date = original_df.index.max()
        expected_minutes = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        original_coverage = len(original_df) / len(expected_minutes) * 100
        filled_coverage = len(filled_df) / len(expected_minutes) * 100
        
        log.info(f"  Coverage improvement: {original_coverage:.1f}% → {filled_coverage:.1f}% (+{filled_coverage-original_coverage:.1f}%)")
        
        # Data quality assessment
        if filled_coverage >= 95:
            quality = "EXCELLENT"
        elif filled_coverage >= 85:
            quality = "GOOD"
        elif filled_coverage >= 70:
            quality = "FAIR"
        else:
            quality = "POOR"
        
        log.info(f"  Final quality assessment: {quality}")
    
    log.info("\n" + "="*80)

def main():
    """Main data cleaning function with comprehensive analysis"""
    log.info("Starting comprehensive data cleaning with detailed analysis")
    
    try:
        # Add synthetic data columns to database
        add_synthetic_data_columns()
        
        # Define symbols to clean
        symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
        symbols_analysis = {}
        
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"PROCESSING {symbol}")
            log.info(f"{'='*60}")
            
            # Load and analyze data
            ohlcv_df = load_and_analyze_ohlcv_data(symbol)
            orderbook_df = load_and_analyze_orderbook_data(symbol)
            
            if ohlcv_df is None:
                log.warning(f"Skipping {symbol} - no OHLCV data")
                continue
            
            # ===== PRE-FILL ANALYSIS =====
            log.info(f"\n{'='*40}")
            log.info("PRE-FILL COMPREHENSIVE ANALYSIS")
            log.info(f"{'='*40}")
            
            # Create pre-fill visualizations
            create_pre_fill_analysis(symbol, ohlcv_df, orderbook_df)
            
            # Detailed quality analysis
            analyze_orderbook_quality(symbol, orderbook_df)
            validate_ohlcv_coherence(symbol, ohlcv_df)
            cross_validate_ohlcv_orderbook(symbol, ohlcv_df, orderbook_df)
            
            # ===== GAP IDENTIFICATION AND FILLING =====
            log.info(f"\n{'='*40}")
            log.info("GAP IDENTIFICATION AND FORWARD FILL")
            log.info(f"{'='*40}")
            
            # Identify gaps for filling
            gaps_to_fill, gaps_to_skip = identify_gaps_for_filling(symbol, ohlcv_df)
            
            # Perform forward fill
            filled_records, filled_df = perform_forward_fill(symbol, ohlcv_df, gaps_to_fill)
            
            # Insert synthetic data into database
            if filled_records:
                insert_synthetic_data(symbol, filled_records)
            
            # ===== POST-FILL ANALYSIS =====
            log.info(f"\n{'='*40}")
            log.info("POST-FILL COMPREHENSIVE ANALYSIS")
            log.info(f"{'='*40}")
            
            # Create post-fill visualizations
            create_post_fill_analysis(symbol, ohlcv_df, filled_df, gaps_to_fill, gaps_to_skip)
            
            # Comprehensive orderbook analysis
            comprehensive_orderbook_analysis(symbol, orderbook_df)
            
            # Comprehensive OHLCV analysis (post-fill)
            comprehensive_ohlcv_analysis(symbol, filled_df)
            
            # ===== FINAL VALIDATION =====
            log.info(f"\n{'='*40}")
            log.info("POST-FILL VALIDATION")
            log.info(f"{'='*40}")
            
            # Re-validate data quality after fill
            validate_ohlcv_coherence(symbol, filled_df)
            
            # Store analysis results
            symbols_analysis[symbol] = {
                'original_df': ohlcv_df,
                'filled_df': filled_df,
                'orderbook_df': orderbook_df,
                'gaps_filled': gaps_to_fill,
                'gaps_skipped': gaps_to_skip,
                'filled_records': filled_records
            }
        
        # ===== FINAL SUMMARY =====
        log.info(f"\n{'='*60}")
        log.info("FINAL CLEANING AND ANALYSIS SUMMARY")
        log.info(f"{'='*60}")
        
        # Generate comprehensive summary
        if symbols_analysis:
            generate_cleaning_summary(symbols_analysis)
            
            # Create comparison visualizations
            create_final_comparison_plots(symbols_analysis)
        
        log.info("\n" + "="*80)
        log.info("COMPREHENSIVE DATA CLEANING COMPLETED SUCCESSFULLY!")
        log.info("="*80)
        log.info("\nGenerated plots:")
        
        for symbol in symbols:
            log.info(f"\n{symbol}:")
            log.info(f"  - {symbol}_pre_fill_analysis.png")
            log.info(f"  - {symbol}_post_fill_analysis.png") 
            log.info(f"  - {symbol}_orderbook_analysis.png")
            log.info(f"  - {symbol}_ohlcv_analysis.png")
        
        log.info(f"\nComparison:")
        log.info(f"  - symbols_comparison.png")
        log.info(f"  - data_cleaning_results.png")
        
        log.info(f"\nAll plots saved in: {plots_dir}")
        
        return True
        
    except Exception as e:
        log.error(f"Comprehensive data cleaning failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

def create_final_comparison_plots(symbols_analysis):
    """Create final comparison plots between symbols"""
    log.info("\n=== CREATING FINAL COMPARISON PLOTS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    symbols = list(symbols_analysis.keys())
    colors = ['blue', 'orange']
    
    # 1. Coverage comparison
    for i, (symbol, analysis) in enumerate(symbols_analysis.items()):
        original_df = analysis['original_df']
        filled_df = analysis['filled_df']
        
        # Calculate daily coverage
        start_date = original_df.index.min()
        end_date = original_df.index.max()
        expected_timeline = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        original_coverage = pd.Series(index=expected_timeline, data=False)
        original_coverage.loc[original_df.index] = True
        filled_coverage = pd.Series(index=expected_timeline, data=False)
        filled_coverage.loc[filled_df.index] = True
        
        daily_original = original_coverage.resample('D').mean() * 100
        daily_filled = filled_coverage.resample('D').mean() * 100
        
        axes[0, 0].plot(daily_original.index, daily_original.values, 
                       label=f'{symbol} Original', alpha=0.7, color=colors[i], linestyle='--')
        axes[0, 0].plot(daily_filled.index, daily_filled.values, 
                       label=f'{symbol} Filled', alpha=0.8, color=colors[i], linewidth=2)
    
    axes[0, 0].set_title('Coverage Comparison: Before vs After Fill')
    axes[0, 0].set_ylabel('Coverage %')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 105)
    
    # 2. Synthetic data impact
    synthetic_summary = {}
    for symbol, analysis in symbols_analysis.items():
        filled_df = analysis['filled_df']
        total_records = len(filled_df)
        synthetic_count = filled_df.get('is_synthetic', pd.Series(False, index=filled_df.index)).sum()
        synthetic_pct = (synthetic_count / total_records) * 100 if total_records > 0 else 0
        synthetic_summary[symbol] = synthetic_pct
    
    axes[0, 1].bar(synthetic_summary.keys(), synthetic_summary.values(), 
                   color=colors, alpha=0.7)
    axes[0, 1].set_title('Synthetic Data Percentage by Symbol')
    axes[0, 1].set_ylabel('Synthetic Data %')
    axes[0, 1].set_ylim(0, max(synthetic_summary.values()) * 1.2 if synthetic_summary.values() else 5)
    
    # Add value labels
    for i, (symbol, pct) in enumerate(synthetic_summary.items()):
        axes[0, 1].text(i, pct + 0.1, f'{pct:.1f}%', ha='center', va='bottom')
    
    # 3. Gap treatment comparison
    gap_comparison = {}
    for symbol, analysis in symbols_analysis.items():
        gaps_filled = analysis['gaps_filled']
        gaps_skipped = analysis['gaps_skipped']
        
        filled_minutes = sum(gap['duration'] for gap in gaps_filled)
        skipped_minutes = sum(gap['duration'] for gap in gaps_skipped)
        
        gap_comparison[symbol] = {
            'Filled': filled_minutes,
            'Skipped': skipped_minutes
        }
    
    # Create stacked bar chart
    symbols_list = list(gap_comparison.keys())
    filled_values = [gap_comparison[s]['Filled'] for s in symbols_list]
    skipped_values = [gap_comparison[s]['Skipped'] for s in symbols_list]
    
    x_pos = range(len(symbols_list))
    axes[1, 0].bar(x_pos, filled_values, label='Filled', alpha=0.7, color='green')
    axes[1, 0].bar(x_pos, skipped_values, bottom=filled_values, label='Skipped', alpha=0.7, color='red')
    
    axes[1, 0].set_title('Gap Treatment Comparison (Minutes)')
    axes[1, 0].set_ylabel('Minutes')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([s.split('_')[-2] for s in symbols_list])  # Just the symbol part
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    
    # 4. Final quality assessment
    quality_scores = {}
    for symbol, analysis in symbols_analysis.items():
        filled_df = analysis['filled_df']
        
        # Calculate quality score based on multiple factors
        start_date = filled_df.index.min()
        end_date = filled_df.index.max()
        expected_timeline = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        coverage = len(filled_df) / len(expected_timeline)
        synthetic_ratio = filled_df.get('is_synthetic', pd.Series(False, index=filled_df.index)).mean()
        
        # Quality score: high coverage, low synthetic ratio
        quality_score = coverage * (1 - synthetic_ratio * 0.5)  # Penalize synthetic data
        quality_scores[symbol] = quality_score * 100
    
    axes[1, 1].bar(quality_scores.keys(), quality_scores.values(), 
                   color=colors, alpha=0.7)
    axes[1, 1].set_title('Final Data Quality Score')
    axes[1, 1].set_ylabel('Quality Score')
    axes[1, 1].set_ylim(0, 105)
    
    # Add value labels and quality assessment
    for i, (symbol, score) in enumerate(quality_scores.items()):
        axes[1, 1].text(i, score + 2, f'{score:.1f}', ha='center', va='bottom')
        
        # Add quality assessment text
        if score >= 95:
            quality_text = "EXCELLENT"
            text_color = "green"
        elif score >= 85:
            quality_text = "GOOD"
            text_color = "orange"
        elif score >= 70:
            quality_text = "FAIR"
            text_color = "red"
        else:
            quality_text = "POOR"
            text_color = "darkred"
        
        axes[1, 1].text(i, score - 10, quality_text, ha='center', va='center', 
                       color=text_color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'symbols_comparison.png', dpi=300, bbox_inches='tight')
    log.info(f"Saved symbols comparison to {plots_dir / 'symbols_comparison.png'}")
    plt.close()

def create_cleaning_visualizations(symbols_analysis):
    """Create summary cleaning visualizations (kept for compatibility)"""
    log.info("\n=== CREATING SUMMARY CLEANING VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(len(symbols_analysis), 2, figsize=(16, 6*len(symbols_analysis)))
    if len(symbols_analysis) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (symbol, analysis) in enumerate(symbols_analysis.items()):
        original_df = analysis['original_df']
        filled_df = analysis.get('filled_df', original_df)
        
        # Coverage improvement
        start_date = original_df.index.min()
        end_date = original_df.index.max()
        expected_timeline = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        original_coverage = pd.Series(index=expected_timeline, data=False)
        original_coverage.loc[original_df.index] = True
        filled_coverage = pd.Series(index=expected_timeline, data=False)
        filled_coverage.loc[filled_df.index] = True
        
        daily_original = original_coverage.resample('D').mean()
        daily_filled = filled_coverage.resample('D').mean()
        
        axes[i, 0].plot(daily_original.index, daily_original.values * 100, 
                       label='Original', alpha=0.7, linewidth=1)
        axes[i, 0].plot(daily_filled.index, daily_filled.values * 100, 
                       label='After Fill', alpha=0.7, linewidth=2)
        axes[i, 0].set_title(f'{symbol} - Coverage Improvement')
        axes[i, 0].set_ylabel('Coverage %')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylim(0, 105)
        
        # Data composition
        total_records = len(filled_df)
        synthetic_count = filled_df.get('is_synthetic', pd.Series(False, index=filled_df.index)).sum()
        real_count = total_records - synthetic_count
        
        composition = {'Real Data': real_count, 'Synthetic': synthetic_count}
        colors = ['green', 'orange']
        
        axes[i, 1].pie(composition.values(), labels=composition.keys(), 
                      autopct='%1.1f%%', colors=colors, alpha=0.7)
        axes[i, 1].set_title(f'{symbol} - Final Data Composition')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'data_cleaning_results.png', dpi=300, bbox_inches='tight')
    log.info(f"Saved cleaning summary to {plots_dir / 'data_cleaning_results.png'}")
    plt.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)