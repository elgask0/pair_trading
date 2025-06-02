#!/usr/bin/env python3
"""
Data analysis script - VERSION WITH IMPROVED PLOTS
Analyzes cleaned data and generates comprehensive reports and visualizations
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_cleaned_data(symbol: str) -> Dict:
    """Load cleaned data for analysis"""
    log.info(f"Loading cleaned data for {symbol}...")
    
    with db_manager.get_session() as session:
        # OHLCV data - ALL HISTORICAL DATA
        ohlcv_query = text("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        
        # Orderbook data with quality information - ALL HISTORICAL DATA
        orderbook_query = text("""
            SELECT timestamp, bid1_price, bid1_size, ask1_price, ask1_size,
                   liquidity_quality, valid_for_trading, spread_pct, threshold_p80
            FROM orderbook 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        
        try:
            ohlcv_df = pd.read_sql(ohlcv_query, session.bind, params={"symbol": symbol}, index_col='timestamp')
        except Exception as e:
            log.warning(f"Error loading OHLCV data for {symbol}: {e}")
            ohlcv_df = pd.DataFrame()
        
        try:
            orderbook_df = pd.read_sql(orderbook_query, session.bind, params={"symbol": symbol}, index_col='timestamp')
        except Exception as e:
            log.warning(f"Error loading orderbook data for {symbol}: {e}")
            orderbook_df = pd.DataFrame()
        
        # Quality summary - handle case where no quality data exists
        quality_summary = {}
        try:
            quality_result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as trading_ready,
                    COUNT(CASE WHEN liquidity_quality = 'Excellent' THEN 1 END) as excellent,
                    COUNT(CASE WHEN liquidity_quality = 'Good' THEN 1 END) as good,
                    COUNT(CASE WHEN liquidity_quality = 'Fair' THEN 1 END) as fair,
                    COUNT(CASE WHEN liquidity_quality = 'Poor' THEN 1 END) as poor,
                    COUNT(CASE WHEN liquidity_quality = 'Invalid' THEN 1 END) as invalid,
                    AVG(CASE WHEN spread_pct IS NOT NULL AND spread_pct <= 10.0 THEN spread_pct END) as avg_spread,
                    MAX(threshold_p80) as threshold_p80
                FROM orderbook 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            quality_summary = {
                'total_records': quality_result.total_records or 0,
                'trading_ready': quality_result.trading_ready or 0,
                'excellent': quality_result.excellent or 0,
                'good': quality_result.good or 0,
                'fair': quality_result.fair or 0,
                'poor': quality_result.poor or 0,
                'invalid': quality_result.invalid or 0,
                'avg_spread': quality_result.avg_spread or 0,
                'threshold_p80': quality_result.threshold_p80 or 0
            }
        except Exception as e:
            log.warning(f"Error loading quality summary for {symbol}: {e}")
            quality_summary = {
                'total_records': len(orderbook_df),
                'trading_ready': 0, 'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'invalid': 0,
                'avg_spread': 0, 'threshold_p80': 0
            }
        
        log.info(f"Loaded {len(ohlcv_df):,} OHLCV and {len(orderbook_df):,} orderbook records for {symbol}")
        
        return {
            'ohlcv': ohlcv_df,
            'orderbook': orderbook_df,
            'quality_summary': quality_summary
        }

def analyze_spread_quality(symbol: str, data: Dict) -> Dict:
    """Analyze spread quality after cleaning"""
    log.info(f"Analyzing spread quality for {symbol}...")
    
    orderbook_df = data['orderbook']
    quality_summary = data['quality_summary']
    
    if len(orderbook_df) == 0:
        log.warning(f"No orderbook data for {symbol}")
        return None
    
    # Calculate basic spreads if not already done
    if 'spread_pct' not in orderbook_df.columns or orderbook_df['spread_pct'].isna().all():
        log.info(f"Calculating spreads for {symbol}...")
        valid_quotes = orderbook_df.dropna(subset=['bid1_price', 'ask1_price'])
        if len(valid_quotes) > 0:
            spreads = (valid_quotes['ask1_price'] - valid_quotes['bid1_price']) / valid_quotes['bid1_price'] * 100
            # Update orderbook_df with calculated spreads
            orderbook_df.loc[valid_quotes.index, 'spread_pct'] = spreads
    
    # Filter valid spreads for detailed analysis
    valid_spreads = orderbook_df[
        (orderbook_df['spread_pct'].notna()) & 
        (orderbook_df['spread_pct'] <= 10.0) &
        (orderbook_df['spread_pct'] > 0)
    ]['spread_pct']
    
    if len(valid_spreads) == 0:
        log.warning(f"No valid spread data for {symbol}")
        return None
    
    # Use threshold from quality summary or calculate P80
    threshold = quality_summary.get('threshold_p80', 0)
    if threshold <= 0:
        threshold = valid_spreads.quantile(0.80)
        log.info(f"Calculated P80 threshold for {symbol}: {threshold:.4f}%")
    
    # Count trading-ready records
    trading_ready_count = quality_summary.get('trading_ready', 0)
    if trading_ready_count == 0 and 'valid_for_trading' in orderbook_df.columns:
        trading_ready_count = (orderbook_df['valid_for_trading'] == True).sum()
    
    # Calculate statistics
    analysis = {
        'symbol': symbol,
        'threshold_p80': threshold,
        'total_records': len(orderbook_df),
        'valid_spreads_count': len(valid_spreads),
        'trading_ready_count': trading_ready_count,
        'trading_ready_pct': trading_ready_count / len(orderbook_df) * 100 if len(orderbook_df) > 0 else 0,
        'spread_stats': {
            'all_valid': {
                'mean': valid_spreads.mean(),
                'median': valid_spreads.median(),
                'std': valid_spreads.std(),
                'min': valid_spreads.min(),
                'max': valid_spreads.max(),
                'p25': valid_spreads.quantile(0.25),
                'p75': valid_spreads.quantile(0.75),
                'p80': valid_spreads.quantile(0.80),
                'p95': valid_spreads.quantile(0.95),
                'p99': valid_spreads.quantile(0.99)
            }
        },
        'quality_breakdown': quality_summary
    }
    
    # Log results
    log.info(f"{symbol} Spread Quality Analysis:")
    log.info(f"  P80 Threshold: {threshold:.4f}%")
    log.info(f"  Trading Ready: {analysis['trading_ready_count']:,} ({analysis['trading_ready_pct']:.1f}%)")
    log.info(f"  Valid Spreads - Mean: {analysis['spread_stats']['all_valid']['mean']:.4f}%, Median: {analysis['spread_stats']['all_valid']['median']:.4f}%")
    
    return analysis

def analyze_ohlcv_patterns(symbol: str, data: Dict) -> Dict:
    """Analyze OHLCV patterns and characteristics"""
    log.info(f"Analyzing OHLCV patterns for {symbol}...")
    
    ohlcv_df = data['ohlcv'].copy()
    
    if len(ohlcv_df) == 0:
        log.warning(f"No OHLCV data for {symbol}")
        return None
    
    # Basic statistics
    returns = ohlcv_df['close'].pct_change().dropna()
    volume_positive = ohlcv_df[ohlcv_df['volume'] > 0]['volume']
    
    # Price analysis
    price_analysis = {
        'price_range': {
            'min': float(ohlcv_df['low'].min()),
            'max': float(ohlcv_df['high'].max()),
            'mean': float(ohlcv_df['close'].mean()),
            'current': float(ohlcv_df['close'].iloc[-1]) if len(ohlcv_df) > 0 else None
        },
        'returns': {
            'mean': float(returns.mean() * 100),
            'std': float(returns.std() * 100),
            'skew': float(returns.skew()) if len(returns) > 0 else 0,
            'kurtosis': float(returns.kurtosis()) if len(returns) > 0 else 0,
            'min': float(returns.min() * 100),
            'max': float(returns.max() * 100),
            'sharpe_approx': float(returns.mean() / returns.std()) if returns.std() > 0 else 0
        },
        'volume': {
            'mean': float(volume_positive.mean()) if len(volume_positive) > 0 else 0,
            'median': float(volume_positive.median()) if len(volume_positive) > 0 else 0,
            'zero_volume_pct': float((ohlcv_df['volume'] == 0).mean() * 100),
            'total_volume': float(ohlcv_df['volume'].sum())
        }
    }
    
    # Time-based patterns
    ohlcv_df['hour'] = ohlcv_df.index.hour
    ohlcv_df['day_of_week'] = ohlcv_df.index.dayofweek
    
    hourly_volume = ohlcv_df.groupby('hour')['volume'].mean()
    daily_volume = ohlcv_df.groupby('day_of_week')['volume'].mean()
    
    patterns = {
        'hourly_volume': hourly_volume.to_dict(),
        'daily_volume': daily_volume.to_dict(),
        'peak_volume_hour': int(hourly_volume.idxmax()) if len(hourly_volume) > 0 else None,
        'peak_volume_day': int(daily_volume.idxmax()) if len(daily_volume) > 0 else None
    }
    
    analysis = {
        'symbol': symbol,
        'data_period': {
            'start': ohlcv_df.index.min(),
            'end': ohlcv_df.index.max(),
            'total_records': len(ohlcv_df),
            'total_days': (ohlcv_df.index.max() - ohlcv_df.index.min()).days
        },
        'price_analysis': price_analysis,
        'patterns': patterns
    }
    
    # Log results
    log.info(f"{symbol} OHLCV Analysis:")
    log.info(f"  Data Period: {analysis['data_period']['start']} to {analysis['data_period']['end']}")
    log.info(f"  Total Days: {analysis['data_period']['total_days']}")
    log.info(f"  Records: {analysis['data_period']['total_records']:,}")
    log.info(f"  Price Range: ${price_analysis['price_range']['min']:.6f} - ${price_analysis['price_range']['max']:.6f}")
    log.info(f"  Returns - Mean: {price_analysis['returns']['mean']:.4f}%, Std: {price_analysis['returns']['std']:.4f}%")
    log.info(f"  Volume - Zero volume: {price_analysis['volume']['zero_volume_pct']:.1f}%")
    
    return analysis

def create_spread_analysis_plot(symbol: str, analysis: Dict, data: Dict):
    """Create comprehensive spread analysis plots with IMPROVED FORMATTING"""
    log.info(f"Creating spread analysis plot for {symbol}...")
    
    orderbook_df = data['orderbook']
    
    # Ensure we have spread data
    if 'spread_pct' not in orderbook_df.columns:
        log.warning(f"No spread data available for plotting {symbol}")
        return
    
    valid_spreads = orderbook_df[
        (orderbook_df['spread_pct'].notna()) & 
        (orderbook_df['spread_pct'] <= 10.0) &
        (orderbook_df['spread_pct'] > 0)
    ]['spread_pct']
    
    if len(valid_spreads) == 0:
        log.warning(f"No valid spreads to plot for {symbol}")
        return
    
    threshold = analysis['threshold_p80']
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)
    
    # 1. Spread distribution with quality zones (IMPROVED)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate smart bins and limits
    spread_99 = valid_spreads.quantile(0.99)
    spread_display = valid_spreads[valid_spreads <= spread_99]  # Remove extreme outliers for display
    
    bins = min(50, max(20, len(spread_display) // 100))
    n, bins_edges, patches = ax1.hist(spread_display, bins=bins, alpha=0.7, 
                                     edgecolor='black', density=True, color='skyblue')
    
    # Mark quality zones with better visibility
    if threshold > 0:
        ax1.axvline(threshold * 0.5, color='green', linestyle='-', linewidth=3, alpha=0.8,
                   label=f'Excellent: ≤{threshold*0.5:.3f}%')
        ax1.axvline(threshold, color='orange', linestyle='-', linewidth=3, alpha=0.8,
                   label=f'Good: ≤{threshold:.3f}% (P80)')
        ax1.axvline(threshold * 1.5, color='red', linestyle='-', linewidth=3, alpha=0.8,
                   label=f'Fair: ≤{threshold*1.5:.3f}%')
    
    ax1.set_title(f'{symbol_short} - Spread Distribution with Quality Zones', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Spread %', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Set smart y-axis limits
    ax1.set_xlim(0, min(threshold * 2.5 if threshold > 0 else spread_99, spread_99))
    
    # 2. Quality pie chart (IMPROVED)
    ax2 = fig.add_subplot(gs[0, 1])
    
    quality_counts = [
        analysis['quality_breakdown']['excellent'],
        analysis['quality_breakdown']['good'], 
        analysis['quality_breakdown']['fair'],
        analysis['quality_breakdown']['poor'],
        analysis['quality_breakdown']['invalid']
    ]
    quality_labels = ['Excellent', 'Good', 'Fair', 'Poor', 'Invalid']
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#95a5a6']
    
    # Only show non-zero categories
    non_zero_indices = [i for i, count in enumerate(quality_counts) if count > 0]
    if non_zero_indices:
        filtered_counts = [quality_counts[i] for i in non_zero_indices]
        filtered_labels = [quality_labels[i] for i in non_zero_indices]
        filtered_colors = [colors[i] for i in non_zero_indices]
        
        wedges, texts, autotexts = ax2.pie(filtered_counts, labels=filtered_labels, colors=filtered_colors,
                                          autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
        
        # Improve text visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, 'No quality data\navailable', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    ax2.set_title(f'{symbol_short} - Data Quality Distribution', fontsize=12, fontweight='bold')
    
    # 3. Time series of spreads (IMPROVED)
    ax3 = fig.add_subplot(gs[1, :])  # Full width
    
    if len(orderbook_df) > 0:
        # Smart sampling for time series
        max_points = 2000
        if len(orderbook_df) > max_points:
            step = len(orderbook_df) // max_points
            sampled_data = orderbook_df.iloc[::step]
        else:
            sampled_data = orderbook_df
        
        spreads_to_plot = sampled_data[sampled_data['spread_pct'].notna()]
        
        if len(spreads_to_plot) > 0:
            # Plot with better formatting
            ax3.plot(spreads_to_plot.index, spreads_to_plot['spread_pct'], 
                    alpha=0.6, linewidth=0.8, color='steelblue', marker=None)
            
            # Add quality zones as horizontal bands
            if threshold > 0:
                ax3.axhline(threshold, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                           label=f'P80: {threshold:.3f}%')
                ax3.fill_between(spreads_to_plot.index, 0, threshold*0.5, alpha=0.1, color='green', label='Excellent Zone')
                ax3.fill_between(spreads_to_plot.index, threshold*0.5, threshold, alpha=0.1, color='yellow', label='Good Zone')
            
            ax3.set_title(f'{symbol_short} - Historical Spread Evolution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Time', fontsize=11)
            ax3.set_ylabel('Spread %', fontsize=11)
            ax3.legend(fontsize=9, loc='upper right')
            ax3.grid(True, alpha=0.3)
            
            # Smart y-axis limits
            y_max = min(threshold * 4 if threshold > 0 else valid_spreads.quantile(0.95), 
                       valid_spreads.quantile(0.98))
            ax3.set_ylim(0, y_max)
            
            # Format x-axis dates
            if len(spreads_to_plot) > 0:
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Trading readiness over time (IMPROVED)
    ax4 = fig.add_subplot(gs[2, 0])
    
    if len(orderbook_df) > 0 and 'valid_for_trading' in orderbook_df.columns:
        # Daily aggregation with better sampling
        daily_readiness = orderbook_df.groupby(orderbook_df.index.date).agg({
            'valid_for_trading': 'mean'
        })['valid_for_trading'] * 100
        
        # Smart sampling for display
        if len(daily_readiness) > 60:
            step = len(daily_readiness) // 60
            daily_readiness = daily_readiness.iloc[::step]
        
        if len(daily_readiness) > 0:
            ax4.plot(daily_readiness.index, daily_readiness.values, 
                    linewidth=2, marker='o', markersize=3, color='steelblue', alpha=0.8)
            
            # Add target lines
            ax4.axhline(80, color='red', linestyle='--', alpha=0.7, linewidth=2, label='80% Target')
            ax4.axhline(60, color='orange', linestyle=':', alpha=0.7, linewidth=1, label='60% Minimum')
            
            ax4.set_title(f'{symbol_short} - Trading Readiness Over Time', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Date', fontsize=11)
            ax4.set_ylabel('Trading Ready %', fontsize=11)
            ax4.set_ylim(0, 105)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            # Format dates
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No trading readiness\ndata available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_title(f'{symbol_short} - Trading Readiness', fontsize=12, fontweight='bold')
    
    # 5. Spread statistics summary (NEW)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Create statistics table
    stats = analysis['spread_stats']['all_valid']
    stats_text = [
        ['Metric', 'Value'],
        ['Mean Spread', f"{stats['mean']:.4f}%"],
        ['Median Spread', f"{stats['median']:.4f}%"],
        ['P80 Threshold', f"{threshold:.4f}%"],
        ['P95 Spread', f"{stats['p95']:.4f}%"],
        ['Min Spread', f"{stats['min']:.4f}%"],
        ['Max Spread', f"{stats['max']:.4f}%"],
        ['Trading Ready', f"{analysis['trading_ready_pct']:.1f}%"],
        ['Total Records', f"{analysis['total_records']:,}"]
    ]
    
    # Create table with better formatting
    table = ax5.table(cellText=stats_text[1:], colLabels=stats_text[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Style the table
    table[(0, 0)].set_facecolor('#3498db')
    table[(0, 1)].set_facecolor('#3498db')
    table[(0, 0)].set_text_props(weight='bold', color='white')
    table[(0, 1)].set_text_props(weight='bold', color='white')
    
    ax5.set_title(f'{symbol_short} - Spread Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'{symbol_short} - Comprehensive Spread Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    # Save plot with high quality
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{symbol_short}_spread_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    log.info(f"Improved spread analysis plot saved: plots/{symbol_short}_spread_analysis.png")

def create_ohlcv_analysis_plot(symbol: str, analysis: Dict, data: Dict):
    """Create OHLCV analysis plots with IMPROVED FORMATTING"""
    log.info(f"Creating OHLCV analysis plot for {symbol}...")
    
    ohlcv_df = data['ohlcv'].copy()
    
    if len(ohlcv_df) == 0:
        log.warning(f"No OHLCV data to plot for {symbol}")
        return
    
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)
    
    # 1. Price evolution with volume (IMPROVED)
    ax1 = fig.add_subplot(gs[0, :])  # Full width
    
    # Daily aggregation for better readability
    daily_data = ohlcv_df.groupby(ohlcv_df.index.date).agg({
        'close': 'last',
        'volume': 'sum'
    })
    
    # Smart sampling
    if len(daily_data) > 365:
        step = len(daily_data) // 365
        daily_data = daily_data.iloc[::step]
    
    # Price plot
    ax1_twin = ax1.twinx()
    
    price_line = ax1.plot(daily_data.index, daily_data['close'], 
                         linewidth=2, color='steelblue', alpha=0.8, label='Price')
    ax1.set_ylabel('Price (USDT)', fontsize=11, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Volume bars (only show non-zero volume)
    volume_data = daily_data[daily_data['volume'] > 0]
    if len(volume_data) > 0:
        volume_bars = ax1_twin.bar(volume_data.index, volume_data['volume'], 
                                  alpha=0.3, color='orange', width=1, label='Volume')
        ax1_twin.set_ylabel('Daily Volume', fontsize=11, color='orange')
        ax1_twin.tick_params(axis='y', labelcolor='orange')
    
    ax1.set_title(f'{symbol_short} - Price Evolution with Volume ({analysis["data_period"]["total_days"]} days)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Format dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(daily_data)//12)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Return distribution (IMPROVED)
    ax2 = fig.add_subplot(gs[1, 0])
    
    returns = ohlcv_df['close'].pct_change().dropna() * 100
    if len(returns) > 0:
        # Remove extreme outliers for better visualization
        returns_display = returns[(returns >= returns.quantile(0.01)) & (returns <= returns.quantile(0.99))]
        
        n, bins, patches = ax2.hist(returns_display, bins=50, alpha=0.7, edgecolor='black', 
                                   color='lightcoral', density=True)
        
        # Add normal distribution overlay
        mu, sigma = returns_display.mean(), returns_display.std()
        x = np.linspace(returns_display.min(), returns_display.max(), 100)
        normal_curve = ((1/(sigma * np.sqrt(2 * np.pi))) * 
                       np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        ax2.plot(x, normal_curve, 'k--', linewidth=2, alpha=0.8, label='Normal Distribution')
        
        ax2.axvline(0, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Zero Return')
        ax2.set_title(f'{symbol_short} - Return Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('1-minute Returns %', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.legend(fontsize=9)
        
        # Add statistics text
        vol = returns.std()
        skew = returns.skew()
        kurtosis = returns.kurtosis()
        
        stats_text = f'Volatility: {vol:.3f}%\nSkewness: {skew:.2f}\nKurtosis: {kurtosis:.1f}\nSharpe*: {analysis["price_analysis"]["returns"]["sharpe_approx"]:.3f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    
    # 3. Volume patterns by hour (IMPROVED)
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'hourly_volume' in analysis['patterns']:
        hourly_vol = pd.Series(analysis['patterns']['hourly_volume'])
        
        bars = ax3.bar(hourly_vol.index, hourly_vol.values, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Highlight peak hour
        peak_hour = analysis['patterns'].get('peak_volume_hour')
        if peak_hour is not None and peak_hour in hourly_vol.index:
            peak_idx = list(hourly_vol.index).index(peak_hour)
            bars[peak_idx].set_color('red')
            bars[peak_idx].set_alpha(0.9)
        
        ax3.set_title(f'{symbol_short} - Average Volume by Hour\n(Peak: {peak_hour}:00 UTC)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Hour (UTC)', fontsize=11)
        ax3.set_ylabel('Average Volume', fontsize=11)
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 4. Volume distribution (IMPROVED)
    ax4 = fig.add_subplot(gs[2, 0])
    
    volume_positive = ohlcv_df[ohlcv_df['volume'] > 0]['volume']
    if len(volume_positive) > 0:
        # Use log scale for better visualization
        log_volumes = np.log10(volume_positive)
        
        n, bins, patches = ax4.hist(log_volumes, bins=40, alpha=0.7, edgecolor='black', color='lightgreen')
        
        ax4.set_title(f'{symbol_short} - Volume Distribution (Log Scale)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Log10(Volume)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # Add volume statistics
        mean_vol = volume_positive.mean()
        median_vol = volume_positive.median()
        max_vol = volume_positive.max()
        
        stats_text = f'Mean: {mean_vol:.0f}\nMedian: {median_vol:.0f}\nMax: {max_vol:.0f}\nZero Vol: {analysis["price_analysis"]["volume"]["zero_volume_pct"]:.1f}%'
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    
    # 5. Price range analysis (NEW)
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Calculate daily price ranges
    daily_ranges = ohlcv_df.groupby(ohlcv_df.index.date).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    daily_ranges['range_pct'] = (daily_ranges['high'] - daily_ranges['low']) / daily_ranges['close'] * 100
    
    # Remove outliers for better visualization
    range_display = daily_ranges['range_pct']
    range_clean = range_display[range_display <= range_display.quantile(0.95)]
    
    if len(range_clean) > 0:
        n, bins, patches = ax5.hist(range_clean, bins=30, alpha=0.7, edgecolor='black', color='wheat')
        
        mean_range = range_clean.mean()
        ax5.axvline(mean_range, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_range:.2f}%')
        
        ax5.set_title(f'{symbol_short} - Daily Price Range Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Daily Range %', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'{symbol_short} - Comprehensive OHLCV Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    # Save plot with high quality
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{symbol_short}_ohlcv_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    log.info(f"Improved OHLCV analysis plot saved: plots/{symbol_short}_ohlcv_analysis.png")

def create_comparison_report(analyses: Dict):
    """Create IMPROVED comparison report across all symbols"""
    log.info("Creating cross-symbol comparison report...")
    
    if len(analyses) < 2:
        log.warning("Need at least 2 symbols for comparison")
        return
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Extract data for comparison
    symbols = []
    thresholds = []
    trading_ready_pcts = []
    avg_spreads = []
    total_records = []
    data_periods = []
    volatilities = []
    
    for symbol, data in analyses.items():
        if data and 'spread_analysis' in data and data['spread_analysis']:
            symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
            symbols.append(symbol_short)
            spread_analysis = data['spread_analysis']
            thresholds.append(spread_analysis['threshold_p80'])
            trading_ready_pcts.append(spread_analysis['trading_ready_pct'])
            avg_spreads.append(spread_analysis['spread_stats']['all_valid']['mean'])
            total_records.append(spread_analysis['total_records'])
            
            if 'ohlcv_analysis' in data and data['ohlcv_analysis']:
                data_periods.append(data['ohlcv_analysis']['data_period']['total_days'])
                volatilities.append(data['ohlcv_analysis']['price_analysis']['returns']['std'])
            else:
                data_periods.append(0)
                volatilities.append(0)
    
    if len(symbols) == 0:
        log.warning("No valid data for comparison")
        return
    
    # Use distinct colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'][:len(symbols)]
    
    # 1. P80 Thresholds comparison (IMPROVED)
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(symbols, thresholds, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('P80 Thresholds Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Threshold %', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars1, thresholds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{value:.4f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Trading readiness comparison (IMPROVED)
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(symbols, trading_ready_pcts, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Trading Readiness Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Trading Ready %', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.8, linewidth=2, label='80% Target')
    ax2.axhline(y=60, color='orange', linestyle=':', alpha=0.8, linewidth=1, label='60% Minimum')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars2, trading_ready_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Data coverage comparison (IMPROVED)
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(symbols, data_periods, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title('Historical Data Coverage', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Days of Data', fontsize=11)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars3, data_periods):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{value}d', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Spread vs Volatility scatter (NEW)
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(avg_spreads, volatilities, c=colors, s=100, alpha=0.8, edgecolors='black')
    
    for i, symbol in enumerate(symbols):
        ax4.annotate(symbol, (avg_spreads[i], volatilities[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_title('Spread vs Volatility', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Average Spread %', fontsize=11)
    ax4.set_ylabel('Volatility %', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 5. Trading efficiency radar-like comparison (NEW)
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Normalize metrics for comparison (0-100 scale)
    norm_readiness = np.array(trading_ready_pcts)
    norm_coverage = np.array(data_periods) / max(data_periods) * 100 if max(data_periods) > 0 else np.zeros(len(symbols))
    norm_records = np.array(total_records) / max(total_records) * 100 if max(total_records) > 0 else np.zeros(len(symbols))
    
    x = np.arange(len(symbols))
    width = 0.25
    
    bars_readiness = ax5.bar(x - width, norm_readiness, width, label='Trading Ready %', color='steelblue', alpha=0.8)
    bars_coverage = ax5.bar(x, norm_coverage, width, label='Coverage (normalized)', color='orange', alpha=0.8)
    bars_records = ax5.bar(x + width, norm_records, width, label='Records (normalized)', color='green', alpha=0.8)
    
    ax5.set_title('Trading Efficiency Comparison', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Normalized Score (0-100)', fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels(symbols, rotation=45)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Enhanced summary table (IMPROVED)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create comprehensive table data
    table_data = []
    for i, symbol in enumerate(symbols):
        # Determine status based on trading readiness
        if trading_ready_pcts[i] >= 75:
            status = "🟢 Excellent"
        elif trading_ready_pcts[i] >= 60:
            status = "🟡 Good"
        elif trading_ready_pcts[i] >= 40:
            status = "🟠 Fair"
        else:
            status = "🔴 Poor"
        
        table_data.append([
            symbol,
            f"{thresholds[i]:.4f}%",
            f"{trading_ready_pcts[i]:.1f}%",
            f"{avg_spreads[i]:.4f}%",
            f"{data_periods[i]}d",
            status
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Symbol', 'P80 Threshold', 'Ready %', 'Avg Spread', 'Days', 'Status'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    
    # Style the table header
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows based on status
    for i, row in enumerate(table_data):
        status = row[5]
        if "Excellent" in status:
            color = '#d5f4e6'
        elif "Good" in status:
            color = '#fff3cd'
        elif "Fair" in status:
            color = '#f8d7da'
        else:
            color = '#f5c6cb'
        
        for j in range(6):
            table[(i+1, j)].set_facecolor(color)
    
    ax6.set_title('Analysis Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('Cross-Symbol Trading Analysis Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot with high quality
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'symbols_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    log.info("Improved comparison plot saved: plots/symbols_comparison.png")

def generate_comprehensive_report(analyses: Dict):
    """Generate comprehensive analysis report"""
    log.info("\n" + "="*80)
    log.info("COMPREHENSIVE DATA ANALYSIS REPORT")
    log.info("="*80)
    
    total_symbols = len(analyses)
    valid_analyses = {k: v for k, v in analyses.items() if v and 'spread_analysis' in v and v['spread_analysis']}
    
    log.info(f"Analyzed {len(valid_analyses)}/{total_symbols} symbols successfully")
    
    if not valid_analyses:
        log.warning("No valid analyses to report")
        return
    
    # Overall statistics
    total_records = sum(a['spread_analysis']['total_records'] for a in valid_analyses.values())
    total_trading_ready = sum(a['spread_analysis']['trading_ready_count'] for a in valid_analyses.values())
    avg_trading_ready_pct = sum(a['spread_analysis']['trading_ready_pct'] for a in valid_analyses.values()) / len(valid_analyses)
    
    # Calculate total historical period
    min_start = None
    max_end = None
    for analysis_data in valid_analyses.values():
        if 'ohlcv_analysis' in analysis_data and analysis_data['ohlcv_analysis']:
            start = analysis_data['ohlcv_analysis']['data_period']['start']
            end = analysis_data['ohlcv_analysis']['data_period']['end']
            if min_start is None or start < min_start:
                min_start = start
            if max_end is None or end > max_end:
                max_end = end
    
    log.info(f"\nOVERALL STATISTICS:")
    log.info(f"  Historical Period: {min_start} to {max_end}")
    if min_start and max_end:
        total_days = (max_end - min_start).days
        log.info(f"  Total Days Covered: {total_days}")
    log.info(f"  Total records analyzed: {total_records:,}")
    log.info(f"  Total trading-ready records: {total_trading_ready:,}")
    log.info(f"  Average trading readiness: {avg_trading_ready_pct:.1f}%")
    
    # Symbol-by-symbol analysis
    log.info(f"\nSYMBOL ANALYSIS:")
    log.info("-" * 50)
    
    for symbol, analysis_data in valid_analyses.items():
        symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
        spread_analysis = analysis_data['spread_analysis']
        ohlcv_analysis = analysis_data.get('ohlcv_analysis')
        
        log.info(f"\n{symbol_short}:")
        log.info(f"  Data Quality:")
        log.info(f"    P80 Threshold: {spread_analysis['threshold_p80']:.4f}%")
        log.info(f"    Trading Ready: {spread_analysis['trading_ready_count']:,} ({spread_analysis['trading_ready_pct']:.1f}%)")
        log.info(f"    Average Spread: {spread_analysis['spread_stats']['all_valid']['mean']:.4f}%")
        
        # Quality breakdown
        quality = spread_analysis['quality_breakdown']
        total = quality['total_records']
        if total > 0:
            log.info(f"    Quality Breakdown:")
            log.info(f"      Excellent: {quality['excellent']:,} ({quality['excellent']/total*100:.1f}%)")
            log.info(f"      Good: {quality['good']:,} ({quality['good']/total*100:.1f}%)")
            log.info(f"      Fair: {quality['fair']:,} ({quality['fair']/total*100:.1f}%)")
            log.info(f"      Poor: {quality['poor']:,} ({quality['poor']/total*100:.1f}%)")
            log.info(f"      Invalid: {quality['invalid']:,} ({quality['invalid']/total*100:.1f}%)")
        
        # OHLCV insights
        if ohlcv_analysis:
            price_analysis = ohlcv_analysis['price_analysis']
            data_period = ohlcv_analysis['data_period']
            log.info(f"  Market Activity:")
            log.info(f"    Historical Period: {data_period['total_days']} days ({data_period['start']} to {data_period['end']})")
            log.info(f"    Price Range: ${price_analysis['price_range']['min']:.6f} - ${price_analysis['price_range']['max']:.6f}")
            log.info(f"    Volatility: {price_analysis['returns']['std']:.4f}%")
            log.info(f"    Zero Volume: {price_analysis['volume']['zero_volume_pct']:.1f}%")
            
            patterns = ohlcv_analysis['patterns']
            if patterns.get('peak_volume_hour') is not None:
                log.info(f"    Peak Volume Hour: {patterns['peak_volume_hour']}:00 UTC")
        
        # Trading recommendation
        if spread_analysis['trading_ready_pct'] >= 75:
            recommendation = "EXCELLENT - Highly suitable for trading"
        elif spread_analysis['trading_ready_pct'] >= 60:
            recommendation = "GOOD - Suitable for trading with normal risk"
        elif spread_analysis['trading_ready_pct'] >= 40:
            recommendation = "FAIR - Consider higher risk tolerance"
        else:
            recommendation = "POOR - High spread risk, needs review"
        
        log.info(f"  Recommendation: {recommendation}")
    
    # Trading guidelines
    log.info(f"\nTRADING GUIDELINES:")
    log.info("-" * 50)
    log.info(f"1. Use this filter for optimal data: WHERE valid_for_trading = TRUE")
    log.info(f"2. This ensures spreads ≤ P80 threshold for each symbol")
    log.info(f"3. Consider time-based patterns for execution timing")
    log.info(f"4. Monitor spread quality in real-time during trading")
    log.info(f"5. Historical analysis covers full dataset - not just recent data")
    
    # Data quality insights
    log.info(f"\nDATA QUALITY INSIGHTS:")
    log.info("-" * 50)
    best_quality = max(valid_analyses.items(), key=lambda x: x[1]['spread_analysis']['trading_ready_pct'])
    worst_quality = min(valid_analyses.items(), key=lambda x: x[1]['spread_analysis']['trading_ready_pct'])
    
    best_symbol = best_quality[0].split('_')[-2]
    worst_symbol = worst_quality[0].split('_')[-2]
    best_pct = best_quality[1]['spread_analysis']['trading_ready_pct']
    worst_pct = worst_quality[1]['spread_analysis']['trading_ready_pct']
    
    log.info(f"Best Data Quality: {best_symbol} ({best_pct:.1f}% trading ready)")
    log.info(f"Worst Data Quality: {worst_symbol} ({worst_pct:.1f}% trading ready)")
    
    # Volume patterns summary
    log.info(f"\nVOLUME PATTERNS SUMMARY:")
    log.info("-" * 50)
    for symbol, analysis_data in valid_analyses.items():
        if 'ohlcv_analysis' in analysis_data and analysis_data['ohlcv_analysis']:
            symbol_short = symbol.split('_')[-2]
            patterns = analysis_data['ohlcv_analysis']['patterns']
            peak_hour = patterns.get('peak_volume_hour')
            if peak_hour is not None:
                log.info(f"{symbol_short}: Peak volume at {peak_hour}:00 UTC")
    
    # Cross-symbol insights (if multiple symbols)
    if len(valid_analyses) > 1:
        log.info(f"\nCROSS-SYMBOL INSIGHTS:")
        log.info("-" * 50)
        
        # Compare thresholds
        thresholds = [a['spread_analysis']['threshold_p80'] for a in valid_analyses.values()]
        threshold_spread = max(thresholds) - min(thresholds)
        log.info(f"P80 Threshold Range: {min(thresholds):.4f}% to {max(thresholds):.4f}% (spread: {threshold_spread:.4f}%)")
        
        # Compare trading readiness
        readiness_pcts = [a['spread_analysis']['trading_ready_pct'] for a in valid_analyses.values()]
        readiness_spread = max(readiness_pcts) - min(readiness_pcts)
        log.info(f"Trading Readiness Range: {min(readiness_pcts):.1f}% to {max(readiness_pcts):.1f}% (spread: {readiness_spread:.1f}%)")
        
        # Identify best pairs for trading
        symbols_sorted = sorted(valid_analyses.items(), key=lambda x: x[1]['spread_analysis']['trading_ready_pct'], reverse=True)
        top_symbols = [s[0].split('_')[-2] for s in symbols_sorted[:2]]
        log.info(f"Recommended symbols for pair trading: {' vs '.join(top_symbols)}")
    
    # File locations
    log.info(f"\nGENERATED FILES:")
    log.info("-" * 50)
    log.info(f"Individual analysis plots: plots/[SYMBOL]_spread_analysis.png")
    log.info(f"OHLCV analysis plots: plots/[SYMBOL]_ohlcv_analysis.png")
    log.info(f"Comparison report: plots/symbols_comparison.png")
    log.info(f"Analysis logs: logs/validation.log")
    
    # Performance summary
    log.info(f"\nPERFORMANCE SUMMARY:")
    log.info("-" * 50)
    log.info(f"Total analysis completed in: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Database records processed: {total_records:,}")
    log.info(f"Charts generated: {len(valid_analyses) * 2 + 1}")
    log.info(f"Data coverage: Full historical dataset")
    
    # Final recommendations
    log.info(f"\nFINAL RECOMMENDATIONS:")
    log.info("-" * 50)
    if avg_trading_ready_pct >= 75:
        log.info(f"✅ EXCELLENT data quality across symbols - Ready for production trading")
    elif avg_trading_ready_pct >= 60:
        log.info(f"✅ GOOD data quality - Suitable for trading with standard risk management")
    elif avg_trading_ready_pct >= 40:
        log.info(f"⚠️  FAIR data quality - Consider additional filtering or risk controls")
    else:
        log.info(f"❌ POOR data quality - Review data sources and improve filtering")
    
    log.info(f"📊 Use P80-based quality filtering for optimal results")
    log.info(f"📈 Monitor real-time spread quality during live trading")
    log.info(f"🔄 Regular re-analysis recommended as new data arrives")

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze cleaned data")
    parser.add_argument("--symbol", type=str, help="Specific symbol to analyze")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    log.info("Starting comprehensive data analysis...")
    
    # Get symbols to analyze
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
        log.error("No symbols to analyze")
        return False
    
    log.info(f"Analyzing {len(symbols)} symbols: {', '.join([s.split('_')[-2] for s in symbols])}")
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    analyses = {}
    
    try:
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"ANALYZING {symbol}")
            log.info(f"{'='*60}")
            
            # Load cleaned data
            try:
                data = load_cleaned_data(symbol)
                
                if not data or len(data.get('orderbook', [])) == 0:
                    log.warning(f"No data available for {symbol}")
                    continue
                
                analysis_results = {}
                
                # Analyze spread quality
                try:
                    spread_analysis = analyze_spread_quality(symbol, data)
                    if spread_analysis:
                        analysis_results['spread_analysis'] = spread_analysis
                        
                        # Create spread analysis plot
                        if not args.no_plots:
                            try:
                                create_spread_analysis_plot(symbol, spread_analysis, data)
                            except Exception as plot_error:
                                log.warning(f"Failed to create spread plot for {symbol}: {plot_error}")
                except Exception as spread_error:
                    log.error(f"Failed to analyze spreads for {symbol}: {spread_error}")
                
                # Analyze OHLCV patterns
                try:
                    ohlcv_analysis = analyze_ohlcv_patterns(symbol, data)
                    if ohlcv_analysis:
                        analysis_results['ohlcv_analysis'] = ohlcv_analysis
                        
                        # Create OHLCV analysis plot
                        if not args.no_plots:
                            try:
                                create_ohlcv_analysis_plot(symbol, ohlcv_analysis, data)
                            except Exception as plot_error:
                                log.warning(f"Failed to create OHLCV plot for {symbol}: {plot_error}")
                except Exception as ohlcv_error:
                    log.error(f"Failed to analyze OHLCV for {symbol}: {ohlcv_error}")
                
                analyses[symbol] = analysis_results
                
            except Exception as data_error:
                log.error(f"Failed to load data for {symbol}: {data_error}")
                continue
        
        # Create comparison report
        if len(analyses) > 1 and not args.no_plots:
            try:
                create_comparison_report(analyses)
            except Exception as comp_error:
                log.warning(f"Failed to create comparison report: {comp_error}")
        
        # Generate comprehensive report
        try:
            generate_comprehensive_report(analyses)
        except Exception as report_error:
            log.error(f"Failed to generate comprehensive report: {report_error}")
        
        log.info(f"\nData analysis completed!")
        log.info(f"Analyzed: {len([a for a in analyses.values() if a])}/{len(symbols)} symbols")
        
        return True
        
    except Exception as e:
        log.error(f"Data analysis failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)