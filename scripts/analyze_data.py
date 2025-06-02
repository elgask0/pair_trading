#!/usr/bin/env python3
"""
Data analysis script
Analyzes cleaned data and generates comprehensive reports and visualizations
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

def load_cleaned_data(symbol: str) -> Dict:
    """Load cleaned data for analysis"""
    log.info(f"Loading cleaned data for {symbol}...")
    
    with db_manager.get_session() as session:
        # OHLCV data
        ohlcv_query = text("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        ohlcv_df = pd.read_sql(ohlcv_query, session.bind, params={"symbol": symbol}, index_col='timestamp')
        
        # Orderbook data with quality information
        orderbook_query = text("""
            SELECT timestamp, bid1_price, bid1_size, ask1_price, ask1_size,
                   liquidity_quality, valid_for_trading, spread_pct, threshold_p80
            FROM orderbook 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        orderbook_df = pd.read_sql(orderbook_query, session.bind, params={"symbol": symbol}, index_col='timestamp')
        
        # Quality summary
        quality_summary = session.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as trading_ready,
                COUNT(CASE WHEN liquidity_quality = 'Excellent' THEN 1 END) as excellent,
                COUNT(CASE WHEN liquidity_quality = 'Good' THEN 1 END) as good,
                COUNT(CASE WHEN liquidity_quality = 'Fair' THEN 1 END) as fair,
                COUNT(CASE WHEN liquidity_quality = 'Poor' THEN 1 END) as poor,
                COUNT(CASE WHEN liquidity_quality = 'Invalid' THEN 1 END) as invalid,
                AVG(CASE WHEN spread_pct IS NOT NULL THEN spread_pct END) as avg_spread,
                MIN(threshold_p80) as threshold_p80
            FROM orderbook 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        log.info(f"Loaded {len(ohlcv_df):,} OHLCV and {len(orderbook_df):,} orderbook records")
        
        return {
            'ohlcv': ohlcv_df,
            'orderbook': orderbook_df,
            'quality_summary': {
                'total_records': quality_summary.total_records or 0,
                'trading_ready': quality_summary.trading_ready or 0,
                'excellent': quality_summary.excellent or 0,
                'good': quality_summary.good or 0,
                'fair': quality_summary.fair or 0,
                'poor': quality_summary.poor or 0,
                'invalid': quality_summary.invalid or 0,
                'avg_spread': quality_summary.avg_spread or 0,
                'threshold_p80': quality_summary.threshold_p80 or 0
            }
        }

def analyze_spread_quality(symbol: str, data: Dict) -> Dict:
    """Analyze spread quality after cleaning"""
    log.info(f"Analyzing spread quality for {symbol}...")
    
    orderbook_df = data['orderbook']
    quality_summary = data['quality_summary']
    
    if len(orderbook_df) == 0:
        log.warning(f"No orderbook data for {symbol}")
        return None
    
    # Filter valid spreads for detailed analysis
    valid_spreads = orderbook_df[orderbook_df['spread_pct'].notna()]['spread_pct']
    trading_ready_spreads = orderbook_df[orderbook_df['valid_for_trading'] == True]['spread_pct']
    
    if len(valid_spreads) == 0:
        log.warning(f"No valid spread data for {symbol}")
        return None
    
    threshold = quality_summary['threshold_p80']
    
    # Calculate statistics
    analysis = {
        'symbol': symbol,
        'threshold_p80': threshold,
        'total_records': len(orderbook_df),
        'valid_spreads_count': len(valid_spreads),
        'trading_ready_count': len(trading_ready_spreads),
        'trading_ready_pct': len(trading_ready_spreads) / len(orderbook_df) * 100,
        'spread_stats': {
            'all_valid': {
                'mean': valid_spreads.mean(),
                'median': valid_spreads.median(),
                'std': valid_spreads.std(),
                'min': valid_spreads.min(),
                'max': valid_spreads.max(),
                'p25': valid_spreads.quantile(0.25),
                'p75': valid_spreads.quantile(0.75),
                'p95': valid_spreads.quantile(0.95),
                'p99': valid_spreads.quantile(0.99)
            }
        },
        'quality_breakdown': quality_summary
    }
    
    # Trading-ready spreads stats
    if len(trading_ready_spreads) > 0:
        analysis['spread_stats']['trading_ready'] = {
            'mean': trading_ready_spreads.mean(),
            'median': trading_ready_spreads.median(),
            'std': trading_ready_spreads.std(),
            'min': trading_ready_spreads.min(),
            'max': trading_ready_spreads.max()
        }
    
    # Log results
    log.info(f"{symbol} Spread Quality Analysis:")
    log.info(f"  P80 Threshold: {threshold:.4f}%")
    log.info(f"  Trading Ready: {analysis['trading_ready_count']:,} ({analysis['trading_ready_pct']:.1f}%)")
    log.info(f"  Valid Spreads - Mean: {analysis['spread_stats']['all_valid']['mean']:.4f}%, Median: {analysis['spread_stats']['all_valid']['median']:.4f}%")
    
    if 'trading_ready' in analysis['spread_stats']:
        tr_stats = analysis['spread_stats']['trading_ready']
        log.info(f"  Trading Ready Spreads - Mean: {tr_stats['mean']:.4f}%, Median: {tr_stats['median']:.4f}%")
    
    return analysis

def analyze_ohlcv_patterns(symbol: str, data: Dict) -> Dict:
    """Analyze OHLCV patterns and characteristics"""
    log.info(f"Analyzing OHLCV patterns for {symbol}...")
    
    ohlcv_df = data['ohlcv']
    
    if len(ohlcv_df) == 0:
        log.warning(f"No OHLCV data for {symbol}")
        return None
    
    # Basic statistics
    returns = ohlcv_df['close'].pct_change().dropna()
    volume_positive = ohlcv_df[ohlcv_df['volume'] > 0]['volume']
    
    # Price analysis
    price_analysis = {
        'price_range': {
            'min': ohlcv_df['low'].min(),
            'max': ohlcv_df['high'].max(),
            'mean': ohlcv_df['close'].mean(),
            'current': ohlcv_df['close'].iloc[-1] if len(ohlcv_df) > 0 else None
        },
        'returns': {
            'mean': returns.mean() * 100,
            'std': returns.std() * 100,
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min() * 100,
            'max': returns.max() * 100,
            'sharpe_approx': returns.mean() / returns.std() if returns.std() > 0 else 0
        },
        'volume': {
            'mean': volume_positive.mean() if len(volume_positive) > 0 else 0,
            'median': volume_positive.median() if len(volume_positive) > 0 else 0,
            'zero_volume_pct': (ohlcv_df['volume'] == 0).mean() * 100,
            'total_volume': ohlcv_df['volume'].sum()
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
        'peak_volume_hour': hourly_volume.idxmax() if len(hourly_volume) > 0 else None,
        'peak_volume_day': daily_volume.idxmax() if len(daily_volume) > 0 else None
    }
    
    analysis = {
        'symbol': symbol,
        'data_period': {
            'start': ohlcv_df.index.min(),
            'end': ohlcv_df.index.max(),
            'total_records': len(ohlcv_df)
        },
        'price_analysis': price_analysis,
        'patterns': patterns
    }
    
    # Log results
    log.info(f"{symbol} OHLCV Analysis:")
    log.info(f"  Data Period: {analysis['data_period']['start']} to {analysis['data_period']['end']}")
    log.info(f"  Records: {analysis['data_period']['total_records']:,}")
    log.info(f"  Price Range: ${price_analysis['price_range']['min']:.6f} - ${price_analysis['price_range']['max']:.6f}")
    log.info(f"  Returns - Mean: {price_analysis['returns']['mean']:.4f}%, Std: {price_analysis['returns']['std']:.4f}%")
    log.info(f"  Volume - Zero volume: {price_analysis['volume']['zero_volume_pct']:.1f}%")
    
    return analysis

def create_spread_analysis_plot(symbol: str, analysis: Dict, data: Dict):
    """Create comprehensive spread analysis plots"""
    log.info(f"Creating spread analysis plot for {symbol}...")
    
    orderbook_df = data['orderbook']
    valid_spreads = orderbook_df[orderbook_df['spread_pct'].notna()]['spread_pct']
    threshold = analysis['threshold_p80']
    
    if len(valid_spreads) == 0:
        log.warning(f"No valid spreads to plot for {symbol}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    # 1. Spread distribution with quality zones
    axes[0, 0].hist(valid_spreads, bins=100, alpha=0.7, edgecolor='black', density=True)
    
    # Mark quality zones
    axes[0, 0].axvline(threshold * 0.5, color='green', linestyle='-', linewidth=2, 
                      label=f'Excellent: ≤{threshold*0.5:.4f}%')
    axes[0, 0].axvline(threshold, color='orange', linestyle='-', linewidth=2, 
                      label=f'Good: ≤{threshold:.4f}% (P80)')
    axes[0, 0].axvline(threshold * 1.5, color='red', linestyle='-', linewidth=2, 
                      label=f'Fair: ≤{threshold*1.5:.4f}%')
    
    axes[0, 0].set_title(f'{symbol_short} - Spread Distribution with Quality Zones')
    axes[0, 0].set_xlabel('Spread %')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. Quality pie chart
    quality_counts = [
        analysis['quality_breakdown']['excellent'],
        analysis['quality_breakdown']['good'], 
        analysis['quality_breakdown']['fair'],
        analysis['quality_breakdown']['poor'],
        analysis['quality_breakdown']['invalid']
    ]
    quality_labels = ['Excellent', 'Good', 'Fair', 'Poor', 'Invalid']
    colors = ['green', 'lightgreen', 'yellow', 'red', 'gray']
    
    # Only show non-zero categories
    non_zero_indices = [i for i, count in enumerate(quality_counts) if count > 0]
    filtered_counts = [quality_counts[i] for i in non_zero_indices]
    filtered_labels = [quality_labels[i] for i in non_zero_indices]
    filtered_colors = [colors[i] for i in non_zero_indices]
    
    axes[0, 1].pie(filtered_counts, labels=filtered_labels, colors=filtered_colors,
                   autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title(f'{symbol_short} - Data Quality Distribution')
    
    # 3. Time series of spreads (last 7 days if available)
    recent_data = orderbook_df.last('7D') if len(orderbook_df) > 0 else orderbook_df
    if len(recent_data) > 0:
        recent_spreads = recent_data[recent_data['spread_pct'].notna()]
        if len(recent_spreads) > 0:
            axes[1, 0].plot(recent_spreads.index, recent_spreads['spread_pct'], 
                           alpha=0.7, linewidth=1)
            axes[1, 0].axhline(threshold, color='orange', linestyle='--', 
                              label=f'P80: {threshold:.4f}%')
            axes[1, 0].set_title(f'{symbol_short} - Recent Spread Evolution')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Spread %')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Trading readiness over time (hourly aggregation)
    if len(orderbook_df) > 0:
        hourly_readiness = orderbook_df.groupby(orderbook_df.index.hour).agg({
            'valid_for_trading': 'mean'
        })['valid_for_trading'] * 100
        
        axes[1, 1].bar(hourly_readiness.index, hourly_readiness.values, alpha=0.7)
        axes[1, 1].set_title(f'{symbol_short} - Trading Readiness by Hour')
        axes[1, 1].set_xlabel('Hour (UTC)')
        axes[1, 1].set_ylabel('Trading Ready %')
        axes[1, 1].set_xticks(range(0, 24, 2))
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{symbol_short}_spread_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Spread analysis plot saved: plots/{symbol_short}_spread_analysis.png")

def create_ohlcv_analysis_plot(symbol: str, analysis: Dict, data: Dict):
    """Create OHLCV analysis plots"""
    log.info(f"Creating OHLCV analysis plot for {symbol}...")
    
    ohlcv_df = data['ohlcv']
    
    if len(ohlcv_df) == 0:
        log.warning(f"No OHLCV data to plot for {symbol}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    # 1. Price evolution (last 30 days or all data if less)
    recent_ohlcv = ohlcv_df.last('30D') if len(ohlcv_df) > 1440 else ohlcv_df
    if len(recent_ohlcv) > 0:
        daily_prices = recent_ohlcv.groupby(recent_ohlcv.index.date)['close'].last()
        axes[0, 0].plot(daily_prices.index, daily_prices.values, linewidth=2, alpha=0.8)
        axes[0, 0].set_title(f'{symbol_short} - Price Evolution')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Return distribution
    returns = ohlcv_df['close'].pct_change().dropna() * 100
    if len(returns) > 0:
        axes[0, 1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title(f'{symbol_short} - Return Distribution')
        axes[0, 1].set_xlabel('1-minute Returns %')
        axes[0, 1].set_ylabel('Frequency')
        
        # Add statistics text
        vol = returns.std()
        axes[0, 1].text(0.7, 0.9, f'Volatility: {vol:.3f}%\nSkew: {returns.skew():.2f}', 
                       transform=axes[0, 1].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 3. Volume patterns by hour
    if 'hourly_volume' in analysis['patterns']:
        hourly_vol = pd.Series(analysis['patterns']['hourly_volume'])
        axes[1, 0].bar(hourly_vol.index, hourly_vol.values, alpha=0.7)
        axes[1, 0].set_title(f'{symbol_short} - Average Volume by Hour')
        axes[1, 0].set_xlabel('Hour (UTC)')
        axes[1, 0].set_ylabel('Average Volume')
        axes[1, 0].set_xticks(range(0, 24, 2))
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Volume distribution (non-zero only)
    volume_positive = ohlcv_df[ohlcv_df['volume'] > 0]['volume']
    if len(volume_positive) > 0:
        axes[1, 1].hist(volume_positive, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title(f'{symbol_short} - Volume Distribution')
        axes[1, 1].set_xlabel('Volume')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{symbol_short}_ohlcv_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"OHLCV analysis plot saved: plots/{symbol_short}_ohlcv_analysis.png")

def create_comparison_report(analyses: Dict):
    """Create comparison report across all symbols"""
    log.info("Creating cross-symbol comparison report...")
    
    if len(analyses) < 2:
        log.warning("Need at least 2 symbols for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for comparison
    symbols = []
    thresholds = []
    trading_ready_pcts = []
    avg_spreads = []
    total_records = []
    
    for symbol, data in analyses.items():
        if data and 'spread_analysis' in data and data['spread_analysis']:
            symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
            symbols.append(symbol_short)
            spread_analysis = data['spread_analysis']
            thresholds.append(spread_analysis['threshold_p80'])
            trading_ready_pcts.append(spread_analysis['trading_ready_pct'])
            avg_spreads.append(spread_analysis['spread_stats']['all_valid']['mean'])
            total_records.append(spread_analysis['total_records'])
    
    if len(symbols) == 0:
        log.warning("No valid data for comparison")
        return
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown'][:len(symbols)]
    
    # 1. P80 Thresholds comparison
    bars1 = axes[0, 0].bar(symbols, thresholds, color=colors, alpha=0.7)
    axes[0, 0].set_title('P80 Thresholds Comparison')
    axes[0, 0].set_ylabel('Threshold %')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for bar, value in zip(bars1, thresholds):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Trading readiness comparison
    bars2 = axes[0, 1].bar(symbols, trading_ready_pcts, color=colors, alpha=0.7)
    axes[0, 1].set_title('Trading Readiness Comparison')
    axes[0, 1].set_ylabel('Trading Ready %')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Target')
    axes[0, 1].legend()
    for bar, value in zip(bars2, trading_ready_pcts):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Average spreads vs thresholds
    x_pos = range(len(symbols))
    width = 0.35
    bars3 = axes[1, 0].bar([x - width/2 for x in x_pos], avg_spreads, width, 
                          color=colors, alpha=0.7, label='Avg Spread')
    bars4 = axes[1, 0].bar([x + width/2 for x in x_pos], thresholds, width,
                          color=colors, alpha=0.3, label='P80 Threshold')
    axes[1, 0].set_title('Average Spread vs P80 Threshold')
    axes[1, 0].set_ylabel('Spread %')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(symbols, rotation=45)
    axes[1, 0].legend()
    
    # 4. Summary table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = []
    for i, symbol in enumerate(symbols):
        table_data.append([
            symbol,
            f"{thresholds[i]:.4f}%",
            f"{trading_ready_pcts[i]:.1f}%",
            f"{avg_spreads[i]:.4f}%",
            f"{total_records[i]:,}"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                           colLabels=['Symbol', 'P80 Threshold', 'Trading Ready', 'Avg Spread', 'Records'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Analysis Summary')
    
    plt.tight_layout()
    
    # Save comparison plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'symbols_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info("Comparison plot saved: plots/symbols_comparison.png")

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
    
    log.info(f"\nOVERALL STATISTICS:")
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
        log.info(f"    Quality Breakdown:")
        log.info(f"      Excellent: {quality['excellent']:,} ({quality['excellent']/total*100:.1f}%)")
        log.info(f"      Good: {quality['good']:,} ({quality['good']/total*100:.1f}%)")
        log.info(f"      Fair: {quality['fair']:,} ({quality['fair']/total*100:.1f}%)")
        log.info(f"      Poor: {quality['poor']:,} ({quality['poor']/total*100:.1f}%)")
        log.info(f"      Invalid: {quality['invalid']:,} ({quality['invalid']/total*100:.1f}%)")
        
        # OHLCV insights
        if ohlcv_analysis:
            price_analysis = ohlcv_analysis['price_analysis']
            log.info(f"  Market Activity:")
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
    
    # File locations
    log.info(f"\nGENERATED FILES:")
    log.info("-" * 50)
    log.info(f"Individual analysis plots: plots/[SYMBOL]_spread_analysis.png")
    log.info(f"OHLCV analysis plots: plots/[SYMBOL]_ohlcv_analysis.png")
    log.info(f"Comparison report: plots/symbols_comparison.png")
    log.info(f"Analysis logs: logs/validation.log")

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
            data = load_cleaned_data(symbol)
            
            if not data or len(data.get('orderbook', [])) == 0:
                log.warning(f"No data available for {symbol}")
                continue
            
            analysis_results = {}
            
            # Analyze spread quality
            spread_analysis = analyze_spread_quality(symbol, data)
            if spread_analysis:
                analysis_results['spread_analysis'] = spread_analysis
                
                # Create spread analysis plot
                if not args.no_plots:
                    create_spread_analysis_plot(symbol, spread_analysis, data)
            
            # Analyze OHLCV patterns
            ohlcv_analysis = analyze_ohlcv_patterns(symbol, data)
            if ohlcv_analysis:
                analysis_results['ohlcv_analysis'] = ohlcv_analysis
                
                # Create OHLCV analysis plot
                if not args.no_plots:
                    create_ohlcv_analysis_plot(symbol, ohlcv_analysis, data)
            
            analyses[symbol] = analysis_results
        
        # Create comparison report
        if len(analyses) > 1 and not args.no_plots:
            create_comparison_report(analyses)
        
        # Generate comprehensive report
        generate_comprehensive_report(analyses)
        
        log.info(f"\nData analysis completed successfully!")
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