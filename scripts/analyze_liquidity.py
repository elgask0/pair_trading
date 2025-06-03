#!/usr/bin/env python3
"""
Liquidity analysis script - COMPREHENSIVE MARKET DEPTH ANALYSIS
Analyzes market liquidity using orderbook data and mark prices
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
from typing import Dict, List, Optional, Tuple
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

# Set matplotlib style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def calculate_slippage_for_size(orderbook_row: pd.Series, size_usd: float, side: str = 'buy') -> Dict:
    """
    Calculate slippage for a given order size
    
    Args:
        orderbook_row: Row from orderbook with all levels
        size_usd: Size in USD to execute
        side: 'buy' or 'sell'
    
    Returns:
        Dict with execution details
    """
    
    levels = []
    prefix = 'ask' if side == 'buy' else 'bid'
    
    # Extract all available levels
    for i in range(1, 11):
        price = orderbook_row.get(f'{prefix}{i}_price')
        size = orderbook_row.get(f'{prefix}{i}_size')
        
        if pd.notna(price) and pd.notna(size) and price > 0 and size > 0:
            levels.append({
                'level': i,
                'price': price,
                'size': size,
                'size_usd': price * size
            })
    
    if not levels:
        return {
            'can_execute': False,
            'total_size_usd': 0,
            'avg_price': None,
            'slippage_pct': None,
            'levels_used': 0,
            'max_level': 0,
            'liquidity_exhausted': True
        }
    
    # Sort levels by price (ascending for buys, descending for sells)
    levels.sort(key=lambda x: x['price'], reverse=(side == 'sell'))
    
    # Calculate execution
    remaining_size = size_usd
    executed_value = 0
    executed_size = 0
    levels_used = []
    
    for level in levels:
        if remaining_size <= 0:
            break
            
        level_size_usd = level['size_usd']
        
        if level_size_usd >= remaining_size:
            # This level can fill the entire remaining order
            executed_value += remaining_size
            executed_size += remaining_size / level['price']
            levels_used.append(level['level'])
            remaining_size = 0
        else:
            # Use entire level
            executed_value += level_size_usd
            executed_size += level['size']
            levels_used.append(level['level'])
            remaining_size -= level_size_usd
    
    # Check if order could be filled
    if remaining_size > 0:
        return {
            'can_execute': False,
            'total_size_usd': size_usd - remaining_size,
            'avg_price': executed_value / executed_size if executed_size > 0 else None,
            'slippage_pct': None,
            'levels_used': len(levels_used),
            'max_level': max(levels_used) if levels_used else 0,
            'liquidity_exhausted': True
        }
    
    # Calculate average execution price and slippage
    avg_price = executed_value / executed_size
    best_price = levels[0]['price']
    slippage_pct = abs(avg_price - best_price) / best_price * 100
    
    return {
        'can_execute': True,
        'total_size_usd': size_usd,
        'avg_price': avg_price,
        'best_price': best_price,
        'slippage_pct': slippage_pct,
        'levels_used': len(levels_used),
        'max_level': max(levels_used),
        'liquidity_exhausted': False
    }

def calculate_max_size_for_slippage(orderbook_row: pd.Series, max_slippage_pct: float, side: str = 'buy') -> Dict:
    """
    Calculate maximum order size for a given slippage tolerance
    
    Args:
        orderbook_row: Row from orderbook with all levels
        max_slippage_pct: Maximum acceptable slippage percentage
        side: 'buy' or 'sell'
    
    Returns:
        Dict with maximum executable size
    """
    
    levels = []
    prefix = 'ask' if side == 'buy' else 'bid'
    
    # Extract all available levels
    for i in range(1, 11):
        price = orderbook_row.get(f'{prefix}{i}_price')
        size = orderbook_row.get(f'{prefix}{i}_size')
        
        if pd.notna(price) and pd.notna(size) and price > 0 and size > 0:
            levels.append({
                'level': i,
                'price': price,
                'size': size,
                'size_usd': price * size
            })
    
    if not levels:
        return {
            'max_size_usd': 0,
            'levels_used': 0,
            'actual_slippage_pct': 0
        }
    
    # Sort levels by price
    levels.sort(key=lambda x: x['price'], reverse=(side == 'sell'))
    best_price = levels[0]['price']
    
    # Calculate cumulative execution
    total_value = 0
    total_size = 0
    levels_used = 0
    
    for level in levels:
        # Check if adding this level would exceed slippage
        new_total_value = total_value + level['size_usd']
        new_total_size = total_size + level['size']
        new_avg_price = new_total_value / new_total_size
        new_slippage_pct = abs(new_avg_price - best_price) / best_price * 100
        
        if new_slippage_pct > max_slippage_pct:
            # Can only partially use this level
            # Binary search for the exact amount
            low, high = 0, level['size']
            best_partial_size = 0
            
            for _ in range(20):  # Binary search iterations
                mid = (low + high) / 2
                test_value = total_value + mid * level['price']
                test_size = total_size + mid
                test_avg_price = test_value / test_size
                test_slippage = abs(test_avg_price - best_price) / best_price * 100
                
                if test_slippage <= max_slippage_pct:
                    best_partial_size = mid
                    low = mid
                else:
                    high = mid
            
            if best_partial_size > 0:
                total_value += best_partial_size * level['price']
                total_size += best_partial_size
                levels_used += 1
            break
        else:
            # Use entire level
            total_value = new_total_value
            total_size = new_total_size
            levels_used += 1
    
    if total_size == 0:
        return {
            'max_size_usd': 0,
            'levels_used': 0,
            'actual_slippage_pct': 0
        }
    
    avg_price = total_value / total_size
    actual_slippage = abs(avg_price - best_price) / best_price * 100
    
    return {
        'max_size_usd': total_value,
        'levels_used': levels_used,
        'actual_slippage_pct': actual_slippage,
        'avg_price': avg_price,
        'best_price': best_price
    }

def load_liquidity_data(symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
    """Load orderbook and mark price data for liquidity analysis"""
    log.info(f"Loading liquidity data for {symbol}...")
    
    with db_manager.get_session() as session:
        # Build date filter
        date_filter = ""
        params = {"symbol": symbol}
        
        if start_date:
            date_filter += " AND timestamp >= :start_date"
            params["start_date"] = start_date
        if end_date:
            date_filter += " AND timestamp <= :end_date"
            params["end_date"] = end_date
        
        # Load orderbook data
        orderbook_query = text(f"""
            SELECT 
                timestamp,
                bid1_price, bid1_size, bid2_price, bid2_size, bid3_price, bid3_size,
                bid4_price, bid4_size, bid5_price, bid5_size, bid6_price, bid6_size,
                bid7_price, bid7_size, bid8_price, bid8_size, bid9_price, bid9_size,
                bid10_price, bid10_size,
                ask1_price, ask1_size, ask2_price, ask2_size, ask3_price, ask3_size,
                ask4_price, ask4_size, ask5_price, ask5_size, ask6_price, ask6_size,
                ask7_price, ask7_size, ask8_price, ask8_size, ask9_price, ask9_size,
                ask10_price, ask10_size
            FROM orderbook 
            WHERE symbol = :symbol {date_filter}
            ORDER BY timestamp
        """)
        
        orderbook_df = pd.read_sql(orderbook_query, session.bind, params=params, index_col='timestamp')
        
        # Load mark prices
        mark_price_query = text(f"""
            SELECT timestamp, mark_price, is_valid
            FROM mark_prices 
            WHERE symbol = :symbol 
            AND is_valid = TRUE {date_filter}
            ORDER BY timestamp
        """)
        
        mark_price_df = pd.read_sql(mark_price_query, session.bind, params=params, index_col='timestamp')
        
        log.info(f"Loaded {len(orderbook_df):,} orderbook and {len(mark_price_df):,} mark price records")
        
        return {
            'orderbook': orderbook_df,
            'mark_prices': mark_price_df
        }

def analyze_liquidity_metrics(symbol: str, data: Dict, sizes_usd: List[float], slippage_targets: List[float]) -> Dict:
    """Analyze liquidity metrics for various order sizes and slippage targets"""
    log.info(f"Analyzing liquidity metrics for {symbol}...")
    
    orderbook_df = data['orderbook']
    mark_prices_df = data['mark_prices']
    
    # Merge data on timestamp
    merged_df = orderbook_df.join(mark_prices_df, how='inner')
    
    if len(merged_df) == 0:
        log.warning(f"No matching data between orderbook and mark prices for {symbol}")
        return None
    
    log.info(f"Analyzing {len(merged_df):,} timestamps with complete data")
    
    # Initialize results storage
    slippage_results = {size: {'buy': [], 'sell': []} for size in sizes_usd}
    max_size_results = {slippage: {'buy': [], 'sell': []} for slippage in slippage_targets}
    timestamps = []
    
    # Process each timestamp
    for idx, (timestamp, row) in enumerate(merged_df.iterrows()):
        if idx % 10000 == 0:
            log.info(f"Processing {idx:,} / {len(merged_df):,} timestamps...")
        
        timestamps.append(timestamp)
        
        # Calculate slippage for different order sizes
        for size in sizes_usd:
            for side in ['buy', 'sell']:
                result = calculate_slippage_for_size(row, size, side)
                slippage_results[size][side].append(result)
        
        # Calculate max size for different slippage targets
        for slippage in slippage_targets:
            for side in ['buy', 'sell']:
                result = calculate_max_size_for_slippage(row, slippage, side)
                max_size_results[slippage][side].append(result)
    
    # Convert to DataFrames for easier analysis
    analysis_results = {
        'timestamps': timestamps,
        'slippage_by_size': {},
        'max_size_by_slippage': {},
        'mark_prices': merged_df['mark_price'].values
    }
    
    # Process slippage results
    for size in sizes_usd:
        for side in ['buy', 'sell']:
            results = slippage_results[size][side]
            df = pd.DataFrame(results)
            df.index = timestamps
            analysis_results['slippage_by_size'][f'{size}_{side}'] = df
    
    # Process max size results
    for slippage in slippage_targets:
        for side in ['buy', 'sell']:
            results = max_size_results[slippage][side]
            df = pd.DataFrame(results)
            df.index = timestamps
            analysis_results['max_size_by_slippage'][f'{slippage}_{side}'] = df
    
    return analysis_results

def calculate_liquidity_statistics(analysis_results: Dict) -> Dict:
    """Calculate summary statistics for liquidity metrics"""
    log.info("Calculating liquidity statistics...")
    
    stats = {
        'slippage_stats': {},
        'max_size_stats': {},
        'execution_success_rate': {}
    }
    
    # Slippage statistics by order size
    for key, df in analysis_results['slippage_by_size'].items():
        size, side = key.rsplit('_', 1)
        
        # Filter successful executions
        successful = df[df['can_execute'] == True]
        success_rate = len(successful) / len(df) * 100 if len(df) > 0 else 0
        
        if len(successful) > 0:
            slippage_values = successful['slippage_pct'].dropna()
            
            stats['slippage_stats'][key] = {
                'mean': slippage_values.mean(),
                'median': slippage_values.median(),
                'std': slippage_values.std(),
                'p95': slippage_values.quantile(0.95),
                'p99': slippage_values.quantile(0.99),
                'max': slippage_values.max(),
                'success_rate': success_rate
            }
        else:
            stats['slippage_stats'][key] = {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'p95': np.nan,
                'p99': np.nan,
                'max': np.nan,
                'success_rate': 0
            }
        
        stats['execution_success_rate'][key] = success_rate
    
    # Max size statistics by slippage target
    for key, df in analysis_results['max_size_by_slippage'].items():
        slippage, side = key.rsplit('_', 1)
        
        max_sizes = df['max_size_usd'].dropna()
        
        if len(max_sizes) > 0:
            stats['max_size_stats'][key] = {
                'mean': max_sizes.mean(),
                'median': max_sizes.median(),
                'std': max_sizes.std(),
                'p5': max_sizes.quantile(0.05),
                'p10': max_sizes.quantile(0.10),
                'min': max_sizes.min()
            }
        else:
            stats['max_size_stats'][key] = {
                'mean': 0,
                'median': 0,
                'std': 0,
                'p5': 0,
                'p10': 0,
                'min': 0
            }
    
    return stats

def create_liquidity_plots(symbol: str, analysis_results: Dict, stats: Dict, sizes_usd: List[float], slippage_targets: List[float]):
    """Create comprehensive liquidity analysis plots"""
    log.info(f"Creating liquidity analysis plots for {symbol}...")
    
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.3, wspace=0.25)
    
    # 1. Slippage by Order Size - Time Series
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot rolling mean slippage for different sizes
    for size in sizes_usd[:3]:  # Show first 3 sizes to avoid clutter
        buy_df = analysis_results['slippage_by_size'][f'{size}_buy']
        sell_df = analysis_results['slippage_by_size'][f'{size}_sell']
        
        # Filter successful executions
        buy_slippage = buy_df[buy_df['can_execute'] == True]['slippage_pct']
        sell_slippage = sell_df[sell_df['can_execute'] == True]['slippage_pct']
        
        if len(buy_slippage) > 0:
            # Calculate rolling mean (1 hour window)
            buy_rolling = buy_slippage.rolling('1H', min_periods=10).mean()
            ax1.plot(buy_rolling.index, buy_rolling.values, label=f'Buy ${size}', alpha=0.8)
        
        if len(sell_slippage) > 0:
            sell_rolling = sell_slippage.rolling('1H', min_periods=10).mean()
            ax1.plot(sell_rolling.index, sell_rolling.values, label=f'Sell ${size}', alpha=0.8, linestyle='--')
    
    ax1.set_title(f'{symbol_short} - Slippage Over Time (1H Rolling Mean)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Slippage %', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(analysis_results['timestamps'])//1440//10)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Slippage Distribution by Size
    ax2 = fig.add_subplot(gs[1, 0])
    
    slippage_data = []
    labels = []
    
    for size in sizes_usd:
        buy_df = analysis_results['slippage_by_size'][f'{size}_buy']
        buy_slippage = buy_df[buy_df['can_execute'] == True]['slippage_pct'].dropna()
        
        if len(buy_slippage) > 0:
            slippage_data.append(buy_slippage.values)
            labels.append(f'${size}')
    
    if slippage_data:
        bp = ax2.boxplot(slippage_data, labels=labels, patch_artist=True, showfliers=False)
        
        # Color the boxes
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    ax2.set_title(f'{symbol_short} - Slippage Distribution by Order Size (Buy)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Order Size (USD)', fontsize=11)
    ax2.set_ylabel('Slippage %', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Execution Success Rate
    ax3 = fig.add_subplot(gs[1, 1])
    
    sizes_str = [f'${s}' for s in sizes_usd]
    buy_success = [stats['execution_success_rate'].get(f'{s}_buy', 0) for s in sizes_usd]
    sell_success = [stats['execution_success_rate'].get(f'{s}_sell', 0) for s in sizes_usd]
    
    x = np.arange(len(sizes_str))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, buy_success, width, label='Buy', color='green', alpha=0.7)
    bars2 = ax3.bar(x + width/2, sell_success, width, label='Sell', color='red', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax3.set_title(f'{symbol_short} - Execution Success Rate by Order Size', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Order Size', fontsize=11)
    ax3.set_ylabel('Success Rate %', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(sizes_str)
    ax3.legend()
    ax3.set_ylim(0, 110)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Maximum Executable Size by Slippage Tolerance
    ax4 = fig.add_subplot(gs[2, :])
    
    # Time series of max executable size for different slippage tolerances
    for slippage in slippage_targets[:3]:  # Show first 3 targets
        buy_df = analysis_results['max_size_by_slippage'][f'{slippage}_buy']
        
        # Calculate rolling mean
        max_sizes = buy_df['max_size_usd'].rolling('1H', min_periods=10).mean()
        
        ax4.plot(max_sizes.index, max_sizes.values, 
                label=f'{slippage}% slippage', alpha=0.8, linewidth=2)
    
    ax4.set_title(f'{symbol_short} - Maximum Executable Size by Slippage Tolerance (1H Rolling Mean)', 
                 fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time', fontsize=11)
    ax4.set_ylabel('Max Size (USD)', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(analysis_results['timestamps'])//1440//10)))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # 5. Liquidity Heatmap - Hour of Day vs Day of Week
    ax5 = fig.add_subplot(gs[3, :])
    
    # Calculate average max size by hour and day of week
    timestamps = pd.DatetimeIndex(analysis_results['timestamps'])
    max_size_df = analysis_results['max_size_by_slippage']['0.5_buy']  # Use 0.5% slippage as reference
    
    # Create hour/day aggregation
    liquidity_matrix = pd.DataFrame(index=range(24), columns=range(7))
    
    for hour in range(24):
        for dow in range(7):
            mask = (timestamps.hour == hour) & (timestamps.dayofweek == dow)
            if mask.sum() > 0:
                liquidity_matrix.loc[hour, dow] = max_size_df.loc[mask, 'max_size_usd'].mean()
    
    # Convert to numeric and plot heatmap
    liquidity_matrix = liquidity_matrix.astype(float)
    
    sns.heatmap(liquidity_matrix, ax=ax5, cmap='YlOrRd', annot=False, fmt='.0f',
                cbar_kws={'label': 'Avg Max Size (USD)'})
    
    ax5.set_title(f'{symbol_short} - Liquidity Heatmap (0.5% Slippage Tolerance)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Day of Week', fontsize=11)
    ax5.set_ylabel('Hour of Day (UTC)', fontsize=11)
    ax5.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    # 6. Bid-Ask Imbalance
    ax6 = fig.add_subplot(gs[4, 0])
    
    # Calculate bid/ask volume imbalance
    orderbook_df = data['orderbook']
    
    # Calculate total bid and ask volumes
    bid_volume = orderbook_df[[f'bid{i}_size' for i in range(1, 11)]].sum(axis=1)
    ask_volume = orderbook_df[[f'ask{i}_size' for i in range(1, 11)]].sum(axis=1)
    
    # Calculate imbalance
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) * 100
    imbalance_clean = imbalance.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Plot histogram
    ax6.hist(imbalance_clean, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax6.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax6.set_title(f'{symbol_short} - Order Book Imbalance Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Imbalance % (Positive = More Bids)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    # 7. Liquidity Score Summary
    ax7 = fig.add_subplot(gs[4, 1])
    ax7.axis('off')
    
    # Create summary statistics table
    summary_data = []
    
    # Average slippage for common sizes
    for size in [100, 500, 1000]:
        buy_stats = stats['slippage_stats'].get(f'{size}_buy', {})
        sell_stats = stats['slippage_stats'].get(f'{size}_sell', {})
        
        summary_data.append([
            f'${size}',
            f"{buy_stats.get('mean', 0):.3f}%",
            f"{sell_stats.get('mean', 0):.3f}%",
            f"{buy_stats.get('success_rate', 0):.1f}%"
        ])
    
    # Create table
    table = ax7.table(cellText=summary_data,
                     colLabels=['Size', 'Avg Buy Slip', 'Avg Sell Slip', 'Success Rate'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.5, 1, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Style the table
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax7.set_title(f'{symbol_short} - Liquidity Summary', fontsize=12, fontweight='bold')
    
    # 8. Depth Evolution
    ax8 = fig.add_subplot(gs[5, :])
    
    # Calculate total depth at different price levels
    depth_levels = [0.1, 0.2, 0.5, 1.0]  # Percentage from best price
    
    for level_pct in depth_levels[:3]:
        level_depths = []
        
        for idx, row in orderbook_df.iterrows():
            # Calculate depth within level_pct of best price
            best_bid = row.get('bid1_price', 0)
            best_ask = row.get('ask1_price', 0)
            
            if best_bid > 0 and best_ask > 0:
                bid_threshold = best_bid * (1 - level_pct/100)
                ask_threshold = best_ask * (1 + level_pct/100)
                
                total_depth_usd = 0
                
                # Sum bid depth
                for i in range(1, 11):
                    price = row.get(f'bid{i}_price', 0)
                    size = row.get(f'bid{i}_size', 0)
                    if price >= bid_threshold and price > 0 and size > 0:
                        total_depth_usd += price * size
                
                # Sum ask depth
                for i in range(1, 11):
                    price = row.get(f'ask{i}_price', 0)
                    size = row.get(f'ask{i}_size', 0)
                    if price <= ask_threshold and price > 0 and size > 0:
                        total_depth_usd += price * size
                
                level_depths.append(total_depth_usd)
            else:
                level_depths.append(0)
        
        # Convert to series and plot rolling mean
        depth_series = pd.Series(level_depths, index=orderbook_df.index)
        depth_rolling = depth_series.rolling('1H', min_periods=10).mean()

        ax8.plot(depth_rolling.index, depth_rolling.values,
                 label=f'±{level_pct}% from mid', alpha=0.8, linewidth=2)

    ax8.set_title(f'{symbol_short} - Order Book Depth Evolution (1H Rolling Mean)', 
                 fontsize=14, fontweight='bold')
    ax8.set_xlabel('Time', fontsize=11)
    ax8.set_ylabel('Total Depth (USD)', fontsize=11)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.set_yscale('log')

    # Format x-axis
    ax8.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax8.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(orderbook_df)//1440//10)))
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle(f'{symbol_short} - Comprehensive Liquidity Analysis', fontsize=16, fontweight='bold', y=0.995)

    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{symbol_short}_liquidity_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    log.info(f"Liquidity analysis plot saved: plots/{symbol_short}_liquidity_analysis.png")

def create_detailed_slippage_plots(symbol: str, analysis_results: Dict, sizes_usd: List[float]):
   """Create detailed slippage analysis plots"""
   log.info(f"Creating detailed slippage plots for {symbol}...")
   
   symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
   
   fig = plt.figure(figsize=(18, 14))
   gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)
   
   # 1. Slippage Curves - Average slippage vs Order Size
   ax1 = fig.add_subplot(gs[0, :])
   
   # Calculate average slippage for each size
   buy_slippages = []
   sell_slippages = []
   sizes_successful = []
   
   for size in sizes_usd:
       buy_df = analysis_results['slippage_by_size'][f'{size}_buy']
       sell_df = analysis_results['slippage_by_size'][f'{size}_sell']
       
       buy_successful = buy_df[buy_df['can_execute'] == True]['slippage_pct']
       sell_successful = sell_df[sell_df['can_execute'] == True]['slippage_pct']
       
       if len(buy_successful) > 0 and len(sell_successful) > 0:
           buy_slippages.append(buy_successful.mean())
           sell_slippages.append(sell_successful.mean())
           sizes_successful.append(size)
   
   if sizes_successful:
       ax1.plot(sizes_successful, buy_slippages, 'o-', label='Buy', color='green', 
               linewidth=2, markersize=8, alpha=0.8)
       ax1.plot(sizes_successful, sell_slippages, 's-', label='Sell', color='red', 
               linewidth=2, markersize=8, alpha=0.8)
       
       # Add percentile bands
       buy_p95 = []
       sell_p95 = []
       
       for size in sizes_successful:
           buy_df = analysis_results['slippage_by_size'][f'{size}_buy']
           sell_df = analysis_results['slippage_by_size'][f'{size}_sell']
           
           buy_successful = buy_df[buy_df['can_execute'] == True]['slippage_pct']
           sell_successful = sell_df[sell_df['can_execute'] == True]['slippage_pct']
           
           buy_p95.append(buy_successful.quantile(0.95))
           sell_p95.append(sell_successful.quantile(0.95))
       
       ax1.fill_between(sizes_successful, buy_slippages, buy_p95, alpha=0.2, color='green', 
                       label='Buy 95th percentile')
       ax1.fill_between(sizes_successful, sell_slippages, sell_p95, alpha=0.2, color='red',
                       label='Sell 95th percentile')
   
   ax1.set_title(f'{symbol_short} - Slippage vs Order Size', fontsize=14, fontweight='bold')
   ax1.set_xlabel('Order Size (USD)', fontsize=11)
   ax1.set_ylabel('Slippage %', fontsize=11)
   ax1.set_xscale('log')
   ax1.legend(fontsize=9)
   ax1.grid(True, alpha=0.3)
   
   # 2. Levels Used Distribution
   ax2 = fig.add_subplot(gs[1, 0])
   
   # Collect levels used data
   levels_data = []
   
   for size in [100, 500, 1000]:  # Select specific sizes
       buy_df = analysis_results['slippage_by_size'][f'{size}_buy']
       successful = buy_df[buy_df['can_execute'] == True]
       
       if len(successful) > 0:
           levels_used = successful['levels_used'].value_counts().sort_index()
           levels_data.append({
               'size': size,
               'levels': levels_used
           })
   
   if levels_data:
       width = 0.25
       x_base = np.arange(1, 11)
       
       for i, data in enumerate(levels_data):
           x = x_base + i * width
           levels = data['levels']
           
           # Fill missing levels with 0
           y = [levels.get(j, 0) for j in range(1, 11)]
           
           ax2.bar(x, y, width, label=f"${data['size']}", alpha=0.8)
   
   ax2.set_title(f'{symbol_short} - Order Book Levels Used (Buy Side)', fontsize=12, fontweight='bold')
   ax2.set_xlabel('Number of Levels Used', fontsize=11)
   ax2.set_ylabel('Frequency', fontsize=11)
   ax2.set_xticks(x_base + width)
   ax2.set_xticklabels(x_base)
   ax2.legend(fontsize=9)
   ax2.grid(True, alpha=0.3, axis='y')
   
   # 3. Slippage Time Pattern
   ax3 = fig.add_subplot(gs[1, 1])
   
   # Analyze slippage by hour of day
   buy_df = analysis_results['slippage_by_size']['500_buy']  # Use $500 as reference
   successful = buy_df[buy_df['can_execute'] == True].copy()
   
   if len(successful) > 0:
       successful['hour'] = successful.index.hour
       hourly_slippage = successful.groupby('hour')['slippage_pct'].agg(['mean', 'std'])
       
       hours = hourly_slippage.index
       means = hourly_slippage['mean']
       stds = hourly_slippage['std']
       
       ax3.plot(hours, means, 'o-', linewidth=2, markersize=6, color='steelblue', label='Mean')
       ax3.fill_between(hours, means - stds, means + stds, alpha=0.3, color='steelblue', 
                       label='±1 Std Dev')
   
   ax3.set_title(f'{symbol_short} - Intraday Slippage Pattern ($500 Buy)', fontsize=12, fontweight='bold')
   ax3.set_xlabel('Hour of Day (UTC)', fontsize=11)
   ax3.set_ylabel('Slippage %', fontsize=11)
   ax3.set_xticks(range(0, 24, 3))
   ax3.legend(fontsize=9)
   ax3.grid(True, alpha=0.3)
   
   # 4. Slippage vs Market Conditions
   ax4 = fig.add_subplot(gs[2, :])
   
   # Analyze relationship between slippage and order book imbalance
   orderbook_df = analysis_results['slippage_by_size']['500_buy']
   
   if len(orderbook_df) > 0:
       # Calculate order book imbalance for each timestamp
       imbalances = []
       slippages = []
       
       for idx, row in orderbook_df.iterrows():
           if row['can_execute'] and pd.notna(row['slippage_pct']):
               # Find corresponding orderbook data
               orderbook_row = data['orderbook'].loc[idx]
               
               # Calculate simple imbalance
               bid_vol = sum(orderbook_row.get(f'bid{i}_size', 0) for i in range(1, 6))
               ask_vol = sum(orderbook_row.get(f'ask{i}_size', 0) for i in range(1, 6))
               
               if bid_vol + ask_vol > 0:
                   imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
                   imbalances.append(imbalance)
                   slippages.append(row['slippage_pct'])
       
       if imbalances:
           # Create 2D histogram
           h = ax4.hist2d(imbalances, slippages, bins=[30, 30], cmap='YlOrRd', alpha=0.8)
           cbar = plt.colorbar(h[3], ax=ax4)
           cbar.set_label('Frequency', fontsize=10)
           
           # Add regression line
           z = np.polyfit(imbalances, slippages, 1)
           p = np.poly1d(z)
           x_line = np.linspace(min(imbalances), max(imbalances), 100)
           ax4.plot(x_line, p(x_line), 'b--', linewidth=2, alpha=0.8, 
                   label=f'Linear fit: {z[0]:.3f}x + {z[1]:.3f}')
   
   ax4.set_title(f'{symbol_short} - Slippage vs Order Book Imbalance ($500 Buy)', 
                fontsize=14, fontweight='bold')
   ax4.set_xlabel('Order Book Imbalance (Bid-Ask) / (Bid+Ask)', fontsize=11)
   ax4.set_ylabel('Slippage %', fontsize=11)
   ax4.legend(fontsize=9)
   ax4.grid(True, alpha=0.3)
   
   plt.suptitle(f'{symbol_short} - Detailed Slippage Analysis', fontsize=16, fontweight='bold', y=0.995)
   
   # Save plot
   plots_dir = Path("plots")
   plots_dir.mkdir(exist_ok=True)
   plt.savefig(plots_dir / f'{symbol_short}_slippage_details.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
   plt.close()
   
   log.info(f"Detailed slippage plot saved: plots/{symbol_short}_slippage_details.png")

def create_liquidity_depth_plots(symbol: str, data: Dict, analysis_results: Dict):
   """Create liquidity depth visualization plots"""
   log.info(f"Creating liquidity depth plots for {symbol}...")
   
   symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
   
   fig = plt.figure(figsize=(18, 10))
   gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
   
   # 1. Average Order Book Shape
   ax1 = fig.add_subplot(gs[0, :])
   
   orderbook_df = data['orderbook']
   
   # Calculate average depth at each level
   bid_depths = []
   ask_depths = []
   
   for i in range(1, 11):
       # Calculate average size * price for each level
       bid_depth = (orderbook_df[f'bid{i}_price'] * orderbook_df[f'bid{i}_size']).mean()
       ask_depth = (orderbook_df[f'ask{i}_price'] * orderbook_df[f'ask{i}_size']).mean()
       
       if pd.notna(bid_depth):
           bid_depths.append(bid_depth)
       if pd.notna(ask_depth):
           ask_depths.append(ask_depth)
   
   levels = range(1, len(bid_depths) + 1)
   
   # Create bar plot
   width = 0.35
   x = np.arange(len(levels))
   
   bars1 = ax1.bar(x - width/2, bid_depths, width, label='Bid', color='green', alpha=0.7)
   bars2 = ax1.bar(x + width/2, ask_depths, width, label='Ask', color='red', alpha=0.7)
   
   ax1.set_title(f'{symbol_short} - Average Order Book Depth by Level', fontsize=14, fontweight='bold')
   ax1.set_xlabel('Level', fontsize=11)
   ax1.set_ylabel('Average Depth (USD)', fontsize=11)
   ax1.set_xticks(x)
   ax1.set_xticklabels(levels)
   ax1.legend(fontsize=9)
   ax1.grid(True, alpha=0.3, axis='y')
   ax1.set_yscale('log')
   
   # 2. Liquidity Concentration
   ax2 = fig.add_subplot(gs[1, 0])
   
   # Calculate what percentage of total liquidity is in each level
   total_bid_liquidity = sum(bid_depths) if bid_depths else 1
   total_ask_liquidity = sum(ask_depths) if ask_depths else 1
   
   bid_pct = [d/total_bid_liquidity*100 for d in bid_depths]
   ask_pct = [d/total_ask_liquidity*100 for d in ask_depths]
   
   # Calculate cumulative percentages
   bid_cumsum = np.cumsum(bid_pct)
   ask_cumsum = np.cumsum(ask_pct)
   
   ax2.plot(levels[:len(bid_cumsum)], bid_cumsum, 'o-', label='Bid Cumulative', 
           color='green', linewidth=2, markersize=6)
   ax2.plot(levels[:len(ask_cumsum)], ask_cumsum, 's-', label='Ask Cumulative', 
           color='red', linewidth=2, markersize=6)
   
   # Add reference lines
   ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
   ax2.axhline(y=80, color='gray', linestyle=':', alpha=0.5, label='80%')
   
   ax2.set_title(f'{symbol_short} - Liquidity Concentration by Level', fontsize=12, fontweight='bold')
   ax2.set_xlabel('Level', fontsize=11)
   ax2.set_ylabel('Cumulative Liquidity %', fontsize=11)
   ax2.set_ylim(0, 105)
   ax2.legend(fontsize=9)
   ax2.grid(True, alpha=0.3)
   
   # 3. Maximum Executable Size Distribution
   ax3 = fig.add_subplot(gs[1, 1])
   
   # Get max executable sizes for 0.5% slippage
   max_size_df = analysis_results['max_size_by_slippage']['0.5_buy']
   max_sizes = max_size_df['max_size_usd'].dropna()
   
   if len(max_sizes) > 0:
       # Create histogram
       counts, bins, patches = ax3.hist(np.log10(max_sizes), bins=50, alpha=0.7, 
                                       color='purple', edgecolor='black')
       
       # Add percentile lines
       p10 = np.log10(max_sizes.quantile(0.10))
       p50 = np.log10(max_sizes.quantile(0.50))
       p90 = np.log10(max_sizes.quantile(0.90))
       
       ax3.axvline(p10, color='red', linestyle='--', linewidth=2, alpha=0.8, label='P10')
       ax3.axvline(p50, color='orange', linestyle='-', linewidth=2, alpha=0.8, label='P50')
       ax3.axvline(p90, color='green', linestyle='--', linewidth=2, alpha=0.8, label='P90')
       
       # Add statistics text
       stats_text = f'P10: ${10**p10:,.0f}\nP50: ${10**p50:,.0f}\nP90: ${10**p90:,.0f}'
       ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", 
               facecolor="lightgray", alpha=0.8))
   
   ax3.set_title(f'{symbol_short} - Max Executable Size Distribution (0.5% Slippage)', 
                fontsize=12, fontweight='bold')
   ax3.set_xlabel('Log10(Max Size USD)', fontsize=11)
   ax3.set_ylabel('Frequency', fontsize=11)
   ax3.legend(fontsize=9)
   ax3.grid(True, alpha=0.3)
   
   plt.suptitle(f'{symbol_short} - Order Book Depth Analysis', fontsize=16, fontweight='bold', y=0.995)
   
   # Save plot
   plots_dir = Path("plots")
   plots_dir.mkdir(exist_ok=True)
   plt.savefig(plots_dir / f'{symbol_short}_liquidity_depth.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
   plt.close()
   
   log.info(f"Liquidity depth plot saved: plots/{symbol_short}_liquidity_depth.png")

def generate_liquidity_report(symbol: str, stats: Dict, analysis_results: Dict):
   """Generate comprehensive liquidity analysis report"""
   log.info(f"\n{'='*80}")
   log.info(f"LIQUIDITY ANALYSIS REPORT - {symbol}")
   log.info(f"{'='*80}")
   
   symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
   
   # Overall summary
   log.info(f"\n📊 OVERALL LIQUIDITY SUMMARY:")
   log.info(f"Data Period: {analysis_results['timestamps'][0]} to {analysis_results['timestamps'][-1]}")
   log.info(f"Total Snapshots Analyzed: {len(analysis_results['timestamps']):,}")
   
   # Slippage summary
   log.info(f"\n💹 SLIPPAGE ANALYSIS:")
   log.info(f"{'Size (USD)':<12} {'Buy Mean':<12} {'Buy P95':<12} {'Sell Mean':<12} {'Sell P95':<12} {'Success %':<10}")
   log.info("-" * 70)
   
   for size in [50, 100, 200, 500, 1000, 2000]:
       buy_stats = stats['slippage_stats'].get(f'{size}_buy', {})
       sell_stats = stats['slippage_stats'].get(f'{size}_sell', {})
       success = stats['execution_success_rate'].get(f'{size}_buy', 0)
       
       log.info(f"${size:<11} {buy_stats.get('mean', 0):<11.4f}% {buy_stats.get('p95', 0):<11.4f}% "
               f"{sell_stats.get('mean', 0):<11.4f}% {sell_stats.get('p95', 0):<11.4f}% {success:<9.1f}%")
   
   # Max executable size summary
   log.info(f"\n📈 MAXIMUM EXECUTABLE SIZE (USD):")
   log.info(f"{'Slippage %':<12} {'Buy Mean':<15} {'Buy P10':<15} {'Sell Mean':<15} {'Sell P10':<15}")
   log.info("-" * 75)
   
   for slippage in [0.1, 0.5, 1.0, 2.0, 3.0]:
       buy_stats = stats['max_size_stats'].get(f'{slippage}_buy', {})
       sell_stats = stats['max_size_stats'].get(f'{slippage}_sell', {})
       
       log.info(f"{slippage:<11.1f}% ${buy_stats.get('mean', 0):<14,.0f} ${buy_stats.get('p10', 0):<14,.0f} "
               f"${sell_stats.get('mean', 0):<14,.0f} ${sell_stats.get('p10', 0):<14,.0f}")
   
   # Liquidity quality assessment
   log.info(f"\n🏆 LIQUIDITY QUALITY ASSESSMENT:")
   
   # Calculate overall liquidity score
   avg_success_rate = np.mean([stats['execution_success_rate'].get(f'{s}_buy', 0) 
                               for s in [100, 500, 1000]])
   avg_slippage_500 = stats['slippage_stats'].get('500_buy', {}).get('mean', 0)
   avg_max_size_half_pct = stats['max_size_stats'].get('0.5_buy', {}).get('mean', 0)
   
   if avg_success_rate >= 95 and avg_slippage_500 < 0.1 and avg_max_size_half_pct > 10000:
       quality = "EXCELLENT"
       grade = "A+"
   elif avg_success_rate >= 90 and avg_slippage_500 < 0.2 and avg_max_size_half_pct > 5000:
       quality = "VERY GOOD"
       grade = "A"
   elif avg_success_rate >= 80 and avg_slippage_500 < 0.5 and avg_max_size_half_pct > 2000:
       quality = "GOOD"
       grade = "B"
   elif avg_success_rate >= 70 and avg_slippage_500 < 1.0 and avg_max_size_half_pct > 1000:
       quality = "FAIR"
       grade = "C"
   else:
       quality = "POOR"
       grade = "D"
   
   log.info(f"Overall Liquidity Grade: {grade} ({quality})")
   log.info(f"  - Average execution success rate (≤$1000): {avg_success_rate:.1f}%")
   log.info(f"  - Average slippage for $500 orders: {avg_slippage_500:.4f}%")
   log.info(f"  - Average max size at 0.5% slippage: ${avg_max_size_half_pct:,.0f}")
   
   # Trading recommendations
   log.info(f"\n📋 TRADING RECOMMENDATIONS:")
   
   if grade in ['A+', 'A']:
       log.info(f"✅ {symbol_short} has EXCELLENT liquidity for algorithmic trading")
       log.info(f"  - Can execute orders up to $1000 with minimal slippage")
       log.info(f"  - Suitable for high-frequency strategies")
       log.info(f"  - Low market impact for typical retail sizes")
   elif grade == 'B':
       log.info(f"✅ {symbol_short} has GOOD liquidity for most trading strategies")
       log.info(f"  - Can execute orders up to $500 reliably")
       log.info(f"  - Consider using limit orders for larger sizes")
       log.info(f"  - Monitor slippage during volatile periods")
   elif grade == 'C':
       log.info(f"⚠️ {symbol_short} has FAIR liquidity - trade with caution")
       log.info(f"  - Limit order sizes to $200 or less for market orders")
       log.info(f"  - Use limit orders for better execution")
       log.info(f"  - Avoid trading during low liquidity periods")
   else:
       log.info(f"❌ {symbol_short} has POOR liquidity - not recommended for active trading")
       log.info(f"  - Very high slippage risk")
       log.info(f"  - Limited executable size")
       log.info(f"  - Consider alternative instruments")
   
   # Best trading times
   log.info(f"\n⏰ OPTIMAL TRADING TIMES:")
   
   # Analyze liquidity by hour
   hourly_liquidity = {}
   for hour in range(24):
       hour_mask = pd.DatetimeIndex(analysis_results['timestamps']).hour == hour
       hour_data = []
       
       for key in ['0.5_buy', '0.5_sell']:
           df = analysis_results['max_size_by_slippage'][key]
           hour_values = df.loc[hour_mask, 'max_size_usd']
           if len(hour_values) > 0:
               hour_data.extend(hour_values.values)
       
       if hour_data:
           hourly_liquidity[hour] = np.mean(hour_data)
   
   if hourly_liquidity:
       sorted_hours = sorted(hourly_liquidity.items(), key=lambda x: x[1], reverse=True)
       
       log.info(f"Best liquidity hours (UTC):")
       for hour, liquidity in sorted_hours[:5]:
           log.info(f"  - {hour:02d}:00-{hour:02d}:59: Avg max size ${liquidity:,.0f}")
       
       log.info(f"\nWorst liquidity hours (UTC):")
       for hour, liquidity in sorted_hours[-3:]:
           log.info(f"  - {hour:02d}:00-{hour:02d}:59: Avg max size ${liquidity:,.0f}")
   
   log.info(f"\n📁 GENERATED FILES:")
   log.info(f"  - Comprehensive liquidity analysis: plots/{symbol_short}_liquidity_analysis.png")
   log.info(f"  - Detailed slippage analysis: plots/{symbol_short}_slippage_details.png")
   log.info(f"  - Order book depth analysis: plots/{symbol_short}_liquidity_depth.png")
   
   log.info(f"\n{'='*80}")

def main():
   """Main liquidity analysis function"""
   import argparse
   
   parser = argparse.ArgumentParser(description="Analyze market liquidity using orderbook data")
   parser.add_argument("--symbol", type=str, help="Specific symbol to analyze")
   parser.add_argument("--days", type=int, default=7, help="Number of days to analyze (default: 7)")
   parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
   
   args = parser.parse_args()
   
   log.info("Starting comprehensive liquidity analysis...")
   
   # Get symbols to analyze
   if args.symbol:
       symbols = [args.symbol]
   else:
       try:
           active_pairs = settings.get_active_pairs()
           symbols = []
           for pair in active_pairs:
               symbols.extend([pair.symbol1, pair.symbol2])
           symbols = list(set(symbols))
       except Exception as e:
           log.error(f"Could not load symbols from config: {e}")
           symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
   
   if not symbols:
       log.error("No symbols to analyze")
       return False
   
   # Define analysis parameters
   sizes_usd = [50, 100, 200, 500, 1000, 2000]
   slippage_targets = [0.1, 0.5, 1.0, 2.0, 3.0]
   
   # Calculate date range
   end_date = datetime.now()
   start_date = end_date - timedelta(days=args.days)
   
   log.info(f"Analyzing {len(symbols)} symbols from {start_date} to {end_date}")
   log.info(f"Order sizes: {sizes_usd}")
   log.info(f"Slippage targets: {slippage_targets}")
   
   # Create plots directory
   plots_dir = Path("plots")
   plots_dir.mkdir(exist_ok=True)
   
   try:
       for symbol in symbols:
           log.info(f"\n{'='*60}")
           log.info(f"ANALYZING LIQUIDITY FOR {symbol}")
           log.info(f"{'='*60}")
           
           # Load data
           try:
               data = load_liquidity_data(symbol, start_date, end_date)
               
               if len(data['orderbook']) == 0:
                   log.warning(f"No orderbook data for {symbol}")
                   continue
               
               if len(data['mark_prices']) == 0:
                   log.warning(f"No mark price data for {symbol}")
                   continue
               
               # Analyze liquidity metrics
               analysis_results = analyze_liquidity_metrics(symbol, data, sizes_usd, slippage_targets)
               
               if not analysis_results:
                   log.warning(f"No analysis results for {symbol}")
                   continue
               
               # Calculate statistics
               stats = calculate_liquidity_statistics(analysis_results)
               
               # Generate plots
               if not args.no_plots:
                   create_liquidity_plots(symbol, analysis_results, stats, sizes_usd, slippage_targets)
                   create_detailed_slippage_plots(symbol, analysis_results, sizes_usd)
                   create_liquidity_depth_plots(symbol, data, analysis_results)
               
               # Generate report
               generate_liquidity_report(symbol, stats, analysis_results)
               
           except Exception as e:
               log.error(f"Failed to analyze {symbol}: {e}")
               import traceback
               log.error(traceback.format_exc())
               continue
       
       log.info(f"\n🎉 Liquidity analysis completed successfully!")
       log.info(f"Analyzed: {len(symbols)} symbols")
       log.info(f"Time period: {args.days} days")
       
       return True
       
   except Exception as e:
       log.error(f"Liquidity analysis failed: {e}")
       import traceback
       log.error(traceback.format_exc())
       return False

if __name__ == "__main__":
   success = main()
   sys.exit(0 if success else 1)