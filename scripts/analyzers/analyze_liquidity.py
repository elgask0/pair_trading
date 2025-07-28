#!/usr/bin/env python3
"""
Enhanced liquidity analysis script - IMPROVED VERSION WITH DATABASE SYMBOL SOURCING
Analyzes market liquidity using ALL available orderbook data with comprehensive slippage analysis
UPDATED: Uses database (symbol_info table) instead of YAML for symbol sourcing
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on PYTHONPATH for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

# Enhanced matplotlib settings
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_palette("husl")

def get_data_range(symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get full data range for symbol"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as total_records
            FROM orderbook 
            WHERE symbol = :symbol
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
        """), {'symbol': symbol}).fetchone()
        
        return result.min_ts, result.max_ts, result.total_records

def load_all_orderbook_data(symbol: str, max_records: int = 100000) -> pd.DataFrame:
    """Load ALL available orderbook data for comprehensive analysis"""
    min_date, max_date, total_records = get_data_range(symbol)
    
    if not min_date or not max_date:
        log.warning(f"No orderbook data found for {symbol}")
        return pd.DataFrame()
    
    log.info(f"Loading orderbook data for {symbol}:")
    log.info(f"  Period: {min_date} to {max_date}")
    log.info(f"  Total records available: {total_records:,}")
    
    # If too much data, sample intelligently
    sample_clause = ""
    if total_records > max_records:
        sample_rate = max_records / total_records
        sample_clause = f"TABLESAMPLE SYSTEM ({sample_rate * 100:.2f})"
        log.info(f"  Sampling {sample_rate:.1%} of data ({max_records:,} records)")
    
    with db_manager.get_session() as session:
        query = text(f"""
            SELECT 
                timestamp,
                bid1_price, bid1_size, bid2_price, bid2_size, bid3_price, bid3_size,
                bid4_price, bid4_size, bid5_price, bid5_size,
                ask1_price, ask1_size, ask2_price, ask2_size, ask3_price, ask3_size,
                ask4_price, ask4_size, ask5_price, ask5_size
            FROM orderbook {sample_clause}
            WHERE symbol = :symbol 
            AND timestamp >= :start_date 
            AND timestamp <= :end_date
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params={
            'symbol': symbol,
            'start_date': min_date,
            'end_date': max_date
        }, index_col='timestamp')
        
        log.info(f"  Loaded: {len(df):,} orderbook snapshots")
        return df

def calculate_enhanced_slippage(row: pd.Series, order_size_usd: float, side: str = 'buy') -> Dict:
    """Enhanced slippage calculation with more detailed metrics"""
    
    levels = []
    prefix = 'ask' if side == 'buy' else 'bid'
    
    # Extract top 5 levels
    for i in range(1, 6):
        price = row.get(f'{prefix}{i}_price')
        size = row.get(f'{prefix}{i}_size')
        
        if pd.notna(price) and pd.notna(size) and price > 0 and size > 0:
            levels.append({
                'price': price,
                'size': size,
                'value_usd': price * size,
                'level': i
            })
    
    if not levels:
        return {
            'can_execute': False,
            'slippage_pct': np.nan,
            'slippage_bps': np.nan,
            'levels_used': 0,
            'liquidity_consumed_pct': 0,
            'worst_price': np.nan,
            'price_impact': np.nan
        }
    
    # Sort by price (best first)
    levels.sort(key=lambda x: x['price'], reverse=(side == 'sell'))
    
    # Calculate execution
    remaining_usd = order_size_usd
    total_value = 0
    total_coins = 0
    levels_used = 0
    total_liquidity = sum(level['value_usd'] for level in levels)
    
    for level in levels:
        if remaining_usd <= 0:
            break
            
        available_usd = level['value_usd']
        
        if available_usd >= remaining_usd:
            # This level can fill the remaining order
            coins_bought = remaining_usd / level['price']
            total_value += remaining_usd
            total_coins += coins_bought
            levels_used += 1
            remaining_usd = 0
        else:
            # Use entire level
            total_value += available_usd
            total_coins += level['size']
            levels_used += 1
            remaining_usd -= available_usd
    
    if remaining_usd > 0:
        return {
            'can_execute': False,
            'slippage_pct': np.nan,
            'slippage_bps': np.nan,
            'levels_used': levels_used,
            'liquidity_consumed_pct': 100,  # Not enough liquidity
            'worst_price': levels[-1]['price'] if levels else np.nan,
            'price_impact': np.nan
        }
    
    # Calculate enhanced metrics
    avg_price = total_value / total_coins
    best_price = levels[0]['price']
    worst_price = levels[levels_used-1]['price'] if levels_used > 0 else best_price
    
    slippage_pct = abs(avg_price - best_price) / best_price * 100
    slippage_bps = slippage_pct * 100  # basis points
    liquidity_consumed_pct = (order_size_usd / total_liquidity) * 100 if total_liquidity > 0 else 100
    price_impact = abs(worst_price - best_price) / best_price * 100
    
    return {
        'can_execute': True,
        'slippage_pct': slippage_pct,
        'slippage_bps': slippage_bps,
        'levels_used': levels_used,
        'liquidity_consumed_pct': liquidity_consumed_pct,
        'worst_price': worst_price,
        'price_impact': price_impact,
        'avg_price': avg_price,
        'best_price': best_price
    }

def analyze_comprehensive_slippage(symbol: str, orderbook_df: pd.DataFrame) -> Dict:
    """Comprehensive slippage analysis with detailed statistics"""
    log.info(f"Analyzing comprehensive slippage metrics for {symbol}...")
    
    # Extended order sizes for better analysis
    order_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]  # USD
    
    results = {
        'timestamps': orderbook_df.index.tolist(),
        'detailed_slippage': {},
        'distribution_stats': {},
        'execution_reliability': {}
    }
    
    for size in order_sizes:
        log.info(f"  Processing ${size} orders...")
        
        buy_data = {
            'slippage_pct': [],
            'slippage_bps': [],
            'levels_used': [],
            'liquidity_consumed': [],
            'price_impact': [],
            'execution_success': []
        }
        
        sell_data = {
            'slippage_pct': [],
            'slippage_bps': [],
            'levels_used': [],
            'liquidity_consumed': [],
            'price_impact': [],
            'execution_success': []
        }
        
        # Process each orderbook snapshot
        for idx, row in orderbook_df.iterrows():
            # Buy side analysis
            buy_result = calculate_enhanced_slippage(row, size, 'buy')
            sell_result = calculate_enhanced_slippage(row, size, 'sell')
            
            for data, result in [(buy_data, buy_result), (sell_data, sell_result)]:
                data['execution_success'].append(result['can_execute'])
                
                if result['can_execute']:
                    data['slippage_pct'].append(result['slippage_pct'])
                    data['slippage_bps'].append(result['slippage_bps'])
                    data['levels_used'].append(result['levels_used'])
                    data['liquidity_consumed'].append(result['liquidity_consumed_pct'])
                    data['price_impact'].append(result['price_impact'])
        
        # Calculate comprehensive statistics
        def calc_stats(data_list):
            if not data_list:
                return {
                    'mean': np.nan, 'median': np.nan, 'std': np.nan,
                    'p75': np.nan, 'p90': np.nan, 'p95': np.nan, 'p99': np.nan,
                    'min': np.nan, 'max': np.nan
                }
            
            arr = np.array(data_list)
            return {
                'mean': np.mean(arr),
                'median': np.median(arr),
                'std': np.std(arr),
                'p75': np.percentile(arr, 75),
                'p90': np.percentile(arr, 90),
                'p95': np.percentile(arr, 95),
                'p99': np.percentile(arr, 99),
                'min': np.min(arr),
                'max': np.max(arr)
            }
        
        results['detailed_slippage'][size] = {
            'buy': {
                'slippage_pct_stats': calc_stats(buy_data['slippage_pct']),
                'slippage_bps_stats': calc_stats(buy_data['slippage_bps']),
                'levels_used_stats': calc_stats(buy_data['levels_used']),
                'liquidity_consumed_stats': calc_stats(buy_data['liquidity_consumed']),
                'execution_rate': np.mean(buy_data['execution_success']) * 100,
                'raw_data': buy_data  # For box plots
            },
            'sell': {
                'slippage_pct_stats': calc_stats(sell_data['slippage_pct']),
                'slippage_bps_stats': calc_stats(sell_data['slippage_bps']),
                'levels_used_stats': calc_stats(sell_data['levels_used']),
                'liquidity_consumed_stats': calc_stats(sell_data['liquidity_consumed']),
                'execution_rate': np.mean(sell_data['execution_success']) * 100,
                'raw_data': sell_data  # For box plots
            }
        }
    
    return results

def create_enhanced_visualization(symbol: str, analysis_results: Dict):
    """Create comprehensive visualization with box plots and detailed metrics"""
    log.info(f"Creating enhanced visualization for {symbol}...")
    
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    # Create larger figure with more subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'{symbol_short} - Comprehensive Liquidity Analysis', fontsize=20, fontweight='bold')
    
    # 1. Slippage Box Plots - Buy Side (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_slippage_boxplots(ax1, analysis_results, 'buy', 'Slippage %')
    
    # 2. Slippage Box Plots - Sell Side (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_slippage_boxplots(ax2, analysis_results, 'sell', 'Slippage %')
    
    # 3. Execution Success Rates (Top Center-Right)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_execution_rates(ax3, analysis_results)
    
    # 4. Levels Used Distribution (Top Far-Right)
    ax4 = fig.add_subplot(gs[0, 3])
    plot_levels_used(ax4, analysis_results)
    
    # 5. Slippage vs Order Size - Mean (Middle Left)
    ax5 = fig.add_subplot(gs[1, 0])
    plot_slippage_vs_size(ax5, analysis_results, 'mean')
    
    # 6. Slippage vs Order Size - P95 (Middle Center-Left)
    ax6 = fig.add_subplot(gs[1, 1])
    plot_slippage_vs_size(ax6, analysis_results, 'p95')
    
    # 7. Price Impact Analysis (Middle Center-Right)
    ax7 = fig.add_subplot(gs[1, 2])
    plot_price_impact(ax7, analysis_results)
    
    # 8. Liquidity Consumption (Middle Right)
    ax8 = fig.add_subplot(gs[1, 3])
    plot_liquidity_consumption(ax8, analysis_results)
    
    # 9. Trading Quality Summary Table (Bottom)
    ax9 = fig.add_subplot(gs[2, :])
    create_quality_summary_table(ax9, symbol_short, analysis_results)
    
    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{symbol_short}_enhanced_liquidity_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    log.info(f"Enhanced visualization saved: plots/{symbol_short}_enhanced_liquidity_analysis.png")

def plot_slippage_boxplots(ax, analysis_results: Dict, side: str, metric: str):
    """Create box plots for slippage distribution"""
    order_sizes = list(analysis_results['detailed_slippage'].keys())
    slippage_data = []
    labels = []
    
    for size in order_sizes:
        data = analysis_results['detailed_slippage'][size][side]['raw_data']['slippage_pct']
        if data:  # Only include if we have data
            slippage_data.append(data)
            labels.append(f'${size}')
    
    if slippage_data:
        bp = ax.boxplot(slippage_data, tick_labels=labels, patch_artist=True, showfliers=False)
        
        # Color the boxes
        colors = ['lightblue' if side == 'buy' else 'lightcoral' for _ in bp['boxes']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Slippage %')
        ax.set_title(f'{side.title()} Side Slippage Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add mean markers
        for i, data in enumerate(slippage_data):
            mean_val = np.mean(data)
            ax.plot(i+1, mean_val, 'ro' if side == 'sell' else 'bo', markersize=8, label='Mean' if i == 0 else "")
        
        if len(slippage_data) > 0:
            ax.legend()
        # Ajustar ejes: usar rangos de bigotes para límites
        caps = bp['caps']
        cap_vals = [cap.get_ydata()[0] for cap in caps]
        if cap_vals:
            ymin = min(cap_vals)
            ymax = max(cap_vals)
            margin = (ymax - ymin) * 0.1 if (ymax - ymin) > 0 else 0.1
            ax.set_ylim(max(0, ymin - margin), ymax + margin)

def plot_execution_rates(ax, analysis_results: Dict):
    """Plot execution success rates"""
    order_sizes = list(analysis_results['detailed_slippage'].keys())
    buy_rates = [analysis_results['detailed_slippage'][size]['buy']['execution_rate'] for size in order_sizes]
    sell_rates = [analysis_results['detailed_slippage'][size]['sell']['execution_rate'] for size in order_sizes]
    
    x = np.arange(len(order_sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, buy_rates, width, label='Buy', color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, sell_rates, width, label='Sell', color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Success Rate %')
    ax.set_title('Order Execution Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([f'${size}' for size in order_sizes])
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

def plot_levels_used(ax, analysis_results: Dict):
    """Plot distribution of orderbook levels used"""
    order_sizes = list(analysis_results['detailed_slippage'].keys())
    
    # Get average levels used for each size
    buy_levels = []
    sell_levels = []
    
    for size in order_sizes:
        buy_data = analysis_results['detailed_slippage'][size]['buy']['raw_data']['levels_used']
        sell_data = analysis_results['detailed_slippage'][size]['sell']['raw_data']['levels_used']
        
        buy_levels.append(np.mean(buy_data) if buy_data else 0)
        sell_levels.append(np.mean(sell_data) if sell_data else 0)
    
    x = np.arange(len(order_sizes))
    width = 0.35
    
    ax.bar(x - width/2, buy_levels, width, label='Buy', color='lightblue', alpha=0.8)
    ax.bar(x + width/2, sell_levels, width, label='Sell', color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Avg Levels Used')
    ax.set_title('Orderbook Depth Consumption')
    ax.set_xticks(x)
    ax.set_xticklabels([f'${size}' for size in order_sizes])
    ax.legend()

def plot_slippage_vs_size(ax, analysis_results: Dict, stat_type: str):
    """Plot slippage vs order size for specific statistic"""
    order_sizes = list(analysis_results['detailed_slippage'].keys())
    
    buy_slippages = []
    sell_slippages = []
    
    for size in order_sizes:
        buy_stat = analysis_results['detailed_slippage'][size]['buy']['slippage_pct_stats'][stat_type]
        sell_stat = analysis_results['detailed_slippage'][size]['sell']['slippage_pct_stats'][stat_type]
        
        buy_slippages.append(buy_stat if not np.isnan(buy_stat) else 0)
        sell_slippages.append(sell_stat if not np.isnan(sell_stat) else 0)
    
    ax.plot(order_sizes, buy_slippages, 'o-', label='Buy', color='blue', linewidth=2, markersize=6)
    ax.plot(order_sizes, sell_slippages, 's-', label='Sell', color='red', linewidth=2, markersize=6)
    
    ax.set_xlabel('Order Size (USD)')
    ax.set_ylabel(f'{stat_type.upper()} Slippage %')
    ax.set_title(f'Slippage vs Order Size ({stat_type.upper()})')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_price_impact(ax, analysis_results: Dict):
    """Plot price impact analysis"""
    order_sizes = list(analysis_results['detailed_slippage'].keys())
    
    # Calculate combined price impact
    price_impacts = []
    for size in order_sizes:
        buy_impact = analysis_results['detailed_slippage'][size]['buy']['raw_data']['price_impact']
        sell_impact = analysis_results['detailed_slippage'][size]['sell']['raw_data']['price_impact']
        
        combined_impact = buy_impact + sell_impact
        avg_impact = np.mean(combined_impact) if combined_impact else 0
        price_impacts.append(avg_impact)
    
    bars = ax.bar([f'${size}' for size in order_sizes], price_impacts, 
                  color='orange', alpha=0.7)
    
    ax.set_ylabel('Avg Price Impact %')
    ax.set_title('Market Price Impact by Order Size')
    
    # Add value labels
    for bar, impact in zip(bars, price_impacts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{impact:.3f}%', ha='center', va='bottom', fontsize=9)

def plot_liquidity_consumption(ax, analysis_results: Dict):
    """Plot liquidity consumption analysis"""
    order_sizes = list(analysis_results['detailed_slippage'].keys())
    
    # Get P95 liquidity consumption
    consumption_p95 = []
    for size in order_sizes:
        buy_consumption = analysis_results['detailed_slippage'][size]['buy']['liquidity_consumed_stats']['p95']
        sell_consumption = analysis_results['detailed_slippage'][size]['sell']['liquidity_consumed_stats']['p95']
        
        avg_consumption = np.mean([buy_consumption, sell_consumption])
        consumption_p95.append(avg_consumption if not np.isnan(avg_consumption) else 0)
    
    colors = ['green' if x < 20 else 'orange' if x < 50 else 'red' for x in consumption_p95]
    bars = ax.bar([f'${size}' for size in order_sizes], consumption_p95, color=colors, alpha=0.7)
    
    ax.set_ylabel('P95 Liquidity Consumed %')
    ax.set_title('Liquidity Consumption (P95)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='High consumption')
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Medium consumption')
    ax.legend()

def create_quality_summary_table(ax, symbol_short: str, analysis_results: Dict):
    """Create comprehensive quality summary table"""
    ax.axis('off')
    
    order_sizes = list(analysis_results['detailed_slippage'].keys())
    
    # Prepare table data
    table_data = [['Order Size', 'Buy Exec %', 'Sell Exec %', 'Avg Slippage %', 'P95 Slippage %', 'Quality Grade']]
    
    for size in order_sizes:
        buy_exec = analysis_results['detailed_slippage'][size]['buy']['execution_rate']
        sell_exec = analysis_results['detailed_slippage'][size]['sell']['execution_rate']
        
        buy_slippage = analysis_results['detailed_slippage'][size]['buy']['slippage_pct_stats']['mean']
        sell_slippage = analysis_results['detailed_slippage'][size]['sell']['slippage_pct_stats']['mean']
        avg_slippage = np.mean([buy_slippage, sell_slippage]) if not np.isnan(buy_slippage) and not np.isnan(sell_slippage) else np.nan
        
        buy_p95 = analysis_results['detailed_slippage'][size]['buy']['slippage_pct_stats']['p95']
        sell_p95 = analysis_results['detailed_slippage'][size]['sell']['slippage_pct_stats']['p95']
        p95_slippage = np.mean([buy_p95, sell_p95]) if not np.isnan(buy_p95) and not np.isnan(sell_p95) else np.nan
        
        # Calculate quality grade
        if min(buy_exec, sell_exec) >= 95 and avg_slippage <= 0.1:
            grade = "A"
        elif min(buy_exec, sell_exec) >= 90 and avg_slippage <= 0.2:
            grade = "B"
        elif min(buy_exec, sell_exec) >= 80 and avg_slippage <= 0.5:
            grade = "C"
        else:
            grade = "D"
        
        table_data.append([
            f'${size}',
            f'{buy_exec:.1f}%' if not np.isnan(buy_exec) else 'N/A',
            f'{sell_exec:.1f}%' if not np.isnan(sell_exec) else 'N/A',
            f'{avg_slippage:.3f}%' if not np.isnan(avg_slippage) else 'N/A',
            f'{p95_slippage:.3f}%' if not np.isnan(p95_slippage) else 'N/A',
            grade
        ])
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 2.0)
    
    # Style table header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code quality grades
    grade_colors = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e67e22', 'D': '#e74c3c'}
    for i in range(1, len(table_data)):
        grade = table_data[i][-1]
        if grade in grade_colors:
            table[(i, len(table_data[0])-1)].set_facecolor(grade_colors[grade])
            table[(i, len(table_data[0])-1)].set_text_props(weight='bold', color='white')
    
    ax.set_title(f'{symbol_short} - Trading Quality Summary', fontsize=14, fontweight='bold', pad=20)

def generate_enhanced_report(symbol: str, analysis_results: Dict):
    """Generate enhanced analysis report"""
    log.info(f"\n{'='*80}")
    log.info(f"ENHANCED LIQUIDITY ANALYSIS REPORT - {symbol}")
    log.info(f"{'='*80}")
    
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    # Overall statistics
    order_sizes = list(analysis_results['detailed_slippage'].keys())
    log.info(f"\n📊 ANALYSIS COVERAGE:")
    log.info(f"  Symbol: {symbol_short}")
    log.info(f"  Data points analyzed: {len(analysis_results['timestamps']):,}")
    log.info(f"  Order sizes tested: {order_sizes}")
    log.info(f"  Analysis period: {min(analysis_results['timestamps'])} to {max(analysis_results['timestamps'])}")
    
    # Detailed metrics table
    log.info(f"\n💹 DETAILED SLIPPAGE METRICS:")
    log.info(f"{'Size':<8} {'Side':<4} {'Exec %':<8} {'Mean':<8} {'P95':<8} {'P99':<8} {'Levels':<7}")
    log.info("-" * 60)
    
    for size in order_sizes:
        for side in ['buy', 'sell']:
            stats = analysis_results['detailed_slippage'][size][side]
            exec_rate = stats['execution_rate']
            slippage_mean = stats['slippage_pct_stats']['mean']
            slippage_p95 = stats['slippage_pct_stats']['p95']
            slippage_p99 = stats['slippage_pct_stats']['p99']
            levels_mean = stats['levels_used_stats']['mean']
            
            log.info(f"${size:<7} {side:<4} "
                    f"{exec_rate:>7.1f} "
                    f"{slippage_mean:>7.3f} "
                    f"{slippage_p95:>7.3f} "
                    f"{slippage_p99:>7.3f} "
                    f"{levels_mean:>6.2f}")
    
    # Trading recommendations
    log.info(f"\n📋 ENHANCED TRADING RECOMMENDATIONS:")
    
    # Find optimal order sizes
    viable_sizes = []
    for size in order_sizes:
        buy_exec = analysis_results['detailed_slippage'][size]['buy']['execution_rate']
        sell_exec = analysis_results['detailed_slippage'][size]['sell']['execution_rate']
        avg_slippage = np.mean([
            analysis_results['detailed_slippage'][size]['buy']['slippage_pct_stats']['mean'],
            analysis_results['detailed_slippage'][size]['sell']['slippage_pct_stats']['mean']
        ])
        if min(buy_exec, sell_exec) >= 90 and avg_slippage <= 0.5:
            viable_sizes.append((size, min(buy_exec, sell_exec), avg_slippage))

    if viable_sizes:
        optimal_size = max(viable_sizes, key=lambda x: x[0])  # Largest viable size
        log.info(f"  ✅ Optimal order size: ${optimal_size[0]} (Exec: {optimal_size[1]:.1f}%, Slippage: {optimal_size[2]:.3f}%)")
        log.info(f"  ✅ Viable sizes: {[f'${s[0]}' for s in viable_sizes]}")
    else:
        log.warning(f"  ⚠️ No order sizes meet quality criteria (90% exec, <0.5% slippage)")

    # Risk warnings
    high_slippage_sizes = []
    for size in order_sizes:
        p95_slippage = np.mean([
            analysis_results['detailed_slippage'][size]['buy']['slippage_pct_stats']['p95'],
            analysis_results['detailed_slippage'][size]['sell']['slippage_pct_stats']['p95']
        ])
        if p95_slippage > 1.0:  # P95 slippage > 1%
            high_slippage_sizes.append(size)

    if high_slippage_sizes:
        log.warning(f"  ⚠️ High slippage risk for sizes: {[f'${s}' for s in high_slippage_sizes]}")

    # Market structure insights
    log.info(f"\n🏗️ MARKET STRUCTURE INSIGHTS:")

    # Calculate average levels used across all sizes
    total_levels_data = []
    for size in order_sizes:
        for side in ['buy', 'sell']:
            levels_data = analysis_results['detailed_slippage'][size][side]['raw_data']['levels_used']
            total_levels_data.extend(levels_data)

    if total_levels_data:
        avg_levels = np.mean(total_levels_data)
        log.info(f"  Average orderbook levels consumed: {avg_levels:.2f}")

        if avg_levels < 1.5:
            log.info(f"  ✅ Excellent depth - most orders fill in top level")
        elif avg_levels < 2.5:
            log.info(f"  ✅ Good depth - orders typically use 2-3 levels")
        else:
            log.warning(f"  ⚠️ Fragmented liquidity - orders spread across many levels")

    # Asymmetry analysis
    log.info(f"\n⚖️ BUY/SELL ASYMMETRY ANALYSIS:")
    for size in [500, 1000, 2000]:  # Key sizes
        if size in order_sizes:
            buy_slippage = analysis_results['detailed_slippage'][size]['buy']['slippage_pct_stats']['mean']
            sell_slippage = analysis_results['detailed_slippage'][size]['sell']['slippage_pct_stats']['mean']

            if not (np.isnan(buy_slippage) or np.isnan(sell_slippage)):
                asymmetry = abs(buy_slippage - sell_slippage) / max(buy_slippage, sell_slippage) * 100
                bias = "buy" if buy_slippage > sell_slippage else "sell"

                log.info(f"  ${size}: {asymmetry:.1f}% asymmetry (higher {bias} slippage)")

                if asymmetry > 20:
                    log.warning(f"    ⚠️ Significant bias toward {bias} side")

def main():
    """Main enhanced analysis function - UPDATED WITH DATABASE SYMBOL SOURCING"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced liquidity analysis with comprehensive slippage metrics")
    parser.add_argument("--symbol", type=str, help="Specific symbol to analyze")
    parser.add_argument("--max-records", type=int, default=100000, help="Maximum records to analyze (default: 100k)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    log.info("Starting enhanced liquidity analysis...")
    
    # UPDATED: Get symbols from database (same logic as ingest_data.py)
    if args.symbol:
        symbols = [args.symbol]
        log.info(f"Analyzing specific symbol: {args.symbol}")
    else:
        log.info("🔍 Getting symbols from database...")
        
        # Try to get active symbols from database first
        symbols = settings.get_active_symbols_from_db()
        
        if symbols:
            log.info(f"✅ Found {len(symbols)} active symbols in database")
        else:
            # Fallback: get all symbols from database
            log.info("⚠️ No active symbols found, trying all symbols in database...")
            symbols = settings.get_symbols_from_db()
            
            if symbols:
                log.info(f"✅ Found {len(symbols)} total symbols in database")
            else:
                # Final fallback: use YAML (for cases where DB is not populated yet)
                log.warning("⚠️ No symbols found in database, falling back to YAML configuration...")
                try:
                    active_pairs = settings.get_active_pairs()
                    symbols = list(set([pair.symbol1 for pair in active_pairs] + [pair.symbol2 for pair in active_pairs]))
                    log.info(f"✅ Found {len(symbols)} symbols from YAML configuration")
                except Exception as yaml_error:
                    log.error(f"Failed to load symbols from YAML: {yaml_error}")
                    # Ultimate fallback
                    symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
                    log.warning(f"Using default symbols: {symbols}")
    
    # Log data source
    if args.symbol:
        log.info(f"📋 Data source: Manual symbol specification")
    elif settings.get_active_symbols_from_db():
        log.info(f"📋 Data source: Database (active symbols from active pairs)")
    elif settings.get_symbols_from_db():
        log.info(f"📋 Data source: Database (all available symbols)")
    else:
        log.info(f"📋 Data source: YAML configuration (fallback)")
    
    if not symbols:
        log.error("No symbols to analyze")
        return False
    
    log.info(f"Analyzing enhanced liquidity for {len(symbols)} symbols")
    
    # Display symbol names nicely
    symbol_names = []
    for s in symbols:
        if '_' in s:
            # Extract meaningful part (e.g., MEXCFTS_PERP_GIGA_USDT -> GIGA)
            parts = s.split('_')
            if len(parts) >= 3:
                symbol_names.append(parts[-2])  # Get GIGA from MEXCFTS_PERP_GIGA_USDT
            else:
                symbol_names.append(s)
        else:
            symbol_names.append(s)
    
    log.info(f"📊 Processing symbols: {symbol_names}")
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    try:
        for symbol in symbols:
            log.info(f"\n{'='*80}")
            log.info(f"ENHANCED ANALYSIS: {symbol}")
            log.info(f"{'='*80}")
            
            # Load ALL available data
            orderbook_df = load_all_orderbook_data(symbol, args.max_records)
            
            if len(orderbook_df) == 0:
                log.warning(f"No orderbook data for {symbol}")
                continue
            
            # Enhanced slippage analysis
            analysis_results = analyze_comprehensive_slippage(symbol, orderbook_df)
            
            # Generate enhanced visualizations
            if not args.no_plots:
                create_enhanced_visualization(symbol, analysis_results)
            
            # Generate enhanced report
            generate_enhanced_report(symbol, analysis_results)
        
        log.info(f"\n🎉 Enhanced liquidity analysis completed!")
        log.info(f"Key improvements:")
        log.info(f"  ✅ Analyzed ALL available data (not just 7 days)")
        log.info(f"  ✅ Box plots show slippage distributions")
        log.info(f"  ✅ Bidirectional analysis (buy vs sell)")
        log.info(f"  ✅ Extended order sizes (100-10000 USD)")
        log.info(f"  ✅ Comprehensive quality grading")
        log.info(f"  ✅ Market structure insights")
        log.info(f"  ✅ Uses database for symbol sourcing")
        log.info(f"Check plots/ directory for enhanced visualizations")
        
        return True
        
    except Exception as e:
        log.error(f"Enhanced liquidity analysis failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)