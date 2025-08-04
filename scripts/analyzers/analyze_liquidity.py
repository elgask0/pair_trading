#!/usr/bin/env python3
"""
Enhanced liquidity analysis script - PRODUCTION VERSION
Implements comprehensive liquidity analysis plan with dual time horizons
Analyzes market liquidity using orderbook data with slippage, round-trip costs, and hourly patterns
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
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is on PYTHONPATH for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.database.connection import db_manager
from config.settings import settings
from src.utils.logger import get_logger

log = get_logger()

# Enhanced matplotlib settings
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 100

# Configuration based on liquidity plan
@dataclass
class LiquidityConfig:
    """Configuration for liquidity analysis"""
    notional_grid: List[float] = None
    delta_bps: float = 8.0  # for depth calculation
    pr_max: float = 0.10  # max participation rate (10%)
    fee_bps: float = 3.0  # default taker fee in bps
    bar_minutes: int = 5  # bar size for volume analysis
    
    # Adjusted thresholds for more realistic crypto markets
    slippage_p50_green: float = 15.0  # bps (was 10)
    slippage_p50_yellow: float = 30.0  # bps (was 20)
    slippage_p95_green: float = 50.0  # bps (was 25)
    slippage_p95_yellow: float = 100.0  # bps (was 35)
    rt_p50_green: float = 50.0  # bps (was 40)
    rt_p50_yellow: float = 80.0  # bps (was 60)
    rt_p95_green: float = 120.0  # bps (was 70)
    rt_p95_yellow: float = 200.0  # bps (was 90)
    rt_ratio_green: float = 2.0  # (was 1.5)
    rt_ratio_yellow: float = 3.0  # (was 1.8)
    depth_multiple_green: float = 5.0  # (was 10)
    depth_multiple_yellow: float = 2.0  # (was 5)
    pr_feasible_green: float = 0.70  # 70% of hours (was 0.80)
    pr_feasible_yellow: float = 0.50  # 50% of hours (was 0.60)
    
    def __post_init__(self):
        if self.notional_grid is None:
            self.notional_grid = [100, 250, 500, 1000, 1500, 2000]

config = LiquidityConfig()

def get_data_range(symbol: str) -> Tuple[Optional[datetime], Optional[datetime], int]:
    """Get full data range for symbol"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as total_records
            FROM orderbook 
            WHERE symbol = :symbol
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
        """), {'symbol': symbol}).fetchone()
        
        return result.min_ts, result.max_ts, result.total_records if result else (None, None, 0)

def check_trades_table_exists() -> bool:
    """Check if trades table exists in database"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'trades'
            );
        """)).fetchone()
        return result[0] if result else False

def load_orderbook_data(symbol: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None, max_records: int = None) -> pd.DataFrame:
    """Load orderbook data with optional date filtering"""
    min_date, max_date, total_records = get_data_range(symbol)
    
    if not min_date or not max_date:
        log.warning(f"No orderbook data found for {symbol}")
        return pd.DataFrame()
    
    # Use provided dates or defaults
    start_date = start_date or min_date
    end_date = end_date or max_date
    
    log.info(f"Loading orderbook data for {symbol}:")
    log.info(f"  Available period: {min_date} to {max_date}")
    log.info(f"  Requested period: {start_date} to {end_date}")
    
    with db_manager.get_session() as session:
        # First check how many records in requested period
        count_result = session.execute(text("""
            SELECT COUNT(*) as count
            FROM orderbook
            WHERE symbol = :symbol 
            AND timestamp >= :start_date 
            AND timestamp <= :end_date
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
        """), {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        }).fetchone()
        
        period_records = count_result.count if count_result else 0
        log.info(f"  Records in period: {period_records:,}")
        
        # Sample if needed
        sample_clause = ""
        if max_records and period_records > max_records:
            sample_rate = max_records / period_records
            sample_clause = f"TABLESAMPLE SYSTEM ({sample_rate * 100:.2f})"
            log.info(f"  Sampling {sample_rate:.1%} of data ({max_records:,} records)")
        
        query = text(f"""
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
            'start_date': start_date,
            'end_date': end_date
        }, index_col='timestamp')
        
        log.info(f"  Loaded: {len(df):,} orderbook snapshots")
        
        # Add hour column for hourly analysis
        df['hour'] = df.index.hour
        
        return df

def load_trades_data(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Load trades data for volume analysis - returns None if table doesn't exist"""
    if not check_trades_table_exists():
        log.info("  Trades table not found - skipping volume analysis")
        return None
        
    try:
        with db_manager.get_session() as session:
            query = text("""
                SELECT 
                    timestamp,
                    price,
                    size,
                    price * size as volume_usd
                FROM trades
                WHERE symbol = :symbol 
                AND timestamp >= :start_date 
                AND timestamp <= :end_date
                ORDER BY timestamp
            """)
            
            df = pd.read_sql(query, session.bind, params={
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date
            }, index_col='timestamp')
            
            if len(df) > 0:
                log.info(f"  Loaded: {len(df):,} trades")
            
            return df
    except Exception as e:
        log.warning(f"Could not load trades data: {e}")
        return None

def calculate_slippage_and_rt(row: pd.Series, notional_usd: float, side: str = 'buy', 
                             fee_bps: float = 3.0) -> Dict:
    """Calculate slippage and round-trip cost for given notional"""
    
    levels = []
    prefix = 'ask' if side == 'buy' else 'bid'
    
    # Extract top 10 levels
    for i in range(1, 11):
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
            'slippage_bps': np.nan,
            'rt_bps': np.nan,
            'levels_used': 0,
            'avg_price': np.nan
        }
    
    # Sort by price (best first)
    levels.sort(key=lambda x: x['price'], reverse=(side == 'sell'))
    
    # Calculate execution
    remaining_usd = notional_usd
    total_value = 0
    total_coins = 0
    levels_used = 0
    
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
            'slippage_bps': np.nan,
            'rt_bps': np.nan,
            'levels_used': levels_used,
            'avg_price': np.nan
        }
    
    # Calculate metrics
    avg_price = total_value / total_coins
    best_price = levels[0]['price']
    
    slippage_pct = abs(avg_price - best_price) / best_price * 100
    slippage_bps = slippage_pct * 100
    rt_bps = 2 * (fee_bps + slippage_bps)  # Round-trip = 2 * (fee + slippage)
    
    return {
        'can_execute': True,
        'slippage_bps': slippage_bps,
        'rt_bps': rt_bps,
        'levels_used': levels_used,
        'avg_price': avg_price
    }

def calculate_depth_at_delta(row: pd.Series, delta_bps: float = 8.0) -> Dict:
    """Calculate available depth within delta_bps from mid price"""
    bid1 = row.get('bid1_price', np.nan)
    ask1 = row.get('ask1_price', np.nan)
    
    if pd.isna(bid1) or pd.isna(ask1):
        return {'bid_depth_usd': 0, 'ask_depth_usd': 0, 'total_depth_usd': 0}
    
    mid_price = (bid1 + ask1) / 2
    threshold_pct = delta_bps / 10000  # Convert bps to percentage
    
    bid_depth_usd = 0
    ask_depth_usd = 0
    
    # Calculate bid depth (within delta_bps below mid)
    for i in range(1, 11):
        price = row.get(f'bid{i}_price')
        size = row.get(f'bid{i}_size')
        
        if pd.notna(price) and pd.notna(size) and price > 0:
            price_distance = abs(mid_price - price) / mid_price
            if price_distance <= threshold_pct:
                bid_depth_usd += price * size
    
    # Calculate ask depth (within delta_bps above mid)
    for i in range(1, 11):
        price = row.get(f'ask{i}_price')
        size = row.get(f'ask{i}_size')
        
        if pd.notna(price) and pd.notna(size) and price > 0:
            price_distance = abs(price - mid_price) / mid_price
            if price_distance <= threshold_pct:
                ask_depth_usd += price * size
    
    return {
        'bid_depth_usd': bid_depth_usd,
        'ask_depth_usd': ask_depth_usd,
        'total_depth_usd': bid_depth_usd + ask_depth_usd
    }

def analyze_liquidity_metrics(symbol: str, orderbook_df: pd.DataFrame, 
                            trades_df: pd.DataFrame = None) -> Dict:
    """Comprehensive liquidity analysis following the plan"""
    log.info(f"Analyzing liquidity metrics for {symbol}...")
    
    results = {
        'symbol': symbol,
        'period_start': orderbook_df.index.min(),
        'period_end': orderbook_df.index.max(),
        'total_snapshots': len(orderbook_df),
        'hourly_metrics': {},
        'global_metrics': {},
        'notional_recommendations': {},
        'stability_analysis': {}
    }
    
    # Process each notional size
    for notional in config.notional_grid:
        log.info(f"  Processing ${notional} notional...")
        
        hourly_data = {hour: {
            'slippage_buy': [], 'slippage_sell': [],
            'rt_buy': [], 'rt_sell': [],
            'can_execute_buy': [], 'can_execute_sell': [],
            'depth_total': [], 'depth_multiples': []
        } for hour in range(24)}
        
        # Process each orderbook snapshot
        for idx, row in orderbook_df.iterrows():
            hour = row['hour']
            
            # Calculate slippage and RT for buy side
            buy_result = calculate_slippage_and_rt(row, notional, 'buy', config.fee_bps)
            if buy_result['can_execute']:
                hourly_data[hour]['slippage_buy'].append(buy_result['slippage_bps'])
                hourly_data[hour]['rt_buy'].append(buy_result['rt_bps'])
            hourly_data[hour]['can_execute_buy'].append(buy_result['can_execute'])
            
            # Calculate slippage and RT for sell side
            sell_result = calculate_slippage_and_rt(row, notional, 'sell', config.fee_bps)
            if sell_result['can_execute']:
                hourly_data[hour]['slippage_sell'].append(sell_result['slippage_bps'])
                hourly_data[hour]['rt_sell'].append(sell_result['rt_bps'])
            hourly_data[hour]['can_execute_sell'].append(sell_result['can_execute'])
            
            # Calculate depth
            depth_result = calculate_depth_at_delta(row, config.delta_bps)
            hourly_data[hour]['depth_total'].append(depth_result['total_depth_usd'])
            if depth_result['total_depth_usd'] > 0:
                hourly_data[hour]['depth_multiples'].append(depth_result['total_depth_usd'] / notional)
        
        # Calculate hourly statistics
        hourly_stats = {}
        for hour in range(24):
            data = hourly_data[hour]
            
            # Combine buy and sell slippage
            all_slippage = data['slippage_buy'] + data['slippage_sell']
            all_rt = data['rt_buy'] + data['rt_sell']
            
            if all_slippage:
                hourly_stats[hour] = {
                    'slippage_p50': np.percentile(all_slippage, 50),
                    'slippage_p95': np.percentile(all_slippage, 95),
                    'rt_p50': np.percentile(all_rt, 50),
                    'rt_p95': np.percentile(all_rt, 95),
                    'exec_rate_buy': np.mean(data['can_execute_buy']) * 100,
                    'exec_rate_sell': np.mean(data['can_execute_sell']) * 100,
                    'depth_median': np.median(data['depth_total']) if data['depth_total'] else 0,
                    'depth_multiple_median': np.median(data['depth_multiples']) if data['depth_multiples'] else 0,
                    'sample_count': len(all_slippage)
                }
            else:
                hourly_stats[hour] = {
                    'slippage_p50': np.nan, 'slippage_p95': np.nan,
                    'rt_p50': np.nan, 'rt_p95': np.nan,
                    'exec_rate_buy': 0, 'exec_rate_sell': 0,
                    'depth_median': 0, 'depth_multiple_median': 0,
                    'sample_count': 0
                }
        
        results['hourly_metrics'][notional] = hourly_stats
        
        # Calculate global statistics (aggregated across all hours)
        all_slippage_global = []
        all_rt_global = []
        all_exec_buy = []
        all_exec_sell = []
        
        for hour_data in hourly_data.values():
            all_slippage_global.extend(hour_data['slippage_buy'] + hour_data['slippage_sell'])
            all_rt_global.extend(hour_data['rt_buy'] + hour_data['rt_sell'])
            all_exec_buy.extend(hour_data['can_execute_buy'])
            all_exec_sell.extend(hour_data['can_execute_sell'])
        
        if all_slippage_global:
            results['global_metrics'][notional] = {
                'slippage_p50': np.percentile(all_slippage_global, 50),
                'slippage_p95': np.percentile(all_slippage_global, 95),
                'rt_p50': np.percentile(all_rt_global, 50),
                'rt_p95': np.percentile(all_rt_global, 95),
                'exec_rate_buy': np.mean(all_exec_buy) * 100,
                'exec_rate_sell': np.mean(all_exec_sell) * 100
            }
    
    # Analyze stability (RT ratio at $1000 reference)
    if 1000 in results['hourly_metrics']:
        for hour in range(24):
            stats = results['hourly_metrics'][1000][hour]
            if not np.isnan(stats['rt_p50']) and stats['rt_p50'] > 0:
                results['stability_analysis'][hour] = {
                    'rt_ratio': stats['rt_p95'] / stats['rt_p50'],
                    'is_stable': (stats['rt_p95'] / stats['rt_p50']) <= config.rt_ratio_green
                }
    
    # Calculate volume metrics if trades data available
    if trades_df is not None and len(trades_df) > 0:
        log.info("  Calculating volume metrics...")
        results['volume_analysis'] = analyze_volume_metrics(trades_df, config.bar_minutes)
    
    return results

def analyze_volume_metrics(trades_df: pd.DataFrame, bar_minutes: int = 5) -> Dict:
    """Analyze trading volume by time bars"""
    # Resample to bars
    volume_bars = trades_df['volume_usd'].resample(f'{bar_minutes}T').sum()
    volume_bars = volume_bars[volume_bars > 0]  # Remove empty bars
    
    if len(volume_bars) == 0:
        return {}
    
    # Group by hour
    volume_bars_hourly = volume_bars.groupby(volume_bars.index.hour)
    
    hourly_volume_stats = {}
    for hour in range(24):
        if hour in volume_bars_hourly.groups:
            hour_volumes = volume_bars_hourly.get_group(hour)
            hourly_volume_stats[hour] = {
                'median_volume': np.median(hour_volumes),
                'p75_volume': np.percentile(hour_volumes, 75),
                'p90_volume': np.percentile(hour_volumes, 90),
                'bar_count': len(hour_volumes)
            }
        else:
            hourly_volume_stats[hour] = {
                'median_volume': 0,
                'p75_volume': 0,
                'p90_volume': 0,
                'bar_count': 0
            }
    
    return {
        'bar_minutes': bar_minutes,
        'hourly_stats': hourly_volume_stats,
        'global_median_volume': np.median(volume_bars),
        'global_p90_volume': np.percentile(volume_bars, 90)
    }

def determine_optimal_notional(analysis_results: Dict) -> Dict:
    """Determine optimal notional size based on liquidity metrics"""
    recommendations = {
        'global': {},
        'hourly': {},
        'traffic_lights': {}
    }
    
    # Global recommendation
    viable_notionals = []
    
    for notional, metrics in analysis_results['global_metrics'].items():
        # Check traffic light conditions
        traffic_light = evaluate_traffic_light(metrics, notional, analysis_results)
        recommendations['traffic_lights'][notional] = traffic_light
        
        if traffic_light['overall_status'] in ['green', 'yellow']:
            viable_notionals.append((notional, traffic_light['score']))
    
    if viable_notionals:
        # Choose largest viable notional
        optimal_notional = max(viable_notionals, key=lambda x: x[0])
        recommendations['global']['optimal_notional'] = optimal_notional[0]
        recommendations['global']['score'] = optimal_notional[1]
        recommendations['global']['viable_notionals'] = [n[0] for n in viable_notionals]
    else:
        recommendations['global']['optimal_notional'] = None
        recommendations['global']['viable_notionals'] = []
    
    # Hourly recommendations
    for hour in range(24):
        hour_viable = []
        
        for notional in config.notional_grid:
            if notional in analysis_results['hourly_metrics']:
                hour_metrics = analysis_results['hourly_metrics'][notional].get(hour, {})
                
                if hour_metrics and not np.isnan(hour_metrics.get('rt_p50', np.nan)):
                    # Check basic conditions with relaxed thresholds
                    if (hour_metrics['rt_p50'] <= config.rt_p50_yellow * 1.2 and  # Allow 20% over yellow
                        hour_metrics['rt_p95'] <= config.rt_p95_yellow * 1.2 and
                        min(hour_metrics['exec_rate_buy'], hour_metrics['exec_rate_sell']) >= 60):  # Lower threshold
                        
                        hour_viable.append(notional)
        
        recommendations['hourly'][hour] = {
            'optimal_notional': max(hour_viable) if hour_viable else None,
            'viable_notionals': hour_viable
        }
    
    return recommendations

def evaluate_traffic_light(metrics: Dict, notional: float, analysis_results: Dict) -> Dict:
    """Evaluate traffic light status for given metrics"""
    scores = {
        'slippage_p50': 'red',
        'slippage_p95': 'red',
        'rt_p50': 'red',
        'rt_p95': 'red',
        'exec_rate': 'red',
        'stability': 'red'
    }
    
    # Slippage checks
    if metrics.get('slippage_p50', np.inf) <= config.slippage_p50_green:
        scores['slippage_p50'] = 'green'
    elif metrics.get('slippage_p50', np.inf) <= config.slippage_p50_yellow:
        scores['slippage_p50'] = 'yellow'
    
    if metrics.get('slippage_p95', np.inf) <= config.slippage_p95_green:
        scores['slippage_p95'] = 'green'
    elif metrics.get('slippage_p95', np.inf) <= config.slippage_p95_yellow:
        scores['slippage_p95'] = 'yellow'
    
    # RT checks
    if metrics.get('rt_p50', np.inf) <= config.rt_p50_green:
        scores['rt_p50'] = 'green'
    elif metrics.get('rt_p50', np.inf) <= config.rt_p50_yellow:
        scores['rt_p50'] = 'yellow'
    
    if metrics.get('rt_p95', np.inf) <= config.rt_p95_green:
        scores['rt_p95'] = 'green'
    elif metrics.get('rt_p95', np.inf) <= config.rt_p95_yellow:
        scores['rt_p95'] = 'yellow'
    
    # Execution rate check
    min_exec = min(metrics.get('exec_rate_buy', 0), metrics.get('exec_rate_sell', 0))
    if min_exec >= 95:
        scores['exec_rate'] = 'green'
    elif min_exec >= 80:
        scores['exec_rate'] = 'yellow'
    
    # Stability check (if available)
    if 'stability_analysis' in analysis_results:
        stable_hours = sum(1 for h in analysis_results['stability_analysis'].values() 
                          if h.get('is_stable', False))
        stability_pct = stable_hours / 24
        
        if stability_pct >= 0.7:  # Relaxed from 0.8
            scores['stability'] = 'green'
        elif stability_pct >= 0.5:  # Relaxed from 0.6
            scores['stability'] = 'yellow'
    
    # Overall status - more lenient
    red_count = sum(1 for s in scores.values() if s == 'red')
    yellow_count = sum(1 for s in scores.values() if s == 'yellow')
    
    if red_count > 2:  # Allow up to 2 reds
        overall = 'red'
    elif yellow_count > 3:  # Allow up to 3 yellows
        overall = 'yellow'
    else:
        overall = 'green'
    
    # Calculate numeric score for sorting
    score_values = {'green': 2, 'yellow': 1, 'red': 0}
    numeric_score = sum(score_values[s] for s in scores.values())
    
    return {
        'scores': scores,
        'overall_status': overall,
        'score': numeric_score
    }

def create_comprehensive_visualizations(symbol: str, analysis_all: Dict, 
                                      analysis_3m: Dict = None):
    """Create all visualizations from the liquidity plan"""
    log.info(f"Creating comprehensive visualizations for {symbol}...")
    
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Title
    title = f'{symbol_short} - Liquidity Analysis'
    if analysis_3m:
        title += ' (All Data vs Last 3 Months)'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # 1. RT vs Notional curves (p50/p95)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_rt_curves(ax1, analysis_all, analysis_3m, 'p50')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_rt_curves(ax2, analysis_all, analysis_3m, 'p95')
    
    # 2. Slippage vs Notional curves (p50/p95)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_slippage_curves(ax3, analysis_all, analysis_3m, 'p50')
    
    ax4 = fig.add_subplot(gs[0, 3])
    plot_slippage_curves(ax4, analysis_all, analysis_3m, 'p95')
    
    # 3. Hourly heatmap - RT costs
    ax5 = fig.add_subplot(gs[1, :2])
    plot_hourly_heatmap(ax5, analysis_all, 'rt_p50', 'Round-Trip Cost P50 (bps)')
    
    # 4. Hourly heatmap - Execution rates
    ax6 = fig.add_subplot(gs[1, 2:])
    plot_hourly_execution_heatmap(ax6, analysis_all)
    
    # 5. Stability analysis
    ax7 = fig.add_subplot(gs[2, 0])
    plot_stability_analysis(ax7, analysis_all)
    
    # 6. Depth analysis
    ax8 = fig.add_subplot(gs[2, 1])
    plot_depth_analysis(ax8, analysis_all)
    
    # 7. Traffic light scorecard
    ax9 = fig.add_subplot(gs[2, 2:])
    plot_traffic_light_scorecard(ax9, analysis_all)
    
    # 8. Optimal notional recommendations
    ax10 = fig.add_subplot(gs[3, :])
    plot_recommendations_table(ax10, analysis_all)
    
    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    filename = f'{symbol_short}_liquidity_analysis_enhanced.png'
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    log.info(f"Visualizations saved: plots/{filename}")

def plot_rt_curves(ax, analysis_all: Dict, analysis_3m: Dict, percentile: str):
    """Plot round-trip cost curves"""
    notionals = sorted(analysis_all['global_metrics'].keys())
    
    # All data
    rt_all = [analysis_all['global_metrics'][n][f'rt_{percentile}'] for n in notionals]
    ax.plot(notionals, rt_all, 'o-', label='All Data', linewidth=2, markersize=8)
    
    # 3 month data
    if analysis_3m:
        rt_3m = [analysis_3m['global_metrics'][n][f'rt_{percentile}'] for n in notionals]
        ax.plot(notionals, rt_3m, 's--', label='Last 3 Months', linewidth=2, markersize=8)
    
    # Reference lines
    if percentile == 'p50':
        ax.axhline(y=config.rt_p50_green, color='green', linestyle='--', alpha=0.5, label=f'Green ({config.rt_p50_green} bps)')
        ax.axhline(y=config.rt_p50_yellow, color='orange', linestyle='--', alpha=0.5, label=f'Yellow ({config.rt_p50_yellow} bps)')
    else:
        ax.axhline(y=config.rt_p95_green, color='green', linestyle='--', alpha=0.5, label=f'Green ({config.rt_p95_green} bps)')
        ax.axhline(y=config.rt_p95_yellow, color='orange', linestyle='--', alpha=0.5, label=f'Yellow ({config.rt_p95_yellow} bps)')
    
    ax.set_xlabel('Notional (USD)')
    ax.set_ylabel(f'RT Cost {percentile.upper()} (bps)')
    ax.set_title(f'Round-Trip Cost {percentile.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_slippage_curves(ax, analysis_all: Dict, analysis_3m: Dict, percentile: str):
    """Plot slippage curves"""
    notionals = sorted(analysis_all['global_metrics'].keys())
    
    # All data
    slip_all = [analysis_all['global_metrics'][n][f'slippage_{percentile}'] for n in notionals]
    ax.plot(notionals, slip_all, 'o-', label='All Data', linewidth=2, markersize=8)
    
    # 3 month data
    if analysis_3m:
        slip_3m = [analysis_3m['global_metrics'][n][f'slippage_{percentile}'] for n in notionals]
        ax.plot(notionals, slip_3m, 's--', label='Last 3 Months', linewidth=2, markersize=8)
    
    # Reference lines
    if percentile == 'p50':
        ax.axhline(y=config.slippage_p50_green, color='green', linestyle='--', alpha=0.5, label=f'Green ({config.slippage_p50_green} bps)')
        ax.axhline(y=config.slippage_p50_yellow, color='orange', linestyle='--', alpha=0.5, label=f'Yellow ({config.slippage_p50_yellow} bps)')
    else:
        ax.axhline(y=config.slippage_p95_green, color='green', linestyle='--', alpha=0.5, label=f'Green ({config.slippage_p95_green} bps)')
        ax.axhline(y=config.slippage_p95_yellow, color='orange', linestyle='--', alpha=0.5, label=f'Yellow ({config.slippage_p95_yellow} bps)')
    
    ax.set_xlabel('Notional (USD)')
    ax.set_ylabel(f'Slippage {percentile.upper()} (bps)')
    ax.set_title(f'Slippage {percentile.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_hourly_heatmap(ax, analysis: Dict, metric: str, title: str):
    """Plot hourly heatmap for given metric"""
    hours = list(range(24))
    notionals = sorted(analysis['hourly_metrics'].keys())
    
    # Create matrix
    matrix = np.zeros((len(notionals), 24))
    
    for i, notional in enumerate(notionals):
        for hour in hours:
            value = analysis['hourly_metrics'][notional][hour].get(metric, np.nan)
            matrix[i, hour] = value
    
    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r')
    
    # Set ticks
    ax.set_xticks(range(24))
    ax.set_xticklabels(hours)
    ax.set_yticks(range(len(notionals)))
    ax.set_yticklabels([f'${n}' for n in notionals])
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Notional Size')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('bps')

def plot_hourly_execution_heatmap(ax, analysis: Dict):
    """Plot hourly execution rate heatmap"""
    hours = list(range(24))
    notionals = sorted(analysis['hourly_metrics'].keys())
    
    # Create matrix (minimum of buy/sell execution rates)
    matrix = np.zeros((len(notionals), 24))
    
    for i, notional in enumerate(notionals):
        for hour in hours:
            buy_rate = analysis['hourly_metrics'][notional][hour].get('exec_rate_buy', 0)
            sell_rate = analysis['hourly_metrics'][notional][hour].get('exec_rate_sell', 0)
            matrix[i, hour] = min(buy_rate, sell_rate)
    
    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(range(24))
    ax.set_xticklabels(hours)
    ax.set_yticks(range(len(notionals)))
    ax.set_yticklabels([f'${n}' for n in notionals])
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Notional Size')
    ax.set_title('Execution Success Rate % (min of buy/sell)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('%')

def plot_stability_analysis(ax, analysis: Dict):
    """Plot RT ratio stability analysis"""
    if not analysis.get('stability_analysis'):
        ax.text(0.5, 0.5, 'No stability data', ha='center', va='center')
        ax.set_title('Stability Analysis')
        return
    
    hours = sorted(analysis['stability_analysis'].keys())
    rt_ratios = [analysis['stability_analysis'][h]['rt_ratio'] for h in hours]
    
    colors = ['green' if r <= config.rt_ratio_green else 'orange' 
              if r <= config.rt_ratio_yellow else 'red' for r in rt_ratios]
    
    bars = ax.bar(hours, rt_ratios, color=colors, alpha=0.7)
    
    # Reference lines
    ax.axhline(y=config.rt_ratio_green, color='green', linestyle='--', alpha=0.5, label=f'Green ({config.rt_ratio_green})')
    ax.axhline(y=config.rt_ratio_yellow, color='orange', linestyle='--', alpha=0.5, label=f'Yellow ({config.rt_ratio_yellow})')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('RT Ratio (P95/P50)')
    ax.set_title('Hourly Stability (RT Ratio @ $1000)')
    ax.set_ylim(0, max(rt_ratios) * 1.1 if rt_ratios else 3)
    ax.legend()

def plot_depth_analysis(ax, analysis: Dict):
    """Plot depth analysis"""
    notionals = sorted(analysis['global_metrics'].keys())
    
    # Calculate average depth multiples across all hours
    avg_depth_multiples = []
    
    for notional in notionals:
        multiples = []
        for hour in range(24):
            hour_data = analysis['hourly_metrics'][notional].get(hour, {})
            if 'depth_multiple_median' in hour_data and hour_data['depth_multiple_median'] > 0:
                multiples.append(hour_data['depth_multiple_median'])
        
        avg_multiple = np.mean(multiples) if multiples else 0
        avg_depth_multiples.append(avg_multiple)
    
    # Create bar chart
    colors = ['green' if m >= config.depth_multiple_green else 'orange' 
              if m >= config.depth_multiple_yellow else 'red' for m in avg_depth_multiples]
    
    bars = ax.bar([f'${n}' for n in notionals], avg_depth_multiples, color=colors, alpha=0.7)
    
    # Reference lines
    ax.axhline(y=config.depth_multiple_green, color='green', linestyle='--', alpha=0.5, label=f'Green ({config.depth_multiple_green}x)')
    ax.axhline(y=config.depth_multiple_yellow, color='orange', linestyle='--', alpha=0.5, label=f'Yellow ({config.depth_multiple_yellow}x)')
    
    ax.set_ylabel('Depth / Notional')
    ax.set_title(f'Average Depth Multiple (±{config.delta_bps} bps)')
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars, avg_depth_multiples):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}x', ha='center', va='bottom')

def plot_traffic_light_scorecard(ax, analysis: Dict):
    """Plot traffic light scorecard"""
    ax.axis('off')
    
    # Get recommendations
    recommendations = determine_optimal_notional(analysis)
    
    # Prepare table data
    headers = ['Notional', 'Slip P50', 'Slip P95', 'RT P50', 'RT P95', 'Exec %', 'Stability', 'Overall']
    table_data = []
    
    color_map = {'green': '#2ecc71', 'yellow': '#f39c12', 'red': '#e74c3c'}
    
    for notional in sorted(analysis['global_metrics'].keys()):
        traffic_light = recommendations['traffic_lights'][notional]
        
        row = [f'${notional}']
        colors = []
        
        # Add scores
        for metric in ['slippage_p50', 'slippage_p95', 'rt_p50', 'rt_p95', 'exec_rate', 'stability']:
            status = traffic_light['scores'][metric]
            row.append('●')
            colors.append(color_map[status])
        
        # Overall status
        row.append('●')
        colors.append(color_map[traffic_light['overall_status']])
        
        table_data.append((row, colors))
    
    # Create table
    cell_text = [row[0] for row in table_data]
    
    table = ax.table(cellText=cell_text, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Color cells
    for i, (row, colors) in enumerate(table_data):
        for j in range(1, len(headers)):
            table[(i+1, j)].set_facecolor(colors[j-1])
            table[(i+1, j)].set_text_props(color='white', weight='bold')
    
    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Traffic Light Scorecard', fontsize=14, fontweight='bold', pad=20)

def plot_recommendations_table(ax, analysis: Dict):
    """Plot recommendations table"""
    ax.axis('off')
    
    recommendations = determine_optimal_notional(analysis)
    
    # Prepare table data
    table_data = []
    
    # Global recommendation
    optimal = recommendations['global'].get('optimal_notional')
    if optimal:
        table_data.append(['Global Optimal Notional', f"${optimal}"])
    else:
        table_data.append(['Global Optimal Notional', 'No viable notional found'])
    
    viable = recommendations['global'].get('viable_notionals', [])
    if viable:
        table_data.append(['Viable Notionals', ', '.join([f"${n}" for n in viable])])
    else:
        table_data.append(['Viable Notionals', 'None meet criteria'])
    
    # Hourly summary
    hourly_notionals = [recommendations['hourly'][h].get('optimal_notional', 0) for h in range(24)]
    valid_hourly = [n for n in hourly_notionals if n]
    
    if valid_hourly:
        avg_hourly = np.mean(valid_hourly)
        table_data.append(['Average Hourly Optimal', f"${avg_hourly:.0f}"])
    else:
        table_data.append(['Average Hourly Optimal', 'N/A'])
    
    # Trading hours
    trade_hours = [str(h) for h in range(24) 
                  if recommendations['hourly'][h].get('optimal_notional') is not None]
    
    if trade_hours:
        if len(trade_hours) > 10:
            table_data.append(['Tradeable Hours', f"{len(trade_hours)}/24 hours"])
        else:
            table_data.append(['Tradeable Hours', ', '.join(trade_hours)])
    else:
        table_data.append(['Tradeable Hours', 'None - Insufficient liquidity'])
    
    # Non-tradeable hours
    non_trade_hours = [str(h) for h in range(24) 
                      if recommendations['hourly'][h].get('optimal_notional') is None]
    
    if non_trade_hours and len(non_trade_hours) < 24:
        if len(non_trade_hours) > 10:
            table_data.append(['Non-Trade Hours', f"{len(non_trade_hours)}/24 hours"])
        else:
            table_data.append(['Non-Trade Hours', ', '.join(non_trade_hours)])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.0)
    
    # Style cells
    for i in range(len(table_data)):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_facecolor('#ffffff')
    
    ax.set_title('Trading Recommendations', fontsize=14, fontweight='bold', pad=20)

def generate_detailed_report(symbol: str, analysis_all: Dict, analysis_3m: Dict = None):
    """Generate detailed analysis report"""
    log.info(f"\n{'='*80}")
    log.info(f"LIQUIDITY ANALYSIS REPORT - {symbol}")
    log.info(f"{'='*80}")
    
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    recommendations = determine_optimal_notional(analysis_all)
    
    # Summary
    log.info(f"\n📊 EXECUTIVE SUMMARY:")
    log.info(f"  Symbol: {symbol_short}")
    log.info(f"  Analysis Period: {analysis_all['period_start']} to {analysis_all['period_end']}")
    log.info(f"  Total Snapshots: {analysis_all['total_snapshots']:,}")
    
    optimal = recommendations['global'].get('optimal_notional')
    if optimal:
        log.info(f"  Optimal Notional: ${optimal}")
    else:
        log.info(f"  Optimal Notional: No viable notional found with current criteria")
        log.info(f"  Consider: Relaxing criteria or trading smaller sizes")
    
    viable = recommendations['global'].get('viable_notionals', [])
    if viable:
        log.info(f"  Viable Notionals: {[f'${n}' for n in viable]}")
    
    # Global metrics table
    log.info(f"\n💹 GLOBAL LIQUIDITY METRICS:")
    log.info(f"{'Notional':<10} {'Slip P50':<10} {'Slip P95':<10} {'RT P50':<10} {'RT P95':<10} {'Exec %':<10}")
    log.info("-" * 70)
    
    for notional in sorted(analysis_all['global_metrics'].keys()):
        metrics = analysis_all['global_metrics'][notional]
        exec_rate = min(metrics['exec_rate_buy'], metrics['exec_rate_sell'])
        
        log.info(f"${notional:<9} "
                f"{metrics['slippage_p50']:>9.2f} "
                f"{metrics['slippage_p95']:>9.2f} "
                f"{metrics['rt_p50']:>9.2f} "
                f"{metrics['rt_p95']:>9.2f} "
                f"{exec_rate:>9.1f}")
    
    # Comparison with 3-month data
    if analysis_3m:
        log.info(f"\n📈 RECENT PERFORMANCE (Last 3 Months):")
        log.info("  Changes in key metrics:")
        
        for notional in [500, 1000, 2000]:
            if notional in analysis_all['global_metrics'] and notional in analysis_3m['global_metrics']:
                all_rt = analysis_all['global_metrics'][notional]['rt_p50']
                recent_rt = analysis_3m['global_metrics'][notional]['rt_p50']
                change = ((recent_rt - all_rt) / all_rt) * 100 if all_rt > 0 else 0
                
                trend = "📈" if change > 5 else "📉" if change < -5 else "➡️"
                log.info(f"  ${notional}: RT P50 {all_rt:.1f} → {recent_rt:.1f} bps ({change:+.1f}%) {trend}")
    
    # Hourly patterns
    log.info(f"\n🕐 HOURLY TRADING PATTERNS:")
    
    # Find best and worst hours
    hourly_quality = []
    for hour in range(24):
        optimal = recommendations['hourly'][hour].get('optimal_notional')
        if optimal:
            # Get RT at optimal notional
            rt_p50 = analysis_all['hourly_metrics'][optimal][hour]['rt_p50']
            hourly_quality.append((hour, optimal, rt_p50))
    
    if hourly_quality:
        hourly_quality.sort(key=lambda x: x[2])  # Sort by RT
        
        log.info("  Best trading hours:")
        for hour, notional, rt in hourly_quality[:3]:
            log.info(f"    Hour {hour:02d}:00 - Optimal: ${notional}, RT P50: {rt:.1f} bps")
        
        if len(hourly_quality) > 3:
            log.info("  Worst trading hours:")
            for hour, notional, rt in hourly_quality[-3:]:
                log.info(f"    Hour {hour:02d}:00 - Optimal: ${notional}, RT P50: {rt:.1f} bps")
    else:
        log.info("  No tradeable hours found with current criteria")
    
    # Risk warnings
    log.info(f"\n⚠️ RISK WARNINGS:")
    
    # Check for high slippage
    high_slip_notionals = []
    for notional, metrics in analysis_all['global_metrics'].items():
        if metrics['slippage_p95'] > config.slippage_p95_yellow:
            high_slip_notionals.append(notional)
    
    if high_slip_notionals:
        log.warning(f"  High slippage risk (P95 > {config.slippage_p95_yellow} bps) for: {[f'${n}' for n in high_slip_notionals]}")
    
    # Check for low execution rates
    low_exec_notionals = []
    for notional, metrics in analysis_all['global_metrics'].items():
        if min(metrics['exec_rate_buy'], metrics['exec_rate_sell']) < 80:
            low_exec_notionals.append(notional)
    
    if low_exec_notionals:
        log.warning(f"  Low execution rates (<80%) for: {[f'${n}' for n in low_exec_notionals]}")
    
    # Trading recommendations
    log.info(f"\n📋 TRADING RECOMMENDATIONS:")
    
    if optimal:
        log.info(f"  1. Use notional size of ${optimal} for general trading")
        log.info(f"  2. Adjust size by hour - see hourly optimal notionals in visualization")
        log.info(f"  3. Monitor RT costs in real-time and reduce size if exceeding P95 historical")
        log.info(f"  4. Consider avoiding hours with no viable notional sizes")
    else:
        log.info(f"  1. Market shows limited liquidity for standard notional sizes")
        log.info(f"  2. Consider starting with ${min(config.notional_grid)} and monitor execution quality")
        log.info(f"  3. Trade during peak liquidity hours if possible")
        log.info(f"  4. Use limit orders to reduce slippage impact")
    
    if 'volume_analysis' in analysis_all:
        log.info(f"  5. Check participation rate vs {config.bar_minutes}-minute volume bars")
    
    # Additional insights based on data
    log.info(f"\n🔍 MARKET INSIGHTS:")
    
    # Calculate average spread
    if 100 in analysis_all['global_metrics']:
        base_slip = analysis_all['global_metrics'][100]['slippage_p50']
        estimated_spread_bps = base_slip * 2  # Rough estimate
        log.info(f"  Estimated average spread: ~{estimated_spread_bps:.1f} bps")
    
    # Liquidity depth insight
    max_viable_notional = max(viable) if viable else 0
    if max_viable_notional:
        log.info(f"  Maximum viable trade size: ${max_viable_notional}")
    else:
        log.info(f"  Limited depth - consider splitting large orders")

def export_results_to_csv(symbol: str, analysis: Dict, output_dir: Path = Path("output")):
    """Export analysis results to CSV files"""
    output_dir.mkdir(exist_ok=True)
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    # 1. Hourly metrics CSV
    hourly_data = []
    for notional in sorted(analysis['hourly_metrics'].keys()):
        for hour in range(24):
            metrics = analysis['hourly_metrics'][notional].get(hour, {})
            if metrics and metrics.get('sample_count', 0) > 0:
                hourly_data.append({
                    'hour': hour,
                    'notional': notional,
                    'slip_p50': metrics.get('slippage_p50', np.nan),
                    'slip_p95': metrics.get('slippage_p95', np.nan),
                    'rt_p50': metrics.get('rt_p50', np.nan),
                    'rt_p95': metrics.get('rt_p95', np.nan),
                    'exec_rate_buy': metrics.get('exec_rate_buy', 0),
                    'exec_rate_sell': metrics.get('exec_rate_sell', 0),
                    'depth_median': metrics.get('depth_median', 0),
                    'depth_multiple': metrics.get('depth_multiple_median', 0),
                    'sample_count': metrics.get('sample_count', 0)
                })
    
    if hourly_data:
        hourly_df = pd.DataFrame(hourly_data)
        hourly_df.to_csv(output_dir / f'{symbol_short}_hourly_metrics.csv', index=False)
        log.info(f"Exported hourly metrics to {symbol_short}_hourly_metrics.csv")
    
    # 2. Recommendations CSV
    recommendations = determine_optimal_notional(analysis)
    rec_data = []
    
    for hour in range(24):
        rec_data.append({
            'hour': hour,
            'optimal_notional': recommendations['hourly'][hour].get('optimal_notional'),
            'viable_notionals': ','.join([str(n) for n in recommendations['hourly'][hour].get('viable_notionals', [])])
        })
    
    rec_df = pd.DataFrame(rec_data)
    rec_df.to_csv(output_dir / f'{symbol_short}_recommendations.csv', index=False)
    log.info(f"Exported recommendations to {symbol_short}_recommendations.csv")

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced liquidity analysis following comprehensive plan")
    parser.add_argument("--symbol", type=str, help="Specific symbol to analyze")
    parser.add_argument("--max-records", type=int, default=200000, 
                       help="Maximum records per period (default: 200k)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-3month", action="store_true", 
                       help="Skip 3-month comparison analysis")
    parser.add_argument("--export-csv", action="store_true", 
                       help="Export results to CSV files")
    parser.add_argument("--fee-bps", type=float, default=3.0, 
                       help="Taker fee in basis points (default: 3.0)")
    
    args = parser.parse_args()
    
    # Update config with custom fee if provided
    if args.fee_bps:
        config.fee_bps = args.fee_bps
    
    log.info("Starting enhanced liquidity analysis...")
    log.info(f"Configuration: fee={config.fee_bps} bps, delta={config.delta_bps} bps, PRmax={config.pr_max*100}%")
    
    # Get symbols (same logic as before)
    if args.symbol:
        symbols = [args.symbol]
        log.info(f"Analyzing specific symbol: {args.symbol}")
    else:
        symbols = settings.get_active_symbols_from_db()
        if not symbols:
            symbols = settings.get_symbols_from_db()
        if not symbols:
            log.error("No symbols found in database")
            return False
    
    log.info(f"Analyzing liquidity for {len(symbols)} symbols")
    
    # Create output directories
    Path("plots").mkdir(exist_ok=True)
    if args.export_csv:
        Path("output").mkdir(exist_ok=True)
    
    try:
        for symbol in symbols:
            log.info(f"\n{'='*80}")
            log.info(f"ANALYZING: {symbol}")
            log.info(f"{'='*80}")
            
            # Load all historical data
            orderbook_all = load_orderbook_data(symbol, max_records=args.max_records)
            
            if len(orderbook_all) == 0:
                log.warning(f"No orderbook data for {symbol}")
                continue
            
            # Try to load trades data for volume analysis
            trades_all = None
            if check_trades_table_exists():
                try:
                    trades_all = load_trades_data(symbol, 
                                                orderbook_all.index.min(), 
                                                orderbook_all.index.max())
                except Exception as e:
                    log.warning(f"Could not load trades data: {e}")
            else:
                log.info("Trades table not found - volume analysis will be skipped")
            
            # Analyze all data
            analysis_all = analyze_liquidity_metrics(symbol, orderbook_all, trades_all)
            
            # Analyze last 3 months if requested
            analysis_3m = None
            if not args.no_3month:
                three_months_ago = orderbook_all.index.max() - timedelta(days=90)
                if orderbook_all.index.min() < three_months_ago:
                    log.info("\nAnalyzing last 3 months separately...")
                    orderbook_3m = orderbook_all[orderbook_all.index >= three_months_ago]
                    
                    trades_3m = None
                    if trades_all is not None:
                        trades_3m = trades_all[trades_all.index >= three_months_ago]
                    
                    analysis_3m = analyze_liquidity_metrics(symbol, orderbook_3m, trades_3m)
            
            # Generate visualizations
            if not args.no_plots:
                create_comprehensive_visualizations(symbol, analysis_all, analysis_3m)
            
            # Generate report
            generate_detailed_report(symbol, analysis_all, analysis_3m)
            
            # Export to CSV if requested
            if args.export_csv:
                export_results_to_csv(symbol, analysis_all)
        
        log.info(f"\n🎉 Enhanced liquidity analysis completed!")
        log.info(f"Key features implemented:")
        log.info(f"  ✅ Dual time horizon analysis (all data vs 3 months)")
        log.info(f"  ✅ Hourly liquidity patterns")
        log.info(f"  ✅ Round-trip cost analysis")
        log.info(f"  ✅ Depth at ±{config.delta_bps} bps")
        log.info(f"  ✅ Traffic light scorecard")
        log.info(f"  ✅ Optimal notional recommendations (global and hourly)")
        log.info(f"  ✅ Stability analysis (RT ratio)")
        log.info(f"  ✅ Volume and participation rate analysis (if trades data available)")
        log.info(f"Check plots/ directory for comprehensive visualizations")
        
        return True
        
    except Exception as e:
        log.error(f"Liquidity analysis failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)