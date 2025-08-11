#!/usr/bin/env python3
"""
Liquidity Analysis - Complete Implementation with 20% Sampling
- Box plots for slippage and round-trip analysis (no outliers, better axes)
- 4-color traffic light system (blue/green/yellow/red)
- 20% data sampling (distributed across full time period)
- Improved visualizations with proper axis scaling
- Enhanced visualizations focused on useful metrics
- Mean lines in distributions, statistics table, timing tracking
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
import time
from contextlib import contextmanager
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

# 🕐 TIMING CONTEXT MANAGER
@contextmanager
def timer(description: str):
    """Context manager for timing operations"""
    start_time = time.time()
    log.info(f"⏳ Starting: {description}")
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        log.info(f"✅ Completed: {description} ({elapsed_time:.2f}s)")

# 🎯 CONFIGURATION WITH 4-COLOR SYSTEM
@dataclass
class LiquidityConfig:
    """Refined liquidity analysis configuration"""
    notional_grid: List[float] = None
    pr_max: float = 0.10  # max participation rate (10%)
    fee_bps: float = 2.0  # taker fee: 0.02% = 2 bps
    bar_minutes: int = 5  # bar size for volume analysis
    sample_rate: float = 0.20  # 20% sampling rate
    
    # 🎯 4-COLOR THRESHOLDS (SLIPPAGE)
    slippage_blue: float = 5.0      # < 0.05% = 5 bps
    slippage_green: float = 10.0    # < 0.1% = 10 bps  
    slippage_yellow: float = 30.0   # 0.1-0.3% = 10-30 bps
    # slippage_red: > 30 bps
    
    # 🎯 4-COLOR THRESHOLDS (ROUND-TRIP)
    # RT = slippage*2 + fees (4 bps total for round-trip)
    rt_blue: float = 14.0       # < 5*2 + 4 = 14 bps
    rt_green: float = 24.0      # < 10*2 + 4 = 24 bps
    rt_yellow: float = 64.0     # < 30*2 + 4 = 64 bps
    # rt_red: > 64 bps
    
    # Execution rate thresholds
    exec_rate_excellent: float = 95.0
    exec_rate_good: float = 80.0
    
    def __post_init__(self):
        if self.notional_grid is None:
            # 8 points up to 5k, more spacing at higher values
            self.notional_grid = [100, 250, 500, 1000, 1500, 2500, 3500, 5000]


config = LiquidityConfig()

# ===== UTILS: DISCOVER SYMBOLS =====

def get_available_symbols() -> List[str]:
    """Discover all available symbols. Tries symbol_info first, falls back to orderbook."""
    with db_manager.get_session() as session:
        try:
            rows = session.execute(text(
                """
                SELECT DISTINCT symbol_id 
                FROM symbol_info 
                ORDER BY symbol_id
                """
            )).fetchall()
            symbols = [r[0] for r in rows]
            if symbols:
                return symbols
        except Exception as e:
            log.warning(f"⚠️ Could not query symbol_info: {e}. Falling back to orderbook.")
        rows = session.execute(text(
            """
            SELECT DISTINCT symbol 
            FROM orderbook
            ORDER BY symbol
            """
        )).fetchall()
        return [r[0] for r in rows]

# ===== CORE DATA LOADING FUNCTIONS =====

def get_data_range(symbol: str) -> Tuple[Optional[datetime], Optional[datetime], int]:
    """Get full data range for symbol"""
    with timer(f"Getting data range for {symbol}"):
        with db_manager.get_session() as session:
            result = session.execute(text("""
                SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as total_records
                FROM orderbook 
                WHERE symbol = :symbol
                AND bid1_price IS NOT NULL 
                AND ask1_price IS NOT NULL
            """), {'symbol': symbol}).fetchone()
            
            if result:
                log.info(f"📊 Data range: {result.min_ts} to {result.max_ts} ({result.total_records:,} records)")
            
            return result.min_ts, result.max_ts, result.total_records if result else (None, None, 0)

def check_trades_table_exists() -> bool:
    """Check if trades table exists in database"""
    with timer("Checking trades table existence"):
        with db_manager.get_session() as session:
            result = session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'trades'
                );
            """)).fetchone()
            exists = result[0] if result else False
            log.info(f"🗃️ Trades table exists: {exists}")
            return exists

def load_orderbook_data(symbol: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Load orderbook data with 20% sampling distributed across full time period"""
    with timer(f"Loading orderbook data for {symbol}"):
        min_date, max_date, total_records = get_data_range(symbol)
        
        if not min_date or not max_date:
            log.warning(f"❌ No orderbook data found for {symbol}")
            return pd.DataFrame()
        
        # Use provided dates or defaults
        start_date = start_date or min_date
        end_date = end_date or max_date
        
        log.info(f"📈 Loading orderbook data for {symbol}")
        log.info(f"  📅 Period: {start_date} to {end_date}")
        log.info(f"  🎯 Sampling: {config.sample_rate:.0%} (distributed across full period)")
        
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
            expected_sample = int(period_records * config.sample_rate)
            
            log.info(f"  📊 Total records in period: {period_records:,}")
            log.info(f"  🎯 Expected sample size: ~{expected_sample:,} records")
            
            # Use TABLESAMPLE for efficient sampling
            sample_clause = f"TABLESAMPLE SYSTEM ({config.sample_rate * 100:.1f})"
            
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
            
            actual_sample_rate = len(df) / period_records if period_records > 0 else 0
            log.info(f"  ✅ Loaded: {len(df):,} orderbook snapshots ({actual_sample_rate:.1%} of total)")
            
            # Add hour column for hourly analysis
            df['hour'] = df.index.hour
            
            return df

def load_trades_data(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Load trades data for volume analysis with sampling"""
    with timer(f"Loading trades data for {symbol}"):
        if not check_trades_table_exists():
            log.info("  ⚠️ Trades table not found - volume analysis will be skipped")
            return None
            
        try:
            with db_manager.get_session() as session:
                # Also sample trades data for consistency
                sample_clause = f"TABLESAMPLE SYSTEM ({config.sample_rate * 100:.1f})"
                
                query = text(f"""
                    SELECT 
                        timestamp,
                        price,
                        size,
                        price * size as volume_usd
                    FROM trades {sample_clause}
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
                    log.info(f"  ✅ Loaded: {len(df):,} trades (sampled)")
                else:
                    log.info(f"  ⚠️ No trades data found")
                
                return df
        except Exception as e:
            log.warning(f"❌ Could not load trades data: {e}")
            return None

# ===== CORE CALCULATION FUNCTIONS =====

def calculate_slippage_and_rt(row: pd.Series, notional_usd: float, side: str = 'buy', 
                             fee_bps: float = 2.0) -> Dict:
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

def calculate_bid_ask_spread(row: pd.Series) -> float:
    """Calculate bid-ask spread in bps"""
    bid1 = row.get('bid1_price', np.nan)
    ask1 = row.get('ask1_price', np.nan)
    
    if pd.isna(bid1) or pd.isna(ask1) or bid1 <= 0 or ask1 <= 0:
        return np.nan
    
    mid_price = (bid1 + ask1) / 2
    spread_bps = ((ask1 - bid1) / mid_price) * 10000
    
    return spread_bps

def analyze_liquidity_metrics(symbol: str, orderbook_df: pd.DataFrame, 
                            trades_df: pd.DataFrame = None) -> Dict:
    """Comprehensive liquidity analysis"""
    with timer(f"Analyzing liquidity metrics for {symbol}"):
        log.info(f"🔍 Analyzing liquidity metrics for {symbol}...")
        log.info(f"  📊 Processing {len(orderbook_df):,} orderbook snapshots")
        log.info(f"  🎯 Analyzing {len(config.notional_grid)} notional sizes: {config.notional_grid}")
        
        results = {
            'symbol': symbol,
            'period_start': orderbook_df.index.min(),
            'period_end': orderbook_df.index.max(),
            'total_snapshots': len(orderbook_df),
            'hourly_metrics': {},
            'global_metrics': {},
            'raw_data': {},  # Store raw data for box plots
            'statistics_summary': {}  # Store all statistics for summary table
        }
        
        # Calculate bid-ask spreads
        with timer("Calculating bid-ask spreads"):
            log.info("  📏 Calculating bid-ask spreads...")
            spreads = []
            for idx, row in orderbook_df.iterrows():
                spread = calculate_bid_ask_spread(row)
                if not np.isnan(spread):
                    spreads.append(spread)
            
            results['spreads'] = spreads
            log.info(f"    ✅ Calculated {len(spreads):,} valid spreads")
        
        # Process each notional size
        for i, notional in enumerate(config.notional_grid):
            with timer(f"Processing ${notional:,} notional ({i+1}/{len(config.notional_grid)})"):
                log.info(f"  💰 Processing ${notional:,} notional ({i+1}/{len(config.notional_grid)})...")
                
                hourly_data = {hour: {
                    'slippage_buy': [], 'slippage_sell': [],
                    'rt_buy': [], 'rt_sell': [],
                    'can_execute_buy': [], 'can_execute_sell': []
                } for hour in range(24)}
                
                all_slippage = []
                all_rt = []
                all_exec_buy = []
                all_exec_sell = []
                
                # Process each orderbook snapshot
                for idx, row in orderbook_df.iterrows():
                    hour = row['hour']
                    
                    # Calculate slippage and RT for buy side
                    buy_result = calculate_slippage_and_rt(row, notional, 'buy', config.fee_bps)
                    if buy_result['can_execute']:
                        hourly_data[hour]['slippage_buy'].append(buy_result['slippage_bps'])
                        hourly_data[hour]['rt_buy'].append(buy_result['rt_bps'])
                        all_slippage.append(buy_result['slippage_bps'])
                        all_rt.append(buy_result['rt_bps'])
                    hourly_data[hour]['can_execute_buy'].append(buy_result['can_execute'])
                    all_exec_buy.append(buy_result['can_execute'])
                    
                    # Calculate slippage and RT for sell side
                    sell_result = calculate_slippage_and_rt(row, notional, 'sell', config.fee_bps)
                    if sell_result['can_execute']:
                        hourly_data[hour]['slippage_sell'].append(sell_result['slippage_bps'])
                        hourly_data[hour]['rt_sell'].append(sell_result['rt_bps'])
                        all_slippage.append(sell_result['slippage_bps'])
                        all_rt.append(sell_result['rt_bps'])
                    hourly_data[hour]['can_execute_sell'].append(sell_result['can_execute'])
                    all_exec_sell.append(sell_result['can_execute'])
                
                log.info(f"    ✅ Processed all {len(orderbook_df):,} snapshots")
                if len(orderbook_df) > 0:
                    log.info(f"    📈 Valid executions: {len(all_slippage):,} ({len(all_slippage)/len(orderbook_df)/2:.1%})")
                
                # Store raw data for box plots
                results['raw_data'][notional] = {
                    'slippage': all_slippage,
                    'rt': all_rt
                }
                
                # Calculate hourly statistics
                hourly_stats = {}
                for hour in range(24):
                    data = hourly_data[hour]
                    combined_slippage = data['slippage_buy'] + data['slippage_sell']
                    combined_rt = data['rt_buy'] + data['rt_sell']
                    
                    if combined_slippage:
                        hourly_stats[hour] = {
                            'slippage_p50': np.percentile(combined_slippage, 50),
                            'slippage_p95': np.percentile(combined_slippage, 95),
                            'rt_p50': np.percentile(combined_rt, 50),
                            'rt_p95': np.percentile(combined_rt, 95),
                            'exec_rate_buy': np.mean(data['can_execute_buy']) * 100,
                            'exec_rate_sell': np.mean(data['can_execute_sell']) * 100,
                            'sample_count': len(combined_slippage)
                        }
                    else:
                        hourly_stats[hour] = {
                            'slippage_p50': np.nan, 'slippage_p95': np.nan,
                            'rt_p50': np.nan, 'rt_p95': np.nan,
                            'exec_rate_buy': 0, 'exec_rate_sell': 0,
                            'sample_count': 0
                        }
                
                results['hourly_metrics'][notional] = hourly_stats
                
                # Calculate global statistics
                if all_slippage:
                    global_stats = {
                        'slippage_mean': np.mean(all_slippage),
                        'slippage_p50': np.percentile(all_slippage, 50),
                        'slippage_p95': np.percentile(all_slippage, 95),
                        'rt_mean': np.mean(all_rt),
                        'rt_p50': np.percentile(all_rt, 50),
                        'rt_p95': np.percentile(all_rt, 95),
                        'exec_rate_buy': np.mean(all_exec_buy) * 100,
                        'exec_rate_sell': np.mean(all_exec_sell) * 100
                    }
                    results['global_metrics'][notional] = global_stats
                    
                    # Store for summary table
                    results['statistics_summary'][notional] = global_stats
                    
                    log.info(f"    📊 Stats: Slip={global_stats['slippage_p50']:.1f}bps, RT={global_stats['rt_p50']:.1f}bps")
        
        log.info(f"✅ Analysis completed for {symbol}")
        return results

# ===== TRAFFIC LIGHT EVALUATION =====

def evaluate_traffic_light_4colors(metrics: Dict, metric_type: str = 'slippage') -> str:
    """Evaluate 4-color traffic light status"""
    if metric_type == 'slippage':
        p50_val = metrics.get('slippage_p50', np.inf)
        if p50_val < config.slippage_blue:
            return 'blue'
        elif p50_val < config.slippage_green:
            return 'green'
        elif p50_val < config.slippage_yellow:
            return 'yellow'
        else:
            return 'red'
    
    elif metric_type == 'rt':
        p50_val = metrics.get('rt_p50', np.inf)
        if p50_val < config.rt_blue:
            return 'blue'
        elif p50_val < config.rt_green:
            return 'green'
        elif p50_val < config.rt_yellow:
            return 'yellow'
        else:
            return 'red'
    
    return 'red'

def create_traffic_light_analysis(analysis_all: Dict, analysis_3m: Dict = None) -> Dict:
    """Create comprehensive traffic light analysis"""
    with timer("Creating traffic light analysis"):
        traffic_lights = {
            'slippage': {},
            'rt': {}
        }
        
        for notional in config.notional_grid:
            if notional in analysis_all['global_metrics']:
                # Historical data
                slippage_all_p50 = evaluate_traffic_light_4colors(analysis_all['global_metrics'][notional], 'slippage')
                slippage_all_p95 = evaluate_traffic_light_4colors(
                    {'slippage_p50': analysis_all['global_metrics'][notional]['slippage_p95']}, 'slippage'
                )
                rt_all_p50 = evaluate_traffic_light_4colors(analysis_all['global_metrics'][notional], 'rt')
                rt_all_p95 = evaluate_traffic_light_4colors(
                    {'rt_p50': analysis_all['global_metrics'][notional]['rt_p95']}, 'rt'
                )
                
                # 3M data (if available)
                slippage_3m_p50 = slippage_all_p50  # Default fallback
                slippage_3m_p95 = slippage_all_p95
                rt_3m_p50 = rt_all_p50
                rt_3m_p95 = rt_all_p95
                
                if analysis_3m and notional in analysis_3m['global_metrics']:
                    slippage_3m_p50 = evaluate_traffic_light_4colors(analysis_3m['global_metrics'][notional], 'slippage')
                    slippage_3m_p95 = evaluate_traffic_light_4colors(
                        {'slippage_p50': analysis_3m['global_metrics'][notional]['slippage_p95']}, 'slippage'
                    )
                    rt_3m_p50 = evaluate_traffic_light_4colors(analysis_3m['global_metrics'][notional], 'rt')
                    rt_3m_p95 = evaluate_traffic_light_4colors(
                        {'rt_p50': analysis_3m['global_metrics'][notional]['rt_p95']}, 'rt'
                    )
                
                traffic_lights['slippage'][notional] = {
                    'p50_all': slippage_all_p50,
                    'p95_all': slippage_all_p95,
                    'p50_3m': slippage_3m_p50,
                    'p95_3m': slippage_3m_p95
                }
                
                traffic_lights['rt'][notional] = {
                    'p50_all': rt_all_p50,
                    'p95_all': rt_all_p95,
                    'p50_3m': rt_3m_p50,
                    'p95_3m': rt_3m_p95
                }
        
        return traffic_lights

# ===== VISUALIZATION FUNCTIONS =====

def create_comprehensive_visualizations(symbol: str, analysis_all: Dict, analysis_3m: Dict = None):
    """Create comprehensive liquidity visualizations"""
    with timer(f"Creating visualizations for {symbol}"):
        log.info(f"🎨 Creating comprehensive visualizations for {symbol}...")
        
        symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
        
        # Create figure with improved layout - 6 rows now (added statistics table)
        fig = plt.figure(figsize=(28, 24))
        gs = fig.add_gridspec(6, 4, hspace=0.4, wspace=0.3)
        
        # Title
        title = f'{symbol_short} - Liquidity Analysis (20% Sample) with Statistics'
        if analysis_3m:
            title += ' (Historical vs Recent 3M)'
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # 1. Box plots for slippage (historical and 3M)
        ax1 = fig.add_subplot(gs[0, :2])
        plot_slippage_boxplots(ax1, analysis_all, analysis_3m, 'historical')
        
        ax2 = fig.add_subplot(gs[0, 2:])
        plot_slippage_boxplots(ax2, analysis_all, analysis_3m, '3m')
        
        # 2. Box plots for round-trip (historical and 3M)
        ax3 = fig.add_subplot(gs[1, :2])
        plot_rt_boxplots(ax3, analysis_all, analysis_3m, 'historical')
        
        ax4 = fig.add_subplot(gs[1, 2:])
        plot_rt_boxplots(ax4, analysis_all, analysis_3m, '3m')
        
        # 3. Heatmap hourly patterns
        ax5 = fig.add_subplot(gs[2, :2])
        plot_hourly_heatmap(ax5, analysis_all, analysis_3m)
        
        # 4. Bid-ask spread distribution (improved)
        ax6 = fig.add_subplot(gs[2, 2:])
        plot_spread_distribution(ax6, analysis_all, analysis_3m)
        
        # 5. Slippage distribution (improved - historical vs 3M)
        ax7 = fig.add_subplot(gs[3, :2])
        plot_slippage_distribution_improved(ax7, analysis_all, analysis_3m)
        
        # 6. Cost breakdown (fixed for 2k issue)
        ax8 = fig.add_subplot(gs[3, 2:])
        plot_cost_breakdown(ax8, analysis_all)
        
        # 7. Execution rates (fixed for 2k issue)
        ax9 = fig.add_subplot(gs[4, :2])
        plot_execution_rates(ax9, analysis_all, analysis_3m)
        
        # 8. Traffic light tables
        ax10 = fig.add_subplot(gs[4, 2:])
        plot_traffic_light_combined(ax10, analysis_all, analysis_3m)
        
        # 9. Statistics summary table (full width)
        ax11 = fig.add_subplot(gs[5, :])
        plot_statistics_summary_table(ax11, analysis_all, analysis_3m)
        
        # Save plot
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        filename = f'{symbol_short}_liquidity_analysis.png'
        plt.savefig(plots_dir / filename, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        
        log.info(f"✅ Visualizations saved: plots/{filename}")

def plot_slippage_boxplots(ax, analysis_all: Dict, analysis_3m: Dict, period: str):
    """Plot slippage box plots with improved axes and no outliers"""
    analysis = analysis_all if period == 'historical' else analysis_3m
    if not analysis:
        analysis = analysis_all  # Fallback
    
    notionals = sorted(analysis['raw_data'].keys())
    slippage_data = [analysis['raw_data'][n]['slippage'] for n in notionals]
    
    # Filter out empty data
    valid_data = []
    valid_labels = []
    for i, data in enumerate(slippage_data):
        if data:  # Only include non-empty data
            valid_data.append(data)
            valid_labels.append(f'${notionals[i]/1000:.0f}k' if notionals[i] >= 1000 else f'${notionals[i]}')
    
    if not valid_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Slippage Distribution - {period.title()}')
        return
    
    # Create box plot without outliers
    bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True, showfliers=False)
    
    # Calculate reasonable Y-axis limits
    all_data = [val for sublist in valid_data for val in sublist]
    y_max = min(np.percentile(all_data, 95) * 1.2, config.slippage_yellow * 2)
    
    # Color boxes based on median values
    for i, (patch, data) in enumerate(zip(bp['boxes'], valid_data)):
        median_val = np.median(data)
        color = get_traffic_light_color(median_val, 'slippage')
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
        # Add median value as text
        ax.text(i + 1, median_val + y_max * 0.02, f'{median_val:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax.set_ylabel('Slippage (bps)')
    ax.set_title(f'Slippage Distribution - {period.title()}')
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axhline(y=config.slippage_blue, color='blue', linestyle='--', alpha=0.7, linewidth=1, label=f'Blue: {config.slippage_blue}bps')
    ax.axhline(y=config.slippage_green, color='green', linestyle='--', alpha=0.7, linewidth=1, label=f'Green: {config.slippage_green}bps')
    ax.axhline(y=config.slippage_yellow, color='orange', linestyle='--', alpha=0.7, linewidth=1, label=f'Yellow: {config.slippage_yellow}bps')
    
    # Add legend for reference lines
    ax.legend(fontsize=8, loc='upper right')
    
    # Rotate x-labels for better readability
    ax.tick_params(axis='x', rotation=45)

def plot_rt_boxplots(ax, analysis_all: Dict, analysis_3m: Dict, period: str):
    """Plot round-trip box plots with improved axes and no outliers"""
    analysis = analysis_all if period == 'historical' else analysis_3m
    if not analysis:
        analysis = analysis_all  # Fallback
    
    notionals = sorted(analysis['raw_data'].keys())
    rt_data = [analysis['raw_data'][n]['rt'] for n in notionals]
    
    # Filter out empty data
    valid_data = []
    valid_labels = []
    for i, data in enumerate(rt_data):
        if data:  # Only include non-empty data
            valid_data.append(data)
            valid_labels.append(f'${notionals[i]/1000:.0f}k' if notionals[i] >= 1000 else f'${notionals[i]}')
    
    if not valid_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Round-Trip Cost Distribution - {period.title()}')
        return
    
    # Create box plot without outliers
    bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True, showfliers=False)
    
    # Calculate reasonable Y-axis limits
    all_data = [val for sublist in valid_data for val in sublist]
    y_max = min(np.percentile(all_data, 95) * 1.2, config.rt_yellow * 2)
    
    # Color boxes based on median values
    for i, (patch, data) in enumerate(zip(bp['boxes'], valid_data)):
        median_val = np.median(data)
        color = get_traffic_light_color(median_val, 'rt')
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
        # Add median value as text
        ax.text(i + 1, median_val + y_max * 0.02, f'{median_val:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax.set_ylabel('Round-Trip Cost (bps)')
    ax.set_title(f'Round-Trip Cost Distribution - {period.title()}')
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    
    # Add reference lines
    ax.axhline(y=config.rt_blue, color='blue', linestyle='--', alpha=0.7, linewidth=1, label=f'Blue: {config.rt_blue}bps')
    ax.axhline(y=config.rt_green, color='green', linestyle='--', alpha=0.7, linewidth=1, label=f'Green: {config.rt_green}bps')
    ax.axhline(y=config.rt_yellow, color='orange', linestyle='--', alpha=0.7, linewidth=1, label=f'Yellow: {config.rt_yellow}bps')
    
    # Add legend for reference lines
    ax.legend(fontsize=8, loc='upper right')
    
    # Rotate x-labels for better readability
    ax.tick_params(axis='x', rotation=45)

def plot_hourly_heatmap(ax, analysis_all: Dict, analysis_3m: Dict):
    """Plot hourly RT heatmap comparing historical vs 3M"""
    if 1000 not in analysis_all['hourly_metrics']:
        ax.text(0.5, 0.5, 'No hourly data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Hourly RT Patterns')
        return
    
    hours = list(range(24))
    
    # Historical data
    rt_hist = []
    for hour in hours:
        hour_data = analysis_all['hourly_metrics'][1000].get(hour, {})
        rt_p50 = hour_data.get('rt_p50', np.nan)
        rt_hist.append(rt_p50 if not np.isnan(rt_p50) else 0)
    
    # 3M data
    rt_3m = rt_hist.copy()  # Default fallback
    if analysis_3m and 1000 in analysis_3m['hourly_metrics']:
        rt_3m = []
        for hour in hours:
            hour_data = analysis_3m['hourly_metrics'][1000].get(hour, {})
            rt_p50 = hour_data.get('rt_p50', np.nan)
            rt_3m.append(rt_p50 if not np.isnan(rt_p50) else 0)
    
    # Create heatmap data
    heatmap_data = np.array([rt_hist, rt_3m])
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    
    # Set labels
    ax.set_xticks(range(0, 24, 2))  # Every 2 hours
    ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Historical', '3 Months'])
    ax.set_xlabel('Hour of Day')
    ax.set_title('RT Cost Heatmap ($1000) - Historical vs 3M')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='RT Cost (bps)')
    
    # Add value annotations (every 2 hours for readability)
    for i in range(2):
        for j in range(0, 24, 2):
            if heatmap_data[i, j] > 0:
                ax.text(j, i, f'{heatmap_data[i, j]:.0f}',
                        ha="center", va="center", color="black", fontsize=8)

def plot_spread_distribution(ax, analysis_all: Dict, analysis_3m: Dict):
    """Plot bid-ask spread distribution comparing historical vs 3M with mean lines"""
    spreads_all = analysis_all.get('spreads', [])
    
    if not spreads_all:
        ax.text(0.5, 0.5, 'No spread data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Bid-Ask Spread Distribution')
        return
    
    # Calculate reasonable x-axis limit
    x_max = np.percentile(spreads_all, 95) * 1.2
    
    # Plot historical
    n_hist, bins_hist, patches_hist = ax.hist(spreads_all, bins=40, alpha=0.6, label='Historical', range=(0, x_max), density=True)
    
    # Add mean line for historical in the same color as the histogram
    mean_hist = np.mean(spreads_all)
    hist_color = patches_hist[0].get_facecolor()
    ax.axvline(mean_hist, linestyle='-', linewidth=2, color=hist_color, alpha=1.0, label=f'Hist Mean: {mean_hist:.1f} bps')
    
    # 3M data if available
    if analysis_3m:
        spreads_3m = analysis_3m.get('spreads', [])
        if spreads_3m:
            n_3m, bins_3m, patches_3m = ax.hist(spreads_3m, bins=40, alpha=0.6, label='3 Months', range=(0, x_max), density=True)
            
            # Add mean line for 3M in the same color as the histogram
            mean_3m = np.mean(spreads_3m)
            color_3m = patches_3m[0].get_facecolor()
            ax.axvline(mean_3m, linestyle='-', linewidth=2, color=color_3m, alpha=1.0, label=f'3M Mean: {mean_3m:.1f} bps')
    
    # Add median lines
    median_spread = np.median(spreads_all)
    ax.axvline(median_spread, linestyle='--', alpha=0.8, linewidth=2, label=f'Hist Median: {median_spread:.1f} bps')
    
    if analysis_3m and analysis_3m.get('spreads'):
        median_3m = np.median(analysis_3m['spreads'])
        ax.axvline(median_3m, linestyle='--', alpha=0.8, linewidth=2, label=f'3M Median: {median_3m:.1f} bps')
    
    ax.set_xlabel('Bid-Ask Spread (bps)')
    ax.set_ylabel('Density')
    ax.set_title('Bid-Ask Spread Distribution with Mean/Median Lines')
    ax.set_xlim(0, x_max)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

def plot_slippage_distribution_improved(ax, analysis_all: Dict, analysis_3m: Dict):
    """Plot slippage distribution comparing historical vs 3M for key notionals with mean lines"""
    key_notionals = [500, 1000, 2000]
    
    # Calculate reasonable x-axis limit
    all_slippage = []
    for notional in key_notionals:
        if notional in analysis_all['raw_data']:
            all_slippage.extend(analysis_all['raw_data'][notional]['slippage'])
    
    if not all_slippage:
        ax.text(0.5, 0.5, 'No slippage data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Slippage Distribution Comparison')
        return
    
    x_max = min(np.percentile(all_slippage, 95) * 1.2, config.slippage_yellow * 2)
    
    for notional in key_notionals:
        if notional in analysis_all['raw_data']:
            slippage_data = analysis_all['raw_data'][notional]['slippage']
            if slippage_data:
                n_hist, bins_hist, patches_hist = ax.hist(slippage_data, bins=30, alpha=0.4, label=f'${notional:,} (Hist)', density=True, range=(0, x_max))
                
                # Add mean line for historical in the same color as the histogram
                mean_hist = np.mean(slippage_data)
                color_hist = patches_hist[0].get_facecolor()
                ax.axvline(mean_hist, linestyle='-', linewidth=1.5, color=color_hist, alpha=1.0)
        
        # 3M data
        if analysis_3m and notional in analysis_3m['raw_data']:
            slippage_3m = analysis_3m['raw_data'][notional]['slippage']
            if slippage_3m:
                n_3m, bins_3m, patches_3m = ax.hist(slippage_3m, bins=30, alpha=0.6, label=f'${notional:,} (3M)', density=True, range=(0, x_max))
                
                # Add mean line for 3M in the same color as the histogram
                mean_3m = np.mean(slippage_3m)
                color_3m = patches_3m[0].get_facecolor()
                ax.axvline(mean_3m, linestyle='-', linewidth=1.5, color=color_3m, alpha=1.0)
    
    ax.set_xlabel('Slippage (bps)')
    ax.set_ylabel('Density')
    ax.set_title('Slippage Distribution - Historical vs 3M (with Means)')
    ax.set_xlim(0, x_max)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

def plot_cost_breakdown(ax, analysis: Dict):
    """Plot cost breakdown (fees vs slippage) - fixed for missing data"""
    key_notionals = [500, 1000, 1500]  # Removed 2000 to avoid issues
    
    slip_costs = []
    fee_costs = []
    valid_notionals = []
    
    for notional in key_notionals:
        if notional in analysis['global_metrics']:
            slip_p50 = analysis['global_metrics'][notional]['slippage_p50']
            slip_costs.append(slip_p50)
            fee_costs.append(config.fee_bps)
            valid_notionals.append(notional)
    
    if not valid_notionals:
        ax.text(0.5, 0.5, 'No cost data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Trading Cost Breakdown (One-Way)')
        return
    
    x = np.arange(len(valid_notionals))
    width = 0.6
    
    # Stacked bar chart
    ax.bar(x, fee_costs, width, label='Fee Cost', alpha=0.7)
    ax.bar(x, slip_costs, width, bottom=fee_costs, label='Slippage Cost', alpha=0.7)
    
    ax.set_xlabel('Notional Size')
    ax.set_ylabel('Cost (bps)')
    ax.set_title('Trading Cost Breakdown (One-Way)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'${n/1000:.0f}k' if n >= 1000 else f'${n}' for n in valid_notionals])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add total cost labels
    for i, (fee, slip) in enumerate(zip(fee_costs, slip_costs)):
        total = fee + slip
        ax.text(i, total + 0.5, f'{total:.1f}', ha='center', va='bottom', fontweight='bold')

def plot_execution_rates(ax, analysis_all: Dict, analysis_3m: Dict):
    """Plot execution rates comparison - fixed for missing data"""
    notionals = [500, 1000, 1500]  # Removed 2000 to avoid issues
    
    # Historical execution rates
    hist_exec = []
    recent_exec = []
    valid_notionals = []
    
    for notional in notionals:
        if notional in analysis_all['global_metrics']:
            hist_rate = min(
                analysis_all['global_metrics'][notional]['exec_rate_buy'],
                analysis_all['global_metrics'][notional]['exec_rate_sell']
            )
            hist_exec.append(hist_rate)
            
            if analysis_3m and notional in analysis_3m['global_metrics']:
                recent_rate = min(
                    analysis_3m['global_metrics'][notional]['exec_rate_buy'],
                    analysis_3m['global_metrics'][notional]['exec_rate_sell']
                )
                recent_exec.append(recent_rate)
            else:
                recent_exec.append(hist_rate)  # Fallback
            
            valid_notionals.append(notional)
    
    if not valid_notionals:
        ax.text(0.5, 0.5, 'No execution data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Execution Quality Comparison')
        return
    
    x = np.arange(len(valid_notionals))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hist_exec, width, label='Historical', alpha=0.7)
    bars2 = ax.bar(x + width/2, recent_exec, width, label='Recent 3M', alpha=0.7)
    
    ax.set_xlabel('Notional Size')
    ax.set_ylabel('Execution Rate (%)')
    ax.set_title('Execution Quality Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'${n/1000:.0f}k' if n >= 1000 else f'${n}' for n in valid_notionals])
    ax.set_ylim(0, 105)  # Fixed Y-axis
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

def plot_traffic_light_combined(ax, analysis_all: Dict, analysis_3m: Dict):
    """Plot combined traffic light table for both slippage and RT"""
    ax.axis('off')
    
    traffic_lights = create_traffic_light_analysis(analysis_all, analysis_3m)
    
    # Prepare table data - show key notionals only
    key_notionals = [500, 1000, 1500, 2500]
    headers = ['Notional', 'Slip P50', 'Slip 3M', 'RT P50', 'RT 3M']
    table_data = []
    
    color_map = {
        'blue': '#4A90E2',
        'green': '#7ED321',
        'yellow': '#F5A623',
        'red': '#D0021B'
    }
    
    for notional in key_notionals:
        if (notional in traffic_lights['slippage'] and 
            notional in traffic_lights['rt']):
            
            slip_lights = traffic_lights['slippage'][notional]
            rt_lights = traffic_lights['rt'][notional]
            
            row = [f'${notional/1000:.0f}k' if notional >= 1000 else f'${notional}']
            colors = ['white']  # First column (notional) is white
            
            # Add traffic light status for each column
            colors.append(color_map[slip_lights['p50_all']])   # Slip P50
            colors.append(color_map[slip_lights['p50_3m']])    # Slip 3M
            colors.append(color_map[rt_lights['p50_all']])     # RT P50
            colors.append(color_map[rt_lights['p50_3m']])      # RT 3M
            
            row.extend(['●', '●', '●', '●'])
            table_data.append((row, colors))
    
    if not table_data:
        ax.text(0.5, 0.5, 'No traffic light data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Traffic Light Summary')
        return
    
    # Create table
    cell_text = [row[0] for row in table_data]
    
    table = ax.table(cellText=cell_text, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Color cells
    for i, (row, colors) in enumerate(table_data):
        for j in range(len(headers)):
            if j > 0:  # Skip first column (notional)
                table[(i+1, j)].set_facecolor(colors[j])
                table[(i+1, j)].set_text_props(color='white', weight='bold')
    
    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Traffic Light Summary (P50 Values)', fontsize=12, fontweight='bold', pad=20)

def plot_statistics_summary_table(ax, analysis_all: Dict, analysis_3m: Dict):
    """Plot comprehensive statistics summary table"""
    ax.axis('off')
    
    # Prepare comprehensive statistics data
    key_notionals = sorted([n for n in config.notional_grid if n in analysis_all.get('global_metrics', {})])
    
    if not key_notionals:
        ax.text(0.5, 0.5, 'No statistics data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Statistics Summary Table')
        return
    
    # Create headers
    headers = ['Notional', 'Slip Mean', 'Slip P50', 'Slip P95', 'RT Mean', 'RT P50', 'RT P95', 'Exec Buy%', 'Exec Sell%']
    
    # Prepare data rows
    table_data = []
    cell_colors = []
    
    for notional in key_notionals:
        if notional in analysis_all['global_metrics']:
            stats = analysis_all['global_metrics'][notional]
            
            # Format notional
            notional_str = f'${notional/1000:.0f}k' if notional >= 1000 else f'${notional}'
            
            # Extract values
            slip_mean = stats['slippage_mean']
            slip_p50 = stats['slippage_p50']
            slip_p95 = stats['slippage_p95']
            rt_mean = stats['rt_mean']
            rt_p50 = stats['rt_p50']
            rt_p95 = stats['rt_p95']
            exec_buy = stats['exec_rate_buy']
            exec_sell = stats['exec_rate_sell']
            
            # Create row
            row = [
                notional_str,
                f'{slip_mean:.1f}',
                f'{slip_p50:.1f}',
                f'{slip_p95:.1f}',
                f'{rt_mean:.1f}',
                f'{rt_p50:.1f}',
                f'{rt_p95:.1f}',
                f'{exec_buy:.1f}',
                f'{exec_sell:.1f}'
            ]
            
            # Create color row (based on traffic light for key metrics)
            colors = [
                'lightgray',  # Notional
                '#E8F4F8',    # Slip Mean (light blue)
                get_traffic_light_color(slip_p50, 'slippage'),  # Slip P50 (traffic light)
                '#FFE8E8',    # Slip P95 (light red)
                '#E8F4F8',    # RT Mean (light blue)
                get_traffic_light_color(rt_p50, 'rt'),  # RT P50 (traffic light)
                '#FFE8E8',    # RT P95 (light red)
                '#E8F8E8' if exec_buy >= 90 else '#FFF8E8' if exec_buy >= 80 else '#FFE8E8',  # Exec Buy
                '#E8F8E8' if exec_sell >= 90 else '#FFF8E8' if exec_sell >= 80 else '#FFE8E8'  # Exec Sell
            ]
            
            table_data.append(row)
            cell_colors.append(colors)
    
    if not table_data:
        ax.text(0.5, 0.5, 'No statistics data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Statistics Summary Table')
        return
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    
    # Color cells
    for i, colors in enumerate(cell_colors):
        for j, color in enumerate(colors):
            table[(i+1, j)].set_facecolor(color)
            # Set text color based on background
            table[(i+1, j)].set_text_props(color='black')
    
    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=10)
    
    ax.set_title('📊 Comprehensive Statistics Summary (All Values in bps except Execution %)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_text = (
        "Color Code: Blue/Green/Yellow/Red = Traffic Light Status (P50 values)\n"
        "Light Blue = Mean values, Light Red = P95 values, Green/Yellow/Red = Execution rates"
    )
    ax.text(0.5, -0.1, legend_text, ha='center', va='top', transform=ax.transAxes, 
            fontsize=10, style='italic')

def get_traffic_light_color(value: float, metric_type: str) -> str:
    """Get traffic light color for matplotlib"""
    color_map = {
        'blue': '#4A90E2',
        'green': '#7ED321', 
        'yellow': '#F5A623',
        'red': '#D0021B'
    }
    
    if metric_type == 'slippage':
        if value < config.slippage_blue:
            return color_map['blue']
        elif value < config.slippage_green:
            return color_map['green']
        elif value < config.slippage_yellow:
            return color_map['yellow']
        else:
            return color_map['red']
    
    elif metric_type == 'rt':
        if value < config.rt_blue:
            return color_map['blue']
        elif value < config.rt_green:
            return color_map['green']
        elif value < config.rt_yellow:
            return color_map['yellow']
        else:
            return color_map['red']
    
    return color_map['red']

# ===== REPORTING FUNCTIONS =====

def generate_refined_report(symbol: str, analysis_all: Dict, analysis_3m: Dict = None):
    """Generate refined analysis report with enhanced logging"""
    with timer("Generating refined report"):
        log.info(f"\n{'='*80}")
        log.info(f"📊 LIQUIDITY ANALYSIS - {symbol} (20% SAMPLE)")
        log.info(f"{'='*80}")
        
        symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
        
        # Executive Summary
        log.info(f"\n📋 EXECUTIVE SUMMARY")
        log.info(f"  🎯 Symbol: {symbol_short}")
        log.info(f"  📅 Analysis Period: {analysis_all['period_start']} to {analysis_all['period_end']}")
        log.info(f"  📊 Sample Size: {analysis_all['total_snapshots']:,} snapshots ({config.sample_rate:.0%})")
        log.info(f"  💰 Fee Structure: {config.fee_bps} bps ({config.fee_bps/100:.2f}%)")
        log.info(f"  🎲 Notional Grid: {len(config.notional_grid)} sizes from ${min(config.notional_grid):,} to ${max(config.notional_grid):,}")
        
        # Detailed Statistics Summary
        log.info(f"\n📊 DETAILED STATISTICS SUMMARY")
        log.info(f"{'Size':<8} {'SlipMean':<9} {'SlipP50':<8} {'SlipP95':<8} {'RTMean':<8} {'RTP50':<7} {'RTP95':<7} {'ExecB%':<7} {'ExecS%':<7}")
        log.info("-" * 85)
        
        for notional in sorted(analysis_all.get('global_metrics', {}).keys()):
            if notional in analysis_all['global_metrics']:
                stats = analysis_all['global_metrics'][notional]
                size_str = f"${notional:,}"
                
                log.info(f"{size_str:<8} "
                        f"{stats['slippage_mean']:>7.1f}  "
                        f"{stats['slippage_p50']:>6.1f}  "
                        f"{stats['slippage_p95']:>6.1f}  "
                        f"{stats['rt_mean']:>6.1f}  "
                        f"{stats['rt_p50']:>5.1f}  "
                        f"{stats['rt_p95']:>5.1f}  "
                        f"{stats['exec_rate_buy']:>5.1f}  "
                        f"{stats['exec_rate_sell']:>5.1f}")
        
        # Liquidity Quality Assessment
        log.info(f"\n💹 LIQUIDITY QUALITY ASSESSMENT (4-Color System)")
        log.info(f"{'Size':<8} {'Slip P50':<9} {'RT P50':<8} {'Exec %':<8} {'Quality':<10}")
        log.info("-" * 50)
        
        for notional in [500, 1000, 1500, 2500]:  # Key notionals
            if notional in analysis_all['global_metrics']:
                metrics = analysis_all['global_metrics'][notional]
                
                slip_p50 = metrics['slippage_p50']
                rt_p50 = metrics['rt_p50']
                exec_rate = min(metrics['exec_rate_buy'], metrics['exec_rate_sell'])
                
                slip_color = evaluate_traffic_light_4colors({'slippage_p50': slip_p50}, 'slippage')
                rt_color = evaluate_traffic_light_4colors({'rt_p50': rt_p50}, 'rt')
                
                color_icons = {'blue': '🔵', 'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
                
                log.info(f"${notional:<7} "
                        f"{slip_p50:>6.1f} {color_icons[slip_color]} "
                        f"{rt_p50:>6.1f} {color_icons[rt_color]} "
                        f"{exec_rate:>6.1f}% "
                        f"{slip_color.upper()}/{rt_color.upper()}")
        
        # Traffic light summary for P95 (tail risk)
        key_notionals_tl = [500, 1000, 1500, 2500]
        log.info(f"\n🚦 TRAFFIC LIGHT (P95) – tail risk")
        log.info(f"{'Size':<8} {'Slip P95':<10} {'RT P95':<10}")
        log.info("-" * 32)
        for n in key_notionals_tl:
            if n in analysis_all.get('global_metrics', {}):
                stats = analysis_all['global_metrics'][n]
                slip95 = stats['slippage_p95']
                rt95 = stats['rt_p95']
                slip95_color = evaluate_traffic_light_4colors({'slippage_p50': slip95}, 'slippage')
                rt95_color = evaluate_traffic_light_4colors({'rt_p50': rt95}, 'rt')
                color_icons = {'blue': '🔵', 'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
                log.info(f"{('$'+format(n, ',')):<8} {slip95:>6.1f} {color_icons[slip95_color]}   {rt95:>6.1f} {color_icons[rt95_color]}")
        
        # Spread Analysis
        if 'spreads' in analysis_all and analysis_all['spreads']:
            spreads = analysis_all['spreads']
            mean_spread = np.mean(spreads)
            median_spread = np.median(spreads)
            p90_spread = np.percentile(spreads, 90)
            p95_spread = np.percentile(spreads, 95)
            p99_spread = np.percentile(spreads, 99)
            log.info(f"\n📏 BID-ASK SPREAD ANALYSIS")
            log.info(f"  📊 Mean Spread: {mean_spread:.1f} bps")
            log.info(f"  📊 Median Spread: {median_spread:.1f} bps")
            log.info(f"  📊 P90 Spread: {p90_spread:.1f} bps")
            log.info(f"  📊 P95 Spread: {p95_spread:.1f} bps")
            log.info(f"  📊 P99 Spread: {p99_spread:.1f} bps")
            log.info(f"  📊 Total Samples: {len(spreads):,}")
        
        # Hourly overview (RT P50) at a reference notional
        ref_notional = 1000
        if ref_notional not in analysis_all.get('hourly_metrics', {}):
            if analysis_all.get('hourly_metrics'):
                ref_notional = sorted(analysis_all['hourly_metrics'].keys(), key=lambda n: abs(n-1000))[0]
        hourly = analysis_all.get('hourly_metrics', {}).get(ref_notional, {})
        rows = []
        for h in range(24):
            m = hourly.get(h, {})
            rt50 = m.get('rt_p50', np.nan)
            if not np.isnan(rt50):
                rows.append((h, rt50, m.get('rt_p95', np.nan), m.get('exec_rate_buy', 0.0), m.get('exec_rate_sell', 0.0), m.get('sample_count', 0)))
        if rows:
            rows_sorted = sorted(rows, key=lambda x: x[1])
            best = rows_sorted[:3]
            worst = rows_sorted[-3:][::-1]
            log.info(f"\n⏱️ HOURLY OVERVIEW (RT P50) @ ${ref_notional:,}")
            log.info(f"  {'Hour':<6} {'RT_P50':>7} {'RT_P95':>7} {'ExecB%':>7} {'ExecS%':>7} {'Samples':>8}")
            log.info("  Best:")
            for h, rt50, rt95, eb, es, sc in best:
                log.info(f"    {h:02d}:00   {rt50:>6.1f}  {rt95:>6.1f}   {eb:>6.1f}   {es:>6.1f}   {sc:>8}")
            log.info("  Worst:")
            for h, rt50, rt95, eb, es, sc in worst:
                log.info(f"    {h:02d}:00   {rt50:>6.1f}  {rt95:>6.1f}   {eb:>6.1f}   {es:>6.1f}   {sc:>8}")
            log.info(f"  Coverage: {len(rows)}/24 hours with data")
        
        # Key Recommendations
        log.info(f"\n📋 KEY RECOMMENDATIONS")
        
        # Find optimal notional
        optimal_candidates = []
        for notional in config.notional_grid:
            if notional in analysis_all['global_metrics']:
                metrics = analysis_all['global_metrics'][notional]
                rt_color = evaluate_traffic_light_4colors({'rt_p50': metrics['rt_p50']}, 'rt')
                exec_rate = min(metrics['exec_rate_buy'], metrics['exec_rate_sell'])
                
                if rt_color in ['blue', 'green'] and exec_rate >= 85:
                    optimal_candidates.append((notional, metrics['rt_p50']))
        
        if optimal_candidates:
            optimal_notional, optimal_cost = max(optimal_candidates, key=lambda x: x[0])
            log.info(f"  🎯 Optimal Size: ${optimal_notional:,} (RT: {optimal_cost:.1f} bps)")
            
            # Risk assessment
            p95_cost = analysis_all['global_metrics'][optimal_notional]['rt_p95']
            if p95_cost > config.rt_yellow:
                log.info(f"  ⚠️ Risk Warning: P95 cost can reach {p95_cost:.1f} bps")
            else:
                log.info(f"  ✅ Low Risk: P95 cost controlled at {p95_cost:.1f} bps")
        else:
            log.info(f"  ❌ No optimal size found - consider smaller notionals or market timing")
        
        # Performance Summary
        log.info(f"\n⚡ PERFORMANCE SUMMARY")
        log.info(f"  📊 Total Analysis Time: Check timing logs above")
        log.info(f"  🎯 Sampling Efficiency: {config.sample_rate:.0%} of full dataset")
        log.info(f"  📈 Notionals Analyzed: {len([n for n in config.notional_grid if n in analysis_all.get('global_metrics', {})])}/{len(config.notional_grid)}")
        
        # Bottom Line
        log.info(f"\n🎯 BOTTOM LINE")
        if optimal_candidates:
            best_color = evaluate_traffic_light_4colors({'rt_p50': optimal_candidates[-1][1]}, 'rt')
            if best_color in ['blue', 'green']:
                log.info(f"  ✅ SUITABLE for systematic trading up to ${optimal_candidates[-1][0]:,}")
            else:
                log.info(f"  ⚠️ MARGINAL liquidity - monitor costs carefully")
        else:
            log.info(f"  🛑 INSUFFICIENT liquidity for systematic pair trading")

# ===== MAIN FUNCTION =====


def main():
    """Main analysis function with comprehensive timing and multi-symbol support"""
    start_time = time.time()

    import argparse

    parser = argparse.ArgumentParser(description="Liquidity analysis with statistics table and timing")
    # Make symbol optional and allow comma-separated list
    parser.add_argument("--symbol", type=str, required=False,
                        help="Symbol(s) to analyze (comma-separated). If omitted, analyze ALL available symbols.")

    args = parser.parse_args()

    log.info("🚀 Starting liquidity analysis")
    if args.symbol:
        log.info(f"  🎯 Symbols: {args.symbol}")
    else:
        log.info("  🎯 Symbols: [ALL] (auto-discovered)")
    log.info(f"  📊 Sampling: {config.sample_rate:.0%} (distributed across full period)")
    log.info(f"  💰 Fee: {config.fee_bps} bps ({config.fee_bps/100:.2f}%)")
    log.info(f"  📈 Notional range: ${min(config.notional_grid):,} - ${max(config.notional_grid):,}")

    # Create output directory
    Path("plots").mkdir(exist_ok=True)

    try:
        # Resolve symbol list
        if args.symbol:
            symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
        else:
            symbols = get_available_symbols()
            log.info(f"  🔎 Auto-discovered {len(symbols)} symbols")

        if not symbols:
            log.error("❌ No symbols found to analyze")
            return False

        successes = 0

        for idx, symbol in enumerate(symbols, 1):
            log.info(f"\n{'='*80}")
            log.info(f"🔍 ANALYZING: {symbol} ({idx}/{len(symbols)})")
            log.info(f"{'='*80}")

            try:
                # Load all historical data (with sampling)
                with timer("Loading historical orderbook data"):
                    orderbook_all = load_orderbook_data(symbol)

                if len(orderbook_all) == 0:
                    log.error(f"❌ No orderbook data found for {symbol}")
                    continue

                # Analyze all data
                with timer("Analyzing historical data"):
                    log.info("📊 Processing historical data...")
                    analysis_all = analyze_liquidity_metrics(symbol, orderbook_all, None)

                # Analyze last 3 months
                analysis_3m = None
                three_months_ago = orderbook_all.index.max() - timedelta(days=90)
                if orderbook_all.index.min() < three_months_ago:
                    with timer("Processing recent 3-month data"):
                        log.info("📊 Processing recent 3-month data...")
                        orderbook_3m = orderbook_all[orderbook_all.index >= three_months_ago]
                        analysis_3m = analyze_liquidity_metrics(symbol, orderbook_3m, None)

                # Create visualizations
                with timer("Creating comprehensive visualizations"):
                    log.info("🎨 Creating visualizations...")
                    create_comprehensive_visualizations(symbol, analysis_all, analysis_3m)

                # Generate report
                generate_refined_report(symbol, analysis_all, analysis_3m)

                successes += 1

            except Exception as e:
                log.error(f"❌ Analysis failed for {symbol}: {e}")
                import traceback
                log.error(traceback.format_exc())
                continue

        total_time = time.time() - start_time
        log.info(f"\n{'='*80}")
        if successes > 0:
            log.info(f"🎉 ANALYSIS COMPLETED: {successes}/{len(symbols)} symbols analyzed successfully")
            log.info(f"⏱️ Total Execution Time: {total_time:.2f} seconds")
            log.info("📊 Check plots/ directory for visualizations")
            log.info(f"⚡ Performance: {config.sample_rate:.0%} sampling provided significant speedup")
            log.info(f"{'='*80}")
            return True
        else:
            log.error("🛑 No symbols analyzed successfully")
            log.info(f"⏱️ Total Execution Time: {total_time:.2f} seconds")
            log.info(f"{'='*80}")
            return False

    except Exception as e:
        total_time = time.time() - start_time
        log.error(f"❌ Analysis failed after {total_time:.2f}s: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)