#!/usr/bin/env python3
"""
📊 Professional Pair Trading Analysis - DYNAMIC SPREAD VERSION
Análisis con spreads dinámicos calculados con OLS rolling
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
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller, kpss
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf, pacf
import argparse
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger

log = get_validation_logger()

# Configuración mejorada de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 100

def _infer_samples_per_day(index: pd.DatetimeIndex) -> float:
    """Return the mean number of observations per calendar day."""
    if len(index) < 2:
        return 1440.0  # Default: minute data
    
    # Calculate actual frequency
    total_days = (index[-1] - index[0]).total_seconds() / 86400
    if total_days <= 0:
        return 1440.0
    
    return len(index) / total_days

def load_mark_prices_data(symbol1: str, symbol2: str) -> Tuple[pd.Series, pd.Series]:
    """Cargar mark prices de la base de datos (como en analyze_markprices.py)"""
    log.info(f"Loading mark prices for {symbol1} / {symbol2}")
    
    with db_manager.get_session() as session:
        # Cargar mark prices directamente
        query = text("""
            SELECT 
                m1.timestamp,
                m1.mark_price as price1,
                m2.mark_price as price2
            FROM mark_prices m1
            JOIN mark_prices m2 ON m1.timestamp = m2.timestamp
            WHERE m1.symbol = :symbol1
            AND m2.symbol = :symbol2
            AND m1.mark_price > 0
            AND m2.mark_price > 0
            AND m1.valid_for_trading = TRUE
            AND m2.valid_for_trading = TRUE
            ORDER BY m1.timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params={
            'symbol1': symbol1, 'symbol2': symbol2
        }, parse_dates=['timestamp'])
        
        if df.empty:
            log.error("No aligned mark prices data found")
            return pd.Series(), pd.Series()
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Resample to regular intervals
        df_resampled = df.resample('1T').last()
        df_aligned = df_resampled.dropna()
        
        prices1 = df_aligned['price1']
        prices2 = df_aligned['price2']
        
        log.info(f"Loaded {len(df_aligned):,} aligned mark price observations")
        log.info(f"Price ranges - {symbol1}: ${prices1.min():.2f} to ${prices1.max():.2f}")
        log.info(f"Price ranges - {symbol2}: ${prices2.min():.2f} to ${prices2.max():.2f}")
        
        return prices1, prices2

def calculate_dynamic_spread(log_p1: pd.Series, log_p2: pd.Series, window: int) -> Dict:
    """Calculate spread using rolling OLS parameters"""
    log.info(f"Calculating dynamic spread with {window}-sample window...")
    
    # Initialize arrays
    spreads = []
    alphas = []
    betas = []
    r_squareds = []
    timestamps = []
    
    # Rolling calculation
    for i in range(window, len(log_p1)):
        window_p1 = log_p1.iloc[i-window:i]
        window_p2 = log_p2.iloc[i-window:i]
        
        # OLS regression for this window
        X = np.column_stack([np.ones(len(window_p2)), window_p2.values])
        model = OLS(window_p1.values, X).fit()
        
        alpha = model.params[0]
        beta = model.params[1]
        r_squared = model.rsquared
        
        # Calculate spread for current point using window's parameters
        spread_value = log_p1.iloc[i] - (beta * log_p2.iloc[i] + alpha)
        
        spreads.append(spread_value)
        alphas.append(alpha)
        betas.append(beta)
        r_squareds.append(r_squared)
        timestamps.append(log_p1.index[i])
    
    # Convert to Series
    spread_series = pd.Series(spreads, index=timestamps)
    alpha_series = pd.Series(alphas, index=timestamps)
    beta_series = pd.Series(betas, index=timestamps)
    r_squared_series = pd.Series(r_squareds, index=timestamps)
    
    return {
        'spread': spread_series,
        'alpha': alpha_series,
        'beta': beta_series,
        'r_squared': r_squared_series
    }

def calculate_rolling_stationarity_dynamic(dynamic_spread: pd.Series, name: str,
                                         test_window_days: int = 30,
                                         step_days: int = 1) -> Dict:
    """Calculate stationarity tests on dynamic spread"""
    log.info(f"Running rolling stationarity tests on dynamic spread ({test_window_days}-day test window)")
    
    samples_per_day = _infer_samples_per_day(dynamic_spread.index)
    test_window_points = int(test_window_days * samples_per_day)
    step_points = max(int(step_days * samples_per_day), 1)
    
    if len(dynamic_spread) < test_window_points + step_points:
        log.warning("Insufficient data for rolling stationarity tests")
        return {'error': 'Insufficient data'}
    
    results = {
        'test_window_days': test_window_days,
        'adf_pvalues': [],
        'kpss_pvalues': [],
        'timestamps': []
    }
    
    # Rolling tests
    for i in range(test_window_points, len(dynamic_spread), step_points):
        test_data = dynamic_spread.iloc[i-test_window_points:i]
        timestamp = dynamic_spread.index[i-1]
        
        # ADF test
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(test_data, autolag='AIC')
            results['adf_pvalues'].append(adf_p)
            results['timestamps'].append(timestamp)
        except:
            continue
        
        # KPSS test
        try:
            kpss_stat, kpss_p, _, _ = kpss(test_data, regression='c')
            results['kpss_pvalues'].append(kpss_p)
        except:
            results['kpss_pvalues'].append(np.nan)
    
    # Convert to Series
    if results['timestamps']:
        results['adf_pvalues'] = pd.Series(results['adf_pvalues'], index=results['timestamps'])
        results['kpss_pvalues'] = pd.Series(results['kpss_pvalues'], index=results['timestamps'])
        
        # Summary statistics
        adf_stationary_pct = (results['adf_pvalues'] < 0.05).mean() * 100
        kpss_stationary_pct = (results['kpss_pvalues'] > 0.05).mean() * 100
        
        results['summary'] = {
            'adf_stationary_pct': adf_stationary_pct,
            'kpss_stationary_pct': kpss_stationary_pct,
            'adf_latest_pvalue': results['adf_pvalues'].iloc[-1],
            'kpss_latest_pvalue': results['kpss_pvalues'].iloc[-1],
            'observations': len(results['adf_pvalues'])
        }
        
        log.info(f"  Dynamic spread ADF: {adf_stationary_pct:.1f}% of windows stationary")
        log.info(f"  Dynamic spread KPSS: {kpss_stationary_pct:.1f}% of windows stationary")
    
    return results

def calculate_half_life_dynamic(dynamic_spread: pd.Series) -> Dict:
    """Calculate half-life on dynamic spread"""
    log.info("Calculating half-life on dynamic spread...")
    
    spread_clean = dynamic_spread.dropna()
    if len(spread_clean) < 20:
        return {'error': 'Insufficient data'}
    
    # AR(1) regression
    spread_lag = spread_clean.shift(1)
    delta_spread = spread_clean.diff()
    
    valid_data = pd.DataFrame({
        'delta': delta_spread,
        'lag': spread_lag
    }).dropna()
    
    if len(valid_data) < 10:
        return {'error': 'Insufficient valid data'}
    
    # OLS regression
    X = np.column_stack([np.ones(len(valid_data)), valid_data['lag'].values])
    y = valid_data['delta'].values
    
    model = OLS(y, X).fit()
    alpha_hat, beta_hat = model.params
    
    # Calculate half-life
    if beta_hat >= 0:
        halflife_obs = np.inf
    else:
        halflife_obs = -np.log(2) / beta_hat
    
    # Convert to time units
    samples_per_day = _infer_samples_per_day(spread_clean.index)
    halflife_days = halflife_obs / samples_per_day if samples_per_day > 0 else np.inf
    halflife_hours = halflife_days * 24
    
    return {
        'ar1': {
            'alpha': alpha_hat,
            'beta': beta_hat,
            'r_squared': model.rsquared,
            'beta_pvalue': model.pvalues[1],
            'half_life_obs': halflife_obs,
            'half_life_days': halflife_days,
            'half_life_hours': halflife_hours,
        }
    }

def create_enhanced_visualization(symbol1: str, symbol2: str,
                                prices1: pd.Series, prices2: pd.Series,
                                dynamic_spreads: Dict[int, Dict],
                                rolling_stationarity: Dict[int, Dict],
                                half_life_results: Dict[int, Dict]):
    """Create visualization with dynamic spreads"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.5, 1, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    s1 = symbol1.split('_')[-2] if '_' in symbol1 else symbol1
    s2 = symbol2.split('_')[-2] if '_' in symbol2 else symbol2
    
    # 1. Price Series (Mark Prices)
    ax1 = fig.add_subplot(gs[0, :])
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{s1} Mark Price', color=color1)
    ax1.plot(prices1.index, prices1.values, color=color1, alpha=0.7, linewidth=1, label=s1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale('log')
    
    ax1_twin = ax1.twinx()
    color2 = 'tab:orange'
    ax1_twin.set_ylabel(f'{s2} Mark Price', color=color2)
    ax1_twin.plot(prices2.index, prices2.values, color=color2, alpha=0.7, linewidth=1, label=s2)
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.set_yscale('log')
    
    ax1.set_title(f'Mark Prices: {s1} vs {s2} (from database)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Dynamic Spreads Comparison
    ax2 = fig.add_subplot(gs[1, :])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(dynamic_spreads)))
    
    for i, (window_days, data) in enumerate(sorted(dynamic_spreads.items())):
        spread = data['spread']
        ax2.plot(spread.index, spread.values, alpha=0.7, linewidth=1, 
                color=colors[i], label=f'{window_days}d window')
    
    ax2.set_title('Dynamic Spreads (OLS Rolling)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Spread Value')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Beta Evolution
    ax3 = fig.add_subplot(gs[2, 0])
    
    for i, (window_days, data) in enumerate(sorted(dynamic_spreads.items())):
        beta = data['beta']
        ax3.plot(beta.index, beta.values, alpha=0.7, linewidth=1,
                color=colors[i], label=f'{window_days}d')
    
    ax3.set_title('Rolling Beta (Hedge Ratio)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Beta')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. R-squared Evolution
    ax4 = fig.add_subplot(gs[2, 1])
    
    for i, (window_days, data) in enumerate(sorted(dynamic_spreads.items())):
        r2 = data['r_squared']
        ax4.plot(r2.index, r2.values, alpha=0.7, linewidth=1,
                color=colors[i], label=f'{window_days}d')
    
    ax4.set_title('Rolling R² (Model Fit)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('R²')
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Z-Scores for main window (e.g., 30 days)
    ax5 = fig.add_subplot(gs[3, :])
    
    main_window = 30  # or choose the middle window
    if main_window in dynamic_spreads:
        spread = dynamic_spreads[main_window]['spread']
        
        # Calculate rolling statistics for z-score
        samples_per_day = _infer_samples_per_day(spread.index)
        window_samples = int(30 * samples_per_day)
        
        spread_mean = spread.rolling(window_samples).mean()
        spread_std = spread.rolling(window_samples).std()
        z_scores = (spread - spread_mean) / spread_std
        
        ax5.plot(z_scores.index, z_scores.values, 'b-', alpha=0.7, linewidth=1)
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax5.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='±2σ')
        ax5.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        ax5.axhline(y=3, color='darkred', linestyle=':', alpha=0.5, label='±3σ')
        ax5.axhline(y=-3, color='darkred', linestyle=':', alpha=0.5)
        ax5.fill_between(z_scores.index, -2, 2, alpha=0.1, color='green')
        
    ax5.set_title(f'Z-Score of Dynamic Spread ({main_window}d window)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Z-Score')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Rolling ADF p-values for different windows
    ax6 = fig.add_subplot(gs[4, 0])
    
    for window_days, stationarity_data in rolling_stationarity.items():
        if 'adf_pvalues' in stationarity_data and isinstance(stationarity_data['adf_pvalues'], pd.Series):
            adf_pvals = stationarity_data['adf_pvalues']
            ax6.plot(adf_pvals.index, adf_pvals.values, alpha=0.7, linewidth=1, 
                    label=f'{window_days}d spread')
    
    ax6.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% significance')
    ax6.axhline(y=0.01, color='darkred', linestyle=':', alpha=0.5, label='1% significance')
    ax6.fill_between(ax6.get_xlim(), 0, 0.05, alpha=0.2, color='green')
    ax6.set_title('Rolling ADF Tests on Dynamic Spreads', fontsize=12, fontweight='bold')
    ax6.set_ylabel('p-value')
    ax6.set_ylim(0, 0.5)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Half-life comparison
    ax7 = fig.add_subplot(gs[4, 1])
    
    windows = []
    half_lives = []
    
    for window_days, hl_data in half_life_results.items():
        if 'ar1' in hl_data:
            windows.append(window_days)
            half_lives.append(hl_data['ar1']['half_life_days'])
    
    if windows and half_lives:
        bars = ax7.bar(windows, half_lives, alpha=0.7, color='purple')
        ax7.set_xlabel('Window Size (days)')
        ax7.set_ylabel('Half-life (days)')
        ax7.set_title('Half-life by Window Size', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, hl in zip(bars, half_lives):
            if hl < np.inf:
                ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{hl:.1f}', ha='center', va='bottom')
    
    # Main title
    fig.suptitle(f'Dynamic Spread Analysis: {s1}/{s2} (Using Mark Prices)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.tight_layout()
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"{s1}_{s2}_dynamic_spread_analysis.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info(f"Dynamic spread visualization saved: {filename}")

def main():
    """Main function with dynamic spread analysis"""
    parser = argparse.ArgumentParser(description="Pair Trading Analysis with Dynamic Spreads")
    parser.add_argument("--symbol1", type=str, required=True)
    parser.add_argument("--symbol2", type=str, required=True)
    parser.add_argument("--windows", nargs='+', type=int, default=[7, 15, 30, 60, 90],
                       help="OLS rolling windows in days")
    parser.add_argument("--test-window", type=int, default=30,
                       help="Window for stationarity tests in days")
    parser.add_argument("--no-plots", action="store_true")
    
    args = parser.parse_args()
    
    log.info("🚀 Dynamic Spread Pair Trading Analysis")
    log.info(f"Analyzing pair: {args.symbol1} / {args.symbol2}")
    log.info(f"OLS windows: {args.windows} days")
    
    try:
        # Load mark prices data
        prices1, prices2 = load_mark_prices_data(args.symbol1, args.symbol2)
        if prices1.empty or prices2.empty:
            log.error("No mark prices data available")
            return False
        
        # Calculate log prices
        log_p1 = np.log(prices1)
        log_p2 = np.log(prices2)
        
        # Calculate dynamic spreads for each window
        log.info("\n" + "="*60)
        log.info("CALCULATING DYNAMIC SPREADS")
        log.info("="*60)
        
        dynamic_spreads = {}
        samples_per_day = _infer_samples_per_day(log_p1.index)
        
        for window_days in args.windows:
            window_samples = int(window_days * samples_per_day)
            
            if len(log_p1) < window_samples * 2:
                log.warning(f"Insufficient data for {window_days}-day window")
                continue
            
            log.info(f"\nProcessing {window_days}-day window ({window_samples} samples)...")
            dynamic_spreads[window_days] = calculate_dynamic_spread(log_p1, log_p2, window_samples)
            
            # Show statistics
            spread = dynamic_spreads[window_days]['spread']
            beta = dynamic_spreads[window_days]['beta']
            log.info(f"  Spread range: {spread.min():.4f} to {spread.max():.4f}")
            log.info(f"  Beta range: {beta.min():.4f} to {beta.max():.4f}")
            log.info(f"  Beta current: {beta.iloc[-1]:.4f}")
        
        # Run stationarity tests on dynamic spreads
        log.info("\n" + "="*60)
        log.info("STATIONARITY TESTS ON DYNAMIC SPREADS")
        log.info("="*60)
        
        rolling_stationarity = {}
        
        for window_days, spread_data in dynamic_spreads.items():
            log.info(f"\nTesting {window_days}-day dynamic spread...")
            rolling_stationarity[window_days] = calculate_rolling_stationarity_dynamic(
                spread_data['spread'], 
                f"{window_days}d dynamic spread",
                test_window_days=args.test_window
            )
        
        # Calculate half-life for each dynamic spread
        log.info("\n" + "="*60)
        log.info("HALF-LIFE ANALYSIS ON DYNAMIC SPREADS")
        log.info("="*60)
        
        half_life_results = {}
        
        for window_days, spread_data in dynamic_spreads.items():
            log.info(f"\nCalculating half-life for {window_days}-day spread...")
            hl_result = calculate_half_life_dynamic(spread_data['spread'])
            half_life_results[window_days] = hl_result
            
            if 'ar1' in hl_result:
                log.info(f"  Half-life: {hl_result['ar1']['half_life_days']:.2f} days")
                log.info(f"  R²: {hl_result['ar1']['r_squared']:.4f}")
        
        # Generate visualization
        if not args.no_plots:
            log.info("\n" + "="*60)
            log.info("CREATING VISUALIZATION")
            log.info("="*60)
            create_enhanced_visualization(
                args.symbol1, args.symbol2,
                prices1, prices2,
                dynamic_spreads,
                rolling_stationarity,
                half_life_results
            )
        
        # Summary report
        log.info("\n" + "="*60)
        log.info("DYNAMIC SPREAD ANALYSIS SUMMARY")
        log.info("="*60)
        
        for window_days in sorted(dynamic_spreads.keys()):
            log.info(f"\n{window_days}-day window:")
            
            # Current parameters
            current_beta = dynamic_spreads[window_days]['beta'].iloc[-1]
            current_alpha = dynamic_spreads[window_days]['alpha'].iloc[-1]
            log.info(f"  Current beta: {current_beta:.4f}")
            log.info(f"  Current alpha: {current_alpha:.4f}")
            
            # Stationarity
            if window_days in rolling_stationarity and 'summary' in rolling_stationarity[window_days]:
                summary = rolling_stationarity[window_days]['summary']
                log.info(f"  Stationarity: {summary['adf_stationary_pct']:.1f}% of periods")
            
            # Half-life
            if window_days in half_life_results and 'ar1' in half_life_results[window_days]:
                hl = half_life_results[window_days]['ar1']['half_life_days']
                log.info(f"  Half-life: {hl:.2f} days")
        
        log.info("\n✅ Dynamic spread analysis completed successfully!")
        return True
        
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)