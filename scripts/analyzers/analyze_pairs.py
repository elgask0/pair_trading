#!/usr/bin/env python3
"""
📊 Pair Analysis - FINAL VERSION WITH BEST GRAPHICS
Análisis optimizado con los mejores datos disponibles y visualizaciones profesionales
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
import argparse
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger

log = get_validation_logger()

# Enhanced plotting settings
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 100
sns.set_palette("husl")

def load_mark_prices_data(symbol: str) -> pd.DataFrame:
    """Cargar datos de mark prices optimizado"""
    with db_manager.get_session() as session:
        # Check available columns
        mp_columns_result = session.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'mark_prices'
        """)).fetchall()
        
        available_columns = [row[0] for row in mp_columns_result]
        
        # Build select based on available columns
        select_columns = ['timestamp', 'mark_price']
        if 'liquidity_score' in available_columns:
            select_columns.append('liquidity_score')
        if 'valid_for_trading' in available_columns:
            select_columns.append('valid_for_trading')
        elif 'is_valid' in available_columns:
            select_columns.append('is_valid as valid_for_trading')
        
        query = text(f"""
            SELECT {', '.join(select_columns)}
            FROM mark_prices 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params={'symbol': symbol}, index_col='timestamp')
        log.info(f"Loaded {len(df):,} mark_prices records for {symbol}")
        return df

def create_minute_alignment_optimized(df1: pd.DataFrame, df2: pd.DataFrame, s1: str, s2: str) -> Tuple[pd.Series, pd.Series]:
    """Alineamiento optimizado para mejores resultados"""
    log.info(f"\n🔄 OPTIMIZED ALIGNMENT: {s1} vs {s2}")
    
    prices1 = df1['mark_price']
    prices2 = df2['mark_price']
    
    # Find temporal overlap
    start_time = max(prices1.index.min(), prices2.index.min())
    end_time = min(prices1.index.max(), prices2.index.max())
    
    # Filter to overlap period
    overlap1 = prices1[(prices1.index >= start_time) & (prices1.index <= end_time)]
    overlap2 = prices2[(prices2.index >= start_time) & (prices2.index <= end_time)]
    
    log.info(f"  Overlap period: {start_time.date()} to {end_time.date()}")
    log.info(f"  Data in overlap: {s1}={len(overlap1):,}, {s2}={len(overlap2):,}")
    
    # Create minute grid
    minute_grid = pd.date_range(
        start=start_time.floor('T'), 
        end=end_time.ceil('T'), 
        freq='T'
    )
    
    # Round to minutes and group
    rounded1 = overlap1.copy()
    rounded2 = overlap2.copy()
    rounded1.index = rounded1.index.round('T')
    rounded2.index = rounded2.index.round('T')
    
    # Group by minute (use last value if multiple)
    grouped1 = rounded1.groupby(rounded1.index).last()
    grouped2 = rounded2.groupby(rounded2.index).last()
    
    # Reindex to full grid
    grid1 = grouped1.reindex(minute_grid)
    grid2 = grouped2.reindex(minute_grid)
    
    # Smart filling strategy
    filled1 = grid1.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
    filled2 = grid2.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
    
    # Interpolate small gaps
    filled1 = filled1.interpolate(method='linear', limit=10)
    filled2 = filled2.interpolate(method='linear', limit=10)
    
    # Final forward/backward fill
    filled1 = filled1.fillna(method='ffill', limit=15).fillna(method='bfill', limit=15)
    filled2 = filled2.fillna(method='ffill', limit=15).fillna(method='bfill', limit=15)
    
    # Find valid data points
    both_valid = filled1.notna() & filled2.notna()
    final_aligned1 = filled1[both_valid]
    final_aligned2 = filled2[both_valid]
    
    log.info(f"  Final aligned points: {len(final_aligned1):,}")
    log.info(f"  Coverage: {len(final_aligned1)/len(minute_grid)*100:.1f}% of possible minutes")
    log.info(f"  Data utilization: {len(final_aligned1)/max(len(df1), len(df2))*100:.1f}% of original data")
    
    return final_aligned1, final_aligned2

def calculate_comprehensive_stats(prices1: pd.Series, prices2: pd.Series, s1: str, s2: str) -> Dict:
    """Calcular estadísticas comprehensivas para trading"""
    log.info(f"\n📈 COMPREHENSIVE STATISTICS: {s1} vs {s2}")
    
    if len(prices1) != len(prices2) or len(prices1) < 10:
        log.error(f"Insufficient data for analysis")
        return {}
    
    # Clean data
    valid_data = pd.DataFrame({'price1': prices1, 'price2': prices2}).dropna()
    log.info(f"Using {len(valid_data):,} clean data points")
    
    # Basic correlation
    correlation = valid_data['price1'].corr(valid_data['price2'])
    
    # Linear regression
    X = valid_data['price2'].values.reshape(-1, 1)
    y = valid_data['price1'].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    alpha = reg.intercept_
    beta = reg.coef_[0]
    r_squared = reg.score(X, y)
    predictions = reg.predict(X)
    
    # Spread analysis
    spread = y - predictions
    spread_series = pd.Series(spread, index=valid_data.index)
    
    # Z-score calculation
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std
    z_score_series = pd.Series(z_score, index=valid_data.index)
    
    # Advanced statistics
    # Cointegration test (simplified)
    from scipy.stats import jarque_bera, normaltest
    
    # Test spread normality
    jb_stat, jb_pvalue = jarque_bera(spread)
    norm_stat, norm_pvalue = normaltest(spread)
    
    # Half-life calculation (mean reversion)
    spread_lagged = spread[:-1]
    spread_diff = np.diff(spread)
    half_life_reg = LinearRegression()
    half_life_reg.fit(spread_lagged.reshape(-1, 1), spread_diff)
    lambda_param = -half_life_reg.coef_[0]
    half_life = np.log(2) / lambda_param if lambda_param > 0 else np.inf
    
    # Trading signal analysis
    current_z = z_score[-1]
    
    # Signal classification
    if abs(current_z) > 3:
        signal_strength = "VERY STRONG"
    elif abs(current_z) > 2:
        signal_strength = "STRONG"
    elif abs(current_z) > 1:
        signal_strength = "MODERATE"
    else:
        signal_strength = "WEAK"
    
    signal_direction = "LONG spread (short GIGA, long SPX)" if current_z < -1 else \
                      "SHORT spread (long GIGA, short SPX)" if current_z > 1 else \
                      "NEUTRAL"
    
    results = {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'correlation': correlation,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'current_z_score': current_z,
        'half_life_days': half_life / (24 * 60) if half_life != np.inf else np.inf,  # Convert minutes to days
        'spread_normality_pvalue': norm_pvalue,
        'signal_strength': signal_strength,
        'signal_direction': signal_direction,
        'spread': spread_series,
        'z_score': z_score_series,
        'aligned_prices1': pd.Series(valid_data['price1'], index=valid_data.index),
        'aligned_prices2': pd.Series(valid_data['price2'], index=valid_data.index),
        'predictions': pd.Series(predictions, index=valid_data.index),
        'data_points': len(valid_data)
    }
    
    # Log key statistics
    log.info(f"  Regression: {s1} = {alpha:.6f} + {beta:.6f} × {s2}")
    log.info(f"  R²: {r_squared:.4f}, Correlation: {correlation:.4f}")
    log.info(f"  Current Z-score: {current_z:.2f}")
    log.info(f"  Half-life: {results['half_life_days']:.1f} days")
    log.info(f"  Signal: {signal_strength} {signal_direction}")
    
    return results

def create_professional_charts(df1: pd.DataFrame, df2: pd.DataFrame, stats: Dict, s1: str, s2: str):
    """Crear gráficos profesionales de alta calidad"""
    log.info(f"\n📊 Creating professional charts for {s1} vs {s2}")
    
    # Create figure with optimized layout
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                          height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1])
    
    # Color scheme
    color1 = '#2E86AB'  # Blue
    color2 = '#F24236'  # Red
    color_spread = '#A23B72'  # Purple
    color_zscore = '#F18F01'  # Orange
    
    # 1. TIME SERIES WITH DUAL AXES (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot first symbol
    line1 = ax1.plot(df1.index, df1['mark_price'], 
                     color=color1, linewidth=1.2, alpha=0.9, label=f'{s1}')
    ax1.set_ylabel(f'{s1} Price (USD)', color=color1, fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot second symbol on twin axis
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(df2.index, df2['mark_price'], 
                          color=color2, linewidth=1.2, alpha=0.9, label=f'{s2}')
    ax1_twin.set_ylabel(f'{s2} Price (USD)', color=color2, fontweight='bold', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
    
    ax1.set_title(f'Price Evolution: {s1} vs {s2}', fontweight='bold', fontsize=14)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. SCATTER PLOT WITH REGRESSION (Top Center)
    ax2 = fig.add_subplot(gs[0, 1])
    
    aligned_p1 = stats['aligned_prices1']
    aligned_p2 = stats['aligned_prices2']
    predictions = stats['predictions']
    
    # Scatter plot with density coloring
    scatter = ax2.scatter(aligned_p2, aligned_p1, 
                         c=np.arange(len(aligned_p1)), cmap='viridis', 
                         alpha=0.6, s=15, edgecolors='none')
    
    # Regression line
    ax2.plot(aligned_p2, predictions, color=color2, linewidth=3, 
             label='Regression Line', alpha=0.8)
    
    ax2.set_xlabel(f'{s2} Price (USD)', fontweight='bold')
    ax2.set_ylabel(f'{s1} Price (USD)', fontweight='bold')
    ax2.set_title(f'Price Relationship: {s1} vs {s2}', fontweight='bold', fontsize=14)
    
    # Add equation and statistics
    equation_text = (f'{s1} = {stats["alpha"]:.4f} + {stats["beta"]:.4f} × {s2}\n'
                    f'R² = {stats["r_squared"]:.4f}\n'
                    f'Correlation = {stats["correlation"]:.4f}\n'
                    f'Data Points: {stats["data_points"]:,}')
    
    ax2.text(0.05, 0.95, equation_text, transform=ax2.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
            verticalalignment='top', fontsize=11, fontweight='bold')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. SPREAD ANALYSIS (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    spread = stats['spread']
    spread_mean = stats['spread_mean']
    spread_std = stats['spread_std']
    
    ax3.plot(spread.index, spread.values, color=color_spread, linewidth=1.5, alpha=0.8)
    ax3.axhline(y=spread_mean, color='black', linestyle='-', alpha=0.8, 
                label=f'Mean: {spread_mean:.4f}')
    ax3.axhline(y=spread_mean + 2*spread_std, color=color2, linestyle='--', alpha=0.7, 
                label=f'+2σ: {spread_mean + 2*spread_std:.4f}')
    ax3.axhline(y=spread_mean - 2*spread_std, color=color2, linestyle='--', alpha=0.7, 
                label=f'-2σ: {spread_mean - 2*spread_std:.4f}')
    
    # Fill confidence intervals
    ax3.fill_between(spread.index, 
                     spread_mean - 2*spread_std, 
                     spread_mean + 2*spread_std, 
                     alpha=0.15, color=color2)
    
    ax3.set_ylabel('Spread (Actual - Predicted)', fontweight='bold')
    ax3.set_title(f'Regression Spread Analysis', fontweight='bold', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Z-SCORE WITH TRADING SIGNALS (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    
    z_score = stats['z_score']
    
    # Plot z-score
    ax4.plot(z_score.index, z_score.values, color=color_zscore, linewidth=1.5, alpha=0.8)
    
    # Trading level lines
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax4.axhline(y=2, color=color2, linestyle='--', alpha=0.8, label='Entry Level ±2σ')
    ax4.axhline(y=-2, color=color2, linestyle='--', alpha=0.8)
    ax4.axhline(y=3, color='darkred', linestyle=':', alpha=0.8, label='Strong Signal ±3σ')
    ax4.axhline(y=-3, color='darkred', linestyle=':', alpha=0.8)
    
    # Fill zones
    ax4.fill_between(z_score.index, -2, 2, alpha=0.1, color='green', label='Normal Zone')
    ax4.fill_between(z_score.index, 2, 3, alpha=0.15, color='orange', label='Entry Zone')
    ax4.fill_between(z_score.index, -3, -2, alpha=0.15, color='orange')
    ax4.fill_between(z_score.index[z_score > 3], 3, z_score[z_score > 3], 
                     alpha=0.2, color='red', label='Strong Signal')
    ax4.fill_between(z_score.index[z_score < -3], z_score[z_score < -3], -3, 
                     alpha=0.2, color='red')
    
    ax4.set_ylabel('Z-Score', fontweight='bold')
    ax4.set_title(f'Trading Signals (Current: {stats["current_z_score"]:.2f})', 
                  fontweight='bold', fontsize=14)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # 5. SPREAD DISTRIBUTION (Middle Center)
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Histogram of spread
    ax5.hist(spread.values, bins=50, density=True, alpha=0.7, color=color_spread, 
             edgecolor='black', linewidth=0.5)
    
    # Overlay normal distribution
    x_norm = np.linspace(spread.min(), spread.max(), 100)
    y_norm = stats_scipy.norm.pdf(x_norm, spread_mean, spread_std)
    ax5.plot(x_norm, y_norm, color='red', linewidth=2, label='Normal Distribution')
    
    ax5.axvline(x=spread_mean, color='black', linestyle='-', alpha=0.8, label='Mean')
    ax5.axvline(x=spread_mean + 2*spread_std, color=color2, linestyle='--', alpha=0.7)
    ax5.axvline(x=spread_mean - 2*spread_std, color=color2, linestyle='--', alpha=0.7)
    
    ax5.set_xlabel('Spread Value', fontweight='bold')
    ax5.set_ylabel('Density', fontweight='bold')
    ax5.set_title(f'Spread Distribution\n(Normality p-value: {stats["spread_normality_pvalue"]:.4f})', 
                  fontweight='bold', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. ROLLING CORRELATION (Middle Right)
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Calculate rolling correlation
    window = min(1440, len(aligned_p1) // 10)  # 1 day or 10% of data
    rolling_corr = aligned_p1.rolling(window=window).corr(aligned_p2)
    
    ax6.plot(rolling_corr.index, rolling_corr.values, color=color1, linewidth=1.5)
    ax6.axhline(y=0.7, color=color2, linestyle='--', alpha=0.7, label='Strong Correlation (0.7)')
    ax6.axhline(y=stats['correlation'], color='black', linestyle='-', alpha=0.8, 
                label=f'Overall: {stats["correlation"]:.3f}')
    
    ax6.set_ylabel('Rolling Correlation', fontweight='bold')
    ax6.set_title(f'Correlation Stability\n(Window: {window} points)', fontweight='bold', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # 7. TRADING SUMMARY TABLE (Bottom)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create comprehensive summary table
    summary_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Data Points', f'{stats["data_points"]:,}', 'Amount of aligned data'],
        ['R-Squared', f'{stats["r_squared"]:.4f}', 'Explains ' + f'{stats["r_squared"]*100:.1f}% of variance'],
        ['Correlation', f'{stats["correlation"]:.4f}', 'Strong' if stats["correlation"] > 0.7 else 'Moderate'],
        ['Current Z-Score', f'{stats["current_z_score"]:.2f}', stats["signal_strength"]],
        ['Half-Life', f'{stats["half_life_days"]:.1f} days', 'Mean reversion speed'],
        ['Signal Direction', stats["signal_direction"], 'Trading recommendation'],
        ['Spread Std Dev', f'{stats["spread_std"]:.6f}', 'Volatility measure'],
        ['Beta Coefficient', f'{stats["beta"]:.4f}', 'Price sensitivity'],
        ['Alpha Intercept', f'{stats["alpha"]:.6f}', 'Systematic difference']
    ]
    
    # Create table
    table = ax7.table(cellText=summary_data[1:], colLabels=summary_data[0],
                      cellLoc='center', loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code based on values
    for i in range(1, len(summary_data)):
        if 'Strong' in summary_data[i][2] or 'STRONG' in summary_data[i][1]:
            table[(i, 2)].set_facecolor('#2ecc71')
            table[(i, 2)].set_text_props(weight='bold', color='white')
        elif 'VERY STRONG' in summary_data[i][1]:
            table[(i, 1)].set_facecolor('#e74c3c')
            table[(i, 1)].set_text_props(weight='bold', color='white')
    
    ax7.set_title(f'Trading Analysis Summary: {s1} / {s2}', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle(f'Complete Pair Trading Analysis: {s1} / {s2}', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save high-quality plot
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"{s1}_{s2}_professional_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    log.info(f"Professional analysis chart saved: {filename}")
    
    return filename

def create_complete_analysis(symbol1: str, symbol2: str):
    """Análisis completo con gráficos profesionales"""
    
    log.info(f"Creating complete professional analysis for {symbol1} / {symbol2}")
    
    s1 = symbol1.split('_')[-2] if '_' in symbol1 else symbol1
    s2 = symbol2.split('_')[-2] if '_' in symbol2 else symbol2
    
    # Load data
    log.info(f"Loading mark prices data...")
    df_marks1 = load_mark_prices_data(symbol1)
    df_marks2 = load_mark_prices_data(symbol2)
    
    if df_marks1.empty or df_marks2.empty:
        log.error("No mark prices data available")
        return
    
    log.info(f"Data loaded: {s1}={len(df_marks1):,}, {s2}={len(df_marks2):,}")
    
    # Optimize alignment
    aligned_prices1, aligned_prices2 = create_minute_alignment_optimized(df_marks1, df_marks2, s1, s2)
    
    if aligned_prices1.empty or aligned_prices2.empty:
        log.error("No aligned data available")
        return
    
    # Calculate comprehensive statistics
    stats = calculate_comprehensive_stats(aligned_prices1, aligned_prices2, s1, s2)
    
    if not stats:
        log.error("Could not calculate statistics")
        return
    
    # Create professional charts
    chart_file = create_professional_charts(df_marks1, df_marks2, stats, s1, s2)
    
    # Final summary
    log.info(f"\n🎉 PROFESSIONAL ANALYSIS COMPLETE!")
    log.info(f"📊 Key Results:")
    log.info(f"   • Data Points: {stats['data_points']:,}")
    log.info(f"   • R²: {stats['r_squared']:.4f}")
    log.info(f"   • Correlation: {stats['correlation']:.4f}")
    log.info(f"   • Current Signal: {stats['signal_strength']} {stats['signal_direction']}")
    log.info(f"   • Half-life: {stats['half_life_days']:.1f} days")
    log.info(f"📈 Chart saved: {chart_file}")
    
    return stats

# Add scipy.stats import at the top
from scipy import stats as stats_scipy

def main():
    """Función principal para análisis profesional"""
    parser = argparse.ArgumentParser(description="Professional Pair Analysis with Best Graphics")
    parser.add_argument("--symbol1", type=str, required=True)
    parser.add_argument("--symbol2", type=str, required=True)
    
    args = parser.parse_args()
    
    log.info("🎨 Starting PROFESSIONAL Pair Analysis with Best Graphics")
    log.info(f"Symbols: {args.symbol1} / {args.symbol2}")
    
    try:
        create_complete_analysis(args.symbol1, args.symbol2)
        
        log.info("✅ Professional analysis with best graphics completed!")
        log.info("🖼️ Check plots/ directory for high-quality visualizations")
        return True
        
    except Exception as e:
        log.error(f"❌ Analysis failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)