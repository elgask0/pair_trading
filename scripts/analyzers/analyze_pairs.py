#!/usr/bin/env python3
"""
Advanced Pair Trading Analysis Script
Analyzes statistical relationships between two trading symbols with comprehensive metrics
"""

import argparse
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sqlalchemy import text

# Database imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.database.connection import db_manager 
from src.utils.logger import setup_logger

# Setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
logger = setup_logger(__name__)

# Try to import advanced statistical packages
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from arch.unitroot import PhillipsPerron
    ADVANCED_STATS = True
    logger.info("Advanced statistical packages loaded successfully")
except ImportError:
    ADVANCED_STATS = False
    logger.warning("Advanced statistical packages not available. Install with: pip install statsmodels arch")

def create_analysis_directories(base_dir: str) -> dict:
    """Create directory structure for analysis outputs"""
    directories = {
        'base': base_dir,
        'overview': os.path.join(base_dir, 'overview'),
        'windows': os.path.join(base_dir, 'windows'),
        'comparison': os.path.join(base_dir, 'comparison'),
        'advanced': os.path.join(base_dir, 'advanced'),
        'correlation': os.path.join(base_dir, 'correlation'),
        'residuals': os.path.join(base_dir, 'residuals')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Create subdirectories for different window sizes
    for window in [3, 7, 15, 30, 60, 90]:
        window_dir = os.path.join(directories['windows'], f'{window}d')
        os.makedirs(window_dir, exist_ok=True)
    
    logger.info(f"Created analysis directory structure in: {base_dir}")
    return directories

def load_mark_prices_data(symbol: str, limit: int = None) -> pd.DataFrame:
    """Load mark prices data for a symbol - FIXED VERSION"""
    try:
        with db_manager.get_session() as session:
            query = """
            SELECT timestamp, mark_price
            FROM mark_prices 
            WHERE symbol = :symbol 
            ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Use pandas to read SQL directly
            df = pd.read_sql(
                text(query), 
                session.bind, 
                params={'symbol': symbol},
                index_col='timestamp'
            )
            
            if df.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return df
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            df = df.sort_index()
            
            logger.info(f"Loaded {len(df):,} mark_prices records for {symbol}")
            return df
            
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def create_minute_alignment_optimized(df1: pd.DataFrame, df2: pd.DataFrame, 
                                    symbol1: str, symbol2: str) -> pd.DataFrame:
    """Create optimized minute-level alignment of two price series"""
    try:
        logger.info(f"\n🔄 OPTIMIZED ALIGNMENT: {symbol1.split('_')[-2]} vs {symbol2.split('_')[-2]}")
        
        # Find overlap period
        start_time = max(df1.index.min(), df2.index.min())
        end_time = min(df1.index.max(), df2.index.max())
        
        logger.info(f"  Overlap period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        # Filter to overlap period
        df1_overlap = df1[(df1.index >= start_time) & (df1.index <= end_time)].copy()
        df2_overlap = df2[(df2.index >= start_time) & (df2.index <= end_time)].copy()
        
        logger.info(f"  Data in overlap: {symbol1.split('_')[-2]}={len(df1_overlap):,}, {symbol2.split('_')[-2]}={len(df2_overlap):,}")
        
        # Create minute-level index
        minute_index = pd.date_range(start=start_time, end=end_time, freq='1T')
        
        # Resample to minute frequency using forward fill
        df1_resampled = df1_overlap.resample('1T').last().reindex(minute_index, method='ffill')
        df2_resampled = df2_overlap.resample('1T').last().reindex(minute_index, method='ffill')
        
        # Combine and drop NaN values
        combined = pd.DataFrame({
            f'{symbol1.split("_")[-2]}_price': df1_resampled['mark_price'],
            f'{symbol2.split("_")[-2]}_price': df2_resampled['mark_price']
        })
        
        # Drop rows where either price is NaN
        combined_clean = combined.dropna()
        
        if combined_clean.empty:
            logger.error("No overlapping data points found after alignment")
            return pd.DataFrame()
        
        # Calculate statistics
        total_possible_minutes = len(minute_index)
        coverage_pct = (len(combined_clean) / total_possible_minutes) * 100
        utilization_pct = (len(combined_clean) / (len(df1_overlap) + len(df2_overlap))) * 200  # *200 because we're comparing against sum of both
        
        logger.info(f"  Final aligned points: {len(combined_clean):,}")
        logger.info(f"  Coverage: {coverage_pct:.1f}% of possible minutes")
        logger.info(f"  Data utilization: {utilization_pct:.1f}% of original data")
        
        return combined_clean
        
    except Exception as e:
        logger.error(f"Error in alignment: {e}")
        return pd.DataFrame()

def calculate_comprehensive_stats(df: pd.DataFrame, symbol1: str, symbol2: str) -> dict:
    """Calculate comprehensive statistical measures"""
    try:
        logger.info(f"\n📈 COMPREHENSIVE STATISTICS: {symbol1.split('_')[-2]} vs {symbol2.split('_')[-2]}")
        logger.info(f"Using {len(df):,} clean data points")
        
        if len(df) < 100:
            logger.warning("Insufficient data for reliable statistics")
            return {}
        
        # Get column names
        cols = df.columns.tolist()
        x_col, y_col = cols[0], cols[1]  # First symbol, Second symbol
        
        x = df[x_col].values
        y = df[y_col].values
        
        # Basic statistics
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Linear regression: y = alpha + beta * x
        X = x.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        alpha = reg.intercept_
        beta = reg.coef_[0]
        r_squared = r2_score(y, reg.predict(X))
        
        # Calculate spread and residuals
        spread = y - (alpha + beta * x)
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        # Current values and z-score
        current_spread = spread[-1]  # Use array indexing instead of iloc
        zscore_current = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Half-life calculation (Ornstein-Uhlenbeck mean reversion)
        try:
            spread_series = pd.Series(spread)
            spread_lag = spread_series.shift(1).dropna()
            spread_diff = spread_series.diff().dropna()
            
            if len(spread_lag) > 0 and len(spread_diff) > 0:
                # Align the series
                min_len = min(len(spread_lag), len(spread_diff))
                spread_lag_aligned = spread_lag.iloc[:min_len]
                spread_diff_aligned = spread_diff.iloc[-min_len:]
                
                # Calculate half-life
                reg_hl = LinearRegression().fit(spread_lag_aligned.values.reshape(-1, 1), spread_diff_aligned.values)
                lambda_coef = reg_hl.coef_[0]
                half_life = -np.log(2) / lambda_coef if lambda_coef < 0 else float('inf')
                half_life_days = half_life / (24 * 60)  # Convert minutes to days
            else:
                half_life_days = float('inf')
        except:
            half_life_days = float('inf')
        
        # Trading signal based on z-score
        if abs(zscore_current) < 1:
            signal = "NEUTRAL"
            signal_strength = "WEAK"
        elif abs(zscore_current) < 2:
            signal = "LONG spread" if zscore_current < 0 else "SHORT spread"
            signal_strength = "MODERATE"
        else:
            signal = "LONG spread" if zscore_current < 0 else "SHORT spread"
            signal_strength = "VERY STRONG"
        
        # Log key results
        logger.info(f"  Regression: {y_col.split('_')[0]} = {alpha:.6f} + {beta:.6f} × {x_col.split('_')[0]}")
        logger.info(f"  R²: {r_squared:.4f}, Correlation: {correlation:.4f}")
        logger.info(f"  Current Z-score: {zscore_current:.2f}")
        logger.info(f"  Half-life: {half_life_days:.1f} days")
        logger.info(f"  Signal: {signal_strength} {signal}")
        
        results = {
            'correlation': correlation,
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'zscore_current': zscore_current,
            'half_life': half_life_days,
            'signal': signal,
            'signal_strength': signal_strength,
            'spread': spread,
            'x_values': x,
            'y_values': y,
            'regression_line': alpha + beta * x
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating comprehensive stats: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}

def calculate_rolling_stats(df: pd.DataFrame, window_days: int, symbol1: str, symbol2: str) -> dict:
    """Calculate rolling statistics for a given window"""
    try:
        window_minutes = window_days * 24 * 60
        
        if len(df) < window_minutes:
            logger.warning(f"Not enough data for {window_days}-day window")
            return None
        
        # Get column names
        cols = df.columns.tolist()
        x_col, y_col = cols[0], cols[1]
        
        # Calculate rolling correlations
        rolling_corr = df[x_col].rolling(window=window_minutes).corr(df[y_col])
        
        # Calculate rolling regression coefficients
        def rolling_regression(window_data):
            if len(window_data) < 10:  # Minimum data points
                return pd.Series([np.nan, np.nan, np.nan], index=['alpha', 'beta', 'r_squared'])
            
            x = window_data[x_col].values
            y = window_data[y_col].values
            
            try:
                X = x.reshape(-1, 1)
                reg = LinearRegression().fit(X, y)
                alpha = reg.intercept_
                beta = reg.coef_[0]
                r_squared = r2_score(y, reg.predict(X))
                return pd.Series([alpha, beta, r_squared], index=['alpha', 'beta', 'r_squared'])
            except:
                return pd.Series([np.nan, np.nan, np.nan], index=['alpha', 'beta', 'r_squared'])
        
        rolling_reg = df.rolling(window=window_minutes).apply(rolling_regression)
        
        # Calculate rolling z-scores
        def calculate_rolling_zscore(window_data):
            if len(window_data) < 10:
                return np.nan
            
            x = window_data[x_col].values
            y = window_data[y_col].values
            
            try:
                # Get regression parameters for this window
                X = x.reshape(-1, 1)
                reg = LinearRegression().fit(X, y)
                alpha = reg.intercept_
                beta = reg.coef_[0]
                
                # Calculate spread
                spread = y - (alpha + beta * x)
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                
                # Current z-score (last value in window)
                current_spread = spread[-1]
                zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
                return zscore
            except:
                return np.nan
        
        rolling_zscore = df.rolling(window=window_minutes).apply(calculate_rolling_zscore)
        
        # Clean up the results
        results = {
            'correlation': rolling_corr.dropna(),
            'alpha': rolling_reg['alpha'].dropna() if 'alpha' in rolling_reg.columns else pd.Series(dtype=float),
            'beta': rolling_reg['beta'].dropna() if 'beta' in rolling_reg.columns else pd.Series(dtype=float),
            'r_squared': rolling_reg['r_squared'].dropna() if 'r_squared' in rolling_reg.columns else pd.Series(dtype=float),
            'zscore': rolling_zscore.dropna(),
            'window_days': window_days
        }
        
        # Add summary statistics
        if len(results['correlation']) > 0:
            results['correlation_mean'] = results['correlation'].mean()
            results['correlation_std'] = results['correlation'].std()
            results['correlation_current'] = results['correlation'].iloc[-1]
        
        if len(results['zscore']) > 0:
            results['zscore_mean'] = results['zscore'].mean()
            results['zscore_std'] = results['zscore'].std()
            results['zscore_current'] = results['zscore'].iloc[-1]
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating rolling stats for {window_days}d window: {e}")
        return None

def calculate_hurst_exponent(spread_series):
    """Calculate Hurst exponent for mean reversion analysis"""
    try:
        # Convert to numpy array if pandas Series
        if hasattr(spread_series, 'values'):
            spread_array = spread_series.values
        else:
            spread_array = np.array(spread_series)
        
        # Remove any NaN values
        spread_clean = spread_array[~np.isnan(spread_array)]
        
        if len(spread_clean) < 100:
            return np.nan
        
        lags = range(2, min(100, len(spread_clean)//4))
        tau = []
        
        for lag in lags:
            # Calculate the variance of the differences
            pp = np.subtract(spread_clean[lag:], spread_clean[:-lag])
            tau.append(np.sqrt(np.std(pp)))
        
        # Linear regression on log-log plot
        tau = np.array(tau)
        lags = np.array(lags)
        
        # Remove any invalid values
        valid_idx = (tau > 0) & (lags > 0) & np.isfinite(tau) & np.isfinite(lags)
        if np.sum(valid_idx) < 10:
            return np.nan
        
        log_lags = np.log(lags[valid_idx])
        log_tau = np.log(tau[valid_idx])
        
        # Fit line
        coeffs = np.polyfit(log_lags, log_tau, 1)
        hurst = coeffs[0]
        
        return hurst
        
    except Exception as e:
        logger.warning(f"Error calculating Hurst exponent: {e}")
        return np.nan

def calculate_variance_ratio(spread_series, k=2):
    """Calculate variance ratio test for mean reversion"""
    try:
        if hasattr(spread_series, 'values'):
            spread_array = spread_series.values
        else:
            spread_array = np.array(spread_series)
        
        spread_clean = spread_array[~np.isnan(spread_array)]
        
        if len(spread_clean) < k * 10:
            return np.nan
        
        # Calculate returns
        returns = np.diff(spread_clean)
        n = len(returns)
        
        if n < k * 5:
            return np.nan
        
        # Variance of 1-period returns
        var_1 = np.var(returns, ddof=1)
        
        # Variance of k-period returns
        k_returns = []
        for i in range(0, n - k + 1, k):
            k_return = np.sum(returns[i:i+k])
            k_returns.append(k_return)
        
        if len(k_returns) < 5:
            return np.nan
        
        var_k = np.var(k_returns, ddof=1)
        
        # Variance ratio
        vr = (var_k / k) / var_1 if var_1 > 0 else np.nan
        
        return vr
        
    except Exception as e:
        logger.warning(f"Error calculating variance ratio: {e}")
        return np.nan

def calculate_advanced_spread_metrics(df: pd.DataFrame, window_days: int) -> dict:
    """Calculate advanced spread metrics"""
    try:
        logger.info(f"Calculating advanced metrics for {window_days}-day window...")
        
        window_minutes = window_days * 24 * 60
        
        if len(df) < window_minutes * 2:  # Need more data for advanced metrics
            return {}
        
        # Get the most recent window of data
        recent_data = df.tail(window_minutes)
        
        if len(recent_data) < 100:
            return {}
        
        # Get column names
        cols = df.columns.tolist()
        x_col, y_col = cols[0], cols[1]
        
        # Calculate basic regression for spread
        x = recent_data[x_col].values
        y = recent_data[y_col].values
        
        X = x.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        alpha = reg.intercept_
        beta = reg.coef_[0]
        
        # Calculate spread
        spread = y - (alpha + beta * x)
        
        # Advanced metrics
        metrics = {}
        
        # Hurst exponent
        metrics['hurst_exponent'] = calculate_hurst_exponent(spread)
        
        # Variance ratio
        metrics['variance_ratio_2'] = calculate_variance_ratio(spread, k=2)
        metrics['variance_ratio_4'] = calculate_variance_ratio(spread, k=4)
        
        # Half-life estimation
        try:
            spread_series = pd.Series(spread)
            spread_lag = spread_series.shift(1).dropna()
            spread_diff = spread_series.diff().dropna()
            
            if len(spread_lag) > 10 and len(spread_diff) > 10:
                min_len = min(len(spread_lag), len(spread_diff))
                spread_lag_aligned = spread_lag.iloc[:min_len]
                spread_diff_aligned = spread_diff.iloc[-min_len:]
                
                reg_hl = LinearRegression().fit(spread_lag_aligned.values.reshape(-1, 1), spread_diff_aligned.values)
                lambda_coef = reg_hl.coef_[0]
                half_life = -np.log(2) / lambda_coef if lambda_coef < 0 else float('inf')
                metrics['half_life'] = half_life / (24 * 60)  # Convert to days
            else:
                metrics['half_life'] = np.nan
        except:
            metrics['half_life'] = np.nan
        
        # Spread statistics
        metrics['spread_skewness'] = stats.skew(spread)
        metrics['spread_kurtosis'] = stats.kurtosis(spread)
        metrics['spread_jarque_bera'] = stats.jarque_bera(spread)[0]
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating advanced spread metrics: {e}")
        return {}

def create_overview_figure(df: pd.DataFrame, stats: dict, symbol1: str, symbol2: str, output_path: str):
    """Create comprehensive overview figure"""
    try:
        logger.info(f"Creating overview figure for {symbol1.split('_')[-2]} vs {symbol2.split('_')[-2]}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get column names
        cols = df.columns.tolist()
        x_col, y_col = cols[0], cols[1]
        s1_name = symbol1.split('_')[-2]
        s2_name = symbol2.split('_')[-2]
        
        # 1. Price Time Series (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(df.index, df[x_col], label=f'{s1_name} Price', color='blue', alpha=0.7)
        line2 = ax1_twin.plot(df.index, df[y_col], label=f'{s2_name} Price', color='red', alpha=0.7)
        
        ax1.set_ylabel(f'{s1_name} Price', color='blue')
        ax1_twin.set_ylabel(f'{s2_name} Price', color='red')
        ax1.set_title('Price Time Series', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 2. Scatter Plot with Regression (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.scatter(stats['x_values'], stats['y_values'], alpha=0.5, s=1)
        ax2.plot(stats['x_values'], stats['regression_line'], color='red', linewidth=2)
        ax2.set_xlabel(f'{s1_name} Price')
        ax2.set_ylabel(f'{s2_name} Price')
        ax2.set_title(f'Regression: R² = {stats["r_squared"]:.4f}', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Spread Time Series (second row, spans all columns)
        ax3 = fig.add_subplot(gs[1, :])
        spread_series = pd.Series(stats['spread'], index=df.index)
        ax3.plot(spread_series.index, spread_series.values, color='green', alpha=0.7)
        ax3.axhline(y=stats['spread_mean'], color='black', linestyle='--', alpha=0.5, label='Mean')
        ax3.axhline(y=stats['spread_mean'] + stats['spread_std'], color='red', linestyle='--', alpha=0.5, label='+1σ')
        ax3.axhline(y=stats['spread_mean'] - stats['spread_std'], color='red', linestyle='--', alpha=0.5, label='-1σ')
        ax3.axhline(y=stats['spread_mean'] + 2*stats['spread_std'], color='red', linestyle='-', alpha=0.7, label='+2σ')
        ax3.axhline(y=stats['spread_mean'] - 2*stats['spread_std'], color='red', linestyle='-', alpha=0.7, label='-2σ')
        ax3.set_ylabel('Spread')
        ax3.set_title(f'Spread Time Series (Current Z-score: {stats["zscore_current"]:.2f})', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Z-Score Time Series (third row, left)
        ax4 = fig.add_subplot(gs[2, 0])
        zscore_series = (spread_series - stats['spread_mean']) / stats['spread_std']
        ax4.plot(zscore_series.index, zscore_series.values, color='purple', alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axhline(y=1, color='orange', linestyle='--', alpha=0.7)
        ax4.axhline(y=-1, color='orange', linestyle='--', alpha=0.7)
        ax4.axhline(y=2, color='red', linestyle='-', alpha=0.7)
        ax4.axhline(y=-2, color='red', linestyle='-', alpha=0.7)
        ax4.set_ylabel('Z-Score')
        ax4.set_title('Z-Score Evolution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Spread Distribution (third row, center)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(stats['spread'], bins=50, alpha=0.7, color='green', density=True)
        ax5.axvline(x=stats['spread_mean'], color='black', linestyle='--', label='Mean')
        ax5.axvline(x=stats['spread_mean'] + stats['spread_std'], color='red', linestyle='--', alpha=0.7)
        ax5.axvline(x=stats['spread_mean'] - stats['spread_std'], color='red', linestyle='--', alpha=0.7)
        
        # Overlay normal distribution
        x_norm = np.linspace(stats['spread'].min(), stats['spread'].max(), 100)
        y_norm = stats.norm.pdf(x_norm, stats['spread_mean'], stats['spread_std'])
        ax5.plot(x_norm, y_norm, 'r-', linewidth=2, alpha=0.8, label='Normal')
        
        ax5.set_xlabel('Spread Value')
        ax5.set_ylabel('Density')
        ax5.set_title('Spread Distribution', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics Summary (third row, right)
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        stats_text = f"""
        REGRESSION ANALYSIS
        α (intercept): {stats['alpha']:.6f}
        β (slope): {stats['beta']:.6f}
        R²: {stats['r_squared']:.4f}
        Correlation: {stats['correlation']:.4f}
        
        SPREAD STATISTICS
        Mean: {stats['spread_mean']:.6f}
        Std Dev: {stats['spread_std']:.6f}
        Current Z-score: {stats['zscore_current']:.2f}
        
        TRADING SIGNAL
        Half-life: {stats['half_life']:.1f} days
        Signal: {stats['signal_strength']} {stats['signal']}
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 7. Rolling Correlation (bottom row, spans all columns)
        ax7 = fig.add_subplot(gs[3, :])
        
        # Calculate 24-hour rolling correlation
        window_size = 24 * 60  # 24 hours in minutes
        if len(df) > window_size:
            rolling_corr = df[x_col].rolling(window=window_size).corr(df[y_col])
            ax7.plot(rolling_corr.index, rolling_corr.values, color='blue', alpha=0.7)
            ax7.axhline(y=stats['correlation'], color='red', linestyle='--', alpha=0.7, label=f'Overall: {stats["correlation"]:.3f}')
            ax7.set_ylabel('Rolling Correlation')
            ax7.set_title('24-Hour Rolling Correlation', fontsize=12, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Insufficient data for rolling correlation', 
                    ha='center', va='center', transform=ax7.transAxes)
        
        # Format x-axis for time series plots
        for ax in [ax1, ax3, ax4, ax7]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df) // (10 * 24 * 60))))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Main title
        fig.suptitle(f'Comprehensive Pair Analysis: {s1_name} vs {s2_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Overview figure saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating overview figure: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def create_window_figure(df: pd.DataFrame, rolling_data: dict, window_days: int, 
                       symbol1: str, symbol2: str, output_path: str):
    """Create figure for specific rolling window analysis"""
    try:
        logger.info(f"Creating {window_days}-day window figure for {symbol1.split('_')[-2]} vs {symbol2.split('_')[-2]}")
        
        if not rolling_data or len(rolling_data.get('correlation', [])) == 0:
            logger.warning(f"No data available for {window_days}-day window figure")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{window_days}-Day Rolling Window Analysis: {symbol1.split("_")[-2]} vs {symbol2.split("_")[-2]}', 
                    fontsize=16, fontweight='bold')
        
        s1_name = symbol1.split('_')[-2]
        s2_name = symbol2.split('_')[-2]
        
        # 1. Rolling Correlation
        ax1 = axes[0, 0]
        if len(rolling_data['correlation']) > 0:
            ax1.plot(rolling_data['correlation'].index, rolling_data['correlation'].values, 
                    color='blue', alpha=0.7, linewidth=1)
            ax1.axhline(y=rolling_data.get('correlation_mean', 0), color='red', linestyle='--', 
                       alpha=0.7, label=f'Mean: {rolling_data.get("correlation_mean", 0):.3f}')
            ax1.set_ylabel('Correlation')
            ax1.set_title(f'{window_days}d Rolling Correlation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Rolling Beta
        ax2 = axes[0, 1]
        if len(rolling_data.get('beta', [])) > 0:
            ax2.plot(rolling_data['beta'].index, rolling_data['beta'].values, 
                    color='green', alpha=0.7, linewidth=1)
            beta_mean = rolling_data['beta'].mean()
            ax2.axhline(y=beta_mean, color='red', linestyle='--', alpha=0.7, label=f'Mean: {beta_mean:.3f}')
            ax2.set_ylabel('Beta')
            ax2.set_title(f'{window_days}d Rolling Beta')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No beta data', ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Rolling R-squared
        ax3 = axes[1, 0]
        if len(rolling_data.get('r_squared', [])) > 0:
            ax3.plot(rolling_data['r_squared'].index, rolling_data['r_squared'].values, 
                    color='purple', alpha=0.7, linewidth=1)
            r2_mean = rolling_data['r_squared'].mean()
            ax3.axhline(y=r2_mean, color='red', linestyle='--', alpha=0.7, label=f'Mean: {r2_mean:.3f}')
            ax3.set_ylabel('R²')
            ax3.set_title(f'{window_days}d Rolling R²')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No R² data', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Rolling Z-Score
        ax4 = axes[1, 1]
        if len(rolling_data.get('zscore', [])) > 0:
            ax4.plot(rolling_data['zscore'].index, rolling_data['zscore'].values, 
                    color='orange', alpha=0.7, linewidth=1)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='±1σ')
            ax4.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            ax4.axhline(y=2, color='red', linestyle='-', alpha=0.7, label='±2σ')
            ax4.axhline(y=-2, color='red', linestyle='-', alpha=0.7)
            ax4.set_ylabel('Z-Score')
            ax4.set_title(f'{window_days}d Rolling Z-Score')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No Z-score data', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"{window_days}-day window figure saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating window figure: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def create_comparison_figure(all_rolling_data: dict, symbol1: str, symbol2: str, output_path: str):
    """Create comparison figure across all windows"""
    try:
        logger.info(f"Creating comparison figure for all windows: {symbol1.split('_')[-2]} vs {symbol2.split('_')[-2]}")
        
        # Filter valid data
        valid_windows = {k: v for k, v in all_rolling_data.items() if v is not None and len(v.get('correlation', [])) > 0}
        
        if not valid_windows:
            logger.warning("No valid rolling data for comparison figure")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Multi-Window Comparison: {symbol1.split("_")[-2]} vs {symbol2.split("_")[-2]}', 
                    fontsize=16, fontweight='bold')
        
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        # 1. Correlation comparison
        ax1 = axes[0, 0]
        for i, (window, data) in enumerate(valid_windows.items()):
            if len(data.get('correlation', [])) > 0:
                color = colors[i % len(colors)]
                # Sample data for better visualization if too dense
                corr_data = data['correlation']
                if len(corr_data) > 1000:
                    step = len(corr_data) // 1000
                    corr_data = corr_data.iloc[::step]
                
                ax1.plot(corr_data.index, corr_data.values, 
                        label=f'{window}d', color=color, alpha=0.7, linewidth=1)
        
        ax1.set_ylabel('Correlation')
        ax1.set_title('Rolling Correlation Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 2. Beta comparison
        ax2 = axes[0, 1]
        for i, (window, data) in enumerate(valid_windows.items()):
            if len(data.get('beta', [])) > 0:
                color = colors[i % len(colors)]
                beta_data = data['beta']
                if len(beta_data) > 1000:
                    step = len(beta_data) // 1000
                    beta_data = beta_data.iloc[::step]
                
                ax2.plot(beta_data.index, beta_data.values, 
                        label=f'{window}d', color=color, alpha=0.7, linewidth=1)
        
        ax2.set_ylabel('Beta')
        ax2.set_title('Rolling Beta Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 3. R-squared comparison
        ax3 = axes[1, 0]
        for i, (window, data) in enumerate(valid_windows.items()):
            if len(data.get('r_squared', [])) > 0:
                color = colors[i % len(colors)]
                r2_data = data['r_squared']
                if len(r2_data) > 1000:
                    step = len(r2_data) // 1000
                    r2_data = r2_data.iloc[::step]
                
                ax3.plot(r2_data.index, r2_data.values, 
                        label=f'{window}d', color=color, alpha=0.7, linewidth=1)
        
        ax3.set_ylabel('R²')
        ax3.set_title('Rolling R² Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 4. Z-score comparison
        ax4 = axes[1, 1]
        for i, (window, data) in enumerate(valid_windows.items()):
            if len(data.get('zscore', [])) > 0:
                color = colors[i % len(colors)]
                zscore_data = data['zscore']
                if len(zscore_data) > 1000:
                    step = len(zscore_data) // 1000
                    zscore_data = zscore_data.iloc[::step]
                
                ax4.plot(zscore_data.index, zscore_data.values, 
                        label=f'{window}d', color=color, alpha=0.7, linewidth=1)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='±2σ')
        ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Z-Score')
        ax4.set_title('Rolling Z-Score Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Format x-axis
        for ax in axes.flat:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Comparison figure saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating comparison figure: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def create_advanced_metrics_table(all_rolling_data: dict, s1: str, s2: str, output_dir: str):
    """Create advanced metrics comparison table"""
    try:
        logger.info(f"Creating advanced metrics table for {s1.split('_')[-2]} vs {s2.split('_')[-2]}")
        
        # Filter valid data
        valid_data = {k: v for k, v in all_rolling_data.items() if v is not None}
        
        if not valid_data:
            logger.warning("No valid rolling data for advanced metrics table")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        windows = list(valid_data.keys())
        metrics = ['Correlation', 'R²', 'Beta', 'Z-score Current', 'Signal Strength']
        
        # Create table data
        table_data = []
        for metric in metrics:
            row = [metric]
            for window in windows:
                data = valid_data[window]
                try:
                    if metric == 'Correlation':
                        value = data.get('correlation_current', data.get('correlation_mean', 0))
                        row.append(f"{value:.4f}" if not pd.isna(value) else "N/A")
                    elif metric == 'R²':
                        if 'r_squared' in data and len(data['r_squared']) > 0:
                            value = data['r_squared'].iloc[-1] if hasattr(data['r_squared'], 'iloc') else data['r_squared']
                            row.append(f"{value:.4f}" if not pd.isna(value) else "N/A")
                        else:
                            row.append("N/A")
                    elif metric == 'Beta':
                        if 'beta' in data and len(data['beta']) > 0:
                            value = data['beta'].iloc[-1] if hasattr(data['beta'], 'iloc') else data['beta']
                            row.append(f"{value:.4f}" if not pd.isna(value) else "N/A")
                        else:
                            row.append("N/A")
                    elif metric == 'Z-score Current':
                        value = data.get('zscore_current', 0)
                        row.append(f"{value:.2f}" if not pd.isna(value) else "N/A")
                    elif metric == 'Signal Strength':
                        zscore = data.get('zscore_current', 0)
                        if abs(zscore) < 1:
                            row.append("WEAK")
                        elif abs(zscore) < 2:
                            row.append("MODERATE")
                        else:
                            row.append("STRONG")
                    else:
                        row.append("N/A")
                except Exception as e:
                    row.append("ERROR")
                    logger.warning(f"Error processing {metric} for {window}d window: {e}")
            table_data.append(row)
        
        # Create headers
        headers = ['Metric'] + [f"{w}d" for w in windows]
        
        # Ensure we have the right number of columns
        max_cols = len(headers)
        for row in table_data:
            while len(row) < max_cols:
                row.append("N/A")
            if len(row) > max_cols:
                row = row[:max_cols]
        
        # Create table
        stat_table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Apply formatting
        stat_table.auto_set_font_size(False)
        stat_table.set_fontsize(11)
        stat_table.scale(1.2, 2.5)
        
        # Get actual table dimensions
        table_cells = stat_table._cells
        
        # Format headers (row 0) - check if cells exist
        for i in range(len(headers)):
            cell_key = (0, i)
            if cell_key in table_cells:
                stat_table[cell_key].set_facecolor('#2C3E50')
                stat_table[cell_key].set_text_props(weight='bold', color='white')
        
        # Format data cells
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                cell_key = (i, j)
                if cell_key in table_cells:
                    if j == 0:  # First column (metric names)
                        stat_table[cell_key].set_facecolor('#34495E')
                        stat_table[cell_key].set_text_props(weight='bold', color='white')
                    else:
                        # Color code based on values
                        cell_text = table_data[i-1][j]
                        if 'STRONG' in cell_text:
                            stat_table[cell_key].set_facecolor('#E74C3C')
                            stat_table[cell_key].set_text_props(color='white', weight='bold')
                        elif 'MODERATE' in cell_text:
                            stat_table[cell_key].set_facecolor('#F39C12')
                            stat_table[cell_key].set_text_props(color='white')
                        elif 'WEAK' in cell_text:
                            stat_table[cell_key].set_facecolor('#95A5A6')
                            stat_table[cell_key].set_text_props(color='white')
                        else:
                            stat_table[cell_key].set_facecolor('#ECF0F1')
        
        plt.title(f'Advanced Metrics Comparison: {s1.split("_")[-2]} vs {s2.split("_")[-2]}', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Save table
        filename = f"advanced_metrics_{s1.split('_')[-2]}_{s2.split('_')[-2]}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Advanced metrics table saved: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating advanced metrics table: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_complete_analysis(symbol1: str, symbol2: str) -> dict:
    """Create complete pair analysis with all metrics and visualizations"""
    try:
        logger.info(f"Creating complete analysis with advanced statistics for {symbol1} / {symbol2}")
        
        # Create directory structure
        s1_short = symbol1.split('_')[-2]
        s2_short = symbol2.split('_')[-2]
        base_dir = f"plots/pair_analysis/{s1_short}_{s2_short}"
        directories = create_analysis_directories(base_dir)
        
        # Load data
        logger.info("Loading mark prices data...")
        df1 = load_mark_prices_data(symbol1)
        df2 = load_mark_prices_data(symbol2)
        
        if df1.empty or df2.empty:
            logger.error("Failed to load data for one or both symbols")
            return {}
        
        logger.info(f"Data loaded: {s1_short}={len(df1):,}, {s2_short}={len(df2):,}")
        
        # Align data
        aligned_df = create_minute_alignment_optimized(df1, df2, symbol1, symbol2)
        
        if aligned_df.empty:
            logger.error("Failed to align data")
            return {}
        
        # Calculate comprehensive statistics
        comprehensive_stats = calculate_comprehensive_stats(aligned_df, symbol1, symbol2)
        
        if not comprehensive_stats:
            logger.error("Failed to calculate comprehensive statistics")
            return {}
        
        # Create overview figure
        overview_path = os.path.join(directories['overview'], f"overview_{s1_short}_{s2_short}.png")
        create_overview_figure(aligned_df, comprehensive_stats, symbol1, symbol2, overview_path)
        
        # Calculate rolling statistics for different windows
        logger.info("Calculating rolling statistics with advanced metrics for all windows...")
        windows = [3, 7, 15, 30, 60, 90]
        all_rolling_data = {}
        
        for window in windows:
            logger.info(f"  Processing {window}-day window with advanced analysis...")
            
            # Calculate rolling stats
            rolling_stats = calculate_rolling_stats(aligned_df, window, symbol1, symbol2)
            
            if rolling_stats is not None:
                # Add advanced metrics
                advanced_metrics = calculate_advanced_spread_metrics(aligned_df, window)
                if advanced_metrics:
                    rolling_stats.update(advanced_metrics)
                
                all_rolling_data[window] = rolling_stats
                
                # Create window-specific figure
                window_path = os.path.join(directories['windows'], f'{window}d', f"rolling_{window}d_{s1_short}_{s2_short}.png")
                create_window_figure(aligned_df, rolling_stats, window, symbol1, symbol2, window_path)
            else:
                logger.warning(f"No data available for {window}-day window")
                all_rolling_data[window] = None
        
        # Create comparison figure
        comparison_path = os.path.join(directories['comparison'], f"comparison_all_windows_{s1_short}_{s2_short}.png")
        create_comparison_figure(all_rolling_data, symbol1, symbol2, comparison_path)
        
        # Create advanced metrics table
        advanced_file = create_advanced_metrics_table(all_rolling_data, symbol1, symbol2, directories['advanced'])
        
        # Compile results
        results = {
            'symbols': (symbol1, symbol2),
            'data_points': len(aligned_df),
            'comprehensive_stats': comprehensive_stats,
            'rolling_data': all_rolling_data,
            'directories': directories,
            'files': {
                'overview': overview_path,
                'comparison': comparison_path,
                'advanced_table': advanced_file
            }
        }
        
        logger.info(f"✅ Complete analysis finished for {s1_short} vs {s2_short}")
        logger.info(f"📊 Generated {len([f for f in results['files'].values() if f])} main files")
        logger.info(f"📁 Output directory: {base_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in complete analysis: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}

def main():
    """Main execution function"""
    try:
        parser = argparse.ArgumentParser(description='Advanced Pair Trading Analysis')
        parser.add_argument('--symbol1', required=True, help='First symbol to analyze')
        parser.add_argument('--symbol2', required=True, help='Second symbol to analyze')
        parser.add_argument('--limit', type=int, default=None, help='Limit number of records per symbol')
        
        args = parser.parse_args()
        
        # Log analysis start
        logger.info("🔬 Starting ADVANCED STATISTICAL Pair Analysis")
        logger.info(f"Symbols: {args.symbol1} / {args.symbol2}")
        logger.info("Rolling Windows: 3, 7, 15, 30, 60, 90 days")
        
        if ADVANCED_STATS:
            logger.info("Advanced Tests: ADF, PP, KPSS, Engle-Granger, Johansen")
        logger.info("Trading Metrics: Hurst, Half-life, Variance Ratio, etc.")
        
        # Run complete analysis
        results = create_complete_analysis(args.symbol1, args.symbol2)
        
        if results:
            logger.info("🎉 Analysis completed successfully!")
            
            # Print summary
            stats = results['comprehensive_stats']
            s1_name = args.symbol1.split('_')[-2]
            s2_name = args.symbol2.split('_')[-2]
            
            print(f"\n" + "="*60)
            print(f"PAIR ANALYSIS SUMMARY: {s1_name} vs {s2_name}")
            print(f"="*60)
            print(f"Data Points: {results['data_points']:,}")
            print(f"Correlation: {stats['correlation']:.4f}")
            print(f"R-squared: {stats['r_squared']:.4f}")
            print(f"Current Z-score: {stats['zscore_current']:.2f}")
            print(f"Half-life: {stats['half_life']:.1f} days")
            print(f"Trading Signal: {stats['signal_strength']} {stats['signal']}")
            print(f"Output Directory: {results['directories']['base']}")
            print(f"="*60)
            
        else:
            logger.error("❌ Analysis failed - no results generated")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)