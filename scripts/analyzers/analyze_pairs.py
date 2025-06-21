#!/usr/bin/env python3
"""
📊 Professional Pair Trading Analysis - ENHANCED VERSION
Análisis exhaustivo con métricas estadísticas avanzadas y visualización mejorada
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

def inspect_database_schema():
    """Inspeccionar esquema de la base de datos"""
    log.info("Inspecting database schema...")
    with db_manager.get_session() as session:
        schema_query = text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'mark_prices'
            ORDER BY ordinal_position
        """)
        columns = session.execute(schema_query).fetchall()
        log.info("mark_prices table columns:")
        for col in columns:
            log.info(f"  {col.column_name}: {col.data_type}")
        
        sample_query = text("""
            SELECT symbol, COUNT(*) as count, MIN(timestamp) as min_date, MAX(timestamp) as max_date
            FROM mark_prices 
            WHERE symbol LIKE '%GIGA%' OR symbol LIKE '%SPX%'
            GROUP BY symbol
        """)
        sample_data = session.execute(sample_query).fetchall()
        log.info("\nAvailable data for GIGA/SPX symbols:")
        for row in sample_data:
            log.info(f"  {row.symbol}: {row.count:,} records ({row.min_date} to {row.max_date})")

def load_complete_dataset(symbol1: str, symbol2: str) -> Tuple[pd.Series, pd.Series]:
    """Cargar dataset completo con mejor manejo de datos"""
    log.info(f"Loading complete dataset for {symbol1} / {symbol2}")
    
    with db_manager.get_session() as session:
        # Verificar cobertura de datos
        check_query = text("""
            SELECT symbol, COUNT(*) as total_records,
                   MIN(timestamp) as start_date,
                   MAX(timestamp) as end_date
            FROM mark_prices 
            WHERE symbol IN (:symbol1, :symbol2)
            AND mark_price IS NOT NULL
            AND mark_price > 0
            GROUP BY symbol
        """)
        
        coverage_df = pd.read_sql(check_query, session.bind, params={
            'symbol1': symbol1, 'symbol2': symbol2
        })
        
        if coverage_df.empty or len(coverage_df) < 2:
            log.error(f"Insufficient data for symbols {symbol1} and {symbol2}")
            return pd.Series(), pd.Series()
        
        for _, row in coverage_df.iterrows():
            log.info(f"  {row['symbol']}: {row['total_records']:,} records "
                    f"({row['start_date']} to {row['end_date']})")
        
        # Cargar datos
        data_query = text("""
            SELECT timestamp, symbol, mark_price
            FROM mark_prices 
            WHERE symbol IN (:symbol1, :symbol2)
            AND mark_price IS NOT NULL
            AND mark_price > 0
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(data_query, session.bind, params={
            'symbol1': symbol1, 'symbol2': symbol2
        }, parse_dates=['timestamp'])
        
        if df.empty:
            log.error("No data loaded from database")
            return pd.Series(), pd.Series()
        
        # Pivot y alinear
        df_pivot = df.pivot(index='timestamp', columns='symbol', values='mark_price')
        
        # Resample a intervalos regulares (1 minuto) para evitar problemas
        df_resampled = df_pivot.resample('1T').last()
        df_aligned = df_resampled.dropna()
        
        if df_aligned.empty:
            log.error("No aligned data available")
            return pd.Series(), pd.Series()
        
        prices1 = df_aligned[symbol1]
        prices2 = df_aligned[symbol2]
        
        # Calcular estadísticas
        period_days = (df_aligned.index[-1] - df_aligned.index[0]).total_seconds() / 86400
        samples_per_day = len(df_aligned) / period_days if period_days > 0 else 0
        
        log.info(f"Final aligned dataset: {len(df_aligned):,} observations over {period_days:.1f} days")
        log.info(f"Data frequency: ~{samples_per_day:.1f} observations per day")
        log.info(f"Estimated data interval: ~{1440/samples_per_day:.1f} minutes")
        
        return prices1, prices2

def calculate_stationarity_tests(series: pd.Series, name: str) -> Dict:
    """Tests de estacionariedad comprehensivos"""
    log.info(f"Running stationarity tests for {name}...")
    results = {}
    
    # ADF Test
    try:
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, adf_icbest = adfuller(
            series.dropna(), autolag='AIC', maxlag=int(len(series)**0.25)
        )
        results['adf'] = {
            'statistic': adf_stat,
            'p_value': adf_p,
            'lags_used': adf_lags,
            'n_obs': adf_nobs,
            'critical_values': dict(zip(['1%', '5%', '10%'], adf_crit.values())),
            'ic_best': adf_icbest,
            'is_stationary': adf_p < 0.05,
            'is_stationary_1pct': adf_p < 0.01
        }
        log.info(f"  ADF test: p-value = {adf_p:.6f} ({'stationary' if adf_p < 0.05 else 'non-stationary'})")
    except Exception as e:
        log.warning(f"ADF test failed for {name}: {e}")
        results['adf'] = {'error': str(e)}
    
    # KPSS Test
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series.dropna(), regression='c', nlags='auto')
        results['kpss'] = {
            'statistic': kpss_stat,
            'p_value': kpss_p,
            'lags_used': kpss_lags,
            'critical_values': dict(zip(['10%', '5%', '2.5%', '1%'], kpss_crit.values())),
            'is_stationary': kpss_p > 0.05,  # KPSS: null hypothesis is stationarity
            'is_stationary_5pct': kpss_p > 0.05
        }
        log.info(f"  KPSS test: p-value = {kpss_p:.6f} ({'stationary' if kpss_p > 0.05 else 'non-stationary'})")
    except Exception as e:
        log.warning(f"KPSS test failed for {name}: {e}")
        results['kpss'] = {'error': str(e)}
    
    return results

def calculate_cointegration_analysis(log_p1: pd.Series, log_p2: pd.Series) -> Dict:
    """Static cointegration analysis via Engle-Granger"""
    log.info("Performing cointegration analysis on log prices...")
    
    # Debug info
    log.info(f"Price ranges - Symbol1: {np.exp(log_p1).min():.2f} to {np.exp(log_p1).max():.2f}")
    log.info(f"Log price ranges - Symbol1: {log_p1.min():.4f} to {log_p1.max():.4f}")
    
    # Perform Engle-Granger test
    stat, pvalue, crit_values = coint(log_p1, log_p2)
    
    # Estimate cointegration equation using OLS
    X = np.column_stack([np.ones(len(log_p2)), log_p2.values])
    model = OLS(log_p1.values, X).fit()
    alpha, beta = model.params[0], model.params[1]
    r2 = model.rsquared
    beta_std_err = model.bse[1]
    
    log.info(f"Engle-Granger test: p-value = {pvalue:.6f} ({'cointegrated at 5%' if pvalue < 0.05 else 'not cointegrated'})")
    log.info(f"Cointegration equation: log(P1) = {alpha:.4f} + {beta:.4f} * log(P2) + ε (R² = {r2:.4f})")
    
    return {
        'statistic': stat,
        'p_value': pvalue,
        'critical_values': crit_values,
        'alpha': alpha,
        'beta': beta,
        'beta_std_error': beta_std_err,
        'r_squared': r2,
        'cointegration_equation': {
            'alpha': alpha,
            'beta': beta
        }
    }

def calculate_half_life_analysis(spread: pd.Series) -> Dict:
    """Calculate half-life with proper time units conversion"""
    log.info("Calculating half-life analysis...")
    
    spread_clean = spread.dropna()
    if len(spread_clean) < 20:
        log.warning("Not enough data to estimate half-life.")
        return {'error': 'Insufficient data'}
    
    # AR(1) regression: ΔS_t = α + β·S_{t-1} + ε_t
    spread_lag = spread_clean.shift(1)
    delta_spread = spread_clean.diff()
    
    # Align data
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
    
    # Calculate half-life in observations
    if beta_hat >= 0:
        halflife_obs = np.inf
    else:
        halflife_obs = -np.log(2) / beta_hat
    
    # Convert to time units
    samples_per_day = _infer_samples_per_day(spread_clean.index)
    halflife_days = halflife_obs / samples_per_day if samples_per_day > 0 else np.inf
    halflife_hours = halflife_days * 24
    
    # OU process parameters
    theta = -beta_hat  # mean reversion speed
    mu = -alpha_hat / beta_hat if beta_hat != 0 else spread_clean.mean()  # long-term mean
    
    # Estimate volatility
    residuals = y - model.predict(X)
    sigma = np.std(residuals) * np.sqrt(samples_per_day * 252)  # annualized
    
    results = {
        'ar1': {
            'alpha': alpha_hat,
            'beta': beta_hat,
            'r_squared': model.rsquared,
            'beta_pvalue': model.pvalues[1],
            'half_life': halflife_obs,  # Keep for compatibility
            'half_life_obs': halflife_obs,
            'half_life_days': halflife_days,
            'half_life_hours': halflife_hours,
        },
        'ou': {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': halflife_obs,  # Keep for compatibility
            'half_life_days': halflife_days,
            'r_squared': model.rsquared
        }
    }
    
    log.info(f"  Half-life (AR1 method): {halflife_days:.2f} days ({halflife_hours:.1f} hours)")
    log.info(f"  Half-life (OU method): {halflife_days:.2f} days")
    
    return results

def calculate_rolling_stationarity_tests(series: pd.Series, name: str,
                                       window_days: int = 30,
                                       step_days: int = 1) -> Dict:
    """Fixed rolling stationarity tests"""
    log.info(f"Running rolling stationarity tests for {name} ({window_days}-day window, step={step_days} days)...")
    
    samples_per_day = _infer_samples_per_day(series.index)
    window_points = int(window_days * samples_per_day)
    step_points = max(int(step_days * samples_per_day), 1)
    
    # Check if we have enough data (at least 2 windows)
    min_required = window_points + step_points
    if len(series) < min_required:
        log.warning(f"Insufficient data for rolling stationarity tests ({len(series)} < {min_required})")
        return {'error': 'Insufficient data', 'window_points': window_points, 'data_points': len(series)}
    
    results = {
        'window_days': window_days,
        'window_points': window_points,
        'step_days': step_days,
        'adf_pvalues': [],
        'kpss_pvalues': [],
        'timestamps': []
    }
    
    series_clean = series.dropna()
    
    # Rolling window analysis
    for i in range(window_points, len(series_clean), step_points):
        window_data = series_clean.iloc[i-window_points:i]
        timestamp = series_clean.index[i-1]
        
        # ADF test
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(window_data, autolag='AIC')
            results['adf_pvalues'].append(adf_p)
            results['timestamps'].append(timestamp)
        except:
            continue
        
        # KPSS test
        try:
            kpss_stat, kpss_p, _, _ = kpss(window_data, regression='c')
            results['kpss_pvalues'].append(kpss_p)
        except:
            results['kpss_pvalues'].append(np.nan)
    
    # Convert to Series
    if results['timestamps']:
        results['adf_pvalues'] = pd.Series(results['adf_pvalues'], index=results['timestamps'])
        results['kpss_pvalues'] = pd.Series(results['kpss_pvalues'], index=results['timestamps'])
        
        # Calculate summary statistics
        adf_stationary_pct = (results['adf_pvalues'] < 0.05).mean() * 100
        kpss_stationary_pct = (results['kpss_pvalues'] > 0.05).mean() * 100
        
        results['summary'] = {
            'adf_stationary_pct': adf_stationary_pct,
            'kpss_stationary_pct': kpss_stationary_pct,
            'adf_latest_pvalue': results['adf_pvalues'].iloc[-1],
            'kpss_latest_pvalue': results['kpss_pvalues'].iloc[-1],
            'observations': len(results['adf_pvalues'])
        }
        
        log.info(f"  Rolling ADF: {adf_stationary_pct:.1f}% of windows stationary")
        log.info(f"  Rolling KPSS: {kpss_stationary_pct:.1f}% of windows stationary")
    else:
        results['summary'] = {
            'adf_stationary_pct': 0,
            'kpss_stationary_pct': 0,
            'observations': 0
        }
    
    return results

def calculate_rolling_cointegration(log_p1: pd.Series, log_p2: pd.Series,
                                  window_days: int = 30,
                                  step_days: int = 1) -> Dict:
    """Calculate rolling cointegration tests"""
    log.info(f"Running rolling cointegration tests ({window_days}-day window)...")
    
    samples_per_day = _infer_samples_per_day(log_p1.index)
    window_points = int(window_days * samples_per_day)
    step_points = max(int(step_days * samples_per_day), 1)
    
    if len(log_p1) < window_points + step_points:
        log.warning("Insufficient data for rolling cointegration")
        return {'error': 'Insufficient data'}
    
    results = {
        'timestamps': [],
        'pvalues': [],
        'betas': [],
        'r_squared': []
    }
    
    for i in range(window_points, len(log_p1), step_points):
        window_p1 = log_p1.iloc[i-window_points:i]
        window_p2 = log_p2.iloc[i-window_points:i]
        timestamp = log_p1.index[i-1]
        
        try:
            # Engle-Granger test
            stat, pvalue, _ = coint(window_p1, window_p2)
            
            # OLS for beta
            X = np.column_stack([np.ones(len(window_p2)), window_p2])
            model = OLS(window_p1, X).fit()
            beta = model.params[1]
            r2 = model.rsquared
            
            results['timestamps'].append(timestamp)
            results['pvalues'].append(pvalue)
            results['betas'].append(beta)
            results['r_squared'].append(r2)
        except:
            continue
    
    if results['timestamps']:
        results['pvalues'] = pd.Series(results['pvalues'], index=results['timestamps'])
        results['betas'] = pd.Series(results['betas'], index=results['timestamps'])
        results['r_squared'] = pd.Series(results['r_squared'], index=results['timestamps'])
        
        coint_pct = (results['pvalues'] < 0.05).mean() * 100
        
        results['summary'] = {
            'cointegrated_pct': coint_pct,
            'latest_pvalue': results['pvalues'].iloc[-1],
            'latest_beta': results['betas'].iloc[-1],
            'beta_stability': results['betas'].std(),
            'observations': len(results['pvalues'])
        }
        
        log.info(f"  {coint_pct:.1f}% of windows show cointegration")
        log.info(f"  Beta stability (std): {results['summary']['beta_stability']:.4f}")
    else:
        results['summary'] = {
            'cointegrated_pct': 0,
            'observations': 0
        }
    
    return results

def calculate_hurst_exponent(series: pd.Series, max_lags: int = 20) -> Dict:
    """Calculate Hurst exponent with improved R/S analysis"""
    log.info("Calculating Hurst exponent...")
    
    series_clean = series.dropna().values
    if len(series_clean) < 100:
        log.warning("Not enough data for reliable Hurst estimation")
        return {'error': 'Insufficient data'}
    
    # R/S analysis
    lags = range(2, min(max_lags, len(series_clean) // 2))
    rs_values = []
    
    for lag in lags:
        # Calculate R/S for this lag
        nblocks = len(series_clean) // lag
        if nblocks < 2:
            continue
            
        rs_lag = []
        for i in range(nblocks):
            block = series_clean[i*lag:(i+1)*lag]
            if len(block) < 2:
                continue
                
            # Demean the block
            mean = np.mean(block)
            block_demean = block - mean
            
            # Calculate cumulative sum
            cumsum = np.cumsum(block_demean)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(block, ddof=1)
            
            if S > 0:
                rs_lag.append(R / S)
        
        if rs_lag:
            rs_values.append(np.mean(rs_lag))
        else:
            rs_values.append(np.nan)
    
    # Remove NaN values
    valid_indices = ~np.isnan(rs_values)
    lags_valid = np.array(list(lags))[valid_indices]
    rs_valid = np.array(rs_values)[valid_indices]
    
    if len(lags_valid) < 3:
        return {'error': 'Not enough valid R/S values'}
    
    # Log-log regression
    log_lags = np.log(lags_valid)
    log_rs = np.log(rs_valid)
    
    # OLS regression
    X = np.column_stack([np.ones(len(log_lags)), log_lags])
    model = OLS(log_rs, X).fit()
    
    hurst = model.params[1]
    r_squared = model.rsquared
    p_value = model.pvalues[1]
    
    # Interpretation
    if hurst < 0.4:
        interpretation = "Mean-reverting (anti-persistent)"
    elif hurst < 0.6:
        interpretation = "Random walk (geometric Brownian motion)"
    else:
        interpretation = "Trending (persistent)"
    
    results = {
        'hurst': hurst,
        'r_squared': r_squared,
        'p_value': p_value,
        'interpretation': interpretation,
        'n_lags': len(lags_valid)
    }
    
    log.info(f"  Hurst exponent (R/S): {hurst:.4f} (R² = {r_squared:.4f})")
    log.info(f"  → {interpretation}")
    
    return results

def calculate_rolling_window_analysis(log_p1: pd.Series, log_p2: pd.Series, 
                                    windows: List[int],
                                    alpha: float = 0.0,
                                    beta: float = 1.0) -> Dict:
    """Enhanced rolling window analysis"""
    log.info("Calculating rolling window analysis...")
    
    results = {}
    samples_per_day = _infer_samples_per_day(log_p1.index)
    
    for window_days in windows:
        window_samples = int(window_days * samples_per_day)
        
        if len(log_p1) < window_samples * 2:
            log.warning(f"  Insufficient data for {window_days}-day window")
            continue
        
        log.info(f"  Processing {window_days}-day window ({window_samples} samples)...")
        
        # Initialize arrays
        correlations = []
        betas = []
        r_squared = []
        volatilities_1 = []
        volatilities_2 = []
        timestamps = []
        
        # Calculate spread using cointegration parameters
        spread = log_p1 - (beta * log_p2 + alpha)
        
        # Rolling calculations
        for i in range(window_samples, len(log_p1)):
            window_p1 = log_p1.iloc[i-window_samples:i]
            window_p2 = log_p2.iloc[i-window_samples:i]
            window_spread = spread.iloc[i-window_samples:i]
            
            # Correlation
            corr = window_p1.corr(window_p2)
            correlations.append(corr)
            
            # Beta (hedge ratio) via OLS
            X = np.column_stack([np.ones(len(window_p2)), window_p2.values])
            model = OLS(window_p1.values, X).fit()
            betas.append(model.params[1])
            r_squared.append(model.rsquared)
            
            # Volatilities (annualized)
            vol1 = window_p1.pct_change().std() * np.sqrt(252 * samples_per_day)
            vol2 = window_p2.pct_change().std() * np.sqrt(252 * samples_per_day)
            volatilities_1.append(vol1)
            volatilities_2.append(vol2)
            
            timestamps.append(log_p1.index[i])
        
        # Convert to Series
        timestamps = pd.DatetimeIndex(timestamps)
        correlations = pd.Series(correlations, index=timestamps)
        betas = pd.Series(betas, index=timestamps)
        r_squared = pd.Series(r_squared, index=timestamps)
        vol_ratio = pd.Series(np.array(volatilities_1) / np.array(volatilities_2), index=timestamps)
        
        # Calculate z-scores
        spread_mean = spread.rolling(window_samples).mean()
        spread_std = spread.rolling(window_samples).std()
        z_scores = (spread - spread_mean) / spread_std
        
        # Store results
        results[window_days] = {
            'correlations': correlations,
            'betas': betas,
            'r_squared': r_squared,
            'vol_ratio': vol_ratio,
            'z_scores': z_scores.iloc[window_samples:],
            'summary': {
                'corr_mean': correlations.mean(),
                'corr_std': correlations.std(),
                'corr_current': correlations.iloc[-1],
                'corr_min': correlations.min(),
                'corr_max': correlations.max(),
                'beta_mean': betas.mean(),
                'beta_std': betas.std(),
                'beta_current': betas.iloc[-1],
                'r2_mean': r_squared.mean(),
                'r2_current': r_squared.iloc[-1],
                'z_current': z_scores.iloc[-1],
                'z_2sigma_pct': (np.abs(z_scores) > 2).mean() * 100,
                'z_3sigma_pct': (np.abs(z_scores) > 3).mean() * 100,
                'vol_ratio_mean': vol_ratio.mean(),
                'vol_ratio_current': vol_ratio.iloc[-1]
            }
        }
    
    return results

def create_professional_visualization(symbol1: str, symbol2: str,
                                    log_p1: pd.Series, log_p2: pd.Series,
                                    alpha: float, beta: float,
                                    stationarity_results: Dict,
                                    rolling_stationarity: Dict,
                                    cointegration_results: Dict,
                                    rolling_cointegration: Dict,
                                    half_life_results: Dict,
                                    hurst_results: Dict,
                                    rolling_results: Dict,
                                    summary: Dict):
    """Create enhanced visualization with dual axes"""
    
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.5, 1, 1, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Extract short names
    s1 = symbol1.split('_')[-2] if '_' in symbol1 else symbol1
    s2 = symbol2.split('_')[-2] if '_' in symbol2 else symbol2
    
    # Calculate spread
    spread = log_p1 - (beta * log_p2 + alpha)
    
    # 1. Price Series with Dual Axes
    ax1 = fig.add_subplot(gs[0, :])
    
    # Convert log prices back to regular prices for visualization
    p1 = np.exp(log_p1)
    p2 = np.exp(log_p2)
    
    # First axis for symbol1
    color1 = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{s1} Price', color=color1)
    ax1.plot(p1.index, p1.values, color=color1, alpha=0.7, linewidth=1, label=s1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Second axis for symbol2
    ax1_twin = ax1.twinx()
    color2 = 'tab:orange'
    ax1_twin.set_ylabel(f'{s2} Price', color=color2)
    ax1_twin.plot(p2.index, p2.values, color=color2, alpha=0.7, linewidth=1, label=s2)
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.set_yscale('log')
    
    ax1.set_title(f'Price Series: {s1} vs {s2} (Log Scale)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 2. Spread Analysis with Bands
    ax2 = fig.add_subplot(gs[1, :])
    
    # Calculate rolling statistics
    window = 30  # days
    samples_per_day = _infer_samples_per_day(spread.index)
    window_samples = int(window * samples_per_day)
    
    spread_mean = spread.rolling(window_samples).mean()
    spread_std = spread.rolling(window_samples).std()
    
    # Plot spread
    ax2.plot(spread.index, spread.values, 'k-', alpha=0.7, linewidth=1, label='Spread')
    ax2.plot(spread_mean.index, spread_mean.values, 'b--', alpha=0.8, label=f'{window}d MA')
    
    # Add bands
    ax2.fill_between(spread.index, 
                     spread_mean + 2*spread_std, 
                     spread_mean - 2*spread_std,
                     alpha=0.2, color='gray', label='±2σ bands')
    ax2.fill_between(spread.index,
                     spread_mean + 3*spread_std,
                     spread_mean - 3*spread_std,
                     alpha=0.1, color='gray', label='±3σ bands')
    
    ax2.axhline(y=spread.mean(), color='r', linestyle=':', alpha=0.7, label='Long-term mean')
    ax2.set_title(f'Spread = log({s1}) - ({beta:.4f} × log({s2}) + {alpha:.4f})', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Spread Value')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Z-Score
    ax3 = fig.add_subplot(gs[2, :])
    z_scores = (spread - spread_mean) / spread_std
    ax3.plot(z_scores.index, z_scores.values, 'b-', alpha=0.7, linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='±2σ')
    ax3.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=3, color='darkred', linestyle=':', alpha=0.5, label='±3σ')
    ax3.axhline(y=-3, color='darkred', linestyle=':', alpha=0.5)
    ax3.fill_between(z_scores.index, -2, 2, alpha=0.1, color='green')
    ax3.set_title('Z-Score of Spread', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Z-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Rolling Correlation
    ax4 = fig.add_subplot(gs[3, 0])
    if 30 in rolling_results:
        corr = rolling_results[30]['correlations']
        ax4.plot(corr.index, corr.values, 'g-', alpha=0.7)
        ax4.axhline(y=corr.mean(), color='r', linestyle='--', alpha=0.5, 
                    label=f'Mean: {corr.mean():.3f}')
        ax4.fill_between(corr.index, corr.mean() - corr.std(), corr.mean() + corr.std(),
                        alpha=0.2, color='gray')
        ax4.set_title('30-Day Rolling Correlation', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Correlation')
        ax4.set_ylim(max(-1, corr.min() - 0.1), min(1, corr.max() + 0.1))
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Rolling Beta (Hedge Ratio)
    ax5 = fig.add_subplot(gs[3, 1])
    if 30 in rolling_results:
        betas = rolling_results[30]['betas']
        ax5.plot(betas.index, betas.values, 'purple', alpha=0.7)
        ax5.axhline(y=beta, color='r', linestyle='--', alpha=0.5, 
                    label=f'Static: {beta:.3f}')
        ax5.axhline(y=betas.mean(), color='orange', linestyle='--', alpha=0.5,
                    label=f'Mean: {betas.mean():.3f}')
        ax5.set_title('30-Day Rolling Beta (Hedge Ratio)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Beta')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Rolling Stationarity Tests
    ax6 = fig.add_subplot(gs[4, 0])
    if 'adf_pvalues' in rolling_stationarity.get('spread', {}):
        adf_pvals = rolling_stationarity['spread']['adf_pvalues']
        if isinstance(adf_pvals, pd.Series) and len(adf_pvals) > 0:
            ax6.plot(adf_pvals.index, adf_pvals.values, 'b-', alpha=0.7, label='ADF p-value')
            ax6.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% significance')
            ax6.axhline(y=0.01, color='darkred', linestyle=':', alpha=0.5, label='1% significance')
            ax6.fill_between(adf_pvals.index, 0, 0.05, alpha=0.2, color='green')
            ax6.set_title('Rolling ADF Test for Spread Stationarity', fontsize=12, fontweight='bold')
            ax6.set_ylabel('p-value')
            ax6.set_ylim(0, min(1, adf_pvals.max() + 0.1))
            ax6.legend()
            ax6.grid(True, alpha=0.3)
    
    # 7. Rolling Cointegration
    ax7 = fig.add_subplot(gs[4, 1])
    if 'pvalues' in rolling_cointegration and isinstance(rolling_cointegration['pvalues'], pd.Series):
        coint_pvals = rolling_cointegration['pvalues']
        ax7.plot(coint_pvals.index, coint_pvals.values, 'g-', alpha=0.7, label='Cointegration p-value')
        ax7.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% significance')
        ax7.axhline(y=0.01, color='darkred', linestyle=':', alpha=0.5, label='1% significance')
        ax7.fill_between(coint_pvals.index, 0, 0.05, alpha=0.2, color='green')
        ax7.set_title('Rolling Cointegration Test', fontsize=12, fontweight='bold')
        ax7.set_ylabel('p-value')
        ax7.set_ylim(0, min(1, coint_pvals.max() + 0.1))
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Summary Statistics Box
    ax8 = fig.add_subplot(gs[5, :])
    ax8.axis('off')
    
    # Create summary text
    summary_text = f"""
    PAIR TRADING ANALYSIS SUMMARY
    
    Pair: {s1}/{s2}
    Overall Score: {summary['overall_score']:.1f}/100
    Recommendation: {summary['recommendation']}
    Risk Level: {summary['risk_level']}
    
    KEY METRICS:
    • Cointegration p-value: {cointegration_results.get('p_value', np.nan):.4f}
    • Half-life: {half_life_results.get('ar1', {}).get('half_life_days', np.nan):.1f} days
    • Hurst Exponent: {hurst_results.get('hurst', np.nan):.4f} ({hurst_results.get('interpretation', 'N/A')})
    • Current Z-score: {z_scores.iloc[-1]:.2f}
    • 30d Correlation: {rolling_results.get(30, {}).get('summary', {}).get('corr_current', np.nan):.3f}
    • Beta Stability: {rolling_results.get(30, {}).get('summary', {}).get('beta_std', np.nan):.4f}
    
    SCORING BREAKDOWN:
    • Stationarity: {summary['scores']['stationarity']:.1f}/100
    • Cointegration: {summary['scores']['cointegration']:.1f}/100
    • Half-life: {summary['scores']['half_life']:.1f}/100
    • Correlation Stability: {summary['scores']['correlation_stability']:.1f}/100
    • Hurst: {summary['scores']['hurst']:.1f}/100
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle(f'Comprehensive Pair Trading Analysis: {s1}/{s2}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.tight_layout()
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"{s1}_{s2}_comprehensive_analysis.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info(f"Comprehensive visualization saved: {filename}")

def generate_professional_summary(symbol1: str, symbol2: str,
                                stationarity_results: Dict,
                                rolling_stationarity: Dict,
                                cointegration_results: Dict,
                                rolling_cointegration: Dict,
                                half_life_results: Dict,
                                hurst_results: Dict,
                                rolling_results: Dict) -> Dict:
    """Generate comprehensive trading summary with scores"""
    
    # Extract short names
    s1 = symbol1.split('_')[-2] if '_' in symbol1 else symbol1
    s2 = symbol2.split('_')[-2] if '_' in symbol2 else symbol2
    
    scores = {}
    
    # 1. Stationarity Score (0-100)
    spread_adf = stationarity_results.get('spread', {}).get('adf', {})
    spread_kpss = stationarity_results.get('spread', {}).get('kpss', {})
    
    stationarity_score = 0
    if spread_adf.get('p_value', 1) < 0.01:
        stationarity_score += 50
    elif spread_adf.get('p_value', 1) < 0.05:
        stationarity_score += 30
    elif spread_adf.get('p_value', 1) < 0.10:
        stationarity_score += 15
    
    if spread_kpss.get('p_value', 0) > 0.10:
        stationarity_score += 50
    elif spread_kpss.get('p_value', 0) > 0.05:
        stationarity_score += 30
    elif spread_kpss.get('p_value', 0) > 0.01:
        stationarity_score += 15
    
    scores['stationarity'] = min(stationarity_score, 100)
    
    # 2. Cointegration Score (0-100)
    coint_pvalue = cointegration_results.get('p_value', 1)
    if coint_pvalue < 0.01:
        scores['cointegration'] = 100
    elif coint_pvalue < 0.05:
        scores['cointegration'] = 70
    elif coint_pvalue < 0.10:
        scores['cointegration'] = 40
    else:
        scores['cointegration'] = 20
    
    # 3. Half-life Score (0-100)
    hl_days = half_life_results.get('ar1', {}).get('half_life_days', np.inf)
    if hl_days < 2:
        scores['half_life'] = 100
    elif hl_days < 5:
        scores['half_life'] = 80
    elif hl_days < 10:
        scores['half_life'] = 60
    elif hl_days < 20:
        scores['half_life'] = 40
    elif hl_days < 30:
        scores['half_life'] = 20
    else:
        scores['half_life'] = 0
    
    # 4. Correlation Stability Score (0-100)
    if 30 in rolling_results:
        corr_std = rolling_results[30]['summary']['corr_std']
        if corr_std < 0.05:
            scores['correlation_stability'] = 100
        elif corr_std < 0.10:
            scores['correlation_stability'] = 80
        elif corr_std < 0.15:
            scores['correlation_stability'] = 60
        elif corr_std < 0.20:
            scores['correlation_stability'] = 40
        else:
            scores['correlation_stability'] = 20
    else:
        scores['correlation_stability'] = 0
    
    # 5. Hurst Score (0-100)
    hurst = hurst_results.get('hurst', 0.5)
    if 0.3 < hurst < 0.7:  # Near random walk is good for mean reversion
        scores['hurst'] = 100
    elif 0.2 < hurst < 0.8:
        scores['hurst'] = 70
    else:
        scores['hurst'] = 30
    
    # Overall Score
    overall_score = np.mean(list(scores.values()))
    
    # Recommendation
    if overall_score >= 80:
        recommendation = "🟢 EXCELLENT - Highly suitable for pair trading"
        risk_level = "LOW"
        confidence = "HIGH"
        proceed = True
    elif overall_score >= 60:
        recommendation = "🟡 GOOD - Suitable with careful monitoring"
        risk_level = "MEDIUM"
        confidence = "MEDIUM"
        proceed = True
    elif overall_score >= 40:
        recommendation = "🟠 MARGINAL - Consider alternatives"
        risk_level = "MEDIUM-HIGH"
        confidence = "LOW"
        proceed = False
    else:
        recommendation = "🔴 POOR - Not suitable for systematic trading"
        risk_level = "HIGH"
        confidence = "VERY LOW"
        proceed = False
    
    return {
        'pair_name': f"{s1}/{s2}",
        'overall_score': overall_score,
        'scores': scores,
        'recommendation': recommendation,
        'risk_level': risk_level,
        'confidence_level': confidence,
        'proceed_to_backtest': proceed,
        'timestamp': datetime.now()
    }

def generate_detailed_report(symbol1: str, symbol2: str,
                           stationarity_results: Dict,
                           rolling_stationarity: Dict,
                           cointegration_results: Dict,
                           rolling_cointegration: Dict,
                           half_life_results: Dict,
                           hurst_results: Dict,
                           rolling_results: Dict,
                           summary: Dict):
    """Generate detailed text report"""
    
    # Extract short names
    s1 = symbol1.split('_')[-2] if '_' in symbol1 else symbol1
    s2 = symbol2.split('_')[-2] if '_' in symbol2 else symbol2
    
    log.info("\n" + "="*100)
    log.info("COMPREHENSIVE PAIR TRADING ANALYSIS REPORT")
    log.info("="*100)
    log.info(f"Pair: {summary['pair_name']}")
    log.info(f"Analysis Timestamp: {summary['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Overall Score: {summary['overall_score']:.1f}/100")
    log.info(f"Recommendation: {summary['recommendation']}")
    log.info(f"Risk Category: {summary['risk_level']}")
    log.info(f"Confidence Level: {summary['confidence_level']}")
    log.info(f"Proceed to Backtest: {'YES ✅' if summary['proceed_to_backtest'] else 'NO ❌'}")
    
    log.info("\n🎯 DETAILED SCORING BREAKDOWN:")
    for metric, score in summary['scores'].items():
        log.info(f"  {metric.replace('_', ' ').title()}: {score:.1f}/100")
    
    # Stationarity Tests
    log.info("\n🔬 STATIONARITY TESTS (STATIC):")
    for series_name, results in stationarity_results.items():
        log.info(f"  {series_name.upper()}:")
        
        adf = results.get('adf', {})
        if 'error' not in adf:
            log.info(f"    ADF: statistic={adf.get('statistic', np.nan):.4f}, "
                    f"p-value={adf.get('p_value', np.nan):.6f}")
            log.info(f"         {'✅' if adf.get('is_stationary', False) else '❌'} "
                    f"{'Stationary' if adf.get('is_stationary', False) else 'Non-stationary'} at 5%")
            log.info(f"         {'✅' if adf.get('is_stationary_1pct', False) else '❌'} "
                    f"{'Stationary' if adf.get('is_stationary_1pct', False) else 'Non-stationary'} at 1%")
        
        kpss = results.get('kpss', {})
        if 'error' not in kpss:
            log.info(f"    KPSS: statistic={kpss.get('statistic', np.nan):.4f}, "
                    f"p-value={kpss.get('p_value', np.nan):.6f}")
            log.info(f"          {'✅' if kpss.get('is_stationary', False) else '❌'} "
                    f"{'Stationary' if kpss.get('is_stationary', False) else 'Non-stationary'}")
    
    # Rolling Stationarity
    log.info("\n🔄 STATIONARITY TESTS (ROLLING 30-DAY):")
    for series_name in ['symbol1_log', 'symbol2_log', 'spread']:
        rolling_stats = rolling_stationarity.get(series_name, {})
        if 'summary' in rolling_stats and rolling_stats['summary']['observations'] > 0:
            log.info(f"  {series_name.upper()}:")
            log.info(f"    ADF: {rolling_stats['summary']['adf_stationary_pct']:.1f}% of windows stationary")
            log.info(f"    KPSS: {rolling_stats['summary']['kpss_stationary_pct']:.1f}% of windows stationary")
    
    # Cointegration
    log.info("\n🔗 COINTEGRATION ANALYSIS:")
    log.info("  Static Engle-Granger Test:")
    log.info(f"    Statistic: {cointegration_results.get('statistic', np.nan):.4f}")
    log.info(f"    P-value: {cointegration_results.get('p_value', np.nan):.6f}")
    log.info(f"    Result: {'✅ Cointegrated at 5%' if cointegration_results.get('p_value', 1) < 0.05 else '❌ Not cointegrated at 5%'}")
    log.info(f"    Result: {'✅ Cointegrated at 1%' if cointegration_results.get('p_value', 1) < 0.01 else '❌ Not cointegrated at 1%'}")
    log.info(f"    Cointegration Equation: log({s1}) = {cointegration_results.get('alpha', 0):.4f} + "
            f"{cointegration_results.get('beta', 1):.4f} * log({s2}) + ε")
    log.info(f"    R²: {cointegration_results.get('r_squared', 0):.4f}")
    log.info(f"    Beta Standard Error: {cointegration_results.get('beta_std_error', 0):.4f}")
    
    # Half-life
    log.info("\n⏰ MEAN REVERSION ANALYSIS:")
    if 'error' not in half_life_results:
        ar1 = half_life_results.get('ar1', {})
        ou = half_life_results.get('ou', {})
        
        log.info("  AR(1) Method:")
        log.info(f"    Half-life: {ar1.get('half_life_days', np.nan):.2f} days "
                f"({ar1.get('half_life_hours', np.nan):.1f} hours)")
        log.info(f"    Alpha: {ar1.get('alpha', np.nan):.6f}")
        log.info(f"    Beta: {ar1.get('beta', np.nan):.6f}")
        log.info(f"    R²: {ar1.get('r_squared', np.nan):.4f}")
        log.info(f"    Beta p-value: {ar1.get('beta_pvalue', np.nan):.6f}")
        
        # Interpretation
        hl_days = ar1.get('half_life_days', np.inf)
        if hl_days < 2:
            log.info("    → FAST mean reversion (suitable for intraday strategies)")
        elif hl_days < 7:
            log.info("    → MODERATE mean reversion (suitable for daily strategies)")
        elif hl_days < 30:
            log.info("    → SLOW mean reversion (suitable for weekly strategies)")
        else:
            log.info("    → VERY SLOW or NO mean reversion")
        
        log.info("  Ornstein-Uhlenbeck Method:")
        log.info(f"    Half-life: {ou.get('half_life_days', np.nan):.2f} days")
        log.info(f"    Mean reversion speed (θ): {ou.get('theta', np.nan):.6f}")
        log.info(f"    Long-term mean (μ): {ou.get('mu', np.nan):.4f}")
        log.info(f"    Volatility (σ): {ou.get('sigma', np.nan):.4f}")
        log.info(f"    R²: {ou.get('r_squared', np.nan):.4f}")
    
    # Hurst
    log.info("\n📈 HURST EXPONENT ANALYSIS:")
    if 'error' not in hurst_results:
        log.info("  R/S Analysis:")
        log.info(f"    Hurst Exponent: {hurst_results.get('hurst', np.nan):.4f}")
        log.info(f"    R²: {hurst_results.get('r_squared', np.nan):.4f}")
        log.info(f"    P-value: {hurst_results.get('p_value', np.nan):.6f}")
        log.info(f"    Data points used: {hurst_results.get('n_lags', 0)}")
        log.info(f"    → {hurst_results.get('interpretation', 'Unknown')}")
    
    # Rolling Analysis Summary
    log.info("\n📊 ROLLING ANALYSIS SUMMARY:")
    for window in sorted(rolling_results.keys()):
        results = rolling_results[window]
        summary_stats = results['summary']
        log.info(f"  {window}-day Rolling Window:")
        log.info(f"    Correlation: current={summary_stats['corr_current']:.3f}, "
                f"mean={summary_stats['corr_mean']:.3f}")
        log.info(f"                stability={summary_stats['corr_std']:.3f}, "
                f"range=[{summary_stats['corr_min']:.3f}, {summary_stats['corr_max']:.3f}]")
        log.info(f"    Beta: current={summary_stats['beta_current']:.3f}, "
                f"mean={summary_stats['beta_mean']:.3f}, "
                f"stability={summary_stats['beta_std']:.3f}")
        log.info(f"    R²: current={summary_stats['r2_current']:.3f}, "
                f"mean={summary_stats['r2_mean']:.3f}")
        log.info(f"    Z-score: current={summary_stats['z_current']:.3f}")
        log.info(f"    Exceedances: >2σ={summary_stats['z_2sigma_pct']:.1f}%, "
                f">3σ={summary_stats['z_3sigma_pct']:.1f}%")
        log.info(f"    Vol Ratio: current={summary_stats['vol_ratio_current']:.3f}, "
                f"mean={summary_stats['vol_ratio_mean']:.3f}")
    
    # Trading Recommendations
    log.info("\n💡 PROFESSIONAL TRADING RECOMMENDATIONS:")
    if summary['proceed_to_backtest']:
        log.info(f"  ✅ RECOMMENDATION: Proceed with backtesting and paper trading")
        log.info(f"  🔍 Confidence Level: {summary['confidence_level']}")
        log.info(f"  ⚠️ Risk Category: {summary['risk_level']}")
        
        log.info("\n📋 SUGGESTED PARAMETERS:")
        hl_days = half_life_results.get('ar1', {}).get('half_life_days', 10)
        log.info(f"    • Entry threshold: ±2.0 to ±2.5 z-score")
        log.info(f"    • Exit threshold: ±0.5 to 0.0 z-score")
        log.info(f"    • Stop loss: ±3.5 to ±4.0 z-score")
        log.info(f"    • Holding period target: {hl_days*0.5:.0f} to {hl_days*2:.0f} days")
        log.info(f"    • Position sizing: Use Kelly Criterion with safety factor")
        log.info(f"    • Rebalance frequency: Daily or when beta changes >10%")
    else:
        log.info(f"  ❌ RECOMMENDATION: Do not proceed with this pair")
        log.info(f"  🔍 Confidence Level: {summary['confidence_level']}")
        log.info(f"  ⚠️ Risk Category: {summary['risk_level']}")
        
        log.info("\n🚨 IDENTIFIED ISSUES:")
        for metric, score in summary['scores'].items():
            if score < 60:
                log.info(f"    • {metric.replace('_', ' ').title()}: {score:.1f}/100 (needs >60)")
        
        log.info("\n🔄 ALTERNATIVE APPROACHES:")
        log.info("    • Try different pairs in same sector")
        log.info("    • Consider different time frequencies")
        log.info("    • Analyze fundamental correlation drivers")
        log.info("    • Look for structural breaks in relationship")
    
    # Market Context
    if 30 in rolling_results:
        z_current = rolling_results[30]['summary']['z_current']
        log.info("\n📈 MARKET CONTEXT:")
        log.info(f"  Current market position: Z-score = {z_current:.2f}")
        if abs(z_current) < 0.5:
            log.info("    📊 NEUTRAL - No immediate trading opportunity")
        elif abs(z_current) < 1.5:
            log.info("    📊 MILD signal - Monitor closely")
        elif abs(z_current) < 2.0:
            log.info("    📊 MODERATE signal - Prepare for potential trade")
        elif abs(z_current) < 3.0:
            log.info("    📊 STRONG signal - Consider entry if other conditions met")
        else:
            log.info("    ⚠️ EXTREME signal - Exercise caution, possible regime change")
    
    # Risk Warnings
    log.info("\n⚠️ RISK CONSIDERATIONS:")
    log.info("  • Monitor correlation breakdown during market stress")
    log.info("  • Be aware of sector-specific events affecting both assets")
    log.info("  • Consider liquidity constraints during volatile periods")
    log.info("  • Implement position sizing based on portfolio volatility")
    log.info("  • Regular revalidation of statistical relationships")
    
    log.info("\n" + "="*100)
    log.info("END OF COMPREHENSIVE ANALYSIS REPORT")
    log.info("="*100 + "\n")

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description="Professional Pair Trading Analysis - ENHANCED VERSION")
    parser.add_argument("--symbol1", type=str, required=True, 
                       help="First symbol (e.g., MEXCFTS_PERP_GIGA_USDT)")
    parser.add_argument("--symbol2", type=str, required=True,
                       help="Second symbol (e.g., MEXCFTS_PERP_SPX_USDT)")
    parser.add_argument("--windows", nargs='+', type=int, default=[3, 7, 15, 30, 60],
                       help="Rolling windows in days (default: 3 7 15 30 60)")
    parser.add_argument("--rolling-window", type=int, default=30,
                       help="Window for rolling statistical tests in days (default: 30)")
    parser.add_argument("--rolling-step", type=int, default=1,
                        help="Step in days between successive rolling-test evaluations (default: 1)")
    parser.add_argument("--inspect-schema", action="store_true",
                       help="Inspect database schema before analysis")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    
    args = parser.parse_args()
    
    log.info("🚀 Professional Pair Trading Analysis - ENHANCED VERSION")
    log.info(f"Analyzing pair: {args.symbol1} / {args.symbol2}")
    log.info(f"Rolling windows: {args.windows} days")
    log.info(f"Statistical tests rolling window: {args.rolling_window} days")
    log.info(f"Rolling test step size: {args.rolling_step} day(s)")
    
    try:
        # Inspect schema if requested
        if args.inspect_schema:
            inspect_database_schema()
        
        # Load data
        prices1, prices2 = load_complete_dataset(args.symbol1, args.symbol2)
        if prices1.empty or prices2.empty:
            log.error("No data available for analysis")
            return False
        
        # Calculate log prices
        log_p1 = np.log(prices1)
        log_p2 = np.log(prices2)
        
        # Cointegration analysis
        cointegration_results = calculate_cointegration_analysis(log_p1, log_p2)
        alpha = cointegration_results.get('cointegration_equation', {}).get('alpha', 0.0)
        beta = cointegration_results.get('cointegration_equation', {}).get('beta', 1.0)
        
        # Calculate spread
        spread = log_p1 - (beta * log_p2 + alpha)
        
        log.info(f"Using cointegration residual spread: spread = log_p1 - ({beta:.4f} * log_p2 + {alpha:.4f})")
        log.info(f"Using LOG PRICES for analysis:")
        log.info(f"  {args.symbol1}: price range {prices1.min():.2f} to {prices1.max():.2f}")
        log.info(f"  {args.symbol1}: log price range {log_p1.min():.4f} to {log_p1.max():.4f}")
        log.info(f"  Spread range: {spread.min():.4f} to {spread.max():.4f}")
        
        # Run all analyses
        log.info("\n" + "="*60)
        log.info("STATIONARITY TESTS (STATIC)")
        log.info("="*60)
        stationarity_results = {
            'symbol1_log': calculate_stationarity_tests(log_p1, f"log({args.symbol1})"),
            'symbol2_log': calculate_stationarity_tests(log_p2, f"log({args.symbol2})"),
            'spread': calculate_stationarity_tests(spread, "spread")
        }
        
        log.info("\n" + "="*60)
        log.info(f"STATIONARITY TESTS (ROLLING {args.rolling_window}-DAY)")
        log.info("="*60)
        rolling_stationarity = {
            'symbol1_log': calculate_rolling_stationarity_tests(
                log_p1, f"log({args.symbol1})", 
                window_days=args.rolling_window, step_days=args.rolling_step
            ),
            'symbol2_log': calculate_rolling_stationarity_tests(
                log_p2, f"log({args.symbol2})", 
                window_days=args.rolling_window, step_days=args.rolling_step
            ),
            'spread': calculate_rolling_stationarity_tests(
                spread, "spread", 
                window_days=args.rolling_window, step_days=args.rolling_step
            )
        }
        
        log.info("\n" + "="*60)
        log.info("COINTEGRATION ANALYSIS (STATIC)")
        log.info("="*60)
        # Already calculated above
        
        log.info("\n" + "="*60)
        log.info(f"COINTEGRATION ANALYSIS (ROLLING {args.rolling_window}-DAY)")
        log.info("="*60)
        rolling_cointegration = calculate_rolling_cointegration(
            log_p1, log_p2, 
            window_days=args.rolling_window, step_days=args.rolling_step
        )
        
        log.info("\n" + "="*60)
        log.info("HALF-LIFE ANALYSIS")
        log.info("="*60)
        half_life_results = calculate_half_life_analysis(spread)
        
        log.info("\n" + "="*60)
        log.info("HURST EXPONENT ANALYSIS")
        log.info("="*60)
        hurst_results = calculate_hurst_exponent(spread)
        
        log.info("\n" + "="*60)
        log.info("ROLLING WINDOW ANALYSIS")
        log.info("="*60)
        rolling_results = calculate_rolling_window_analysis(
            log_p1, log_p2, args.windows,
            alpha=alpha, beta=beta
        )
        
        log.info("\n" + "="*60)
        log.info("GENERATING PROFESSIONAL SUMMARY")
        log.info("="*60)
        summary = generate_professional_summary(
            args.symbol1, args.symbol2,
            stationarity_results, rolling_stationarity,
            cointegration_results, rolling_cointegration,
            half_life_results, hurst_results, rolling_results
        )
        
        if not args.no_plots:
            log.info("\n" + "="*60)
            log.info("CREATING COMPREHENSIVE VISUALIZATION")
            log.info("="*60)
            create_professional_visualization(
                args.symbol1, args.symbol2,
                log_p1, log_p2, alpha, beta,
                stationarity_results, rolling_stationarity,
                cointegration_results, rolling_cointegration,
                half_life_results, hurst_results, rolling_results, summary
            )
        
        # Generate detailed report
        generate_detailed_report(
            args.symbol1, args.symbol2,
            stationarity_results, rolling_stationarity,
            cointegration_results, rolling_cointegration,
            half_life_results, hurst_results, rolling_results, summary
        )
        
        log.info("✅ Professional pair analysis completed successfully!")
        log.info(f"\n🎯 QUICK SUMMARY:")
        log.info(f"Pair: {summary['pair_name']}")
        log.info(f"Score: {summary['overall_score']:.1f}/100")
        log.info(f"Recommendation: {summary['recommendation']}")
        log.info(f"Proceed to Backtest: {'YES ✅' if summary['proceed_to_backtest'] else 'NO ❌'}")
        
        if not args.no_plots:
            s1 = args.symbol1.split('_')[-2] if '_' in args.symbol1 else args.symbol1
            s2 = args.symbol2.split('_')[-2] if '_' in args.symbol2 else args.symbol2
            log.info(f"📊 Visualization saved: plots/{s1}_{s2}_comprehensive_analysis.png")
        
        return True
        
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)