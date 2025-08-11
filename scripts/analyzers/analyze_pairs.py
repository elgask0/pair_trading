#!/usr/bin/env python3
"""
🎯 DUAL-PERIOD PAIR TRADING ANALYSIS - Full Period vs Recent 3 Months
Enhanced with Cointegration and Signal Quality Analysis (FIXED Layout & Johansen)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
import time
from contextlib import contextmanager
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import combinations
warnings.filterwarnings('ignore')

# Ensure project root is on PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.database.connection import db_manager
from config.settings import settings
from src.utils.logger import get_logger

log = get_logger()

# Enhanced matplotlib settings
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 100

# ========== TIMING UTILITIES ==========
@contextmanager
def timer(description: str):
    """Context manager for timing operations"""
    start_time = time.time()
    log.info(f"⏳ {description}...")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        log.info(f"✅ {description} completed in {elapsed:.2f}s")

# ========== CONFIGURATION ==========
DEFAULT_WINDOWS = [3, 5, 7, 10, 15, 20, 30]  # Extended windows
MIN_OBSERVATIONS = 100
Z_ENTRY_THRESHOLD = 2.0
Z_EXIT_THRESHOLD = 0.0
Z_STOP_THRESHOLD = 4.0

@dataclass
class TrafficLightThresholds:
    """SUPER RELAXED thresholds for high-volatility memecoins"""
    # Beta stability
    beta_blue: float = 0.05      
    beta_green: float = 0.20     
    beta_yellow: float = 0.35    
    
    # R-squared
    r2_blue: float = 0.80        
    r2_green: float = 0.65       
    r2_yellow: float = 0.50      
    
    # Correlation
    corr_blue: float = 0.7      
    corr_green: float = 0.5     
    corr_yellow: float = 0.4    
    
    # ADF p-value
    adf_blue: float = 0.01
    adf_green: float = 0.05
    adf_yellow: float = 0.10
    
    # KPSS p-value
    kpss_blue: float = 0.05
    kpss_green: float = 0.02 
    kpss_yellow: float = 0.005
    
    # Half-life
    hl_blue_min: float = 0.05    
    hl_blue_max: float = 0.15    
    hl_green_max: float = 0.30   
    hl_yellow_max: float = 0.45
    
    # Hurst exponent
    hurst_blue_min: float = 0.30 
    hurst_blue_max: float = 0.50 
    hurst_green_max: float = 0.60 
    hurst_yellow_max: float = 0.70
    
thresholds = TrafficLightThresholds()

# Color schemes for multiple windows
WINDOW_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# ========== CORE STATISTICAL FUNCTIONS ==========
def robust_zscore(series: pd.Series, window: int, method: str = 'mad') -> pd.Series:
    """Calculate robust z-score using median and MAD or EWMA"""
    if method == 'mad':
        rolling_median = series.rolling(window=window).median()
        rolling_mad = series.rolling(window=window).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        zscore = 0.6745 * (series - rolling_median) / rolling_mad
    elif method == 'ewma':
        span = max(1, window // 4)
        ewma_mean = series.ewm(span=span).mean()
        ewma_std = series.ewm(span=span).std()
        zscore = (series - ewma_mean) / ewma_std
    else:
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        zscore = (series - rolling_mean) / rolling_std
    
    return zscore.fillna(0)

def adf_gls_test(series: pd.Series) -> Tuple[float, float]:
    """ADF-GLS test (more powerful than standard ADF)"""
    try:
        from arch.unitroot import DFGLS
        result = DFGLS(series.dropna())
        return result.stat, result.pvalue
    except ImportError:
        result = adfuller(series.dropna(), autolag='AIC')
        return result[0], result[1]

def johansen_cointegration_test(price1: pd.Series, price2: pd.Series) -> Dict:
    """IMPROVED: Johansen cointegration test with realistic thresholds for memecoins"""
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        # Prepare data
        data = np.column_stack([price1.dropna(), price2.dropna()])
        
        # Ensure we have enough data
        if len(data) < 50:
            return {'error': 'Insufficient data for cointegration test'}
        
        # Run Johansen test
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        
        # IMPROVED: Handle different statsmodels versions with better fallback
        try:
            # Try new format first (cvt and cve attributes)
            if hasattr(result, 'cvt') and hasattr(result, 'cve'):
                trace_crit_5pct = result.cvt[0, 1]
                trace_crit_1pct = result.cvt[0, 2]
                eigen_crit_5pct = result.cve[0, 1] 
                eigen_crit_1pct = result.cve[0, 2]
            elif hasattr(result, 'critical_values'):
                trace_crit_5pct = result.critical_values[0][1]
                trace_crit_1pct = result.critical_values[0][2]
                eigen_crit_5pct = result.critical_values[1][1]
                eigen_crit_1pct = result.critical_values[1][2]
            else:
                raise AttributeError("No critical values found")
                
        except (AttributeError, IndexError, TypeError):
            # IMPROVED: More realistic fallback values for memecoins
            # These are more lenient than traditional academic values
            trace_crit_5pct = 12.0   # Reduced from 15.495 (more permissive)
            trace_crit_1pct = 16.0   # Reduced from 20.262
            eigen_crit_5pct = 11.0   # Reduced from 14.265  
            eigen_crit_1pct = 15.0   # Reduced from 18.520
            log.warning(f"      Using relaxed critical values for memecoin cointegration")
        
        # IMPROVED: More nuanced cointegration strength assessment
        trace_stat = float(result.lr1[0])
        eigen_stat = float(result.lr2[0])
        
        # Calculate cointegration strength ratio
        trace_ratio = trace_stat / trace_crit_5pct
        eigen_ratio = eigen_stat / eigen_crit_5pct
        
        # IMPROVED: More realistic interpretation for memecoins
        if trace_ratio >= 1.0:  # Above 5% critical value
            if trace_stat > trace_crit_1pct:
                strength = "Strong"
            else:
                strength = "Moderate"
        elif trace_ratio >= 0.7:  # 70% of critical value (more lenient)
            strength = "Weak-Moderate"
        elif trace_ratio >= 0.4:  # 40% of critical value (very lenient for memecoins)
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        return {
            'trace_stat': trace_stat,
            'trace_crit_5pct': float(trace_crit_5pct),
            'trace_crit_1pct': float(trace_crit_1pct),
            'eigen_stat': eigen_stat, 
            'eigen_crit_5pct': float(eigen_crit_5pct),
            'eigen_crit_1pct': float(eigen_crit_1pct),
            'trace_ratio': trace_ratio,
            'eigen_ratio': eigen_ratio,
            'cointegrated_5pct': trace_stat > trace_crit_5pct,
            'cointegrated_1pct': trace_stat > trace_crit_1pct,
            'cointegrated_relaxed': trace_ratio >= 0.4,  # NEW: More lenient threshold
            'cointegration_strength': strength
        }
    except Exception as e:
        log.warning(f"      Johansen cointegration test failed: {e}")
        return {'error': str(e)}

def analyze_signal_quality(zscore: pd.Series, threshold: float = 2.0, resample_minutes: int = 1) -> Dict:
    """FIXED: Analyze trading signal quality with proper time conversion"""
    try:
        if len(zscore) < 10:
            return {}
        
        # Generate signals
        signals = np.where(zscore.abs() > threshold, np.sign(-zscore), 0)
        signal_changes = np.diff(signals)
        
        # Signal frequency
        signal_frequency = float((signals != 0).mean())
        
        # Signal duration analysis
        signal_durations = []
        current_duration = 0
        current_signal = 0
        
        for signal in signals:
            if signal != 0 and signal == current_signal:
                current_duration += 1
            else:
                if current_duration > 0:
                    signal_durations.append(current_duration)
                current_duration = 1 if signal != 0 else 0
                current_signal = signal
        
        if current_duration > 0:
            signal_durations.append(current_duration)
        
        # Signal consistency (less switching)
        signal_consistency = float(1 - (signal_changes != 0).mean()) if len(signal_changes) > 0 else 1.0
        
        # False signal analysis (signals that reverse quickly)
        false_signals = 0
        for i in range(len(signals) - 2):
            if signals[i] != 0 and signals[i+1] == 0 and signals[i+2] == -signals[i]:
                false_signals += 1
        
        false_signal_rate = float(false_signals / len(signals)) if len(signals) > 0 else 0.0
        
        # Signal strength analysis
        strong_signals = (zscore.abs() > threshold * 1.5).sum()
        extreme_signals = (zscore.abs() > threshold * 2).sum()
        
        # FIXED: Convert periods to hours based on resample frequency
        minutes_per_period = resample_minutes
        avg_duration_periods = float(np.mean(signal_durations)) if signal_durations else 0.0
        max_duration_periods = float(max(signal_durations)) if signal_durations else 0.0
        
        return {
            'signal_frequency': signal_frequency,
            'avg_signal_duration_periods': avg_duration_periods,
            'avg_signal_duration_hours': avg_duration_periods * minutes_per_period / 60.0,
            'max_signal_duration_periods': max_duration_periods,
            'max_signal_duration_hours': max_duration_periods * minutes_per_period / 60.0,
            'signal_consistency': signal_consistency,
            'false_signal_rate': false_signal_rate,
            'strong_signal_pct': float(strong_signals / len(zscore) * 100),
            'extreme_signal_pct': float(extreme_signals / len(zscore) * 100),
            'total_signals': float(len(signal_durations)),
            'signal_volatility': float(np.std(signals)) if len(signals) > 0 else 0.0
        }
        
    except Exception as e:
        log.warning(f"      Signal quality analysis failed: {e}")
        return {}

def estimate_ou_halflife(series: pd.Series) -> float:
    """Simplified OU half-life estimation"""
    try:
        series_clean = series.dropna()
        if len(series_clean) < 10:
            return np.nan
        
        def neg_log_likelihood(kappa):
            if kappa <= 0:
                return 1e6
            
            mu = series_clean.mean()
            diff = series_clean.diff().dropna()
            drift = series_clean.shift(1).dropna()
            
            min_len = min(len(diff), len(drift))
            diff = diff.iloc[:min_len]
            drift = drift.iloc[:min_len]
            
            expected_diff = -kappa * (drift - mu)
            residuals = diff - expected_diff
            sigma_sq = residuals.var()
            
            if sigma_sq <= 0:
                return 1e6
            
            log_lik = -0.5 * len(residuals) * np.log(2 * np.pi * sigma_sq) - 0.5 * np.sum(residuals**2) / sigma_sq
            return -log_lik
        
        result = minimize_scalar(neg_log_likelihood, bounds=(0.001, 10), method='bounded')
        
        if result.success:
            kappa_opt = result.x
            return np.log(2) / kappa_opt if kappa_opt > 0 else np.inf
        else:
            return np.nan
    
    except Exception as e:
        return np.nan

def resample_data(df: pd.DataFrame, resample_minutes: int) -> pd.DataFrame:
    """Resample OHLCV data to specified minute frequency"""
    if resample_minutes <= 1:
        return df
    
    log.info(f"    Resampling from 1min to {resample_minutes}min...")
    
    resample_rule = f'{resample_minutes}min'
    
    resampled = df.resample(resample_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Recalculate derived columns
    resampled['returns'] = resampled['close'].pct_change()
    resampled['log_returns'] = np.log(resampled['close'] / resampled['close'].shift(1))
    resampled['log_close'] = np.log(resampled['close'])
    
    log.info(f"    Resampled: {len(df)} -> {len(resampled)} records")
    
    return resampled

def get_available_symbols() -> List[str]:
    """Get all available symbols from database"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT DISTINCT symbol_id 
            FROM symbol_info 
            ORDER BY symbol_id
        """)).fetchall()
        return [row[0] for row in result]

# ========== ENHANCED DATA STRUCTURES ==========
@dataclass
class WindowMetrics:
    """Enhanced metrics with cointegration and signal quality"""
    window_days: int
    period_type: str  # 'full' or '3m'
    
    # Core regression metrics
    alpha: pd.Series = field(default_factory=pd.Series)
    beta: pd.Series = field(default_factory=pd.Series)
    spread: pd.Series = field(default_factory=pd.Series)
    zscore: pd.Series = field(default_factory=pd.Series)
    zscore_robust: pd.Series = field(default_factory=pd.Series)
    r_squared: pd.Series = field(default_factory=pd.Series)
    
    # Core statistical tests
    adf_statistic: float = np.nan
    adf_pvalue: float = np.nan
    adf_gls_statistic: float = np.nan
    adf_gls_pvalue: float = np.nan
    kpss_statistic: float = np.nan
    kpss_pvalue: float = np.nan
    
    # Mean reversion metrics
    half_life: float = np.nan
    half_life_ols: float = np.nan
    half_life_ou: float = np.nan
    hurst_exponent: float = np.nan
    variance_ratio: float = np.nan
    
    # Correlation metrics
    correlation_pearson: pd.Series = field(default_factory=pd.Series)
    corr_pearson_mean: float = np.nan
    correlation_spearman: pd.Series = field(default_factory=pd.Series)
    corr_spearman_mean: float = np.nan
    
    # Stability metrics
    alpha_mean: float = np.nan
    alpha_std: float = np.nan
    beta_mean: float = np.nan
    beta_std: float = np.nan
    spread_mean: float = np.nan
    spread_std: float = np.nan
    
    # Enhanced autocorrelation
    acf_values: np.ndarray = field(default_factory=lambda: np.array([]))
    ljung_box_pvalue: float = np.nan
    ljung_box_statistic: float = np.nan
    
    # Trading zones analysis
    zone_neutral_pct: float = np.nan
    zone_trading_pct: float = np.nan
    zone_extreme_pct: float = np.nan
    zone_neutral_pct_robust: float = np.nan
    zone_trading_pct_robust: float = np.nan
    zone_extreme_pct_robust: float = np.nan
    
    # Hourly patterns
    hourly_stats: Dict = field(default_factory=dict)
    
    # Signal quality metrics
    signal_quality: Dict = field(default_factory=dict)
    
    # Enhanced volatility metrics
    spread_volatility: float = np.nan
    zscore_volatility: float = np.nan
    max_drawdown: float = np.nan
    
    # Overall assessment
    is_tradeable: bool = False
    tradeable_score: float = np.inf
    
    # Traffic light assessments
    traffic_lights: Dict[str, str] = field(default_factory=dict)

@dataclass
class DualPeriodAnalysisResults:
    """Enhanced analysis results with cointegration"""
    symbol1: str
    symbol2: str
    pair_name: str
    
    # Data info
    data_start: datetime
    data_end: datetime
    total_observations: int
    resample_minutes: int = 1
    
    # 3-month period info
    period_3m_start: datetime = field(default_factory=datetime.now)
    period_3m_observations: int = 0
    
    # Window results for full period
    full_period_metrics: Dict[int, WindowMetrics] = field(default_factory=dict)
    
    # Window results for 3-month period
    three_month_metrics: Dict[int, WindowMetrics] = field(default_factory=dict)
    
    # Cointegration results
    cointegration_results: Dict = field(default_factory=dict)
    
    # Analysis warnings
    warnings: List[str] = field(default_factory=list)
    
    # Best windows for each period
    best_full_window: Optional[int] = None
    best_3m_window: Optional[int] = None
    
    # Current market state
    current_zscore: float = np.nan
    current_zscore_robust: float = np.nan
    current_alpha: float = np.nan
    current_beta: float = np.nan
    
    # Overall assessment
    is_suitable_full: bool = False
    is_suitable_3m: bool = False
    suitability_reasons_full: List[str] = field(default_factory=list)
    suitability_reasons_3m: List[str] = field(default_factory=list)

# ========== ENHANCED CORE ANALYSIS CLASS ==========
class DualPeriodPairAnalyzer:
    """Enhanced dual-period pair trading analyzer"""
    
    def __init__(self, symbol1: str, symbol2: str, 
                 days: int = 365, 
                 windows: List[int] = None,
                 robust_zscore_method: str = 'mad',
                 resample_minutes: int = 1):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.symbol1_short = symbol1.split('_')[-2] if '_' in symbol1 else symbol1
        self.symbol2_short = symbol2.split('_')[-2] if '_' in symbol2 else symbol2
        self.pair_name = f"{self.symbol1_short}-{self.symbol2_short}"
        
        self.days = days
        self.windows = windows or DEFAULT_WINDOWS
        self.robust_zscore_method = robust_zscore_method
        self.resample_minutes = resample_minutes
        
        self.df1: Optional[pd.DataFrame] = None
        self.df2: Optional[pd.DataFrame] = None
        self.aligned_data: Optional[pd.DataFrame] = None
        self.aligned_data_3m: Optional[pd.DataFrame] = None
        
        self.results = DualPeriodAnalysisResults(
            symbol1=symbol1,
            symbol2=symbol2,
            pair_name=self.pair_name,
            data_start=datetime.now(),
            data_end=datetime.now(),
            total_observations=0,
            resample_minutes=resample_minutes
        )
    
    def load_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """Load and resample OHLCV data for a symbol"""
        log.info(f"  Loading data for {symbol}...")
        
        cutoff = datetime.utcnow() - timedelta(days=self.days)
        
        with db_manager.get_session() as session:
            result = session.execute(text("""
                SELECT 
                    MIN(timestamp) as min_ts, 
                    MAX(timestamp) as max_ts,
                    COUNT(*) as total_records
                FROM ohlcv
                WHERE symbol = :symbol
                  AND timestamp >= :cutoff
            """), {'symbol': symbol, 'cutoff': cutoff}).fetchone()
            
            if not result or not result.min_ts:
                log.warning(f"  No data found for {symbol}")
                return pd.DataFrame()
            
            log.info(f"    Period: {result.min_ts} to {result.max_ts}")
            log.info(f"    Records: {result.total_records:,}")
            
            query = text("""
                SELECT 
                    timestamp,
                    open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = :symbol
                  AND timestamp >= :cutoff
                ORDER BY timestamp
            """)
            
            df = pd.read_sql(query, session.bind, 
                           params={'symbol': symbol, 'cutoff': cutoff}, 
                           index_col='timestamp')
            
            # Resample if needed
            if self.resample_minutes > 1:
                df = resample_data(df, self.resample_minutes)
            else:
                # Calculate derived columns for 1min data
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                df['log_close'] = np.log(df['close'])
            
            log.info(f"    Final records: {len(df):,}")
            
            return df
    
    def align_data(self) -> bool:
        """Align data from both symbols for both periods"""
        if self.df1 is None or self.df2 is None:
            return False
        
        common_index = self.df1.index.intersection(self.df2.index)
        
        if len(common_index) < MIN_OBSERVATIONS:
            log.error(f"  Insufficient common data: {len(common_index)}")
            return False
        
        # Full period data
        self.aligned_data = pd.DataFrame({
            f'{self.symbol1_short}_close': self.df1.loc[common_index, 'close'],
            f'{self.symbol2_short}_close': self.df2.loc[common_index, 'close'],
            f'{self.symbol1_short}_log_close': self.df1.loc[common_index, 'log_close'],
            f'{self.symbol2_short}_log_close': self.df2.loc[common_index, 'log_close'],
            f'{self.symbol1_short}_returns': self.df1.loc[common_index, 'returns'],
            f'{self.symbol2_short}_returns': self.df2.loc[common_index, 'returns'],
        })
        
        # 3-month period data
        cutoff_3m = datetime.utcnow() - timedelta(days=90)
        common_index_3m = common_index[common_index >= cutoff_3m]
        
        self.aligned_data_3m = self.aligned_data.loc[common_index_3m]
        
        # Store in results
        self.results.data_start = self.aligned_data.index[0]
        self.results.data_end = self.aligned_data.index[-1]
        self.results.total_observations = len(self.aligned_data)
        
        self.results.period_3m_start = self.aligned_data_3m.index[0] if len(self.aligned_data_3m) > 0 else cutoff_3m
        self.results.period_3m_observations = len(self.aligned_data_3m)
        
        log.info(f"  Aligned full period: {len(self.aligned_data):,} timestamps")
        log.info(f"  Aligned 3-month period: {len(self.aligned_data_3m):,} timestamps")
        
        return True
    
    def run_cointegration_analysis(self):
        """Run Johansen cointegration test"""
        log.info(f"    Running cointegration analysis...")
        
        log_price1 = self.aligned_data[f'{self.symbol1_short}_log_close']
        log_price2 = self.aligned_data[f'{self.symbol2_short}_log_close']
        
        self.results.cointegration_results = johansen_cointegration_test(log_price1, log_price2)
        
        if 'error' not in self.results.cointegration_results:
            coint = self.results.cointegration_results
            log.info(f"      Cointegration: {coint['cointegration_strength']} "
                    f"(Trace: {coint['trace_stat']:.3f} vs Crit: {coint['trace_crit_5pct']:.3f})")
            log.info(f"      Trace Ratio: {coint.get('trace_ratio', 0):.2f} (>0.4 = acceptable for memecoins)")
        else:
            log.warning(f"      Cointegration test failed: {self.results.cointegration_results['error']}")
    
    def analyze_window(self, window_days: int, data: pd.DataFrame, period_type: str) -> WindowMetrics:
        """Enhanced window analysis with signal quality metrics"""
        log.info(f"    Analyzing {window_days}d window for {period_type} period...")
        
        metrics = WindowMetrics(window_days=window_days, period_type=period_type)
        window_periods = window_days * 24 * 60 // self.resample_minutes
        
        log_price1 = data[f'{self.symbol1_short}_log_close']
        log_price2 = data[f'{self.symbol2_short}_log_close']
        returns1 = data[f'{self.symbol1_short}_returns']
        returns2 = data[f'{self.symbol2_short}_returns']
        
        # ===== 1. ROLLING REGRESSION ON LOG PRICES =====
        y = log_price1
        X = sm.add_constant(log_price2)
        
        roll = RollingOLS(y, X, window=window_periods)
        res = roll.fit()
        
        metrics.alpha = res.params.iloc[:, 0].dropna()
        metrics.beta = res.params.iloc[:, 1].dropna()
        metrics.r_squared = res.rsquared.dropna()
        
        if len(metrics.beta) < MIN_OBSERVATIONS:
            log.warning(f"      Insufficient data after rolling window for {window_days}d")
            return metrics
        
        # Calculate spread
        common_index = metrics.alpha.index.intersection(metrics.beta.index).intersection(log_price1.index).intersection(log_price2.index)
        
        if len(common_index) < MIN_OBSERVATIONS:
            log.warning(f"      Insufficient aligned data for spread calculation for {window_days}d")
            return metrics
        
        aligned_alpha = metrics.alpha.loc[common_index]
        aligned_beta = metrics.beta.loc[common_index]
        aligned_log_price1 = log_price1.loc[common_index]
        aligned_log_price2 = log_price2.loc[common_index]
        
        metrics.spread = aligned_log_price1 - aligned_alpha - aligned_beta * aligned_log_price2
        metrics.spread = metrics.spread.dropna()
        
        # Z-score calculations
        if len(metrics.spread) >= window_periods:
            spread_mean = metrics.spread.rolling(window_periods).mean()
            spread_std = metrics.spread.rolling(window_periods).std()
            metrics.zscore = (metrics.spread - spread_mean) / spread_std
            metrics.zscore = metrics.zscore.dropna()
            
            metrics.zscore_robust = robust_zscore(
                metrics.spread, window_periods, method=self.robust_zscore_method
            )
        
        # Statistics
        metrics.alpha_mean = metrics.alpha.mean()
        metrics.alpha_std = metrics.alpha.std()
        metrics.beta_mean = metrics.beta.mean()
        metrics.beta_std = metrics.beta.std()
        metrics.spread_mean = metrics.spread.mean()
        metrics.spread_std = metrics.spread.std()
        
        # Enhanced volatility metrics
        metrics.spread_volatility = metrics.spread.std()
        if len(metrics.zscore) > 0:
            metrics.zscore_volatility = metrics.zscore.std()
            # Calculate max drawdown
            rolling_max = metrics.zscore.expanding().max()
            drawdown = metrics.zscore - rolling_max
            metrics.max_drawdown = drawdown.min()
        
        # ===== 2. CORE STATIONARITY TESTS =====
        # Standard ADF Test
        adf_result = adfuller(metrics.spread.dropna(), autolag='AIC')
        metrics.adf_statistic = adf_result[0]
        metrics.adf_pvalue = adf_result[1]
        
        # ADF-GLS test
        metrics.adf_gls_statistic, metrics.adf_gls_pvalue = adf_gls_test(metrics.spread)
        
        # KPSS Test
        try:
            nlags = min(12, int(len(metrics.spread)**0.25))
            kpss_result = kpss(metrics.spread.dropna(), regression='c', nlags=nlags)
            metrics.kpss_statistic = kpss_result[0]
            metrics.kpss_pvalue = kpss_result[1]
        except Exception as e:
            log.warning(f"      KPSS test failed for {window_days}d: {e}")
        
        # ===== 3. MEAN REVERSION METRICS =====
        # Traditional half-life (OLS method)
        metrics.half_life_ols = self.calculate_half_life_ols(metrics.spread)
        
        # Simplified OU estimation
        metrics.half_life_ou = estimate_ou_halflife(metrics.spread)
        metrics.half_life = metrics.half_life_ou if not np.isnan(metrics.half_life_ou) else metrics.half_life_ols
        
        # Hurst exponent
        metrics.hurst_exponent = self.calculate_hurst_exponent(metrics.spread)
        
        # Variance ratio
        metrics.variance_ratio = self.calculate_variance_ratio(metrics.spread)
        
        # ===== 4. ENHANCED CORRELATION ANALYSIS =====
        metrics.correlation_pearson = returns1.rolling(window_periods).corr(returns2).dropna()
        metrics.corr_pearson_mean = metrics.correlation_pearson.mean()
        
        # Add Spearman correlation
        try:
            metrics.correlation_spearman = returns1.rolling(window_periods).apply(
                lambda x: returns2.loc[x.index].corr(x, method='spearman'), raw=False
            ).dropna()
            metrics.corr_spearman_mean = metrics.correlation_spearman.mean()
        except Exception as e:
            log.warning(f"      Spearman correlation failed for {window_days}d: {e}")
            metrics.corr_spearman_mean = np.nan
        
        # ===== 5. ENHANCED AUTOCORRELATION ANALYSIS =====
        try:
            max_lags = min(20, len(metrics.spread)//4)
            if max_lags > 0:
                metrics.acf_values = acf(metrics.spread.dropna(), nlags=max_lags)
            
            # Enhanced Ljung-Box test with better error handling
            if len(metrics.spread.dropna()) > 20:
                try:
                    lags = min(10, len(metrics.spread.dropna())//10)
                    log.debug(f"      Running Ljung-Box test with {lags} lags on {len(metrics.spread.dropna())} observations")
                    
                    lb_result = acorr_ljungbox(metrics.spread.dropna(), lags=lags, return_df=True)
                    
                    if isinstance(lb_result, pd.DataFrame):
                        metrics.ljung_box_statistic = lb_result['lb_stat'].iloc[-1]
                        metrics.ljung_box_pvalue = lb_result['lb_pvalue'].iloc[-1]
                    else:
                        # Fallback for older statsmodels versions
                        lb_result_old = acorr_ljungbox(metrics.spread.dropna(), lags=lags, return_df=False)
                        if isinstance(lb_result_old, tuple) and len(lb_result_old) >= 2:
                            metrics.ljung_box_statistic = float(lb_result_old[0])
                            metrics.ljung_box_pvalue = float(lb_result_old[1])
                    
                    log.debug(f"      Ljung-Box: stat={metrics.ljung_box_statistic:.4f}, p-val={metrics.ljung_box_pvalue:.4f}")
                    
                except Exception as e:
                    log.warning(f"      Ljung-Box test failed for {window_days}d: {type(e).__name__}: {str(e)}")
                    metrics.ljung_box_pvalue = np.nan
                    metrics.ljung_box_statistic = np.nan
            else:
                log.warning(f"      Insufficient data for Ljung-Box test: {len(metrics.spread.dropna())} observations")
                
        except Exception as e:
            log.warning(f"      Autocorrelation analysis failed for {window_days}d: {e}")
        
        # ===== 6. TRADING ZONES ANALYSIS =====
        if len(metrics.zscore.dropna()) > 0:
            z_abs = metrics.zscore.abs()
            metrics.zone_neutral_pct = (z_abs < 2).mean() * 100
            metrics.zone_trading_pct = ((z_abs >= 2) & (z_abs < 4)).mean() * 100
            metrics.zone_extreme_pct = (z_abs >= 4).mean() * 100
        
        if len(metrics.zscore_robust.dropna()) > 0:
            z_robust_abs = metrics.zscore_robust.abs()
            metrics.zone_neutral_pct_robust = (z_robust_abs < 2).mean() * 100
            metrics.zone_trading_pct_robust = ((z_robust_abs >= 2) & (z_robust_abs < 4)).mean() * 100
            metrics.zone_extreme_pct_robust = (z_robust_abs >= 4).mean() * 100
        
        # ===== 7. HOURLY PATTERNS =====
        z_for_hourly = metrics.zscore_robust if len(metrics.zscore_robust) > 0 else metrics.zscore
        metrics.hourly_stats = self.analyze_hourly_patterns(z_for_hourly)
        
        # ===== 8. ENHANCED: SIGNAL QUALITY METRICS (FIXED TIME CONVERSION) =====
        z_for_signals = metrics.zscore_robust if len(metrics.zscore_robust) > 0 else metrics.zscore
        metrics.signal_quality = analyze_signal_quality(z_for_signals, resample_minutes=self.resample_minutes)
        
        # ===== 9. TRAFFIC LIGHT EVALUATION =====
        metrics.traffic_lights = self.evaluate_traffic_lights(metrics)
        
        # ===== 10. TRADABILITY ASSESSMENT =====
        metrics.is_tradeable = self.assess_tradability(metrics)
        metrics.tradeable_score = self.calculate_tradeable_score(metrics)
        
        return metrics
    
    def calculate_half_life_ols(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion (OLS method)"""
        try:
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            spread_lag = spread_lag[1:]
            spread_diff = spread_diff[1:]
            
            model = sm.OLS(spread_diff, sm.add_constant(spread_lag)).fit()
            halflife = -np.log(2) / model.params[1] if model.params[1] < 0 else np.inf
            
            return halflife if halflife > 0 else np.inf
        except:
            return np.inf
    
    def calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 50) -> float:
        """Calculate Hurst exponent"""
        try:
            series = series.dropna()
            lags = range(2, min(max_lag, len(series)//4))
            tau = []
            
            for lag in lags:
                std_dev = []
                for start in range(0, len(series) - lag, lag):
                    std_dev.append(np.std(series[start:start+lag]))
                if std_dev:
                    tau.append(np.mean(std_dev))
            
            if len(tau) < 3:
                return 0.5
                
            reg = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            return reg[0]
        except:
            return 0.5
    
    def calculate_variance_ratio(self, series: pd.Series, lag: int = 2) -> float:
        """Calculate variance ratio test statistic"""
        try:
            series = series.dropna()
            
            var_1 = np.var(series.diff().dropna())
            var_k = np.var(series.diff(lag).dropna()) / lag
            
            vr = var_k / var_1 if var_1 > 0 else 1.0
            
            return vr
        except:
            return 1.0
    
    def analyze_hourly_patterns(self, zscore: pd.Series) -> Dict:
        """Analyze hourly patterns in z-score"""
        try:
            hourly_stats = {}
            zscore_df = zscore.to_frame('zscore')
            zscore_df['hour'] = zscore_df.index.hour
            
            for hour in range(24):
                hour_data = zscore_df[zscore_df['hour'] == hour]['zscore']
                if len(hour_data) > 0:
                    hourly_stats[hour] = {
                        'mean': hour_data.mean(),
                        'std': hour_data.std(),
                        'count': len(hour_data),
                        'extreme_pct': (hour_data.abs() > 2).mean() * 100
                    }
            
            return hourly_stats
        except:
            return {}
    
    def evaluate_traffic_lights(self, metrics: WindowMetrics) -> Dict[str, str]:
        """Evaluate 4-color traffic lights with ULTRA RELAXED KPSS thresholds"""
        lights = {}
        
        # Beta stability
        if metrics.beta_std < thresholds.beta_blue:
            lights['beta_stability'] = 'blue'
        elif metrics.beta_std < thresholds.beta_green:
            lights['beta_stability'] = 'green'
        elif metrics.beta_std < thresholds.beta_yellow:
            lights['beta_stability'] = 'yellow'
        else:
            lights['beta_stability'] = 'red'
        
        # R-squared
        r2_mean = metrics.r_squared.mean()
        if r2_mean > thresholds.r2_blue:
            lights['r_squared'] = 'blue'
        elif r2_mean > thresholds.r2_green:
            lights['r_squared'] = 'green'
        elif r2_mean > thresholds.r2_yellow:
            lights['r_squared'] = 'yellow'
        else:
            lights['r_squared'] = 'red'
        
        # ADF p-value (use ADF-GLS if available)
        adf_pval = metrics.adf_gls_pvalue if not np.isnan(metrics.adf_gls_pvalue) else metrics.adf_pvalue
        if adf_pval < thresholds.adf_blue:
            lights['adf'] = 'blue'
        elif adf_pval < thresholds.adf_green:
            lights['adf'] = 'green'
        elif adf_pval < thresholds.adf_yellow:
            lights['adf'] = 'yellow'
        else:
            lights['adf'] = 'red'
        
        # KPSS p-value (ULTRA RELAXED for high-volatility memecoins)
        if metrics.kpss_pvalue > thresholds.kpss_blue:
            lights['kpss'] = 'blue'
        elif metrics.kpss_pvalue > thresholds.kpss_green:
            lights['kpss'] = 'green'
        elif metrics.kpss_pvalue > thresholds.kpss_yellow:
            lights['kpss'] = 'yellow'
        else:
            lights['kpss'] = 'red'
        
        # Half-life
        window_periods = metrics.window_days * 24 * 60 // self.resample_minutes
        hl_pct = metrics.half_life / window_periods if window_periods > 0 else 1.0
        
        if thresholds.hl_blue_min <= hl_pct <= thresholds.hl_blue_max:
            lights['half_life'] = 'blue'
        elif hl_pct <= thresholds.hl_green_max:
            lights['half_life'] = 'green'
        elif hl_pct <= thresholds.hl_yellow_max:
            lights['half_life'] = 'yellow'
        else:
            lights['half_life'] = 'red'
        
        # Correlation
        if metrics.corr_pearson_mean > thresholds.corr_blue:
            lights['correlation'] = 'blue'
        elif metrics.corr_pearson_mean > thresholds.corr_green:
            lights['correlation'] = 'green'
        elif metrics.corr_pearson_mean > thresholds.corr_yellow:
            lights['correlation'] = 'yellow'
        else:
            lights['correlation'] = 'red'
        
        # Hurst exponent
        if thresholds.hurst_blue_min <= metrics.hurst_exponent <= thresholds.hurst_blue_max:
            lights['hurst'] = 'blue'
        elif metrics.hurst_exponent <= thresholds.hurst_green_max:
            lights['hurst'] = 'green'
        elif metrics.hurst_exponent <= thresholds.hurst_yellow_max:
            lights['hurst'] = 'yellow'
        else:
            lights['hurst'] = 'red'
        
        return lights
    
    def assess_tradability(self, metrics: WindowMetrics) -> bool:
        """More realistic tradability assessment"""
        
        # Core metrics with weights
        core_metrics = {
            'adf': 3,           # Most important
            'r_squared': 2,     # Very important  
            'beta_stability': 2, # Very important
            'half_life': 1,     # Important
            'correlation': 1,   # Important
        }
        
        score = 0
        max_score = 0
        
        for metric, weight in core_metrics.items():
            color = metrics.traffic_lights.get(metric, 'red')
            max_score += weight * 3
            
            if color == 'blue':
                score += weight * 3
            elif color == 'green':
                score += weight * 2
            elif color == 'yellow':
                score += weight * 1
            # red = 0 points
        
        # Lower threshold to 50%
        tradeable = (score / max_score) >= 0.5
        
        # More relaxed safety checks
        extreme_pct = metrics.zone_extreme_pct_robust if not np.isnan(metrics.zone_extreme_pct_robust) else metrics.zone_extreme_pct
        if extreme_pct > 12:
            return False
            
        if metrics.corr_pearson_mean < 0.20:
            return False
        
        # Keep ADF requirement (critical for stationarity)
        adf_color = metrics.traffic_lights.get('adf', 'red')
        if adf_color == 'red':
            return False
        
        return tradeable
    
    def calculate_tradeable_score(self, metrics: WindowMetrics) -> float:
        """Calculate a tradability score (lower is better)"""
        score = 0.0
        
        adf_pval = metrics.adf_gls_pvalue if not np.isnan(metrics.adf_gls_pvalue) else metrics.adf_pvalue
        
        score += adf_pval * 10
        score += (1 - metrics.r_squared.mean()) * 5
        score += metrics.beta_std * 15
        score += abs(metrics.hurst_exponent - 0.4) * 10
        
        extreme_pct = metrics.zone_extreme_pct_robust if not np.isnan(metrics.zone_extreme_pct_robust) else metrics.zone_extreme_pct
        score += extreme_pct * 2
        
        return score
    
    def validate_pair_quality(self):
        """IMPROVED: More nuanced warning system with relaxed cointegration"""
        warnings = []
        
        # IMPROVED: More nuanced cointegration warning
        if 'error' not in self.results.cointegration_results:
            coint = self.results.cointegration_results
            trace_ratio = coint.get('trace_ratio', 0)
            
            if not coint.get('cointegrated_5pct', False):
                if coint.get('cointegrated_relaxed', False):  # NEW: Check relaxed threshold
                    warnings.append(f"📊 Moderate cointegration detected (Ratio: {trace_ratio:.2f}, acceptable for memecoins)")
                else:
                    warnings.append(f"📊 Weak cointegration (Ratio: {trace_ratio:.2f} < 0.4 threshold)")
        else:
            warnings.append(f"🔧 Cointegration test needs attention: {self.results.cointegration_results['error']}")
        
        # KPSS Warnings (more nuanced)
        kpss_red_count = 0
        for window, metrics in self.results.full_period_metrics.items():
            if metrics.kpss_pvalue < thresholds.kpss_yellow:
                kpss_red_count += 1
        
        if kpss_red_count == len(self.results.full_period_metrics):
            warnings.append(f"⚠️  All windows show non-stationarity (KPSS < {thresholds.kpss_yellow}). Normal for high-volatility memecoins.")
        elif kpss_red_count > len(self.results.full_period_metrics) // 2:
            warnings.append(f"⚠️  {kpss_red_count}/{len(self.results.full_period_metrics)} windows have stationarity concerns")
        
        # Signal Quality Warnings (more targeted)
        low_quality_windows = []
        for window, metrics in self.results.full_period_metrics.items():
            if metrics.signal_quality:
                sig = metrics.signal_quality
                if (sig.get('false_signal_rate', 0) > 0.15 or 
                    sig.get('signal_frequency', 0) < 0.03):
                    low_quality_windows.append(f"{window}d")
        
        if low_quality_windows:
            warnings.append(f"📡 Signal quality concerns: {', '.join(low_quality_windows)} windows")
        
        # Data Quality Warnings
        if self.results.period_3m_observations < 1000:
            warnings.append(f"📅 Limited recent data: {self.results.period_3m_observations:,} 3M observations")
        
        self.results.warnings = warnings
    
    def analyze_all_windows(self):
        """Enhanced analysis including cointegration and validation"""
        log.info(f"\nAnalyzing pair: {self.pair_name}")
        log.info(f"  Windows: {self.windows} days")
        log.info(f"  Resample: {self.resample_minutes} min")
        log.info(f"  Z-score method: {self.robust_zscore_method}")
        
        # Run cointegration analysis first
        with timer("Cointegration analysis"):
            self.run_cointegration_analysis()
        
        # Analyze full period
        log.info(f"\n=== FULL PERIOD ANALYSIS ===")
        for window in self.windows:
            metrics = self.analyze_window(window, self.aligned_data, "FULL")
            self.results.full_period_metrics[window] = metrics
        
        # Analyze 3-month period
        if len(self.aligned_data_3m) >= MIN_OBSERVATIONS:
            log.info(f"\n=== 3-MONTH PERIOD ANALYSIS ===")
            for window in self.windows:
                metrics = self.analyze_window(window, self.aligned_data_3m, "3M")
                self.results.three_month_metrics[window] = metrics
        else:
            log.warning("Insufficient data for 3-month analysis")
        
        # Validate pair quality and generate warnings
        self.validate_pair_quality()
        
        self.select_best_windows()
        self.assess_overall_suitability()
        self.get_current_market_state()
    
    def select_best_windows(self):
        """Select the best windows for each period"""
        # Best full period window
        best_score_full = float('inf')
        best_window_full = None
        
        for window, metrics in self.results.full_period_metrics.items():
            if metrics.is_tradeable and metrics.tradeable_score < best_score_full:
                best_score_full = metrics.tradeable_score
                best_window_full = window
        
        if best_window_full:
            self.results.best_full_window = best_window_full
            log.info(f"Best full period window: {best_window_full}d (score: {best_score_full:.2f})")
        
        # Best 3-month window
        best_score_3m = float('inf')
        best_window_3m = None
        
        for window, metrics in self.results.three_month_metrics.items():
            if metrics.is_tradeable and metrics.tradeable_score < best_score_3m:
                best_score_3m = metrics.tradeable_score
                best_window_3m = window
        
        if best_window_3m:
            self.results.best_3m_window = best_window_3m
            log.info(f"Best 3-month window: {best_window_3m}d (score: {best_score_3m:.2f})")
    
    def assess_overall_suitability(self):
        """Assess overall pair suitability for both periods"""
        # Full period assessment
        tradeable_windows_full = [w for w, m in self.results.full_period_metrics.items() if m.is_tradeable]
        
        if len(tradeable_windows_full) >= 1:
            self.results.is_suitable_full = True
            self.results.suitability_reasons_full.append(f"{len(tradeable_windows_full)} tradeable windows found")
        else:
            self.results.is_suitable_full = False
            self.results.suitability_reasons_full.append(f"Only {len(tradeable_windows_full)} tradeable windows")
        
        # 3-month period assessment
        tradeable_windows_3m = [w for w, m in self.results.three_month_metrics.items() if m.is_tradeable]
        
        if len(tradeable_windows_3m) >= 1:
            self.results.is_suitable_3m = True
            self.results.suitability_reasons_3m.append(f"{len(tradeable_windows_3m)} tradeable windows found")
        else:
            self.results.is_suitable_3m = False
            self.results.suitability_reasons_3m.append(f"Only {len(tradeable_windows_3m)} tradeable windows")
    
    def get_current_market_state(self):
        """Get current market state from best window"""
        if self.results.best_full_window:
            metrics = self.results.full_period_metrics[self.results.best_full_window]
            
            if len(metrics.zscore_robust) > 0:
                self.results.current_zscore_robust = metrics.zscore_robust.iloc[-1]
            
            self.results.current_zscore = metrics.zscore.iloc[-1] if len(metrics.zscore) > 0 else np.nan
            self.results.current_alpha = metrics.alpha.iloc[-1] if len(metrics.alpha) > 0 else np.nan
            self.results.current_beta = metrics.beta.iloc[-1] if len(metrics.beta) > 0 else np.nan
    
    def create_comprehensive_visualization(self):
        """FIXED: Create enhanced visualization with proper layout spacing"""
        log.info(f"\nCreating comprehensive dual-period visualization...")
        
        plots_dir = Path("plots") / "pair_analysis"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # FIXED: Create figure with better spacing (8 rows to avoid overlap)
        fig = plt.figure(figsize=(50, 40))
        gs = fig.add_gridspec(8, 6, hspace=0.45, wspace=0.3)  # Increased hspace
        
        fig.suptitle(f'{self.pair_name} - ENHANCED DUAL-PERIOD ANALYSIS', 
                    fontsize=28, fontweight='bold', y=0.995)
        
        # Add analysis info
        period_text = f"Full: {self.results.data_start.strftime('%Y-%m-%d')} to {self.results.data_end.strftime('%Y-%m-%d')} ({self.results.total_observations:,} obs) | "
        period_text += f"3M: {self.results.period_3m_start.strftime('%Y-%m-%d')} to {self.results.data_end.strftime('%Y-%m-%d')} ({self.results.period_3m_observations:,} obs)"
        fig.text(0.5, 0.985, period_text, ha='center', fontsize=11, style='italic')
        
        info_text = f"Windows: {self.windows} | Resample: {self.resample_minutes}min | Method: {self.robust_zscore_method.upper()}"
        fig.text(0.5, 0.975, info_text, ha='center', fontsize=10, style='italic', color='blue')
        
        # ROW 0: Price comparison and correlation
        ax1 = fig.add_subplot(gs[0, :3])
        self.plot_dual_period_price_comparison(ax1)
        
        ax2 = fig.add_subplot(gs[0, 3:])
        self.plot_correlation_evolution(ax2)
        
        # ROW 1: Window analyses for both periods (show top 4 windows)
        for i, window in enumerate(self.windows[:4]):
            col_start = i * 1.5
            ax = fig.add_subplot(gs[1, int(col_start):int(col_start+1.5)])
            self.plot_window_comparison(ax, window)
        
        # ROW 2: Statistical tests tables (full period)
        ax_tests_full = fig.add_subplot(gs[2, :6])
        self.plot_statistical_tests_table(ax_tests_full, "full")
        
        # ROW 3: Statistical tests 3M 
        ax_tests_3m = fig.add_subplot(gs[3, :6])
        self.plot_statistical_tests_table(ax_tests_3m, "3m")
        
        # ROW 4: Traffic light matrices (side by side)
        ax_tl_full = fig.add_subplot(gs[4, :3])
        self.plot_traffic_light_matrix(ax_tl_full, "full")
        
        ax_tl_3m = fig.add_subplot(gs[4, 3:])
        self.plot_traffic_light_matrix(ax_tl_3m, "3m")
        
        # ROW 5: FIXED - Cointegration and Signal Quality (side by side with proper spacing)
        ax_coint = fig.add_subplot(gs[5, :3])  # Left half
        self.plot_cointegration_analysis(ax_coint)
        
        ax_signals = fig.add_subplot(gs[5, 3:])  # Right half
        self.plot_signal_quality_dashboard(ax_signals)
        
        # ROW 6: Distribution analysis and trading zones
        ax_dist = fig.add_subplot(gs[6, :2])
        self.plot_spread_distributions(ax_dist)
        
        ax_zones = fig.add_subplot(gs[6, 2:4])
        self.plot_trading_zones(ax_zones)
        
        ax_hourly = fig.add_subplot(gs[6, 4:])
        self.plot_hourly_heatmap(ax_hourly)
        
        # ROW 7: Enhanced executive summary (new row to avoid overlap)
        ax_summary = fig.add_subplot(gs[7, :])
        self.plot_enhanced_executive_summary(ax_summary)
        
        # Save figure
        filename = f'{self.pair_name}_ENHANCED_DUAL_PERIOD_ANALYSIS_{self.resample_minutes}min.png'
        plt.savefig(plots_dir / filename, dpi=120, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        log.info(f"Enhanced visualization saved: plots/pair_analysis/{filename}")
    
    # ========== PLOTTING METHODS (IMPROVED) ==========
    
    def plot_dual_period_price_comparison(self, ax):
        """Plot price comparison with improved legend and 3M highlighting"""
        log_price1 = self.aligned_data[f'{self.symbol1_short}_log_close']
        log_price2 = self.aligned_data[f'{self.symbol2_short}_log_close']
        
        ax2 = ax.twinx()
        
        # Plot full period (lighter colors)
        line1 = ax.plot(log_price1.index, log_price1, label=f'{self.symbol1_short} (Full)', 
                       linewidth=1, alpha=0.5, color='lightblue')
        line2 = ax2.plot(log_price2.index, log_price2, label=f'{self.symbol2_short} (Full)', 
                        linewidth=1, alpha=0.5, color='orange')
        
        # Highlight 3-month period (darker, thicker lines)
        if len(self.aligned_data_3m) > 0:
            log_price1_3m = self.aligned_data_3m[f'{self.symbol1_short}_log_close']
            log_price2_3m = self.aligned_data_3m[f'{self.symbol2_short}_log_close']
            
            line3 = ax.plot(log_price1_3m.index, log_price1_3m, 
                   linewidth=2.5, alpha=0.9, color='darkblue', label=f'{self.symbol1_short} (3M)')
            line4 = ax2.plot(log_price2_3m.index, log_price2_3m, 
                    linewidth=2.5, alpha=0.9, color='darkorange', label=f'{self.symbol2_short} (3M)')
        
        ax.set_title(f'Dual-Period Log-Price Comparison', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'Log Price - {self.symbol1_short}', color='darkblue')
        ax2.set_ylabel(f'Log Price - {self.symbol2_short}', color='darkorange')
        
        ax.tick_params(axis='y', labelcolor='darkblue')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        ax.grid(True, alpha=0.3)
    
    def plot_correlation_evolution(self, ax):
        """Plot correlation evolution for all windows with adjusted Y-axis"""
        # Plot all windows in different colors
        for i, window in enumerate(self.windows):
            color = WINDOW_COLORS[i % len(WINDOW_COLORS)]
            
            # Full period correlation
            if window in self.results.full_period_metrics:
                metrics = self.results.full_period_metrics[window]
                if len(metrics.correlation_pearson) > 0:
                    corr_sample = metrics.correlation_pearson.iloc[::max(1, len(metrics.correlation_pearson)//200)]
                    ax.plot(corr_sample.index, corr_sample, 
                           label=f'{window}d (Full)', alpha=0.6, linewidth=1, color=color, linestyle='-')
            
            # 3-month correlation (dashed line, same color)
            if window in self.results.three_month_metrics:
                metrics = self.results.three_month_metrics[window]
                if len(metrics.correlation_pearson) > 0:
                    corr_sample = metrics.correlation_pearson.iloc[::max(1, len(metrics.correlation_pearson)//200)]
                    ax.plot(corr_sample.index, corr_sample, 
                           label=f'{window}d (3M)', alpha=0.8, linewidth=1.5, color=color, linestyle='--')
        
        ax.set_title('Return Correlation Evolution (All Windows)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Pearson Correlation')
        
        # Adjusted Y-axis for better visibility
        ax.set_ylim(-0.8, 1.0)
        ax.axhline(y=0.6, color='green', linestyle=':', alpha=0.5, label='Good (0.6)')
        ax.axhline(y=0.4, color='orange', linestyle=':', alpha=0.5, label='Fair (0.4)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    
    def plot_window_comparison(self, ax, window: int):
        """Plot window analysis comparison"""
        ax.set_title(f'{window}d Window Comparison', fontweight='bold', fontsize=11)
        
        # Plot full period if available
        if window in self.results.full_period_metrics:
            metrics_full = self.results.full_period_metrics[window]
            zscore_full = metrics_full.zscore_robust if len(metrics_full.zscore_robust) > 0 else metrics_full.zscore
            
            if len(zscore_full) > 0:
                sample_rate = max(1, len(zscore_full) // 500)
                zscore_sample = zscore_full.iloc[::sample_rate]
                ax.plot(zscore_sample.index, zscore_sample, color='blue', linewidth=1, alpha=0.6,
                       label='Full Period')
        
        # Plot 3-month period if available
        if window in self.results.three_month_metrics:
            metrics_3m = self.results.three_month_metrics[window]
            zscore_3m = metrics_3m.zscore_robust if len(metrics_3m.zscore_robust) > 0 else metrics_3m.zscore
            
            if len(zscore_3m) > 0:
                sample_rate = max(1, len(zscore_3m) // 500)
                zscore_sample = zscore_3m.iloc[::sample_rate]
                ax.plot(zscore_sample.index, zscore_sample, color='green', linewidth=1.5, alpha=0.8,
                       label='3-Month')
        
        # Add threshold lines
        ax.axhline(y=Z_ENTRY_THRESHOLD, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=-Z_ENTRY_THRESHOLD, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=Z_STOP_THRESHOLD, color='red', linestyle=':', alpha=0.7)
        ax.axhline(y=-Z_STOP_THRESHOLD, color='red', linestyle=':', alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_ylabel('Z-Score')
        ax.set_ylim(-6, 6)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # Add tradability status
        full_tradeable = window in self.results.full_period_metrics and self.results.full_period_metrics[window].is_tradeable
        three_m_tradeable = window in self.results.three_month_metrics and self.results.three_month_metrics[window].is_tradeable
        
        status_text = f"Full: {'YES' if full_tradeable else 'NO'} | 3M: {'YES' if three_m_tradeable else 'NO'}"
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    def plot_statistical_tests_table(self, ax, period_type: str):
        """Plot enhanced statistical tests table"""
        ax.axis('off')
        
        title = f"STATISTICAL TESTS - {'FULL PERIOD' if period_type == 'full' else '3-MONTH PERIOD'}"
        metrics_dict = self.results.full_period_metrics if period_type == 'full' else self.results.three_month_metrics
        
        headers = ['Window', 'ADF', 'ADF-GLS', 'KPSS', 'LB Stat', 'LB P-val', 'Tradeable']
        table_data = []
        
        for window in self.windows:
            if window in metrics_dict:
                m = metrics_dict[window]
                
                row = [
                    f'{window}d',
                    f'{m.adf_pvalue:.4f}' if not np.isnan(m.adf_pvalue) else 'N/A',
                    f'{m.adf_gls_pvalue:.4f}' if not np.isnan(m.adf_gls_pvalue) else 'N/A',
                    f'{m.kpss_pvalue:.4f}' if not np.isnan(m.kpss_pvalue) else 'N/A',
                    f'{m.ljung_box_statistic:.2f}' if not np.isnan(m.ljung_box_statistic) else 'N/A',
                    f'{m.ljung_box_pvalue:.4f}' if not np.isnan(m.ljung_box_pvalue) else 'N/A',
                    'YES' if m.is_tradeable else 'NO'
                ]
                table_data.append(row)
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 2.0)
            
            # Color cells with ULTRA RELAXED KPSS coloring
            for i in range(len(table_data)):
                # Color tradeable column
                tradeable = table_data[i][6] == 'YES'
                table[(i+1, 6)].set_facecolor('lightgreen' if tradeable else 'lightcoral')
                
                # Color p-value columns (ADF, ADF-GLS, LB p-val)
                for j in [1, 2, 5]:
                    if j < len(table_data[i]) and table_data[i][j] != 'N/A':
                        try:
                            pval = float(table_data[i][j])
                            if pval < 0.01:
                                color = '#4A90E2'  # Blue
                            elif pval < 0.05:
                                color = '#7ED321'  # Green
                            elif pval < 0.10:
                                color = '#F5A623'  # Yellow
                            else:
                                color = '#D0021B'  # Red
                            table[(i+1, j)].set_facecolor(color)
                            table[(i+1, j)].set_text_props(color='white')
                        except:
                            pass
                
                # KPSS (ULTRA RELAXED thresholds)
                if table_data[i][3] != 'N/A':
                    try:
                        pval = float(table_data[i][3])
                        if pval > thresholds.kpss_blue:  # > 0.05
                            color = '#4A90E2'  # Blue
                        elif pval > thresholds.kpss_green:  # > 0.02
                            color = '#7ED321'  # Green 
                        elif pval > thresholds.kpss_yellow:  # > 0.005
                            color = '#F5A623'  # Yellow
                        else:  # <= 0.005
                            color = '#D0021B'  # Red
                        table[(i+1, 3)].set_facecolor(color)
                        table[(i+1, 3)].set_text_props(color='white')
                    except:
                        pass
                
                # LB Statistic (neutral coloring)
                if table_data[i][4] != 'N/A':
                    table[(i+1, 4)].set_facecolor('#E8E8E8')
            
            # Style header
            for j in range(len(headers)):
                table[(0, j)].set_facecolor('#2C3E50')
                table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    def plot_traffic_light_matrix(self, ax, period_type: str):
        """Plot traffic light matrix with improved spacing"""
        ax.axis('off')
        
        title = f"TRAFFIC LIGHTS - {'FULL PERIOD' if period_type == 'full' else '3-MONTH PERIOD'}"
        metrics_dict = self.results.full_period_metrics if period_type == 'full' else self.results.three_month_metrics
        
        metrics_names = ['Beta\nStability', 'R²', 'ADF', 'KPSS', 'Half-life', 'Correlation', 'Hurst']
        
        matrix_data = []
        windows_list = []
        
        for window in self.windows:
            if window in metrics_dict:
                metrics = metrics_dict[window]
                lights = metrics.traffic_lights
                
                row = [
                    lights.get('beta_stability', 'gray'),
                    lights.get('r_squared', 'gray'),
                    lights.get('adf', 'gray'),
                    lights.get('kpss', 'gray'),
                    lights.get('half_life', 'gray'),
                    lights.get('correlation', 'gray'),
                    lights.get('hurst', 'gray')
                ]
                matrix_data.append(row)
                windows_list.append(f'{window}d')
        
        if matrix_data:
            color_map = {
                'blue': '#4A90E2',
                'green': '#7ED321',
                'yellow': '#F5A623',
                'red': '#D0021B',
                'gray': '#95A5A6'
            }
            
            table = ax.table(cellText=[['●'] * len(metrics_names) for _ in matrix_data],
                           rowLabels=windows_list,
                           colLabels=metrics_names,
                           cellLoc='center', loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.3, 2.2)
            
            # Color cells
            for i, row_colors in enumerate(matrix_data):
                for j, color in enumerate(row_colors):
                    table[(i+1, j)].set_facecolor(color_map[color])
                    table[(i+1, j)].set_text_props(color='white', fontsize=16, weight='bold')
            
            # Style headers
            for j in range(len(metrics_names)):
                table[(0, j)].set_facecolor('#34495e')
                table[(0, j)].set_text_props(weight='bold', color='white', fontsize=9)
            
            for i in range(len(windows_list)):
                table[(i+1, -1)].set_facecolor('#34495e')
                table[(i+1, -1)].set_text_props(weight='bold', color='white')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    def plot_cointegration_analysis(self, ax):
        """IMPROVED: Plot cointegration test results with relaxed interpretation"""
        ax.axis('off')
        
        if 'error' not in self.results.cointegration_results:
            coint = self.results.cointegration_results
            trace_ratio = coint.get('trace_ratio', 0)
            
            # Create cointegration summary table with relaxed interpretation
            headers = ['Test', 'Statistic', 'Critical 5%', 'Ratio', 'Result']
            data = [
                ['Johansen Trace', f"{coint['trace_stat']:.3f}", f"{coint['trace_crit_5pct']:.3f}", 
                 f"{trace_ratio:.2f}", coint['cointegration_strength']],
                ['Relaxed Threshold', '0.40', '1.00', 
                 f"{trace_ratio:.2f}", 'Pass' if trace_ratio >= 0.4 else 'Fail']
            ]
            
            table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)  # Smaller font to fit
            table.scale(1.1, 1.8)  # Smaller scale
            
            # Color results with more nuanced interpretation
            for i in range(len(data)):
                result_cell = table[(i+1, 4)]
                result = data[i][4]
                
                if 'Strong' in result or result == 'Pass':
                    result_cell.set_facecolor('lightgreen')
                elif 'Moderate' in result or 'Weak-Moderate' in result:
                    result_cell.set_facecolor('lightyellow')
                elif 'Weak' in result and trace_ratio >= 0.4:
                    result_cell.set_facecolor('lightblue')  # Acceptable for memecoins
                else:
                    result_cell.set_facecolor('lightcoral')
            
            # Style header
            for j in range(len(headers)):
                table[(0, j)].set_facecolor('#2C3E50')
                table[(0, j)].set_text_props(weight='bold', color='white', fontsize=7)
            
            # Add interpretation note
            interpretation = f"Memecoin-Adjusted: {'✅ ACCEPTABLE' if trace_ratio >= 0.4 else '❌ WEAK'}"
            ax.text(0.5, 0.15, interpretation, ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        else:
            # Show error message
            ax.text(0.5, 0.5, f"Cointegration Test Failed:\n{self.results.cointegration_results['error']}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax.set_title('Cointegration Analysis\n(Memecoin-Adjusted)', fontweight='bold', fontsize=12, pad=20)
    
    def plot_signal_quality_dashboard(self, ax):
        """ENHANCED: Plot signal quality dashboard with time conversion (properly spaced)"""
        ax.axis('off')
        
        if self.results.best_full_window and self.results.best_full_window in self.results.full_period_metrics:
            metrics = self.results.full_period_metrics[self.results.best_full_window]
            
            if metrics.signal_quality:
                sig = metrics.signal_quality
                
                # Create signal quality table with time conversion
                headers = ['Signal Metric', 'Value', 'Quality']
                data = [
                    ['Signal Frequency', f"{sig.get('signal_frequency', 0):.1%}", 
                     'Good' if sig.get('signal_frequency', 0) > 0.1 else 'Fair' if sig.get('signal_frequency', 0) > 0.05 else 'Low'],
                    ['Avg Duration (Hours)', f"{sig.get('avg_signal_duration_hours', 0):.1f}h", 
                     'Good' if sig.get('avg_signal_duration_hours', 0) > 2 else 'Fair' if sig.get('avg_signal_duration_hours', 0) > 0.5 else 'Short'],
                    ['False Signal Rate', f"{sig.get('false_signal_rate', 0):.1%}", 
                     'Low' if sig.get('false_signal_rate', 0) < 0.05 else 'Fair' if sig.get('false_signal_rate', 0) < 0.1 else 'High'],
                    ['Signal Consistency', f"{sig.get('signal_consistency', 0):.1%}", 
                     'High' if sig.get('signal_consistency', 0) > 0.8 else 'Fair' if sig.get('signal_consistency', 0) > 0.6 else 'Low']
                ]
                
                table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)  # Smaller font
                table.scale(1.1, 1.8)  # Proper scaling to match cointegration table
                
                # Color quality assessments
                for i in range(len(data)):
                    quality_cell = table[(i+1, 2)]
                    quality = data[i][2]
                    if quality in ['Good', 'High', 'Low'] and 'False Signal' not in data[i][0]:
                        quality_cell.set_facecolor('lightgreen')
                    elif quality in ['Fair']:
                        quality_cell.set_facecolor('lightyellow')
                    else:
                        if 'False Signal' in data[i][0] and quality == 'Low':
                            quality_cell.set_facecolor('lightgreen')  # Low false signal rate is good
                        else:
                            quality_cell.set_facecolor('lightcoral')
                
                # Style header
                for j in range(len(headers)):
                    table[(0, j)].set_facecolor('#2C3E50')
                    table[(0, j)].set_text_props(weight='bold', color='white', fontsize=7)
            else:
                ax.text(0.5, 0.5, "Signal quality metrics not available", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
        else:
            ax.text(0.5, 0.5, "No tradeable window available", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
        
        ax.set_title('Signal Quality Dashboard\n(Time-Adjusted)', fontweight='bold', fontsize=12, pad=20)
    
    def plot_spread_distributions(self, ax):
        """Plot spread distributions by window and period"""
        # Full period spreads
        for i, window in enumerate(self.windows[:4]):
            if window in self.results.full_period_metrics:
                spread = self.results.full_period_metrics[window].spread.dropna()
                if len(spread) > 0:
                    color = WINDOW_COLORS[i % len(WINDOW_COLORS)]
                    ax.hist(spread, bins=30, alpha=0.5, label=f'Full {window}d', 
                           density=True, color=color)
        
        ax.set_xlabel('Spread Value')
        ax.set_ylabel('Density')
        ax.set_title('Spread Distributions (Log-price)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_trading_zones(self, ax):
        """Plot time spent in trading zones comparison"""
        windows_list = []
        full_data = []
        three_m_data = []
        
        for window in self.windows:
            # Full period zones
            if window in self.results.full_period_metrics:
                metrics = self.results.full_period_metrics[window]
                full_data.append([
                    metrics.zone_neutral_pct_robust if not np.isnan(metrics.zone_neutral_pct_robust) else metrics.zone_neutral_pct,
                    metrics.zone_trading_pct_robust if not np.isnan(metrics.zone_trading_pct_robust) else metrics.zone_trading_pct,
                    metrics.zone_extreme_pct_robust if not np.isnan(metrics.zone_extreme_pct_robust) else metrics.zone_extreme_pct
                ])
            else:
                full_data.append([0, 0, 0])
            
            # 3-month zones
            if window in self.results.three_month_metrics:
                metrics = self.results.three_month_metrics[window]
                three_m_data.append([
                    metrics.zone_neutral_pct_robust if not np.isnan(metrics.zone_neutral_pct_robust) else metrics.zone_neutral_pct,
                    metrics.zone_trading_pct_robust if not np.isnan(metrics.zone_trading_pct_robust) else metrics.zone_trading_pct,
                    metrics.zone_extreme_pct_robust if not np.isnan(metrics.zone_extreme_pct_robust) else metrics.zone_extreme_pct
                ])
            else:
                three_m_data.append([0, 0, 0])
            
            windows_list.append(f'{window}d')
        
        if full_data:
            full_data = np.array(full_data)
            three_m_data = np.array(three_m_data)
            
            x = np.arange(len(windows_list))
            width = 0.35
            
            # Full period bars (left)
            ax.bar(x - width/2, full_data[:, 0], width/2, label='Neutral (Full)', color='green', alpha=0.7)
            ax.bar(x - width/2, full_data[:, 1], width/2, bottom=full_data[:, 0], 
                  label='Trading (Full)', color='yellow', alpha=0.7)
            ax.bar(x - width/2, full_data[:, 2], width/2, bottom=full_data[:, 0] + full_data[:, 1],
                  label='Extreme (Full)', color='red', alpha=0.7)
            
            # 3-month bars (right)
            ax.bar(x + width/2, three_m_data[:, 0], width/2, label='Neutral (3M)', color='green', alpha=1.0)
            ax.bar(x + width/2, three_m_data[:, 1], width/2, bottom=three_m_data[:, 0], 
                  label='Trading (3M)', color='yellow', alpha=1.0)
            ax.bar(x + width/2, three_m_data[:, 2], width/2, bottom=three_m_data[:, 0] + three_m_data[:, 1],
                  label='Extreme (3M)', color='red', alpha=1.0)
            
            ax.set_xlabel('Window')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Trading Zones: Full vs 3-Month', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(windows_list)
            ax.legend(loc='best', fontsize=7)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y')
    
    def plot_hourly_heatmap(self, ax):
        """Plot hourly pattern heatmap for best window"""
        if self.results.best_full_window and self.results.best_full_window in self.results.full_period_metrics:
            hourly = self.results.full_period_metrics[self.results.best_full_window].hourly_stats
            
            if hourly:
                hours = list(range(24))
                extreme_pcts = [hourly.get(h, {}).get('extreme_pct', 0) for h in hours]
                
                heatmap_data = np.array(extreme_pcts).reshape(1, 24)
                
                im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', 
                             vmin=0, vmax=max(extreme_pcts)*1.2 if extreme_pcts else 1)
                
                ax.set_xticks(range(0, 24, 2))
                ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)])
                ax.set_yticks([0])
                ax.set_yticklabels(['Signal %'])
                ax.set_xlabel('Hour of Day')
                ax.set_title(f'Hourly Trading Signals (Full {self.results.best_full_window}d)', 
                           fontsize=11, fontweight='bold')
                
                plt.colorbar(im, ax=ax, label='% Time |z|>2')
                
                for h in range(24):
                    if h in hourly:
                        ax.text(h, 0, f'{extreme_pcts[h]:.0f}', ha='center', va='center', 
                                color='white' if extreme_pcts[h] > max(extreme_pcts)*0.5 else 'black',
                               fontsize=7)
    
    def plot_enhanced_executive_summary(self, ax):
        """ENHANCED: Plot comprehensive executive summary with improved cointegration interpretation"""
        ax.axis('off')
        
        summary_text = "ENHANCED DUAL-PERIOD EXECUTIVE SUMMARY\n"
        summary_text += "=" * 100 + "\n\n"
        
        # IMPROVED: Cointegration Summary with relaxed interpretation
        if 'error' not in self.results.cointegration_results:
            coint = self.results.cointegration_results
            trace_ratio = coint.get('trace_ratio', 0)
            
            summary_text += f"COINTEGRATION ANALYSIS (MEMECOIN-ADJUSTED):\n"
            summary_text += f"  STATUS: {coint['cointegration_strength']} cointegration detected\n"
            summary_text += f"  Johansen Trace: {coint['trace_stat']:.3f} (Critical 5%: {coint['trace_crit_5pct']:.3f})\n"
            summary_text += f"  Trace Ratio: {trace_ratio:.2f} ({'✅ ACCEPTABLE' if trace_ratio >= 0.4 else '❌ WEAK'} for memecoins)\n"
            summary_text += f"  Traditional Test: {'PASSED' if coint['cointegrated_5pct'] else 'FAILED'}\n"
            summary_text += f"  Relaxed Test: {'✅ PASSED' if coint.get('cointegrated_relaxed', False) else '❌ FAILED'} (Ratio ≥ 0.4)\n"
        else:
            summary_text += f"COINTEGRATION ANALYSIS:\n"
            summary_text += f"  STATUS: Test failed - {self.results.cointegration_results['error']}\n"
        
        summary_text += "\n"
        
        # Full period summary
        summary_text += "FULL PERIOD ANALYSIS:\n"
        if self.results.is_suitable_full:
            summary_text += f"  STATUS: ✅ SUITABLE FOR TRADING\n"
            summary_text += f"  BEST WINDOW: {self.results.best_full_window or 'None'}d\n"
        else:
            summary_text += f"  STATUS: ❌ NOT SUITABLE FOR TRADING\n"
        
        for reason in self.results.suitability_reasons_full:
            summary_text += f"  • {reason}\n"
        
        if self.results.best_full_window:
            metrics = self.results.full_period_metrics[self.results.best_full_window]
            adf_pval = metrics.adf_gls_pvalue if not np.isnan(metrics.adf_gls_pvalue) else metrics.adf_pvalue
            summary_text += f"  Key Metrics: ADF={adf_pval:.4f}, R²={metrics.r_squared.mean():.3f}, β-std={metrics.beta_std:.3f}\n"
            summary_text += f"  Half-life: {metrics.half_life:.0f}p, Correlation: {metrics.corr_pearson_mean:.3f}\n"
            
            # Enhanced: Signal quality with time conversion
            if metrics.signal_quality:
                sig = metrics.signal_quality
                summary_text += f"  Signals: Freq={sig.get('signal_frequency', 0):.1%}, Duration={sig.get('avg_signal_duration_hours', 0):.1f}h, False={sig.get('false_signal_rate', 0):.1%}\n"
        
        summary_text += "\n"
        
        # 3-month period summary
        summary_text += "3-MONTH PERIOD ANALYSIS:\n"
        if self.results.three_month_metrics:
            if self.results.is_suitable_3m:
                summary_text += f"  STATUS: ✅ SUITABLE FOR TRADING\n"
                summary_text += f"  BEST WINDOW: {self.results.best_3m_window or 'None'}d\n"
            else:
                summary_text += f"  STATUS: ❌ NOT SUITABLE FOR TRADING\n"
            
            for reason in self.results.suitability_reasons_3m:
                summary_text += f"  • {reason}\n"
            
            if self.results.best_3m_window:
                metrics = self.results.three_month_metrics[self.results.best_3m_window]
                adf_pval = metrics.adf_gls_pvalue if not np.isnan(metrics.adf_gls_pvalue) else metrics.adf_pvalue
                summary_text += f"  Key Metrics: ADF={adf_pval:.4f}, R²={metrics.r_squared.mean():.3f}, β-std={metrics.beta_std:.3f}\n"
                summary_text += f"  Half-life: {metrics.half_life:.0f}p, Correlation: {metrics.corr_pearson_mean:.3f}\n"
        else:
            summary_text += f"  STATUS: INSUFFICIENT DATA\n"
        
        summary_text += "\n"
        
        # ENHANCED: Warnings Section with improved cointegration interpretation
        if self.results.warnings:
            summary_text += "ANALYSIS WARNINGS:\n"
            for warning in self.results.warnings:
                summary_text += f"  {warning}\n"
            summary_text += "\n"
        
        # Trading recommendations with enhanced cointegration logic
        summary_text += "TRADING RECOMMENDATIONS:\n"
        if self.results.is_suitable_full or self.results.is_suitable_3m:
            preferred_window = self.results.best_3m_window or self.results.best_full_window
            preferred_period = "3-month" if self.results.best_3m_window else "full period"
            summary_text += f"  • ✅ RECOMMENDED FOR TRADING\n"
            summary_text += f"  • RECOMMENDED WINDOW: {preferred_window}d ({preferred_period})\n"
            summary_text += f"  • ENTRY THRESHOLD: |Z_robust| > {Z_ENTRY_THRESHOLD}\n"
            summary_text += f"  • EXIT THRESHOLD: Z crosses {Z_EXIT_THRESHOLD}\n"
            summary_text += f"  • STOP LOSS: |Z_robust| > {Z_STOP_THRESHOLD}\n"
            
            if preferred_window:
                if preferred_window in self.results.three_month_metrics:
                    best_metrics = self.results.three_month_metrics[preferred_window]
                else:
                    best_metrics = self.results.full_period_metrics[preferred_window]
                
                summary_text += f"  • CURRENT BETA: {best_metrics.beta.iloc[-1]:.4f}\n"
                summary_text += f"  • EXPECTED HOLDING: {best_metrics.half_life:.0f} periods ({best_metrics.half_life * self.resample_minutes / 60:.1f} hours)\n"
                
                # Enhanced: Cointegration-based recommendations
                if 'error' not in self.results.cointegration_results:
                    coint = self.results.cointegration_results
                    trace_ratio = coint.get('trace_ratio', 0)
                    if trace_ratio >= 0.7:
                        summary_text += f"  ✅ GOOD LONG-TERM STABILITY: High cointegration ratio {trace_ratio:.2f}\n"
                    elif trace_ratio >= 0.4:
                        summary_text += f"  ⚠️  MODERATE STABILITY: Acceptable cointegration for memecoins ({trace_ratio:.2f})\n"
                    else:
                        summary_text += f"  ⚠️  SHORT-TERM ONLY: Low cointegration - consider shorter holding periods\n"
                
                # Enhanced: Signal quality recommendations
                if best_metrics.signal_quality:
                    sig = best_metrics.signal_quality
                    if sig.get('false_signal_rate', 0) > 0.1:
                        summary_text += f"  ⚠️  SIGNAL QUALITY: {sig['false_signal_rate']:.1%} false signals - consider longer confirmation\n"
                    if sig.get('avg_signal_duration_hours', 0) < 1:
                        summary_text += f"  ⚠️  SHORT SIGNALS: Average {sig['avg_signal_duration_hours']:.1f}h duration - fast execution needed\n"
                    if sig.get('signal_frequency', 0) > 0.15:
                        summary_text += f"  ✅ HIGH ACTIVITY: {sig['signal_frequency']:.1%} signal frequency - good for active trading\n"
        else:
            summary_text += f"  • ❌ PAIR NOT RECOMMENDED FOR TRADING\n"
            summary_text += f"  • CONSIDER ALTERNATIVE PAIRS OR PARAMETER ADJUSTMENTS\n"
        
        # Key insights with enhanced interpretation
        summary_text += "\nKEY ENHANCEMENTS:\n"
        summary_text += f"  • MEMECOIN-ADJUSTED: Relaxed cointegration thresholds (0.4 ratio vs 1.0 traditional)\n"
        summary_text += f"  • TIME-ADJUSTED: Signal durations converted to hours for {self.resample_minutes}min timeframe\n"
        summary_text += f"  • ULTRA-RELAXED KPSS: Accepts p-values > 0.005 for high-volatility assets\n"
        summary_text += f"  • COMPREHENSIVE: Johansen cointegration + signal quality + dual-period analysis\n"
        summary_text += f"  • REALISTIC: Designed for memecoin volatility patterns and behavior\n"
        summary_text += f"  • DUAL-PERIOD: Full vs 3M analysis reveals market regime changes\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=8, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        
        ax.set_title('ENHANCED DUAL-PERIOD EXECUTIVE SUMMARY', fontsize=16, fontweight='bold')
    
    def generate_enhanced_report(self):
        """ENHANCED: Generate comprehensive analysis report with improved cointegration interpretation"""
        log.info(f"\n{'='*80}")
        log.info(f"ENHANCED DUAL-PERIOD ANALYSIS REPORT")
        log.info(f"{'='*80}")
        
        log.info(f"\nPair: {self.pair_name}")
        log.info(f"Full period: {self.results.data_start} to {self.results.data_end} ({self.results.total_observations:,} obs)")
        log.info(f"3M period: {self.results.period_3m_start} to {self.results.data_end} ({self.results.period_3m_observations:,} obs)")
        
        # ENHANCED: Cointegration Results with improved interpretation
        log.info(f"\nCOINTEGRATION ANALYSIS (MEMECOIN-ADJUSTED):")
        if 'error' not in self.results.cointegration_results:
            coint = self.results.cointegration_results
            trace_ratio = coint.get('trace_ratio', 0)
            
            log.info(f"  Traditional Test: {'✅ Cointegrated' if coint['cointegrated_5pct'] else '❌ Not Cointegrated'} ({coint['cointegration_strength']})")
            log.info(f"  Relaxed Test: {'✅ ACCEPTABLE' if coint.get('cointegrated_relaxed', False) else '❌ WEAK'} for memecoins")
            log.info(f"  Trace Statistic: {coint['trace_stat']:.3f} (Critical 5%: {coint['trace_crit_5pct']:.3f})")
            log.info(f"  Trace Ratio: {trace_ratio:.2f} (≥0.4 acceptable, ≥0.7 good, ≥1.0 strong)")
            log.info(f"  Interpretation: {'Strong long-term relationship' if trace_ratio >= 0.7 else 'Moderate relationship - suitable for memecoins' if trace_ratio >= 0.4 else 'Weak relationship - short-term only'}")
        else:
            log.info(f"  ❌ Test failed: {self.results.cointegration_results['error']}")
        
        # Full period results
        log.info(f"\nFULL PERIOD RESULTS:")
        tradeable_full = [w for w, m in self.results.full_period_metrics.items() if m.is_tradeable]
        log.info(f"  Tradeable windows: {len(tradeable_full)}/{len(self.results.full_period_metrics)}")
        log.info(f"  Best window: {self.results.best_full_window or 'None'}d")
        log.info(f"  Overall suitable: {'YES' if self.results.is_suitable_full else 'NO'}")
        
        if self.results.best_full_window:
            metrics = self.results.full_period_metrics[self.results.best_full_window]
            adf_pval = metrics.adf_gls_pvalue if not np.isnan(metrics.adf_gls_pvalue) else metrics.adf_pvalue
            log.info(f"  Best metrics: ADF={adf_pval:.4f}, R²={metrics.r_squared.mean():.3f}, β-std={metrics.beta_std:.3f}")
            
            # ENHANCED: Signal quality reporting with time conversion
            if metrics.signal_quality:
                sig = metrics.signal_quality
                log.info(f"  Signal Quality:")
                log.info(f"    Signal Frequency: {sig.get('signal_frequency', 0):.1%}")
                log.info(f"    Avg Duration: {sig.get('avg_signal_duration_hours', 0):.1f} hours ({sig.get('avg_signal_duration_periods', 0):.1f} periods)")
                log.info(f"    Max Duration: {sig.get('max_signal_duration_hours', 0):.1f} hours")
                log.info(f"    False Signal Rate: {sig.get('false_signal_rate', 0):.1%}")
                log.info(f"    Consistency: {sig.get('signal_consistency', 0):.1%}")
        
        # 3-month period results
        log.info(f"\n3-MONTH PERIOD RESULTS:")
        if self.results.three_month_metrics:
            tradeable_3m = [w for w, m in self.results.three_month_metrics.items() if m.is_tradeable]
            log.info(f"  Tradeable windows: {len(tradeable_3m)}/{len(self.results.three_month_metrics)}")
            log.info(f"  Best window: {self.results.best_3m_window or 'None'}d")
            log.info(f"  Overall suitable: {'YES' if self.results.is_suitable_3m else 'NO'}")
            
            if self.results.best_3m_window:
                metrics = self.results.three_month_metrics[self.results.best_3m_window]
                adf_pval = metrics.adf_gls_pvalue if not np.isnan(metrics.adf_gls_pvalue) else metrics.adf_pvalue
                log.info(f"  Best metrics: ADF={adf_pval:.4f}, R²={metrics.r_squared.mean():.3f}, β-std={metrics.beta_std:.3f}")
        else:
            log.info(f"  Insufficient data for 3-month analysis")
        
        # ENHANCED: Warnings reporting with improved interpretation
        if self.results.warnings:
            log.info(f"\nANALYSIS WARNINGS:")
            for warning in self.results.warnings:
                log.info(f"  {warning}")
        
        # Final recommendation with enhanced logic
        log.info(f"\nFINAL RECOMMENDATION:")
        if self.results.is_suitable_3m or self.results.is_suitable_full:
            preferred = self.results.best_3m_window or self.results.best_full_window
            preferred_period = "3-month" if self.results.best_3m_window else "full period"
            log.info(f"  ✅ RECOMMENDED FOR TRADING")
            log.info(f"  USE WINDOW: {preferred}d ({preferred_period} analysis)")
            log.info(f"  PREFER RECENT ANALYSIS: {'YES' if self.results.best_3m_window else 'NO'}")
            
            # Enhanced cointegration-based recommendations
            if 'error' not in self.results.cointegration_results:
                coint = self.results.cointegration_results
                trace_ratio = coint.get('trace_ratio', 0)
                if trace_ratio >= 0.7:
                    log.info(f"  ✅ EXCELLENT STABILITY: Strong cointegration (ratio: {trace_ratio:.2f}) - suitable for longer holds")
                elif trace_ratio >= 0.4:
                    log.info(f"  ✅ GOOD FOR MEMECOINS: Acceptable cointegration (ratio: {trace_ratio:.2f}) - moderate risk")
                else:
                    log.info(f"  ⚠️  HIGH RISK: Weak cointegration (ratio: {trace_ratio:.2f}) - short-term trades only")
            
            # Enhanced signal quality warnings with time context
            if preferred and preferred in (self.results.three_month_metrics if self.results.best_3m_window else self.results.full_period_metrics):
                preferred_metrics = (self.results.three_month_metrics if self.results.best_3m_window else self.results.full_period_metrics)[preferred]
                if preferred_metrics.signal_quality:
                    sig = preferred_metrics.signal_quality
                    if sig.get('false_signal_rate', 0) > 0.1:
                        log.info(f"  ⚠️  SIGNAL QUALITY: {sig['false_signal_rate']:.1%} false signals - consider longer confirmation")
                    if sig.get('avg_signal_duration_hours', 0) < 1:
                        log.info(f"  ⚠️  FAST SIGNALS: Average {sig['avg_signal_duration_hours']:.1f}h duration - quick execution needed")
                    if sig.get('signal_frequency', 0) > 0.15:
                        log.info(f"  ✅ HIGH ACTIVITY: {sig['signal_frequency']:.1%} signal frequency - good for active trading")
        else:
            log.info(f"  ❌ NOT RECOMMENDED FOR TRADING")
            log.info(f"  REASON: No suitable windows found in either period")
            
            # Enhanced guidance for failed pairs
            if 'error' not in self.results.cointegration_results:
                coint = self.results.cointegration_results
                trace_ratio = coint.get('trace_ratio', 0)
                if trace_ratio < 0.2:
                    log.info(f"  SUGGESTION: Try different pairs - very low cointegration ({trace_ratio:.2f})")
                elif trace_ratio < 0.4:
                    log.info(f"  SUGGESTION: Consider shorter timeframes or different parameters")
        
        log.info(f"\n{'='*80}\n")
    
    def run_analysis(self):
        """Enhanced analysis pipeline with improved cointegration and signal quality metrics"""
        try:
            with timer("Loading and resampling data"):
                log.info(f"\nLoading data...")
                self.df1 = self.load_ohlcv_data(self.symbol1)
                self.df2 = self.load_ohlcv_data(self.symbol2)
                
                if self.df1.empty or self.df2.empty:
                    log.error("Failed to load data for one or both symbols")
                    return False
            
            with timer("Aligning dual-period data"):
                if not self.align_data():
                    log.error("Failed to align data")
                    return False
            
            with timer("Enhanced dual-period window analysis"):
                self.analyze_all_windows()
                
                if not self.results.full_period_metrics:
                    log.error("No valid analysis results")
                    return False
            
            with timer("Creating enhanced visualization"):
                self.create_comprehensive_visualization()
            
            with timer("Generating enhanced report"):
                self.generate_enhanced_report()
            
            return True
            
        except Exception as e:
            log.error(f"Enhanced dual-period analysis failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False

# ========== MAIN FUNCTION ==========
def main():
    """Main execution function for enhanced dual-period analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Dual-Period Pair Trading Analysis - Fixed Layout & Improved Johansen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_pairs2.py BTC_USDT ETH_USDT
  python analyze_pairs2.py BTC_USDT ETH_USDT --days 180 --resample 15
  python analyze_pairs2.py BTC_USDT ETH_USDT --windows 3,7,15,30 --resample 5
  python analyze_pairs2.py --auto  # Analyze all available pairs automatically
        """
    )
    
    parser.add_argument("symbol1", type=str, nargs='?', help="First symbol (e.g., BTC_USDT)")
    parser.add_argument("symbol2", type=str, nargs='?', help="Second symbol (e.g., ETH_USDT)")
    parser.add_argument("--auto", action="store_true", 
                       help="Automatically analyze all unique symbol combinations")
    parser.add_argument("--days", type=int, default=365,
                       help="Number of days to analyze (default: 365)")
    parser.add_argument("--windows", type=str,
                       default=",".join(str(w) for w in DEFAULT_WINDOWS),
                       help="Analysis windows in days, comma-separated (default: 3,5,7,10,15,20,30)")
    parser.add_argument("--robust-method", type=str, default="mad",
                       choices=["mad", "ewma", "traditional"],
                       help="Robust z-score method (default: mad)")
    parser.add_argument("--resample", type=int, default=1,
                       help="Resample frequency in minutes (default: 1)")
    
    args = parser.parse_args()
    
    # Clean header logs
    log.info("=" * 80)
    log.info("ENHANCED DUAL-PERIOD PAIR TRADING ANALYSIS")
    log.info("Features: Fixed Layout + Improved Johansen + Time-Adjusted Signals")
    log.info("=" * 80)
    
    try:
        windows = [int(w.strip()) for w in args.windows.split(",") if w.strip()]
        
        if args.auto:
            # Auto-analyze all symbol combinations
            log.info("AUTO MODE: Analyzing all available symbol combinations...")
            symbols = get_available_symbols()
            log.info(f"Found {len(symbols)} symbols: {symbols}")
            
            symbol_pairs = list(combinations(symbols, 2))
            log.info(f"Generated {len(symbol_pairs)} unique pairs")
            
            successful_analyses = 0
            
            for i, (symbol1, symbol2) in enumerate(symbol_pairs, 1):
                log.info(f"\n[{i}/{len(symbol_pairs)}] Analyzing {symbol1} vs {symbol2}")
                
                analyzer = DualPeriodPairAnalyzer(
                    symbol1=symbol1,
                    symbol2=symbol2,
                    days=args.days,
                    windows=windows,
                    robust_zscore_method=args.robust_method,
                    resample_minutes=args.resample
                )
                
                try:
                    success = analyzer.run_analysis()
                    if success:
                        successful_analyses += 1
                        log.info(f"✅ Enhanced analysis completed for {symbol1}-{symbol2}")
                    else:
                        log.warning(f"❌ Enhanced analysis failed for {symbol1}-{symbol2}")
                except Exception as e:
                    log.error(f"❌ Error analyzing {symbol1}-{symbol2}: {e}")
                    continue
            
            log.info(f"\n{'='*80}")
            log.info(f"ENHANCED AUTO ANALYSIS COMPLETED!")
            log.info(f"Successfully analyzed: {successful_analyses}/{len(symbol_pairs)} pairs")
            log.info(f"Check plots/pair_analysis/ directory for all visualizations")
            log.info(f"Features: Fixed Layout + Improved Johansen + Time-Adjusted Signals")
            log.info(f"{'='*80}")
            
            return successful_analyses > 0
            
        else:
            # Manual mode - specific symbols
            if not args.symbol1 or not args.symbol2:
                log.error("Error: Please provide both symbol1 and symbol2, or use --auto mode")
                return False
            
            log.info(f"Symbol 1: {args.symbol1}")
            log.info(f"Symbol 2: {args.symbol2}")
            log.info(f"Analysis period: {args.days} days")
            log.info(f"Windows: {args.windows}")
            log.info(f"Resample: {args.resample} minutes")
            log.info(f"Robust method: {args.robust_method}")
            
            analyzer = DualPeriodPairAnalyzer(
                symbol1=args.symbol1,
                symbol2=args.symbol2,
                days=args.days,
                windows=windows,
                robust_zscore_method=args.robust_method,
                resample_minutes=args.resample
            )
            
            success = analyzer.run_analysis()
            
            if success:
                log.info("\n✅ ENHANCED DUAL-PERIOD ANALYSIS COMPLETED SUCCESSFULLY!")
                log.info("Check plots/pair_analysis/ directory for comprehensive visualizations")
                log.info("Enhanced with: Fixed Layout + Improved Johansen + Time-Adjusted Signals")
                return True
            else:
                log.error("\n❌ ENHANCED DUAL-PERIOD ANALYSIS FAILED!")
                return False
            
    except Exception as e:
        log.error(f"Fatal error: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
