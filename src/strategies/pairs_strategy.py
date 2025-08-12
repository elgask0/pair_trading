#!/usr/bin/env python3
"""
Pairs Trading Strategy - CORREGIDO: Ventanas exactas según resample
"""

from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.strategies.base_strategy import BaseStrategy
from src.utils.logger import get_logger

log = get_logger()


class PairsTradingStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        defaults = {
            'lookback_window': 20,      # DÍAS
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,      # 🔧 CORREGIDO: era 0.0
            'stop_loss_threshold': 4.0,
            'min_correlation': 0.5,
            'use_robust_zscore': True,
            'hedge_ratio_window': 60,   # DÍAS
            'rebalance_frequency': 'never',
            'signal_confirmation_bars': 0,
            'max_holding_periods': 0,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

        # Estado
        self.hedge_ratio: float = 1.0
        self.spread: Optional[pd.Series] = None
        self.current_position: Optional[str] = None
        self.last_rebalance_ts: Optional[pd.Timestamp] = None
        
        # 🔧 NUEVO: Configuración de resample
        self.resample_minutes: int = 15  # Default, será sobrescrito
        
        # Debug
        self.z_score_history = []

    def set_resample_minutes(self, resample_minutes: int):
        """🔧 NUEVO: Recibir resample_minutes del engine"""
        self.resample_minutes = resample_minutes
        log.info(f"📊 Strategy resample set to: {resample_minutes} minutes")

    def _periods_from_days(self, days: int) -> int:
        """🔧 CORREGIDO: Usar resample_minutes exacto del config"""
        periods = int(days * 24 * 60 / self.resample_minutes)
        log.debug(f"Days to periods: {days}d → {periods}p @ {self.resample_minutes}min")
        return periods

    def _slice_periods(self, series: pd.Series, end_idx: int, periods: int) -> pd.Series:
        """Slice por períodos hacia atrás desde end_idx"""
        start_idx = max(0, end_idx - periods + 1)
        return series.iloc[start_idx:end_idx + 1]

    def _enough_history(self, end_idx: int, periods: int, min_points: int = 10) -> bool:
        """Verificar si hay suficiente historia"""
        if self.data is None or self.data.empty:
            return False
        available = end_idx + 1  # +1 porque end_idx es 0-indexed
        return available >= periods and periods >= min_points

    def initialize(self):
        # 🔧 CALCULAR períodos exactos para logging
        lookback_days = int(self.params.get('lookback_window', 20))
        lookback_periods = self._periods_from_days(lookback_days)
        
        hedge_days = int(self.params.get('hedge_ratio_window', 60))
        hedge_periods = self._periods_from_days(hedge_days)
        
        log.info(f"🚀 STRATEGY INIT | Resample: {self.resample_minutes}min")
        log.info(f"📊 Lookback: {lookback_days}d = {lookback_periods}p | "
                f"Hedge: {hedge_days}d = {hedge_periods}p")
        log.info(f"🎯 Entry: ±{self.params.get('entry_threshold', 2.0)} | "
                f"Exit: ±{self.params.get('exit_threshold', 0.5)} | "
                f"Stop: ±{self.params.get('stop_loss_threshold', 4.0)}")
        
        self._calculate_initial_hedge_ratio()
        self._recompute_full_spread()
        self.last_rebalance_ts = None
        self.current_position = None
        self.signals_history = []
        self.z_score_history = []

    def _calculate_initial_hedge_ratio(self):
        """🔧 CORREGIDO: Usar períodos exactos"""
        if self.data is None or self.data.empty:
            self.hedge_ratio = 1.0
            return

        days = int(self.params.get('hedge_ratio_window', 60))
        periods = self._periods_from_days(days)
        
        # Usar primeros períodos disponibles
        initial = self.data.iloc[:min(periods, len(self.data))]

        if len(initial) < 10:
            log.warning(f"⚠️ Insufficient data ({len(initial)}/{periods} periods), using hedge ratio = 1.0")
            self.hedge_ratio = 1.0
            return

        try:
            y = initial['log_price1']
            X = sm.add_constant(initial['log_price2'])
            model = sm.OLS(y, X).fit()
            self.hedge_ratio = float(model.params.iloc[1])
            r_squared = float(model.rsquared)
            
            log.info(f"📈 HEDGE RATIO CALCULATED:")
            log.info(f"   Value: {self.hedge_ratio:.6f}")
            log.info(f"   R²: {r_squared:.4f}")
            log.info(f"   Window: {len(initial)}/{periods} periods ({days}d @ {self.resample_minutes}min)")
            log.info(f"   Alpha: {float(model.params.iloc[0]):.6f}")
            
        except Exception as e:
            log.error(f"❌ Error calculating hedge ratio: {e}")
            self.hedge_ratio = 1.0

    def _recompute_full_spread(self):
        """Calcular spread completo"""
        if self.data is not None and not self.data.empty:
            self.spread = self.data['log_price1'] - self.hedge_ratio * self.data['log_price2']
            log.info(f"📊 SPREAD COMPUTED:")
            log.info(f"   Total observations: {len(self.spread)}")
            log.info(f"   Mean: {self.spread.mean():.6f}")
            log.info(f"   Std: {self.spread.std():.6f}")
            log.info(f"   Range: [{self.spread.min():.6f}, {self.spread.max():.6f}]")

    def _zscore_at(self, idx: int) -> float:
        """🔧 CORREGIDO: Z-score con períodos exactos"""
        if self.spread is None:
            return 0.0

        days = int(self.params.get('lookback_window', 20))
        periods = self._periods_from_days(days)
        
        if not self._enough_history(idx, periods):
            return 0.0

        # 🔧 CORREGIDO: Usar slice exacto por períodos
        window = self._slice_periods(self.spread, idx, periods)

        if self.params.get('use_robust_zscore', True):
            # Z-score robusto usando MAD
            med = window.median()
            mad = np.median(np.abs(window - med))
            if mad == 0:
                return 0.0
            zscore = 0.6745 * (self.spread.iloc[idx] - med) / mad
        else:
            # Z-score tradicional
            mean = window.mean()
            std = window.std()
            if std == 0 or np.isnan(std):
                return 0.0
            zscore = (self.spread.iloc[idx] - mean) / std
        
        # 🔧 DEBUG: Guardar para análisis
        self.z_score_history.append({
            'idx': idx,
            'zscore': float(zscore),
            'window_size': len(window),
            'window_periods_requested': periods,
            'window_mean': float(window.mean()),
            'window_std': float(window.std()),
            'current_spread': float(self.spread.iloc[idx])
        })
        
        return float(zscore)

    def _correlation_ok(self, idx: int) -> bool:
        """🔧 CORREGIDO: Correlación con períodos exactos"""
        days = int(self.params.get('lookback_window', 20))
        periods = self._periods_from_days(days)
        
        if not self._enough_history(idx, periods):
            return True
            
        returns1 = self._slice_periods(self.data['returns1'], idx, periods)
        returns2 = self._slice_periods(self.data['returns2'], idx, periods)
        
        corr = returns1.corr(returns2)
        min_corr = float(self.params.get('min_correlation', 0.5))
        
        return (corr >= min_corr) if not np.isnan(corr) else True

    def generate_signal(self, timestamp: datetime, row: pd.Series, index: int) -> Optional[Dict]:
        """🔧 CORREGIDO: Generación de señales con debugging mejorado"""
        z = self._zscore_at(index)
        
        # guardar para diagnóstico
        self.signals_history.append({
            'timestamp': timestamp,
            'z_score': float(z),
            'spread': float(self.spread.iloc[index]) if self.spread is not None else 0.0,
            'hedge_ratio': float(self.hedge_ratio),
            'price1': float(row['price1']),
            'price2': float(row['price2'])
        })

        # 🔧 DEBUG: Log cada 500 observaciones con info detallada
        if index % 500 == 0:
            recent_z = [x['zscore'] for x in self.z_score_history[-50:]] if self.z_score_history else []
            if recent_z:
                lookback_days = int(self.params.get('lookback_window', 20))
                lookback_periods = self._periods_from_days(lookback_days)
                
                log.info(f"📊 Z-SCORE DEBUG | idx:{index}")
                log.info(f"   Current Z: {z:.3f}")
                log.info(f"   Recent Z range: [{min(recent_z):.2f}, {max(recent_z):.2f}]")
                log.info(f"   Lookback: {lookback_days}d = {lookback_periods}p @ {self.resample_minutes}min")
                log.info(f"   Position: {self.current_position or 'None'}")
                log.info(f"   Spread: {float(self.spread.iloc[index]):.6f}")

        # filtro de correlación
        if not self._correlation_ok(index):
            if self.current_position:
                log.info(f"⚠️ LOW CORR EXIT")
                self.current_position = None
                return {
                    'action': 'CLOSE', 
                    'position_id': 'current', 
                    'reason': 'low_correlation', 
                    'z_score': float(z)
                }
            return {'action': 'HOLD', 'reason': 'low_correlation'}

        entry = float(self.params.get('entry_threshold', 2.0))
        exit_thr = float(self.params.get('exit_threshold', 0.5))
        stop = float(self.params.get('stop_loss_threshold', 4.0))

        signal: Optional[Dict[str, Any]] = None

        if not self.current_position:
            if z <= -entry:
                self.current_position = 'LONG'
                log.info(f"📈 LONG SIGNAL | Z: {z:.4f} ≤ -{entry} | HR: {self.hedge_ratio:.6f}")
                
                signal = {
                    'action': 'LONG',
                    'strength': min(abs(z) / stop, 1.0),
                    'reason': 'spread_undervalued',
                    'z_score': float(z),
                    'hedge_ratio': float(self.hedge_ratio),
                }
            elif z >= entry:
                self.current_position = 'SHORT'
                log.info(f"📉 SHORT SIGNAL | Z: {z:.4f} ≥ +{entry} | HR: {self.hedge_ratio:.6f}")
                
                signal = {
                    'action': 'SHORT',
                    'strength': min(abs(z) / stop, 1.0),
                    'reason': 'spread_overvalued',
                    'z_score': float(z),
                    'hedge_ratio': float(self.hedge_ratio),
                }
        else:
            if self.current_position == 'LONG':
                if z > exit_thr:
                    log.info(f"🔄 LONG EXIT | Z: {z:.4f} > {exit_thr} (reversion)")
                    self.current_position = None
                    signal = {
                        'action': 'CLOSE',
                        'position_id': 'current',
                        'reason': 'exit_signal',
                        'z_score': float(z),
                    }
                elif z < -stop:
                    log.info(f"🛑 LONG STOP | Z: {z:.4f} < -{stop} (stop loss)")
                    self.current_position = None
                    signal = {
                        'action': 'CLOSE',
                        'position_id': 'current',
                        'reason': 'stop_loss',
                        'z_score': float(z),
                    }
            elif self.current_position == 'SHORT':
                if z < -exit_thr:
                    log.info(f"🔄 SHORT EXIT | Z: {z:.4f} < -{exit_thr} (reversion)")
                    self.current_position = None
                    signal = {
                        'action': 'CLOSE',
                        'position_id': 'current',
                        'reason': 'exit_signal',
                        'z_score': float(z),
                    }
                elif z > stop:
                    log.info(f"🛑 SHORT STOP | Z: {z:.4f} > +{stop} (stop loss)")
                    self.current_position = None
                    signal = {
                        'action': 'CLOSE',
                        'position_id': 'current',
                        'reason': 'stop_loss',
                        'z_score': float(z),
                    }

        return signal or {'action': 'HOLD'}

    def get_strategy_metrics(self) -> Dict[str, Any]:
        df = self.get_signals_history()
        if df.empty:
            return {}
            
        # Z-score debugging info
        recent_z = [x['zscore'] for x in self.z_score_history] if self.z_score_history else []
        
        # 🔧 NUEVO: Info de configuración para debug
        lookback_days = int(self.params.get('lookback_window', 20))
        lookback_periods = self._periods_from_days(lookback_days)
        
        return {
            'avg_z_score': float(df['z_score'].mean()),
            'max_z_score': float(df['z_score'].max()),
            'min_z_score': float(df['z_score'].min()),
            'z_score_std': float(df['z_score'].std()),
            'current_hedge_ratio': float(self.hedge_ratio),
            'spread_mean': float(df['spread'].mean()) if 'spread' in df.columns else 0.0,
            'spread_std': float(df['spread'].std()) if 'spread' in df.columns else 0.0,
            'signals_generated': len(df),
            'extreme_z_scores': int((df['z_score'].abs() > 3).sum()) if len(df) > 0 else 0,
            # 🔧 DEBUG INFO
            'z_calc_count': len(self.z_score_history),
            'z_calc_range': [float(min(recent_z)), float(max(recent_z))] if recent_z else [0, 0],
            'resample_minutes': self.resample_minutes,
            'lookback_days': lookback_days,
            'lookback_periods': lookback_periods
        }