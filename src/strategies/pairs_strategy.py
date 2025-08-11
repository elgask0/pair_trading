#!/usr/bin/env python3
"""
Pairs Trading Strategy - Ventanas basadas en días (lookback/hedge/rebalance)
y manejo robusto de 'rebalance_frequency' para evitar Timedelta out-of-bounds.
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
    """
    Estrategia de pairs basada en z-score del spread con hedge ratio por OLS.
    - lookback_window: días para z-score
    - hedge_ratio_window: días para calcular hedge ratio
    - rebalance_frequency: días entre rebalanceos (valores muy grandes => desactivado)
    """

    def __init__(self, params: Dict[str, Any] = None):
        defaults = {
            'lookback_window': 20,      # DÍAS
            'entry_threshold': 2.0,
            'exit_threshold': 0.0,
            'stop_loss_threshold': 4.0,
            'min_correlation': 0.5,
            'use_robust_zscore': True,
            'hedge_ratio_window': 60,   # DÍAS
            'rebalance_frequency': 5,   # DÍAS (>=100_000 => desactivado)
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

    # ---------------------- utilidades ----------------------
    @staticmethod
    def _as_days(value: Any) -> int:
        """Convierte config a días (int), devolviendo 0 si no aplica."""
        if value is None:
            return 0
        if isinstance(value, (int, float)) and np.isfinite(value):
            return int(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {'never', 'none', 'off', 'disable', 'disabled'}:
                return 0
            # intenta parsear número en string
            try:
                return int(float(v))
            except Exception:
                return 0
        return 0

    def _slice_days(self, series: pd.Series, end_ts: pd.Timestamp, days: int) -> pd.Series:
        start_ts = end_ts - pd.Timedelta(days=int(days))
        return series.loc[start_ts:end_ts]

    def _enough_history(self, end_ts: pd.Timestamp, days: int, min_points: int = 10) -> bool:
        if self.data is None or self.data.empty:
            return False
        window = self._slice_days(self.data['price1'], end_ts, days)
        return len(window) >= min_points

    # ------------------- inicialización ---------------------
    def initialize(self):
        log.info("Initializing Pairs Trading Strategy (time-based windows)")
        log.info(f"Parameters: {self.params}")
        self._calculate_initial_hedge_ratio()
        self._recompute_full_spread()
        self.last_rebalance_ts = None
        self.current_position = None
        self.signals_history = []

    def _calculate_initial_hedge_ratio(self):
        if self.data is None or self.data.empty:
            self.hedge_ratio = 1.0
            return

        days = self._as_days(self.params.get('hedge_ratio_window', 60))
        start = self.data.index[0]
        end = start + pd.Timedelta(days=days)
        initial = self.data.loc[start:end]

        if len(initial) < 10:
            log.warning("Insufficient data for hedge ratio calculation; using 1.0")
            self.hedge_ratio = 1.0
            return

        y = initial['log_price1']
        X = sm.add_constant(initial['log_price2'])
        model = sm.OLS(y, X).fit()
        # usar iloc para evitar FutureWarning
        self.hedge_ratio = float(model.params.iloc[1])
        log.info(f"Initial hedge ratio (first {days}d): {self.hedge_ratio:.4f}")

    def _recompute_full_spread(self):
        self.spread = self.data['log_price1'] - self.hedge_ratio * self.data['log_price2']

    # ------------------- cálculos rolling -------------------
    def _zscore_at(self, idx: int) -> float:
        end_ts = self.data.index[idx]
        days = self._as_days(self.params.get('lookback_window', 20))
        if not self._enough_history(end_ts, days):
            return 0.0

        w = self._slice_days(self.spread, end_ts, days)

        if self.params.get('use_robust_zscore', True):
            med = w.median()
            mad = np.median(np.abs(w - med))
            if mad == 0:
                return 0.0
            return float(0.6745 * (self.spread.iloc[idx] - med) / mad)

        mean = w.mean()
        std = w.std(ddof=0)
        if std == 0 or np.isnan(std):
            return 0.0
        return float((self.spread.iloc[idx] - mean) / std)

    def _correlation_ok(self, idx: int) -> bool:
        end_ts = self.data.index[idx]
        days = self._as_days(self.params.get('lookback_window', 20))
        if not self._enough_history(end_ts, days):
            return True
        r1 = self._slice_days(self.data['returns1'], end_ts, days)
        r2 = self._slice_days(self.data['returns2'], end_ts, days)
        corr = r1.corr(r2)
        return (corr >= float(self.params.get('min_correlation', 0.5))) if not np.isnan(corr) else True

    def _rebalance_disabled(self) -> bool:
        """True si el rebalanceo está desactivado por config."""
        freq_days = self._as_days(self.params.get('rebalance_frequency', 0))
        # pandas Timedelta aguanta ~106_751 días; por encima, desactiva.
        return freq_days <= 0 or freq_days >= 100_000

    def _should_rebalance(self, idx: int) -> bool:
        if self._rebalance_disabled():
            return False
        freq_days = self._as_days(self.params.get('rebalance_frequency', 5))
        now = self.data.index[idx]
        if self.last_rebalance_ts is None:
            return True
        return (now - self.last_rebalance_ts) >= pd.Timedelta(days=freq_days)

    def _rebalance_hedge_ratio(self, idx: int):
        days = self._as_days(self.params.get('hedge_ratio_window', 60))
        end_ts = self.data.index[idx]
        if not self._enough_history(end_ts, days):
            return
        recent = self._slice_days(self.data, end_ts, days)
        try:
            y = recent['log_price1']
            X = sm.add_constant(recent['log_price2'])
            model = sm.OLS(y, X).fit()
            self.hedge_ratio = float(model.params.iloc[1])
            self.last_rebalance_ts = end_ts
            # Actualizamos spread desde el rebalanceo hacia delante
            self.spread.loc[end_ts:] = (
                self.data['log_price1'].loc[end_ts:] -
                self.hedge_ratio * self.data['log_price2'].loc[end_ts:]
            )
        except Exception as e:
            log.warning(f"Failed to rebalance hedge ratio: {e}")

    # ------------------- señal principal -------------------
    def generate_signal(self, timestamp: datetime, row: pd.Series, index: int) -> Optional[Dict]:
        z = self._zscore_at(index)

        # guardar para diagnóstico/plots
        self.signals_history.append({
            'timestamp': timestamp,
            'z_score': float(z),
            'spread': float(self.spread.iloc[index]),
            'hedge_ratio': float(self.hedge_ratio),
            'price1': float(row['price1']),
            'price2': float(row['price2']),
        })

        # filtro de correlación
        if not self._correlation_ok(index):
            if self.current_position:
                self.current_position = None
                return {'action': 'CLOSE', 'position_id': 'current', 'reason': 'low_correlation', 'z_score': float(z)}
            return {'action': 'HOLD', 'reason': 'low_correlation'}

        # rebalanceo si procede
        if self._should_rebalance(index):
            self._rebalance_hedge_ratio(index)

        entry = float(self.params.get('entry_threshold', 2.0))
        exit_thr = float(self.params.get('exit_threshold', 0.0))
        stop = float(self.params.get('stop_loss_threshold', 4.0))

        signal: Optional[Dict[str, Any]] = None

        if not self.current_position:
            if z <= -entry:
                self.current_position = 'LONG'
                signal = {
                    'action': 'LONG',
                    'strength': min(abs(z) / stop, 1.0),
                    'reason': 'spread_undervalued',
                    'z_score': float(z),
                    'hedge_ratio': float(self.hedge_ratio),
                }
            elif z >= entry:
                self.current_position = 'SHORT'
                signal = {
                    'action': 'SHORT',
                    'strength': min(abs(z) / stop, 1.0),
                    'reason': 'spread_overvalued',
                    'z_score': float(z),
                    'hedge_ratio': float(self.hedge_ratio),
                }
        else:
            if self.current_position == 'LONG':
                if z > exit_thr or z < -stop:
                    self.current_position = None
                    signal = {
                        'action': 'CLOSE',
                        'position_id': 'current',
                        'reason': 'exit_signal' if z > exit_thr else 'stop_loss',
                        'z_score': float(z),
                    }
            elif self.current_position == 'SHORT':
                if z < exit_thr or z > stop:
                    self.current_position = None
                    signal = {
                        'action': 'CLOSE',
                        'position_id': 'current',
                        'reason': 'exit_signal' if z < exit_thr else 'stop_loss',
                        'z_score': float(z),
                    }

        return signal or {'action': 'HOLD'}

    # ------------------- métricas -------------------
    def get_strategy_metrics(self) -> Dict[str, Any]:
        df = self.get_signals_history()
        if df.empty:
            return {}
        return {
            'avg_z_score': float(df['z_score'].mean()),
            'max_z_score': float(df['z_score'].max()),
            'min_z_score': float(df['z_score'].min()),
            'current_hedge_ratio': float(self.hedge_ratio),
            'spread_mean': float(df['spread'].mean()),
            'spread_std': float(df['spread'].std()),
        }