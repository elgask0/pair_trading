#!/usr/bin/env python3
"""
Pairs Trading Strategy - Estrategia de pairs trading con cointegración
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
import statsmodels.api as sm
from scipy import stats

from src.strategies.base_strategy import BaseStrategy
from src.utils.logger import get_logger

log = get_logger()

class PairsTradingStrategy(BaseStrategy):
    """
    Estrategia de pairs trading basada en cointegración y z-score
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'lookback_window': 20,     # Ventana para calcular z-score (días)
            'entry_threshold': 2.0,     # Z-score para entrada
            'exit_threshold': 0.0,      # Z-score para salida
            'stop_loss_threshold': 4.0, # Z-score para stop loss
            'min_correlation': 0.5,     # Correlación mínima requerida
            'use_robust_zscore': True,  # Usar z-score robusto (MAD)
            'hedge_ratio_window': 60,   # Ventana para calcular hedge ratio
            'rebalance_frequency': 5,   # Rebalancear hedge ratio cada N días
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(default_params)
        
        # Estado interno
        self.hedge_ratio = None
        self.spread = None
        self.zscore = None
        self.current_position = None
        self.last_rebalance = None
        self.spread_mean = None
        self.spread_std = None
        
    def initialize(self):
        """Inicializar la estrategia"""
        log.info("Initializing Pairs Trading Strategy")
        log.info(f"Parameters: {self.params}")
        
        # Pre-calcular series necesarias
        self._calculate_initial_hedge_ratio()
        self._calculate_spread_series()
        
        self.last_rebalance = 0
        self.current_position = None
        
    def _calculate_initial_hedge_ratio(self):
        """Calcular hedge ratio inicial usando OLS"""
        window = self.params['hedge_ratio_window']
        
        if len(self.data) < window:
            log.warning(f"Insufficient data for hedge ratio calculation")
            self.hedge_ratio = 1.0
            return
        
        # Usar los primeros N días para calcular hedge ratio inicial
        initial_data = self.data.iloc[:window]
        
        y = initial_data['log_price1']
        X = sm.add_constant(initial_data['log_price2'])
        
        model = sm.OLS(y, X).fit()
        self.hedge_ratio = model.params[1]
        
        log.info(f"Initial hedge ratio: {self.hedge_ratio:.4f}")
        
    def _calculate_spread_series(self):
        """Pre-calcular serie de spread para todo el período"""
        # Calcular spread usando hedge ratio
        self.spread = pd.Series(index=self.data.index, dtype=float)
        
        # Para simplicidad, usar hedge ratio fijo inicialmente
        # En versión avanzada, recalcular dinámicamente
        self.spread = self.data['log_price1'] - self.hedge_ratio * self.data['log_price2']
        
    def _calculate_zscore(self, index: int) -> float:
        """Calcular z-score del spread"""
        window = self.params['lookback_window']
        
        if index < window:
            return 0.0
        
        # Obtener ventana de spread
        spread_window = self.spread.iloc[max(0, index - window):index + 1]
        
        if self.params['use_robust_zscore']:
            # Z-score robusto usando MAD
            median = spread_window.median()
            mad = np.median(np.abs(spread_window - median))
            if mad == 0:
                return 0.0
            zscore = 0.6745 * (spread_window.iloc[-1] - median) / mad
        else:
            # Z-score tradicional
            mean = spread_window.mean()
            std = spread_window.std()
            if std == 0:
                return 0.0
            zscore = (spread_window.iloc[-1] - mean) / std
        
        return zscore
    
    def _check_correlation(self, index: int) -> bool:
        """Verificar si la correlación es suficiente"""
        window = self.params['lookback_window']
        
        if index < window:
            return True
        
        returns1 = self.data['returns1'].iloc[max(0, index - window):index + 1]
        returns2 = self.data['returns2'].iloc[max(0, index - window):index + 1]
        
        correlation = returns1.corr(returns2)
        
        return correlation >= self.params['min_correlation']
    
    def _should_rebalance(self, index: int) -> bool:
        """Determinar si es momento de rebalancear hedge ratio"""
        if self.last_rebalance is None:
            return True
            
        days_since_rebalance = index - self.last_rebalance
        return days_since_rebalance >= self.params['rebalance_frequency']
    
    def _rebalance_hedge_ratio(self, index: int):
        """Recalcular hedge ratio"""
        window = self.params['hedge_ratio_window']
        
        if index < window:
            return
        
        # Obtener datos recientes
        recent_data = self.data.iloc[max(0, index - window):index + 1]
        
        y = recent_data['log_price1']
        X = sm.add_constant(recent_data['log_price2'])
        
        try:
            model = sm.OLS(y, X).fit()
            self.hedge_ratio = model.params[1]
            self.last_rebalance = index
            
            # Recalcular spread con nuevo hedge ratio
            self.spread.iloc[index:] = (
                self.data['log_price1'].iloc[index:] - 
                self.hedge_ratio * self.data['log_price2'].iloc[index:]
            )
            
        except Exception as e:
            log.warning(f"Failed to rebalance hedge ratio: {e}")
    
    def generate_signal(self, timestamp: datetime, row: pd.Series, index: int) -> Optional[Dict]:
        """Generar señal de trading basada en z-score del spread"""
        
        # Skip primeras observaciones
        if index < self.params['lookback_window']:
            return {'action': 'HOLD', 'reason': 'warming_up'}
        
        # Rebalancear hedge ratio si es necesario
        if self._should_rebalance(index):
            self._rebalance_hedge_ratio(index)
        
        # Calcular z-score actual
        zscore = self._calculate_zscore(index)
        
        # Verificar correlación
        if not self._check_correlation(index):
            # Si hay posición abierta y baja correlación, cerrar
            if self.current_position:
                self.current_position = None
                return {
                    'action': 'CLOSE',
                    'position_id': 'current',
                    'reason': 'low_correlation',
                    'z_score': zscore
                }
            return {'action': 'HOLD', 'reason': 'low_correlation'}
        
        # Guardar señal en historial
        self.signals_history.append({
            'timestamp': timestamp,
            'z_score': zscore,
            'spread': self.spread.iloc[index],
            'hedge_ratio': self.hedge_ratio,
            'price1': row['price1'],
            'price2': row['price2']
        })
        
        # Lógica de señales
        signal = None
        
        # Si no hay posición abierta
        if not self.current_position:
            
            # Señal LONG: spread muy negativo (comprar asset1, vender asset2)
            if zscore < -self.params['entry_threshold']:
                self.current_position = 'LONG'
                signal = {
                    'action': 'LONG',
                    'strength': min(abs(zscore) / self.params['stop_loss_threshold'], 1.0),
                    'reason': 'spread_undervalued',
                    'z_score': zscore,
                    'hedge_ratio': self.hedge_ratio
                }
                
            # Señal SHORT: spread muy positivo (vender asset1, comprar asset2)
            elif zscore > self.params['entry_threshold']:
                self.current_position = 'SHORT'
                signal = {
                    'action': 'SHORT',
                    'strength': min(abs(zscore) / self.params['stop_loss_threshold'], 1.0),
                    'reason': 'spread_overvalued',
                    'z_score': zscore,
                    'hedge_ratio': self.hedge_ratio
                }
        
        # Si hay posición abierta
        else:
            # Posición LONG
            if self.current_position == 'LONG':
                # Cerrar si z-score cruza el threshold de salida o stop loss
                if zscore > self.params['exit_threshold'] or zscore < -self.params['stop_loss_threshold']:
                    self.current_position = None
                    signal = {
                        'action': 'CLOSE',
                        'position_id': 'current',
                        'reason': 'exit_signal' if zscore > self.params['exit_threshold'] else 'stop_loss',
                        'z_score': zscore
                    }
            
            # Posición SHORT
            elif self.current_position == 'SHORT':
                # Cerrar si z-score cruza el threshold de salida o stop loss
                if zscore < self.params['exit_threshold'] or zscore > self.params['stop_loss_threshold']:
                    self.current_position = None
                    signal = {
                        'action': 'CLOSE',
                        'position_id': 'current',
                        'reason': 'exit_signal' if zscore < self.params['exit_threshold'] else 'stop_loss',
                        'z_score': zscore
                    }
        
        # Si no hay señal, mantener
        if not signal:
            signal = {'action': 'HOLD', 'reason': 'no_signal', 'z_score': zscore}
        
        return signal
    
    def get_strategy_metrics(self) -> Dict:
        """Obtener métricas específicas de la estrategia"""
        signals_df = self.get_signals_history()
        
        if signals_df.empty:
            return {}
        
        metrics = {
            'avg_z_score': signals_df['z_score'].mean(),
            'max_z_score': signals_df['z_score'].max(),
            'min_z_score': signals_df['z_score'].min(),
            'current_hedge_ratio': self.hedge_ratio,
            'spread_mean': signals_df['spread'].mean(),
            'spread_std': signals_df['spread'].std()
        }
        
        return metrics