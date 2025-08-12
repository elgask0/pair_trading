#!/usr/bin/env python3
"""
Pairs Trading Strategy - VERSIÓN OPTIMIZADA v2.0
Con filtros avanzados, hedge ratio robusto y gestión de volatilidad
"""

from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.strategies.base_strategy import BaseStrategy
from src.utils.logger import get_logger

log = get_logger()


class PairsTradingStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        defaults = {
            # PARÁMETROS OPTIMIZADOS
            'lookback_window': 5,           # DÍAS (no períodos) - más reactivo
            'entry_threshold': 2.5,         # Más conservador
            'exit_threshold': 0.3,          # Salida antes de la media
            'stop_loss_threshold': 3.5,     # Stop más ajustado
            'min_correlation': 0.6,         # Correlación más estricta
            'use_robust_zscore': True,      # Usar MAD para robustez
            'hedge_ratio_window': 30,       # DÍAS - ventana más corta
            'rebalance_frequency': 'never',
            
            # NUEVOS FILTROS
            'min_spread_volatility': 0.001,  # Volatilidad mínima del spread
            'max_spread_volatility': 0.05,   # Volatilidad máxima
            'volume_filter': True,           # Filtrar por volumen
            'min_volume_percentile': 20,     # Percentil mínimo de volumen
            'use_dynamic_thresholds': True,  # Ajustar thresholds por volatilidad
            'volatility_lookback': 10,       # Días para calcular volatilidad
            'max_hedge_ratio': 2.0,          # Cap para hedge ratio
            'min_hedge_ratio': 0.2,          # Floor para hedge ratio
            'use_half_life_filter': True,    # Filtrar por half-life del spread
            'min_half_life': 1.0,            # Días mínimos de mean reversion
            'max_half_life': 15.0,           # Días máximos de mean reversion
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

        # Estado
        self.hedge_ratio: float = 1.0
        self.current_hedge_ratio: float = 1.0
        self.spread: Optional[pd.Series] = None
        self.current_position: Optional[str] = None
        self.position_hedge_ratio: Optional[float] = None
        self.entry_timestamp: Optional[datetime] = None
        self.entry_z: Optional[float] = None
        self.entry_spread: Optional[float] = None
        
        # Configuración
        self.resample_minutes: int = 15
        
        # Cachés y buffers
        self.spread_cache = {}
        self.zscore_cache = {}
        self.volatility_cache = {}
        self.volume_stats = {}
        self.half_life_cache = {}
        
        # Tracking
        self.z_score_history = []
        self.hedge_ratio_history = []
        self.rejected_signals = []

    def set_resample_minutes(self, resample_minutes: int):
        """Recibir resample_minutes del engine"""
        self.resample_minutes = resample_minutes
        log.info(f"📊 Strategy resample set to: {resample_minutes} minutes")

    def _periods_from_days(self, days: float) -> int:
        """Convertir días a períodos según resample - CORREGIDO"""
        if self.resample_minutes <= 0:
            self.resample_minutes = 15  # Fallback seguro
        
        periods_per_day = (24 * 60) / self.resample_minutes
        periods = int(days * periods_per_day)
        return max(1, periods)

    def initialize(self):
        """Inicialización mejorada con validaciones"""
        lookback_days = float(self.params.get('lookback_window', 5))
        lookback_periods = self._periods_from_days(lookback_days)
        
        hedge_days = float(self.params.get('hedge_ratio_window', 30))
        hedge_periods = self._periods_from_days(hedge_days)
        
        log.info(f"🚀 STRATEGY INIT | Resample: {self.resample_minutes}min")
        log.info(f"📊 Lookback: {lookback_days}d = {lookback_periods}p | "
                f"Hedge: {hedge_days}d = {hedge_periods}p")
        log.info(f"🎯 Entry: ±{self.params.get('entry_threshold')} | "
                f"Exit: ±{self.params.get('exit_threshold')} | "
                f"Stop: ±{self.params.get('stop_loss_threshold')}")
        log.info(f"🔧 Robust Z: {self.params.get('use_robust_zscore')} | "
                f"Dynamic Thresholds: {self.params.get('use_dynamic_thresholds')} | "
                f"Volume Filter: {self.params.get('volume_filter')}")
        
        # Calcular estadísticas iniciales de volumen
        if self.data is not None and self.params.get('volume_filter'):
            self._calculate_volume_stats()
        
        # Reset estado
        self.current_position = None
        self.position_hedge_ratio = None
        self.signals_history = []
        self.z_score_history = []
        self.hedge_ratio_history = []

    def _calculate_volume_stats(self):
        """Calcular estadísticas de volumen para filtrado"""
        if self.data is None:
            return
        
        vol1 = self.data['volume1'].rolling(window=96).mean()  # Media 24h @ 15min
        vol2 = self.data['volume2'].rolling(window=96).mean()  # Media 24h @ 15min
        
        self.volume_stats = {
            'vol1_percentiles': vol1.quantile([0.2, 0.5, 0.8]).to_dict(),
            'vol2_percentiles': vol2.quantile([0.2, 0.5, 0.8]).to_dict(),
            'vol1_mean': vol1.mean(),
            'vol2_mean': vol2.mean()
        }

    def _calculate_half_life(self, spread_series: pd.Series) -> float:
        """Calcular half-life del spread usando Ornstein-Uhlenbeck"""
        if len(spread_series) < 20:
            return np.nan
        
        try:
            # OU process: dS = theta*(mu - S)*dt + sigma*dW
            # AR(1): S_t = a + b*S_{t-1} + epsilon
            spread_lag = spread_series.shift(1)
            spread_diff = spread_series - spread_lag
            
            # Remover NaN
            valid_idx = ~(spread_lag.isna() | spread_diff.isna())
            spread_lag = spread_lag[valid_idx]
            spread_diff = spread_diff[valid_idx]
            
            if len(spread_lag) < 10:
                return np.nan
            
            # Regresión para obtener theta
            X = sm.add_constant(spread_lag)
            model = sm.OLS(spread_diff, X).fit()
            
            theta = -model.params.iloc[1] if len(model.params) > 1 else 0
            
            if theta <= 0:
                return np.nan  # No mean-reverting
            
            # Half-life en períodos
            half_life_periods = np.log(2) / theta
            
            # Convertir a días
            periods_per_day = (24 * 60) / self.resample_minutes
            half_life_days = half_life_periods / periods_per_day
            
            return float(half_life_days)
            
        except:
            return np.nan

    def _calculate_robust_hedge_ratio(self, idx: int) -> float:
        """Hedge ratio robusto con regularización y validación"""
        if self.data is None or self.data.empty:
            return 1.0

        days = float(self.params.get('hedge_ratio_window', 30))
        periods = self._periods_from_days(days)
        
        # Asegurar suficiente historia
        if idx < periods:
            start_idx = max(0, idx - 100)  # Usar al menos 100 observaciones
            end_idx = idx + 1
        else:
            start_idx = idx - periods + 1
            end_idx = idx + 1
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        if len(window_data) < 50:  # Mínimo 50 observaciones
            return 1.0
        
        try:
            # Método 1: OLS con validación
            y = window_data['log_price1']
            X = sm.add_constant(window_data['log_price2'])
            model = sm.OLS(y, X).fit()
            
            # Validación de calidad
            if model.rsquared < 0.5:  # R² mínimo
                return 1.0
            
            hedge_ratio_ols = float(model.params.iloc[1])
            
            # Método 2: Correlación-based (más estable)
            returns1 = window_data['returns1'].dropna()
            returns2 = window_data['returns2'].dropna()
            
            if len(returns1) > 20:
                corr = returns1.corr(returns2)
                std1 = returns1.std()
                std2 = returns2.std()
                
                if std2 > 0:
                    hedge_ratio_corr = corr * (std1 / std2)
                else:
                    hedge_ratio_corr = 1.0
            else:
                hedge_ratio_corr = hedge_ratio_ols
            
            # Combinar métodos (weighted average)
            weight_ols = model.rsquared  # Peso basado en R²
            weight_corr = 1 - weight_ols
            
            hedge_ratio = (hedge_ratio_ols * weight_ols + 
                          hedge_ratio_corr * weight_corr)
            
            # Aplicar límites
            max_hr = float(self.params.get('max_hedge_ratio', 2.0))
            min_hr = float(self.params.get('min_hedge_ratio', 0.2))
            hedge_ratio = np.clip(hedge_ratio, min_hr, max_hr)
            
            # Suavizado exponencial para evitar cambios bruscos
            if self.hedge_ratio_history:
                last_hr = self.hedge_ratio_history[-1]['hedge_ratio']
                alpha = 0.3  # Factor de suavizado
                hedge_ratio = alpha * hedge_ratio + (1 - alpha) * last_hr
            
            # Guardar para análisis
            self.hedge_ratio_history.append({
                'idx': idx,
                'hedge_ratio': hedge_ratio,
                'hedge_ratio_ols': hedge_ratio_ols,
                'hedge_ratio_corr': hedge_ratio_corr if 'hedge_ratio_corr' in locals() else hedge_ratio_ols,
                'r_squared': float(model.rsquared),
                'correlation': float(corr) if 'corr' in locals() else 0,
                'window_size': len(window_data)
            })
            
            return hedge_ratio
            
        except Exception as e:
            log.warning(f"Error calculating robust hedge ratio at idx {idx}: {e}")
            return self.hedge_ratio_history[-1]['hedge_ratio'] if self.hedge_ratio_history else 1.0

    def _calculate_spread_at(self, idx: int, hedge_ratio: float) -> float:
        """Calcular spread con hedge ratio específico"""
        if self.data is None:
            return 0.0
        
        log_price1 = self.data.iloc[idx]['log_price1']
        log_price2 = self.data.iloc[idx]['log_price2']
        
        return log_price1 - hedge_ratio * log_price2

    def _calculate_robust_zscore(self, idx: int, hedge_ratio: float) -> float:
        """Z-score robusto usando MAD o std según configuración"""
        if self.data is None:
            return 0.0

        days = float(self.params.get('lookback_window', 5))
        periods = self._periods_from_days(days)
        
        # Validación crítica
        if periods < 2:
            log.error(f"❌ Lookback periods too small: {periods} (days={days}, resample={self.resample_minutes})")
            periods = max(20, periods)  # Mínimo 20 períodos
        
        if idx < periods:
            return 0.0
        
        # Calcular spreads para la ventana
        start_idx = idx - periods + 1
        window_spreads = []
        
        for i in range(start_idx, idx + 1):
            spread_i = self._calculate_spread_at(i, hedge_ratio)
            window_spreads.append(spread_i)
        
        window_spreads = np.array(window_spreads)
        current_spread = window_spreads[-1]
        
        if self.params.get('use_robust_zscore', True):
            # Método robusto usando MAD (Median Absolute Deviation)
            median = np.median(window_spreads)
            mad = np.median(np.abs(window_spreads - median))
            
            # Convertir MAD a escala de std (factor 1.4826)
            mad_std = mad * 1.4826
            
            # Evitar división por cero
            if mad_std < 1e-8:
                mad_std = np.std(window_spreads)
                if mad_std < 1e-8:
                    return 0.0
            
            zscore = (current_spread - median) / mad_std
            
        else:
            # Método tradicional
            mean = np.mean(window_spreads)
            std = np.std(window_spreads)
            
            if std < 1e-8:
                return 0.0
            
            zscore = (current_spread - mean) / std
        
        # Cap para evitar valores extremos
        zscore = np.clip(zscore, -5, 5)
        
        # Guardar para análisis (limitar frecuencia)
        if idx % 10 == 0 or abs(zscore) > 2:
            self.z_score_history.append({
                'idx': idx,
                'zscore': float(zscore),
                'spread': float(current_spread),
                'hedge_ratio': hedge_ratio,
                'lookback_periods': periods
            })
        
        return float(zscore)

    def _calculate_spread_volatility(self, idx: int, hedge_ratio: float) -> float:
        """Calcular volatilidad del spread"""
        days = float(self.params.get('volatility_lookback', 10))
        periods = self._periods_from_days(days)
        
        if idx < periods:
            return 0.01  # Valor por defecto
        
        start_idx = idx - periods + 1
        spreads = []
        
        for i in range(start_idx, idx + 1):
            spread_i = self._calculate_spread_at(i, hedge_ratio)
            spreads.append(spread_i)
        
        return float(np.std(spreads))

    def _check_volume_filter(self, idx: int) -> bool:
        """Verificar si el volumen es suficiente"""
        if not self.params.get('volume_filter', True):
            return True
        
        if self.data is None or idx < 96:  # Necesitamos 24h de historia
            return True
        
        # Volumen promedio últimas 4 horas
        vol1_recent = self.data.iloc[idx-16:idx+1]['volume1'].mean()
        vol2_recent = self.data.iloc[idx-16:idx+1]['volume2'].mean()
        
        # Comparar con percentil mínimo
        min_percentile = self.params.get('min_volume_percentile', 20) / 100
        
        if not self.volume_stats:
            return True
        
        vol1_threshold = self.volume_stats['vol1_mean'] * min_percentile
        vol2_threshold = self.volume_stats['vol2_mean'] * min_percentile
        
        return vol1_recent > vol1_threshold and vol2_recent > vol2_threshold

    def _get_dynamic_thresholds(self, volatility: float) -> Dict[str, float]:
        """Ajustar thresholds según volatilidad"""
        if not self.params.get('use_dynamic_thresholds', True):
            return {
                'entry': float(self.params.get('entry_threshold', 2.5)),
                'exit': float(self.params.get('exit_threshold', 0.3)),
                'stop': float(self.params.get('stop_loss_threshold', 3.5))
            }
        
        # Volatilidad normalizada (asumiendo 1% como referencia)
        vol_factor = volatility / 0.01
        vol_factor = np.clip(vol_factor, 0.5, 2.0)  # Limitar ajuste
        
        # Ajustar thresholds (más estrictos con alta volatilidad)
        base_entry = float(self.params.get('entry_threshold', 2.5))
        base_exit = float(self.params.get('exit_threshold', 0.3))
        base_stop = float(self.params.get('stop_loss_threshold', 3.5))
        
        return {
            'entry': base_entry * np.sqrt(vol_factor),  # Más conservador con alta vol
            'exit': base_exit / np.sqrt(vol_factor),    # Salir antes con alta vol
            'stop': base_stop * np.sqrt(vol_factor)     # Stop más amplio con alta vol
        }

    def _correlation_ok(self, idx: int) -> bool:
        """Verificar correlación mínima con ventana adaptativa"""
        days = float(self.params.get('lookback_window', 5))
        periods = self._periods_from_days(days)
        
        if idx < periods:
            return True
        
        start_idx = idx - periods + 1
        returns1 = self.data.iloc[start_idx:idx+1]['returns1'].dropna()
        returns2 = self.data.iloc[start_idx:idx+1]['returns2'].dropna()
        
        if len(returns1) < 10 or len(returns2) < 10:
            return True
        
        corr = returns1.corr(returns2)
        min_corr = float(self.params.get('min_correlation', 0.6))
        
        return (corr >= min_corr) if not np.isnan(corr) else False

    def generate_signal(self, timestamp: datetime, row: pd.Series, index: int) -> Optional[Dict]:
        """Generar señales con filtros avanzados y validación mejorada"""
        
        # Skip señales tempranas
        min_warmup = max(100, self._periods_from_days(10))
        if index < min_warmup:
            return {'action': 'HOLD', 'reason': 'warmup_period'}
        
        # Actualizar hedge ratio si no hay posición
        if not self.current_position:
            self.current_hedge_ratio = self._calculate_robust_hedge_ratio(index)
        
        hedge_ratio_to_use = self.position_hedge_ratio if self.current_position else self.current_hedge_ratio
        
        # Calcular métricas principales
        z = self._calculate_robust_zscore(index, hedge_ratio_to_use)
        current_spread = self._calculate_spread_at(index, hedge_ratio_to_use)
        spread_volatility = self._calculate_spread_volatility(index, hedge_ratio_to_use)
        
        # Guardar para histórico (menos frecuente para reducir spam)
        if index % 50 == 0 or (self.signals_history and len(self.signals_history) > 0 and 
                               abs(z - self.signals_history[-1].get('z_score', 0)) > 0.5):
            self.signals_history.append({
                'timestamp': timestamp,
                'z_score': float(z),
                'spread': float(current_spread),
                'hedge_ratio': float(hedge_ratio_to_use),
                'spread_volatility': float(spread_volatility),
                'has_position': self.current_position is not None,
                'price1': float(row['price1']),
                'price2': float(row['price2'])
            })
        
        # Debug periódico (menos frecuente)
        if index % 1000 == 0 and index > 0:
            log.info(
                f"📊 PERIODIC | idx:{index} | Z:{z:.3f} | Vol:{spread_volatility:.4f} | "
                f"HR:{self.current_hedge_ratio:.4f} | Pos:{self.current_position or 'None'}"
            )
        
        # === FILTROS DE CALIDAD ===
        
        # 1. Correlación
        if not self._correlation_ok(index):
            if self.current_position:
                # Solo loguear si es crítico
                if abs(z) > 2:
                    log.info(f"⚠️ LOW CORR EXIT | Z:{z:.3f}")
                self.current_position = None
                self.position_hedge_ratio = None
                return {
                    'action': 'CLOSE',
                    'position_id': 'current',
                    'reason': 'low_correlation',
                    'z_score': float(z)
                }
            return {'action': 'HOLD', 'reason': 'low_correlation'}
        
        # 2. Filtro de volatilidad
        min_vol = float(self.params.get('min_spread_volatility', 0.001))
        max_vol = float(self.params.get('max_spread_volatility', 0.05))
        
        if spread_volatility < min_vol:
            self.rejected_signals.append({'reason': 'low_volatility', 'z': z})
            return {'action': 'HOLD', 'reason': 'low_spread_volatility'}
        
        if spread_volatility > max_vol:
            if self.current_position:
                log.info(f"⚠️ HIGH VOL EXIT | Vol:{spread_volatility:.4f} > {max_vol}")
                self.current_position = None
                self.position_hedge_ratio = None
                return {
                    'action': 'CLOSE',
                    'position_id': 'current',
                    'reason': 'high_volatility',
                    'z_score': float(z)
                }
            return {'action': 'HOLD', 'reason': 'high_spread_volatility'}
        
        # 3. Filtro de volumen
        if not self._check_volume_filter(index):
            self.rejected_signals.append({'reason': 'low_volume', 'z': z})
            return {'action': 'HOLD', 'reason': 'insufficient_volume'}
        
        # 4. Filtro de half-life (solo para nuevas entradas)
        if not self.current_position and self.params.get('use_half_life_filter', True):
            # Calcular half-life cada 100 períodos o si no está en caché
            cache_key = f"{index//100}_{hedge_ratio_to_use:.4f}"
            
            if cache_key not in self.half_life_cache:
                lookback_periods = self._periods_from_days(30)
                if index >= lookback_periods:
                    spread_series = pd.Series([
                        self._calculate_spread_at(i, hedge_ratio_to_use) 
                        for i in range(index - lookback_periods + 1, index + 1)
                    ])
                    half_life = self._calculate_half_life(spread_series)
                    self.half_life_cache[cache_key] = half_life
                else:
                    half_life = np.nan
            else:
                half_life = self.half_life_cache[cache_key]
            
            if not np.isnan(half_life):
                min_hl = float(self.params.get('min_half_life', 1.0))
                max_hl = float(self.params.get('max_half_life', 15.0))
                
                if half_life < min_hl or half_life > max_hl:
                    self.rejected_signals.append({'reason': 'half_life_filter', 'z': z, 'hl': half_life})
                    return {'action': 'HOLD', 'reason': f'half_life_out_of_range_{half_life:.1f}d'}
        
        # === SEÑALES DE TRADING ===
        
        # Obtener thresholds dinámicos
        thresholds = self._get_dynamic_thresholds(spread_volatility)
        entry_thr = thresholds['entry']
        exit_thr = thresholds['exit']
        stop_thr = thresholds['stop']
        
        signal: Optional[Dict[str, Any]] = None
        
        if not self.current_position:
            # === APERTURA DE POSICIONES ===
            
            if z <= -entry_thr:
                # LONG: Spread infravalorado
                self.current_position = 'LONG'
                self.position_hedge_ratio = self.current_hedge_ratio
                self.entry_spread = current_spread
                self.entry_z = z
                self.entry_timestamp = timestamp
                
                log.info(f"📈 LONG | Z:{z:.3f}≤-{entry_thr:.2f} | "
                        f"HR:{self.position_hedge_ratio:.4f} | Vol:{spread_volatility:.4f}")
                
                signal = {
                    'action': 'LONG',
                    'strength': min(abs(z) / stop_thr, 1.0),
                    'reason': 'spread_undervalued',
                    'z_score': float(z),
                    'hedge_ratio': float(self.position_hedge_ratio),
                    'spread': float(current_spread),
                    'volatility': float(spread_volatility),
                    'thresholds': thresholds
                }
                
            elif z >= entry_thr:
                # SHORT: Spread sobrevalorado
                self.current_position = 'SHORT'
                self.position_hedge_ratio = self.current_hedge_ratio
                self.entry_spread = current_spread
                self.entry_z = z
                self.entry_timestamp = timestamp
                
                log.info(f"📉 SHORT | Z:{z:.3f}≥{entry_thr:.2f} | "
                        f"HR:{self.position_hedge_ratio:.4f} | Vol:{spread_volatility:.4f}")
                
                signal = {
                    'action': 'SHORT',
                    'strength': min(abs(z) / stop_thr, 1.0),
                    'reason': 'spread_overvalued',
                    'z_score': float(z),
                    'hedge_ratio': float(self.position_hedge_ratio),
                    'spread': float(current_spread),
                    'volatility': float(spread_volatility),
                    'thresholds': thresholds
                }
        else:
            # === CIERRE DE POSICIONES ===
            
            # Calcular tiempo en posición
            if self.entry_timestamp:
                time_in_position = (timestamp - self.entry_timestamp).total_seconds() / 3600  # horas
            else:
                time_in_position = 0
            
            # Verificar coherencia (sin spam de logs)
            if hasattr(self, 'entry_spread') and hasattr(self, 'entry_z'):
                spread_change = current_spread - self.entry_spread
                z_change = z - self.entry_z
                
                # Solo alertar si es significativo y no spam
                if abs(spread_change) > 0.1 and abs(z_change) > 1.0:
                    if not hasattr(self, '_last_coherence_warning') or \
                       (timestamp - self._last_coherence_warning).total_seconds() > 3600:  # Max 1 por hora
                        
                        expected_direction = np.sign(spread_change)
                        actual_direction = np.sign(z_change)
                        
                        if expected_direction != actual_direction:
                            log.debug(f"Z-Score check: Spread Δ{spread_change:+.6f}, Z Δ{z_change:+.3f}")
                            self._last_coherence_warning = timestamp
            
            close_signal = False
            close_reason = None
            
            if self.current_position == 'LONG':
                # LONG: Salida normal o stop loss
                if z >= exit_thr:
                    close_signal = True
                    close_reason = 'exit_signal'
                    if abs(z - exit_thr) > 0.1:  # Solo log si es significativo
                        log.info(f"🔄 LONG EXIT | Z:{z:.3f}≥{exit_thr:.2f}")
                    
                elif z < -stop_thr:
                    close_signal = True
                    close_reason = 'stop_loss'
                    log.info(f"🛑 LONG STOP | Z:{z:.3f}<-{stop_thr:.2f}")
                    
                # Time stop (opcional)
                elif time_in_position > 24 * 7:  # 7 días máximo
                    close_signal = True
                    close_reason = 'time_stop'
                    log.info(f"⏰ LONG TIME STOP | {time_in_position/24:.1f} days")
                    
            elif self.current_position == 'SHORT':
                # SHORT: Salida normal o stop loss
                if z <= -exit_thr:
                    close_signal = True
                    close_reason = 'exit_signal'
                    if abs(z + exit_thr) > 0.1:  # Solo log si es significativo
                        log.info(f"🔄 SHORT EXIT | Z:{z:.3f}≤-{exit_thr:.2f}")
                    
                elif z > stop_thr:
                    close_signal = True
                    close_reason = 'stop_loss'
                    log.info(f"🛑 SHORT STOP | Z:{z:.3f}>{stop_thr:.2f}")
                    
                # Time stop
                elif time_in_position > 24 * 7:  # 7 días máximo
                    close_signal = True
                    close_reason = 'time_stop'
                    log.info(f"⏰ SHORT TIME STOP | {time_in_position/24:.1f} days")
            
            if close_signal:
                self.current_position = None
                self.position_hedge_ratio = None
                self.entry_spread = None
                self.entry_z = None
                self.entry_timestamp = None
                
                signal = {
                    'action': 'CLOSE',
                    'position_id': 'current',
                    'reason': close_reason,
                    'z_score': float(z),
                    'spread': float(current_spread),
                    'time_in_position_hours': float(time_in_position)
                }
        
        return signal or {'action': 'HOLD'}

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Métricas extendidas de la estrategia"""
        df = self.get_signals_history()
        
        metrics = {
            'signals_generated': len(df) if not df.empty else 0,
            'current_hedge_ratio': float(self.current_hedge_ratio),
            'position_hedge_ratio': float(self.position_hedge_ratio) if self.position_hedge_ratio else None,
            'resample_minutes': self.resample_minutes,
            'lookback_days': float(self.params.get('lookback_window', 5)),
            'lookback_periods': self._periods_from_days(self.params.get('lookback_window', 5)),
        }
        
        if not df.empty and 'z_score' in df:
            metrics.update({
                'avg_z_score': float(df['z_score'].mean()),
                'max_z_score': float(df['z_score'].max()),
                'min_z_score': float(df['z_score'].min()),
                'z_score_std': float(df['z_score'].std()),
                'extreme_z_scores': int((df['z_score'].abs() > 3).sum()),
            })
        
        if self.hedge_ratio_history:
            recent_hr = [x['hedge_ratio'] for x in self.hedge_ratio_history[-100:]]
            metrics.update({
                'hedge_ratio_mean': float(np.mean(recent_hr)),
                'hedge_ratio_std': float(np.std(recent_hr)),
                'hedge_ratio_min': float(np.min(recent_hr)),
                'hedge_ratio_max': float(np.max(recent_hr)),
            })
        
        if self.rejected_signals:
            rejection_reasons = pd.Series([x['reason'] for x in self.rejected_signals])
            metrics['rejection_reasons'] = rejection_reasons.value_counts().to_dict()
            metrics['total_rejections'] = len(self.rejected_signals)
        
        if hasattr(self, 'half_life_cache') and self.half_life_cache:
            half_lives = [hl for hl in self.half_life_cache.values() if not np.isnan(hl)]
            if half_lives:
                metrics['avg_half_life_days'] = float(np.mean(half_lives))
                metrics['median_half_life_days'] = float(np.median(half_lives))
        
        return metrics