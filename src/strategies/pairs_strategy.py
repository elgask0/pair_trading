#!/usr/bin/env python3
"""
Pairs Trading Strategy - VERSIÓN MEJORADA
Con filtros de R², time stops, y volatility scaling
"""

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.strategies.base_strategy import BaseStrategy
from src.utils.logger import get_logger

log = get_logger()


class PairsTradingStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        defaults = {
            # Ventanas temporales
            "lookback_window": 7,  # DÍAS para z-score
            "hedge_ratio_window": 7,  # DÍAS para hedge ratio
            
            # Thresholds de trading
            "entry_threshold": 2.0,
            "exit_threshold": 0.0,
            "stop_loss_threshold": 4.0,
            
            # NUEVOS: Filtros de calidad
            "min_correlation": 0.6,
            "min_correlation_rolling": 0.6,  # Para 24h rolling
            "min_r_squared": 0.5,  # Mínimo R² para hedge ratio
            "max_hedge_ratio": 5.0,  # Máximo beta aceptable
            "min_hedge_ratio": 0.1,  # Mínimo beta aceptable
            
            # NUEVOS: Risk management
            "max_position_hours": 48,  # Time stop
            "volatility_lookback_hours": 24,  # Para calcular vol
            "high_volatility_multiplier": 1.5,  # Threshold para reducir size
            "position_size_reduction": 0.5,  # Reducción cuando hay alta vol
            
            # Otros
            "use_robust_zscore": False,
            "rebalance_frequency": "never",
            "signal_confirmation_bars": 0,
            "max_holding_periods": 0,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

        # Estado
        self.hedge_ratio: float = 1.0
        self.current_hedge_ratio: Optional[float] = None
        self.spread: Optional[pd.Series] = None
        self.current_position: Optional[str] = None
        self.position_hedge_ratio: Optional[float] = None
        
        # NUEVO: Tracking mejorado
        self.entry_timestamp: Optional[datetime] = None
        self.entry_price1: Optional[float] = None
        self.entry_price2: Optional[float] = None
        self.last_r_squared: float = 0.0
        self.current_volatility: float = 1.0
        self.normal_volatility: float = 1.0

        # Configuración
        self.resample_minutes: int = 15

        # Debug
        self.z_score_history = []
        self.hedge_ratio_history = []
        self.volatility_history = []

    def set_resample_minutes(self, resample_minutes: int):
        """Recibir resample_minutes del engine"""
        self.resample_minutes = resample_minutes
        log.info(f"📊 Strategy resample set to: {resample_minutes} minutes")

    def _periods_from_days(self, days: float) -> int:
        """Convertir días a períodos según resample"""
        periods = int(days * 24 * 60 / self.resample_minutes)
        return max(1, periods)
    
    def _periods_from_hours(self, hours: float) -> int:
        """Convertir horas a períodos según resample"""
        periods = int(hours * 60 / self.resample_minutes)
        return max(1, periods)

    def initialize(self):
        """Inicialización con nuevos parámetros"""
        lookback_days = int(self.params.get("lookback_window", 7))
        lookback_periods = self._periods_from_days(lookback_days)

        hedge_days = int(self.params.get("hedge_ratio_window", 7))
        hedge_periods = self._periods_from_days(hedge_days)

        log.info(f"🚀 STRATEGY INIT | Resample: {self.resample_minutes}min")
        log.info(
            f"📊 Lookback: {lookback_days}d = {lookback_periods}p | "
            f"Hedge: {hedge_days}d = {hedge_periods}p"
        )
        log.info(
            f"🎯 Entry: ±{self.params.get('entry_threshold', 2.0)} | "
            f"Exit: ±{self.params.get('exit_threshold', 0.0)} | "
            f"Stop: ±{self.params.get('stop_loss_threshold', 4.0)}"
        )
        log.info(
            f"🛡️ Min R²: {self.params.get('min_r_squared', 0.5)} | "
            f"Min Corr: {self.params.get('min_correlation_rolling', 0.6)} | "
            f"Max Hours: {self.params.get('max_position_hours', 48)}"
        )

        self.current_position = None
        self.position_hedge_ratio = None
        self.entry_timestamp = None
        self.signals_history = []
        self.z_score_history = []
        self.hedge_ratio_history = []
        self.volatility_history = []

    def _calculate_hedge_ratio_at(self, idx: int) -> Optional[float]:
        """
        MEJORADO: Calcular hedge ratio con validación de R² y beta
        """
        if self.data is None or idx < 2:
            return None

        days = int(self.params.get("hedge_ratio_window", 7))
        periods = self._periods_from_days(days)

        # Solo usar datos históricos [t-n, t-1]
        if idx < periods:
            start_idx = 0
            end_idx = idx
        else:
            start_idx = idx - periods
            end_idx = idx

        # Necesitamos mínimo 20 observaciones
        if end_idx - start_idx < 20:
            return None

        window_data = self.data.iloc[start_idx:end_idx]

        try:
            y = window_data["log_price1"]
            X = sm.add_constant(window_data["log_price2"])
            model = sm.OLS(y, X).fit()
            
            hedge_ratio = float(model.params.iloc[1])
            r_squared = float(model.rsquared)
            p_value = float(model.pvalues.iloc[1])
            
            # NUEVO: Validación de R²
            min_r_squared = float(self.params.get("min_r_squared", 0.5))
            if r_squared < min_r_squared:
               # log.warning(f"⚠️ R² too low: {r_squared:.3f} < {min_r_squared}")
                self.last_r_squared = r_squared
                return None
            
            # NUEVO: Validación de beta extremo
            max_hr = float(self.params.get("max_hedge_ratio", 5.0))
            min_hr = float(self.params.get("min_hedge_ratio", 0.1))
            if hedge_ratio < min_hr or hedge_ratio > max_hr:
                log.warning(f"⚠️ Extreme beta: {hedge_ratio:.3f} not in [{min_hr}, {max_hr}]")
                return None
            
            # NUEVO: Validación de significancia
            if p_value > 0.05:
                log.warning(f"⚠️ Beta not significant: p-value={p_value:.4f}")
                return None

            # Guardar para debug
            self.hedge_ratio_history.append({
                "idx": idx,
                "hedge_ratio": hedge_ratio,
                "r_squared": r_squared,
                "p_value": p_value,
                "window_size": len(window_data),
                "valid": True
            })
            
            self.last_r_squared = r_squared
            return hedge_ratio

        except Exception as e:
            log.warning(f"Error calculating hedge ratio at idx {idx}: {e}")
            return None

    def _calculate_spread_at(self, idx: int, hedge_ratio: float) -> float:
        """Calcular spread con hedge ratio específico"""
        if self.data is None:
            return 0.0

        log_price1 = self.data.iloc[idx]["log_price1"]
        log_price2 = self.data.iloc[idx]["log_price2"]

        return log_price1 - hedge_ratio * log_price2

    def _zscore_at(self, idx: int, hedge_ratio: float) -> float:
        """
        Calcular z-score SIN look-ahead bias
        """
        if self.data is None or idx < 2:
            return 0.0

        days = int(self.params.get("lookback_window", 7))
        periods = self._periods_from_days(days)

        if idx < periods + 1:
            return 0.0

        # Ventana histórica [t-n, t-1] EXCLUYENDO t0
        start_idx = idx - periods
        end_idx = idx

        # Calcular spreads históricos
        historical_spreads = []
        for i in range(start_idx, end_idx):
            spread_i = self._calculate_spread_at(i, hedge_ratio)
            historical_spreads.append(spread_i)

        historical_spreads = np.array(historical_spreads)

        # Estadísticas históricas
        mean_hist = np.mean(historical_spreads)
        std_hist = np.std(historical_spreads)

        # Safeguard
        min_std = abs(mean_hist) * 0.001 if mean_hist != 0 else 0.0001
        std_hist = max(std_hist, min_std)

        if std_hist == 0:
            return 0.0

        # Spread actual
        current_spread = self._calculate_spread_at(idx, hedge_ratio)

        # Z-score
        zscore = (current_spread - mean_hist) / std_hist
        zscore = np.clip(zscore, -5, 5)

        # Debug
        self.z_score_history.append({
            "idx": idx,
            "zscore": float(zscore),
            "spread": float(current_spread),
            "mean_hist": float(mean_hist),
            "std_hist": float(std_hist),
            "hedge_ratio": hedge_ratio
        })

        return float(zscore)

    def _correlation_ok(self, idx: int) -> bool:
        """
        MEJORADO: Verificar correlación rolling de 24h
        """
        hours = 24
        periods = self._periods_from_hours(hours)

        if idx < periods + 1:
            return True

        # Ventana de 24h
        start_idx = idx - periods
        end_idx = idx
        
        returns1 = self.data.iloc[start_idx:end_idx]["returns1"]
        returns2 = self.data.iloc[start_idx:end_idx]["returns2"]

        corr = returns1.corr(returns2)
        min_corr = float(self.params.get("min_correlation_rolling", 0.6))

        if not np.isnan(corr) and corr < min_corr:
            log.debug(f"Low correlation: {corr:.3f} < {min_corr}")
            return False
        
        return True

    def _calculate_current_volatility(self, idx: int) -> float:
        """
        NUEVO: Calcular volatilidad actual para ajuste de posición
        """
        hours = float(self.params.get("volatility_lookback_hours", 24))
        periods = self._periods_from_hours(hours)
        
        if idx < periods:
            return 1.0
        
        returns = self.data.iloc[idx-periods:idx]["returns1"]
        
        # Volatilidad anualizada
        vol = returns.std() * np.sqrt(365 * 24 * 60 / self.resample_minutes)
        
        # Guardar historial
        self.volatility_history.append({
            "idx": idx,
            "volatility": float(vol)
        })
        
        return float(vol)

    def _check_time_stop(self, timestamp: datetime, z: float) -> Optional[Dict]:
        """
        NUEVO: Verificar time-based stop loss
        """
        if not self.current_position or not self.entry_timestamp:
            return None
        
        max_hours = float(self.params.get("max_position_hours", 48))
        duration = timestamp - self.entry_timestamp
        
        if duration > timedelta(hours=max_hours):
            # Estimar PnL aproximado (simplificado)
            # En producción, obtendríamos el PnL real del portfolio
            
            log.info(f"⏰ TIME STOP | Duration: {duration} > {max_hours}h | Z:{z:.3f}")
            
            self.current_position = None
            self.position_hedge_ratio = None
            self.entry_timestamp = None
            self.entry_price1 = None
            self.entry_price2 = None
            
            return {
                "action": "CLOSE",
                "position_id": "current",
                "reason": "time_stop",
                "z_score": float(z),
                "duration_hours": duration.total_seconds() / 3600
            }
        
        return None

    def generate_signal(self, timestamp: datetime, row: pd.Series, index: int) -> Optional[Dict]:
        """
        MEJORADO: Generar señales con todos los filtros nuevos
        """
        
        # NUEVO: Calcular volatilidad actual
        self.current_volatility = self._calculate_current_volatility(index)
        
        # Si no hay posición, calcular nuevo hedge ratio
        if not self.current_position:
            self.current_hedge_ratio = self._calculate_hedge_ratio_at(index)
            
            # NUEVO: Si no hay hedge ratio válido, no tradear
            if self.current_hedge_ratio is None:
                return {"action": "HOLD", "reason": "invalid_hedge_ratio"}

        hedge_ratio_to_use = (
            self.position_hedge_ratio
            if self.current_position
            else self.current_hedge_ratio
        )
        
        # Si no hay hedge ratio válido, no hacer nada
        if hedge_ratio_to_use is None:
            return {"action": "HOLD", "reason": "no_valid_hedge_ratio"}

        # Calcular z-score y spread
        z = self._zscore_at(index, hedge_ratio_to_use)
        current_spread = self._calculate_spread_at(index, hedge_ratio_to_use)

        # Guardar para diagnóstico
        self.signals_history.append({
            "timestamp": timestamp,
            "z_score": float(z),
            "spread": float(current_spread),
            "hedge_ratio": float(hedge_ratio_to_use),
            "r_squared": self.last_r_squared,
            "volatility": self.current_volatility,
            "has_position": self.current_position is not None,
            "price1": float(row["price1"]),
            "price2": float(row["price2"]),
        })

        # Debug
        if index % 500 == 0 and index > 0:
            log.info(
                f"📊 DEBUG | idx:{index} | Z:{z:.3f} | Spread:{current_spread:.6f} | "
                f"HR:{hedge_ratio_to_use:.4f} | R²:{self.last_r_squared:.3f} | "
                f"Vol:{self.current_volatility:.2f} | Pos:{self.current_position or 'None'}"
            )

        # NUEVO: Check time stop si hay posición
        if self.current_position:
            time_stop_signal = self._check_time_stop(timestamp, z)
            if time_stop_signal:
                return time_stop_signal

        # Check correlación
        if not self._correlation_ok(index):
            if self.current_position:
                log.info(f"⚠️ LOW CORR EXIT | Z:{z:.3f}")
                self.current_position = None
                self.position_hedge_ratio = None
                self.entry_timestamp = None
                self.entry_price1 = None
                self.entry_price2 = None
                return {
                    "action": "CLOSE",
                    "position_id": "current",
                    "reason": "low_correlation",
                    "z_score": float(z),
                }
            return {"action": "HOLD", "reason": "low_correlation"}

        # Parámetros de trading
        entry = float(self.params.get("entry_threshold", 2.0))
        exit_thr = float(self.params.get("exit_threshold", 0.0))
        stop = float(self.params.get("stop_loss_threshold", 4.0))

        signal: Optional[Dict[str, Any]] = None

        if not self.current_position:
            # === ENTRADA ===
            if z <= -entry:
                # LONG
                self.current_position = "LONG"
                self.position_hedge_ratio = self.current_hedge_ratio
                self.entry_spread = current_spread
                self.entry_z = z
                self.entry_timestamp = timestamp
                self.entry_price1 = row["price1"]
                self.entry_price2 = row["price2"]

                log.info(
                    f"📈 LONG SIGNAL | Z:{z:.4f} | HR:{self.position_hedge_ratio:.4f} | "
                    f"R²:{self.last_r_squared:.3f} | Vol:{self.current_volatility:.2f}"
                )

                signal = {
                    "action": "LONG",
                    "strength": min(abs(z) / stop, 1.0),
                    "reason": "spread_undervalued",
                    "z_score": float(z),
                    "hedge_ratio": float(self.position_hedge_ratio),
                    "spread": float(current_spread),
                    "r_squared": self.last_r_squared
                }

            elif z >= entry:
                # SHORT
                self.current_position = "SHORT"
                self.position_hedge_ratio = self.current_hedge_ratio
                self.entry_spread = current_spread
                self.entry_z = z
                self.entry_timestamp = timestamp
                self.entry_price1 = row["price1"]
                self.entry_price2 = row["price2"]

                log.info(
                    f"📉 SHORT SIGNAL | Z:{z:.4f} | HR:{self.position_hedge_ratio:.4f} | "
                    f"R²:{self.last_r_squared:.3f} | Vol:{self.current_volatility:.2f}"
                )

                signal = {
                    "action": "SHORT",
                    "strength": min(abs(z) / stop, 1.0),
                    "reason": "spread_overvalued",
                    "z_score": float(z),
                    "hedge_ratio": float(self.position_hedge_ratio),
                    "spread": float(current_spread),
                    "r_squared": self.last_r_squared
                }
            
            # NUEVO: Ajuste por volatilidad
            if signal and signal["action"] in ["LONG", "SHORT"]:
                high_vol_mult = float(self.params.get("high_volatility_multiplier", 1.5))
                
                # Calcular volatilidad "normal" (percentil 50 de las últimas observaciones)
                if len(self.volatility_history) > 100:
                    recent_vols = [v["volatility"] for v in self.volatility_history[-100:]]
                    normal_vol = np.percentile(recent_vols, 50)
                else:
                    normal_vol = 1.0
                
                if self.current_volatility > normal_vol * high_vol_mult:
                    size_reduction = float(self.params.get("position_size_reduction", 0.5))
                    signal["position_size_multiplier"] = size_reduction
                    log.info(f"⚠️ HIGH VOL | Reducing position size to {size_reduction:.0%}")
                else:
                    signal["position_size_multiplier"] = 1.0

        else:
            # === SALIDA ===
            if self.current_position == "LONG":
                if z >= exit_thr:
                    log.info(f"🔄 LONG EXIT | Z:{z:.4f} ≥ {exit_thr}")
                    self.current_position = None
                    self.position_hedge_ratio = None
                    self.entry_timestamp = None
                    self.entry_price1 = None
                    self.entry_price2 = None
                    signal = {
                        "action": "CLOSE",
                        "position_id": "current",
                        "reason": "exit_signal",
                        "z_score": float(z),
                        "spread": float(current_spread),
                    }
                elif z < -stop:
                    log.info(f"🛑 LONG STOP | Z:{z:.4f} < -{stop}")
                    self.current_position = None
                    self.position_hedge_ratio = None
                    self.entry_timestamp = None
                    self.entry_price1 = None
                    self.entry_price2 = None
                    signal = {
                        "action": "CLOSE",
                        "position_id": "current",
                        "reason": "stop_loss",
                        "z_score": float(z),
                        "spread": float(current_spread),
                    }

            elif self.current_position == "SHORT":
                if z <= -exit_thr:
                    log.info(f"🔄 SHORT EXIT | Z:{z:.4f} ≤ {-exit_thr}")
                    self.current_position = None
                    self.position_hedge_ratio = None
                    self.entry_timestamp = None
                    self.entry_price1 = None
                    self.entry_price2 = None
                    signal = {
                        "action": "CLOSE",
                        "position_id": "current",
                        "reason": "exit_signal",
                        "z_score": float(z),
                        "spread": float(current_spread),
                    }
                elif z > stop:
                    log.info(f"🛑 SHORT STOP | Z:{z:.4f} > +{stop}")
                    self.current_position = None
                    self.position_hedge_ratio = None
                    self.entry_timestamp = None
                    self.entry_price1 = None
                    self.entry_price2 = None
                    signal = {
                        "action": "CLOSE",
                        "position_id": "current",
                        "reason": "stop_loss",
                        "z_score": float(z),
                        "spread": float(current_spread),
                    }

        return signal or {"action": "HOLD"}

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Métricas mejoradas de la estrategia"""
        df = self.get_signals_history()
        if df.empty:
            return {}

        recent_z = [x["zscore"] for x in self.z_score_history[-100:]] if self.z_score_history else []
        recent_hr = [x["hedge_ratio"] for x in self.hedge_ratio_history[-10:]] if self.hedge_ratio_history else []
        recent_vol = [x["volatility"] for x in self.volatility_history[-100:]] if self.volatility_history else []
        recent_r2 = [x["r_squared"] for x in self.hedge_ratio_history[-10:] if "r_squared" in x] if self.hedge_ratio_history else []

        return {
            "avg_z_score": float(df["z_score"].mean()) if "z_score" in df else 0,
            "max_z_score": float(df["z_score"].max()) if "z_score" in df else 0,
            "min_z_score": float(df["z_score"].min()) if "z_score" in df else 0,
            "z_score_std": float(df["z_score"].std()) if "z_score" in df else 0,
            "current_hedge_ratio": float(self.current_hedge_ratio) if self.current_hedge_ratio else 0,
            "position_hedge_ratio": float(self.position_hedge_ratio) if self.position_hedge_ratio else None,
            "hedge_ratio_mean": float(np.mean(recent_hr)) if recent_hr else 1.0,
            "hedge_ratio_std": float(np.std(recent_hr)) if recent_hr else 0.0,
            "avg_r_squared": float(np.mean(recent_r2)) if recent_r2 else 0.0,
            "current_volatility": self.current_volatility,
            "avg_volatility": float(np.mean(recent_vol)) if recent_vol else 1.0,
            "signals_generated": len(df),
            "extreme_z_scores": int((df["z_score"].abs() > 3).sum()) if "z_score" in df and len(df) > 0 else 0,
            "resample_minutes": self.resample_minutes,
            "lookback_days": int(self.params.get("lookback_window", 7)),
            "lookback_periods": self._periods_from_days(int(self.params.get("lookback_window", 7))),
        }