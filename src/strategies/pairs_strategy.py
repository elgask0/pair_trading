#!/usr/bin/env python3
"""
Pairs Trading Strategy - Con validación de coherencia
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
            "lookback_window": 7,  # DÍAS para z-score
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,  # Exit direccional
            "stop_loss_threshold": 4.0,
            "min_correlation": 0.5,
            "use_robust_zscore": False,  # Usar std tradicional
            "hedge_ratio_window": 60,  # DÍAS para hedge ratio
            "rebalance_frequency": "never",  # No rebalancear posiciones existentes
            "signal_confirmation_bars": 0,
            "max_holding_periods": 0,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

        # Estado
        self.hedge_ratio: float = 1.0
        self.current_hedge_ratio: float = 1.0  # Para nuevas posiciones
        self.spread: Optional[pd.Series] = None
        self.current_position: Optional[str] = None
        self.position_hedge_ratio: Optional[float] = (
            None  # Hedge ratio de la posición actual
        )

        # Configuración
        self.resample_minutes: int = 15

        # Debug
        self.z_score_history = []
        self.hedge_ratio_history = []

    def set_resample_minutes(self, resample_minutes: int):
        """Recibir resample_minutes del engine"""
        self.resample_minutes = resample_minutes
        log.info(f"📊 Strategy resample set to: {resample_minutes} minutes")

    def _periods_from_days(self, days: int) -> int:
        """Convertir días a períodos según resample"""
        periods = int(days * 24 * 60 / self.resample_minutes)
        return max(1, periods)

    def initialize(self):
        """Inicialización - NO usar primeros 60 días, usar ventana rolling"""
        lookback_days = int(self.params.get("lookback_window", 7))
        lookback_periods = self._periods_from_days(lookback_days)

        hedge_days = int(self.params.get("hedge_ratio_window", 60))
        hedge_periods = self._periods_from_days(hedge_days)

        log.info(f"🚀 STRATEGY INIT | Resample: {self.resample_minutes}min")
        log.info(
            f"📊 Lookback: {lookback_days}d = {lookback_periods}p | "
            f"Hedge: {hedge_days}d = {hedge_periods}p"
        )
        log.info(
            f"🎯 Entry: ±{self.params.get('entry_threshold', 2.0)} | "
            f"Exit: ±{self.params.get('exit_threshold', 0.5)} | "
            f"Stop: ±{self.params.get('stop_loss_threshold', 4.0)}"
        )

        # NO calcular hedge ratio inicial con primeros datos
        # Se calculará dinámicamente cuando sea necesario
        self.current_position = None
        self.position_hedge_ratio = None
        self.signals_history = []
        self.z_score_history = []
        self.hedge_ratio_history = []

    def _calculate_hedge_ratio_at(self, idx: int) -> float:
        """Calcular hedge ratio usando ventana ROLLING de los ÚLTIMOS días"""
        if self.data is None or self.data.empty:
            return 1.0

        days = int(self.params.get("hedge_ratio_window", 60))
        periods = self._periods_from_days(days)

        # Asegurar que tenemos suficiente historia
        if idx < periods:
            # Si no hay suficiente historia, usar lo que hay
            start_idx = 0
            end_idx = idx + 1
        else:
            # Usar ventana rolling de los ÚLTIMOS períodos
            start_idx = idx - periods + 1
            end_idx = idx + 1

        window_data = self.data.iloc[start_idx:end_idx]

        if len(window_data) < 20:  # Mínimo 20 observaciones para regresión
            return 1.0

        try:
            y = window_data["log_price1"]
            X = sm.add_constant(window_data["log_price2"])
            model = sm.OLS(y, X).fit()
            hedge_ratio = float(model.params.iloc[1])

            # Guardar para debug
            self.hedge_ratio_history.append(
                {
                    "idx": idx,
                    "hedge_ratio": hedge_ratio,
                    "r_squared": float(model.rsquared),
                    "window_size": len(window_data),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

            return hedge_ratio

        except Exception as e:
            log.warning(f"Error calculating hedge ratio at idx {idx}: {e}")
            return 1.0

    def _calculate_spread_at(self, idx: int, hedge_ratio: float) -> float:
        """Calcular spread con hedge ratio específico"""
        if self.data is None:
            return 0.0

        log_price1 = self.data.iloc[idx]["log_price1"]
        log_price2 = self.data.iloc[idx]["log_price2"]

        return log_price1 - hedge_ratio * log_price2

    def _zscore_at(self, idx: int, hedge_ratio: float) -> float:
        """Calcular z-score usando ventana rolling y hedge ratio específico"""
        if self.data is None:
            return 0.0

        days = int(self.params.get("lookback_window", 7))
        periods = self._periods_from_days(days)

        # Necesitamos suficiente historia
        if idx < periods:
            return 0.0

        # Calcular spread para la ventana con el hedge ratio dado
        start_idx = idx - periods + 1
        window_spreads = []

        for i in range(start_idx, idx + 1):
            spread_i = self._calculate_spread_at(i, hedge_ratio)
            window_spreads.append(spread_i)

        window_spreads = np.array(window_spreads)
        current_spread = window_spreads[-1]

        # Z-score tradicional con safeguards
        mean = np.mean(window_spreads)
        std = np.std(window_spreads)

        # Safeguard contra std muy pequeño
        min_std = abs(mean) * 0.001 if mean != 0 else 0.0001
        std = max(std, min_std)

        if std == 0:
            return 0.0

        zscore = (current_spread - mean) / std

        # Cap para evitar valores extremos
        zscore = np.clip(zscore, -5, 5)

        # Debug
        self.z_score_history.append(
            {
                "idx": idx,
                "zscore": float(zscore),
                "spread": float(current_spread),
                "mean": float(mean),
                "std": float(std),
                "hedge_ratio": hedge_ratio,
            }
        )

        return float(zscore)

    def _correlation_ok(self, idx: int) -> bool:
        """Verificar correlación mínima"""
        days = int(self.params.get("lookback_window", 7))
        periods = self._periods_from_days(days)

        if idx < periods:
            return True

        start_idx = idx - periods + 1
        returns1 = self.data.iloc[start_idx : idx + 1]["returns1"]
        returns2 = self.data.iloc[start_idx : idx + 1]["returns2"]

        corr = returns1.corr(returns2)
        min_corr = float(self.params.get("min_correlation", 0.5))

        return (corr >= min_corr) if not np.isnan(corr) else True

    def generate_signal(
        self, timestamp: datetime, row: pd.Series, index: int
    ) -> Optional[Dict]:
        """Generar señales con validación de coherencia"""

        # Si NO hay posición, calcular NUEVO hedge ratio con datos RECIENTES
        if not self.current_position:
            self.current_hedge_ratio = self._calculate_hedge_ratio_at(index)

        hedge_ratio_to_use = (
            self.position_hedge_ratio
            if self.current_position
            else self.current_hedge_ratio
        )

        # Calcular z-score y spread actual
        z = self._zscore_at(index, hedge_ratio_to_use)
        current_spread = self._calculate_spread_at(index, hedge_ratio_to_use)

        # Guardar para diagnóstico
        self.signals_history.append(
            {
                "timestamp": timestamp,
                "z_score": float(z),
                "spread": float(current_spread),
                "hedge_ratio": float(hedge_ratio_to_use),
                "current_hedge_ratio": float(self.current_hedge_ratio),
                "position_hedge_ratio": (
                    float(self.position_hedge_ratio)
                    if self.position_hedge_ratio
                    else None
                ),
                "has_position": self.current_position is not None,
                "price1": float(row["price1"]),
                "price2": float(row["price2"]),
            }
        )

        # Debug cada 500 observaciones
        if index % 500 == 0 and index > 0:
            log.info(
                f"📊 DEBUG | idx:{index} | Z:{z:.3f} | Spread:{current_spread:.6f} | "
                f"HR_current:{self.current_hedge_ratio:.4f} | "
                f"HR_position:{(0.0 if self.position_hedge_ratio is None else self.position_hedge_ratio):.4f} | "
                f"Position:{self.current_position or 'None'}"
            )

        # Verificar correlación
        if not self._correlation_ok(index):
            if self.current_position:
                log.info(f"⚠️ LOW CORR EXIT | Z:{z:.3f}")
                self.current_position = None
                self.position_hedge_ratio = None
                self.entry_spread = None if hasattr(self, "entry_spread") else None
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
                # LONG: Spread infravalorado
                self.current_position = "LONG"
                self.position_hedge_ratio = self.current_hedge_ratio
                self.entry_spread = current_spread  # Guardar para validación
                self.entry_z = z

                log.info(
                    f"📈 LONG SIGNAL | Z:{z:.4f} ≤ -{entry} | HR:{self.position_hedge_ratio:.6f} | Spread:{current_spread:.6f}"
                )

                signal = {
                    "action": "LONG",
                    "strength": min(abs(z) / stop, 1.0),
                    "reason": "spread_undervalued",
                    "z_score": float(z),
                    "hedge_ratio": float(self.position_hedge_ratio),
                    "spread": float(current_spread),
                }

            elif z >= entry:
                # SHORT: Spread sobrevalorado
                self.current_position = "SHORT"
                self.position_hedge_ratio = self.current_hedge_ratio
                self.entry_spread = current_spread  # Guardar para validación
                self.entry_z = z

                log.info(
                    f"📉 SHORT SIGNAL | Z:{z:.4f} ≥ +{entry} | HR:{self.position_hedge_ratio:.6f} | Spread:{current_spread:.6f}"
                )

                signal = {
                    "action": "SHORT",
                    "strength": min(abs(z) / stop, 1.0),
                    "reason": "spread_overvalued",
                    "z_score": float(z),
                    "hedge_ratio": float(self.position_hedge_ratio),
                    "spread": float(current_spread),
                }
        else:
            # === SALIDA ===
            # Validación de coherencia
            if hasattr(self, "entry_spread"):
                spread_change = current_spread - self.entry_spread
                z_change = z - self.entry_z if hasattr(self, "entry_z") else 0

                # Para debugging
                if abs(spread_change) > 0.1:  # Si el spread se movió significativamente
                    expected_z_direction = np.sign(spread_change)
                    actual_z_direction = np.sign(z_change)

            if self.current_position == "LONG":
                # LONG: entrada en Z≤-entry, salida en Z≥exit_threshold
                if z >= exit_thr:
                    log.info(
                        f"🔄 LONG EXIT | Z:{z:.4f} ≥ {exit_thr} | Spread:{current_spread:.6f} (from "
                        f"{(self.entry_spread if hasattr(self, 'entry_spread') else 0):.6f})"
                    )
                    self.current_position = None
                    self.position_hedge_ratio = None
                    self.entry_spread = None if hasattr(self, "entry_spread") else None
                    self.entry_z = None if hasattr(self, "entry_z") else None
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
                    self.entry_spread = None if hasattr(self, "entry_spread") else None
                    self.entry_z = None if hasattr(self, "entry_z") else None
                    signal = {
                        "action": "CLOSE",
                        "position_id": "current",
                        "reason": "stop_loss",
                        "z_score": float(z),
                        "spread": float(current_spread),
                    }

            elif self.current_position == "SHORT":
                # SHORT: entrada en Z≥entry, salida en Z≤-exit_threshold
                if z <= -exit_thr:
                    log.info(
                        f"🔄 SHORT EXIT | Z:{z:.4f} ≤ {-exit_thr} | Spread:{current_spread:.6f} (from "
                        f"{(self.entry_spread if hasattr(self, 'entry_spread') else 0):.6f})"
                    )
                    self.current_position = None
                    self.position_hedge_ratio = None
                    self.entry_spread = None if hasattr(self, "entry_spread") else None
                    self.entry_z = None if hasattr(self, "entry_z") else None
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
                    self.entry_spread = None if hasattr(self, "entry_spread") else None
                    self.entry_z = None if hasattr(self, "entry_z") else None
                    signal = {
                        "action": "CLOSE",
                        "position_id": "current",
                        "reason": "stop_loss",
                        "z_score": float(z),
                        "spread": float(current_spread),
                    }

        return signal or {"action": "HOLD"}

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Métricas de la estrategia"""
        df = self.get_signals_history()
        if df.empty:
            return {}

        recent_z = (
            [x["zscore"] for x in self.z_score_history[-100:]]
            if self.z_score_history
            else []
        )
        recent_hr = (
            [x["hedge_ratio"] for x in self.hedge_ratio_history[-10:]]
            if self.hedge_ratio_history
            else []
        )

        return {
            "avg_z_score": float(df["z_score"].mean()) if "z_score" in df else 0,
            "max_z_score": float(df["z_score"].max()) if "z_score" in df else 0,
            "min_z_score": float(df["z_score"].min()) if "z_score" in df else 0,
            "z_score_std": float(df["z_score"].std()) if "z_score" in df else 0,
            "current_hedge_ratio": float(self.current_hedge_ratio),
            "position_hedge_ratio": (
                float(self.position_hedge_ratio) if self.position_hedge_ratio else None
            ),
            "hedge_ratio_mean": float(np.mean(recent_hr)) if recent_hr else 1.0,
            "hedge_ratio_std": float(np.std(recent_hr)) if recent_hr else 0.0,
            "signals_generated": len(df),
            "extreme_z_scores": (
                int((df["z_score"].abs() > 3).sum())
                if "z_score" in df and len(df) > 0
                else 0
            ),
            "resample_minutes": self.resample_minutes,
        }
