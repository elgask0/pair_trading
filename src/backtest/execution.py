#!/usr/bin/env python3
"""
Enhanced Execution Simulator - VERSIÓN CORREGIDA
Ratio-weighted pairs trading con hedge ratio correcto
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from sqlalchemy import text

from src.database.connection import db_manager
from src.utils.logger import get_logger

log = get_logger()

@dataclass
class Position:
    id: str
    timestamp: any
    direction: str
    size1: float
    size2: float
    entry_price1: float
    entry_price2: float
    entry_cost: float
    current_value: float = 0
    unrealized_pnl: float = 0
    hedge_ratio: float = 1.0
    value1: float = 0
    value2: float = 0
    entry_spread_log: float = 0
    

class EnhancedExecutionSimulator:
    def __init__(self,
                 commission_bps: float = 2.0,
                 use_orderbook: bool = True,
                 fallback_slippage_bps: float = 5.0,
                 max_participation_rate: float = 0.1,
                 orderbook_time_tolerance_seconds: int = 300):
        self.commission_bps = commission_bps
        self.use_orderbook = use_orderbook
        self.fallback_slippage_bps = fallback_slippage_bps
        self.max_participation_rate = max_participation_rate
        self.orderbook_time_tolerance_seconds = int(orderbook_time_tolerance_seconds)
        self.orderbook_cache = {}

    def _get_orderbook_snapshot(self, symbol: str, timestamp: datetime) -> Optional[pd.Series]:
        """Obtener snapshot del orderbook más cercano dentro de tolerancia"""
        key = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M')}"
        if key in self.orderbook_cache:
            return self.orderbook_cache[key]

        tol = timedelta(seconds=self.orderbook_time_tolerance_seconds)
        try:
            with db_manager.get_session() as session:
                query = text("""
                    SELECT 
                        bid1_price, bid1_size, bid2_price, bid2_size, 
                        bid3_price, bid3_size, bid4_price, bid4_size,
                        bid5_price, bid5_size,
                        ask1_price, ask1_size, ask2_price, ask2_size,
                        ask3_price, ask3_size, ask4_price, ask4_size,
                        ask5_price, ask5_size
                    FROM orderbook
                    WHERE symbol = :symbol
                      AND timestamp >= :t0
                      AND timestamp <= :t1
                      AND bid1_price IS NOT NULL
                      AND ask1_price IS NOT NULL
                    ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - :tref)))
                    LIMIT 1
                """)
                row = session.execute(query, {
                    'symbol': symbol,
                    't0': timestamp - tol,
                    't1': timestamp + tol,
                    'tref': timestamp
                }).fetchone()

                if row:
                    mapping = getattr(row, "_mapping", None)
                    data = dict(mapping) if mapping is not None else row._asdict()
                    snap = pd.Series(data)
                    self.orderbook_cache[key] = snap
                    return snap
        except Exception as e:
            log.warning(f"Failed to get orderbook snapshot for {symbol}: {e}")
        return None

    def _calculate_market_impact(self,
                                 ob: pd.Series,
                                 notional_usd: float,
                                 side: str) -> Tuple[float, float]:
        """Calcular impacto en el mercado usando hasta 5 niveles del orderbook"""
        levels, prefix = [], ('ask' if side == 'buy' else 'bid')
        
        for i in range(1, 6):
            pk, sk = f'{prefix}{i}_price', f'{prefix}{i}_size'
            if pk in ob and sk in ob:
                p, s = ob[pk], ob[sk]
                if pd.notna(p) and pd.notna(s) and p > 0 and s > 0:
                    levels.append({
                        'price': float(p), 
                        'size': float(s), 
                        'value_usd': float(p) * float(s)
                    })

        if not levels:
            return self.fallback_slippage_bps, 0.0

        levels.sort(key=lambda x: x['price'], reverse=(side == 'sell'))
        remaining = notional_usd
        total_val = total_qty = 0.0

        for lvl in levels:
            if remaining <= 0: 
                break
            if lvl['value_usd'] >= remaining:
                qty = remaining / lvl['price']
                total_val += remaining
                total_qty += qty
                remaining = 0.0
            else:
                total_val += lvl['value_usd']
                total_qty += lvl['size']
                remaining -= lvl['value_usd']

        best = levels[0]['price']
        if total_qty <= 0:
            return self.fallback_slippage_bps, best

        if remaining > 0:
            extra = (remaining / notional_usd) * 100 * 100
            return self.fallback_slippage_bps + extra, best

        avg = total_val / total_qty
        slip_pct = abs(avg - best) / best * 100
        return slip_pct * 100, avg

    def simulate_execution(self, signal: Dict, symbol1: str, symbol2: str,
                           price1: float, price2: float,
                           volume1: float, volume2: float,
                           position_value: float, timestamp: datetime) -> Dict:
        """
        VERSIÓN CORREGIDA: Ratio-weighted pairs trading execution
        
        Para spread = log(P1) - β*log(P2):
        - LONG spread: Comprar Asset1, Vender β-weighted Asset2
        - SHORT spread: Vender Asset1, Comprar β-weighted Asset2
        """
        
        hedge_ratio = signal.get('hedge_ratio', 1.0)
        
        # ====== RATIO-WEIGHTED POSITION SIZING CORRECTO ======
        # 
        # Para un spread = log(P1) - β*log(P2)
        # Si queremos invertir position_value total:
        #
        # Método: Asignar capital proporcional al hedge ratio
        # value1 + value2 = position_value
        # value2 = β * value1 (para mantener el ratio)
        # 
        # Resolviendo:
        # value1 * (1 + β) = position_value
        # value1 = position_value / (1 + β)
        # value2 = β * value1
        
        # Valores nominales objetivo
        value1_target = position_value / (1 + abs(hedge_ratio))
        value2_target = abs(hedge_ratio) * value1_target
        
        # Verificación: value1_target + value2_target ≈ position_value
        
        # Sizes objetivo (antes de slippage)
        size1_target = value1_target / price1
        size2_target = value2_target / price2
        
        # Direcciones de trading
        if signal['action'] == 'LONG':
            # LONG spread = Comprar Asset1, Vender Asset2
            side1, side2 = 'buy', 'sell'
        else:  # SHORT
            # SHORT spread = Vender Asset1, Comprar Asset2
            side1, side2 = 'sell', 'buy'
        
        # ====== CÁLCULO DE SLIPPAGE Y PRECIOS DE EJECUCIÓN ======
        
        if not self.use_orderbook:
            # Slippage fijo
            s1_bps = s2_bps = self.fallback_slippage_bps
            
            if side1 == 'buy':
                px1 = price1 * (1 + s1_bps/10000)
            else:
                px1 = price1 * (1 - s1_bps/10000)
                
            if side2 == 'buy':
                px2 = price2 * (1 + s2_bps/10000)
            else:
                px2 = price2 * (1 - s2_bps/10000)
        else:
            # Slippage basado en orderbook
            ob1 = self._get_orderbook_snapshot(symbol1, timestamp)
            if ob1 is not None:
                s1_bps, px1 = self._calculate_market_impact(ob1, value1_target, side1)
                if px1 == 0:
                    px1 = price1 * (1 + (s1_bps/10000 if side1 == 'buy' else -s1_bps/10000))
            else:
                s1_bps = self.fallback_slippage_bps
                px1 = price1 * (1 + (s1_bps/10000 if side1 == 'buy' else -s1_bps/10000))

            ob2 = self._get_orderbook_snapshot(symbol2, timestamp)
            if ob2 is not None:
                s2_bps, px2 = self._calculate_market_impact(ob2, value2_target, side2)
                if px2 == 0:
                    px2 = price2 * (1 + (s2_bps/10000 if side2 == 'buy' else -s2_bps/10000))
            else:
                s2_bps = self.fallback_slippage_bps
                px2 = price2 * (1 + (s2_bps/10000 if side2 == 'buy' else -s2_bps/10000))
        
        # ====== AJUSTE POR LIQUIDEZ ======
        
        # Recalcular sizes con precios de ejecución
        size1_final = value1_target / px1
        size2_final = value2_target / px2
        
        # Verificar límites de participación
        max_part = self.max_participation_rate
        max_size1 = volume1 * max_part
        max_size2 = volume2 * max_part
        
        liquidity_constrained = False
        scaling_factor = 1.0
        
        if size1_final > max_size1 or size2_final > max_size2:
            # Escalar ambos lados proporcionalmente para mantener el ratio
            scale1 = max_size1 / size1_final if size1_final > 0 else 1.0
            scale2 = max_size2 / size2_final if size2_final > 0 else 1.0
            scaling_factor = min(scale1, scale2)
            
            size1_final *= scaling_factor
            size2_final *= scaling_factor
            value1_target *= scaling_factor
            value2_target *= scaling_factor
            
            liquidity_constrained = True
            log.warning(f"⚠️ LIQUIDITY CONSTRAINT | Position scaled to {scaling_factor:.1%}")
        
        # ====== CÁLCULO DE COSTOS ======
        
        # Valores finales ejecutados
        exec_value1 = size1_final * px1
        exec_value2 = size2_final * px2
        total_exec_value = exec_value1 + exec_value2
        
        # Costos de transacción
        commission = total_exec_value * (self.commission_bps / 10000)
        total_cost = total_exec_value + commission
        
        # ====== MÉTRICAS ADICIONALES ======
        
        # Calcular spread en log-space para validación
        entry_spread_log = np.log(px1) - hedge_ratio * np.log(px2)
        
        # Verificar que el ratio se mantiene
        actual_value_ratio = exec_value2 / exec_value1 if exec_value1 > 0 else 0
        target_value_ratio = abs(hedge_ratio)
        ratio_deviation = abs(actual_value_ratio - target_value_ratio) / target_value_ratio if target_value_ratio > 0 else 0
        
        if ratio_deviation > 0.1:  # Warning si desviación > 10%
            log.warning(f"⚠️ RATIO DEVIATION | Target: {target_value_ratio:.4f} | Actual: {actual_value_ratio:.4f}")
        
        return {
            'success': True,
            'exec_price1': px1,
            'exec_price2': px2,
            'size1': size1_final,
            'size2': size2_final,
            'value1': exec_value1,
            'value2': exec_value2,
            'slippage1_bps': s1_bps,
            'slippage2_bps': s2_bps,
            'avg_slippage_bps': (s1_bps * exec_value1 + s2_bps * exec_value2) / total_exec_value,
            'commission': commission,
            'total_cost': total_cost,
            'liquidity_ok': not liquidity_constrained,
            'scaling_factor': scaling_factor,
            'orderbook_used': self.use_orderbook and (ob1 is not None or ob2 is not None),
            'side1': side1,
            'side2': side2,
            'hedge_ratio': hedge_ratio,
            'entry_spread_log': entry_spread_log,
            'value_ratio_actual': actual_value_ratio,
            'value_ratio_target': target_value_ratio,
            'ratio_deviation': ratio_deviation
        }

    def simulate_close(self, position: Position, symbol1: str, symbol2: str,
                       price1: float, price2: float, timestamp: datetime) -> Dict:
        """
        Cierre de posición pairs trading - mantener consistencia con apertura
        """
        
        # Para cerrar, invertimos las operaciones originales
        if position.direction == 'LONG':
            # Original: compramos Asset1, vendimos Asset2
            # Cierre: vendemos Asset1, compramos Asset2
            side1, side2 = 'sell', 'buy'
        else:  # SHORT
            # Original: vendimos Asset1, compramos Asset2
            # Cierre: compramos Asset1, vendemos Asset2
            side1, side2 = 'buy', 'sell'
        
        # Valores a ejecutar (basados en sizes actuales)
        value1 = position.size1 * price1
        value2 = position.size2 * price2
        
        # Calcular slippage
        if not self.use_orderbook:
            s1_bps = s2_bps = self.fallback_slippage_bps
            
            if side1 == 'buy':
                px1 = price1 * (1 + s1_bps/10000)
            else:
                px1 = price1 * (1 - s1_bps/10000)
                
            if side2 == 'buy':
                px2 = price2 * (1 + s2_bps/10000)
            else:
                px2 = price2 * (1 - s2_bps/10000)
        else:
            ob1 = self._get_orderbook_snapshot(symbol1, timestamp)
            if ob1 is not None:
                s1_bps, px1 = self._calculate_market_impact(ob1, value1, side1)
                if px1 == 0:
                    px1 = price1 * (1 + (s1_bps/10000 if side1 == 'buy' else -s1_bps/10000))
            else:
                s1_bps = self.fallback_slippage_bps
                px1 = price1 * (1 + (s1_bps/10000 if side1 == 'buy' else -s1_bps/10000))
            
            ob2 = self._get_orderbook_snapshot(symbol2, timestamp)
            if ob2 is not None:
                s2_bps, px2 = self._calculate_market_impact(ob2, value2, side2)
                if px2 == 0:
                    px2 = price2 * (1 + (s2_bps/10000 if side2 == 'buy' else -s2_bps/10000))
            else:
                s2_bps = self.fallback_slippage_bps
                px2 = price2 * (1 + (s2_bps/10000 if side2 == 'buy' else -s2_bps/10000))
        
        # Valores de cierre
        close_value1 = position.size1 * px1
        close_value2 = position.size2 * px2
        total_close_value = close_value1 + close_value2
        
        # Comisiones
        commission = total_close_value * (self.commission_bps / 10000)
        
        # Spread al cerrar
        exit_spread_log = np.log(px1) - position.hedge_ratio * np.log(px2)
        
        return {
            'exec_price1': px1,
            'exec_price2': px2,
            'close_value1': close_value1,
            'close_value2': close_value2,
            'close_value': total_close_value,
            'commission': commission,
            'total_cost': commission,
            'side1': side1,
            'side2': side2,
            'slippage1_bps': s1_bps,
            'slippage2_bps': s2_bps,
            'exit_spread_log': exit_spread_log
        }