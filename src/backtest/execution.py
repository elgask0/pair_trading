#!/usr/bin/env python3
"""
Enhanced Execution Simulator - CORREGIDO: Capital allocation para pairs
"""
import pandas as pd
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
    hedge_ratio: float = 1.0  # Añadir para tracking

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
        """Snapshot más cercano dentro de ±tolerancia"""
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
        """Devuelve (slippage_bps, avg_exec_price) usando hasta 5 niveles."""
        levels, prefix = [], ('ask' if side == 'buy' else 'bid')
        for i in range(1, 6):
            pk, sk = f'{prefix}{i}_price', f'{prefix}{i}_size'
            if pk in ob and sk in ob:
                p, s = ob[pk], ob[sk]
                if pd.notna(p) and pd.notna(s) and p > 0 and s > 0:
                    levels.append({'price': float(p), 'size': float(s), 'value_usd': float(p) * float(s)})

        if not levels:
            return self.fallback_slippage_bps, 0.0

        levels.sort(key=lambda x: x['price'], reverse=(side == 'sell'))
        remaining = notional_usd
        total_val = total_qty = 0.0

        for lvl in levels:
            if remaining <= 0: break
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
        """CORREGIDO: Ejecución correcta de pairs trading con hedge ratio"""
        
        hedge_ratio = signal.get('hedge_ratio', 1.0)
        
        # CORRECCIÓN: Para un spread con hedge ratio β
        # Spread = P1 - β*P2
        # Si compramos el spread (LONG):
        #   - Compramos 1 unidad de P1 → costo = P1
        #   - Vendemos β unidades de P2 → ingreso = β*P2
        # 
        # Para asignar capital total position_value:
        # Necesitamos Size1 unidades de P1 y Size2 = β*Size1 unidades de P2
        # Capital en P1: V1 = Size1 * P1
        # Capital en P2: V2 = Size2 * P2 = β * Size1 * P2
        # 
        # Para simplicidad, asignamos proporcionalmente al valor nocional:
        # Total nocional = |V1| + |V2| = Size1 * P1 + β * Size1 * P2 = Size1 * (P1 + β*P2)
        # Size1 = position_value / (P1 + β*P2)
        # Size2 = β * Size1
        
        # Calcular tamaños target
        denominator = price1 + hedge_ratio * price2
        if denominator <= 0:
            log.error(f"Invalid denominator in position sizing: {denominator}")
            return {'success': False, 'error': 'Invalid prices'}
        
        size1_target = position_value / denominator
        size2_target = hedge_ratio * size1_target
        
        # Valores nocionales
        v1 = size1_target * price1
        v2 = size2_target * price2
        
        # Determinar direcciones de trading
        if signal['action'] == 'LONG':
            # LONG spread = comprar symbol1, vender symbol2
            side1, side2 = 'buy', 'sell'
        else:
            # SHORT spread = vender symbol1, comprar symbol2
            side1, side2 = 'sell', 'buy'

        # Calcular slippage y precios de ejecución
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
                s1_bps, px1 = self._calculate_market_impact(ob1, v1, side1)
                if px1 == 0:
                    px1 = price1 * (1 + (s1_bps/10000 if side1 == 'buy' else -s1_bps/10000))
            else:
                s1_bps = self.fallback_slippage_bps
                px1 = price1 * (1 + (s1_bps/10000 if side1 == 'buy' else -s1_bps/10000))

            ob2 = self._get_orderbook_snapshot(symbol2, timestamp)
            if ob2 is not None:
                s2_bps, px2 = self._calculate_market_impact(ob2, v2, side2)
                if px2 == 0:
                    px2 = price2 * (1 + (s2_bps/10000 if side2 == 'buy' else -s2_bps/10000))
            else:
                s2_bps = self.fallback_slippage_bps
                px2 = price2 * (1 + (s2_bps/10000 if side2 == 'buy' else -s2_bps/10000))

        # Recalcular tamaños con precios de ejecución
        size1 = v1 / px1
        size2 = v2 / px2

        # Verificar liquidez
        max_part = self.max_participation_rate
        max1, max2 = volume1 * max_part, volume2 * max_part
        liq_ok = True
        
        if size1 > max1 or size2 > max2:
            scale = min(max1/size1 if size1>0 else 1.0, max2/size2 if size2>0 else 1.0)
            size1 *= scale
            size2 *= scale
            v1 *= scale
            v2 *= scale
            position_value = v1 + v2
            liq_ok = False
            log.warning(f"⚠️ LIQUIDITY CONSTRAINT | Position scaled to {scale:.1%}")

        # Calcular costos
        slip_cost = (v1 * s1_bps/10000) + (v2 * s2_bps/10000)
        comm = position_value * (self.commission_bps/10000)
        total_cost = position_value + slip_cost + comm

        return {
            'success': True,
            'exec_price1': px1,
            'exec_price2': px2,
            'size1': size1,
            'size2': size2,
            'slippage1_bps': s1_bps,
            'slippage2_bps': s2_bps,
            'avg_slippage_bps': (s1_bps + s2_bps) / 2,
            'slippage_cost': slip_cost,
            'commission': comm,
            'total_cost': total_cost,
            'liquidity_ok': liq_ok,
            'orderbook_used': self.use_orderbook and (
                self._get_orderbook_snapshot(symbol1, timestamp) is not None or 
                self._get_orderbook_snapshot(symbol2, timestamp) is not None
            ),
            'side1': side1,
            'side2': side2,
            'v1': v1,
            'v2': v2,
            'hedge_ratio': hedge_ratio
        }

    def simulate_close(self, position: Position, symbol1: str, symbol2: str,
                       price1: float, price2: float, timestamp: datetime) -> Dict:
        """CORREGIDO: Cierre correcto de pairs trading"""
        
        # Para cerrar, invertimos las direcciones de la apertura
        if position.direction == 'LONG':
            # Originalmente: compramos symbol1, vendimos symbol2
            # Para cerrar: vendemos symbol1, compramos symbol2
            side1, side2 = 'sell', 'buy'
        else:
            # Originalmente: vendimos symbol1, compramos symbol2
            # Para cerrar: compramos symbol1, vendemos symbol2
            side1, side2 = 'buy', 'sell'

        v1 = position.size1 * price1
        v2 = position.size2 * price2
        
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
            s1_bps, px1 = self._calculate_market_impact(ob1, v1, side1) if ob1 is not None else (
                self.fallback_slippage_bps, 
                price1*(1 + (self.fallback_slippage_bps/10000 if side1 == 'buy' else -self.fallback_slippage_bps/10000))
            )
            
            ob2 = self._get_orderbook_snapshot(symbol2, timestamp)
            s2_bps, px2 = self._calculate_market_impact(ob2, v2, side2) if ob2 is not None else (
                self.fallback_slippage_bps,
                price2*(1 + (self.fallback_slippage_bps/10000 if side2 == 'buy' else -self.fallback_slippage_bps/10000))
            )

        close_val = position.size1 * px1 + position.size2 * px2
        comm = close_val * (self.commission_bps/10000)
        
        return {
            'exec_price1': px1,
            'exec_price2': px2,
            'close_value': close_val,
            'commission': comm,
            'total_cost': comm,
            'side1': side1,
            'side2': side2
        }