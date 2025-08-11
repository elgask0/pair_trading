#!/usr/bin/env python3
"""
Enhanced Execution Simulator - Usa datos reales de orderbook para slippage dinámico
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
    """Representa una posición abierta (INTERFAZ MÍNIMA para cerrar)"""
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


class EnhancedExecutionSimulator:
    """
    Simulador de ejecución que usa datos reales de orderbook.
    """

    def __init__(self,
                 commission_bps: float = 2.0,
                 use_orderbook: bool = True,
                 fallback_slippage_bps: float = 5.0,
                 max_participation_rate: float = 0.1):
        self.commission_bps = commission_bps
        self.use_orderbook = use_orderbook
        self.fallback_slippage_bps = fallback_slippage_bps
        self.max_participation_rate = max_participation_rate
        self.orderbook_cache = {}

    def _get_orderbook_snapshot(self, symbol: str, timestamp: datetime) -> Optional[pd.Series]:
        """Obtener snapshot de orderbook más cercano al timestamp (±30s)"""
        cache_key = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M')}"
        if cache_key in self.orderbook_cache:
            return self.orderbook_cache[cache_key]

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
                      AND timestamp >= :time_start
                      AND timestamp <= :time_end
                      AND bid1_price IS NOT NULL
                      AND ask1_price IS NOT NULL
                    ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - :target_time)))
                    LIMIT 1
                """)

                row = session.execute(query, {
                    'symbol': symbol,
                    'time_start': timestamp - timedelta(seconds=30),
                    'time_end': timestamp + timedelta(seconds=30),
                    'target_time': timestamp
                }).fetchone()

                if row:
                    # Compatibilidad SQLAlchemy 1.4/2.0
                    mapping = getattr(row, "_mapping", None)
                    data = dict(mapping) if mapping is not None else row._asdict()
                    snapshot = pd.Series(data)
                    self.orderbook_cache[cache_key] = snapshot
                    return snapshot

        except Exception as e:
            log.warning(f"Failed to get orderbook snapshot for {symbol}: {e}")

        return None

    def _calculate_market_impact(self,
                                 orderbook: pd.Series,
                                 notional_usd: float,
                                 side: str) -> Tuple[float, float]:
        """
        Calcular slippage real usando orderbook (top 5 niveles).
        Devuelve (slippage_bps, avg_exec_price)
        """
        levels = []
        prefix = 'ask' if side == 'buy' else 'bid'

        # Extraer niveles del orderbook
        for i in range(1, 5 + 1):
            price_key = f'{prefix}{i}_price'
            size_key = f'{prefix}{i}_size'
            if price_key in orderbook and size_key in orderbook:
                price = orderbook[price_key]
                size = orderbook[size_key]
                if pd.notna(price) and pd.notna(size) and price > 0 and size > 0:
                    levels.append({
                        'price': float(price),
                        'size': float(size),
                        'value_usd': float(price) * float(size)
                    })

        if not levels:
            # No hay datos de orderbook: usar slippage fijo sobre el precio 'mejor' desconocido
            return self.fallback_slippage_bps, 0.0

        # Ordenar por mejor precio
        levels.sort(key=lambda x: x['price'], reverse=(side == 'sell'))

        # Calcular ejecución atravesando el orderbook
        remaining_usd = notional_usd
        total_value = 0.0
        total_coins = 0.0

        for level in levels:
            if remaining_usd <= 0:
                break
            available_usd = level['value_usd']
            if available_usd >= remaining_usd:
                coins = remaining_usd / level['price']
                total_value += remaining_usd
                total_coins += coins
                remaining_usd = 0.0
            else:
                total_value += available_usd
                total_coins += level['size']
                remaining_usd -= available_usd

        best_price = levels[0]['price']

        if total_coins <= 0:
            # Sin llenado; aplica fallback y usa mejor precio como referencia
            return self.fallback_slippage_bps, best_price

        if remaining_usd > 0:
            # Falta de liquidez -> penalizamos con slippage adicional proporcional
            extra_slippage = (remaining_usd / notional_usd) * 100 * 100  # bps
            return self.fallback_slippage_bps + extra_slippage, best_price

        avg_price = total_value / total_coins
        slippage_pct = abs(avg_price - best_price) / best_price * 100
        slippage_bps = slippage_pct * 100
        return slippage_bps, avg_price

    def simulate_execution(self, signal: Dict, symbol1: str, symbol2: str,
                           price1: float, price2: float,
                           volume1: float, volume2: float,
                           position_value: float,
                           timestamp: datetime) -> Dict:
        """
        Simular ejecución con slippage dinámico basado en orderbook real
        """
        hedge_ratio = signal.get('hedge_ratio', 1.0)

        # Dividir capital entre las dos piernas
        value_leg1 = position_value / (1 + hedge_ratio)
        value_leg2 = position_value - value_leg1

        # Obtener slippage real del orderbook si está disponible
        if self.use_orderbook:
            # Symbol 1
            orderbook1 = self._get_orderbook_snapshot(symbol1, timestamp)
            if orderbook1 is not None:
                side1 = 'buy' if signal['action'] == 'LONG' else 'sell'
                slippage1_bps, exec_price1 = self._calculate_market_impact(orderbook1, value_leg1, side1)
                if exec_price1 == 0:
                    exec_price1 = price1 * (1 + (slippage1_bps / 10000 if side1 == 'buy' else -slippage1_bps / 10000))
            else:
                slippage1_bps = self.fallback_slippage_bps
                exec_price1 = price1 * (1 + (slippage1_bps / 10000 if signal['action'] == 'LONG' else -slippage1_bps / 10000))

            # Symbol 2 (pierna opuesta)
            orderbook2 = self._get_orderbook_snapshot(symbol2, timestamp)
            if orderbook2 is not None:
                side2 = 'sell' if signal['action'] == 'LONG' else 'buy'
                slippage2_bps, exec_price2 = self._calculate_market_impact(orderbook2, value_leg2, side2)
                if exec_price2 == 0:
                    exec_price2 = price2 * (1 + (-slippage2_bps / 10000 if side2 == 'sell' else slippage2_bps / 10000))
            else:
                slippage2_bps = self.fallback_slippage_bps
                if signal['action'] == 'LONG':
                    exec_price2 = price2 * (1 - slippage2_bps / 10000)
                else:
                    exec_price2 = price2 * (1 + slippage2_bps / 10000)
        else:
            # Usar slippage fijo si no hay orderbook
            slippage1_bps = self.fallback_slippage_bps
            slippage2_bps = self.fallback_slippage_bps
            if signal['action'] == 'LONG':
                exec_price1 = price1 * (1 + slippage1_bps / 10000)
                exec_price2 = price2 * (1 - slippage2_bps / 10000)
            else:
                exec_price1 = price1 * (1 - slippage1_bps / 10000)
                exec_price2 = price2 * (1 + slippage2_bps / 10000)

        # Calcular tamaños
        size1 = value_leg1 / exec_price1
        size2 = value_leg2 / exec_price2

        # Verificar liquidez contra volumen
        max_participation = self.max_participation_rate
        max_size1 = volume1 * max_participation
        max_size2 = volume2 * max_participation

        liquidity_ok = True
        if size1 > max_size1 or size2 > max_size2:
            scale_factor = min(max_size1 / size1 if size1 > 0 else 1.0,
                               max_size2 / size2 if size2 > 0 else 1.0)
            size1 *= scale_factor
            size2 *= scale_factor
            value_leg1 *= scale_factor
            value_leg2 *= scale_factor
            position_value = value_leg1 + value_leg2
            liquidity_ok = False
            log.warning(f"Position scaled down to {scale_factor:.1%} due to volume constraints")

        # Calcular costos
        slippage_cost = (value_leg1 * slippage1_bps / 10000) + (value_leg2 * slippage2_bps / 10000)
        commission = position_value * (self.commission_bps / 10000) * 2
        total_cost = position_value + commission

        result = {
            'success': True,
            'exec_price1': exec_price1,
            'exec_price2': exec_price2,
            'size1': size1,
            'size2': size2,
            'slippage1_bps': slippage1_bps,
            'slippage2_bps': slippage2_bps,
            'avg_slippage_bps': (slippage1_bps + slippage2_bps) / 2,
            'slippage_cost': slippage_cost,
            'commission': commission,
            'total_cost': total_cost,
            'liquidity_ok': liquidity_ok,
            'orderbook_used': self.use_orderbook and (orderbook1 is not None or orderbook2 is not None)
        }

        log.debug(f"Execution: Slippage={result['avg_slippage_bps']:.1f}bps, "
                  f"Orderbook={'YES' if result['orderbook_used'] else 'NO'}")

        return result

    def simulate_close(self, position: Position, symbol1: str, symbol2: str,
                       price1: float, price2: float, timestamp: datetime) -> Dict:
        """Simular cierre con orderbook real"""
        # Para cerrar, invertimos la dirección
        if position.direction == 'LONG':
            # Cerrando long = vender asset1, comprar asset2
            value1 = position.size1 * price1
            value2 = position.size2 * price2

            if self.use_orderbook:
                orderbook1 = self._get_orderbook_snapshot(symbol1, timestamp)
                if orderbook1 is not None:
                    slippage1_bps, exec_price1 = self._calculate_market_impact(orderbook1, value1, 'sell')
                else:
                    slippage1_bps = self.fallback_slippage_bps
                    exec_price1 = price1 * (1 - slippage1_bps / 10000)

                orderbook2 = self._get_orderbook_snapshot(symbol2, timestamp)
                if orderbook2 is not None:
                    slippage2_bps, exec_price2 = self._calculate_market_impact(orderbook2, value2, 'buy')
                else:
                    slippage2_bps = self.fallback_slippage_bps
                    exec_price2 = price2 * (1 + slippage2_bps / 10000)
            else:
                exec_price1 = price1 * (1 - self.fallback_slippage_bps / 10000)
                exec_price2 = price2 * (1 + self.fallback_slippage_bps / 10000)

        else:
            # Cerrando short = comprar asset1, vender asset2
            value1 = position.size1 * price1
            value2 = position.size2 * price2

            if self.use_orderbook:
                orderbook1 = self._get_orderbook_snapshot(symbol1, timestamp)
                if orderbook1 is not None:
                    slippage1_bps, exec_price1 = self._calculate_market_impact(orderbook1, value1, 'buy')
                else:
                    slippage1_bps = self.fallback_slippage_bps
                    exec_price1 = price1 * (1 + slippage1_bps / 10000)

                orderbook2 = self._get_orderbook_snapshot(symbol2, timestamp)
                if orderbook2 is not None:
                    slippage2_bps, exec_price2 = self._calculate_market_impact(orderbook2, value2, 'sell')
                else:
                    slippage2_bps = self.fallback_slippage_bps
                    exec_price2 = price2 * (1 - slippage2_bps / 10000)
            else:
                exec_price1 = price1 * (1 + self.fallback_slippage_bps / 10000)
                exec_price2 = price2 * (1 - self.fallback_slippage_bps / 10000)

        close_value = position.size1 * exec_price1 + position.size2 * exec_price2
        commission = close_value * (self.commission_bps / 10000)

        return {
            'exec_price1': exec_price1,
            'exec_price2': exec_price2,
            'close_value': close_value,
            'commission': commission,
            'total_cost': commission
        }