#!/usr/bin/env python3
"""
Portfolio Manager - VERSIÓN VERIFICADA
PnL calculation correcto para pairs trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

from src.backtest.execution import Position
from src.utils.logger import get_logger

log = get_logger()

class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.transaction_history = []
        self.equity_history = []
        
        # Métricas adicionales
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
    def open_position(self, timestamp: datetime, direction: str, size1: float, 
                     size2: float, price1: float, price2: float, cost: float,
                     hedge_ratio: float = 1.0, **kwargs) -> Position:
        """
        Abrir posición con tracking mejorado
        """
        position_id = str(uuid.uuid4())[:8]
        
        position = Position(
            id=position_id,
            timestamp=timestamp,
            direction=direction,
            size1=size1,
            size2=size2,
            entry_price1=price1,
            entry_price2=price2,
            entry_cost=cost,
            hedge_ratio=hedge_ratio,
            value1=kwargs.get('value1', size1 * price1),
            value2=kwargs.get('value2', size2 * price2),
            entry_spread_log=kwargs.get('entry_spread_log', 
                                       np.log(price1) - hedge_ratio * np.log(price2))
        )
        
        self.positions[position_id] = position
        self.cash -= cost
        self.total_trades += 1
        
        self.transaction_history.append({
            'timestamp': timestamp,
            'type': 'OPEN',
            'position_id': position_id,
            'direction': direction,
            'size1': size1,
            'size2': size2,
            'price1': price1,
            'price2': price2,
            'hedge_ratio': hedge_ratio,
            'cost': cost,
            'cash_after': self.cash,
            'spread_log': position.entry_spread_log
        })
        
        log.debug(f"Position opened | ID: {position_id} | Direction: {direction} | "
                 f"Spread(log): {position.entry_spread_log:.6f}")
        
        return position
    
    def close_position(self, position_id: str, price1: float, price2: float, 
                      cost: float, **kwargs) -> float:
        """
        Cálculo correcto del PnL para pairs trading
        
        La lógica del PnL es correcta:
        - LONG spread: ganamos si Asset1 sube relativo a Asset2
        - SHORT spread: ganamos si Asset1 baja relativo a Asset2
        """
        
        if position_id not in self.positions:
            log.warning(f"Position {position_id} not found")
            return 0
        
        position = self.positions[position_id]
        
        # ====== CÁLCULO DEL PNL (VERIFICADO CORRECTO) ======
        
        if position.direction == 'LONG':
            # LONG spread = Compramos Asset1, Vendimos Asset2 (short)
            # Asset1: Lo compramos, ganamos si sube
            pnl_asset1 = position.size1 * (price1 - position.entry_price1)
            
            # Asset2: Lo vendimos (short), ganamos si BAJA
            pnl_asset2 = position.size2 * (position.entry_price2 - price2)
            
        else:  # SHORT
            # SHORT spread = Vendimos Asset1 (short), Compramos Asset2
            # Asset1: Lo vendimos (short), ganamos si BAJA
            pnl_asset1 = position.size1 * (position.entry_price1 - price1)
            
            # Asset2: Lo compramos, ganamos si sube
            pnl_asset2 = position.size2 * (price2 - position.entry_price2)
        
        # PnL total antes de costos
        pnl_gross = pnl_asset1 + pnl_asset2
        
        # PnL neto después de costos
        pnl_net = pnl_gross - cost
        
        # ====== VALIDACIÓN DE COHERENCIA ======
        
        # Calcular movimiento del spread
        exit_spread_log = np.log(price1) - position.hedge_ratio * np.log(price2)
        spread_change = exit_spread_log - position.entry_spread_log
        
        # Validar coherencia: el PnL debe ser consistente con el movimiento del spread
        if position.direction == 'LONG':
            # LONG gana si el spread sube
            expected_profitable = spread_change > 0
        else:
            # SHORT gana si el spread baja
            expected_profitable = spread_change < 0
        
        actual_profitable = pnl_gross > 0
        
        # Warning si hay incoherencia significativa
        if expected_profitable != actual_profitable and abs(pnl_gross) > 10:
            log.warning(f"⚠️ PnL COHERENCE CHECK | "
                       f"Direction: {position.direction} | "
                       f"Spread moved {'up' if spread_change > 0 else 'down'} by {spread_change:.6f} | "
                       f"Expected PnL {'positive' if expected_profitable else 'negative'} | "
                       f"Actual PnL: ${pnl_gross:.2f}")
        
        # ====== ACTUALIZACIÓN DEL PORTFOLIO ======
        
        # Recuperar capital inicial + PnL
        self.cash += position.entry_cost + pnl_net
        
        # Actualizar estadísticas
        self.total_pnl += pnl_net
        if pnl_net > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Guardar información detallada
        position.exit_price1 = price1
        position.exit_price2 = price2
        position.exit_spread_log = exit_spread_log
        position.pnl_asset1 = pnl_asset1
        position.pnl_asset2 = pnl_asset2
        position.pnl_gross = pnl_gross
        position.pnl_net = pnl_net
        position.spread_change = spread_change
        
        # Mover a posiciones cerradas
        self.closed_positions.append(position)
        del self.positions[position_id]
        
        # Registrar transacción
        self.transaction_history.append({
            'timestamp': datetime.now(),
            'type': 'CLOSE',
            'position_id': position_id,
            'direction': position.direction,
            'exit_price1': price1,
            'exit_price2': price2,
            'pnl_gross': pnl_gross,
            'pnl_net': pnl_net,
            'pnl_symbol1': pnl_asset1,
            'pnl_symbol2': pnl_asset2,
            'spread_change': spread_change,
            'cash_after': self.cash,
            'coherence_check': expected_profitable == actual_profitable
        })
        
        return pnl_net
    
    def update_positions(self, price1: float, price2: float):
        """
        Mark-to-market de posiciones abiertas
        """
        
        for position in self.positions.values():
            # Calcular PnL no realizado con la misma lógica
            if position.direction == 'LONG':
                # LONG: compramos Asset1, vendimos Asset2
                unrealized_pnl1 = position.size1 * (price1 - position.entry_price1)
                unrealized_pnl2 = position.size2 * (position.entry_price2 - price2)
            else:  # SHORT
                # SHORT: vendimos Asset1, compramos Asset2
                unrealized_pnl1 = position.size1 * (position.entry_price1 - price1)
                unrealized_pnl2 = position.size2 * (price2 - position.entry_price2)
            
            unrealized_pnl_total = unrealized_pnl1 + unrealized_pnl2
            
            # Actualizar valores
            position.current_value = position.size1 * price1 + position.size2 * price2
            position.unrealized_pnl = unrealized_pnl_total
            position.unrealized_pnl1 = unrealized_pnl1
            position.unrealized_pnl2 = unrealized_pnl2
            
            # Calcular spread actual para monitoring
            current_spread_log = np.log(price1) - position.hedge_ratio * np.log(price2)
            position.current_spread_log = current_spread_log
            position.spread_change_unrealized = current_spread_log - position.entry_spread_log
    
    def get_equity(self) -> float:
        """
        Equity total = Cash + Capital invertido + PnL no realizado
        """
        capital_invested = sum(p.entry_cost for p in self.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        equity = self.cash + capital_invested + unrealized_pnl
        
        return equity
    
    def get_positions_value(self) -> float:
        """Valor total de posiciones abiertas"""
        return sum(p.current_value for p in self.positions.values())
    
    def get_total_exposure(self) -> float:
        """Exposición total (suma de valores absolutos)"""
        total = 0
        for p in self.positions.values():
            # Exposición es la suma de los valores absolutos de cada leg
            total += abs(p.size1 * p.entry_price1) + abs(p.size2 * p.entry_price2)
        return total
    
    def get_position_summary(self) -> pd.DataFrame:
        """Resumen detallado de posiciones actuales"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for pos in self.positions.values():
            data.append({
                'id': pos.id,
                'direction': pos.direction,
                'size1': pos.size1,
                'size2': pos.size2,
                'entry_price1': pos.entry_price1,
                'entry_price2': pos.entry_price2,
                'current_value': pos.current_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl1': getattr(pos, 'unrealized_pnl1', 0),
                'unrealized_pnl2': getattr(pos, 'unrealized_pnl2', 0),
                'hedge_ratio': pos.hedge_ratio,
                'entry_spread_log': pos.entry_spread_log,
                'current_spread_log': getattr(pos, 'current_spread_log', 0),
                'spread_change': getattr(pos, 'spread_change_unrealized', 0)
            })
        
        return pd.DataFrame(data)
    
    def get_performance_metrics(self) -> Dict:
        """Métricas de performance del portfolio"""
        if self.total_trades == 0:
            return {}
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calcular métricas de trades cerrados
        if self.closed_positions:
            pnls = [p.pnl_net for p in self.closed_positions if hasattr(p, 'pnl_net')]
            winning_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p < 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0
            
            # Profit factor
            gross_profit = sum(winning_pnls) if winning_pnls else 0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
        else:
            avg_win = avg_loss = profit_factor = expectancy = 0
        
        return {
            'total_trades': self.total_trades,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'current_equity': self.get_equity(),
            'total_return': (self.get_equity() - self.initial_capital) / self.initial_capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'current_cash': self.cash,
            'current_exposure': self.get_total_exposure()
        }
    
    def validate_portfolio_integrity(self) -> Tuple[bool, List[str]]:
        """
        Validación de integridad del portfolio
        """
        issues = []
        
        # Check 1: Cash no debe ser negativo
        if self.cash < 0:
            issues.append(f"Negative cash: ${self.cash:.2f}")
        
        # Check 2: Equity no debe ser negativa
        equity = self.get_equity()
        if equity < 0:
            issues.append(f"Negative equity: ${equity:.2f}")
        
        # Check 3: Todas las posiciones deben tener hedge_ratio válido
        for pos_id, pos in self.positions.items():
            if pos.hedge_ratio <= 0 or pos.hedge_ratio > 10:
                issues.append(f"Position {pos_id} has invalid hedge_ratio: {pos.hedge_ratio}")
        
        is_valid = len(issues) == 0
        return is_valid, issues