#!/usr/bin/env python3
"""
Portfolio Manager - CORREGIDO: Mark-to-market continuo para equity curve suave
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
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
        
    def open_position(self, timestamp: datetime, direction: str, size1: float, 
                     size2: float, price1: float, price2: float, cost: float,
                     hedge_ratio: float = 1.0) -> Position:
        """Abrir posición con tracking de hedge ratio"""
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
            hedge_ratio=hedge_ratio
        )
        
        self.positions[position_id] = position
        self.cash -= cost
        
        self.transaction_history.append({
            'timestamp': timestamp,
            'type': 'OPEN',
            'position_id': position_id,
            'direction': direction,
            'cost': cost,
            'cash_after': self.cash
        })
        
        return position
    
    def close_position(self, position_id: str, price1: float, price2: float, cost: float) -> float:
        """CORREGIDO: PnL calculation para pairs trading"""
        
        if position_id not in self.positions:
            log.warning(f"Position {position_id} not found")
            return 0
        
        position = self.positions[position_id]
        
        # CORRECCIÓN: Cálculo correcto del PnL para pairs
        if position.direction == 'LONG':
            # LONG spread: compramos P1, vendimos P2
            # PnL = (P1_exit - P1_entry) * size1 - (P2_exit - P2_entry) * size2
            pnl_symbol1 = position.size1 * (price1 - position.entry_price1)
            pnl_symbol2 = -position.size2 * (price2 - position.entry_price2)
            pnl_gross = pnl_symbol1 + pnl_symbol2
        else:  # SHORT
            # SHORT spread: vendimos P1, compramos P2
            # PnL = -(P1_exit - P1_entry) * size1 + (P2_exit - P2_entry) * size2
            pnl_symbol1 = -position.size1 * (price1 - position.entry_price1)
            pnl_symbol2 = position.size2 * (price2 - position.entry_price2)
            pnl_gross = pnl_symbol1 + pnl_symbol2
        
        # Restar comisiones
        pnl_net = pnl_gross - cost
        
        # Actualizar cash
        # El cash aumenta por el PnL neto (puede ser negativo)
        self.cash += position.entry_cost + pnl_net
        
        # Mover a posiciones cerradas
        self.closed_positions.append(position)
        del self.positions[position_id]
        
        self.transaction_history.append({
            'timestamp': datetime.now(),
            'type': 'CLOSE',
            'position_id': position_id,
            'pnl_gross': pnl_gross,
            'pnl_net': pnl_net,
            'cash_after': self.cash
        })
        
        return pnl_net
    
    def update_positions(self, price1: float, price2: float):
        """CORREGIDO: Mark-to-market continuo para equity curve suave"""
        
        for position in self.positions.values():
            # Calcular PnL no realizado
            if position.direction == 'LONG':
                # LONG: compramos P1, vendimos P2
                unrealized_pnl1 = position.size1 * (price1 - position.entry_price1)
                unrealized_pnl2 = -position.size2 * (price2 - position.entry_price2)
                unrealized_pnl = unrealized_pnl1 + unrealized_pnl2
            else:  # SHORT
                # SHORT: vendimos P1, compramos P2
                unrealized_pnl1 = -position.size1 * (price1 - position.entry_price1)
                unrealized_pnl2 = position.size2 * (price2 - position.entry_price2)
                unrealized_pnl = unrealized_pnl1 + unrealized_pnl2
            
            # Valor actual de la posición (para reporting)
            position.current_value = position.size1 * price1 + position.size2 * price2
            position.unrealized_pnl = unrealized_pnl
    
    def get_equity(self) -> float:
        """CORREGIDO: Equity = Cash + Capital invertido + PnL no realizado"""
        # El equity total es:
        # 1. Cash disponible
        # 2. Capital invertido en posiciones (entry_cost)
        # 3. PnL no realizado
        
        capital_invested = sum(p.entry_cost for p in self.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        return self.cash + capital_invested + unrealized_pnl
    
    def get_positions_value(self) -> float:
        """Obtener valor total de posiciones abiertas"""
        return sum(p.current_value for p in self.positions.values())
    
    def get_total_exposure(self) -> float:
        """Calcular exposición total"""
        return sum(abs(p.current_value) for p in self.positions.values())
    
    def get_position_summary(self) -> pd.DataFrame:
        """Resumen de posiciones actuales"""
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
                'hedge_ratio': pos.hedge_ratio
            })
        
        return pd.DataFrame(data)