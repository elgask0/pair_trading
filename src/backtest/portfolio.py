#!/usr/bin/env python3
"""
Portfolio Manager - Gestión de portfolio y tracking de posiciones
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
    """Gestiona el portfolio y las posiciones"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.transaction_history = []
        self.equity_history = []
        
    def open_position(self, timestamp: datetime, direction: str, size1: float, 
                     size2: float, price1: float, price2: float, cost: float) -> Position:
        """Abrir nueva posición"""
        
        position_id = str(uuid.uuid4())[:8]
        
        position = Position(
            id=position_id,
            timestamp=timestamp,
            direction=direction,
            size1=size1,
            size2=size2,
            entry_price1=price1,
            entry_price2=price2,
            entry_cost=cost
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
        
        log.debug(f"Opened position {position_id}: {direction} | Cost: ${cost:.2f}")
        
        return position
    
    def close_position(self, position_id: str, price1: float, price2: float, cost: float) -> float:
        """Cerrar posición y calcular PnL"""
        
        if position_id not in self.positions:
            log.warning(f"Position {position_id} not found")
            return 0
        
        position = self.positions[position_id]
        
        # Calcular PnL
        if position.direction == 'LONG':
            pnl = (
                position.size1 * (price1 - position.entry_price1) +
                position.size2 * (price2 - position.entry_price2)
            )
        else:  # SHORT
            pnl = (
                position.size1 * (position.entry_price1 - price1) +
                position.size2 * (position.entry_price2 - price2)
            )
        
        # Restar costos
        pnl -= cost
        
        # Actualizar cash
        close_value = position.size1 * price1 + position.size2 * price2
        self.cash += close_value - cost
        
        # Mover a posiciones cerradas
        self.closed_positions.append(position)
        del self.positions[position_id]
        
        self.transaction_history.append({
            'timestamp': datetime.now(),
            'type': 'CLOSE',
            'position_id': position_id,
            'pnl': pnl,
            'cash_after': self.cash
        })
        
        log.debug(f"Closed position {position_id}: PnL: ${pnl:.2f}")
        
        return pnl
    
    def update_positions(self, price1: float, price2: float):
        """Actualizar valor de posiciones abiertas"""
        
        for position in self.positions.values():
            # Calcular valor actual
            current_value = position.size1 * price1 + position.size2 * price2
            
            # Calcular PnL no realizado
            if position.direction == 'LONG':
                unrealized_pnl = (
                    position.size1 * (price1 - position.entry_price1) +
                    position.size2 * (price2 - position.entry_price2)
                )
            else:  # SHORT
                unrealized_pnl = (
                    position.size1 * (position.entry_price1 - price1) +
                    position.size2 * (position.entry_price2 - price2)
                )
            
            position.current_value = current_value
            position.unrealized_pnl = unrealized_pnl
    
    def get_equity(self) -> float:
        """Calcular equity total (cash + posiciones)"""
        positions_value = sum(p.current_value for p in self.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + unrealized_pnl
    
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
                'unrealized_pnl': pos.unrealized_pnl
            })
        
        return pd.DataFrame(data)