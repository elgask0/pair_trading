#!/usr/bin/env python3
"""
Portfolio Manager - CORREGIDO: PnL calculation definitivo
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
        """CORREGIDO DEFINITIVO: PnL calculation para pairs trading"""
        
        if position_id not in self.positions:
            log.warning(f"Position {position_id} not found")
            return 0
        
        position = self.positions[position_id]
        
        # CORRECCIÓN CRÍTICA: El PnL depende de qué hacemos con cada activo
        if position.direction == 'LONG':
            # LONG spread = Compramos symbol1, Vendimos symbol2
            # En execution.py: side1='buy', side2='sell'
            
            # Symbol1: Lo COMPRAMOS, ganamos si sube
            pnl_symbol1 = position.size1 * (price1 - position.entry_price1)
            
            # Symbol2: Lo VENDIMOS (short), ganamos si BAJA
            pnl_symbol2 = position.size2 * (position.entry_price2 - price2)  # INVERTIDO!
            
        else:  # SHORT
            # SHORT spread = Vendimos symbol1, Compramos symbol2
            # En execution.py: side1='sell', side2='buy'
            
            # Symbol1: Lo VENDIMOS (short), ganamos si BAJA
            pnl_symbol1 = position.size1 * (position.entry_price1 - price1)  # INVERTIDO!
            
            # Symbol2: Lo COMPRAMOS, ganamos si sube
            pnl_symbol2 = position.size2 * (price2 - position.entry_price2)
        
        pnl_gross = pnl_symbol1 + pnl_symbol2
        pnl_net = pnl_gross - cost
        
        # Guardar para debugging
        position.exit_price1 = price1
        position.exit_price2 = price2
        position.pnl_symbol1 = pnl_symbol1
        position.pnl_symbol2 = pnl_symbol2
        position.pnl_gross = pnl_gross
        position.pnl_net = pnl_net
        
        # Log detallado para debugging
        log.debug(f"PnL Debug | {position.direction} | "
                 f"S1: {position.entry_price1:.4f}→{price1:.4f} pnl={pnl_symbol1:.2f} | "
                 f"S2: {position.entry_price2:.4f}→{price2:.4f} pnl={pnl_symbol2:.2f} | "
                 f"Total: {pnl_gross:.2f}")
        
        # Actualizar cash
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
            'pnl_symbol1': pnl_symbol1,
            'pnl_symbol2': pnl_symbol2,
            'cash_after': self.cash
        })
        
        return pnl_net
    
    def update_positions(self, price1: float, price2: float):
        """Mark-to-market continuo - mismo cálculo que close"""
        
        for position in self.positions.values():
            # Usar la misma lógica que close_position
            if position.direction == 'LONG':
                # LONG: compramos S1, vendimos S2
                unrealized_pnl1 = position.size1 * (price1 - position.entry_price1)
                unrealized_pnl2 = position.size2 * (position.entry_price2 - price2)
            else:  # SHORT
                # SHORT: vendimos S1, compramos S2
                unrealized_pnl1 = position.size1 * (position.entry_price1 - price1)
                unrealized_pnl2 = position.size2 * (price2 - position.entry_price2)
            
            unrealized_pnl = unrealized_pnl1 + unrealized_pnl2
            
            position.current_value = position.size1 * price1 + position.size2 * price2
            position.unrealized_pnl = unrealized_pnl
            position.unrealized_pnl1 = unrealized_pnl1
            position.unrealized_pnl2 = unrealized_pnl2
    
    def get_equity(self) -> float:
        """Equity = Cash + Capital invertido + PnL no realizado"""
        capital_invested = sum(p.entry_cost for p in self.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        return self.cash + capital_invested + unrealized_pnl
    
    def get_positions_value(self) -> float:
        """Obtener valor total de posiciones abiertas"""
        return sum(p.current_value for p in self.positions.values())
    
    def get_total_exposure(self) -> float:
        """Calcular exposición total"""
        return sum(abs(p.current_value) for p in self.positions.values())
    
    def get_trade_metrics(self) -> Dict:
        """Obtener métricas de trading detalladas"""
        if not self.closed_positions:
            return {}
        
        pnls = [p.pnl_net for p in self.closed_positions if hasattr(p, 'pnl_net')]
        if not pnls:
            return {}
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        return {
            'total_trades': len(pnls),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else float('inf')
        }
    
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