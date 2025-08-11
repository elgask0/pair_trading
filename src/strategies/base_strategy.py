#!/usr/bin/env python3
"""
Base Strategy - Clase base para todas las estrategias
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

class BaseStrategy(ABC):
    """Clase base abstracta para estrategias de trading"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.data: Optional[pd.DataFrame] = None
        self.signals_history = []
        self.state = {}
        
    def set_data(self, data: pd.DataFrame):
        """Configurar los datos para la estrategia"""
        self.data = data
        
    @abstractmethod
    def initialize(self):
        """Inicializar la estrategia antes del backtest"""
        pass
    
    @abstractmethod
    def generate_signal(self, timestamp: datetime, row: pd.Series, index: int) -> Optional[Dict]:
        """
        Generar señal de trading
        
        Returns:
            Dict con keys:
                - action: 'LONG', 'SHORT', 'CLOSE', 'HOLD'
                - strength: float (0-1)
                - reason: str
                - position_id: str (para CLOSE)
                - z_score: float (opcional)
        """
        pass
    
    def get_signals_history(self) -> pd.DataFrame:
        """Obtener historial de señales generadas"""
        if self.signals_history:
            return pd.DataFrame(self.signals_history)
        return pd.DataFrame()