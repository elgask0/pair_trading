#!/usr/bin/env python3
"""
Metrics Calculator - Cálculo de métricas de performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

class MetricsCalculator:
    """Calcula métricas de performance del backtest"""
    
    def calculate_all_metrics(self, trades: List[Dict], equity_curve: pd.Series, 
                             initial_capital: float) -> Dict[str, float]:
        """Calcular todas las métricas de performance"""
        
        metrics = {}
        
        # Métricas básicas
        metrics['initial_capital'] = initial_capital
        metrics['final_equity'] = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
        metrics['total_pnl'] = metrics['final_equity'] - initial_capital
        metrics['total_return'] = (metrics['final_equity'] / initial_capital - 1)
        
        # Calcular retornos
        returns = equity_curve.pct_change().dropna()
        
        # Sharpe Ratio (asumiendo 0% risk-free rate)
        if len(returns) > 0:
            metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0
        
        # Maximum Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Métricas de trading
        if trades:
            # Separar trades de apertura y cierre
            closes = [t for t in trades if t.get('action') == 'CLOSE']
            
            if closes:
                pnls = [t['pnl'] for t in closes if 'pnl' in t]
                
                if pnls:
                    wins = [pnl for pnl in pnls if pnl > 0]
                    losses = [pnl for pnl in pnls if pnl < 0]
                    
                    metrics['total_trades'] = len(pnls)
                    metrics['win_rate'] = len(wins) / len(pnls) if pnls else 0
                    metrics['avg_win'] = np.mean(wins) if wins else 0
                    metrics['avg_loss'] = np.mean(losses) if losses else 0
                    
                    # Profit Factor
                    gross_profit = sum(wins) if wins else 0
                    gross_loss = abs(sum(losses)) if losses else 0
                    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
                    
                    # Expectancy
                    metrics['expectancy'] = np.mean(pnls) if pnls else 0
                else:
                    metrics['total_trades'] = 0
                    metrics['win_rate'] = 0
                    metrics['avg_win'] = 0
                    metrics['avg_loss'] = 0
                    metrics['profit_factor'] = 0
                    metrics['expectancy'] = 0
            else:
                # No hay trades cerrados
                metrics['total_trades'] = 0
                metrics['win_rate'] = 0
                metrics['avg_win'] = 0
                metrics['avg_loss'] = 0
                metrics['profit_factor'] = 0
                metrics['expectancy'] = 0
        else:
            metrics['total_trades'] = 0
            metrics['win_rate'] = 0
            metrics['avg_win'] = 0
            metrics['avg_loss'] = 0
            metrics['profit_factor'] = 0
            metrics['expectancy'] = 0
        
        # Annual Return (aproximado)
        if len(equity_curve) > 0:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if days > 0:
                metrics['annual_return'] = (metrics['final_equity'] / initial_capital) ** (365 / days) - 1
            else:
                metrics['annual_return'] = 0
        else:
            metrics['annual_return'] = 0
        
        # Calmar Ratio
        if metrics['max_drawdown'] < 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
        
        return metrics