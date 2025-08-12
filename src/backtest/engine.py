#!/usr/bin/env python3
"""
Backtest Engine - LOGS CONDENSADOS (CORREGIDO)
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.backtest.portfolio import Portfolio
from src.backtest.execution import EnhancedExecutionSimulator
from src.backtest.data_loader import DataLoader
from src.strategies.base_strategy import BaseStrategy
from src.analytics.metrics import MetricsCalculator
from src.utils.logger import get_logger

log = get_logger()

@dataclass
class BacktestConfig:
    symbol1: str
    symbol2: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    position_size: float = 0.5
    max_positions: int = 1

    commission_bps: float = 2.0
    use_orderbook: bool = True
    fallback_slippage_bps: float = 5.0
    max_participation_rate: float = 0.1

    resample_minutes: int = 1
    orderbook_time_tolerance_seconds: Optional[int] = None

@dataclass
class BacktestResults:
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, float] = field(default_factory=dict)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    slippage_analysis: Dict[str, float] = field(default_factory=dict)

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = Portfolio(initial_capital=config.initial_capital)

        tol = config.orderbook_time_tolerance_seconds
        if tol is None:
            tol = max(60, int(config.resample_minutes * 60 / 2))

        self.execution = EnhancedExecutionSimulator(
            commission_bps=config.commission_bps,
            use_orderbook=config.use_orderbook,
            fallback_slippage_bps=config.fallback_slippage_bps,
            max_participation_rate=config.max_participation_rate,
            orderbook_time_tolerance_seconds=tol,
        )

        self.data_loader = DataLoader()
        self.metrics_calculator = MetricsCalculator()
        self.results = BacktestResults()

        self.data1: Optional[pd.DataFrame] = None
        self.data2: Optional[pd.DataFrame] = None
        self.aligned_data: Optional[pd.DataFrame] = None

        self.strategy: Optional[BaseStrategy] = None
        self.slippage_tracking = []

    def load_data(self) -> bool:
        try:
            log.info(f"Loading data for {self.config.symbol1} and {self.config.symbol2}")
            log.info(f"Period: {self.config.start_date} to {self.config.end_date}")
            log.info(f"Resample: {self.config.resample_minutes} minutes")

            self.data1 = self.data_loader.load_ohlcv(
                self.config.symbol1, self.config.start_date, self.config.end_date, self.config.resample_minutes
            )
            self.data2 = self.data_loader.load_ohlcv(
                self.config.symbol2, self.config.start_date, self.config.end_date, self.config.resample_minutes
            )

            if self.data1 is None or self.data2 is None or self.data1.empty or self.data2.empty:
                log.error("One or both symbols have no data")
                return False

            self.aligned_data = self._align_data()
            if self.aligned_data is None or len(self.aligned_data) < 100:
                log.error(f"Insufficient data: {0 if self.aligned_data is None else len(self.aligned_data)} rows")
                return False

            log.info(f"Data loaded: {len(self.aligned_data)} aligned observations")
            log.info(f"Date range: {self.aligned_data.index[0]} to {self.aligned_data.index[-1]}")
            return True

        except Exception as e:
            log.error(f"Error loading data: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False

    def _align_data(self) -> pd.DataFrame:
        common_index = self.data1.index.intersection(self.data2.index)
        if len(common_index) == 0:
            log.error("No common timestamps between symbols")
            return pd.DataFrame()

        aligned = pd.DataFrame(index=common_index)
        aligned['price1'] = self.data1.loc[common_index, 'close']
        aligned['price2'] = self.data2.loc[common_index, 'close']
        aligned['volume1'] = self.data1.loc[common_index, 'volume']
        aligned['volume2'] = self.data2.loc[common_index, 'volume']

        aligned['open1'] = self.data1.loc[common_index, 'open']
        aligned['high1'] = self.data1.loc[common_index, 'high']
        aligned['low1'] = self.data1.loc[common_index, 'low']
        aligned['open2'] = self.data2.loc[common_index, 'open']
        aligned['high2'] = self.data2.loc[common_index, 'high']
        aligned['low2'] = self.data2.loc[common_index, 'low']

        aligned['log_price1'] = np.log(aligned['price1'])
        aligned['log_price2'] = np.log(aligned['price2'])
        aligned['returns1'] = aligned['price1'].pct_change()
        aligned['returns2'] = aligned['price2'].pct_change()
        aligned['log_returns1'] = aligned['log_price1'].diff()
        aligned['log_returns2'] = aligned['log_price2'].diff()

        return aligned.dropna()

    def set_strategy(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.strategy.set_data(self.aligned_data)
        
        # 🔧 NUEVO: Pasar resample_minutes a la estrategia
        if hasattr(self.strategy, 'set_resample_minutes'):
            self.strategy.set_resample_minutes(self.config.resample_minutes)
        
        log.info(f"Strategy set: {strategy.__class__.__name__}")
        log.info(f"Strategy params: {getattr(strategy, 'params', {})}")
        log.info(f"Resample minutes passed to strategy: {self.config.resample_minutes}")

    def run(self) -> BacktestResults:
        if not self.strategy:
            raise ValueError("No strategy set. Call set_strategy() first.")

        log.info("=" * 80)
        log.info("STARTING BACKTEST")
        log.info("=" * 80)
        log.info(f"Symbols: {self.config.symbol1} vs {self.config.symbol2}")
        log.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        log.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
        log.info(f"Position size: {self.config.position_size:.1%} of capital")
        log.info(f"Max positions: {self.config.max_positions}")
        log.info(f"Commission: {self.config.commission_bps} bps")
        log.info(f"Orderbook mode: {'ENABLED' if self.config.use_orderbook else 'DISABLED'}")
        log.info(f"Fallback slippage: {self.config.fallback_slippage_bps} bps")
        log.info(f"Max participation rate: {self.config.max_participation_rate:.0%}")
        log.info("=" * 80)

        self.strategy.initialize()

        equity_curve, position_history = [], []
        last_signal_action = 'HOLD'
        signals_generated = {'LONG': 0, 'SHORT': 0, 'CLOSE': 0, 'HOLD': 0}

        log.info("Starting main backtest loop...")
        for i, (timestamp, row) in enumerate(self.aligned_data.iterrows()):
            if i % 1000 == 0 and i > 0:
                progress = (i / len(self.aligned_data)) * 100
                log.info(f"Progress: {progress:.1f}% ({i}/{len(self.aligned_data)})")

            signal = self.strategy.generate_signal(timestamp, row, i)
            if signal:
                signals_generated[signal.get('action', 'HOLD')] += 1
                if signal['action'] != last_signal_action and signal['action'] != 'HOLD':
                    last_signal_action = signal['action']

            if signal and signal['action'] not in ['HOLD', None]:
                trade = self._execute_signal(signal, timestamp, row)
                if trade:
                    self.results.trades.append(trade)

            self.portfolio.update_positions(row['price1'], row['price2'])
            equity_curve.append({
                'timestamp': timestamp,
                'equity': self.portfolio.get_equity(),
                'cash': self.portfolio.cash,
                'positions_value': self.portfolio.get_positions_value(),
                'unrealized_pnl': sum(p.unrealized_pnl for p in self.portfolio.positions.values())
            })
            if self.portfolio.positions:
                position_history.append({
                    'timestamp': timestamp,
                    'positions': len(self.portfolio.positions),
                    'exposure': self.portfolio.get_total_exposure()
                })

        self._close_all_positions(self.aligned_data.index[-1], self.aligned_data.iloc[-1])

        self.results.equity_curve = pd.DataFrame(equity_curve).set_index('timestamp')['equity']
        self.results.positions = pd.DataFrame(position_history) if position_history else pd.DataFrame()

        self.results.metrics = self.metrics_calculator.calculate_all_metrics(
            self.results.trades, self.results.equity_curve, self.config.initial_capital
        )

        self._analyze_slippage()
        self.results.signals = self.strategy.get_signals_history()

        log.info("\nSignals generated summary:")
        for action, count in signals_generated.items():
            log.info(f"  {action}: {count}")

        self._print_enhanced_summary()
        return self.results

    def _execute_signal(self, signal: Dict, timestamp: datetime, row: pd.Series) -> Optional[Dict]:
        """LOGS CONDENSADOS: Una línea por operación"""
        try:
            if signal['action'] in ['LONG', 'SHORT']:
                if len(self.portfolio.positions) >= self.config.max_positions:
                    log.warning(f"🚫 MAX POSITIONS | Current: {len(self.portfolio.positions)}/{self.config.max_positions}")
                    return None
                    
                position_value = self.portfolio.cash * self.config.position_size
                
                execution_result = self.execution.simulate_execution(
                    signal=signal,
                    symbol1=self.config.symbol1,
                    symbol2=self.config.symbol2,
                    price1=row['price1'], price2=row['price2'],
                    volume1=row['volume1'], volume2=row['volume2'],
                    position_value=position_value, timestamp=timestamp
                )
                
                if execution_result['success']:
                    position = self.portfolio.open_position(
                        timestamp=timestamp, direction=signal['action'],
                        size1=execution_result['size1'], size2=execution_result['size2'],
                        price1=execution_result['exec_price1'], price2=execution_result['exec_price2'],
                        cost=execution_result['total_cost']
                    )
                    
                    # LOG CONDENSADO: Una sola línea con toda la info
                    log.info(f"🚀 OPEN {signal['action']} | {timestamp} | ID:{position.id} | Z:{signal.get('z_score', 0):.3f} | HR:{signal.get('hedge_ratio', 1):.4f} | "
                            f"S1:{execution_result['size1']:.2f}@{execution_result['exec_price1']:.6f}({execution_result.get('side1','?')}) | "
                            f"S2:{execution_result['size2']:.2f}@{execution_result['exec_price2']:.6f}({execution_result.get('side2','?')}) | "
                            f"Cost:${execution_result['total_cost']:.2f} | Slip:{execution_result.get('avg_slippage_bps', 0):.1f}bps | "
                            f"Cash:${self.portfolio.cash:.2f}")
                    
                    trade = {
                        'timestamp': timestamp, 'action': signal['action'], 'position_id': position.id,
                        'size1': execution_result['size1'], 'size2': execution_result['size2'],
                        'price1': execution_result['exec_price1'], 'price2': execution_result['exec_price2'],
                        'cost': execution_result['total_cost'],
                        'signal_strength': signal.get('strength', 0), 'z_score': signal.get('z_score', 0),
                        'hedge_ratio': signal.get('hedge_ratio', 1.0),
                        'slippage1_bps': execution_result.get('slippage1_bps', 0),
                        'slippage2_bps': execution_result.get('slippage2_bps', 0),
                        'avg_slippage_bps': execution_result.get('avg_slippage_bps', 0),
                        'orderbook_used': execution_result.get('orderbook_used', False)
                    }
                    
                    self.slippage_tracking.append({
                        'timestamp': timestamp,
                        'slippage_bps': execution_result.get('avg_slippage_bps', 0),
                        'orderbook_used': execution_result.get('orderbook_used', False)
                    })
                    
                    return trade

            elif signal['action'] == 'CLOSE' and self.portfolio.positions:
                position_id = list(self.portfolio.positions.keys())[0]
                position = self.portfolio.positions[position_id]
                
                close_result = self.execution.simulate_close(
                    position=position, symbol1=self.config.symbol1, symbol2=self.config.symbol2,
                    price1=row['price1'], price2=row['price2'], timestamp=timestamp
                )
                
                pnl = self.portfolio.close_position(
                    position_id=position_id, price1=close_result['exec_price1'],
                    price2=close_result['exec_price2'], cost=close_result['total_cost']
                )
                
                # Calcular duración del trade
                duration = timestamp - position.timestamp
                
                # LOG CONDENSADO: Una sola línea con toda la info
                log.info(f"🔄 CLOSE {position.direction} | {timestamp} | ID:{position_id} | Z:{signal.get('z_score', 0):.3f} | "
                        f"PnL:${pnl:.2f} | Duration:{duration} | "
                        f"Reason:{signal.get('reason', 'N/A')} | Cash:${self.portfolio.cash:.2f}")
                
                return {
                    'timestamp': timestamp, 'action': 'CLOSE', 'position_id': position_id,
                    'pnl': pnl, 'price1': close_result['exec_price1'],
                    'price2': close_result['exec_price2'], 'cost': close_result['total_cost'],
                    'z_score': signal.get('z_score', 0), 'reason': signal.get('reason', 'N/A'),
                    'duration_seconds': duration.total_seconds()
                }
                
        except Exception as e:
            log.error(f"❌ Error executing signal: {e}")
            return None

    def _close_all_positions(self, timestamp: datetime, row: pd.Series):
        """Cerrar todas las posiciones al final"""
        for position_id in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[position_id]
            
            close_result = self.execution.simulate_close(
                position=position, symbol1=self.config.symbol1, symbol2=self.config.symbol2,
                price1=row['price1'], price2=row['price2'], timestamp=timestamp
            )
            
            pnl = self.portfolio.close_position(
                position_id=position_id, price1=close_result['exec_price1'],
                price2=close_result['exec_price2'], cost=close_result['total_cost']
            )
            
            duration = timestamp - position.timestamp
            log.info(f"🔚 FINAL CLOSE | ID:{position_id} | PnL:${pnl:.2f} | Duration:{duration}")
            
            self.results.trades.append({
                'timestamp': timestamp, 'action': 'CLOSE_FINAL', 'position_id': position_id,
                'pnl': pnl, 'price1': close_result['exec_price1'],
                'price2': close_result['exec_price2'], 'cost': close_result['total_cost'],
                'reason': 'final_close', 'duration_seconds': duration.total_seconds()
            })

    def _analyze_slippage(self):
        if self.slippage_tracking:
            s = pd.DataFrame(self.slippage_tracking)
            self.results.slippage_analysis = {
                'avg_slippage_bps': s['slippage_bps'].mean(),
                'max_slippage_bps': s['slippage_bps'].max(),
                'min_slippage_bps': s['slippage_bps'].min(),
                'orderbook_usage': s['orderbook_used'].mean() * 100,
                'total_slippage_events': len(s)
            }
            if self.results.trades:
                t = pd.DataFrame([x for x in self.results.trades if x.get('action') in ['LONG','SHORT']])
                if not t.empty and 'avg_slippage_bps' in t.columns:
                    total_value = t['cost'].sum()
                    avg_slip = t['avg_slippage_bps'].mean()
                    self.results.slippage_analysis['total_slippage_cost'] = total_value * (avg_slip / 10000)

    def _print_enhanced_summary(self):
        """REPORTE AMPLIADO CON MÉTRICAS DETALLADAS"""
        m = self.results.metrics
        
        # Análisis detallado de trades
        trades_df = pd.DataFrame(self.results.trades) if self.results.trades else pd.DataFrame()
        
        log.info("\n" + "=" * 100)
        log.info("🔍 ENHANCED BACKTEST RESULTS SUMMARY")
        log.info("=" * 100)
        
        # Performance básica
        log.info(f"\n📊 PERFORMANCE OVERVIEW:")
        log.info(f"  Total Return: {m.get('total_return', 0):.2%} | Annual: {m.get('annual_return', 0):.2%}")
        log.info(f"  Sharpe: {m.get('sharpe_ratio', 0):.2f} | Calmar: {m.get('calmar_ratio', 0):.2f} | Max DD: {m.get('max_drawdown', 0):.2%}")
        
        # Trading Statistics Detalladas
        if not trades_df.empty:
            close_trades = trades_df[trades_df['action'].isin(['CLOSE', 'CLOSE_FINAL'])].copy()
            
            if not close_trades.empty:
                # Análisis por razón de cierre
                exit_reasons = close_trades['reason'].value_counts() if 'reason' in close_trades.columns else {}
                stop_losses = exit_reasons.get('stop_loss', 0)
                normal_exits = exit_reasons.get('exit_signal', 0) + exit_reasons.get('final_close', 0)
                
                # Duración de trades
                if 'duration_seconds' in close_trades.columns:
                    durations_hours = close_trades['duration_seconds'] / 3600
                    avg_duration_hours = durations_hours.mean()
                    median_duration_hours = durations_hours.median()
                    max_duration_hours = durations_hours.max()
                else:
                    avg_duration_hours = median_duration_hours = max_duration_hours = 0
                
                # PnL Analysis
                pnls = close_trades['pnl'].dropna()
                wins = pnls[pnls > 0]
                losses = pnls[pnls < 0]
                
                log.info(f"\n📈 DETAILED TRADING STATS:")
                log.info(f"  Total Trades: {len(close_trades)} | Wins: {len(wins)} ({len(wins)/len(close_trades)*100:.1f}%) | Losses: {len(losses)} ({len(losses)/len(close_trades)*100:.1f}%)")
                log.info(f"  Normal Exits: {normal_exits} | Stop Losses: {stop_losses} | Stop Loss Rate: {stop_losses/len(close_trades)*100:.1f}%")
                log.info(f"  Avg Win: ${wins.mean():.2f} | Avg Loss: ${losses.mean():.2f} | Win/Loss Ratio: {abs(wins.mean()/losses.mean()) if len(losses) > 0 else float('inf'):.2f}")
                log.info(f"  Best Trade: ${pnls.max():.2f} | Worst Trade: ${pnls.min():.2f} | Profit Factor: {m.get('profit_factor', 0):.2f}")
                log.info(f"  Expectancy: ${m.get('expectancy', 0):.2f} | Avg Trade: ${pnls.mean():.2f} | Median Trade: ${pnls.median():.2f}")
                
                log.info(f"\n⏱️ TRADE DURATION ANALYSIS:")
                log.info(f"  Avg Duration: {avg_duration_hours:.1f}h | Median: {median_duration_hours:.1f}h | Max: {max_duration_hours:.1f}h")
                
                # PnL Distribution
                log.info(f"\n💰 PnL DISTRIBUTION:")
                pnl_ranges = [
                    (pnls >= 50, ">= $50"),
                    ((pnls >= 20) & (pnls < 50), "$20-$50"),
                    ((pnls >= 5) & (pnls < 20), "$5-$20"),
                    ((pnls >= 0) & (pnls < 5), "$0-$5"),
                    ((pnls >= -5) & (pnls < 0), "$0 to -$5"),
                    ((pnls >= -20) & (pnls < -5), "-$5 to -$20"),
                    ((pnls >= -50) & (pnls < -20), "-$20 to -$50"),
                    (pnls < -50, "< -$50")
                ]
                
                for condition, label in pnl_ranges:
                    count = condition.sum()
                    pct = count / len(pnls) * 100 if len(pnls) > 0 else 0
                    log.info(f"    {label}: {count} trades ({pct:.1f}%)")
                
                # Z-Score Analysis
                if 'z_score' in close_trades.columns:
                    z_scores = close_trades['z_score'].abs()
                    log.info(f"\n📊 Z-SCORE AT CLOSE ANALYSIS:")
                    log.info(f"  Avg |Z|: {z_scores.mean():.2f} | Median |Z|: {z_scores.median():.2f} | Max |Z|: {z_scores.max():.2f}")
                    log.info(f"  Extreme Closes (|Z|>3): {(z_scores > 3).sum()} ({(z_scores > 3).mean()*100:.1f}%)")
        
        # Strategy Analysis
        strategy_metrics = self.strategy.get_strategy_metrics() if hasattr(self.strategy, 'get_strategy_metrics') else {}
        if strategy_metrics:
            log.info(f"\n🎯 STRATEGY ANALYSIS:")
            log.info(f"  Current Hedge Ratio: {strategy_metrics.get('current_hedge_ratio', 1.0):.4f}")
            log.info(f"  Z-Score Range: {strategy_metrics.get('min_z_score', 0):.2f} to {strategy_metrics.get('max_z_score', 0):.2f}")
            log.info(f"  Avg Z-Score: {strategy_metrics.get('avg_z_score', 0):.3f} | Std: {strategy_metrics.get('z_score_std', 0):.3f}")
            if 'avg_correlation' in strategy_metrics:
                log.info(f"  Avg Correlation: {strategy_metrics.get('avg_correlation', 0):.3f}")
            if 'extreme_z_scores' in strategy_metrics:
                log.info(f"  Extreme Signals (|Z|>3): {strategy_metrics.get('extreme_z_scores', 0)}")
            log.info(f"  Total Signals: {strategy_metrics.get('signals_generated', 0)}")
            
            # 🔧 DEBUG INFO
            if 'resample_minutes' in strategy_metrics:
                log.info(f"  Resample Minutes: {strategy_metrics.get('resample_minutes', 0)}")
                log.info(f"  Lookback Days: {strategy_metrics.get('lookback_days', 0)} = {strategy_metrics.get('lookback_periods', 0)} periods")
        
        # Slippage Analysis
        if self.results.slippage_analysis:
            s = self.results.slippage_analysis
            log.info(f"\n💹 EXECUTION ANALYSIS:")
            log.info(f"  Avg Slippage: {s.get('avg_slippage_bps', 0):.1f} bps | Max: {s.get('max_slippage_bps', 0):.1f} bps")
            log.info(f"  Orderbook Usage: {s.get('orderbook_usage', 0):.1f}% | Slippage Events: {s.get('total_slippage_events', 0)}")
            if 'total_slippage_cost' in s:
                log.info(f"  Total Slippage Cost: ${s['total_slippage_cost']:.2f}")
        
        # Capital Efficiency
        log.info(f"\n💵 CAPITAL EFFICIENCY:")
        log.info(f"  Initial Capital: ${self.config.initial_capital:,.2f} | Final Equity: ${m.get('final_equity', 0):,.2f}")
        log.info(f"  Total PnL: ${m.get('total_pnl', 0):,.2f} | Max Capital Used: {self.config.position_size:.1%}")
        
        if not trades_df.empty:
            total_volume = trades_df[trades_df['action'].isin(['LONG', 'SHORT'])]['cost'].sum() if 'cost' in trades_df.columns else 0
            log.info(f"  Total Volume Traded: ${total_volume:,.2f} | Turnover: {total_volume/self.config.initial_capital:.1f}x")
        
        # Risk Metrics
        if len(self.results.equity_curve) > 0:
            returns = self.results.equity_curve.pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252 * 24 * 60 / self.config.resample_minutes)  # Annualized
                var_95 = returns.quantile(0.05)
                log.info(f"\n📉 RISK METRICS:")
                log.info(f"  Annualized Volatility: {volatility:.2%} | VaR(95%): {var_95:.2%}")
                log.info(f"  Best Day: +{returns.max():.2%} | Worst Day: {returns.min():.2%}")
        
        log.info("=" * 100)