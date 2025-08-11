#!/usr/bin/env python3
"""
Backtest Engine - Motor principal para backtesting con slippage dinámico basado en orderbook
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
    """Configuración del backtest con soporte para orderbook"""
    symbol1: str
    symbol2: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    position_size: float = 0.5  # % del capital por operación
    max_positions: int = 1

    # Execution configuration
    commission_bps: float = 2.0              # Comisión en basis points
    use_orderbook: bool = True               # Usar orderbook para slippage dinámico
    fallback_slippage_bps: float = 5.0       # Slippage cuando no hay orderbook
    max_participation_rate: float = 0.1      # Máximo % del volumen

    # Data configuration
    resample_minutes: int = 1


@dataclass
class BacktestResults:
    """Resultados del backtest con métricas adicionales"""
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, float] = field(default_factory=dict)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    slippage_analysis: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Motor principal de backtest con soporte para orderbook"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = Portfolio(initial_capital=config.initial_capital)

        # Simulador de ejecución con orderbook
        self.execution = EnhancedExecutionSimulator(
            commission_bps=config.commission_bps,
            use_orderbook=config.use_orderbook,
            fallback_slippage_bps=config.fallback_slippage_bps,
            max_participation_rate=config.max_participation_rate,
        )

        self.data_loader = DataLoader()
        self.metrics_calculator = MetricsCalculator()
        self.results = BacktestResults()

        # Data containers
        self.data1: Optional[pd.DataFrame] = None
        self.data2: Optional[pd.DataFrame] = None
        self.aligned_data: Optional[pd.DataFrame] = None

        # Strategy
        self.strategy: Optional[BaseStrategy] = None

        # Tracking slippage
        self.slippage_tracking = []

    def load_data(self) -> bool:
        """Cargar y alinear datos de ambos símbolos"""
        try:
            log.info(f"Loading data for {self.config.symbol1} and {self.config.symbol2}")
            log.info(f"Period: {self.config.start_date} to {self.config.end_date}")
            log.info(f"Resample: {self.config.resample_minutes} minutes")

            # Cargar datos
            self.data1 = self.data_loader.load_ohlcv(
                self.config.symbol1,
                self.config.start_date,
                self.config.end_date,
                self.config.resample_minutes
            )

            self.data2 = self.data_loader.load_ohlcv(
                self.config.symbol2,
                self.config.start_date,
                self.config.end_date,
                self.config.resample_minutes
            )

            if self.data1 is None or self.data2 is None or self.data1.empty or self.data2.empty:
                log.error("One or both symbols have no data")
                return False

            # Alinear datos
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
        """Alinear datos de ambos símbolos"""
        common_index = self.data1.index.intersection(self.data2.index)

        if len(common_index) == 0:
            log.error("No common timestamps between symbols")
            return pd.DataFrame()

        aligned = pd.DataFrame(index=common_index)

        # Precios y volúmenes
        aligned['price1'] = self.data1.loc[common_index, 'close']
        aligned['price2'] = self.data2.loc[common_index, 'close']
        aligned['volume1'] = self.data1.loc[common_index, 'volume']
        aligned['volume2'] = self.data2.loc[common_index, 'volume']

        # OHLC completo para análisis
        aligned['open1'] = self.data1.loc[common_index, 'open']
        aligned['high1'] = self.data1.loc[common_index, 'high']
        aligned['low1'] = self.data1.loc[common_index, 'low']
        aligned['open2'] = self.data2.loc[common_index, 'open']
        aligned['high2'] = self.data2.loc[common_index, 'high']
        aligned['low2'] = self.data2.loc[common_index, 'low']

        # Log prices para cointegración
        aligned['log_price1'] = np.log(aligned['price1'])
        aligned['log_price2'] = np.log(aligned['price2'])

        # Returns
        aligned['returns1'] = aligned['price1'].pct_change()
        aligned['returns2'] = aligned['price2'].pct_change()

        # Log returns
        aligned['log_returns1'] = aligned['log_price1'].diff()
        aligned['log_returns2'] = aligned['log_price2'].diff()

        return aligned.dropna()

    def set_strategy(self, strategy: BaseStrategy):
        """Configurar la estrategia a usar"""
        self.strategy = strategy
        self.strategy.set_data(self.aligned_data)
        log.info(f"Strategy set: {strategy.__class__.__name__}")
        log.info(f"Strategy params: {getattr(strategy, 'params', {})}")

    def run(self) -> BacktestResults:
        """Ejecutar el backtest completo"""
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

        # Inicializar strategy
        self.strategy.initialize()

        # Preparar contenedores
        equity_curve = []
        position_history = []

        # Variables de tracking
        last_signal_action = 'HOLD'
        signals_generated = {'LONG': 0, 'SHORT': 0, 'CLOSE': 0, 'HOLD': 0}

        # Loop principal
        log.info("Starting main backtest loop...")
        for i, (timestamp, row) in enumerate(self.aligned_data.iterrows()):

            # Progress log cada 1000 barras
            if i % 1000 == 0 and i > 0:
                progress = (i / len(self.aligned_data)) * 100
                log.info(f"Progress: {progress:.1f}% ({i}/{len(self.aligned_data)})")

            # Generar señales
            signal = self.strategy.generate_signal(timestamp, row, i)

            if signal:
                signals_generated[signal.get('action', 'HOLD')] += 1

                # Log cambios de señal
                if signal['action'] != last_signal_action and signal['action'] != 'HOLD':
                    log.debug(f"[{timestamp}] Signal: {signal['action']} | Z-score: {signal.get('z_score', 0):.2f}")
                    last_signal_action = signal['action']

            # Ejecutar señal si existe
            if signal and signal['action'] not in ['HOLD', None]:
                trade = self._execute_signal(signal, timestamp, row)
                if trade:
                    self.results.trades.append(trade)
                    log.info(f"[{timestamp}] Trade executed: {trade['action']} | "
                             f"Cost: ${trade.get('cost', 0):.2f} | "
                             f"Slippage: {trade.get('avg_slippage_bps', 0):.1f} bps")

            # Actualizar posiciones
            self.portfolio.update_positions(row['price1'], row['price2'])

            # Guardar estado
            current_equity = self.portfolio.get_equity()
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
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

        log.info("Main loop completed. Closing open positions...")

        # Cerrar posiciones abiertas
        self._close_all_positions(self.aligned_data.index[-1], self.aligned_data.iloc[-1])

        # Preparar resultados
        self.results.equity_curve = pd.DataFrame(equity_curve).set_index('timestamp')['equity']
        self.results.positions = pd.DataFrame(position_history) if position_history else pd.DataFrame()

        # Calcular métricas
        self.results.metrics = self.metrics_calculator.calculate_all_metrics(
            self.results.trades,
            self.results.equity_curve,
            self.config.initial_capital
        )

        # Analizar slippage
        self._analyze_slippage()

        # Guardar señales de la estrategia
        self.results.signals = self.strategy.get_signals_history()

        # Imprimir resumen
        log.info("\nSignals generated summary:")
        for action, count in signals_generated.items():
            log.info(f"  {action}: {count}")

        self._print_summary()

        return self.results

    def _execute_signal(self, signal: Dict, timestamp: datetime, row: pd.Series) -> Optional[Dict]:
        """Ejecutar una señal de trading con slippage dinámico"""
        try:
            # Verificar si podemos abrir nueva posición
            if signal['action'] in ['LONG', 'SHORT']:
                if len(self.portfolio.positions) >= self.config.max_positions:
                    log.debug(f"Cannot open position: max positions ({self.config.max_positions}) reached")
                    return None

                # Calcular tamaño de posición
                position_value = self.portfolio.cash * self.config.position_size

                # Simular ejecución con slippage dinámico
                execution_result = self.execution.simulate_execution(
                    signal=signal,
                    symbol1=self.config.symbol1,
                    symbol2=self.config.symbol2,
                    price1=row['price1'],
                    price2=row['price2'],
                    volume1=row['volume1'],
                    volume2=row['volume2'],
                    position_value=position_value,
                    timestamp=timestamp
                )

                if execution_result['success']:
                    # Abrir posición en portfolio
                    position = self.portfolio.open_position(
                        timestamp=timestamp,
                        direction=signal['action'],
                        size1=execution_result['size1'],
                        size2=execution_result['size2'],
                        price1=execution_result['exec_price1'],
                        price2=execution_result['exec_price2'],
                        cost=execution_result['total_cost']
                    )

                    # Registrar trade
                    trade = {
                        'timestamp': timestamp,
                        'action': signal['action'],
                        'position_id': position.id,
                        'size1': execution_result['size1'],
                        'size2': execution_result['size2'],
                        'price1': execution_result['exec_price1'],
                        'price2': execution_result['exec_price2'],
                        'cost': execution_result['total_cost'],
                        'signal_strength': signal.get('strength', 0),
                        'z_score': signal.get('z_score', 0),
                        'hedge_ratio': signal.get('hedge_ratio', 1.0),
                        'slippage1_bps': execution_result.get('slippage1_bps', 0),
                        'slippage2_bps': execution_result.get('slippage2_bps', 0),
                        'avg_slippage_bps': execution_result.get('avg_slippage_bps', 0),
                        'orderbook_used': execution_result.get('orderbook_used', False)
                    }

                    # Track slippage
                    self.slippage_tracking.append({
                        'timestamp': timestamp,
                        'slippage_bps': execution_result.get('avg_slippage_bps', 0),
                        'orderbook_used': execution_result.get('orderbook_used', False)
                    })

                    return trade

            elif signal['action'] == 'CLOSE':
                # Cerrar posición existente
                if self.portfolio.positions:
                    position_id = list(self.portfolio.positions.keys())[0]
                    position = self.portfolio.positions[position_id]

                    # Simular cierre con orderbook
                    close_result = self.execution.simulate_close(
                        position=position,
                        symbol1=self.config.symbol1,
                        symbol2=self.config.symbol2,
                        price1=row['price1'],
                        price2=row['price2'],
                        timestamp=timestamp
                    )

                    # Cerrar en portfolio
                    pnl = self.portfolio.close_position(
                        position_id=position_id,
                        price1=close_result['exec_price1'],
                        price2=close_result['exec_price2'],
                        cost=close_result['total_cost']
                    )

                    # Registrar trade de cierre
                    trade = {
                        'timestamp': timestamp,
                        'action': 'CLOSE',
                        'position_id': position_id,
                        'pnl': pnl,
                        'price1': close_result['exec_price1'],
                        'price2': close_result['exec_price2'],
                        'cost': close_result['total_cost'],
                        'z_score': signal.get('z_score', 0)
                    }

                    return trade

        except Exception as e:
            log.error(f"Error executing signal: {e}")
            import traceback
            log.error(traceback.format_exc())
            return None

    def _close_all_positions(self, timestamp: datetime, row: pd.Series):
        """Cerrar todas las posiciones abiertas al final del backtest"""
        positions_to_close = list(self.portfolio.positions.keys())

        for position_id in positions_to_close:
            position = self.portfolio.positions[position_id]

            # Simular cierre
            close_result = self.execution.simulate_close(
                position=position,
                symbol1=self.config.symbol1,
                symbol2=self.config.symbol2,
                price1=row['price1'],
                price2=row['price2'],
                timestamp=timestamp
            )

            # Cerrar en portfolio
            pnl = self.portfolio.close_position(
                position_id=position_id,
                price1=close_result['exec_price1'],
                price2=close_result['exec_price2'],
                cost=close_result['total_cost']
            )

            # Registrar trade
            trade = {
                'timestamp': timestamp,
                'action': 'CLOSE_FINAL',
                'position_id': position_id,
                'pnl': pnl,
                'price1': close_result['exec_price1'],
                'price2': close_result['exec_price2'],
                'cost': close_result['total_cost']
            }

            self.results.trades.append(trade)
            log.info(f"Closed final position {position_id}: PnL ${pnl:.2f}")

    def _analyze_slippage(self):
        """Analizar el impacto del slippage"""
        if self.slippage_tracking:
            slippage_df = pd.DataFrame(self.slippage_tracking)

            self.results.slippage_analysis = {
                'avg_slippage_bps': slippage_df['slippage_bps'].mean(),
                'max_slippage_bps': slippage_df['slippage_bps'].max(),
                'min_slippage_bps': slippage_df['slippage_bps'].min(),
                'orderbook_usage': slippage_df['orderbook_used'].mean() * 100,
                'total_slippage_events': len(slippage_df)
            }

            # Calcular costo total de slippage
            if self.results.trades:
                trades_df = pd.DataFrame([t for t in self.results.trades if t.get('action') in ['LONG', 'SHORT']])
                if not trades_df.empty and 'avg_slippage_bps' in trades_df.columns:
                    total_traded_value = trades_df['cost'].sum()
                    avg_slippage = trades_df['avg_slippage_bps'].mean()
                    slippage_cost = total_traded_value * (avg_slippage / 10000)
                    self.results.slippage_analysis['total_slippage_cost'] = slippage_cost

    def _print_summary(self):
        """Imprimir resumen detallado del backtest"""
        metrics = self.results.metrics

        log.info("\n" + "=" * 80)
        log.info("BACKTEST RESULTS SUMMARY")
        log.info("=" * 80)

        log.info(f"\n📊 Performance Metrics:")
        log.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        log.info(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
        log.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        log.info(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        log.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

        log.info(f"\n📈 Trading Statistics:")
        log.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        log.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        log.info(f"  Avg Win: ${metrics.get('avg_win', 0):.2f}")
        log.info(f"  Avg Loss: ${metrics.get('avg_loss', 0):.2f}")
        log.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        log.info(f"  Expectancy: ${metrics.get('expectancy', 0):.2f}")

        # Slippage analysis
        if self.results.slippage_analysis:
            log.info(f"\n💹 Slippage Analysis:")
            log.info(f"  Avg Slippage: {self.results.slippage_analysis.get('avg_slippage_bps', 0):.1f} bps")
            log.info(f"  Max Slippage: {self.results.slippage_analysis.get('max_slippage_bps', 0):.1f} bps")
            log.info(f"  Orderbook Usage: {self.results.slippage_analysis.get('orderbook_usage', 0):.1f}%")

            if 'total_slippage_cost' in self.results.slippage_analysis:
                log.info(f"  Total Slippage Cost: ${self.results.slippage_analysis['total_slippage_cost']:.2f}")

        log.info(f"\n💰 Final Results:")
        log.info(f"  Initial Capital: ${self.config.initial_capital:,.2f}")
        log.info(f"  Final Equity: ${metrics.get('final_equity', 0):,.2f}")
        log.info(f"  Total PnL: ${metrics.get('total_pnl', 0):,.2f}")

        # Strategy specific metrics
        if hasattr(self.strategy, 'get_strategy_metrics'):
            strategy_metrics = self.strategy.get_strategy_metrics()
            if strategy_metrics:
                log.info(f"\n🎯 Strategy Metrics:")
                log.info(f"  Current Hedge Ratio: {strategy_metrics.get('current_hedge_ratio', 1.0):.4f}")
                log.info(f"  Avg Z-Score: {strategy_metrics.get('avg_z_score', 0):.2f}")
                log.info(f"  Max Z-Score: {strategy_metrics.get('max_z_score', 0):.2f}")
                log.info(f"  Min Z-Score: {strategy_metrics.get('min_z_score', 0):.2f}")

        log.info("=" * 80)