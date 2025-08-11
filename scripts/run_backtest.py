#!/usr/bin/env python3
"""
Run Backtest - Script principal para ejecutar backtests (slippage dinámico por orderbook)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timedelta
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.strategies.pairs_strategy import PairsTradingStrategy
from src.utils.logger import get_logger

log = get_logger()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def _to_dt(x):
    return pd.to_datetime(x).to_pydatetime()

def plot_results(results):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    ax1 = axes[0]
    results.equity_curve.plot(ax=ax1, label='Equity', linewidth=2)
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    rolling_max = results.equity_curve.expanding().max()
    drawdown = (results.equity_curve - rolling_max) / rolling_max * 100
    drawdown.plot(ax=ax2, label='Drawdown %', linewidth=1)
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axes[2]
    if results.signals is not None and not results.signals.empty:
        sig_df = results.signals.copy()
        if 'timestamp' in sig_df.columns:
            sig_df = sig_df.set_index('timestamp')
        if 'z_score' in sig_df.columns:
            sig_df['z_score'].plot(ax=ax3, label='Z-Score', alpha=0.7)
            ax3.axhline(y=2, linestyle='--', alpha=0.5, label='Entry Long')
            ax3.axhline(y=-2, linestyle='--', alpha=0.5, label='Entry Short')
            ax3.axhline(y=0, linestyle='-', alpha=0.5)
            ax3.set_title('Z-Score Evolution')
            ax3.set_ylabel('Z-Score')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

    plt.tight_layout()
    outdir = Path('plots/backtest'); outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f'backtest_results_{ts}.png'
    plt.savefig(outdir / fname, dpi=100, bbox_inches='tight')
    plt.show()
    log.info(f"Results plot saved to plots/backtest/{fname}")

def main():
    parser = argparse.ArgumentParser(description='Run pairs trading backtest')
    parser.add_argument('--config', type=str, default='config/backtest_config.yaml')
    parser.add_argument('--symbol1', type=str)
    parser.add_argument('--symbol2', type=str)
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)
    else:
        cfg = {
            'symbol1': args.symbol1 or 'BTC_USDT',
            'symbol2': args.symbol2 or 'ETH_USDT',
            'end_date': datetime.now(),
            'start_date': datetime.now() - timedelta(days=args.days),
            'initial_capital': 10000,
            'position_size': 0.5,
            'max_positions': 1,
            'execution': {
                'commission_bps': 2.0,
                'use_orderbook': False,
                'fallback_slippage_bps': 5.0,
                'max_participation_rate': 0.1,
                'orderbook_time_tolerance_seconds': None
            },
            'data': {'resample_minutes': 5},
            'strategy': {},
            'output': {}
        }

    exec_cfg = (cfg.get('execution') or {})
    data_cfg = (cfg.get('data') or {})
    strat_cfg = (cfg.get('strategy') or {})
    out_cfg = (cfg.get('output') or {})

    config = BacktestConfig(
        symbol1=cfg['symbol1'],
        symbol2=cfg['symbol2'],
        start_date=_to_dt(cfg['start_date']),
        end_date=_to_dt(cfg['end_date']),
        initial_capital=cfg.get('initial_capital', 10000),
        position_size=cfg.get('position_size', 0.5),
        max_positions=cfg.get('max_positions', 1),
        commission_bps=exec_cfg.get('commission_bps', 2.0),
        use_orderbook=exec_cfg.get('use_orderbook', True),
        fallback_slippage_bps=exec_cfg.get('fallback_slippage_bps', 5.0),
        max_participation_rate=exec_cfg.get('max_participation_rate', 0.1),
        resample_minutes=data_cfg.get('resample_minutes', 1),
        orderbook_time_tolerance_seconds=exec_cfg.get('orderbook_time_tolerance_seconds')
    )

    engine = BacktestEngine(config)
    if not engine.load_data():
        log.error("Failed to load data")
        return 1

    strategy_params = {
        'lookback_window': strat_cfg.get('lookback_window', 20),
        'entry_threshold': strat_cfg.get('entry_threshold', 2.0),
        'exit_threshold': strat_cfg.get('exit_threshold', 0.0),
        'stop_loss_threshold': strat_cfg.get('stop_loss_threshold', 4.0),
        'min_correlation': strat_cfg.get('min_correlation', 0.5),
        'use_robust_zscore': strat_cfg.get('use_robust_zscore', True),
        'hedge_ratio_window': strat_cfg.get('hedge_ratio_window', 60),
        'rebalance_frequency': strat_cfg.get('rebalance_frequency', 5),
        'signal_confirmation_bars': strat_cfg.get('signal_confirmation_bars', 0),
        'max_holding_periods': strat_cfg.get('max_holding_periods', 0),
    }
    strategy = PairsTradingStrategy(strategy_params)
    engine.set_strategy(strategy)

    results = engine.run()

    outdir = Path('results'); outdir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if (out_cfg.get('save_trades', True)) and results.trades:
        pd.DataFrame(results.trades).to_csv(outdir / f'trades_{ts}.csv', index=False)
        log.info(f"Trades saved to results/trades_{ts}.csv")

    if out_cfg.get('save_signals', True) and results.signals is not None and not results.signals.empty:
        results.signals.to_csv(outdir / f'signals_{ts}.csv', index=False)
        log.info(f"Signals saved to results/signals_{ts}.csv")

    if out_cfg.get('save_equity_curve', True) and results.equity_curve is not None:
        results.equity_curve.to_csv(outdir / f'equity_curve_{ts}.csv', header=['equity'])
        log.info(f"Equity curve saved to results/equity_curve_{ts}.csv")

    pd.DataFrame([results.metrics]).to_csv(outdir / f'metrics_{ts}.csv', index=False)
    log.info(f"Metrics saved to results/metrics_{ts}.csv")

    if args.plot or out_cfg.get('generate_plots', False):
        plot_results(results)

    return 0

if __name__ == '__main__':
    sys.exit(main())