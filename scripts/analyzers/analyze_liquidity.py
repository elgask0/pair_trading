#!/usr/bin/env python3
"""
Liquidity analysis script - SIMPLIFIED VERSION
Analyzes market liquidity using orderbook data with focus on practical metrics
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Optional, Tuple

 # Ensure project root is on PYTHONPATH for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

# Set matplotlib style
plt.style.use("default")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_orderbook_data(symbol: str, days: int = 7) -> pd.DataFrame:
    """Load recent orderbook data for analysis"""
    log.info(f"Loading orderbook data for {symbol} (last {days} days)...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    with db_manager.get_session() as session:
        query = text(
            """
            SELECT 
                timestamp,
                bid1_price, bid1_size, bid2_price, bid2_size, bid3_price, bid3_size,
                bid4_price, bid4_size, bid5_price, bid5_size,
                ask1_price, ask1_size, ask2_price, ask2_size, ask3_price, ask3_size,
                ask4_price, ask4_size, ask5_price, ask5_size
            FROM orderbook 
            WHERE symbol = :symbol 
            AND timestamp >= :start_date 
            AND timestamp <= :end_date
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            ORDER BY timestamp
        """
        )

        df = pd.read_sql(
            query,
            session.bind,
            params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
            index_col="timestamp",
        )

        log.info(f"Loaded {len(df):,} orderbook snapshots")
        return df


def calculate_simple_slippage(
    row: pd.Series, order_size_usd: float, side: str = "buy"
) -> Dict:
    """
    Calculate slippage for a given order size using top 5 levels
    Simplified version focusing on practical order sizes
    """

    levels = []
    prefix = "ask" if side == "buy" else "bid"

    # Extract top 5 levels only
    for i in range(1, 6):
        price = row.get(f"{prefix}{i}_price")
        size = row.get(f"{prefix}{i}_size")

        if pd.notna(price) and pd.notna(size) and price > 0 and size > 0:
            levels.append({"price": price, "size": size, "value_usd": price * size})

    if not levels:
        return {"can_execute": False, "slippage_pct": None, "levels_used": 0}

    # Sort by price (best first)
    levels.sort(key=lambda x: x["price"], reverse=(side == "sell"))

    # Calculate execution
    remaining_usd = order_size_usd
    total_value = 0
    total_coins = 0
    levels_used = 0

    for level in levels:
        if remaining_usd <= 0:
            break

        available_usd = level["value_usd"]

        if available_usd >= remaining_usd:
            # This level can fill the remaining order
            coins_bought = remaining_usd / level["price"]
            total_value += remaining_usd
            total_coins += coins_bought
            levels_used += 1
            remaining_usd = 0
        else:
            # Use entire level
            total_value += available_usd
            total_coins += level["size"]
            levels_used += 1
            remaining_usd -= available_usd

    if remaining_usd > 0:
        return {"can_execute": False, "slippage_pct": None, "levels_used": levels_used}

    # Calculate slippage
    avg_price = total_value / total_coins
    best_price = levels[0]["price"]
    slippage_pct = abs(avg_price - best_price) / best_price * 100

    return {
        "can_execute": True,
        "slippage_pct": slippage_pct,
        "avg_price": avg_price,
        "best_price": best_price,
        "levels_used": levels_used,
    }


def analyze_liquidity_metrics(symbol: str, orderbook_df: pd.DataFrame) -> Dict:
    """Analyze key liquidity metrics"""
    log.info(f"Analyzing liquidity metrics for {symbol}...")

    # Standard order sizes to test
    order_sizes = [100, 500, 1000, 2000, 5000]  # USD

    results = {
        "timestamps": orderbook_df.index.tolist(),
        "slippage_analysis": {},
        "spread_analysis": {},
        "depth_analysis": {},
    }

    # Calculate slippage for different order sizes
    for size in order_sizes:
        buy_slippages = []
        sell_slippages = []
        execution_rates = []

        for idx, row in orderbook_df.iterrows():
            # Buy side
            buy_result = calculate_simple_slippage(row, size, "buy")
            if buy_result["can_execute"]:
                buy_slippages.append(buy_result["slippage_pct"])

            # Sell side
            sell_result = calculate_simple_slippage(row, size, "sell")
            if sell_result["can_execute"]:
                sell_slippages.append(sell_result["slippage_pct"])

            # Execution rate (can execute both sides)
            can_execute_both = buy_result["can_execute"] and sell_result["can_execute"]
            execution_rates.append(can_execute_both)

        results["slippage_analysis"][size] = {
            "buy_slippage_mean": np.mean(buy_slippages) if buy_slippages else None,
            "buy_slippage_p95": (
                np.percentile(buy_slippages, 95) if buy_slippages else None
            ),
            "sell_slippage_mean": np.mean(sell_slippages) if sell_slippages else None,
            "sell_slippage_p95": (
                np.percentile(sell_slippages, 95) if sell_slippages else None
            ),
            "execution_rate": np.mean(execution_rates) * 100,
        }

    # Calculate spreads
    valid_spreads = []
    for idx, row in orderbook_df.iterrows():
        bid = row.get("bid1_price")
        ask = row.get("ask1_price")
        if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
            spread_pct = (ask - bid) / bid * 100
            if spread_pct < 10:  # Reasonable spread
                valid_spreads.append(spread_pct)

    results["spread_analysis"] = {
        "mean_spread": np.mean(valid_spreads) if valid_spreads else None,
        "median_spread": np.median(valid_spreads) if valid_spreads else None,
        "p95_spread": np.percentile(valid_spreads, 95) if valid_spreads else None,
    }

    # Calculate depth (top 5 levels)
    total_depths = []
    for idx, row in orderbook_df.iterrows():
        bid_depth = 0
        ask_depth = 0

        for i in range(1, 6):
            # Bid depth
            bid_price = row.get(f"bid{i}_price")
            bid_size = row.get(f"bid{i}_size")
            if (
                pd.notna(bid_price)
                and pd.notna(bid_size)
                and bid_price > 0
                and bid_size > 0
            ):
                bid_depth += bid_price * bid_size

            # Ask depth
            ask_price = row.get(f"ask{i}_price")
            ask_size = row.get(f"ask{i}_size")
            if (
                pd.notna(ask_price)
                and pd.notna(ask_size)
                and ask_price > 0
                and ask_size > 0
            ):
                ask_depth += ask_price * ask_size

        total_depths.append(bid_depth + ask_depth)

    results["depth_analysis"] = {
        "mean_depth_usd": np.mean(total_depths) if total_depths else 0,
        "median_depth_usd": np.median(total_depths) if total_depths else 0,
        "min_depth_usd": np.min(total_depths) if total_depths else 0,
    }

    return results


def create_liquidity_summary_plot(symbol: str, analysis_results: Dict):
    """Create a focused liquidity analysis plot"""
    log.info(f"Creating liquidity summary plot for {symbol}...")

    symbol_short = symbol.split("_")[-2] if "_" in symbol else symbol

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"{symbol_short} - Liquidity Analysis Summary", fontsize=16, fontweight="bold"
    )

    # 1. Slippage by Order Size
    order_sizes = list(analysis_results["slippage_analysis"].keys())
    buy_slippages = [
        analysis_results["slippage_analysis"][size]["buy_slippage_mean"]
        for size in order_sizes
    ]
    sell_slippages = [
        analysis_results["slippage_analysis"][size]["sell_slippage_mean"]
        for size in order_sizes
    ]

    # Filter out None values
    valid_sizes = []
    valid_buy_slippages = []
    valid_sell_slippages = []

    for i, size in enumerate(order_sizes):
        if buy_slippages[i] is not None and sell_slippages[i] is not None:
            valid_sizes.append(size)
            valid_buy_slippages.append(buy_slippages[i])
            valid_sell_slippages.append(sell_slippages[i])

    if valid_sizes:
        ax1.plot(
            valid_sizes,
            valid_buy_slippages,
            "o-",
            label="Buy",
            color="green",
            linewidth=2,
        )
        ax1.plot(
            valid_sizes,
            valid_sell_slippages,
            "s-",
            label="Sell",
            color="red",
            linewidth=2,
        )
        ax1.set_xlabel("Order Size (USD)")
        ax1.set_ylabel("Average Slippage %")
        ax1.set_title("Slippage vs Order Size")
        ax1.set_xscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Execution Success Rate
    execution_rates = [
        analysis_results["slippage_analysis"][size]["execution_rate"]
        for size in order_sizes
    ]

    valid_execution = [rate for rate in execution_rates if rate is not None]
    if valid_execution and len(valid_execution) == len(order_sizes):
        colors = [
            "green" if rate >= 95 else "orange" if rate >= 80 else "red"
            for rate in execution_rates
        ]
        bars = ax2.bar(
            [f"${s}" for s in order_sizes], execution_rates, color=colors, alpha=0.7
        )
        ax2.set_ylabel("Success Rate %")
        ax2.set_title("Order Execution Success Rate")
        ax2.set_ylim(0, 105)

        # Add value labels
        for bar, rate in zip(bars, execution_rates):
            if rate is not None:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # 3. Spread Statistics
    spread_stats = analysis_results["spread_analysis"]
    if spread_stats["mean_spread"] is not None:
        metrics = ["Mean", "Median", "P95"]
        values = [
            spread_stats["mean_spread"],
            spread_stats["median_spread"],
            spread_stats["p95_spread"],
        ]

        bars = ax3.bar(metrics, values, color=["blue", "orange", "red"], alpha=0.7)
        ax3.set_ylabel("Spread %")
        ax3.set_title("Bid-Ask Spread Statistics")

        # Add value labels
        for bar, value in zip(bars, values):
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + bar.get_height() * 0.01,
                f"{value:.4f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 4. Liquidity Summary Table
    ax4.axis("off")

    # Create summary data
    depth_stats = analysis_results["depth_analysis"]

    summary_data = [
        ["Metric", "Value"],
        [
            "Avg Spread",
            (
                f"{spread_stats['mean_spread']:.4f}%"
                if spread_stats["mean_spread"]
                else "N/A"
            ),
        ],
        [
            "P95 Spread",
            (
                f"{spread_stats['p95_spread']:.4f}%"
                if spread_stats["p95_spread"]
                else "N/A"
            ),
        ],
        [
            "Avg Depth",
            (
                f"${depth_stats['mean_depth_usd']:,.0f}"
                if depth_stats["mean_depth_usd"]
                else "N/A"
            ),
        ],
        [
            "Min Depth",
            (
                f"${depth_stats['min_depth_usd']:,.0f}"
                if depth_stats["min_depth_usd"]
                else "N/A"
            ),
        ],
    ]

    # Add slippage for $1000 orders
    if 1000 in analysis_results["slippage_analysis"]:
        slippage_1k = analysis_results["slippage_analysis"][1000]
        if slippage_1k["buy_slippage_mean"] is not None:
            summary_data.append(
                ["$1K Buy Slippage", f"{slippage_1k['buy_slippage_mean']:.4f}%"]
            )
        if slippage_1k["execution_rate"] is not None:
            summary_data.append(
                ["$1K Success Rate", f"{slippage_1k['execution_rate']:.1f}%"]
            )

    # Create table
    table = ax4.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Style table
    table[(0, 0)].set_facecolor("#3498db")
    table[(0, 1)].set_facecolor("#3498db")
    table[(0, 0)].set_text_props(weight="bold", color="white")
    table[(0, 1)].set_text_props(weight="bold", color="white")

    ax4.set_title("Liquidity Summary", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(
        plots_dir / f"{symbol_short}_liquidity_summary.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    log.info(
        f"Liquidity summary plot saved: plots/{symbol_short}_liquidity_summary.png"
    )


def generate_liquidity_report(symbol: str, analysis_results: Dict):
    """Generate concise liquidity report"""
    log.info(f"\n{'='*60}")
    log.info(f"LIQUIDITY ANALYSIS REPORT - {symbol}")
    log.info(f"{'='*60}")

    symbol_short = symbol.split("_")[-2] if "_" in symbol else symbol

    # Spread analysis
    spread_stats = analysis_results["spread_analysis"]
    log.info(f"\n📊 SPREAD ANALYSIS:")
    if spread_stats["mean_spread"] is not None:
        log.info(f"  Average Spread: {spread_stats['mean_spread']:.4f}%")
        log.info(f"  Median Spread: {spread_stats['median_spread']:.4f}%")
        log.info(f"  P95 Spread: {spread_stats['p95_spread']:.4f}%")
    else:
        log.warning("  No valid spread data found")

    # Slippage analysis
    log.info(f"\n💹 SLIPPAGE ANALYSIS:")
    log.info(f"{'Size':<8} {'Buy Slip %':<12} {'Sell Slip %':<12} {'Success %':<10}")
    log.info("-" * 45)

    for size, stats in analysis_results["slippage_analysis"].items():
        buy_slip = (
            f"{stats['buy_slippage_mean']:.4f}" if stats["buy_slippage_mean"] else "N/A"
        )
        sell_slip = (
            f"{stats['sell_slippage_mean']:.4f}"
            if stats["sell_slippage_mean"]
            else "N/A"
        )
        success = f"{stats['execution_rate']:.1f}" if stats["execution_rate"] else "N/A"

        log.info(f"${size:<7} {buy_slip:<12} {sell_slip:<12} {success:<10}")

    # Depth analysis
    depth_stats = analysis_results["depth_analysis"]
    log.info(f"\n📈 MARKET DEPTH:")
    log.info(f"  Average Depth (top 5 levels): ${depth_stats['mean_depth_usd']:,.0f}")
    log.info(f"  Median Depth: ${depth_stats['median_depth_usd']:,.0f}")
    log.info(f"  Minimum Depth: ${depth_stats['min_depth_usd']:,.0f}")

    # Overall assessment
    log.info(f"\n🏆 LIQUIDITY ASSESSMENT:")

    # Simple scoring based on key metrics
    score = 0
    max_score = 4

    # Spread quality (25%)
    if spread_stats["mean_spread"] and spread_stats["mean_spread"] < 0.1:
        score += 1
        spread_grade = "Excellent"
    elif spread_stats["mean_spread"] and spread_stats["mean_spread"] < 0.2:
        score += 0.5
        spread_grade = "Good"
    else:
        spread_grade = "Poor"

    # $1000 order execution (25%)
    if 1000 in analysis_results["slippage_analysis"]:
        success_rate = analysis_results["slippage_analysis"][1000]["execution_rate"]
        if success_rate and success_rate >= 95:
            score += 1
            execution_grade = "Excellent"
        elif success_rate and success_rate >= 80:
            score += 0.5
            execution_grade = "Good"
        else:
            execution_grade = "Poor"
    else:
        execution_grade = "Unknown"

    # $1000 slippage quality (25%)
    if 1000 in analysis_results["slippage_analysis"]:
        slippage = analysis_results["slippage_analysis"][1000]["buy_slippage_mean"]
        if slippage and slippage < 0.1:
            score += 1
            slippage_grade = "Excellent"
        elif slippage and slippage < 0.3:
            score += 0.5
            slippage_grade = "Good"
        else:
            slippage_grade = "Poor"
    else:
        slippage_grade = "Unknown"

    # Depth quality (25%)
    if depth_stats["mean_depth_usd"] >= 10000:
        score += 1
        depth_grade = "Excellent"
    elif depth_stats["mean_depth_usd"] >= 5000:
        score += 0.5
        depth_grade = "Good"
    else:
        depth_grade = "Poor"

    final_score = (score / max_score) * 100

    if final_score >= 80:
        overall_grade = "A (Excellent for trading)"
    elif final_score >= 60:
        overall_grade = "B (Good for trading)"
    elif final_score >= 40:
        overall_grade = "C (Fair - use caution)"
    else:
        overall_grade = "D (Poor - not recommended)"

    log.info(f"  Overall Liquidity Grade: {overall_grade}")
    log.info(f"  Component Grades:")
    log.info(f"    Spread Quality: {spread_grade}")
    log.info(f"    Execution Success: {execution_grade}")
    log.info(f"    Slippage Quality: {slippage_grade}")
    log.info(f"    Market Depth: {depth_grade}")

    # Trading recommendations
    log.info(f"\n📋 TRADING RECOMMENDATIONS:")
    if final_score >= 80:
        log.info(f"  ✅ {symbol_short} is excellent for algorithmic trading")
        log.info(f"  ✅ Can handle orders up to $1000+ with minimal slippage")
        log.info(f"  ✅ Suitable for high-frequency strategies")
    elif final_score >= 60:
        log.info(f"  ✅ {symbol_short} is suitable for most trading strategies")
        log.info(f"  ⚠️ Consider limiting order sizes to $500-1000")
        log.info(f"  ⚠️ Monitor slippage during execution")
    elif final_score >= 40:
        log.info(f"  ⚠️ {symbol_short} requires careful order management")
        log.info(f"  ⚠️ Limit orders to $200-500 range")
        log.info(f"  ⚠️ Use limit orders instead of market orders")
    else:
        log.info(f"  ❌ {symbol_short} has poor liquidity")
        log.info(f"  ❌ High slippage risk for any meaningful size")
        log.info(f"  ❌ Consider alternative instruments")


def main():
    """Main simplified liquidity analysis function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze market liquidity (simplified version)"
    )
    parser.add_argument("--symbol", type=str, help="Specific symbol to analyze")
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days to analyze (default: 7)"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    log.info("Starting simplified liquidity analysis...")

    # Get symbols to analyze
    if args.symbol:
        symbols = [args.symbol]
    else:
        try:
            active_pairs = settings.get_active_pairs()
            symbols = []
            for pair in active_pairs:
                symbols.extend([pair.symbol1, pair.symbol2])
            symbols = list(set(symbols))
        except Exception as e:
            log.error(f"Could not load symbols from config: {e}")
            symbols = ["MEXCFTS_PERP_GIGA_USDT", "MEXCFTS_PERP_SPX_USDT"]

    if not symbols:
        log.error("No symbols to analyze")
        return False

    log.info(f"Analyzing liquidity for {len(symbols)} symbols (last {args.days} days)")

    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    try:
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"ANALYZING {symbol}")
            log.info(f"{'='*60}")

            # Load data
            orderbook_df = load_orderbook_data(symbol, args.days)

            if len(orderbook_df) == 0:
                log.warning(f"No orderbook data for {symbol}")
                continue

            # Analyze liquidity
            analysis_results = analyze_liquidity_metrics(symbol, orderbook_df)

            # Generate plots
            if not args.no_plots:
                create_liquidity_summary_plot(symbol, analysis_results)

            # Generate report
            generate_liquidity_report(symbol, analysis_results)

        log.info(f"\n🎉 Liquidity analysis completed!")
        log.info(f"Check plots/ directory for visual summaries")

        return True

    except Exception as e:
        log.error(f"Liquidity analysis failed: {e}")
        import traceback

        log.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
