#!/usr/bin/env python3
"""
Análisis de datos con threshold P80 automático
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import text
from pathlib import Path
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger

log = get_validation_logger()

def add_database_columns():
    """Añadir columnas necesarias"""
    log.info("Adding database columns...")
    
    with db_manager.get_session() as session:
        try:
            session.execute(text("""
                ALTER TABLE orderbook 
                ADD COLUMN IF NOT EXISTS liquidity_quality VARCHAR(20),
                ADD COLUMN IF NOT EXISTS valid_for_trading BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS spread_pct FLOAT,
                ADD COLUMN IF NOT EXISTS threshold_p80 FLOAT
            """))
            log.info("Added columns to orderbook table")
        except Exception as e:
            log.warning(f"Columns may already exist: {e}")

def check_if_already_processed(symbol: str) -> bool:
    """Verificar si el símbolo ya fue procesado"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT COUNT(*) as processed_count
            FROM orderbook 
            WHERE symbol = :symbol 
            AND liquidity_quality IS NOT NULL
        """), {'symbol': symbol}).fetchone()
        
        if result.processed_count > 0:
            log.info(f"{symbol} already processed ({result.processed_count:,} records). Skipping database update.")
            return True
        return False

def load_data(symbol: str):
    """Cargar datos OHLCV y orderbook"""
    log.info(f"Loading data for {symbol}...")
    
    with db_manager.get_session() as session:
        # OHLCV
        ohlcv_query = text("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        ohlcv_df = pd.read_sql(ohlcv_query, session.bind, params={"symbol": symbol}, index_col='timestamp')
        
        # Orderbook
        orderbook_query = text("""
            SELECT timestamp, bid1_price, bid1_size, ask1_price, ask1_size,
                   liquidity_quality, valid_for_trading, spread_pct, threshold_p80
            FROM orderbook 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        orderbook_df = pd.read_sql(orderbook_query, session.bind, params={"symbol": symbol}, index_col='timestamp')
        
        log.info(f"Loaded {len(ohlcv_df):,} OHLCV and {len(orderbook_df):,} orderbook records")
        return ohlcv_df, orderbook_df

def calculate_p80_threshold(orderbook_df: pd.DataFrame) -> float:
    """Calcular threshold basado en percentil 80"""
    
    valid_quotes = orderbook_df.dropna(subset=['bid1_price', 'ask1_price'])
    if len(valid_quotes) == 0:
        return 0.1  # Default fallback
    
    spreads = valid_quotes['ask1_price'] - valid_quotes['bid1_price']
    spread_pct = spreads / valid_quotes['bid1_price'] * 100
    
    # Percentil 80 - el 80% de los datos tendrán spreads menores a este valor
    p80_threshold = spread_pct.quantile(0.80)
    
    return p80_threshold

def analyze_spread_quality(symbol: str, orderbook_df: pd.DataFrame, threshold: float):
    """Analizar calidad de spreads con threshold P80"""
    
    valid_quotes = orderbook_df.dropna(subset=['bid1_price', 'ask1_price'])
    if len(valid_quotes) == 0:
        return None
    
    spreads = valid_quotes['ask1_price'] - valid_quotes['bid1_price']
    spread_pct = spreads / valid_quotes['bid1_price'] * 100
    
    # Aplicar threshold P80
    good_data_mask = spread_pct <= threshold
    good_data_pct = good_data_mask.mean() * 100
    
    # Clasificación basada en P80
    excellent_threshold = threshold * 0.5
    fair_threshold = threshold * 1.5
    
    quality_counts = {
        'excellent': (spread_pct <= excellent_threshold).sum(),
        'good': ((spread_pct > excellent_threshold) & (spread_pct <= threshold)).sum(),
        'fair': ((spread_pct > threshold) & (spread_pct <= fair_threshold)).sum(),
        'poor': (spread_pct > fair_threshold).sum()
    }
    
    total_records = len(spread_pct)
    quality_pct = {k: v/total_records*100 for k, v in quality_counts.items()}
    
    log.info(f"{symbol} Spread Analysis:")
    log.info(f"  P80 threshold: {threshold:.4f}%")
    log.info(f"  Average spread: {spread_pct.mean():.4f}%")
    log.info(f"  Median spread: {spread_pct.median():.4f}%")
    log.info(f"  Good data (≤P80): {good_data_pct:.1f}%")
    log.info(f"  Quality - Excellent: {quality_pct['excellent']:.1f}%, Good: {quality_pct['good']:.1f}%, Fair: {quality_pct['fair']:.1f}%, Poor: {quality_pct['poor']:.1f}%")
    
    return {
        'symbol': symbol,
        'threshold': threshold,
        'spread_pct': spread_pct,
        'good_data_pct': good_data_pct,
        'quality_counts': quality_counts,
        'quality_pct': quality_pct,
        'total_records': total_records,
        'basic_stats': {
            'mean': spread_pct.mean(),
            'median': spread_pct.median(),
            'p25': spread_pct.quantile(0.25),
            'p50': spread_pct.quantile(0.50),
            'p75': spread_pct.quantile(0.75),
            'p80': threshold,
            'p95': spread_pct.quantile(0.95)
        }
    }

def update_database_quality(symbol: str, threshold: float):
    """Actualizar base de datos con información de calidad usando P80"""
    log.info(f"Updating database quality for {symbol} with P80 threshold {threshold:.4f}%...")
    
    with db_manager.get_session() as session:
        # 1. Calcular spread_pct y guardar threshold P80
        session.execute(text("""
            UPDATE orderbook 
            SET spread_pct = ((ask1_price - bid1_price) / bid1_price * 100),
                threshold_p80 = :threshold
            WHERE symbol = :symbol 
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            AND bid1_price > 0
        """), {'symbol': symbol, 'threshold': threshold})
        
        # 2. Clasificar basado en P80
        excellent_threshold = threshold * 0.5
        fair_threshold = threshold * 1.5
        
        # Excellent (≤50% del P80)
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = 'Excellent', valid_for_trading = TRUE
            WHERE symbol = :symbol AND spread_pct <= :excellent_threshold
        """), {'symbol': symbol, 'excellent_threshold': excellent_threshold})
        
        # Good (50% del P80 < spread ≤ P80)
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = 'Good', valid_for_trading = TRUE
            WHERE symbol = :symbol 
            AND spread_pct > :excellent_threshold 
            AND spread_pct <= :threshold
        """), {'symbol': symbol, 'excellent_threshold': excellent_threshold, 'threshold': threshold})
        
        # Fair (P80 < spread ≤ 150% del P80)
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = 'Fair', valid_for_trading = FALSE
            WHERE symbol = :symbol 
            AND spread_pct > :threshold 
            AND spread_pct <= :fair_threshold
        """), {'symbol': symbol, 'threshold': threshold, 'fair_threshold': fair_threshold})
        
        # Poor (>150% del P80)
        session.execute(text("""
            UPDATE orderbook 
            SET liquidity_quality = 'Poor', valid_for_trading = FALSE
            WHERE symbol = :symbol 
            AND spread_pct > :fair_threshold
        """), {'symbol': symbol, 'fair_threshold': fair_threshold})
        
        # Verificar resultados
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN valid_for_trading THEN 1 END) as good_trading,
                COUNT(CASE WHEN liquidity_quality = 'Excellent' THEN 1 END) as excellent,
                COUNT(CASE WHEN liquidity_quality = 'Good' THEN 1 END) as good,
                COUNT(CASE WHEN liquidity_quality = 'Fair' THEN 1 END) as fair,
                COUNT(CASE WHEN liquidity_quality = 'Poor' THEN 1 END) as poor
            FROM orderbook 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        log.info(f"Database updated for {symbol}:")
        log.info(f"  Total records: {result.total:,}")
        log.info(f"  Valid for trading (≤P80): {result.good_trading:,} ({result.good_trading/result.total*100:.1f}%)")
        log.info(f"  Excellent: {result.excellent:,}, Good: {result.good:,}, Fair: {result.fair:,}, Poor: {result.poor:,}")

def create_spread_analysis_plot(analysis: Dict):
    """Crear gráfico de análisis de spreads"""
    
    symbol = analysis['symbol']
    spread_pct = analysis['spread_pct']
    threshold = analysis['threshold']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribución de spreads
    axes[0, 0].hist(spread_pct, bins=100, alpha=0.7, edgecolor='black', density=True)
    axes[0, 0].axvline(threshold, color='red', linestyle='-', linewidth=2, 
                      label=f'P80 Threshold: {threshold:.4f}%')
    axes[0, 0].axvline(analysis['basic_stats']['median'], color='blue', linestyle='--', 
                      label=f'Median: {analysis["basic_stats"]["median"]:.4f}%')
    
    axes[0, 0].set_title(f'{symbol} - Spread Distribution')
    axes[0, 0].set_xlabel('Spread %')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. Percentiles con P80 marcado
    percentiles = np.arange(1, 101)
    spread_values = np.percentile(spread_pct.values, percentiles)
    
    axes[0, 1].plot(percentiles, spread_values, linewidth=2)
    axes[0, 1].axhline(threshold, color='red', linestyle='-', 
                      label=f'P80: {threshold:.4f}%')
    axes[0, 1].axvline(80, color='red', linestyle='--', alpha=0.7)
    
    axes[0, 1].set_title(f'{symbol} - Percentile Analysis')
    axes[0, 1].set_xlabel('Percentile')
    axes[0, 1].set_ylabel('Spread %')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Data availability vs threshold
    thresholds_range = np.linspace(spread_pct.quantile(0.01), spread_pct.quantile(0.99), 50)
    data_availability = [(spread_pct <= t).mean() * 100 for t in thresholds_range]
    
    axes[1, 0].plot(thresholds_range, data_availability, linewidth=2)
    axes[1, 0].axvline(threshold, color='red', linestyle='-', 
                      label=f'P80: {analysis["good_data_pct"]:.1f}%')
    axes[1, 0].set_title(f'{symbol} - Data Availability vs Threshold')
    axes[1, 0].set_xlabel('Threshold %')
    axes[1, 0].set_ylabel('Data Availability %')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Quality pie chart
    quality_labels = list(analysis['quality_pct'].keys())
    quality_values = list(analysis['quality_pct'].values())
    colors = ['green', 'lightgreen', 'yellow', 'red']
    
    axes[1, 1].pie(quality_values, labels=quality_labels, colors=colors, 
                   autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title(f'{symbol} - Quality Distribution (P80 Based)')
    
    plt.tight_layout()
    
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{symbol}_spread_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_ohlcv_analysis_plot(symbol: str, ohlcv_df: pd.DataFrame):
    """Análisis OHLCV simplificado"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Evolución de precios
    daily_prices = ohlcv_df.groupby(ohlcv_df.index.date)['close'].last()
    axes[0, 0].plot(daily_prices.index, daily_prices.values, linewidth=1, alpha=0.8)
    axes[0, 0].set_title(f'{symbol} - Price Evolution')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Distribución de volumen
    real_volume = ohlcv_df['volume'][ohlcv_df['volume'] > 0]
    if len(real_volume) > 0:
        axes[0, 1].hist(real_volume, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'{symbol} - Volume Distribution')
        axes[0, 1].set_xlabel('Volume')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
    
    # 3. Distribución de returns
    returns = ohlcv_df['close'].pct_change().dropna()
    if len(returns) > 0:
        axes[1, 0].hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title(f'{symbol} - Return Distribution')
        axes[1, 0].set_xlabel('1-minute Returns %')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        
        vol = returns.std() * 100
        axes[1, 0].text(0.7, 0.9, f'Volatility: {vol:.3f}%', 
                       transform=axes[1, 0].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 4. Patrones horarios
    hourly_volume = ohlcv_df.groupby(ohlcv_df.index.hour)['volume'].mean()
    axes[1, 1].bar(hourly_volume.index, hourly_volume.values, alpha=0.7)
    axes[1, 1].set_title(f'{symbol} - Hourly Volume Pattern')
    axes[1, 1].set_xlabel('Hour (UTC)')
    axes[1, 1].set_ylabel('Average Volume')
    axes[1, 1].set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    plots_dir = Path("plots")
    plt.savefig(plots_dir / f'{symbol}_ohlcv_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plot(analyses: Dict):
    """Crear gráfico de comparación usando P80"""
    
    if len(analyses) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    symbols = []
    thresholds = []
    good_data_pcts = []
    avg_spreads = []
    
    for symbol, analysis in analyses.items():
        if analysis:
            symbol_short = symbol.split('_')[-2]
            symbols.append(symbol_short)
            thresholds.append(analysis['threshold'])
            good_data_pcts.append(analysis['good_data_pct'])
            avg_spreads.append(analysis['basic_stats']['mean'])
    
    colors = ['blue', 'orange', 'green', 'red'][:len(symbols)]
    
    # 1. P80 Thresholds
    bars1 = axes[0, 0].bar(symbols, thresholds, color=colors, alpha=0.7)
    axes[0, 0].set_title('P80 Thresholds (Auto-calculated)')
    axes[0, 0].set_ylabel('Threshold %')
    for bar, value in zip(bars1, thresholds):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}%', ha='center', va='bottom')
    
    # 2. Good data availability (exactly 80%)
    bars2 = axes[0, 1].bar(symbols, good_data_pcts, color=colors, alpha=0.7)
    axes[0, 1].set_title('Good Data Availability (≤P80)')
    axes[0, 1].set_ylabel('Availability %')
    axes[0, 1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Expected 80%')
    axes[0, 1].legend()
    for bar, value in zip(bars2, good_data_pcts):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom')
    
    # 3. Average spreads vs P80 thresholds
    x_pos = range(len(symbols))
    width = 0.35
    bars3 = axes[1, 0].bar([x - width/2 for x in x_pos], avg_spreads, width, 
                          color=colors, alpha=0.7, label='Avg Spread')
    bars4 = axes[1, 0].bar([x + width/2 for x in x_pos], thresholds, width,
                          color=colors, alpha=0.3, label='P80 Threshold')
    axes[1, 0].set_title('Average Spread vs P80 Threshold')
    axes[1, 0].set_ylabel('Spread %')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(symbols)
    axes[1, 0].legend()
    
    # 4. Summary table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = []
    for i, symbol in enumerate(symbols):
        table_data.append([
            symbol,
            f"{thresholds[i]:.4f}%",
            f"{good_data_pcts[i]:.1f}%",
            f"{avg_spreads[i]:.4f}%"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                           colLabels=['Symbol', 'P80 Threshold', 'Good Data %', 'Avg Spread'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('P80 Analysis Summary')
    
    plt.tight_layout()
    
    plots_dir = Path("plots")
    plt.savefig(plots_dir / 'symbols_comparison_p80.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_ohlcv_quality(symbol: str, ohlcv_df: pd.DataFrame):
    """Análisis básico de calidad OHLCV"""
    
    total_records = len(ohlcv_df)
    
    # Validaciones básicas
    invalid_ohlc = (
        (ohlcv_df['high'] < ohlcv_df['open']) |
        (ohlcv_df['high'] < ohlcv_df['close']) |
        (ohlcv_df['low'] > ohlcv_df['open']) |
        (ohlcv_df['low'] > ohlcv_df['close']) |
        (ohlcv_df['high'] < ohlcv_df['low'])
    ).sum()
    
    zero_volume = (ohlcv_df['volume'] == 0).sum()
    
    # Movimientos extremos
    price_changes = ohlcv_df['close'].pct_change().dropna()
    extreme_moves = (abs(price_changes) > 0.5).sum()
    
    log.info(f"{symbol} OHLCV Quality:")
    log.info(f"  Total records: {total_records:,}")
    log.info(f"  Invalid OHLC: {invalid_ohlc:,} ({invalid_ohlc/total_records*100:.3f}%)")
    log.info(f"  Zero volume: {zero_volume:,} ({zero_volume/total_records*100:.3f}%)")
    log.info(f"  Extreme moves: {extreme_moves:,} ({extreme_moves/len(price_changes)*100:.3f}%)")

def main():
    """Función principal con P80 automático"""
    log.info("Starting P80-based data analysis...")
    
    symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
    analyses = {}
    
    try:
        # Preparar base de datos
        add_database_columns()
        
        # Analizar cada símbolo
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"ANALYZING {symbol}")
            log.info(f"{'='*60}")
            
            # Verificar si ya fue procesado
            already_processed = check_if_already_processed(symbol)
            
            # Cargar datos
            ohlcv_df, orderbook_df = load_data(symbol)
            
            if len(ohlcv_df) == 0 or len(orderbook_df) == 0:
                log.warning(f"No data for {symbol}")
                continue
            
            # Análisis básico OHLCV
            analyze_ohlcv_quality(symbol, ohlcv_df)
            
            # Calcular threshold P80
            p80_threshold = calculate_p80_threshold(orderbook_df)
            log.info(f"Calculated P80 threshold: {p80_threshold:.4f}%")
            
            # Análisis de spreads
            analysis = analyze_spread_quality(symbol, orderbook_df, p80_threshold)
            
            if analysis:
                analyses[symbol] = analysis
                
                # Crear gráficos
                create_spread_analysis_plot(analysis)
                create_ohlcv_analysis_plot(symbol, ohlcv_df)
                
                # Actualizar base de datos solo si no fue procesado antes
                if not already_processed:
                    update_database_quality(symbol, p80_threshold)
                else:
                    log.info(f"Skipping database update for {symbol} (already processed)")
        
        # Crear comparación
        if len(analyses) > 1:
            log.info("\n=== CREATING P80 COMPARISON ===")
            create_comparison_plot(analyses)
        
        # Reporte final
        log.info(f"\n{'='*80}")
        log.info("FINAL P80 ANALYSIS REPORT")
        log.info(f"{'='*80}")
        
        for symbol, analysis in analyses.items():
            symbol_short = symbol.split('_')[-2]
            log.info(f"\n{symbol_short}:")
            log.info(f"  P80 Threshold: {analysis['threshold']:.4f}%")
            log.info(f"  Good Data (≤P80): {analysis['good_data_pct']:.1f}%")
            log.info(f"  Average Spread: {analysis['basic_stats']['mean']:.4f}%")
            log.info(f"  Median Spread: {analysis['basic_stats']['median']:.4f}%")
            
            if analysis['good_data_pct'] >= 75:
                recommendation = "EXCELLENT for trading"
            elif analysis['good_data_pct'] >= 60:
                recommendation = "GOOD for trading" 
            else:
                recommendation = "NEEDS REVIEW"
            
            log.info(f"  Recommendation: {recommendation}")
        
        log.info(f"\nUse this filter for optimal trading: WHERE valid_for_trading = TRUE")
        log.info(f"This gives you data with spreads ≤ P80 threshold")
        log.info(f"Plots saved in: plots/")
        
        return True
        
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)