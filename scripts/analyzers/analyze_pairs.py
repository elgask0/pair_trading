#!/usr/bin/env python3
"""
📊 Pair Analysis - DEBUG ALIGNMENT ISSUES
Análisis completo con logs detallados para entender el problema de alineamiento
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
import argparse
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger

log = get_validation_logger()

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def check_table_columns(table_name: str) -> dict:
    """EXACTAMENTE la misma función que analyze_markprices.py"""
    with db_manager.get_session() as session:
        try:
            result = session.execute(text(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY column_name
            """)).fetchall()
            
            columns = {row.column_name: row.data_type for row in result}
            log.info(f"Table {table_name}: {len(columns)} columns found")
            return columns
            
        except Exception as e:
            log.error(f"Error checking table {table_name}: {e}")
            return {}

def load_mark_prices_data(symbol: str, sample_rate: int = 10) -> pd.DataFrame:
    """EXACTAMENTE la misma función que analyze_markprices.py"""
    with db_manager.get_session() as session:
        # Verificar qué columnas existen primero
        mp_columns = check_table_columns('mark_prices')
        
        # Primero verificar cuántos registros hay
        count_result = session.execute(text("""
            SELECT COUNT(*) as total FROM mark_prices WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        total_records = count_result.total
        
        # Determinar sampling
        if total_records > 100000:
            sample_clause = f"WHERE rn % {sample_rate} = 1"
            log.info(f"Sampling 1 of every {sample_rate} records ({total_records:,} total)")
        else:
            sample_clause = ""
            log.info(f"Loading all {total_records:,} records")
        
        # Construir SELECT basado en columnas disponibles
        select_columns = ['timestamp', 'mark_price', 'orderbook_mid']
        
        if 'ohlcv_volume' in mp_columns:
            select_columns.append('ohlcv_volume')
        
        if 'liquidity_score' in mp_columns:
            select_columns.append('liquidity_score')
        
        if 'valid_for_trading' in mp_columns:
            select_columns.append('valid_for_trading')
        elif 'is_valid' in mp_columns:
            select_columns.append('is_valid as valid_for_trading')
        
        query = text(f"""
            SELECT 
                {', '.join(select_columns)}
            FROM (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (ORDER BY timestamp) as rn
                FROM mark_prices 
                WHERE symbol = :symbol
            ) t
            {sample_clause}
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params={'symbol': symbol}, index_col='timestamp')
        return df

def debug_timestamp_analysis(df1: pd.DataFrame, df2: pd.DataFrame, s1: str, s2: str):
    """Análisis detallado de timestamps para debug"""
    log.info(f"\n🔍 DEBUG TIMESTAMP ANALYSIS:")
    
    # Análisis de timestamps originales
    ts1 = df1.index
    ts2 = df2.index
    
    log.info(f"  {s1} timestamps:")
    log.info(f"    Count: {len(ts1):,}")
    log.info(f"    Range: {ts1.min()} to {ts1.max()}")
    log.info(f"    Sample timestamps:")
    for i in range(min(5, len(ts1))):
        log.info(f"      {i+1}: {ts1[i]}")
    
    log.info(f"  {s2} timestamps:")
    log.info(f"    Count: {len(ts2):,}")
    log.info(f"    Range: {ts2.min()} to {ts2.max()}")
    log.info(f"    Sample timestamps:")
    for i in range(min(5, len(ts2))):
        log.info(f"      {i+1}: {ts2[i]}")
    
    # Análisis de intervalos
    if len(ts1) > 1:
        intervals1 = ts1.to_series().diff().dropna()
        log.info(f"  {s1} intervals:")
        log.info(f"    Mean: {intervals1.mean()}")
        log.info(f"    Median: {intervals1.median()}")
        log.info(f"    Min: {intervals1.min()}")
        log.info(f"    Max: {intervals1.max()}")
    
    if len(ts2) > 1:
        intervals2 = ts2.to_series().diff().dropna()
        log.info(f"  {s2} intervals:")
        log.info(f"    Mean: {intervals2.mean()}")
        log.info(f"    Median: {intervals2.median()}")
        log.info(f"    Min: {intervals2.min()}")
        log.info(f"    Max: {intervals2.max()}")
    
    # Buscar timestamps exactos comunes
    exact_common = ts1.intersection(ts2)
    log.info(f"  Exact common timestamps: {len(exact_common):,}")
    
    if len(exact_common) > 0:
        log.info(f"    Sample common timestamps:")
        for i in range(min(5, len(exact_common))):
            log.info(f"      {i+1}: {exact_common[i]}")

def align_price_data_improved(df1: pd.DataFrame, df2: pd.DataFrame, s1: str, s2: str) -> Tuple[pd.Series, pd.Series]:
    """Alineamiento mejorado de datos de precios con debug detallado"""
    log.info(f"\n🔄 IMPROVED PRICE ALIGNMENT: {s1} vs {s2}")
    
    # Debug inicial
    debug_timestamp_analysis(df1, df2, s1, s2)
    
    prices1 = df1['mark_price']
    prices2 = df2['mark_price']
    
    # PASO 1: Intentar timestamps exactos
    log.info(f"\n📊 STEP 1: Exact timestamp alignment")
    exact_common = prices1.index.intersection(prices2.index)
    log.info(f"  Exact matches: {len(exact_common):,}")
    
    if len(exact_common) > 100:  # Si tenemos suficientes matches exactos
        log.info(f"  ✅ Using exact timestamp alignment")
        return prices1.loc[exact_common], prices2.loc[exact_common]
    
    # PASO 2: Redondear a minutos
    log.info(f"\n📊 STEP 2: Rounding to minutes")
    
    # Crear copias para redondeo
    df1_rounded = df1.copy()
    df2_rounded = df2.copy()
    
    # Redondear índices a minutos
    df1_rounded.index = df1_rounded.index.round('T')
    df2_rounded.index = df2_rounded.index.round('T')
    
    log.info(f"  After rounding:")
    log.info(f"    {s1}: {len(df1_rounded):,} records")
    log.info(f"    {s2}: {len(df2_rounded):,} records")
    
    # Agrupar por minuto (tomar último valor si hay múltiples)
    df1_grouped = df1_rounded.groupby(df1_rounded.index).last()
    df2_grouped = df2_rounded.groupby(df2_rounded.index).last()
    
    log.info(f"  After grouping by minute:")
    log.info(f"    {s1}: {len(df1_grouped):,} unique minutes")
    log.info(f"    {s2}: {len(df2_grouped):,} unique minutes")
    
    # Encontrar minutos comunes
    minute_common = df1_grouped.index.intersection(df2_grouped.index)
    log.info(f"  Common minutes: {len(minute_common):,}")
    
    if len(minute_common) > 100:
        log.info(f"  ✅ Using minute-rounded alignment")
        aligned1 = df1_grouped.loc[minute_common]['mark_price']
        aligned2 = df2_grouped.loc[minute_common]['mark_price']
        
        # Log sample of aligned data
        log.info(f"  Sample aligned data:")
        for i in range(min(5, len(aligned1))):
            ts = aligned1.index[i]
            log.info(f"    {ts}: {s1}={aligned1.iloc[i]:.6f}, {s2}={aligned2.iloc[i]:.6f}")
        
        return aligned1, aligned2
    
    # PASO 3: Usar nearest neighbor matching
    log.info(f"\n📊 STEP 3: Nearest neighbor alignment")
    
    # Encontrar overlap temporal
    start_time = max(prices1.index.min(), prices2.index.min())
    end_time = min(prices1.index.max(), prices2.index.max())
    
    log.info(f"  Temporal overlap: {start_time} to {end_time}")
    
    if start_time >= end_time:
        log.error(f"  ❌ No temporal overlap!")
        return pd.Series(), pd.Series()
    
    # Filtrar a overlap period
    overlap1 = prices1[(prices1.index >= start_time) & (prices1.index <= end_time)]
    overlap2 = prices2[(prices2.index >= start_time) & (prices2.index <= end_time)]
    
    log.info(f"  Data in overlap period:")
    log.info(f"    {s1}: {len(overlap1):,} records")
    log.info(f"    {s2}: {len(overlap2):,} records")
    
    # Usar resample para alinear a intervalos regulares
    freq = '5T'  # 5 minutos para tener más puntos
    log.info(f"  Resampling to {freq} intervals...")
    
    resampled1 = overlap1.resample(freq).last().dropna()
    resampled2 = overlap2.resample(freq).last().dropna()
    
    log.info(f"  After resampling:")
    log.info(f"    {s1}: {len(resampled1):,} intervals")
    log.info(f"    {s2}: {len(resampled2):,} intervals")
    
    # Encontrar intervalos comunes
    final_common = resampled1.index.intersection(resampled2.index)
    log.info(f"  Final common intervals: {len(final_common):,}")
    
    if len(final_common) > 10:
        log.info(f"  ✅ Using {freq} resampled alignment")
        return resampled1.loc[final_common], resampled2.loc[final_common]
    else:
        log.error(f"  ❌ Insufficient aligned data: {len(final_common)} points")
        return pd.Series(), pd.Series()

def calculate_regression_and_spread(prices1: pd.Series, prices2: pd.Series, 
                                  symbol1: str, symbol2: str) -> Dict:
    """Calcular regresión y spread entre dos series de precios"""
    log.info(f"\n📈 REGRESSION AND SPREAD CALCULATION: {symbol1} vs {symbol2}")
    
    if len(prices1) != len(prices2) or len(prices1) < 10:
        log.error(f"Insufficient aligned data: {len(prices1)} points")
        return {}
    
    # Eliminar NaN values
    valid_data = pd.DataFrame({'price1': prices1, 'price2': prices2}).dropna()
    
    if len(valid_data) < 10:
        log.error(f"Insufficient valid data after cleaning: {len(valid_data)} points")
        return {}
    
    log.info(f"Using {len(valid_data):,} aligned data points for regression")
    
    # Log sample of data being used
    log.info(f"Sample of aligned data:")
    for i in range(min(5, len(valid_data))):
        row = valid_data.iloc[i]
        ts = valid_data.index[i]
        log.info(f"  {ts}: price1={row['price1']:.6f}, price2={row['price2']:.6f}")
    
    # Preparar datos para regresión
    X = valid_data['price2'].values.reshape(-1, 1)  # Symbol2 como predictor
    y = valid_data['price1'].values                 # Symbol1 como target
    
    # Regresión lineal
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Coeficientes
    alpha = reg.intercept_
    beta = reg.coef_[0]
    r_squared = reg.score(X, y)
    
    # Predicciones
    y_pred = reg.predict(X)
    
    # Spread = Actual - Predicted
    spread = y - y_pred
    spread_series = pd.Series(spread, index=valid_data.index)
    
    # Estadísticas del spread
    spread_mean = spread.mean()
    spread_std = spread.std()
    
    # Z-score del spread
    z_score = (spread - spread_mean) / spread_std
    z_score_series = pd.Series(z_score, index=valid_data.index)
    
    # Correlación y estadísticas adicionales
    correlation = np.corrcoef(valid_data['price1'], valid_data['price2'])[0, 1]
    
    log.info(f"Regression results:")
    log.info(f"  Equation: {symbol1} = {alpha:.6f} + {beta:.6f} * {symbol2}")
    log.info(f"  R²: {r_squared:.4f}")
    log.info(f"  Correlation: {correlation:.4f}")
    log.info(f"  Spread mean: {spread_mean:.6f}")
    log.info(f"  Spread std: {spread_std:.6f}")
    
    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'correlation': correlation,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'spread': spread_series,
        'z_score': z_score_series,
        'aligned_prices1': pd.Series(valid_data['price1'], index=valid_data.index),
        'aligned_prices2': pd.Series(valid_data['price2'], index=valid_data.index),
        'predictions': pd.Series(y_pred, index=valid_data.index)
    }

def create_complete_pair_analysis(symbol1: str, symbol2: str):
    """Crear análisis completo con ejes Y duales y mejor alineamiento"""
    
    log.info(f"Creating complete pair analysis for {symbol1} / {symbol2}")
    
    s1 = symbol1.split('_')[-2] if '_' in symbol1 else symbol1
    s2 = symbol2.split('_')[-2] if '_' in symbol2 else symbol2
    
    # CARGAR DATOS
    log.info(f"Loading mark prices for {s1}...")
    df_marks1 = load_mark_prices_data(symbol1, sample_rate=10)
    
    log.info(f"Loading mark prices for {s2}...")
    df_marks2 = load_mark_prices_data(symbol2, sample_rate=10)
    
    if df_marks1.empty or df_marks2.empty:
        log.error("No mark prices data available")
        return
    
    log.info(f"Loaded data:")
    log.info(f"  {s1}: {len(df_marks1):,} records")
    log.info(f"  {s2}: {len(df_marks2):,} records")
    
    # Alineamiento mejorado con debug
    aligned_prices1, aligned_prices2 = align_price_data_improved(df_marks1, df_marks2, s1, s2)
    
    if aligned_prices1.empty or aligned_prices2.empty:
        log.error("No aligned data available")
        return
    
    log.info(f"Final aligned data: {len(aligned_prices1):,} common points")
    
    # Calcular regresión y spread
    regression_results = calculate_regression_and_spread(aligned_prices1, aligned_prices2, s1, s2)
    
    if not regression_results:
        log.error("Could not calculate regression")
        return
    
    # Crear figura con 4 subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    
    # ===== SUBPLOT 1: SERIES TEMPORALES CON EJES Y DUALES =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot symbol1 (eje izquierdo)
    ax1.plot(df_marks1.index, df_marks1['mark_price'], 
             color=color1, linewidth=1, alpha=0.8, label=f'{s1}')
    ax1.set_ylabel(f'{s1} Price (USD)', color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot symbol2 (eje derecho)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_marks2.index, df_marks2['mark_price'], 
                  color=color2, linewidth=1, alpha=0.8, label=f'{s2}')
    ax1_twin.set_ylabel(f'{s2} Price (USD)', color=color2, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title(f'Mark Prices: {s1} vs {s2}', fontweight='bold')
    
    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # ===== SUBPLOT 2: SCATTERPLOT =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Scatter plot con línea de regresión
    aligned_p1 = regression_results['aligned_prices1']
    aligned_p2 = regression_results['aligned_prices2']
    predictions = regression_results['predictions']
    
    ax2.scatter(aligned_p2, aligned_p1, alpha=0.6, s=30, color='blue', label='Data')
    ax2.plot(aligned_p2, predictions, color='red', linewidth=2, label='Regression Line')
    
    ax2.set_xlabel(f'{s2} Price (USD)')
    ax2.set_ylabel(f'{s1} Price (USD)')
    ax2.set_title(f'Scatter Plot: {s1} vs {s2}', fontweight='bold')
    
    # Añadir ecuación y R²
    alpha = regression_results['alpha']
    beta = regression_results['beta']
    r_squared = regression_results['r_squared']
    
    equation_text = f'{s1} = {alpha:.4f} + {beta:.4f} × {s2}\nR² = {r_squared:.4f}\nPoints: {len(aligned_p1):,}'
    ax2.text(0.05, 0.95, equation_text, transform=ax2.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 3: SPREAD =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    spread = regression_results['spread']
    spread_mean = regression_results['spread_mean']
    spread_std = regression_results['spread_std']
    
    ax3.plot(spread.index, spread.values, color='green', linewidth=1, alpha=0.8)
    ax3.axhline(y=spread_mean, color='red', linestyle='--', alpha=0.7, label=f'Mean: {spread_mean:.4f}')
    ax3.axhline(y=spread_mean + 2*spread_std, color='orange', linestyle=':', alpha=0.7, label='+2σ')
    ax3.axhline(y=spread_mean - 2*spread_std, color='orange', linestyle=':', alpha=0.7, label='-2σ')
    ax3.fill_between(spread.index, spread_mean - 2*spread_std, spread_mean + 2*spread_std, 
                     alpha=0.1, color='orange')
    
    ax3.set_ylabel('Spread (Actual - Predicted)')
    ax3.set_title(f'Regression Spread: {s1} - Predicted', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # ===== SUBPLOT 4: Z-SCORE =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    z_score = regression_results['z_score']
    
    ax4.plot(z_score.index, z_score.values, color='purple', linewidth=1, alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='±2σ')
    ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
    ax4.axhline(y=3, color='darkred', linestyle=':', alpha=0.7, label='±3σ')
    ax4.axhline(y=-3, color='darkred', linestyle=':', alpha=0.7)
    ax4.fill_between(z_score.index, -2, 2, alpha=0.1, color='green')
    
    ax4.set_ylabel('Z-Score')
    ax4.set_title(f'Spread Z-Score (Trading Signals)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Formato de fechas para todos los subplots con fechas
    for ax in [ax1, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Título principal
    fig.suptitle(f'Complete Pair Analysis: {s1} / {s2} (Improved Alignment)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"{s1}_{s2}_complete_improved.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log.info(f"Complete analysis plot saved: {filename}")
    
    # Estadísticas finales
    log.info(f"\n📊 COMPLETE ANALYSIS SUMMARY:")
    log.info(f"  Aligned data points: {len(aligned_prices1):,}")
    log.info(f"  Regression equation: {s1} = {alpha:.6f} + {beta:.6f} × {s2}")
    log.info(f"  R²: {r_squared:.4f}")
    log.info(f"  Correlation: {regression_results['correlation']:.4f}")
    log.info(f"  Spread statistics:")
    log.info(f"    Mean: {spread_mean:.6f}")
    log.info(f"    Std: {spread_std:.6f}")
    log.info(f"    Current Z-score: {z_score.iloc[-1]:.2f}")
    
    # Trading signals
    current_z = z_score.iloc[-1]
    if abs(current_z) > 3:
        signal = "STRONG SIGNAL"
    elif abs(current_z) > 2:
        signal = "MODERATE SIGNAL"
    else:
        signal = "NO SIGNAL"
    
    direction = "SHORT spread" if current_z > 0 else "LONG spread" if current_z < 0 else "NEUTRAL"
    
    log.info(f"  Trading assessment: {signal} - {direction}")

def main():
    """Función principal con debug detallado"""
    parser = argparse.ArgumentParser(description="Complete Pair Analysis with Debug")
    parser.add_argument("--symbol1", type=str, required=True)
    parser.add_argument("--symbol2", type=str, required=True)
    
    args = parser.parse_args()
    
    log.info("📊 Starting COMPLETE Pair Analysis with DEBUG")
    log.info(f"Includes: Time series + Scatter plot + Spread + Z-score + DEBUG logs")
    log.info(f"Symbols: {args.symbol1} / {args.symbol2}")
    
    try:
        # Crear análisis completo
        create_complete_pair_analysis(args.symbol1, args.symbol2)
        
        log.info("✅ Complete pair analysis with debug finished!")
        log.info("🎯 Check the detailed logs above to understand alignment issues!")
        return True
        
    except Exception as e:
        log.error(f"❌ Analysis failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)