#!/usr/bin/env python3
"""
Análisis COMPLETO de Mark Prices - ADAPTADO PARA ESQUEMA SIMPLIFICADO
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

 # Ensure project root is on PYTHONPATH for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 9

def check_table_columns(table_name: str) -> dict:
    """Verificar qué columnas existen en una tabla"""
    with db_manager.get_session() as session:
        try:
            result = session.execute(text(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY column_name
            """)).fetchall()
            
            columns = {row.column_name: row.data_type for row in result}
            log.info(f"Tabla {table_name}: {len(columns)} columnas encontradas")
            return columns
            
        except Exception as e:
            log.error(f"Error checking table {table_name}: {e}")
            return {}

def get_available_data_info(symbol: str) -> dict:
    """Obtener información de datos disponibles - ADAPTADO para esquema simplificado"""
    info = {}
    
    with db_manager.get_session() as session:
        # Mark prices - usando esquema simplificado
        try:
            # Verificar qué columnas existen primero
            mp_columns = check_table_columns('mark_prices')
            
            # Construir query basado en columnas disponibles
            count_clauses = ['COUNT(*) as count']
            
            if 'valid_for_trading' in mp_columns:
                count_clauses.append('COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as valid_count')
            elif 'is_valid' in mp_columns:
                count_clauses.append('COUNT(CASE WHEN is_valid = TRUE THEN 1 END) as valid_count')
            else:
                count_clauses.append('0 as valid_count')
            
            if 'liquidity_score' in mp_columns:
                count_clauses.append('AVG(liquidity_score) as avg_liquidity')
            else:
                count_clauses.append('0 as avg_liquidity')
            
            if 'ohlcv_volume' in mp_columns:
                count_clauses.append('COUNT(CASE WHEN ohlcv_volume IS NOT NULL THEN 1 END) as volume_count')
            else:
                count_clauses.append('0 as volume_count')
            
            query = f"""
                SELECT 
                    {', '.join(count_clauses)},
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date
                FROM mark_prices 
                WHERE symbol = :symbol
            """
            
            result = session.execute(text(query), {'symbol': symbol}).fetchone()
            
            info['mark_prices'] = {
                'count': result.count,
                'min_date': result.min_date,
                'max_date': result.max_date,
                'valid_count': result.valid_count,
                'avg_liquidity': result.avg_liquidity,
                'has_volume': result.volume_count > 0
            }
        except Exception as e:
            log.error(f"Error getting mark_prices info: {e}")
            info['mark_prices'] = {'count': 0}
        
        # OHLCV
        try:
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as count,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    AVG(volume) as avg_volume
                FROM ohlcv 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            info['ohlcv'] = {
                'count': result.count,
                'min_date': result.min_date,
                'max_date': result.max_date,
                'avg_volume': result.avg_volume
            }
        except Exception as e:
            log.error(f"Error getting ohlcv info: {e}")
            info['ohlcv'] = {'count': 0}
        
        # Orderbook
        try:
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as count,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL THEN 1 END) as valid_quotes
                FROM orderbook 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            info['orderbook'] = {
                'count': result.count,
                'min_date': result.min_date,
                'max_date': result.max_date,
                'valid_quotes': result.valid_quotes
            }
        except Exception as e:
            log.error(f"Error getting orderbook info: {e}")
            info['orderbook'] = {'count': 0}
        
        # Funding rates (para perpetuos)
        if "PERP_" in symbol:
            try:
                result = session.execute(text("""
                    SELECT 
                        COUNT(*) as count,
                        MIN(timestamp) as min_date,
                        MAX(timestamp) as max_date,
                        AVG(funding_rate) as avg_rate
                    FROM funding_rates 
                    WHERE symbol = :symbol
                """), {'symbol': symbol}).fetchone()
                
                info['funding_rates'] = {
                    'count': result.count,
                    'min_date': result.min_date,
                    'max_date': result.max_date,
                    'avg_rate': result.avg_rate
                }
            except Exception as e:
                log.error(f"Error getting funding_rates info: {e}")
                info['funding_rates'] = {'count': 0}
    
    return info

def load_mark_prices_data(symbol: str, sample_rate: int = 10) -> pd.DataFrame:
    """Cargar datos de mark prices - ADAPTADO para esquema simplificado"""
    with db_manager.get_session() as session:
        # Verificar columnas disponibles
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

def load_ohlcv_data(symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    """Cargar datos OHLCV separadamente"""
    with db_manager.get_session() as session:
        where_clause = "WHERE symbol = :symbol"
        params = {'symbol': symbol}
        
        if start_date and end_date:
            where_clause += " AND timestamp BETWEEN :start_date AND :end_date"
            params.update({'start_date': start_date, 'end_date': end_date})
        
        query = text(f"""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv 
            {where_clause}
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params=params, index_col='timestamp')
        return df

def load_orderbook_spreads(symbol: str, sample_rate: int = 50) -> pd.DataFrame:
    """Cargar spreads calculados del orderbook"""
    with db_manager.get_session() as session:
        query = text(f"""
            SELECT 
                timestamp,
                (ask1_price - bid1_price) / bid1_price * 100 as raw_spread_pct,
                bid1_price,
                ask1_price
            FROM (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (ORDER BY timestamp) as rn
                FROM orderbook 
                WHERE symbol = :symbol
                AND bid1_price IS NOT NULL 
                AND ask1_price IS NOT NULL
                AND bid1_price > 0 
                AND ask1_price > 0
                AND bid1_price < ask1_price
            ) t
            WHERE rn % :sample_rate = 1
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params={
            'symbol': symbol, 
            'sample_rate': sample_rate
        }, index_col='timestamp')
        
        return df

def load_funding_rates(symbol: str) -> pd.DataFrame:
    """Cargar funding rates si es perpetuo"""
    if "PERP_" not in symbol:
        return pd.DataFrame()
    
    with db_manager.get_session() as session:
        query = text("""
            SELECT 
                timestamp,
                funding_rate
            FROM funding_rates 
            WHERE symbol = :symbol
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params={'symbol': symbol}, index_col='timestamp')
        return df

def add_invalid_trading_shading(ax, df_marks):
    """Agregar sombreado para periodos no válidos para trading"""
    if 'valid_for_trading' not in df_marks.columns:
        return
    
    # Encontrar periodos donde valid_for_trading es False
    invalid_periods = df_marks[df_marks['valid_for_trading'] == False]
    
    if len(invalid_periods) > 0:
        # Agrupar periodos consecutivos
        invalid_periods = invalid_periods.copy()
        invalid_periods['group'] = (invalid_periods.index.to_series().diff() > pd.Timedelta('1H')).cumsum()
        
        for group_id, group in invalid_periods.groupby('group'):
            start_time = group.index.min()
            end_time = group.index.max()
            
            # Sombrear área
            ax.axvspan(start_time, end_time, alpha=0.2, color='red', 
                      label='Invalid for Trading' if group_id == invalid_periods['group'].iloc[0] else "")

def create_comprehensive_analysis_plot(symbol: str):
    """Crear análisis comprensivo - ADAPTADO para esquema simplificado"""
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    log.info(f"Analizando datos disponibles para {symbol_short}...")
    
    # Obtener info de disponibilidad
    data_info = get_available_data_info(symbol)
    
    # Cargar todos los datos disponibles
    df_marks = load_mark_prices_data(symbol, sample_rate=10)
    df_ohlcv = load_ohlcv_data(symbol)
    df_spreads = load_orderbook_spreads(symbol, sample_rate=50)
    df_funding = load_funding_rates(symbol)
    
    log.info(f"Datos cargados:")
    log.info(f"  Mark prices: {len(df_marks):,}")
    log.info(f"  OHLCV: {len(df_ohlcv):,}")
    log.info(f"  Orderbook spreads: {len(df_spreads):,}")
    log.info(f"  Funding rates: {len(df_funding):,}")
    
    # Crear figura
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'{symbol_short} - Complete Data Analysis (Simplified Schema)', fontsize=16, fontweight='bold')
    
    # 1. SERIES TEMPORALES PRINCIPALES
    ax1 = fig.add_subplot(gs[0, :])
    
    if not df_marks.empty:
        ax1.plot(df_marks.index, df_marks['mark_price'], label='Mark Price (VWAP)', 
                linewidth=1, color='blue', alpha=0.8)
        
        if 'orderbook_mid' in df_marks.columns:
            ax1.plot(df_marks.index, df_marks['orderbook_mid'], label='Orderbook Mid', 
                    linewidth=0.8, color='purple', alpha=0.6)
        
        # Agregar sombreado para periodos no válidos
        add_invalid_trading_shading(ax1, df_marks)
    
    if not df_ohlcv.empty:
        ax1.plot(df_ohlcv.index, df_ohlcv['close'], label='OHLCV Close', 
                linewidth=0.8, color='green', alpha=0.6)
    
    ax1.set_ylabel('Price (USD)')
    ax1.set_title(f'{symbol_short} - Price History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. OHLCV CANDLESTICK (si hay datos)
    ax2 = fig.add_subplot(gs[1, 0])
    
    if not df_ohlcv.empty and len(df_ohlcv) > 0:
        # Tomar muestra para visualización
        if len(df_ohlcv) > 1000:
            step = len(df_ohlcv) // 1000
            ohlcv_sample = df_ohlcv.iloc[::step]
        else:
            ohlcv_sample = df_ohlcv
        
        ax2.plot(ohlcv_sample.index, ohlcv_sample['high'], color='green', alpha=0.5, linewidth=0.8, label='High')
        ax2.plot(ohlcv_sample.index, ohlcv_sample['low'], color='red', alpha=0.5, linewidth=0.8, label='Low')
        ax2.plot(ohlcv_sample.index, ohlcv_sample['close'], color='black', linewidth=1, label='Close')
        ax2.fill_between(ohlcv_sample.index, ohlcv_sample['high'], ohlcv_sample['low'], 
                       alpha=0.1, color='gray')
        ax2.legend()
        ax2.set_title(f'{symbol_short} - OHLCV Range')
        
        # Agregar sombreado para periodos no válidos si hay datos de marks
        if not df_marks.empty:
            add_invalid_trading_shading(ax2, df_marks)
    else:
        ax2.text(0.5, 0.5, f'No OHLCV Data\n({data_info["ohlcv"]["count"]} records in DB)', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{symbol_short} - OHLCV Range')
    
    ax2.set_ylabel('Price (USD)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. SPREADS (orderbook calculado)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Spreads calculados del orderbook
    if not df_spreads.empty:
        ax3.plot(df_spreads.index, df_spreads['raw_spread_pct'], color='orange', 
                linewidth=0.8, alpha=0.7, label='Raw Orderbook Spreads')
        ax3.legend()
        ax3.set_title(f'{symbol_short} - Price Spreads')
        
        # Agregar sombreado para periodos no válidos
        if not df_marks.empty:
            add_invalid_trading_shading(ax3, df_marks)
    else:
        ax3.text(0.5, 0.5, 'No Spread Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'{symbol_short} - Price Spreads')
    
    ax3.set_ylabel('Spread (%)')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. LIQUIDITY SCORES
    ax4 = fig.add_subplot(gs[2, 0])
    
    if not df_marks.empty and 'liquidity_score' in df_marks.columns:
        scores = df_marks['liquidity_score'].dropna()
        if len(scores) > 0:
            ax4.plot(scores.index, scores, color='lightblue', linewidth=0.8, alpha=0.7, label='Score')
            ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='High Quality (>0.7)')
            ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Low Quality (<0.3)')
            
            mean_score = scores.mean()
            ax4.axhline(y=mean_score, color='blue', linestyle=':', alpha=0.7, 
                       label=f'Mean: {mean_score:.3f}')
            ax4.legend()
            
            # Agregar sombreado para periodos no válidos
            add_invalid_trading_shading(ax4, df_marks)
        else:
            ax4.text(0.5, 0.5, 'No Liquidity Score Data', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No Liquidity Score Data', ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_ylabel('Liquidity Score')
    ax4.set_title(f'{symbol_short} - Liquidity Quality')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. FUNDING RATES (si es perpetuo)
    ax5 = fig.add_subplot(gs[2, 1])
    
    if not df_funding.empty:
        ax5.plot(df_funding.index, df_funding['funding_rate'] * 100, color='purple', 
                linewidth=1, alpha=0.8, label='Funding Rate')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        mean_funding = df_funding['funding_rate'].mean() * 100
        ax5.axhline(y=mean_funding, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_funding:.4f}%')
        ax5.legend()
        ax5.set_title(f'{symbol_short} - Funding Rates')
        
        # Agregar sombreado para periodos no válidos
        if not df_marks.empty:
            add_invalid_trading_shading(ax5, df_marks)
    else:
        if "PERP_" in symbol:
            ax5.text(0.5, 0.5, f'No Funding Data\n({data_info.get("funding_rates", {}).get("count", 0)} records)', 
                    ha='center', va='center', transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, 'Not a Perpetual Contract', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title(f'{symbol_short} - Funding Rates')
    
    ax5.set_ylabel('Funding Rate (%)')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. VOLUME (OHLCV vs Mark Prices) - CON EJES DISTINTOS
    ax6 = fig.add_subplot(gs[3, 0])
    ax6_twin = None
    
    volumes_plotted = False
    
    # Volume de OHLCV (eje izquierdo)
    if not df_ohlcv.empty and 'volume' in df_ohlcv.columns:
        volumes = df_ohlcv['volume'].dropna()
        if len(volumes) > 0:
            # Tomar muestra para barras
            if len(volumes) > 500:
                step = len(volumes) // 500
                vol_sample = volumes.iloc[::step]
            else:
                vol_sample = volumes
            
            ax6.bar(vol_sample.index, vol_sample, width=pd.Timedelta(hours=1), 
                   color='lightcoral', alpha=0.7, label='OHLCV Volume')
            ax6.set_ylabel('OHLCV Volume', color='lightcoral')
            ax6.tick_params(axis='y', labelcolor='lightcoral')
            volumes_plotted = True
    
    # Volume de mark prices (eje derecho)
    if not df_marks.empty and 'ohlcv_volume' in df_marks.columns:
        mark_volumes = df_marks['ohlcv_volume'].dropna()
        if len(mark_volumes) > 0:
            ax6_twin = ax6.twinx()
            ax6_twin.plot(mark_volumes.index, mark_volumes, color='blue', 
                         linewidth=1, alpha=0.8, label='Mark Prices Volume')
            ax6_twin.set_ylabel('Mark Prices Volume', color='blue')
            ax6_twin.tick_params(axis='y', labelcolor='blue')
            volumes_plotted = True
    
    if volumes_plotted:
        # Ajustar leyendas
        lines1, labels1 = ax6.get_legend_handles_labels()
        if ax6_twin:
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax6.legend()
        
        # Agregar sombreado para periodos no válidos
        if not df_marks.empty:
            add_invalid_trading_shading(ax6, df_marks)
    else:
        ax6.text(0.5, 0.5, 'No Volume Data', ha='center', va='center', transform=ax6.transAxes)
    
    ax6.set_title(f'{symbol_short} - Trading Volume')
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. STATISTICS & DATA INFO - ADAPTADO
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis('off')
    
    # Estadísticas comprehensivas
    stats_text = f"{symbol_short} - DATA AVAILABILITY\n\n"
    
    # Info de cada tabla
    for table, info in data_info.items():
        count = info.get('count', 0)
        if count > 0:
            min_date = info.get('min_date')
            max_date = info.get('max_date')
            stats_text += f"{table.upper()}:\n"
            stats_text += f"  Records: {count:,}\n"
            if min_date and max_date:
                days = (max_date - min_date).days
                stats_text += f"  Range: {min_date.date()} to {max_date.date()}\n"
                stats_text += f"  Duration: {days} days\n"
            stats_text += "\n"
        else:
            stats_text += f"{table.upper()}: NO DATA\n\n"
    
    # Estadísticas de mark prices - ADAPTADO
    if not df_marks.empty:
        stats_text += f"MARK PRICE STATS:\n"
        prices = df_marks['mark_price'].dropna()
        if len(prices) > 0:
            stats_text += f"  Mean: ${prices.mean():.6f}\n"
            stats_text += f"  Range: ${prices.min():.6f} - ${prices.max():.6f}\n"
        
        if 'valid_for_trading' in df_marks.columns:
            valid_pct = df_marks['valid_for_trading'].mean() * 100
            stats_text += f"  Valid for Trading: {valid_pct:.1f}%\n"
        
        if 'liquidity_score' in df_marks.columns:
            liq_scores = df_marks['liquidity_score'].dropna()
            if len(liq_scores) > 0:
                stats_text += f"  Avg Liquidity: {liq_scores.mean():.3f}\n"
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, 
            verticalalignment='top', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{symbol_short}_comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log.info(f"Análisis comprensivo guardado: plots/{symbol_short}_comprehensive_analysis.png")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive analysis with simplified schema")
    parser.add_argument("--symbol", type=str, help="Analyze specific symbol only")
    
    args = parser.parse_args()
    
    log.info("Iniciando análisis COMPRENSIVO con esquema simplificado...")
    
    # Verificar esquemas de tablas primero
    log.info("Verificando esquemas de tablas...")
    for table in ['mark_prices', 'ohlcv', 'orderbook', 'funding_rates']:
        columns = check_table_columns(table)
        if columns:
            col_list = list(columns.keys())
            display_cols = col_list[:5] + ['...'] if len(col_list) > 5 else col_list
            log.info(f"  {table}: {display_cols}")
        else:
            log.warning(f"  {table}: NO COLUMNS FOUND")
    
    try:
        # Obtener símbolos
        if args.symbol:
            symbols = [args.symbol]
        else:
            try:
                active_pairs = settings.get_active_pairs()
                symbols = list(set([pair.symbol1 for pair in active_pairs] + [pair.symbol2 for pair in active_pairs]))
            except:
                symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
        
        log.info(f"Analizando {len(symbols)} símbolos con esquema simplificado")
        
        # Análisis comprensivo por símbolo
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"PROCESANDO {symbol}")
            log.info(f"{'='*60}")
            
            create_comprehensive_analysis_plot(symbol)
        
        log.info(f"\n🎉 Análisis COMPRENSIVO completado!")
        log.info(f"Gráficas guardadas en: plots/")
        for symbol in symbols:
            symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
            log.info(f"  - {symbol_short}_comprehensive_analysis.png")
        
        return True
        
    except Exception as e:
        log.error(f"Error en análisis: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)