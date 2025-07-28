#!/usr/bin/env python3
"""
🚀 ANÁLISIS OHLCV COMPRENSIVO - SIN MARK PRICES
Análisis profundo de datos OHLCV, volumen, volatilidad y métricas técnicas
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
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is on PYTHONPATH for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.database.connection import db_manager
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 9

def get_comprehensive_data_info(symbol: str) -> dict:
    """Obtener información comprensiva de datos disponibles"""
    info = {}
    
    with db_manager.get_session() as session:
        # OHLCV - análisis detallado
        try:
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    COUNT(DISTINCT DATE(timestamp)) as trading_days,
                    AVG(volume) as avg_volume,
                    SUM(volume) as total_volume,
                    MIN(volume) as min_volume,
                    MAX(volume) as max_volume,
                    AVG(high - low) as avg_range,
                    AVG((high - low) / close * 100) as avg_range_pct,
                    COUNT(CASE WHEN volume = 0 THEN 1 END) as zero_volume_count,
                    AVG(close) as avg_price,
                    MIN(low) as min_price,
                    MAX(high) as max_price
                FROM ohlcv 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            info['ohlcv'] = {
                'total_records': result.total_records or 0,
                'min_date': result.min_date,
                'max_date': result.max_date,
                'trading_days': result.trading_days or 0,
                'avg_volume': result.avg_volume or 0,
                'total_volume': result.total_volume or 0,
                'min_volume': result.min_volume or 0,
                'max_volume': result.max_volume or 0,
                'avg_range': result.avg_range or 0,
                'avg_range_pct': result.avg_range_pct or 0,
                'zero_volume_count': result.zero_volume_count or 0,
                'avg_price': result.avg_price or 0,
                'min_price': result.min_price or 0,
                'max_price': result.max_price or 0
            }
        except Exception as e:
            log.error(f"Error getting OHLCV info: {e}")
            info['ohlcv'] = {'total_records': 0}
        
        # Orderbook - análisis básico
        try:
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL THEN 1 END) as valid_quotes,
                    AVG(CASE WHEN bid1_price > 0 AND ask1_price > 0 
                        THEN (ask1_price - bid1_price) / bid1_price * 100 END) as avg_spread_pct
                FROM orderbook 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            info['orderbook'] = {
                'total_records': result.total_records or 0,
                'min_date': result.min_date,
                'max_date': result.max_date,
                'valid_quotes': result.valid_quotes or 0,
                'avg_spread_pct': result.avg_spread_pct or 0
            }
        except Exception as e:
            log.error(f"Error getting orderbook info: {e}")
            info['orderbook'] = {'total_records': 0}
        
        # Funding rates (para perpetuos)
        if "PERP_" in symbol:
            try:
                result = session.execute(text("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(timestamp) as min_date,
                        MAX(timestamp) as max_date,
                        AVG(funding_rate) as avg_rate,
                        MIN(funding_rate) as min_rate,
                        MAX(funding_rate) as max_rate,
                        STDDEV(funding_rate) as std_rate
                    FROM funding_rates 
                    WHERE symbol = :symbol
                """), {'symbol': symbol}).fetchone()
                
                info['funding_rates'] = {
                    'total_records': result.total_records or 0,
                    'min_date': result.min_date,
                    'max_date': result.max_date,
                    'avg_rate': result.avg_rate or 0,
                    'min_rate': result.min_rate or 0,
                    'max_rate': result.max_rate or 0,
                    'std_rate': result.std_rate or 0
                }
            except Exception as e:
                log.error(f"Error getting funding_rates info: {e}")
                info['funding_rates'] = {'total_records': 0}
    
    return info

def load_ohlcv_comprehensive(symbol: str, sample_rate: int = 10) -> pd.DataFrame:
    """Cargar datos OHLCV con sampling inteligente"""
    with db_manager.get_session() as session:
        # Verificar cantidad total
        count_result = session.execute(text("""
            SELECT COUNT(*) as total FROM ohlcv WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        total_records = count_result.total
        
        if total_records == 0:
            return pd.DataFrame()
        
        # Sampling inteligente
        if total_records > 50000:
            sample_clause = f"WHERE rn % {sample_rate} = 1"
            log.info(f"📊 Sampling OHLCV: 1 of every {sample_rate} records ({total_records:,} total)")
        else:
            sample_clause = ""
            log.info(f"📊 Loading all {total_records:,} OHLCV records")
        
        query = text(f"""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (ORDER BY timestamp) as rn
                FROM ohlcv 
                WHERE symbol = :symbol
            ) t
            {sample_clause}
            ORDER BY timestamp
        """)
        
        df = pd.read_sql(query, session.bind, params={'symbol': symbol}, index_col='timestamp')
        return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular indicadores técnicos completos"""
    if df.empty:
        return df
    
    df = df.copy()
    
    try:
        # 1. Retornos
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Volatilidad rolling
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(1440)  # Anualizada para minutos
        df['volatility_60'] = df['returns'].rolling(60).std() * np.sqrt(1440)
        
        # 3. VWAP (Volume Weighted Average Price)
        df['cumulative_volume'] = df['volume'].cumsum()
        df['cumulative_volume_price'] = (df['close'] * df['volume']).cumsum()
        df['vwap'] = df['cumulative_volume_price'] / df['cumulative_volume']
        
        # 4. Promedios móviles
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # 5. ATR (Average True Range)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)
        
        # 6. RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 7. Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 8. Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 9. Price ranges
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close'] * 100
        
        # 10. Gaps
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = (df['gap'] / df['close'].shift(1)) * 100
        
        log.info(f"✅ Calculated {10} categories of technical indicators")
        
    except Exception as e:
        log.error(f"Error calculating technical indicators: {e}")
    
    return df

def load_orderbook_analysis(symbol: str, sample_rate: int = 100) -> pd.DataFrame:
    """Cargar datos de orderbook para análisis de spreads"""
    with db_manager.get_session() as session:
        query = text(f"""
            SELECT 
                timestamp,
                (ask1_price - bid1_price) / bid1_price * 100 as spread_pct,
                bid1_price,
                ask1_price,
                bid1_size,
                ask1_size,
                (bid1_size + ask1_size) as total_level1_size
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

def load_funding_rates_analysis(symbol: str) -> pd.DataFrame:
    """Cargar funding rates con análisis"""
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
        
        if not df.empty:
            # Calcular métricas de funding
            df['funding_rate_pct'] = df['funding_rate'] * 100
            df['funding_cumulative'] = df['funding_rate'].cumsum()
            df['funding_sma'] = df['funding_rate'].rolling(7).mean()  # 7-period SMA
        
        return df

def create_ohlcv_comprehensive_analysis(symbol: str):
    """Crear análisis comprensivo OHLCV - SIN MARK PRICES"""
    symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
    
    log.info(f"🚀 Iniciando análisis OHLCV comprensivo para {symbol_short}...")
    
    # Obtener información de datos
    data_info = get_comprehensive_data_info(symbol)
    
    # Cargar datos
    df_ohlcv = load_ohlcv_comprehensive(symbol, sample_rate=5)
    df_orderbook = load_orderbook_analysis(symbol, sample_rate=200)
    df_funding = load_funding_rates_analysis(symbol)
    
    if df_ohlcv.empty:
        log.error(f"❌ No OHLCV data found for {symbol}")
        return
    
    # Calcular indicadores técnicos
    df_ohlcv = calculate_technical_indicators(df_ohlcv)
    
    log.info(f"📊 Datos cargados:")
    log.info(f"  OHLCV: {len(df_ohlcv):,} registros")
    log.info(f"  Orderbook: {len(df_orderbook):,} snapshots")
    log.info(f"  Funding: {len(df_funding):,} registros")
    
    # Crear figura comprehensiva
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
    
    fig.suptitle(f'{symbol_short} - Comprehensive OHLCV Analysis', fontsize=20, fontweight='bold')
    
    # 1. PRICE CHART CON CANDLESTICKS MEJORADO (Fila 1, span 2 columnas)
    ax1 = fig.add_subplot(gs[0, :2])
    create_candlestick_chart(ax1, df_ohlcv, symbol_short)
    
    # 2. VOLUME ANALYSIS (Fila 1, columna 3)
    ax2 = fig.add_subplot(gs[0, 2])
    create_volume_analysis(ax2, df_ohlcv, symbol_short)
    
    # 3. VOLATILITY ANALYSIS (Fila 1, columna 4)
    ax3 = fig.add_subplot(gs[0, 3])
    create_volatility_analysis(ax3, df_ohlcv, symbol_short)
    
    # 4. TECHNICAL INDICATORS (Fila 2, span 2 columnas)
    ax4 = fig.add_subplot(gs[1, :2])
    create_technical_indicators_chart(ax4, df_ohlcv, symbol_short)
    
    # 5. RETURNS DISTRIBUTION (Fila 2, columna 3)
    ax5 = fig.add_subplot(gs[1, 2])
    create_returns_distribution(ax5, df_ohlcv, symbol_short)
    
    # 6. RSI & MOMENTUM (Fila 2, columna 4)
    ax6 = fig.add_subplot(gs[1, 3])
    create_rsi_momentum(ax6, df_ohlcv, symbol_short)
    
    # 7. VWAP & PRICE LEVELS (Fila 3, span 2 columnas)
    ax7 = fig.add_subplot(gs[2, :2])
    create_vwap_analysis(ax7, df_ohlcv, symbol_short)
    
    # 8. ORDERBOOK SPREADS (Fila 3, columna 3)
    ax8 = fig.add_subplot(gs[2, 2])
    create_orderbook_spreads(ax8, df_orderbook, symbol_short)
    
    # 9. FUNDING RATES (Fila 3, columna 4)
    ax9 = fig.add_subplot(gs[2, 3])
    create_funding_analysis(ax9, df_funding, symbol_short, symbol)
    
    # 10. BOLLINGER BANDS (Fila 4, span 2 columnas)
    ax10 = fig.add_subplot(gs[3, :2])
    create_bollinger_bands(ax10, df_ohlcv, symbol_short)
    
    # 11. VOLUME INDICATORS (Fila 4, columna 3)
    ax11 = fig.add_subplot(gs[3, 2])
    create_volume_indicators(ax11, df_ohlcv, symbol_short)
    
    # 12. PRICE GAPS ANALYSIS (Fila 4, columna 4)
    ax12 = fig.add_subplot(gs[3, 3])
    create_gaps_analysis(ax12, df_ohlcv, symbol_short)
    
    # 13. COMPREHENSIVE STATISTICS (Fila 5, span todas las columnas)
    ax13 = fig.add_subplot(gs[4, :])
    create_comprehensive_stats_table(ax13, symbol_short, data_info, df_ohlcv, df_orderbook, df_funding)
    
    # Guardar
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    filename = f'{symbol_short}_ohlcv_comprehensive_analysis.png'
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log.info(f"✅ Análisis OHLCV comprensivo guardado: plots/{filename}")

def create_candlestick_chart(ax, df, symbol_short):
    """Crear gráfico de candlesticks mejorado"""
    if df.empty:
        ax.text(0.5, 0.5, 'No OHLCV Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Sampling para visualización
    if len(df) > 2000:
        step = len(df) // 2000
        df_sample = df.iloc[::step]
    else:
        df_sample = df
    
    # Colores para candlesticks
    colors = ['green' if close >= open else 'red' 
              for open, close in zip(df_sample['open'], df_sample['close'])]
    
    # Cuerpos de las velas
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        ax.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=0.5, alpha=0.7)
        
        body_color = 'green' if row['close'] >= row['open'] else 'red'
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        ax.bar(idx, body_height, bottom=body_bottom, color=body_color, alpha=0.7, width=pd.Timedelta(minutes=60))
    
    # Agregar SMA
    if 'sma_20' in df_sample.columns:
        ax.plot(df_sample.index, df_sample['sma_20'], color='blue', linewidth=1, alpha=0.8, label='SMA 20')
    if 'sma_50' in df_sample.columns:
        ax.plot(df_sample.index, df_sample['sma_50'], color='orange', linewidth=1, alpha=0.8, label='SMA 50')
    
    ax.set_title(f'{symbol_short} - Price Action (Candlesticks)', fontweight='bold')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_volume_analysis(ax, df, symbol_short):
    """Análisis de volumen detallado"""
    if df.empty or 'volume' not in df.columns:
        ax.text(0.5, 0.5, 'No Volume Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Volume bars
    volume_sample = df['volume'].iloc[::max(1, len(df)//500)]
    ax.bar(volume_sample.index, volume_sample, alpha=0.6, color='lightblue', width=pd.Timedelta(hours=2))
    
    # Volume SMA
    if 'volume_sma' in df.columns:
        volume_sma_sample = df['volume_sma'].iloc[::max(1, len(df)//500)]
        ax.plot(volume_sma_sample.index, volume_sma_sample, color='red', linewidth=2, label='Volume SMA')
    
    ax.set_title(f'{symbol_short} - Volume Analysis')
    ax.set_ylabel('Volume')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_volatility_analysis(ax, df, symbol_short):
    """Análisis de volatilidad"""
    if df.empty or 'volatility_20' not in df.columns:
        ax.text(0.5, 0.5, 'No Volatility Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    vol_sample = df[['volatility_20', 'volatility_60']].iloc[::max(1, len(df)//1000)]
    
    ax.plot(vol_sample.index, vol_sample['volatility_20'], color='red', linewidth=1, label='Vol 20-period', alpha=0.8)
    ax.plot(vol_sample.index, vol_sample['volatility_60'], color='blue', linewidth=1, label='Vol 60-period', alpha=0.8)
    
    ax.set_title(f'{symbol_short} - Volatility')
    ax.set_ylabel('Volatility (Annualized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_technical_indicators_chart(ax, df, symbol_short):
    """Gráfico de indicadores técnicos principales"""
    if df.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    sample_df = df.iloc[::max(1, len(df)//1000)]
    
    # Precio con EMAs
    ax.plot(sample_df.index, sample_df['close'], color='black', linewidth=1, label='Close', alpha=0.8)
    
    if 'ema_20' in sample_df.columns:
        ax.plot(sample_df.index, sample_df['ema_20'], color='blue', linewidth=1, label='EMA 20', alpha=0.7)
    if 'sma_50' in sample_df.columns:
        ax.plot(sample_df.index, sample_df['sma_50'], color='orange', linewidth=1, label='SMA 50', alpha=0.7)
    
    ax.set_title(f'{symbol_short} - Technical Indicators')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_returns_distribution(ax, df, symbol_short):
    """Distribución de retornos"""
    if df.empty or 'returns' not in df.columns:
        ax.text(0.5, 0.5, 'No Returns Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    returns = df['returns'].dropna()
    if len(returns) == 0:
        ax.text(0.5, 0.5, 'No Valid Returns', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Histograma
    ax.hist(returns * 100, bins=50, alpha=0.7, color='skyblue', density=True)
    
    # Estadísticas
    mean_ret = returns.mean() * 100
    std_ret = returns.std() * 100
    
    ax.axvline(mean_ret, color='red', linestyle='--', label=f'Mean: {mean_ret:.3f}%')
    ax.axvline(mean_ret + std_ret, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_ret+std_ret:.3f}%')
    ax.axvline(mean_ret - std_ret, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_ret-std_ret:.3f}%')
    
    ax.set_title(f'{symbol_short} - Returns Distribution')
    ax.set_xlabel('Returns (%)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_rsi_momentum(ax, df, symbol_short):
    """RSI y momentum"""
    if df.empty or 'rsi' not in df.columns:
        ax.text(0.5, 0.5, 'No RSI Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    rsi_sample = df['rsi'].iloc[::max(1, len(df)//1000)].dropna()
    
    ax.plot(rsi_sample.index, rsi_sample, color='purple', linewidth=1, alpha=0.8)
    ax.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax.axhline(50, color='gray', linestyle='-', alpha=0.5)
    
    ax.set_title(f'{symbol_short} - RSI (14)')
    ax.set_ylabel('RSI')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_vwap_analysis(ax, df, symbol_short):
    """Análisis VWAP"""
    if df.empty or 'vwap' not in df.columns:
        ax.text(0.5, 0.5, 'No VWAP Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    sample_df = df.iloc[::max(1, len(df)//1000)]
    
    ax.plot(sample_df.index, sample_df['close'], color='black', linewidth=1, label='Close', alpha=0.8)
    ax.plot(sample_df.index, sample_df['vwap'], color='blue', linewidth=2, label='VWAP', alpha=0.8)
    
    # Diferencia VWAP
    vwap_diff = ((sample_df['close'] - sample_df['vwap']) / sample_df['vwap'] * 100).fillna(0)
    
    ax2 = ax.twinx()
    ax2.plot(sample_df.index, vwap_diff, color='red', linewidth=1, alpha=0.6, label='Close vs VWAP (%)')
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Deviation from VWAP (%)', color='red')
    
    ax.set_title(f'{symbol_short} - VWAP Analysis')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

def create_orderbook_spreads(ax, df, symbol_short):
    """Análisis de spreads del orderbook"""
    if df.empty:
        ax.text(0.5, 0.5, 'No Orderbook Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    spreads = df['spread_pct'].dropna()
    if len(spreads) == 0:
        ax.text(0.5, 0.5, 'No Valid Spreads', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Time series de spreads
    ax.plot(spreads.index, spreads, color='orange', linewidth=0.8, alpha=0.7)
    
    # Estadísticas
    mean_spread = spreads.mean()
    p95_spread = spreads.quantile(0.95)
    
    ax.axhline(mean_spread, color='blue', linestyle='--', alpha=0.7, label=f'Mean: {mean_spread:.4f}%')
    ax.axhline(p95_spread, color='red', linestyle='--', alpha=0.7, label=f'P95: {p95_spread:.4f}%')
    
    ax.set_title(f'{symbol_short} - Bid-Ask Spreads')
    ax.set_ylabel('Spread (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_funding_analysis(ax, df, symbol_short, symbol):
    """Análisis de funding rates"""
    if "PERP_" not in symbol or df.empty:
        if "PERP_" not in symbol:
            ax.text(0.5, 0.5, 'Not a Perpetual Contract', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No Funding Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{symbol_short} - Funding Rates')
        return
    
    # Funding rate en porcentaje
    ax.plot(df.index, df['funding_rate_pct'], color='purple', linewidth=1, alpha=0.8, label='Funding Rate')
    
    if 'funding_sma' in df.columns:
        ax.plot(df.index, df['funding_sma'] * 100, color='blue', linewidth=1, alpha=0.8, label='7-period SMA')
    
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Estadísticas
    mean_funding = df['funding_rate'].mean() * 100
    ax.axhline(mean_funding, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_funding:.4f}%')
    
    ax.set_title(f'{symbol_short} - Funding Rates')
    ax.set_ylabel('Funding Rate (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_bollinger_bands(ax, df, symbol_short):
    """Bollinger Bands analysis"""
    if df.empty or 'bb_upper' not in df.columns:
        ax.text(0.5, 0.5, 'No Bollinger Bands Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    sample_df = df.iloc[::max(1, len(df)//1000)]
    
    ax.plot(sample_df.index, sample_df['close'], color='black', linewidth=1, label='Close')
    ax.plot(sample_df.index, sample_df['bb_middle'], color='blue', linewidth=1, label='BB Middle (SMA 20)')
    ax.plot(sample_df.index, sample_df['bb_upper'], color='red', linewidth=1, alpha=0.7, label='BB Upper')
    ax.plot(sample_df.index, sample_df['bb_lower'], color='green', linewidth=1, alpha=0.7, label='BB Lower')
    
    # Fill between bands
    ax.fill_between(sample_df.index, sample_df['bb_upper'], sample_df['bb_lower'], alpha=0.1, color='gray')
    
    ax.set_title(f'{symbol_short} - Bollinger Bands')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_volume_indicators(ax, df, symbol_short):
    """Indicadores de volumen"""
    if df.empty or 'volume_ratio' not in df.columns:
        ax.text(0.5, 0.5, 'No Volume Indicators', ha='center', va='center', transform=ax.transAxes)
        return
    
    vol_ratio_sample = df['volume_ratio'].iloc[::max(1, len(df)//1000)].dropna()
    
    ax.plot(vol_ratio_sample.index, vol_ratio_sample, color='darkblue', linewidth=1, alpha=0.8)
    ax.axhline(1, color='gray', linestyle='-', alpha=0.5, label='Normal Volume')
    ax.axhline(1.5, color='orange', linestyle='--', alpha=0.7, label='High Volume (1.5x)')
    ax.axhline(2, color='red', linestyle='--', alpha=0.7, label='Very High Volume (2x)')
    
    ax.set_title(f'{symbol_short} - Volume Ratio')
    ax.set_ylabel('Volume Ratio (vs SMA)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_gaps_analysis(ax, df, symbol_short):
    """Análisis de gaps de precio"""
    if df.empty or 'gap_pct' not in df.columns:
        ax.text(0.5, 0.5, 'No Gaps Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    gaps = df['gap_pct'].dropna()
    if len(gaps) == 0:
        ax.text(0.5, 0.5, 'No Valid Gaps', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Histograma de gaps
    ax.hist(gaps, bins=50, alpha=0.7, color='lightgreen', density=True)
    
    # Estadísticas
    mean_gap = gaps.mean()
    std_gap = gaps.std()
    
    ax.axvline(mean_gap, color='red', linestyle='--', label=f'Mean: {mean_gap:.3f}%')
    ax.axvline(mean_gap + 2*std_gap, color='orange', linestyle=':', alpha=0.7, label=f'+2σ: {mean_gap+2*std_gap:.3f}%')
    ax.axvline(mean_gap - 2*std_gap, color='orange', linestyle=':', alpha=0.7, label=f'-2σ: {mean_gap-2*std_gap:.3f}%')
    
    ax.set_title(f'{symbol_short} - Price Gaps Distribution')
    ax.set_xlabel('Gap (%)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_comprehensive_stats_table(ax, symbol_short, data_info, df_ohlcv, df_orderbook, df_funding):
    """Tabla de estadísticas comprehensivas"""
    ax.axis('off')
    
    # Recopilar estadísticas
    stats_text = f"{symbol_short} - COMPREHENSIVE STATISTICS\n\n"
    
    # OHLCV Stats
    ohlcv_info = data_info.get('ohlcv', {})
    if ohlcv_info.get('total_records', 0) > 0:
        stats_text += f"OHLCV DATA:\n"
        stats_text += f"  Records: {ohlcv_info['total_records']:,}\n"
        stats_text += f"  Trading Days: {ohlcv_info['trading_days']:,}\n"
        stats_text += f"  Avg Volume: {ohlcv_info['avg_volume']:,.0f}\n"
        stats_text += f"  Price Range: ${ohlcv_info['min_price']:.6f} - ${ohlcv_info['max_price']:.6f}\n"
        stats_text += f"  Avg Daily Range: {ohlcv_info['avg_range_pct']:.2f}%\n"
        
        if not df_ohlcv.empty and 'returns' in df_ohlcv.columns:
            returns = df_ohlcv['returns'].dropna()
            if len(returns) > 0:
                stats_text += f"  Daily Volatility: {returns.std()*100:.3f}%\n"
                stats_text += f"  Sharpe Ratio: {(returns.mean()/returns.std()):.2f}\n"
        stats_text += "\n"
    
    # Orderbook Stats
    orderbook_info = data_info.get('orderbook', {})
    if orderbook_info.get('total_records', 0) > 0:
        stats_text += f"ORDERBOOK DATA:\n"
        stats_text += f"  Snapshots: {orderbook_info['total_records']:,}\n"
        stats_text += f"  Valid Quotes: {orderbook_info['valid_quotes']:,}\n"
        stats_text += f"  Avg Spread: {orderbook_info['avg_spread_pct']:.4f}%\n"
        
        if not df_orderbook.empty:
            spreads = df_orderbook['spread_pct'].dropna()
            if len(spreads) > 0:
                stats_text += f"  Spread P95: {spreads.quantile(0.95):.4f}%\n"
                stats_text += f"  Spread StdDev: {spreads.std():.4f}%\n"
        stats_text += "\n"
    
    # Funding Stats
    if "PERP_" in (symbol_short + "_"):
        funding_info = data_info.get('funding_rates', {})
        if funding_info.get('total_records', 0) > 0:
            stats_text += f"FUNDING RATES:\n"
            stats_text += f"  Records: {funding_info['total_records']:,}\n"
            stats_text += f"  Avg Rate: {funding_info['avg_rate']*100:.4f}%\n"
            stats_text += f"  Rate Range: {funding_info['min_rate']*100:.4f}% - {funding_info['max_rate']*100:.4f}%\n"
            stats_text += f"  Rate StdDev: {funding_info['std_rate']*100:.4f}%\n"
            stats_text += "\n"
    
    # Technical Stats
    if not df_ohlcv.empty:
        stats_text += f"TECHNICAL INDICATORS:\n"
        
        if 'rsi' in df_ohlcv.columns:
            rsi_current = df_ohlcv['rsi'].dropna().iloc[-1] if len(df_ohlcv['rsi'].dropna()) > 0 else 0
            stats_text += f"  Current RSI: {rsi_current:.1f}\n"
        
        if 'bb_width' in df_ohlcv.columns:
            bb_width = df_ohlcv['bb_width'].dropna()
            if len(bb_width) > 0:
                stats_text += f"  Avg BB Width: {bb_width.mean():.4f}\n"
        
        if 'atr_14' in df_ohlcv.columns:
            atr = df_ohlcv['atr_14'].dropna()
            if len(atr) > 0:
                current_price = df_ohlcv['close'].iloc[-1] if len(df_ohlcv) > 0 else 1
                atr_pct = (atr.iloc[-1] / current_price * 100) if len(atr) > 0 else 0
                stats_text += f"  Current ATR: {atr_pct:.2f}%\n"
        
        stats_text += "\n"
    
    # Quality Assessment
    stats_text += f"DATA QUALITY ASSESSMENT:\n"
    
    # OHLCV Quality
    if ohlcv_info.get('total_records', 0) > 100000:
        ohlcv_quality = "EXCELLENT"
    elif ohlcv_info.get('total_records', 0) > 10000:
        ohlcv_quality = "GOOD"
    elif ohlcv_info.get('total_records', 0) > 1000:
        ohlcv_quality = "FAIR"
    else:
        ohlcv_quality = "POOR"
    
    stats_text += f"  OHLCV Quality: {ohlcv_quality}\n"
    
    # Orderbook Quality
    avg_spread = orderbook_info.get('avg_spread_pct', 0)
    if avg_spread < 0.01:
        ob_quality = "EXCELLENT"
    elif avg_spread < 0.05:
        ob_quality = "GOOD"
    elif avg_spread < 0.1:
        ob_quality = "FAIR"
    else:
        ob_quality = "POOR"
    
    stats_text += f"  Orderbook Quality: {ob_quality}\n"
    
    # Overall recommendation
    if ohlcv_quality in ["EXCELLENT", "GOOD"] and ob_quality in ["EXCELLENT", "GOOD"]:
        recommendation = "READY for algorithmic trading"
    elif ohlcv_quality in ["EXCELLENT", "GOOD"]:
        recommendation = "SUITABLE for price-based strategies"
    else:
        recommendation = "NEEDS MORE DATA for reliable trading"
    
    stats_text += f"  Recommendation: {recommendation}\n"
    
    # Mostrar stats
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=9, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="🚀 Comprehensive OHLCV Analysis (No Mark Prices)")
    parser.add_argument("--symbol", type=str, help="Analyze specific symbol only")
    
    args = parser.parse_args()
    
    log.info("🚀 Iniciando análisis OHLCV comprensivo (SIN mark prices)...")
    
    try:
        # Obtener símbolos
        if args.symbol:
            symbols = [args.symbol]
        else:
            try:
                # Intentar obtener de la BD primero
                symbols = settings.get_symbols_from_db()
                if not symbols:
                    # Fallback a YAML
                    active_pairs = settings.get_active_pairs()
                    symbols = list(set([pair.symbol1 for pair in active_pairs] + [pair.symbol2 for pair in active_pairs]))
            except:
                symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
        
        log.info(f"🎯 Analizando {len(symbols)} símbolos con análisis OHLCV comprensivo")
        
        # Análisis comprensivo por símbolo
        for symbol in symbols:
            log.info(f"\n{'='*80}")
            log.info(f"📊 PROCESANDO ANÁLISIS OHLCV: {symbol}")
            log.info(f"{'='*80}")
            
            create_ohlcv_comprehensive_analysis(symbol)
        
        log.info(f"\n🎉 Análisis OHLCV COMPRENSIVO completado!")
        log.info(f"📊 Gráficas guardadas en: plots/")
        for symbol in symbols:
            symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
            log.info(f"  ✅ {symbol_short}_ohlcv_comprehensive_analysis.png")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Error en análisis OHLCV: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)