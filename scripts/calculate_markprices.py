#!/usr/bin/env python3
"""
Mark price calculation script - FULL DATA VERSION
Calcula mark prices VWAP para TODOS los datos disponibles
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import db_manager
from src.database.migrations import check_mark_prices_schema
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

def ensure_simplified_mark_prices_schema():
    """Ensure simplified mark prices table exists"""
    log.info("Checking simplified mark prices schema...")
    
    with db_manager.get_session() as session:
        # Check if table exists
        result = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'mark_prices'
        """)).fetchone()
        
        if not result:
            log.info("Creating simplified mark prices table...")
            # Create simplified table
            session.execute(text("""
                CREATE TABLE mark_prices (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    mark_price FLOAT NOT NULL,
                    orderbook_mid FLOAT NOT NULL,
                    ohlcv_close FLOAT,
                    ohlcv_volume FLOAT,
                    liquidity_score FLOAT NOT NULL,
                    valid_for_trading BOOLEAN NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                );
            """))
            
            # Create indexes
            session.execute(text("""
                CREATE INDEX idx_markprice_symbol_timestamp_simple 
                ON mark_prices(symbol, timestamp);
            """))
            session.execute(text("""
                CREATE INDEX idx_markprice_valid_trading 
                ON mark_prices(symbol, valid_for_trading, timestamp);
            """))
            
            log.info("Simplified mark prices table created")
        else:
            log.info("Mark prices table already exists")

def calculate_flexible_vwap(row: pd.Series, min_volume_usd: float = 50.0) -> Dict:
    """
    Calcula VWAP flexible con parámetros más realistas
    
    Args:
        row: Fila del orderbook con niveles bid/ask
        min_volume_usd: Volumen mínimo para incluir nivel (reducido a $50)
    
    Returns:
        Dict con mark_price y liquidity_score
    """
    
    # Extraer niveles válidos
    bid_levels = []
    ask_levels = []
    
    for i in range(1, 11):  # 10 niveles máximo
        # Bid levels
        bid_price = row.get(f'bid{i}_price')
        bid_size = row.get(f'bid{i}_size')
        
        if pd.notna(bid_price) and pd.notna(bid_size) and bid_price > 0 and bid_size > 0:
            volume_usd = bid_price * bid_size
            if volume_usd >= min_volume_usd:
                bid_levels.append({'price': bid_price, 'size': bid_size, 'volume': volume_usd})
        
        # Ask levels
        ask_price = row.get(f'ask{i}_price')
        ask_size = row.get(f'ask{i}_size')
        
        if pd.notna(ask_price) and pd.notna(ask_size) and ask_price > 0 and ask_size > 0:
            volume_usd = ask_price * ask_size
            if volume_usd >= min_volume_usd:
                ask_levels.append({'price': ask_price, 'size': ask_size, 'volume': volume_usd})
    
    # Si no hay suficientes niveles con el volumen mínimo, usar nivel 1 directamente
    if not bid_levels or not ask_levels:
        bid1_price = row.get('bid1_price')
        ask1_price = row.get('ask1_price')
        bid1_size = row.get('bid1_size')
        ask1_size = row.get('ask1_size')
        
        # Verificar que al menos tengamos el nivel 1
        if (pd.notna(bid1_price) and pd.notna(ask1_price) and 
            pd.notna(bid1_size) and pd.notna(ask1_size) and
            bid1_price > 0 and ask1_price > 0 and
            bid1_size > 0 and ask1_size > 0):
            
            # Usar mid price como fallback
            mark_price = (bid1_price + ask1_price) / 2
            
            # Score bajo porque solo usamos nivel 1
            liquidity_score = 0.3
            
            return {
                'mark_price': mark_price,
                'liquidity_score': liquidity_score
            }
        else:
            return {'mark_price': None, 'liquidity_score': 0.0}
    
    # Calcular VWAP para cada lado
    # Bid VWAP
    total_bid_value = sum(level['price'] * level['size'] for level in bid_levels)
    total_bid_size = sum(level['size'] for level in bid_levels)
    bid_vwap = total_bid_value / total_bid_size
    
    # Ask VWAP
    total_ask_value = sum(level['price'] * level['size'] for level in ask_levels)
    total_ask_size = sum(level['size'] for level in ask_levels)
    ask_vwap = total_ask_value / total_ask_size
    
    # Mark price = promedio simple de bid/ask VWAP
    mark_price = (bid_vwap + ask_vwap) / 2
    
    # Calcular liquidity score mejorado
    total_bid_volume = sum(level['volume'] for level in bid_levels)
    total_ask_volume = sum(level['volume'] for level in ask_levels)
    total_volume = total_bid_volume + total_ask_volume
    
    # Score basado en:
    # - Número de niveles activos (40%)
    # - Volumen total (40%) 
    # - Balance bid/ask (20%)
    levels_score = min(1.0, (len(bid_levels) + len(ask_levels)) / 6)  # Normalizado a 3 niveles cada lado
    volume_score = min(1.0, total_volume / 1000)  # Normalizado a $1K (reducido de $20K)
    
    # Balance score (penalizar si un lado tiene mucho más volumen)
    if total_volume > 0:
        balance_ratio = min(total_bid_volume, total_ask_volume) / max(total_bid_volume, total_ask_volume)
        balance_score = balance_ratio
    else:
        balance_score = 0.0
    
    liquidity_score = (levels_score * 0.4 + volume_score * 0.4 + balance_score * 0.2)
    
    return {
        'mark_price': mark_price,
        'liquidity_score': liquidity_score
    }

def get_data_gaps(symbol: str) -> List[Tuple[datetime, datetime]]:
    """Identificar gaps en los datos de mark prices - PROCESAR TODOS LOS DATOS"""
    with db_manager.get_session() as session:
        # Obtener TODOS los datos disponibles en orderbook
        orderbook_range = session.execute(text("""
            SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
            FROM orderbook 
            WHERE symbol = :symbol 
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
        """), {'symbol': symbol}).fetchone()
        
        if not orderbook_range.min_ts:
            log.warning(f"No orderbook data for {symbol}")
            return []
        
        # Obtener rango de mark prices existentes
        mark_range = session.execute(text("""
            SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
            FROM mark_prices 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        gaps = []
        
        if not mark_range.min_ts:
            # No hay mark prices - procesar TODO (sin límite de días)
            gaps.append((orderbook_range.min_ts, orderbook_range.max_ts))
            log.info(f"No existing mark prices - processing FULL range: {orderbook_range.min_ts} to {orderbook_range.max_ts}")
        else:
            # Verificar gaps al inicio y final (sin límites)
            if orderbook_range.min_ts < mark_range.min_ts:
                gaps.append((orderbook_range.min_ts, mark_range.min_ts - timedelta(minutes=1)))
                log.info(f"Gap at beginning: {orderbook_range.min_ts} to {mark_range.min_ts}")
            
            if orderbook_range.max_ts > mark_range.max_ts:
                gaps.append((mark_range.max_ts + timedelta(minutes=1), orderbook_range.max_ts))
                log.info(f"New data at end: {mark_range.max_ts} to {orderbook_range.max_ts}")
        
        return gaps

def calculate_mark_prices_period(symbol: str, start_date: datetime, end_date: datetime) -> int:
    """Calcular mark prices para un período específico - PROCESAMIENTO POR CHUNKS"""
    duration_days = (end_date - start_date).days
    log.info(f"Calculating mark prices for {symbol}: {duration_days} days ({start_date.date()} to {end_date.date()})")
    
    # Contar registros totales primero
    with db_manager.get_session() as session:
        count_result = session.execute(text("""
            SELECT COUNT(*) as total
            FROM orderbook 
            WHERE symbol = :symbol 
            AND timestamp >= :start_date 
            AND timestamp <= :end_date
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
        """), {'symbol': symbol, 'start_date': start_date, 'end_date': end_date}).fetchone()
        
        total_records = count_result.total
        log.info(f"Total orderbook records to process: {total_records:,}")
        
        if total_records == 0:
            log.warning(f"No orderbook data found for {symbol} in period")
            return 0
    
    # Procesar en chunks de 7 días para evitar problemas de memoria
    chunk_days = 7
    total_processed = 0
    current_start = start_date
    
    while current_start < end_date:
        chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
        
        chunk_processed = _process_chunk(symbol, current_start, chunk_end)
        total_processed += chunk_processed
        
        progress = (current_start - start_date).days / duration_days * 100 if duration_days > 0 else 100
        log.info(f"Chunk {current_start.date()} to {chunk_end.date()}: {chunk_processed:,} records | Progress: {progress:.1f}%")
        
        current_start = chunk_end + timedelta(seconds=1)
    
    log.info(f"Completed processing {symbol}: {total_processed:,} mark prices calculated")
    return total_processed

def _process_chunk(symbol: str, start_date: datetime, end_date: datetime) -> int:
    """Procesar un chunk de datos"""
    
    with db_manager.get_session() as session:
        # Cargar datos de orderbook (SIN LÍMITE)
        orderbook_query = text("""
            SELECT 
                ob.timestamp,
                ob.bid1_price, ob.bid1_size, ob.bid2_price, ob.bid2_size, 
                ob.bid3_price, ob.bid3_size, ob.bid4_price, ob.bid4_size,
                ob.bid5_price, ob.bid5_size,
                ob.ask1_price, ob.ask1_size, ob.ask2_price, ob.ask2_size,
                ob.ask3_price, ob.ask3_size, ob.ask4_price, ob.ask4_size,
                ob.ask5_price, ob.ask5_size,
                COALESCE(ob.valid_for_trading, true) as valid_for_trading,
                COALESCE(ob.liquidity_quality, 'Unknown') as liquidity_quality
            FROM orderbook ob
            WHERE ob.symbol = :symbol 
            AND ob.timestamp >= :start_date 
            AND ob.timestamp <= :end_date
            AND ob.bid1_price IS NOT NULL 
            AND ob.ask1_price IS NOT NULL
            ORDER BY ob.timestamp
        """)
        
        orderbook_df = pd.read_sql(orderbook_query, session.bind, params={
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        
        if orderbook_df.empty:
            return 0
        
        # Cargar datos OHLCV para el mismo período
        ohlcv_query = text("""
            SELECT timestamp, close, volume
            FROM ohlcv 
            WHERE symbol = :symbol 
            AND timestamp >= :start_date 
            AND timestamp <= :end_date
            ORDER BY timestamp
        """)
        
        ohlcv_df = pd.read_sql(ohlcv_query, session.bind, params={
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        
        # Convertir OHLCV a diccionario para lookup rápido
        ohlcv_dict = {}
        if not ohlcv_df.empty:
            ohlcv_df.set_index('timestamp', inplace=True)
            ohlcv_dict = ohlcv_df.to_dict('index')
        
        # Procesar mark prices
        mark_prices = []
        
        for idx, row in orderbook_df.iterrows():
            # Calcular VWAP flexible
            vwap_result = calculate_flexible_vwap(row, min_volume_usd=50.0)
            
            if vwap_result['mark_price'] is None:
                continue
            
            timestamp = row['timestamp']
            
            # Calcular orderbook mid
            orderbook_mid = (row['bid1_price'] + row['ask1_price']) / 2
            
            # Obtener datos OHLCV correspondientes
            ohlcv_data = ohlcv_dict.get(timestamp, {})
            ohlcv_close = ohlcv_data.get('close')
            ohlcv_volume = ohlcv_data.get('volume')
            
            # Usar valid_for_trading del orderbook
            valid_for_trading = bool(row.get('valid_for_trading', True))
            
            mark_prices.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'mark_price': float(vwap_result['mark_price']),
                'orderbook_mid': float(orderbook_mid),
                'ohlcv_close': float(ohlcv_close) if ohlcv_close and pd.notna(ohlcv_close) else None,
                'ohlcv_volume': float(ohlcv_volume) if ohlcv_volume and pd.notna(ohlcv_volume) else None,
                'liquidity_score': float(vwap_result['liquidity_score']),
                'valid_for_trading': valid_for_trading
            })
        
        if not mark_prices:
            return 0
        
        # Insertar en lotes más grandes para eficiencia
        batch_size = 5000
        
        for i in range(0, len(mark_prices), batch_size):
            batch = mark_prices[i:i + batch_size]
            
            insert_query = text("""
                INSERT INTO mark_prices (
                    symbol, timestamp, mark_price, orderbook_mid, ohlcv_close,
                    ohlcv_volume, liquidity_score, valid_for_trading
                ) VALUES (
                    :symbol, :timestamp, :mark_price, :orderbook_mid, :ohlcv_close,
                    :ohlcv_volume, :liquidity_score, :valid_for_trading
                ) ON CONFLICT (symbol, timestamp) 
                DO UPDATE SET
                    mark_price = EXCLUDED.mark_price,
                    orderbook_mid = EXCLUDED.orderbook_mid,
                    ohlcv_close = EXCLUDED.ohlcv_close,
                    ohlcv_volume = EXCLUDED.ohlcv_volume,
                    liquidity_score = EXCLUDED.liquidity_score,
                    valid_for_trading = EXCLUDED.valid_for_trading
            """)
            
            session.execute(insert_query, batch)
        
        return len(mark_prices)

def calculate_mark_prices_incremental(symbol: str) -> int:
    """Calcular mark prices incrementalmente"""
    log.info(f"Starting incremental mark price calculation for {symbol}")
    
    # Identificar gaps
    gaps = get_data_gaps(symbol)
    
    if not gaps:
        log.info(f"No missing mark price data for {symbol}")
        return 0
    
    total_processed = 0
    for start_date, end_date in gaps:
        processed = calculate_mark_prices_period(symbol, start_date, end_date)
        total_processed += processed
    
    return total_processed

def get_summary_stats(symbol: str) -> Dict:
    """Obtener estadísticas resumidas"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as valid_trading,
                COUNT(CASE WHEN ohlcv_close IS NOT NULL THEN 1 END) as with_ohlcv,
                AVG(liquidity_score) as avg_liquidity,
                MIN(timestamp) as min_timestamp,
                MAX(timestamp) as max_timestamp,
                AVG(mark_price) as avg_mark_price,
                STDDEV(mark_price) as std_mark_price
            FROM mark_prices 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        return {
            'symbol': symbol,
            'total_records': result.total_records or 0,
            'valid_trading': result.valid_trading or 0,
            'valid_trading_pct': (result.valid_trading or 0) / max(1, result.total_records or 1) * 100,
            'with_ohlcv': result.with_ohlcv or 0,
            'with_ohlcv_pct': (result.with_ohlcv or 0) / max(1, result.total_records or 1) * 100,
            'avg_liquidity': result.avg_liquidity or 0,
            'time_range': {
                'start': result.min_timestamp,
                'end': result.max_timestamp
            },
            'price_stats': {
                'mean': result.avg_mark_price or 0,
                'std': result.std_mark_price or 0
            }
        }

def generate_full_report(summaries: Dict):
    """Generar reporte completo"""
    log.info("\n" + "="*80)
    log.info("FULL DATA MARK PRICES CALCULATION REPORT")
    log.info("="*80)
    
    total_records = sum(s['total_records'] for s in summaries.values())
    total_valid = sum(s['valid_trading'] for s in summaries.values())
    
    log.info(f"Processed {len(summaries)} symbols")
    log.info(f"Total mark prices calculated: {total_records:,}")
    log.info(f"Valid for trading: {total_valid:,} ({total_valid/max(1,total_records)*100:.1f}%)")
    
    for symbol, summary in summaries.items():
        symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
        log.info(f"\n{symbol_short} Summary:")
        log.info(f"  Records: {summary['total_records']:,}")
        log.info(f"  Valid for trading: {summary['valid_trading']:,} ({summary['valid_trading_pct']:.1f}%)")
        log.info(f"  With OHLCV data: {summary['with_ohlcv']:,} ({summary['with_ohlcv_pct']:.1f}%)")
        log.info(f"  Avg liquidity score: {summary['avg_liquidity']:.3f}")
        log.info(f"  Price range: ${summary['price_stats']['mean']:.6f} ±{summary['price_stats']['std']:.6f}")
        log.info(f"  Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")
    
    log.info(f"\nFull Data Processing Features:")
    log.info(f"- Processes ALL available orderbook data")
    log.info(f"- Chunk-based processing for memory efficiency")
    log.info(f"- No time limits - complete historical coverage")
    log.info(f"- Ready for comprehensive backtesting")

def main():
    """Función principal para procesar TODOS los datos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate FULL DATA VWAP mark prices")
    parser.add_argument("--symbol", type=str, help="Specific symbol to process")
    parser.add_argument("--force", action="store_true", help="Force recalculation (truncate table)")
    
    args = parser.parse_args()
    
    log.info("Starting FULL DATA VWAP mark price calculation...")
    
    # Ensure schema
    ensure_simplified_mark_prices_schema()
    
    # Get symbols
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
            symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
    
    if not symbols:
        log.error("No symbols to process")
        return False
    
    log.info(f"Processing FULL DATA mark prices for {len(symbols)} symbols")
    
    summaries = {}
    
    try:
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"PROCESSING {symbol} - FULL DATA")
            log.info(f"{'='*60}")
            
            if args.force:
                # Truncate existing data
                with db_manager.get_session() as session:
                    session.execute(text("DELETE FROM mark_prices WHERE symbol = :symbol"), {'symbol': symbol})
                    log.info(f"Deleted existing mark prices for {symbol}")
            
            # Calculate mark prices for ALL data
            processed_count = calculate_mark_prices_incremental(symbol)
            
            if processed_count > 0:
                log.info(f"Successfully processed {processed_count:,} mark prices for {symbol}")
                summaries[symbol] = get_summary_stats(symbol)
            else:
                log.warning(f"No mark prices processed for {symbol}")
        
        # Generate report
        if summaries:
            generate_full_report(summaries)
        
        log.info(f"\nFull data mark price calculation completed!")
        log.info(f"Processed: {len(summaries)}/{len(symbols)} symbols")
        log.info(f"Now you have mark prices for the complete historical dataset!")
        
        return True
        
    except Exception as e:
        log.error(f"Mark price calculation failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)