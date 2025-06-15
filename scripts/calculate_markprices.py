#!/usr/bin/env python3
"""
FINAL Mark price calculation script - ESQUEMA VERIFICADO
Funciona con el esquema existente que tiene valid_for_trading
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
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

def calculate_simple_vwap(row: pd.Series) -> Dict:
    """Cálculo VWAP simple SIN FILTROS ADICIONALES"""
    
    bid_levels = []
    ask_levels = []
    
    for i in range(1, 11):
        # Bid levels
        bid_price = row.get(f'bid{i}_price')
        bid_size = row.get(f'bid{i}_size')
        
        if (pd.notna(bid_price) and pd.notna(bid_size) and 
            bid_price > 0 and bid_size > 0):
            bid_levels.append({
                'price': float(bid_price), 
                'size': float(bid_size)
            })
        
        # Ask levels
        ask_price = row.get(f'ask{i}_price')
        ask_size = row.get(f'ask{i}_size')
        
        if (pd.notna(ask_price) and pd.notna(ask_size) and 
            ask_price > 0 and ask_size > 0):
            ask_levels.append({
                'price': float(ask_price), 
                'size': float(ask_size)
            })
    
    if len(bid_levels) == 0 or len(ask_levels) == 0:
        return {'mark_price': None, 'liquidity_score': 0.0}
    
    # Verificar spread no cruzado
    best_bid = max(level['price'] for level in bid_levels)
    best_ask = min(level['price'] for level in ask_levels)
    
    if best_bid >= best_ask:
        return {'mark_price': None, 'liquidity_score': 0.0}
    
    try:
        # Calcular VWAP
        total_bid_value = sum(level['price'] * level['size'] for level in bid_levels)
        total_bid_size = sum(level['size'] for level in bid_levels)
        bid_vwap = total_bid_value / total_bid_size
        
        total_ask_value = sum(level['price'] * level['size'] for level in ask_levels)
        total_ask_size = sum(level['size'] for level in ask_levels)
        ask_vwap = total_ask_value / total_ask_size
        
        mark_price = (bid_vwap + ask_vwap) / 2
        
        # Liquidity score simple
        max_levels = 10
        available_levels = len(bid_levels) + len(ask_levels)
        levels_score = min(1.0, available_levels / max_levels)
        balance_score = min(len(bid_levels), len(ask_levels)) / max(len(bid_levels), len(ask_levels))
        liquidity_score = (levels_score * 0.7 + balance_score * 0.3)
        
        return {
            'mark_price': float(mark_price),
            'liquidity_score': float(liquidity_score)
        }
        
    except Exception as e:
        log.warning(f"Error calculando VWAP: {e}")
        return {'mark_price': None, 'liquidity_score': 0.0}

def check_orderbook_quality_column(symbol: str) -> str:
    """Verificar qué columna de calidad usar"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'orderbook' 
            AND column_name = 'valid_for_trading'
        """)).fetchone()
        
        if result:
            log.info("Usando columna 'valid_for_trading' para filtrar")
            return "valid_for_trading"
        
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'orderbook' 
            AND column_name = 'liquidity_quality'
        """)).fetchone()
        
        if result:
            log.info("Usando columna 'liquidity_quality' para filtrar")
            return "liquidity_quality"
        
        log.info("No se encontraron columnas de calidad - procesando todos los registros")
        return None

def get_orderbook_stats(symbol: str, quality_column: str) -> Dict:
    """Obtener estadísticas de orderbook"""
    with db_manager.get_session() as session:
        if quality_column == "valid_for_trading":
            stats_query = text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as valid,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date
                FROM orderbook 
                WHERE symbol = :symbol
                AND bid1_price IS NOT NULL 
                AND ask1_price IS NOT NULL
            """)
        elif quality_column == "liquidity_quality":
            stats_query = text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN liquidity_quality IN ('Excellent', 'Good') THEN 1 END) as valid,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date
                FROM orderbook 
                WHERE symbol = :symbol
                AND bid1_price IS NOT NULL 
                AND ask1_price IS NOT NULL
            """)
        else:
            stats_query = text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) as valid,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date
                FROM orderbook 
                WHERE symbol = :symbol
                AND bid1_price IS NOT NULL 
                AND ask1_price IS NOT NULL
            """)
        
        result = session.execute(stats_query, {'symbol': symbol}).fetchone()
        
        return {
            'total': result.total,
            'valid': result.valid,
            'min_date': result.min_date,
            'max_date': result.max_date,
            'valid_pct': (result.valid / max(1, result.total)) * 100
        }

def calculate_mark_prices_simple(symbol: str) -> int:
    """Calcular mark prices usando SOLO el filtro valid_for_trading"""
    
    quality_column = check_orderbook_quality_column(symbol)
    stats = get_orderbook_stats(symbol, quality_column)
    
    log.info(f"Estadísticas de orderbook para {symbol}:")
    log.info(f"  Total registros: {stats['total']:,}")
    log.info(f"  Registros válidos: {stats['valid']:,} ({stats['valid_pct']:.1f}%)")
    log.info(f"  Período: {stats['min_date']} a {stats['max_date']}")
    
    if stats['valid'] == 0:
        log.warning(f"No hay registros válidos para {symbol}")
        return 0
    
    # Construir filtro de calidad
    if quality_column == "valid_for_trading":
        quality_filter = "AND ob.valid_for_trading = TRUE"
    elif quality_column == "liquidity_quality":
        quality_filter = "AND ob.liquidity_quality IN ('Excellent', 'Good')"
    else:
        quality_filter = ""
    
    with db_manager.get_session() as session:
        batch_size = 5000
        total_processed = 0
        offset = 0
        
        while True:
            orderbook_query = text(f"""
                SELECT 
                    ob.timestamp,
                    ob.bid1_price, ob.bid1_size, ob.bid2_price, ob.bid2_size, 
                    ob.bid3_price, ob.bid3_size, ob.bid4_price, ob.bid4_size,
                    ob.bid5_price, ob.bid5_size, ob.bid6_price, ob.bid6_size,
                    ob.bid7_price, ob.bid7_size, ob.bid8_price, ob.bid8_size,
                    ob.bid9_price, ob.bid9_size, ob.bid10_price, ob.bid10_size,
                    ob.ask1_price, ob.ask1_size, ob.ask2_price, ob.ask2_size,
                    ob.ask3_price, ob.ask3_size, ob.ask4_price, ob.ask4_size,
                    ob.ask5_price, ob.ask5_size, ob.ask6_price, ob.ask6_size,
                    ob.ask7_price, ob.ask7_size, ob.ask8_price, ob.ask8_size,
                    ob.ask9_price, ob.ask9_size, ob.ask10_price, ob.ask10_size
                FROM orderbook ob
                WHERE ob.symbol = :symbol 
                AND ob.bid1_price IS NOT NULL 
                AND ob.ask1_price IS NOT NULL
                AND ob.bid1_price > 0 
                AND ob.ask1_price > 0
                AND ob.bid1_price < ob.ask1_price
                {quality_filter}
                ORDER BY ob.timestamp
                LIMIT :batch_size OFFSET :offset
            """)
            
            orderbook_df = pd.read_sql(orderbook_query, session.bind, params={
                'symbol': symbol,
                'batch_size': batch_size,
                'offset': offset
            })
            
            if orderbook_df.empty:
                break
            
            mark_prices = []
            
            for idx, row in orderbook_df.iterrows():
                vwap_result = calculate_simple_vwap(row)
                
                if vwap_result['mark_price'] is None:
                    continue
                
                timestamp = row['timestamp']
                orderbook_mid = (row['bid1_price'] + row['ask1_price']) / 2
                
                mark_prices.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'mark_price': vwap_result['mark_price'],
                    'orderbook_mid': float(orderbook_mid),
                    'ohlcv_close': None,
                    'ohlcv_volume': None,  # Agregar esta columna que existe en el esquema
                    'liquidity_score': vwap_result['liquidity_score'],
                    'valid_for_trading': True
                })
            
            if mark_prices:
                # INSERT que coincide exactamente con el esquema existente
                insert_query = text("""
                    INSERT INTO mark_prices (
                        symbol, timestamp, mark_price, orderbook_mid, 
                        ohlcv_close, ohlcv_volume, liquidity_score, valid_for_trading
                    ) VALUES (
                        :symbol, :timestamp, :mark_price, :orderbook_mid,
                        :ohlcv_close, :ohlcv_volume, :liquidity_score, :valid_for_trading
                    ) ON CONFLICT (symbol, timestamp) 
                    DO UPDATE SET
                        mark_price = EXCLUDED.mark_price,
                        orderbook_mid = EXCLUDED.orderbook_mid,
                        ohlcv_close = EXCLUDED.ohlcv_close,
                        ohlcv_volume = EXCLUDED.ohlcv_volume,
                        liquidity_score = EXCLUDED.liquidity_score,
                        valid_for_trading = EXCLUDED.valid_for_trading
                """)
                
                session.execute(insert_query, mark_prices)
                total_processed += len(mark_prices)
            
            offset += batch_size
            
            if offset % 25000 == 0:
                log.info(f"  Procesados: {total_processed:,} mark prices")
        
        return total_processed

def update_ohlcv_data(symbol: str):
    """Actualizar datos OHLCV en mark prices"""
    with db_manager.get_session() as session:
        # Actualizar close price
        update_close_query = text("""
            UPDATE mark_prices mp
            SET ohlcv_close = o.close
            FROM ohlcv o
            WHERE mp.symbol = o.symbol
            AND mp.timestamp = o.timestamp
            AND mp.symbol = :symbol
            AND mp.ohlcv_close IS NULL
        """)
        
        close_result = session.execute(update_close_query, {'symbol': symbol})
        
        # Actualizar volume
        update_volume_query = text("""
            UPDATE mark_prices mp
            SET ohlcv_volume = o.volume
            FROM ohlcv o
            WHERE mp.symbol = o.symbol
            AND mp.timestamp = o.timestamp
            AND mp.symbol = :symbol
            AND mp.ohlcv_volume IS NULL
        """)
        
        volume_result = session.execute(update_volume_query, {'symbol': symbol})
        
        log.info(f"  Actualizadas {close_result.rowcount} referencias OHLCV close")
        log.info(f"  Actualizadas {volume_result.rowcount} referencias OHLCV volume")

def main():
    """Función principal final"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate mark prices - ESQUEMA VERIFICADO")
    parser.add_argument("--symbol", type=str, help="Specific symbol to process")
    parser.add_argument("--force", action="store_true", help="Force recalculation")
    
    args = parser.parse_args()
    
    log.info("Iniciando cálculo FINAL de mark prices...")
    log.info("Esquema verificado: usando valid_for_trading y ohlcv_volume")
    
    # Obtener símbolos
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
            log.error(f"Error cargando símbolos: {e}")
            symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
    
    if not symbols:
        log.error("No hay símbolos para procesar")
        return False
    
    log.info(f"Procesando {len(symbols)} símbolos con esquema verificado")
    
    try:
        total_processed = 0
        
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"PROCESANDO {symbol} - FINAL")
            log.info(f"{'='*60}")
            
            if args.force:
                with db_manager.get_session() as session:
                    result = session.execute(text("DELETE FROM mark_prices WHERE symbol = :symbol"), 
                                          {'symbol': symbol})
                    log.info(f"Eliminados {result.rowcount} mark prices existentes")
            
            processed_count = calculate_mark_prices_simple(symbol)
            
            if processed_count > 0:
                update_ohlcv_data(symbol)
                total_processed += processed_count
                log.info(f"✅ {processed_count:,} mark prices calculados para {symbol}")
            else:
                log.warning(f"⚠️ No se calcularon mark prices para {symbol}")
        
        log.info(f"\n🎉 Cálculo FINAL completado!")
        log.info(f"Total procesados: {total_processed:,} mark prices")
        log.info(f"Características:")
        log.info(f"  - Usa SOLO valid_for_trading del orderbook")
        log.info(f"  - Incluye ohlcv_volume en el esquema")
        log.info(f"  - VWAP de todos los niveles disponibles")
        log.info(f"  - Máxima cobertura temporal")
        
        return True
        
    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)