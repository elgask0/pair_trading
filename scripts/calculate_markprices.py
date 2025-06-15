#!/usr/bin/env python3
"""
ULTRA FAST Mark price calculation script - FULLY AUTOMATED VERSION
Sin intervención manual, modo automático completo
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

def calculate_mark_prices_ultrafast(symbol: str) -> int:
    """Calcular mark prices usando SQL directo - ULTRA RÁPIDO"""
    
    log.info(f"Iniciando cálculo ULTRA RÁPIDO para {symbol}...")
    
    with db_manager.get_session() as session:
        # Primero obtener estadísticas
        stats_query = text("""
            SELECT 
                COUNT(*) as valid
            FROM orderbook 
            WHERE symbol = :symbol
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            AND valid_for_trading = TRUE
            AND bid1_price > 0 
            AND ask1_price > 0
            AND bid1_price < ask1_price
        """)
        
        stats = session.execute(stats_query, {'symbol': symbol}).fetchone()
        log.info(f"Registros a procesar: {stats.valid:,}")
        
        if stats.valid == 0:
            return 0
        
        # CALCULAR TODO EN UNA SOLA QUERY SQL - ULTRA RÁPIDO
        log.info("Ejecutando cálculo masivo en SQL...")
        
        insert_query = text("""
            INSERT INTO mark_prices (
                symbol, timestamp, mark_price, orderbook_mid, 
                ohlcv_close, ohlcv_volume, liquidity_score, valid_for_trading
            )
            SELECT 
                :symbol as symbol,
                timestamp,
                
                -- Mark price = promedio de VWAP bid y ask
                (
                    -- VWAP BID (usando niveles disponibles)
                    CASE WHEN 
                        COALESCE(bid1_size, 0) + COALESCE(bid2_size, 0) + COALESCE(bid3_size, 0) + 
                        COALESCE(bid4_size, 0) + COALESCE(bid5_size, 0) > 0
                    THEN
                        (COALESCE(bid1_price * bid1_size, 0) + COALESCE(bid2_price * bid2_size, 0) + 
                         COALESCE(bid3_price * bid3_size, 0) + COALESCE(bid4_price * bid4_size, 0) + 
                         COALESCE(bid5_price * bid5_size, 0)) /
                        (COALESCE(bid1_size, 0) + COALESCE(bid2_size, 0) + COALESCE(bid3_size, 0) + 
                         COALESCE(bid4_size, 0) + COALESCE(bid5_size, 0))
                    ELSE bid1_price
                    END
                    +
                    -- VWAP ASK (usando niveles disponibles)
                    CASE WHEN 
                        COALESCE(ask1_size, 0) + COALESCE(ask2_size, 0) + COALESCE(ask3_size, 0) + 
                        COALESCE(ask4_size, 0) + COALESCE(ask5_size, 0) > 0
                    THEN
                        (COALESCE(ask1_price * ask1_size, 0) + COALESCE(ask2_price * ask2_size, 0) + 
                         COALESCE(ask3_price * ask3_size, 0) + COALESCE(ask4_price * ask4_size, 0) + 
                         COALESCE(ask5_price * ask5_size, 0)) /
                        (COALESCE(ask1_size, 0) + COALESCE(ask2_size, 0) + COALESCE(ask3_size, 0) + 
                         COALESCE(ask4_size, 0) + COALESCE(ask5_size, 0))
                    ELSE ask1_price
                    END
                ) / 2.0 as mark_price,
                
                -- Orderbook mid simple
                (bid1_price + ask1_price) / 2.0 as orderbook_mid,
                
                -- OHLCV inicialmente NULL
                NULL as ohlcv_close,
                NULL as ohlcv_volume,
                
                -- Liquidity score basado en niveles disponibles
                LEAST(1.0, 
                    (CASE WHEN bid1_size IS NOT NULL AND bid1_size > 0 THEN 1 ELSE 0 END +
                     CASE WHEN bid2_size IS NOT NULL AND bid2_size > 0 THEN 1 ELSE 0 END +
                     CASE WHEN bid3_size IS NOT NULL AND bid3_size > 0 THEN 1 ELSE 0 END +
                     CASE WHEN bid4_size IS NOT NULL AND bid4_size > 0 THEN 1 ELSE 0 END +
                     CASE WHEN bid5_size IS NOT NULL AND bid5_size > 0 THEN 1 ELSE 0 END +
                     CASE WHEN ask1_size IS NOT NULL AND ask1_size > 0 THEN 1 ELSE 0 END +
                     CASE WHEN ask2_size IS NOT NULL AND ask2_size > 0 THEN 1 ELSE 0 END +
                     CASE WHEN ask3_size IS NOT NULL AND ask3_size > 0 THEN 1 ELSE 0 END +
                     CASE WHEN ask4_size IS NOT NULL AND ask4_size > 0 THEN 1 ELSE 0 END +
                     CASE WHEN ask5_size IS NOT NULL AND ask5_size > 0 THEN 1 ELSE 0 END) / 8.0
                ) as liquidity_score,
                
                TRUE as valid_for_trading
                
            FROM orderbook 
            WHERE symbol = :symbol
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            AND valid_for_trading = TRUE
            AND bid1_price > 0 
            AND ask1_price > 0
            AND bid1_price < ask1_price
            AND bid1_size > 0
            AND ask1_size > 0
            
            ON CONFLICT (symbol, timestamp) DO NOTHING
        """)
        
        start_time = datetime.now()
        result = session.execute(insert_query, {'symbol': symbol})
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        log.info(f"Cálculo SQL completado en {duration:.1f} segundos")
        log.info(f"Registros procesados: {result.rowcount:,}")
        
        return result.rowcount

def update_ohlcv_ultrafast(symbol: str):
    """Actualización ultra rápida de OHLCV en una sola query"""
    with db_manager.get_session() as session:
        start_time = datetime.now()
        
        update_query = text("""
            UPDATE mark_prices 
            SET 
                ohlcv_close = subq.close,
                ohlcv_volume = subq.volume
            FROM (
                SELECT 
                    timestamp,
                    close,
                    volume
                FROM ohlcv 
                WHERE symbol = :symbol
            ) subq
            WHERE mark_prices.symbol = :symbol
            AND mark_prices.timestamp = subq.timestamp
            AND (mark_prices.ohlcv_close IS NULL OR mark_prices.ohlcv_volume IS NULL)
        """)
        
        result = session.execute(update_query, {'symbol': symbol})
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        log.info(f"OHLCV actualizado en {duration:.1f}s - {result.rowcount:,} registros")

def get_progress_estimate(symbol: str) -> Dict:
    """Estimar progreso si hay datos existentes"""
    with db_manager.get_session() as session:
        # Total registros en orderbook válidos
        total_query = text("""
            SELECT COUNT(*) as total
            FROM orderbook 
            WHERE symbol = :symbol
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            AND valid_for_trading = TRUE
            AND bid1_price > 0 
            AND ask1_price > 0
            AND bid1_price < ask1_price
            AND bid1_size > 0
            AND ask1_size > 0
        """)
        total = session.execute(total_query, {'symbol': symbol}).fetchone().total
        
        # Registros ya procesados en mark_prices
        processed_query = text("""
            SELECT COUNT(*) as processed
            FROM mark_prices 
            WHERE symbol = :symbol
        """)
        processed = session.execute(processed_query, {'symbol': symbol}).fetchone().processed
        
        return {
            'total': total,
            'processed': processed,
            'remaining': total - processed,
            'progress_pct': (processed / max(1, total)) * 100
        }

def main():
    """Función principal ULTRA RÁPIDA - MODO COMPLETAMENTE AUTOMÁTICO"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate mark prices - ULTRA FAST & FULLY AUTOMATED")
    parser.add_argument("--symbol", type=str, help="Specific symbol to process")
    parser.add_argument("--force", action="store_true", help="Force recalculation")
    
    args = parser.parse_args()
    
    log.info("Iniciando cálculo ULTRA RÁPIDO de mark prices...")
    log.info("🤖 MODO COMPLETAMENTE AUTOMÁTICO: Sin intervención manual")
    
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
    
    log.info(f"Procesando {len(symbols)} símbolos con método ULTRA RÁPIDO")
    
    try:
        total_processed = 0
        
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"PROCESANDO {symbol} - ULTRA FAST & AUTOMATED")
            log.info(f"{'='*60}")
            
            # MODO COMPLETAMENTE AUTOMÁTICO: Verificar progreso sin preguntar
            if not args.force:
                progress = get_progress_estimate(symbol)
                if progress['processed'] > 0:
                    log.info(f"Progreso existente: {progress['processed']:,}/{progress['total']:,} ({progress['progress_pct']:.1f}%)")
                    
                    if progress['progress_pct'] >= 99:
                        log.info(f"✅ {symbol} ya está 99%+ completo - saltando procesamiento")
                        continue
                    elif progress['progress_pct'] >= 95:
                        log.info(f"✅ {symbol} está {progress['progress_pct']:.1f}% completo - procesando solo registros faltantes")
                    elif progress['progress_pct'] >= 50:
                        log.info(f"🔄 {symbol} está {progress['progress_pct']:.1f}% completo - continuando automáticamente")
                    else:
                        log.info(f"🔄 {symbol} está {progress['progress_pct']:.1f}% completo - procesando incrementalmente")
                else:
                    log.info(f"🆕 {symbol} - iniciando cálculo desde cero")
            
            if args.force:
                try:
                    with db_manager.get_session() as session:
                        result = session.execute(text("DELETE FROM mark_prices WHERE symbol = :symbol"), 
                                              {'symbol': symbol})
                        log.info(f"🗑️ Eliminados {result.rowcount} mark prices existentes")
                except Exception as e:
                    log.warning(f"Error eliminando datos existentes: {e}")
            
            start_time = datetime.now()
            processed_count = calculate_mark_prices_ultrafast(symbol)
            
            if processed_count > 0:
                update_ohlcv_ultrafast(symbol)
                total_processed += processed_count
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                rate = processed_count / duration if duration > 0 else 0
                
                log.info(f"✅ {processed_count:,} mark prices calculados para {symbol}")
                log.info(f"   Tiempo total: {duration:.1f}s ({rate:.0f} records/sec)")
            else:
                log.info(f"ℹ️ No se agregaron nuevos mark prices para {symbol} (ya existían)")
        
        log.info(f"\n🎉 Cálculo ULTRA RÁPIDO completado!")
        log.info(f"Total procesados: {total_processed:,} mark prices")
        if total_processed > 0:
            log.info(f"Velocidad promedio: ~{total_processed//60:.0f}k records/min")
        
        return True
        
    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)