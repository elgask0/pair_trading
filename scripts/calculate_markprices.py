#!/usr/bin/env python3
"""
ULTRA FAST Mark price calculation script - WITH DIAGNOSTICS (FIXED v2)
Incluye análisis de gaps y períodos vacíos - corregido manejo de parámetros SQL
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

def diagnose_data_gaps(symbol: str) -> Dict:
    """Diagnosticar gaps en los datos de orderbook y mark prices"""
    log.info(f"🔍 Diagnosticando gaps de datos para {symbol}...")
    
    with db_manager.get_session() as session:
        # 1. Obtener rango de fechas de orderbook válido
        orderbook_range_query = text("""
            SELECT 
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                COUNT(DISTINCT DATE(timestamp)) as days_with_data,
                COUNT(*) as total_records,
                COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as valid_records
            FROM orderbook 
            WHERE symbol = :symbol
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
        """)
        
        ob_range = session.execute(orderbook_range_query, {'symbol': symbol}).fetchone()
        
        if not ob_range.min_date or not ob_range.max_date:
            log.warning(f"No orderbook data found for {symbol}")
            return {'error': 'No orderbook data'}
        
        # 2. Analizar gaps por día - VERSIÓN CORREGIDA
        # Primero, obtener todos los días con datos
        daily_counts_query = """
            SELECT 
                DATE(timestamp) as trading_date,
                COUNT(*) as records_count,
                MIN(timestamp) as first_record,
                MAX(timestamp) as last_record,
                COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as valid_count
            FROM orderbook 
            WHERE symbol = %(symbol)s
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY trading_date
        """
        
        daily_counts = pd.read_sql(
            daily_counts_query,
            session.bind,
            params={'symbol': symbol}
        )
        
        # Generar serie completa de fechas
        date_range = pd.date_range(
            start=ob_range.min_date.date(),
            end=ob_range.max_date.date(),
            freq='D'
        )
        
        # Crear DataFrame con todas las fechas
        all_dates_df = pd.DataFrame({
            'date': date_range.date
        })
        
        # Merge con los datos reales
        daily_data = all_dates_df.merge(
            daily_counts.rename(columns={'trading_date': 'date'}),
            on='date',
            how='left'
        )
        
        # Rellenar valores faltantes
        daily_data['records'] = daily_data['records_count'].fillna(0).astype(int)
        daily_data['valid_records'] = daily_data['valid_count'].fillna(0).astype(int)
        
        # Clasificar calidad de datos
        daily_data['data_quality'] = daily_data['records'].apply(
            lambda x: 'NO_DATA' if x == 0 
            else 'SPARSE' if x < 100 
            else 'PARTIAL' if x < 500 
            else 'COMPLETE'
        )
        
        # 3. Identificar períodos problemáticos
        no_data_days = daily_data[daily_data['data_quality'] == 'NO_DATA']
        sparse_days = daily_data[daily_data['data_quality'] == 'SPARSE']
        
        # 4. Analizar gaps consecutivos
        gaps = []
        current_gap = None
        
        for idx, row in daily_data.iterrows():
            if row['data_quality'] in ['NO_DATA', 'SPARSE']:
                if current_gap is None:
                    current_gap = {
                        'start': row['date'],
                        'end': row['date'],
                        'days': 1,
                        'type': row['data_quality']
                    }
                else:
                    current_gap['end'] = row['date']
                    current_gap['days'] += 1
            else:
                if current_gap is not None:
                    gaps.append(current_gap)
                    current_gap = None
        
        if current_gap is not None:
            gaps.append(current_gap)
        
        # 5. Verificar mark prices existentes
        markprice_check_query = text("""
            SELECT 
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                COUNT(*) as total_records,
                COUNT(DISTINCT DATE(timestamp)) as days_with_markprices
            FROM mark_prices 
            WHERE symbol = :symbol
        """)
        
        mp_check = session.execute(markprice_check_query, {'symbol': symbol}).fetchone()
        
        # 6. Análisis específico de junio
        june_analysis_query = """
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as orderbook_records,
                COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as valid_records
            FROM orderbook 
            WHERE symbol = %(symbol)s
            AND timestamp >= '2024-06-01' 
            AND timestamp < '2024-07-01'
            GROUP BY DATE(timestamp)
            ORDER BY date
        """
        
        june_data = pd.read_sql(
            june_analysis_query,
            session.bind,
            params={'symbol': symbol}
        )
        
        # 7. Análisis detallado de días problemáticos
        problem_days_query = """
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_records,
                COUNT(CASE WHEN valid_for_trading = TRUE THEN 1 END) as valid_records,
                MIN(timestamp::time) as first_time,
                MAX(timestamp::time) as last_time,
                COUNT(DISTINCT EXTRACT(HOUR FROM timestamp)) as hours_with_data
            FROM orderbook 
            WHERE symbol = %(symbol)s
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            GROUP BY DATE(timestamp)
            HAVING COUNT(*) < 500
            ORDER BY date
        """
        
        problem_days = pd.read_sql(
            problem_days_query,
            session.bind,
            params={'symbol': symbol}
        )
        
        # 8. Análisis de gaps largos (más de 3 días consecutivos)
        long_gaps = [gap for gap in gaps if gap['days'] >= 3]
        
        return {
            'symbol': symbol,
            'orderbook_range': {
                'min_date': ob_range.min_date,
                'max_date': ob_range.max_date,
                'total_days': (ob_range.max_date - ob_range.min_date).days,
                'days_with_data': ob_range.days_with_data,
                'total_records': ob_range.total_records,
                'valid_records': ob_range.valid_records
            },
            'gaps': gaps,
            'long_gaps': long_gaps,
            'no_data_days': len(no_data_days),
            'sparse_days': len(sparse_days),
            'daily_breakdown': daily_data,
            'june_specific': june_data,
            'problem_days': problem_days,
            'markprices_status': {
                'min_date': mp_check.min_date if mp_check else None,
                'max_date': mp_check.max_date if mp_check else None,
                'total_records': mp_check.total_records if mp_check else 0,
                'days_covered': mp_check.days_with_markprices if mp_check else 0
            }
        }

def generate_gap_report(diagnosis: Dict):
    """Generar reporte detallado de gaps"""
    log.info(f"\n{'='*80}")
    log.info(f"📊 DIAGNÓSTICO DE GAPS - {diagnosis['symbol']}")
    log.info(f"{'='*80}")
    
    # Resumen general
    ob_info = diagnosis['orderbook_range']
    log.info(f"\n📅 RANGO DE DATOS ORDERBOOK:")
    log.info(f"  Período: {ob_info['min_date'].strftime('%Y-%m-%d')} a {ob_info['max_date'].strftime('%Y-%m-%d')}")
    log.info(f"  Días totales: {ob_info['total_days']}")
    log.info(f"  Días con datos: {ob_info['days_with_data']} ({ob_info['days_with_data']/max(1,ob_info['total_days'])*100:.1f}%)")
    log.info(f"  Total registros: {ob_info['total_records']:,}")
    log.info(f"  Registros válidos: {ob_info['valid_records']:,} ({ob_info['valid_records']/max(1,ob_info['total_records'])*100:.1f}%)")
    
    # Gaps encontrados
    if diagnosis['gaps']:
        log.warning(f"\n⚠️ GAPS ENCONTRADOS: {len(diagnosis['gaps'])}")
        for gap in diagnosis['gaps']:
            log.warning(f"  • {gap['start']} a {gap['end']} ({gap['days']} días) - Tipo: {gap['type']}")
        
        # Gaps largos (más problemáticos)
        if diagnosis['long_gaps']:
            log.error(f"\n🚨 GAPS LARGOS (≥3 días):")
            for gap in diagnosis['long_gaps']:
                log.error(f"  • {gap['start']} a {gap['end']} ({gap['days']} días)")
    else:
        log.info(f"\n✅ No se encontraron gaps significativos")
    
    # Días problemáticos
    problem_days = diagnosis.get('problem_days')
    if problem_days is not None and not problem_days.empty:
        log.warning(f"\n⚠️ DÍAS CON DATOS ESCASOS (<500 registros):")
        for _, day in problem_days.iterrows():
            log.warning(f"  • {day['date']}: {int(day['total_records'])} registros "
                       f"({int(day['hours_with_data'])} horas con datos, "
                       f"{day['first_time']} - {day['last_time']})")
    
    # Análisis de junio
    june_data = diagnosis['june_specific']
    if not june_data.empty:
        log.info(f"\n🔍 ANÁLISIS ESPECÍFICO DE JUNIO 2024:")
        
        # Generar rango completo de junio
        june_dates = pd.date_range('2024-06-01', '2024-06-30', freq='D')
        june_dates_df = pd.DataFrame({'date': june_dates.date})
        
        # Merge con datos existentes
        june_complete = june_dates_df.merge(
            june_data,
            on='date',
            how='left'
        )
        june_complete['orderbook_records'] = june_complete['orderbook_records'].fillna(0)
        
        june_missing = june_complete[june_complete['orderbook_records'] == 0]
        if not june_missing.empty:
            log.warning(f"  Días sin datos en junio: {len(june_missing)}")
            for date in june_missing['date']:
                log.warning(f"    - {date}")
        
        june_sparse = june_complete[(june_complete['orderbook_records'] > 0) & 
                                   (june_complete['orderbook_records'] < 100)]
        if not june_sparse.empty:
            log.warning(f"  Días con datos escasos en junio: {len(june_sparse)}")
            for _, row in june_sparse.iterrows():
                log.warning(f"    - {row['date']}: {int(row['orderbook_records'])} registros")
    
    # Estado de mark prices
    mp_info = diagnosis['markprices_status']
    log.info(f"\n💎 ESTADO DE MARK PRICES:")
    if mp_info['total_records'] > 0:
        log.info(f"  Período: {mp_info['min_date'].strftime('%Y-%m-%d %H:%M')} a "
                f"{mp_info['max_date'].strftime('%Y-%m-%d %H:%M')}")
        log.info(f"  Total registros: {mp_info['total_records']:,}")
        log.info(f"  Días cubiertos: {mp_info['days_covered']}")
    else:
        log.warning(f"  No hay mark prices calculados aún")
    
    # Recomendaciones
    log.info(f"\n💡 RECOMENDACIONES:")
    if diagnosis['no_data_days'] > 0:
        log.info(f"  ⚠️ Hay {diagnosis['no_data_days']} días sin datos de orderbook")
        log.info(f"     → Los mark prices no se pueden calcular para estos días")
        log.info(f"     → Matplotlib conectará los puntos creando líneas rectas")
        log.info(f"     → Considera re-ingestar datos para esos períodos con:")
        log.info(f"       python scripts/ingest_data.py --symbol {diagnosis['symbol']}")
    
    if diagnosis['sparse_days'] > 0:
        log.info(f"  ⚠️ Hay {diagnosis['sparse_days']} días con datos escasos")
        log.info(f"     → Los mark prices pueden ser menos confiables estos días")
        log.info(f"     → Verifica la fuente de datos para estos períodos")
    
    if diagnosis['long_gaps']:
        log.error(f"  🚨 Hay {len(diagnosis['long_gaps'])} gaps de 3+ días consecutivos")
        log.error(f"     → Esto afectará significativamente los gráficos y análisis")
        log.error(f"     → PRIORIDAD: Re-ingestar datos para estos períodos")

def calculate_mark_prices_ultrafast(symbol: str, diagnose_only: bool = False) -> int:
    """Calcular mark prices con opción de solo diagnóstico"""
    
    if diagnose_only:
        try:
            # Solo hacer diagnóstico
            diagnosis = diagnose_data_gaps(symbol)
            generate_gap_report(diagnosis)
            
            # Guardar diagnóstico detallado
            diagnosis_dir = Path("diagnostics")
            diagnosis_dir.mkdir(exist_ok=True)
            
            # Guardar breakdown diario
            daily_df = diagnosis['daily_breakdown']
            symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
            daily_df.to_csv(diagnosis_dir / f"{symbol_short}_daily_gaps.csv", index=False)
            log.info(f"📊 Diagnóstico diario guardado en: diagnostics/{symbol_short}_daily_gaps.csv")
            
            # Guardar días problemáticos
            if not diagnosis['problem_days'].empty:
                diagnosis['problem_days'].to_csv(diagnosis_dir / f"{symbol_short}_problem_days.csv", index=False)
                log.info(f"📊 Días problemáticos guardados en: diagnostics/{symbol_short}_problem_days.csv")
            
            return 0
        except Exception as e:
            log.error(f"Error en diagnóstico: {e}")
            import traceback
            log.error(traceback.format_exc())
            return 0
    
    log.info(f"Iniciando cálculo ULTRA RÁPIDO para {symbol}...")
    
    # Primero hacer diagnóstico
    try:
        diagnosis = diagnose_data_gaps(symbol)
        generate_gap_report(diagnosis)
    except Exception as e:
        log.warning(f"No se pudo completar el diagnóstico: {e}")
    
    with db_manager.get_session() as session:
        # Continuar con el cálculo normal...
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
    """Función principal con diagnóstico mejorado"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate mark prices with gap diagnostics")
    parser.add_argument("--symbol", type=str, help="Specific symbol to process")
    parser.add_argument("--force", action="store_true", help="Force recalculation")
    parser.add_argument("--diagnose-only", action="store_true", help="Only run diagnostics, don't calculate")
    
    args = parser.parse_args()
    
    if args.diagnose_only:
        log.info("🔍 Ejecutando solo diagnóstico de gaps...")
    else:
        log.info("Iniciando cálculo de mark prices con diagnóstico...")
    
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
    
    log.info(f"Procesando {len(symbols)} símbolos")
    
    try:
        total_processed = 0
        
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"PROCESANDO {symbol}")
            log.info(f"{'='*60}")
            
            if args.diagnose_only:
                # Solo diagnóstico
                calculate_mark_prices_ultrafast(symbol, diagnose_only=True)
                continue
            
            # Verificar progreso
            if not args.force:
                progress = get_progress_estimate(symbol)
                if progress['processed'] > 0:
                    log.info(f"Progreso existente: {progress['processed']:,}/{progress['total']:,} ({progress['progress_pct']:.1f}%)")
                    
                    if progress['progress_pct'] >= 99:
                        log.info(f"✅ {symbol} ya está 99%+ completo - saltando procesamiento")
                        continue
            
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
                log.info(f"ℹ️ No se agregaron nuevos mark prices para {symbol}")
        
        if not args.diagnose_only:
            log.info(f"\n🎉 Cálculo completado!")
            log.info(f"Total procesados: {total_processed:,} mark prices")
        else:
            log.info(f"\n🔍 Diagnóstico completado!")
            log.info(f"Revisa los archivos en diagnostics/ para más detalles")
            log.info(f"\n💡 Si hay gaps significativos:")
            log.info(f"  1. Los mark prices no se calcularán para esos períodos")
            log.info(f"  2. Los gráficos mostrarán líneas rectas (interpolación)")
            log.info(f"  3. Considera re-ingestar datos para esos períodos")
        
        return True
        
    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)