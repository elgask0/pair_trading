#!/usr/bin/env python3
"""
🔍 DATA INTEGRITY DIAGNOSIS SCRIPT - FIXED VERSION
Analiza cobertura y huecos en datos históricos para todos los tipos de datos
data_end siempre es la fecha actual (hoy)
"""

import sys
import os
import argparse
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sqlalchemy import text
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.utils.logger import setup_logger
from config.settings import settings

# Setup dedicated logger for diagnosis
log = setup_logger("data_diagnosis")

class DataIntegrityDiagnosis:
    """Diagnóstico completo de integridad de datos históricos"""
    
    def __init__(self):
        self.results = defaultdict(dict)
        
    def diagnose_symbol(self, symbol: str) -> Dict:
        """Diagnóstico completo para un símbolo"""
        log.info(f"\n{'='*80}")
        log.info(f"🔍 INICIANDO DIAGNÓSTICO DE INTEGRIDAD: {symbol}")
        log.info(f"{'='*80}")
        
        # Obtener rango de fechas del símbolo
        date_range = self._get_symbol_date_range(symbol)
        if not date_range:
            log.error(f"❌ No se pudo obtener rango de fechas para {symbol}")
            return {}
        
        start_date, end_date = date_range
        log.info(f"📅 Rango a analizar: {start_date.date()} → {end_date.date()}")
        log.info(f"📊 Total días: {(end_date - start_date).days + 1}")
        
        # Diagnóstico por tipo de dato
        results = {
            'symbol': symbol,
            'date_range': {
                'start': start_date,
                'end': end_date,
                'total_days': (end_date - start_date).days + 1
            },
            'ohlcv': self._diagnose_ohlcv(symbol, start_date, end_date),
            'orderbook': self._diagnose_orderbook(symbol, start_date, end_date),
        }
        
        # Funding rates solo para perpetuos
        if "PERP_" in symbol:
            results['funding_rates'] = self._diagnose_funding_rates(symbol, start_date, end_date)
        else:
            log.info(f"ℹ️ {symbol} no es perpetuo - saltando funding rates")
            results['funding_rates'] = {'status': 'N/A - Not perpetual'}
        
        # Resumen general
        self._log_summary(results)
        
        return results
    
    def _get_symbol_date_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """Obtener rango de fechas - data_end siempre es hoy"""
        try:
            # SIEMPRE usar hoy como fecha final
            end_date = datetime.now()
            
            with db_manager.get_session() as session:
                # Intentar obtener data_start de symbol_info
                result = session.execute(text("""
                    SELECT data_start
                    FROM symbol_info
                    WHERE symbol_id = :symbol
                """), {'symbol': symbol}).fetchone()
                
                if result and result.data_start:
                    start_date = result.data_start
                    log.info(f"📊 Usando data_start de symbol_info: {start_date.date()}")
                    return start_date, end_date
                else:
                    # Si no hay symbol_info, usar datos reales
                    log.warning(f"⚠️ No hay data_start en symbol_info para {symbol}, usando datos reales")
                    return self._get_actual_date_range(symbol)
                    
        except Exception as e:
            log.warning(f"Error obteniendo data_start: {e}")
            # Fallback a datos reales
            return self._get_actual_date_range(symbol)
    
    def _get_actual_date_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """Obtener rango real de datos existentes"""
        try:
            # SIEMPRE usar hoy como fecha final
            end_date = datetime.now()
            
            with db_manager.get_session() as session:
                # Buscar fecha más antigua en todas las tablas
                earliest_dates = []
                
                for table in ['ohlcv', 'orderbook', 'funding_rates']:
                    try:
                        result = session.execute(text(f"""
                            SELECT MIN(timestamp) as min_date
                            FROM {table}
                            WHERE symbol = :symbol
                        """), {'symbol': symbol}).fetchone()
                        
                        if result and result.min_date:
                            earliest_dates.append(result.min_date)
                            log.debug(f"  {table}: earliest date = {result.min_date}")
                    except Exception as e:
                        log.debug(f"  Error checking {table}: {e}")
                        continue
                
                if earliest_dates:
                    # Usar la fecha más antigua encontrada
                    start_date = min(earliest_dates)
                    log.info(f"📊 Usando rango real de datos: {start_date.date()} → {end_date.date()}")
                    return start_date, end_date
                else:
                    log.error(f"❌ No se encontraron datos para {symbol} en ninguna tabla")
                    return None
                
        except Exception as e:
            log.error(f"Error obteniendo rango real: {e}")
            return None
    
    def _diagnose_ohlcv(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Diagnóstico de datos OHLCV (1 min candles)"""
        log.info(f"\n📊 DIAGNÓSTICO OHLCV (1-min candles)")
        log.info(f"{'='*50}")
        
        try:
            with db_manager.get_session() as session:
                # 1. Estadísticas generales
                stats_query = text("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record,
                        COUNT(DISTINCT DATE(timestamp)) as days_with_data
                    FROM ohlcv
                    WHERE symbol = :symbol
                    AND timestamp BETWEEN :start_date AND :end_date
                """)
                
                stats = session.execute(stats_query, {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                }).fetchone()
                
                # Si no hay datos, retornar vacío
                if not stats or not stats.total_records:
                    log.warning(f"⚠️ No hay datos OHLCV para {symbol} en el rango especificado")
                    return {
                        'total_records': 0,
                        'expected_records': 0,
                        'coverage_pct': 0,
                        'missing_records': 0,
                        'days_with_data': 0,
                        'missing_days': (end_date - start_date).days + 1,
                        'status': 'NO_DATA'
                    }
                
                # 2. Análisis diario detallado
                daily_query = text("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as minute_count,
                        MIN(timestamp::time) as first_minute,
                        MAX(timestamp::time) as last_minute,
                        COUNT(DISTINCT EXTRACT(HOUR FROM timestamp)) as hours_covered
                    FROM ohlcv
                    WHERE symbol = :symbol
                    AND timestamp BETWEEN :start_date AND :end_date
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """)
                
                daily_df = pd.read_sql(daily_query, session.bind, params={
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                })
                daily_df['date'] = pd.to_datetime(daily_df['date'])
                
                # 3. Calcular métricas
                total_days = (end_date - start_date).days + 1
                expected_minutes_per_day = 1440  # 24 * 60
                total_expected = total_days * expected_minutes_per_day
                total_actual = stats.total_records or 0
                coverage_pct = (total_actual / total_expected * 100) if total_expected > 0 else 0
                
                # 4. Detectar gaps
                all_dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
                # Convert DataFrame dates to plain ``date`` objects to make the set
                # comparison accurate.
                dates_with_data = set(daily_df['date'].dt.date) if not daily_df.empty else set()
                missing_dates = set(all_dates.date) - dates_with_data
                
                # 5. Analizar calidad por día
                perfect_days = 0
                partial_days = 0
                sparse_days = 0
                
                if not daily_df.empty:
                    for _, row in daily_df.iterrows():
                        if row['minute_count'] >= 1440:
                            perfect_days += 1
                        elif row['minute_count'] >= 1000:
                            partial_days += 1
                        else:
                            sparse_days += 1
                
                # 6. Detectar gaps intradiarios grandes
                gap_query = text("""
                    WITH time_diffs AS (
                        SELECT 
                            timestamp,
                            LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                            timestamp - LAG(timestamp) OVER (ORDER BY timestamp) as gap_duration
                        FROM ohlcv
                        WHERE symbol = :symbol
                        AND timestamp BETWEEN :start_date AND :end_date
                    )
                    SELECT 
                        COUNT(*) as gap_count,
                        MAX(gap_duration) as max_gap,
                        AVG(EXTRACT(EPOCH FROM gap_duration)/60) as avg_gap_minutes
                    FROM time_diffs
                    WHERE gap_duration > INTERVAL '1 minute'
                """)
                
                gaps = session.execute(gap_query, {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                }).fetchone()
                
                # Logging detallado
                log.info(f"📈 ESTADÍSTICAS GENERALES OHLCV:")
                log.info(f"  Total registros: {total_actual:,}")
                log.info(f"  Registros esperados: {total_expected:,}")
                log.info(f"  Cobertura: {coverage_pct:.2f}%")
                log.info(f"  Faltan: {total_expected - total_actual:,} minutos")
                
                if stats.first_record and stats.last_record:
                    log.info(f"  Primer registro: {stats.first_record}")
                    log.info(f"  Último registro: {stats.last_record}")
                
                log.info(f"\n📅 ANÁLISIS POR DÍAS:")
                log.info(f"  Días totales: {total_days}")
                log.info(f"  Días con datos: {stats.days_with_data}")
                log.info(f"  Días sin datos: {len(missing_dates)}")
                log.info(f"  Días perfectos (1440 min): {perfect_days}")
                log.info(f"  Días parciales (1000-1439 min): {partial_days}")
                log.info(f"  Días escasos (<1000 min): {sparse_days}")
                
                if missing_dates:
                    log.warning(f"\n⚠️ DÍAS SIN DATOS OHLCV: {len(missing_dates)}")
                    for date in sorted(list(missing_dates))[:10]:
                        log.warning(f"  - {date}")
                    if len(missing_dates) > 10:
                        log.warning(f"  ... y {len(missing_dates) - 10} días más")
                
                if gaps and gaps.gap_count and gaps.gap_count > 0:
                    log.info(f"\n🕳️ ANÁLISIS DE GAPS INTRADIARIOS:")
                    log.info(f"  Total gaps (>1 min): {gaps.gap_count}")
                    log.info(f"  Gap máximo: {gaps.max_gap}")
                    log.info(f"  Gap promedio: {gaps.avg_gap_minutes:.1f} minutos")
                
                # Retornar métricas
                return {
                    'total_records': total_actual,
                    'expected_records': total_expected,
                    'coverage_pct': coverage_pct,
                    'missing_records': total_expected - total_actual,
                    'days_with_data': stats.days_with_data,
                    'missing_days': len(missing_dates),
                    'perfect_days': perfect_days,
                    'partial_days': partial_days,
                    'sparse_days': sparse_days,
                    'gaps_count': gaps.gap_count or 0 if gaps else 0,
                    'max_gap_duration': str(gaps.max_gap) if gaps and gaps.max_gap else None
                }
                
        except Exception as e:
            log.error(f"❌ Error en diagnóstico OHLCV: {e}")
            import traceback
            log.debug(traceback.format_exc())
            return {'error': str(e)}
    
    def _diagnose_orderbook(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Diagnóstico de datos Orderbook (snapshots)"""
        log.info(f"\n📊 DIAGNÓSTICO ORDERBOOK (L2 snapshots)")
        log.info(f"{'='*50}")
        
        try:
            with db_manager.get_session() as session:
                # 1. Estadísticas generales
                stats_query = text("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record,
                        COUNT(DISTINCT DATE(timestamp)) as days_with_data,
                        COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL THEN 1 END) as valid_quotes,
                        AVG(CASE WHEN bid1_price > 0 AND ask1_price > 0 
                            THEN (ask1_price - bid1_price) / bid1_price * 100 END) as avg_spread_pct
                    FROM orderbook
                    WHERE symbol = :symbol
                    AND timestamp BETWEEN :start_date AND :end_date
                """)
                
                stats = session.execute(stats_query, {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                }).fetchone()
                
                # Si no hay datos, retornar vacío
                if not stats or not stats.total_records:
                    log.warning(f"⚠️ No hay datos Orderbook para {symbol} en el rango especificado")
                    return {
                        'total_records': 0,
                        'valid_quotes': 0,
                        'avg_per_day': 0,
                        'days_with_data': 0,
                        'missing_days': (end_date - start_date).days + 1,
                        'status': 'NO_DATA'
                    }
                
                # 2. Análisis diario
                daily_query = text("""
                    WITH ticks AS (
                        SELECT
                            timestamp,
                            DATE(timestamp)   AS day,
                            timestamp::time   AS t_in_day,
                            EXTRACT(EPOCH FROM (
                                timestamp
                                - LAG(timestamp) OVER (
                                    PARTITION BY DATE(timestamp)
                                    ORDER BY timestamp
                                )
                            )) / 60           AS gap_min
                        FROM orderbook
                        WHERE symbol = :symbol
                          AND timestamp BETWEEN :start_date AND :end_date
                    )
                    SELECT
                        day                                          AS date,
                        COUNT(*)                                     AS snapshot_count,
                        MIN(t_in_day)                                AS first_snapshot,
                        MAX(t_in_day)                                AS last_snapshot,
                        COUNT(DISTINCT EXTRACT(HOUR FROM timestamp)) AS hours_covered,
                        AVG(gap_min) FILTER (WHERE gap_min IS NOT NULL) AS avg_interval_minutes
                    FROM ticks
                    GROUP  BY day
                    ORDER  BY day
                """)
                
                daily_df = pd.read_sql(daily_query, session.bind, params={
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                })
                daily_df['date'] = pd.to_datetime(daily_df['date'])
                
                # 3. Calcular métricas
                total_days = (end_date - start_date).days + 1
                expected_per_day = 86400  # Asumiendo 1 snapshot por segundo
                total_actual = stats.total_records or 0
                
                # 4. Detectar días sin datos
                all_dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
                dates_with_data = set(daily_df['date'].dt.date) if not daily_df.empty else set()
                missing_dates = set(all_dates.date) - dates_with_data
                
                # 5. Clasificar días por densidad
                high_density_days = 0
                medium_density_days = 0
                low_density_days = 0
                
                if not daily_df.empty:
                    for _, row in daily_df.iterrows():
                        if row['snapshot_count'] >= 50000:
                            high_density_days += 1
                        elif row['snapshot_count'] >= 10000:
                            medium_density_days += 1
                        else:
                            low_density_days += 1
                
                # 6. Análisis de intervalos
                interval_query = text("""
                    WITH intervals AS (
                        SELECT 
                            timestamp - LAG(timestamp) OVER (ORDER BY timestamp) as interval_duration
                        FROM orderbook
                        WHERE symbol = :symbol
                        AND timestamp BETWEEN :start_date AND :end_date
                    )
                    SELECT 
                        COUNT(*) as interval_count,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM interval_duration)) as median_seconds,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM interval_duration)) as p95_seconds,
                        MAX(EXTRACT(EPOCH FROM interval_duration)) as max_seconds
                    FROM intervals
                    WHERE interval_duration IS NOT NULL
                """)
                
                intervals = session.execute(interval_query, {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                }).fetchone()
                
                # Logging detallado
                log.info(f"📈 ESTADÍSTICAS GENERALES ORDERBOOK:")
                log.info(f"  Total snapshots: {total_actual:,}")
                log.info(f"  Snapshots válidos (bid1/ask1): {stats.valid_quotes:,}")
                log.info(f"  Promedio por día: {total_actual / max(1, stats.days_with_data):.0f}")
                log.info(f"  Spread promedio: {stats.avg_spread_pct:.4f}%" if stats.avg_spread_pct else "  Spread promedio: N/A")
                
                if stats.first_record and stats.last_record:
                    log.info(f"  Primer snapshot: {stats.first_record}")
                    log.info(f"  Último snapshot: {stats.last_record}")
                
                log.info(f"\n📅 ANÁLISIS POR DÍAS:")
                log.info(f"  Días totales: {total_days}")
                log.info(f"  Días con datos: {stats.days_with_data}")
                log.info(f"  Días sin datos: {len(missing_dates)}")
                log.info(f"  Días alta densidad (≥50k): {high_density_days}")
                log.info(f"  Días media densidad (10k-50k): {medium_density_days}")
                log.info(f"  Días baja densidad (<10k): {low_density_days}")
                
                if missing_dates:
                    log.warning(f"\n⚠️ DÍAS SIN DATOS ORDERBOOK: {len(missing_dates)}")
                    for date in sorted(list(missing_dates))[:10]:
                        log.warning(f"  - {date}")
                    if len(missing_dates) > 10:
                        log.warning(f"  ... y {len(missing_dates) - 10} días más")
                
                if intervals and intervals.interval_count and intervals.interval_count > 0:
                    log.info(f"\n⏱️ ANÁLISIS DE INTERVALOS:")
                    log.info(f"  Intervalo mediano: {intervals.median_seconds:.1f} segundos")
                    log.info(f"  Intervalo P95: {intervals.p95_seconds:.1f} segundos")
                    log.info(f"  Intervalo máximo: {intervals.max_seconds:.1f} segundos")
                
                # Días con baja densidad (detalles)
                if low_density_days > 0 and not daily_df.empty:
                    log.warning(f"\n⚠️ DÍAS CON BAJA DENSIDAD (<10k snapshots):")
                    low_density = (
                        daily_df[daily_df['snapshot_count'] < 10000]
                        .sort_values('date', ascending=False)
                        .head(10)
                    )
                    for _, row in low_density.iterrows():
                        log.warning(f"  - {row['date']}: {row['snapshot_count']} snapshots")
                
                return {
                    'total_records': total_actual,
                    'valid_quotes': stats.valid_quotes or 0,
                    'avg_per_day': total_actual / max(1, stats.days_with_data),
                    'days_with_data': stats.days_with_data,
                    'missing_days': len(missing_dates),
                    'high_density_days': high_density_days,
                    'medium_density_days': medium_density_days,
                    'low_density_days': low_density_days,
                    'median_interval_seconds': intervals.median_seconds if intervals and intervals.median_seconds else None,
                    'avg_spread_pct': stats.avg_spread_pct
                }
                
        except Exception as e:
            log.error(f"❌ Error en diagnóstico Orderbook: {e}")
            import traceback
            log.debug(traceback.format_exc())
            return {'error': str(e)}
    
    def _diagnose_funding_rates(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Diagnóstico de Funding Rates (cada 8 horas)"""
        log.info(f"\n📊 DIAGNÓSTICO FUNDING RATES (cada 8h)")
        log.info(f"{'='*50}")
        
        try:
            with db_manager.get_session() as session:
                # 1. Estadísticas generales
                stats_query = text("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record,
                        COUNT(DISTINCT DATE(timestamp)) as days_with_data,
                        AVG(funding_rate) as avg_funding_rate,
                        MIN(funding_rate) as min_funding_rate,
                        MAX(funding_rate) as max_funding_rate
                    FROM funding_rates
                    WHERE symbol = :symbol
                    AND timestamp BETWEEN :start_date AND :end_date
                """)
                
                stats = session.execute(stats_query, {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                }).fetchone()
                
                # Si no hay datos, retornar vacío
                if not stats or not stats.total_records:
                    log.warning(f"⚠️ No hay datos Funding Rates para {symbol} en el rango especificado")
                    return {
                        'total_records': 0,
                        'expected_records': 0,
                        'coverage_pct': 0,
                        'missing_records': 0,
                        'days_with_data': 0,
                        'missing_days': (end_date - start_date).days + 1,
                        'status': 'NO_DATA'
                    }
                
                # 2. Análisis diario
                daily_query = text("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as funding_count,
                        ARRAY_AGG(timestamp::time ORDER BY timestamp) as funding_times
                    FROM funding_rates
                    WHERE symbol = :symbol
                    AND timestamp BETWEEN :start_date AND :end_date
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """)
                
                daily_results = session.execute(daily_query, {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                }).fetchall()
                
                # 3. Calcular métricas
                total_days = (end_date - start_date).days + 1
                expected_per_day = 3  # Cada 8 horas
                total_expected = total_days * expected_per_day
                total_actual = stats.total_records or 0
                coverage_pct = (total_actual / total_expected * 100) if total_expected > 0 else 0
                
                # 4. Detectar días sin datos
                all_dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
                dates_with_data = set(row.date for row in daily_results) if daily_results else set()
                missing_dates = set(all_dates.date) - dates_with_data
                
                # 5. Clasificar días por completitud
                perfect_days = 0
                partial_days = 0
                incomplete_days = 0
                
                for row in daily_results:
                    if row.funding_count == 3:
                        perfect_days += 1
                    elif row.funding_count == 2:
                        partial_days += 1
                    else:
                        incomplete_days += 1
                
                # 6. Detectar horarios anómalos
                expected_hours = [0, 8, 16]  # UTC
                anomalous_times = []
                
                for row in daily_results:
                    if row.funding_times:
                        for funding_time in row.funding_times:
                            hour = funding_time.hour
                            if hour not in expected_hours:
                                anomalous_times.append(f"{row.date} {funding_time}")
                
                # Logging detallado
                log.info(f"📈 ESTADÍSTICAS GENERALES FUNDING:")
                log.info(f"  Total registros: {total_actual:,}")
                log.info(f"  Registros esperados: {total_expected:,}")
                log.info(f"  Cobertura: {coverage_pct:.2f}%")
                log.info(f"  Faltan: {total_expected - total_actual:,} registros")
                
                if stats.first_record and stats.last_record:
                    log.info(f"  Primer registro: {stats.first_record}")
                    log.info(f"  Último registro: {stats.last_record}")
                
                log.info(f"\n💰 TASAS DE FUNDING:")
                log.info(f"  Promedio: {stats.avg_funding_rate:.6f}" if stats.avg_funding_rate else "  Promedio: N/A")
                log.info(f"  Mínimo: {stats.min_funding_rate:.6f}" if stats.min_funding_rate else "  Mínimo: N/A")
                log.info(f"  Máximo: {stats.max_funding_rate:.6f}" if stats.max_funding_rate else "  Máximo: N/A")
                
                log.info(f"\n📅 ANÁLISIS POR DÍAS:")
                log.info(f"  Días totales: {total_days}")
                log.info(f"  Días con datos: {stats.days_with_data}")
                log.info(f"  Días sin datos: {len(missing_dates)}")
                log.info(f"  Días completos (3 funding): {perfect_days}")
                log.info(f"  Días parciales (2 funding): {partial_days}")
                log.info(f"  Días incompletos (1 funding): {incomplete_days}")
                
                if missing_dates:
                    log.warning(f"\n⚠️ DÍAS SIN FUNDING RATES: {len(missing_dates)}")
                    for date in sorted(list(missing_dates))[:10]:
                        log.warning(f"  - {date}")
                    if len(missing_dates) > 10:
                        log.warning(f"  ... y {len(missing_dates) - 10} días más")
                
                if anomalous_times:
                    log.warning(f"\n⚠️ HORARIOS ANÓMALOS (esperado: 00:00, 08:00, 16:00 UTC):")
                    for anomaly in anomalous_times[:10]:
                        log.warning(f"  - {anomaly}")
                    if len(anomalous_times) > 10:
                        log.warning(f"  ... y {len(anomalous_times) - 10} más")
                
                return {
                    'total_records': total_actual,
                    'expected_records': total_expected,
                    'coverage_pct': coverage_pct,
                    'missing_records': total_expected - total_actual,
                    'days_with_data': stats.days_with_data,
                    'missing_days': len(missing_dates),
                    'perfect_days': perfect_days,
                    'partial_days': partial_days,
                    'incomplete_days': incomplete_days,
                    'avg_funding_rate': stats.avg_funding_rate,
                    'anomalous_times_count': len(anomalous_times)
                }
                
        except Exception as e:
            log.error(f"❌ Error en diagnóstico Funding Rates: {e}")
            import traceback
            log.debug(traceback.format_exc())
            return {'error': str(e)}
    
    def _log_summary(self, results: Dict):
        """Log resumen ejecutivo del diagnóstico"""
        symbol = results['symbol']
        
        log.info(f"\n{'='*80}")
        log.info(f"📊 RESUMEN EJECUTIVO: {symbol}")
        log.info(f"{'='*80}")
        
        # OHLCV Summary
        if 'ohlcv' in results and 'error' not in results['ohlcv']:
            ohlcv = results['ohlcv']
            if ohlcv.get('status') == 'NO_DATA':
                log.warning(f"\n📈 OHLCV: NO HAY DATOS")
            else:
                log.info(f"\n📈 OHLCV:")
                log.info(f"  Cobertura: {ohlcv.get('coverage_pct', 0):.2f}%")
                log.info(f"  Calidad: {'EXCELENTE' if ohlcv.get('coverage_pct', 0) > 95 else 'BUENA' if ohlcv.get('coverage_pct', 0) > 80 else 'REGULAR' if ohlcv.get('coverage_pct', 0) > 60 else 'POBRE'}")
                log.info(f"  Días faltantes: {ohlcv.get('missing_days', 0)}")
                log.info(f"  Minutos faltantes: {ohlcv.get('missing_records', 0):,}")
        
        # Orderbook Summary
        if 'orderbook' in results and 'error' not in results['orderbook']:
            orderbook = results['orderbook']
            if orderbook.get('status') == 'NO_DATA':
                log.warning(f"\n📚 ORDERBOOK: NO HAY DATOS")
            else:
                log.info(f"\n📚 ORDERBOOK:")
                log.info(f"  Snapshots totales: {orderbook.get('total_records', 0):,}")
                log.info(f"  Promedio diario: {orderbook.get('avg_per_day', 0):.0f}")
                log.info(f"  Días alta densidad: {orderbook.get('high_density_days', 0)}")
                log.info(f"  Días faltantes: {orderbook.get('missing_days', 0)}")
        
        # Funding Summary
        if 'funding_rates' in results and isinstance(results['funding_rates'], dict) and 'error' not in results['funding_rates']:
            funding = results['funding_rates']
            if funding.get('status') == 'NO_DATA':
                log.warning(f"\n💰 FUNDING RATES: NO HAY DATOS")
            elif 'coverage_pct' in funding:
                log.info(f"\n💰 FUNDING RATES:")
                log.info(f"  Cobertura: {funding.get('coverage_pct', 0):.2f}%")
                log.info(f"  Días completos: {funding.get('perfect_days', 0)}")
                log.info(f"  Días faltantes: {funding.get('missing_days', 0)}")
                log.info(f"  Tasa promedio: {funding.get('avg_funding_rate', 0):.6f}")
        
        # Recomendaciones
        log.info(f"\n💡 RECOMENDACIONES:")
        
        # OHLCV recommendations
        if 'ohlcv' in results and 'error' not in results['ohlcv']:
            if results['ohlcv'].get('status') == 'NO_DATA':
                log.error(f"  ❌ NO HAY DATOS OHLCV - Ejecutar ingesta completa")
            elif results['ohlcv'].get('coverage_pct', 0) < 80:
                log.warning(f"  ⚠️ OHLCV tiene baja cobertura ({results['ohlcv']['coverage_pct']:.1f}%)")
                log.warning(f"     → Ejecutar re-ingesta para días faltantes")
            
            if results['ohlcv'].get('sparse_days', 0) > 5:
                log.warning(f"  ⚠️ {results['ohlcv']['sparse_days']} días con datos escasos en OHLCV")
                log.warning(f"     → Revisar calidad de fuente de datos")
        
        # Orderbook recommendations
        if 'orderbook' in results and 'error' not in results['orderbook']:
            if results['orderbook'].get('status') == 'NO_DATA':
                log.error(f"  ❌ NO HAY DATOS ORDERBOOK - Ejecutar ingesta completa")
            elif results['orderbook'].get('low_density_days', 0) > 10:
                log.warning(f"  ⚠️ {results['orderbook']['low_density_days']} días con baja densidad de orderbook")
                log.warning(f"     → Considerar re-ingesta con mayor frecuencia")
        
        # Funding recommendations
        if 'funding_rates' in results and isinstance(results['funding_rates'], dict):
            if results['funding_rates'].get('status') == 'NO_DATA':
                log.error(f"  ❌ NO HAY DATOS FUNDING - Ejecutar ingesta completa")
            elif results['funding_rates'].get('coverage_pct', 0) < 90:
                log.warning(f"  ⚠️ Funding rates tiene gaps ({results['funding_rates']['coverage_pct']:.1f}% cobertura)")
                log.warning(f"     → Verificar API de MEXC y re-ingestar")
        
        log.info(f"\n✅ Diagnóstico completado para {symbol}")
        log.info(f"{'='*80}\n")

def main():
    """Función principal del diagnóstico"""
    parser = argparse.ArgumentParser(description="🔍 Diagnóstico de integridad de datos históricos")
    parser.add_argument("--symbol", type=str, help="Analizar símbolo específico")
    parser.add_argument("--quick", action="store_true", help="Análisis rápido (últimos 30 días)")
    
    args = parser.parse_args()
    
    log.info("🔍 INICIANDO DIAGNÓSTICO DE INTEGRIDAD DE DATOS")
    log.info(f"Timestamp: {datetime.now()}")
    log.info(f"📌 Nota: data_end siempre es la fecha actual (hoy)")
    
    # Obtener símbolos a analizar
    if args.symbol:
        symbols = [args.symbol]
        log.info(f"Analizando símbolo específico: {args.symbol}")
    else:
        try:
            active_pairs = settings.get_active_pairs()
            symbols = []
            for pair in active_pairs:
                symbols.extend([pair.symbol1, pair.symbol2])
            symbols = list(set(symbols))
            log.info(f"Analizando todos los símbolos activos: {len(symbols)}")
        except Exception as e:
            log.error(f"Error cargando símbolos: {e}")
            symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
            log.info(f"Usando símbolos por defecto: {symbols}")
    
    if not symbols:
        log.error("No hay símbolos para analizar")
        return False
    
    # Crear diagnóstico
    diagnosis = DataIntegrityDiagnosis()
    all_results = {}
    
    # Analizar cada símbolo
    start_time = datetime.now()
    
    for i, symbol in enumerate(symbols):
        symbol_start = datetime.now()
        log.info(f"\n🔄 Procesando símbolo {i+1}/{len(symbols)}: {symbol}")
        
        try:
            results = diagnosis.diagnose_symbol(symbol)
            all_results[symbol] = results
            
            symbol_duration = (datetime.now() - symbol_start).total_seconds()
            log.info(f"⏱️ Tiempo de análisis para {symbol}: {symbol_duration:.1f} segundos")
            
        except Exception as e:
            log.error(f"❌ Error analizando {symbol}: {e}")
            import traceback
            log.debug(traceback.format_exc())
            all_results[symbol] = {'error': str(e)}
    
    # Resumen final global
    total_duration = (datetime.now() - start_time).total_seconds()
    
    log.info(f"\n{'='*80}")
    log.info(f"🏁 DIAGNÓSTICO COMPLETADO")
    log.info(f"{'='*80}")
    log.info(f"Total símbolos analizados: {len(all_results)}")
    log.info(f"Tiempo total: {total_duration:.1f} segundos ({total_duration/60:.1f} minutos)")
    log.info(f"Tiempo promedio por símbolo: {total_duration/len(all_results):.1f} segundos")
    
    # Estadísticas globales
    total_ohlcv_missing = 0
    total_orderbook_missing_days = 0
    total_funding_missing = 0
    symbols_with_issues = 0
    symbols_no_data = 0
    
    for symbol, results in all_results.items():
        has_issues = False
        has_no_data = True
        
        if 'ohlcv' in results and 'error' not in results['ohlcv']:
            if results['ohlcv'].get('status') != 'NO_DATA':
                has_no_data = False
                if results['ohlcv'].get('coverage_pct', 100) < 90:
                    has_issues = True
                    total_ohlcv_missing += results['ohlcv'].get('missing_records', 0)
        
        if 'orderbook' in results and 'error' not in results['orderbook']:
            if results['orderbook'].get('status') != 'NO_DATA':
                has_no_data = False
                missing_days = results['orderbook'].get('missing_days', 0)
                if missing_days > 5:
                    has_issues = True
                    total_orderbook_missing_days += missing_days
        
        if 'funding_rates' in results and isinstance(results['funding_rates'], dict):
            if results['funding_rates'].get('status') != 'NO_DATA':
                has_no_data = False
                if results['funding_rates'].get('coverage_pct', 100) < 90:
                    has_issues = True
                    total_funding_missing += results['funding_rates'].get('missing_records', 0)
        
        if has_issues:
            symbols_with_issues += 1
        if has_no_data:
            symbols_no_data += 1
    
    log.info(f"\n📊 ESTADÍSTICAS GLOBALES:")
    log.info(f"  Símbolos sin datos: {symbols_no_data}/{len(all_results)}")
    log.info(f"  Símbolos con problemas: {symbols_with_issues}/{len(all_results)}")
    log.info(f"  Total minutos OHLCV faltantes: {total_ohlcv_missing:,}")
    log.info(f"  Total días orderbook faltantes: {total_orderbook_missing_days}")
    log.info(f"  Total funding rates faltantes: {total_funding_missing}")
    
    if symbols_no_data > 0:
        log.error(f"\n❌ ACCIÓN CRÍTICA:")
        log.error(f"  {symbols_no_data} símbolos NO TIENEN DATOS")
        log.error(f"  Ejecutar ingesta completa primero:")
        log.error(f"  python scripts/ingest_data.py")
    elif symbols_with_issues > 0:
        log.warning(f"\n⚠️ ACCIÓN REQUERIDA:")
        log.warning(f"  {symbols_with_issues} símbolos necesitan atención")
        log.warning(f"  Revisar logs individuales para detalles específicos")
        log.warning(f"  Considerar ejecutar re-ingesta selectiva")
    else:
        log.info(f"\n✅ TODOS LOS SÍMBOLOS TIENEN BUENA INTEGRIDAD")
    
    log.info(f"\n📁 Logs guardados en: logs/data_diagnosis.log")
    log.info(f"🏁 Diagnóstico finalizado: {datetime.now()}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)