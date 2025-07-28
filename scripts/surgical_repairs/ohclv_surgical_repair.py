#!/usr/bin/env python3
"""
🎯 SURGICAL OHLCV INGESTION - Solo OHLCV para todos los símbolos
Rellena minutos faltantes de OHLCV preservando datos existentes
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from pathlib import Path
import argparse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.data.ingestion import data_ingestion
from config.settings import settings

from src.utils.logger import get_logger
log = get_logger()

class OHLCVSurgicalIngestion:
    """Ingesta quirúrgica solo OHLCV para todos los símbolos"""
    
    def __init__(self):
        self.ingestion = data_ingestion
        self.engine = db_manager.engine
    
    def analyze_ohlcv_gaps(self, symbol: str, min_gap_threshold: int = 1) -> dict:
        """Analiza gaps específicos de OHLCV"""
        log.info(f"🔍 Analizando gaps OHLCV para {symbol}...")
        
        # Obtener rango temporal del símbolo
        with db_manager.get_session() as session:
            range_query = text("""
                SELECT 
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp,
                    COUNT(*) as record_count
                FROM ohlcv 
                WHERE symbol = :symbol
            """)
            
            range_result = session.execute(range_query, {'symbol': symbol}).fetchone()
            
            if not range_result.first_timestamp:
                log.error(f"No OHLCV data found for {symbol}")
                return {}
            
            start_time = range_result.first_timestamp
            end_time = range_result.last_timestamp
            actual_records = range_result.record_count
        
        # Generar días esperados
        all_days = pd.date_range(start=start_time.date(), end=end_time.date(), freq='D')
        expected_minutes_per_day = 1440  # 24 * 60
        
        log.info(f"📊 Período OHLCV: {start_time.date()} → {end_time.date()}")
        log.info(f"📊 Total días en rango: {len(all_days)}")
        log.info(f"📊 Registros actuales: {actual_records:,}")
        
        # Obtener conteo real por día
        with db_manager.get_session() as session:
            daily_counts_query = text("""
                SELECT 
                    DATE(timestamp) as day,
                    COUNT(*) as minute_count,
                    MIN(timestamp) as first_minute,
                    MAX(timestamp) as last_minute
                FROM ohlcv
                WHERE symbol = :symbol
                GROUP BY DATE(timestamp)
                ORDER BY day
            """)
            
            daily_counts_df = pd.read_sql(daily_counts_query, session.bind, params={'symbol': symbol})
            daily_counts_df['day'] = pd.to_datetime(daily_counts_df['day']).dt.date
        
        # Crear análisis completo
        all_days_df = pd.DataFrame({'day': all_days.date})
        complete_analysis = all_days_df.merge(daily_counts_df, on='day', how='left')
        complete_analysis['minute_count'] = complete_analysis['minute_count'].fillna(0)
        complete_analysis['missing_minutes'] = expected_minutes_per_day - complete_analysis['minute_count']
        complete_analysis['missing_pct'] = (complete_analysis['missing_minutes'] / expected_minutes_per_day) * 100
        
        # Clasificar días
        complete_analysis['gap_category'] = complete_analysis['missing_minutes'].apply(
            lambda x: 'Complete' if x == 0 
            else 'Tiny' if x <= 10
            else 'Small' if x <= 100
            else 'Medium' if x <= 500
            else 'Large' if x <= 1000
            else 'Huge' if x < 1440
            else 'Empty'
        )
        
        # Filtrar días con gaps
        days_with_gaps = complete_analysis[complete_analysis['missing_minutes'] >= min_gap_threshold].copy()
        days_with_gaps = days_with_gaps.sort_values('missing_minutes', ascending=False)
        
        # Estadísticas
        total_expected = len(all_days) * expected_minutes_per_day
        coverage_pct = (actual_records / total_expected) * 100
        
        log.info(f"\n📊 ESTADÍSTICAS OHLCV:")
        log.info(f"  Total días analizados: {len(complete_analysis)}")
        log.info(f"  Días completos (0 faltantes): {len(complete_analysis[complete_analysis['missing_minutes'] == 0])}")
        log.info(f"  Días con gaps (≥{min_gap_threshold} faltantes): {len(days_with_gaps)}")
        log.info(f"  Cobertura actual: {coverage_pct:.2f}%")
        
        # Estadísticas por categoría
        category_stats = complete_analysis.groupby('gap_category').agg({
            'missing_minutes': ['count', 'sum', 'mean']
        }).round(1)
        
        log.info(f"\n📋 CATEGORÍAS DE GAPS OHLCV:")
        for category in ['Complete', 'Tiny', 'Small', 'Medium', 'Large', 'Huge', 'Empty']:
            cat_data = complete_analysis[complete_analysis['gap_category'] == category]
            if len(cat_data) > 0:
                total_missing = cat_data['missing_minutes'].sum()
                avg_missing = cat_data['missing_minutes'].mean()
                log.info(f"  {category:>8}: {len(cat_data):>3} días, {total_missing:>8.0f} min total, {avg_missing:>6.1f} min promedio")
        
        return {
            'symbol': symbol,
            'start_date': start_time.date(),
            'end_date': end_time.date(),
            'total_days': len(all_days),
            'complete_analysis': complete_analysis,
            'days_with_gaps': days_with_gaps,
            'total_missing_minutes': complete_analysis['missing_minutes'].sum(),
            'total_actual_minutes': actual_records,
            'coverage_pct': coverage_pct,
            'expected_total': total_expected
        }
    
    def ingest_ohlcv_day(self, symbol: str, target_date):
        """Ingiere OHLCV para un día específico preservando datos existentes"""
        log.info(f"📊 Ingiriendo OHLCV: {symbol} - {target_date}")
        
        try:
            # Convertir fecha
            if isinstance(target_date, str):
                target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
            else:
                target_datetime = datetime.combine(target_date, datetime.min.time())
            
            start_day = target_datetime.replace(hour=0, minute=0, second=0)
            end_day = target_datetime.replace(hour=23, minute=59, second=59)
            
            # Verificar datos existentes antes
            with db_manager.get_session() as session:
                before_query = text("""
                    SELECT COUNT(*) as existing_count
                    FROM ohlcv 
                    WHERE symbol = :symbol 
                    AND DATE(timestamp) = :target_date
                """)
                
                before_count = session.execute(before_query, {
                    'symbol': symbol,
                    'target_date': target_date
                }).fetchone().existing_count
            
            log.info(f"    📋 Registros antes: {before_count}")
            
            # Ingestar OHLCV (usa ON CONFLICT DO UPDATE en el cliente)
            ingested_count = self.ingestion._fetch_ohlcv_range(symbol, start_day, end_day)
            
            # Verificar datos después
            with db_manager.get_session() as session:
                after_count = session.execute(before_query, {
                    'symbol': symbol,
                    'target_date': target_date
                }).fetchone().existing_count
            
            net_improvement = after_count - before_count
            
            log.info(f"    📊 Registros después: {after_count}")
            log.info(f"    ✅ Mejora neta: +{net_improvement} registros")
            
            return net_improvement
            
        except Exception as e:
            log.error(f"❌ Error ingiriendo OHLCV {target_date}: {e}")
            return 0
    
    def ingest_ohlcv_range(self, symbol: str, start_date, end_date):
        """Ingiere OHLCV para un rango preservando datos existentes"""
        log.info(f"📊 Ingiriendo OHLCV rango: {symbol} - {start_date} a {end_date}")
        
        try:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            
            # Convertir a datetime para la ingesta
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # Contar antes
            with db_manager.get_session() as session:
                before_query = text("""
                    SELECT COUNT(*) as existing_count
                    FROM ohlcv 
                    WHERE symbol = :symbol 
                    AND timestamp >= :start_time 
                    AND timestamp <= :end_time
                """)
                
                before_count = session.execute(before_query, {
                    'symbol': symbol,
                    'start_time': start_datetime,
                    'end_time': end_datetime
                }).fetchone().existing_count
            
            log.info(f"    📋 Registros antes del rango: {before_count}")
            
            # Ingestar OHLCV
            ingested_count = self.ingestion._fetch_ohlcv_range(symbol, start_datetime, end_datetime)
            
            # Contar después
            with db_manager.get_session() as session:
                after_count = session.execute(before_query, {
                    'symbol': symbol,
                    'start_time': start_datetime,
                    'end_time': end_datetime
                }).fetchone().existing_count
            
            net_improvement = after_count - before_count
            days_count = (end_date - start_date).days + 1
            
            log.info(f"    📊 Registros después del rango: {after_count}")
            log.info(f"    ✅ Mejora neta: +{net_improvement} registros ({days_count} días)")
            
            return net_improvement
            
        except Exception as e:
            log.error(f"❌ Error ingiriendo rango OHLCV {start_date}-{end_date}: {e}")
            return 0
    
    def execute_ohlcv_plan(self, symbol: str, dry_run=False, max_days=None, min_gap_threshold=1, strategy='individual'):
        """Ejecuta plan de ingesta OHLCV"""
        log.info(f"🎯 PLAN INGESTA OHLCV PARA {symbol}")
        log.info(f"🔧 Modo: {'DRY RUN' if dry_run else 'EJECUCIÓN REAL'}")
        log.info(f"🔧 Umbral mínimo: {min_gap_threshold} minutos faltantes")
        log.info(f"🔧 Estrategia: {strategy}")
        
        # Analizar gaps OHLCV
        analysis = self.analyze_ohlcv_gaps(symbol, min_gap_threshold)
        
        if not analysis:
            return False
        
        days_with_gaps = analysis['days_with_gaps']
        
        if len(days_with_gaps) == 0:
            log.info("✅ No hay días OHLCV con gaps")
            return True
        
        # Limitar días si se especifica
        if max_days and len(days_with_gaps) > max_days:
            log.info(f"⚠️ Limitando a {max_days} días (de {len(days_with_gaps)} totales)")
            days_with_gaps = days_with_gaps.head(max_days)
        
        log.info(f"\n📊 PLAN DE EJECUCIÓN OHLCV:")
        log.info(f"  Días con gaps a procesar: {len(days_with_gaps)}")
        log.info(f"  Total minutos OHLCV faltantes: {days_with_gaps['missing_minutes'].sum():,}")
        
        # Mostrar muestra
        log.info(f"\n📋 MUESTRA DE DÍAS A PROCESAR (OHLCV):")
        log.info(f"  {'Fecha':<12} {'Faltantes':<10} {'% Faltante':<12} {'Categoría':<10} {'Estrategia'}")
        log.info(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*15}")
        
        for _, row in days_with_gaps.head(15).iterrows():
            strategy_type = (
                "Rango" if row['missing_pct'] > 90 
                else "Individual" if row['missing_pct'] > 10
                else "Incremental"
            )
            log.info(f"  {row['day']} {row['missing_minutes']:>8} {row['missing_pct']:>10.1f}% {row['gap_category']:<10} {strategy_type}")
        
        if len(days_with_gaps) > 15:
            log.info(f"  ... y {len(days_with_gaps) - 15} días más")
        
        if dry_run:
            # Calcular impacto potencial
            potential_improvement = (days_with_gaps['missing_minutes'].sum() / analysis['expected_total']) * 100
            final_coverage = analysis['coverage_pct'] + potential_improvement
            
            log.info(f"\n📊 IMPACTO POTENCIAL OHLCV:")
            log.info(f"  Cobertura actual: {analysis['coverage_pct']:.2f}%")
            log.info(f"  Mejora potencial: +{potential_improvement:.2f} puntos")
            log.info(f"  Cobertura esperada: {min(final_coverage, 100):.2f}%")
            
            return True
        
        # Ejecución real
        log.info(f"🚀 EJECUTANDO INGESTA OHLCV...")
        
        if strategy == 'individual':
            return self._execute_individual_ohlcv(symbol, days_with_gaps)
        elif strategy == 'ranges':
            return self._execute_ranges_ohlcv(symbol, days_with_gaps)
        else:
            log.error(f"Estrategia desconocida: {strategy}")
            return False
    
    def _execute_individual_ohlcv(self, symbol: str, days_with_gaps):
        """Ejecuta ingesta OHLCV día por día"""
        log.info(f"🚀 ESTRATEGIA INDIVIDUAL OHLCV")
        
        total_improvement = 0
        successful_days = 0
        
        for i, (_, row) in enumerate(days_with_gaps.iterrows()):
            day = row['day']
            missing_minutes = row['missing_minutes']
            
            log.info(f"\n📅 OHLCV día {i+1}/{len(days_with_gaps)}: {day} ({missing_minutes} min faltantes)")
            
            try:
                improvement = self.ingest_ohlcv_day(symbol, day)
                total_improvement += improvement
                successful_days += 1
                
                # Pausa entre días
                if i < len(days_with_gaps) - 1 and i % 10 == 9:  # Pausa cada 10 días
                    log.info("⏸️ Pausa de 5 segundos...")
                    time.sleep(5)
                elif i < len(days_with_gaps) - 1:
                    time.sleep(1)
                
            except Exception as e:
                log.error(f"❌ Error OHLCV día {day}: {e}")
                continue
        
        log.info(f"\n🎉 INGESTA INDIVIDUAL OHLCV COMPLETADA")
        log.info(f"  Días procesados: {successful_days}/{len(days_with_gaps)}")
        log.info(f"  Mejora total: +{total_improvement:,} registros OHLCV")
        
        return successful_days == len(days_with_gaps)
    
    def _execute_ranges_ohlcv(self, symbol: str, days_with_gaps):
        """Ejecuta ingesta OHLCV por rangos optimizada"""
        log.info(f"🚀 ESTRATEGIA POR RANGOS OHLCV")
        
        # Agrupar días consecutivos
        ranges = self._group_consecutive_days(days_with_gaps['day'].tolist())
        
        log.info(f"📅 Rangos OHLCV identificados: {len(ranges)}")
        
        total_improvement = 0
        successful_ranges = 0
        
        for i, (start_date, end_date) in enumerate(ranges):
            days_in_range = (end_date - start_date).days + 1
            
            log.info(f"\n📅 OHLCV rango {i+1}/{len(ranges)}: {start_date} → {end_date} ({days_in_range} días)")
            
            try:
                improvement = self.ingest_ohlcv_range(symbol, start_date, end_date)
                total_improvement += improvement
                successful_ranges += 1
                
                # Pausa entre rangos
                if i < len(ranges) - 1:
                    log.info("⏸️ Pausa de 10 segundos...")
                    time.sleep(10)
                
            except Exception as e:
                log.error(f"❌ Error OHLCV rango {start_date}-{end_date}: {e}")
                continue
        
        log.info(f"\n🎉 INGESTA POR RANGOS OHLCV COMPLETADA")
        log.info(f"  Rangos procesados: {successful_ranges}/{len(ranges)}")
        log.info(f"  Mejora total: +{total_improvement:,} registros OHLCV")
        
        return successful_ranges == len(ranges)
    
    def _group_consecutive_days(self, days_list, max_gap=2):
        """Agrupa días consecutivos en rangos"""
        if not days_list:
            return []
        
        days_sorted = sorted(days_list)
        ranges = []
        range_start = days_sorted[0]
        range_end = days_sorted[0]
        
        for i in range(1, len(days_sorted)):
            if (days_sorted[i] - range_end).days <= max_gap:
                range_end = days_sorted[i]
            else:
                ranges.append((range_start, range_end))
                range_start = days_sorted[i]
                range_end = days_sorted[i]
        
        ranges.append((range_start, range_end))
        return ranges

def main():
    """Función principal - Solo OHLCV para todos los símbolos"""
    parser = argparse.ArgumentParser(description="🎯 OHLCV Surgical Ingestion - Solo OHLCV todos los símbolos")
    parser.add_argument("--symbol", type=str, help="Símbolo específico (opcional)")
    parser.add_argument("--dry-run", action="store_true", help="Solo analizar, no ejecutar")
    parser.add_argument("--max-days", type=int, help="Máximo días para testing")
    parser.add_argument("--min-gap", type=int, default=10, help="Mínimo minutos faltantes (default: 10)")
    parser.add_argument("--strategy", choices=['individual', 'ranges'], default='ranges', 
                       help="Estrategia: individual o ranges (default: ranges)")
    
    args = parser.parse_args()
    
    log.info("🎯 INGESTA QUIRÚRGICA OHLCV - TODOS LOS SÍMBOLOS")
    log.info("📊 Solo procesará datos OHLCV")
    
    # Obtener TODOS los símbolos
    if args.symbol:
        symbols = [args.symbol]
        log.info(f"🔧 Procesando símbolo específico: {args.symbol}")
    else:
        try:
            active_pairs = settings.get_active_pairs()
            symbols = []
            for pair in active_pairs:
                symbols.extend([pair.symbol1, pair.symbol2])
            symbols = list(set(symbols))  # Eliminar duplicados
            log.info(f"🔧 Procesando TODOS los símbolos: {len(symbols)}")
        except Exception as e:
            log.error(f"Error cargando símbolos: {e}")
            symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
            log.info(f"🔧 Usando símbolos fallback: {symbols}")
    
    if not symbols:
        log.error("No hay símbolos para procesar")
        return False
    
    # Crear instancia
    ohlcv_surgical = OHLCVSurgicalIngestion()
    
    # Estadísticas globales
    total_symbols = len(symbols)
    successful_symbols = 0
    total_global_improvement = 0
    
    log.info(f"\n🚀 INICIANDO PROCESAMIENTO DE {total_symbols} SÍMBOLOS")
    log.info(f"📊 Configuración:")
    log.info(f"  Umbral mínimo: {args.min_gap} minutos")
    log.info(f"  Estrategia: {args.strategy}")
    log.info(f"  Modo: {'DRY RUN' if args.dry_run else 'EJECUCIÓN REAL'}")
    
    # Procesar cada símbolo
    for i, symbol in enumerate(symbols):
        log.info(f"\n{'='*90}")
        log.info(f"SÍMBOLO {i+1}/{total_symbols}: {symbol}")
        log.info(f"{'='*90}")
        
        try:
            success = ohlcv_surgical.execute_ohlcv_plan(
                symbol=symbol,
                dry_run=args.dry_run,
                max_days=args.max_days,
                min_gap_threshold=args.min_gap,
                strategy=args.strategy
            )
            
            if success:
                successful_symbols += 1
                log.info(f"✅ {symbol} OHLCV procesado exitosamente")
            else:
                log.error(f"❌ {symbol} OHLCV falló")
                
        except Exception as e:
            log.error(f"💥 Error procesando {symbol}: {e}")
            continue
        
        # Pausa entre símbolos
        if i < len(symbols) - 1:
            log.info("⏸️ Pausa de 5 segundos entre símbolos...")
            time.sleep(5)
    
    # Resumen final global
    log.info(f"\n🏁 RESUMEN FINAL GLOBAL:")
    log.info(f"  Total símbolos: {total_symbols}")
    log.info(f"  Símbolos exitosos: {successful_symbols}")
    log.info(f"  Tasa de éxito: {successful_symbols/total_symbols*100:.1f}%")
    
    if not args.dry_run and successful_symbols > 0:
        log.info(f"\n💡 PRÓXIMOS PASOS RECOMENDADOS:")
        log.info(f"  1. Verificar integridad: python scripts/validate_data.py")
        log.info(f"  2. Limpiar datos: python scripts/clean_data.py")
        log.info(f"  3. Calcular mark prices: python scripts/calculate_markprices.py")
        log.info(f"  4. Análisis final: jupyter notebook con análisis de completitud")
    
    return successful_symbols == total_symbols

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)