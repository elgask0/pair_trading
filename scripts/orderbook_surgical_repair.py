#!/usr/bin/env python3
"""
🎯 SURGICAL ORDERBOOK RE-INGESTION - Reparación masiva post-abril 12, 2025
Detecta y repara la degradación de datos de orderbook desde abril 12, 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
import argparse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.data.ingestion import data_ingestion
from src.utils.logger import get_ingestion_logger
from config.settings import settings

log = get_ingestion_logger()

class OrderbookSurgicalRepair:
    """Reparación quirúrgica de orderbook post-abril 12, 2025"""
    
    def __init__(self):
        self.ingestion = data_ingestion
        self.engine = db_manager.engine
        
        # Fecha crítica donde empezó el problema
        self.critical_date = datetime(2025, 4, 12)
        
    def analyze_degradation(self, symbol: str) -> dict:
        """Analiza la degradación de datos desde abril 12"""
        log.info(f"🔍 Analizando degradación de orderbook para {symbol}...")
        
        with db_manager.get_session() as session:
            # Obtener estadísticas por día desde marzo para ver el patrón
            daily_stats_query = text("""
                SELECT 
                    DATE(timestamp) as day,
                    COUNT(*) as records_count,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record,
                    COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL THEN 1 END) as valid_quotes
                FROM orderbook 
                WHERE symbol = :symbol 
                AND timestamp >= '2025-03-01'
                GROUP BY DATE(timestamp)
                ORDER BY day
            """)
            
            daily_df = pd.read_sql(daily_stats_query, session.bind, params={'symbol': symbol})
            
            if daily_df.empty:
                log.error(f"No orderbook data found for {symbol}")
                return {}
            
            daily_df['day'] = pd.to_datetime(daily_df['day']).dt.date
            
            # Analizar antes vs después de abril 12
            critical_date = self.critical_date.date()
            
            pre_critical = daily_df[daily_df['day'] < critical_date]
            post_critical = daily_df[daily_df['day'] >= critical_date]
            
            # Calcular estadísticas
            pre_avg = pre_critical['records_count'].mean() if not pre_critical.empty else 0
            post_avg = post_critical['records_count'].mean() if not post_critical.empty else 0
            
            degradation_pct = ((pre_avg - post_avg) / pre_avg * 100) if pre_avg > 0 else 0
            
            # Identificar días problemáticos (menos del 50% del promedio pre-crítico)
            threshold = pre_avg * 0.5
            problematic_days = post_critical[post_critical['records_count'] < threshold]
            
            # Días para re-ingestar (desde abril 12 hasta hoy)
            today = datetime.now().date()
            days_to_reingest = pd.date_range(
                start=critical_date, 
                end=today, 
                freq='D'
            ).date.tolist()
            
            log.info(f"\n📊 ANÁLISIS DE DEGRADACIÓN:")
            log.info(f"  Promedio pre-abril 12: {pre_avg:,.0f} registros/día")
            log.info(f"  Promedio post-abril 12: {post_avg:,.0f} registros/día")
            log.info(f"  Degradación: {degradation_pct:.1f}%")
            log.info(f"  Días problemáticos: {len(problematic_days)}")
            log.info(f"  Días a re-ingestar: {len(days_to_reingest)}")
            
            return {
                'symbol': symbol,
                'pre_critical_avg': pre_avg,
                'post_critical_avg': post_avg,
                'degradation_pct': degradation_pct,
                'problematic_days': problematic_days,
                'days_to_reingest': days_to_reingest,
                'total_days_to_fix': len(days_to_reingest),
                'needs_repair': degradation_pct > 50  # Si perdió más del 50%
            }
    
    def delete_corrupted_data(self, symbol: str, start_date: datetime) -> int:
        """Elimina datos corruptos desde la fecha crítica"""
        log.info(f"🗑️ Eliminando datos corruptos de {symbol} desde {start_date.date()}")
        
        try:
            with db_manager.get_session() as session:
                delete_query = text("""
                    DELETE FROM orderbook 
                    WHERE symbol = :symbol 
                    AND timestamp >= :start_date
                """)
                
                result = session.execute(delete_query, {
                    'symbol': symbol,
                    'start_date': start_date
                })
                
                deleted_count = result.rowcount
                session.commit()
                
                log.info(f"✅ Eliminados {deleted_count:,} registros corruptos de {symbol}")
                return deleted_count
                
        except Exception as e:
            log.error(f"❌ Error eliminando datos de {symbol}: {e}")
            return 0
    
    def reingest_orderbook_range(self, symbol: str, start_date: datetime, end_date: datetime) -> int:
        """Re-ingiere orderbook para un rango específico"""
        log.info(f"📊 Re-ingiriendo orderbook {symbol}: {start_date.date()} → {end_date.date()}")
        
        try:
            total_records = 0
            current_date = start_date.date()
            end_date_only = end_date.date()
            successful_days = 0
            
            # Progress tracking
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False
                log.info("tqdm not available, using simple progress logging")
            
            total_days = (end_date_only - current_date).days + 1
            
            if use_tqdm:
                pbar = tqdm(total=total_days, 
                           desc=f"Orderbook {symbol.split('_')[-2] if '_' in symbol else symbol}", 
                           unit="days")
            
            while current_date <= end_date_only:
                try:
                    # Usar el cliente CoinAPI existente
                    df = self.ingestion.coinapi.get_orderbook_for_date(symbol, current_date.isoformat())
                    
                    if not df.empty:
                        records_count = self.ingestion._insert_orderbook_data(symbol, df)
                        total_records += records_count
                        if records_count > 0:
                            successful_days += 1
                            
                        if use_tqdm:
                            pbar.set_postfix(Records=total_records, Days=successful_days, Current=records_count)
                        else:
                            if current_date.day % 3 == 0:  # Log every 3 days
                                log.info(f"  Progreso: {current_date}, {records_count} records hoy, {total_records:,} total")
                    else:
                        if use_tqdm:
                            pbar.set_postfix(Records=total_records, Days=successful_days, Current=0)
                        else:
                            log.warning(f"  Sin datos para {current_date}")
                    
                    if use_tqdm:
                        pbar.update(1)
                    
                    # Rate limiting agresivo para orderbook
                    time.sleep(0.8)
                    
                except Exception as e:
                    log.warning(f"  Error re-ingiriendo {current_date}: {e}")
                    if use_tqdm:
                        pbar.update(1)
                    time.sleep(2)  # Pausa más larga en caso de error
                
                current_date += timedelta(days=1)
            
            if use_tqdm:
                pbar.close()
            
            log.info(f"✅ Re-ingesta completada: {total_records:,} registros de {successful_days}/{total_days} días")
            return total_records
            
        except Exception as e:
            log.error(f"❌ Error en re-ingesta de {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
            return 0
    
    def execute_surgical_repair(self, symbol: str, dry_run: bool = False, 
                               force_repair: bool = False, start_date: datetime = None) -> bool:
        """Ejecuta reparación quirúrgica completa"""
        log.info(f"🎯 REPARACIÓN QUIRÚRGICA ORDERBOOK: {symbol}")
        log.info(f"🔧 Modo: {'DRY RUN' if dry_run else 'EJECUCIÓN REAL'}")
        
        # Analizar degradación
        analysis = self.analyze_degradation(symbol)
        
        if not analysis:
            log.error(f"No se pudo analizar {symbol}")
            return False
        
        # Verificar si necesita reparación
        if not analysis['needs_repair'] and not force_repair:
            log.info(f"✅ {symbol} no necesita reparación (degradación: {analysis['degradation_pct']:.1f}%)")
            return True
        
        if analysis['degradation_pct'] > 80:
            log.warning(f"🚨 DEGRADACIÓN CRÍTICA en {symbol}: {analysis['degradation_pct']:.1f}%")
        
        # Determinar fechas
        repair_start = start_date if start_date else self.critical_date
        repair_end = datetime.now()
        
        log.info(f"\n📋 PLAN DE REPARACIÓN:")
        log.info(f"  Degradación detectada: {analysis['degradation_pct']:.1f}%")
        log.info(f"  Fecha de inicio: {repair_start.date()}")
        log.info(f"  Fecha final: {repair_end.date()}")
        log.info(f"  Días a procesar: {(repair_end - repair_start).days + 1}")
        log.info(f"  Registros promedio esperados: ~{analysis['pre_critical_avg']:,.0f}/día")
        
        if dry_run:
            log.info(f"\n📊 SIMULACIÓN DE IMPACTO:")
            expected_improvement = analysis['pre_critical_avg'] * len(analysis['days_to_reingest'])
            log.info(f"  Registros esperados post-reparación: ~{expected_improvement:,.0f}")
            log.info(f"  Mejora estimada: +{expected_improvement - analysis['post_critical_avg'] * len(analysis['days_to_reingest']):,.0f} registros")
            return True
        
        # Ejecución real
        log.info(f"\n🚀 EJECUTANDO REPARACIÓN...")
        
        try:
            # 1. Eliminar datos corruptos
            deleted_count = self.delete_corrupted_data(symbol, repair_start)
            
            if deleted_count > 0:
                log.info(f"✅ Limpieza completada: {deleted_count:,} registros eliminados")
            
            # 2. Re-ingestar datos frescos
            log.info(f"📥 Iniciando re-ingesta desde API...")
            new_records = self.reingest_orderbook_range(symbol, repair_start, repair_end)
            
            if new_records > 0:
                log.info(f"✅ Re-ingesta exitosa: {new_records:,} nuevos registros")
                
                # 3. Verificar mejora
                post_repair_analysis = self.analyze_degradation(symbol)
                if post_repair_analysis:
                    improvement = post_repair_analysis['post_critical_avg'] - analysis['post_critical_avg']
                    log.info(f"📈 Mejora lograda: +{improvement:,.0f} registros/día promedio")
                
                return True
            else:
                log.error(f"❌ No se pudieron obtener nuevos datos para {symbol}")
                return False
                
        except Exception as e:
            log.error(f"💥 Error en reparación de {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False
    
    def verify_repair_quality(self, symbol: str) -> dict:
        """Verifica la calidad de la reparación"""
        log.info(f"🔍 Verificando calidad post-reparación para {symbol}...")
        
        with db_manager.get_session() as session:
            # Analizar últimos 7 días
            recent_quality_query = text("""
                SELECT 
                    DATE(timestamp) as day,
                    COUNT(*) as records_count,
                    COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL 
                          AND bid1_price > 0 AND ask1_price > 0 
                          AND bid1_price < ask1_price THEN 1 END) as valid_quotes,
                    AVG(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL 
                        AND bid1_price > 0 AND ask1_price > 0 AND bid1_price < ask1_price
                        THEN (ask1_price - bid1_price) / bid1_price * 100 END) as avg_spread_pct
                FROM orderbook 
                WHERE symbol = :symbol 
                AND timestamp >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(timestamp)
                ORDER BY day DESC
            """)
            
            recent_df = pd.read_sql(recent_quality_query, session.bind, params={'symbol': symbol})
            
            if recent_df.empty:
                return {'error': 'No recent data found'}
            
            avg_records = recent_df['records_count'].mean()
            avg_quality = (recent_df['valid_quotes'].sum() / recent_df['records_count'].sum()) * 100
            avg_spread = recent_df['avg_spread_pct'].mean()
            
            quality_assessment = {
                'avg_records_per_day': avg_records,
                'data_quality_pct': avg_quality,
                'avg_spread_pct': avg_spread,
                'recent_days_analyzed': len(recent_df),
                'assessment': 'excellent' if avg_records > 50000 and avg_quality > 95 
                             else 'good' if avg_records > 20000 and avg_quality > 90
                             else 'fair' if avg_records > 5000 and avg_quality > 80
                             else 'poor'
            }
            
            log.info(f"📊 CALIDAD POST-REPARACIÓN:")
            log.info(f"  Registros/día promedio: {avg_records:,.0f}")
            log.info(f"  Calidad de datos: {avg_quality:.1f}%")
            log.info(f"  Spread promedio: {avg_spread:.4f}%")
            log.info(f"  Evaluación: {quality_assessment['assessment'].upper()}")
            
            return quality_assessment

def main():
    """Función principal de reparación quirúrgica"""
    parser = argparse.ArgumentParser(description="🎯 Orderbook Surgical Repair - Reparación post-abril 12")
    parser.add_argument("--symbol", type=str, help="Símbolo específico (opcional)")
    parser.add_argument("--dry-run", action="store_true", help="Solo simular, no ejecutar")
    parser.add_argument("--force", action="store_true", help="Forzar reparación incluso si no parece necesaria")
    parser.add_argument("--start-date", type=str, help="Fecha inicio (YYYY-MM-DD), default: 2025-04-12")
    parser.add_argument("--verify-only", action="store_true", help="Solo verificar calidad actual")
    
    args = parser.parse_args()
    
    log.info("🎯 REPARACIÓN QUIRÚRGICA DE ORDERBOOK POST-ABRIL 12")
    log.info("🔧 Detecta y repara degradación masiva de datos")
    
    # Parse start date
    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            log.error(f"Fecha inválida: {args.start_date}. Use formato YYYY-MM-DD")
            return False
    
    # Obtener símbolos
    if args.symbol:
        symbols = [args.symbol]
        log.info(f"🎯 Procesando símbolo específico: {args.symbol}")
    else:
        try:
            active_pairs = settings.get_active_pairs()
            symbols = []
            for pair in active_pairs:
                symbols.extend([pair.symbol1, pair.symbol2])
            symbols = list(set(symbols))
            log.info(f"🎯 Procesando TODOS los símbolos: {len(symbols)}")
        except Exception as e:
            log.error(f"Error cargando símbolos: {e}")
            symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
            log.info(f"🎯 Usando símbolos fallback: {symbols}")
    
    if not symbols:
        log.error("No hay símbolos para procesar")
        return False
    
    # Crear instancia de reparador
    repair_tool = OrderbookSurgicalRepair()
    
    # Estadísticas globales
    total_symbols = len(symbols)
    successful_repairs = 0
    
    log.info(f"\n🚀 INICIANDO REPARACIÓN DE {total_symbols} SÍMBOLOS")
    log.info(f"🔧 Configuración:")
    log.info(f"  Fecha crítica: {repair_tool.critical_date.date()}")
    log.info(f"  Fecha inicio: {start_date.date() if start_date else 'default (2025-04-12)'}")
    log.info(f"  Modo: {'DRY RUN' if args.dry_run else 'EJECUCIÓN REAL'}")
    log.info(f"  Forzar: {'Sí' if args.force else 'No'}")
    
    # Procesar cada símbolo
    for i, symbol in enumerate(symbols):
        log.info(f"\n{'='*90}")
        log.info(f"SÍMBOLO {i+1}/{total_symbols}: {symbol}")
        log.info(f"{'='*90}")
        
        try:
            if args.verify_only:
                # Solo verificar calidad
                quality = repair_tool.verify_repair_quality(symbol)
                if 'error' not in quality:
                    log.info(f"✅ Verificación completada para {symbol}")
                continue
            
            # Ejecutar reparación
            success = repair_tool.execute_surgical_repair(
                symbol=symbol,
                dry_run=args.dry_run,
                force_repair=args.force,
                start_date=start_date
            )
            
            if success:
                successful_repairs += 1
                log.info(f"✅ {symbol} reparado exitosamente")
                
                # Verificar calidad post-reparación si no es dry run
                if not args.dry_run:
                    repair_tool.verify_repair_quality(symbol)
            else:
                log.error(f"❌ {symbol} falló en reparación")
                
        except Exception as e:
            log.error(f"💥 Error procesando {symbol}: {e}")
            continue
        
        # Pausa entre símbolos para no sobrecargar la API
        if i < len(symbols) - 1 and not args.dry_run:
            log.info("⏸️ Pausa de 10 segundos entre símbolos...")
            time.sleep(10)
    
    # Resumen final
    log.info(f"\n{'='*90}")
    log.info(f"RESUMEN FINAL DE REPARACIÓN")
    log.info(f"{'='*90}")
    
    if args.verify_only:
        log.info(f"✅ Verificación completada para {total_symbols} símbolos")
    else:
        log.info(f"✅ Símbolos reparados exitosamente: {successful_repairs}/{total_symbols}")
        log.info(f"📊 Tasa de éxito: {successful_repairs/total_symbols*100:.1f}%")
        
        if not args.dry_run and successful_repairs > 0:
            log.info(f"\n💡 PRÓXIMOS PASOS RECOMENDADOS:")
            log.info(f"  1. Verificar con deep analysis: python scripts/orderbook_deep_analysis.py")
            log.info(f"  2. Limpiar datos: python scripts/clean_data.py")
            log.info(f"  3. Recalcular mark prices: python scripts/calculate_markprices.py --force")
            log.info(f"  4. Ejecutar pair analysis actualizado")
    
    log.info(f"\n🎉 Reparación quirúrgica completada!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)