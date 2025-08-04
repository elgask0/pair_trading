#!/usr/bin/env python3
"""
🔍 ENHANCED DATA INTEGRITY DIAGNOSIS + SURGICAL REPAIR - COMPLETE VERSION
Analiza datos + verifica contra API + repara automáticamente datos faltantes
FIXED: Ahora repara también días con datos insuficientes detectados por API
FIXED: Lógica adaptativa para funding rates + exclusión del día actual para orderbook
UPDATED: Usa base de datos en lugar de YAML para obtener símbolos
"""

import sys
import os
import argparse
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from sqlalchemy import text
from pathlib import Path
from collections import defaultdict
import random
import time as time_module

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.data.coinapi_client import coinapi_client
from src.data.mexc_client import mexc_client
from src.data.ingestion import data_ingestion
from config.settings import settings

from src.utils.logger import get_logger
log = get_logger("diagnose_data")

class EnhancedDataDiagnosis:
    """Diagnóstico completo con verificación de API y surgical repair - COMPLETE"""
    
    def __init__(self, enable_surgical_repair: bool = False):
        self.results = defaultdict(dict)
        self.coinapi = coinapi_client
        self.mexc = mexc_client
        self.ingestion = data_ingestion
        self.enable_surgical_repair = enable_surgical_repair
        
        # Control de rate limiting para API calls
        self.api_call_delay = 0.5  # 500ms entre calls
        self.max_sample_days = 10  # Máximo días a verificar por símbolo
        
        # Contadores para surgical repair
        self.repair_stats = {
            'symbols_repaired': 0,
            'total_ohlcv_repaired': 0,
            'total_orderbook_repaired': 0,
            'repair_failures': 0
        }
    
    def diagnose_symbol_enhanced(self, symbol: str, verify_api: bool = True) -> Dict:
        """Diagnóstico completo para un símbolo con verificación de API y optional repair"""
        log.info(f"\n{'='*80}")
        log.info(f"🔍 ENHANCED DIAGNOSIS: {symbol}")
        log.info(f"{'='*80}")
        
        # Obtener rango de fechas del símbolo
        date_range = self._get_symbol_date_range(symbol)
        if not date_range:
            log.error(f"❌ No se pudo obtener rango de fechas para {symbol}")
            return {}
        
        start_date, end_date = date_range
        log.info(f"📅 Rango a analizar: {start_date.date()} → {end_date.date()}")
        log.info(f"📊 Total días: {(end_date - start_date).days + 1}")
        
        # Diagnóstico básico por tipo de dato
        results = {
            'symbol': symbol,
            'date_range': {
                'start': start_date,
                'end': end_date,
                'total_days': (end_date - start_date).days + 1
            },
            'ohlcv': self._diagnose_ohlcv_enhanced(symbol, start_date, end_date, verify_api),
            'orderbook': self._diagnose_orderbook_enhanced(symbol, start_date, end_date, verify_api),
        }
        
        # Funding rates solo para perpetuos
        if "PERP_" in symbol:
            results['funding_rates'] = self._diagnose_funding_enhanced(symbol, start_date, end_date, verify_api)
        else:
            log.info(f"ℹ️ {symbol} no es perpetuo - saltando funding rates")
            results['funding_rates'] = {'status': 'N/A - Not perpetual'}
        
        # SURGICAL REPAIR si está habilitado
        if self.enable_surgical_repair and verify_api:
            repair_results = self._execute_surgical_repair(symbol, results)
            results['surgical_repair'] = repair_results
        
        # Resumen general mejorado
        self._log_enhanced_summary(results)
        
        return results
    
    def _diagnose_ohlcv_enhanced(self, symbol: str, start_date: datetime, end_date: datetime, verify_api: bool) -> Dict:
        """Diagnóstico OHLCV con verificación de API"""
        log.info(f"\n📊 ENHANCED OHLCV DIAGNOSIS")
        log.info(f"{'='*50}")
        
        try:
            # Diagnóstico básico primero
            basic_diagnosis = self._diagnose_ohlcv_basic(symbol, start_date, end_date)
            
            if not verify_api or basic_diagnosis.get('status') == 'NO_DATA':
                return basic_diagnosis
            
            # Verificación de API para días problemáticos
            log.info(f"🔍 Verificando API para días OHLCV problemáticos...")
            
            # Obtener días faltantes y días con pocos datos
            missing_days, sparse_days = self._get_problematic_ohlcv_days(symbol, start_date, end_date)
            
            # Samplear días para verificar (no todos para evitar rate limiting)
            sample_missing = self._sample_days(missing_days, max_samples=5)
            sample_sparse = self._sample_days(sparse_days, max_samples=3)
            
            api_verification = {
                'missing_days_checked': [],
                'sparse_days_checked': [],
                'api_has_data_for_missing': 0,
                'api_confirms_sparse': 0,
                'api_call_failures': 0,
                'significant_discrepancies': 0  # NUEVO: Contador de discrepancias significativas
            }
            
            # Verificar días faltantes
            if sample_missing:
                log.info(f"  Verificando {len(sample_missing)} días faltantes en API...")
                for day in sample_missing:
                    api_result = self._check_ohlcv_api_for_date(symbol, day)
                    api_verification['missing_days_checked'].append({
                        'date': day,
                        'api_has_data': api_result['has_data'],
                        'api_records': api_result['record_count'],
                        'api_error': api_result.get('error')
                    })
                    
                    if api_result['has_data']:
                        api_verification['api_has_data_for_missing'] += 1
                    if api_result.get('error'):
                        api_verification['api_call_failures'] += 1
                    
                    time_module.sleep(self.api_call_delay)
            
            # MEJORADO: Verificar días escasos con detección de discrepancias significativas
            if sample_sparse:
                log.info(f"  Verificando {len(sample_sparse)} días escasos en API...")
                for day_info in sample_sparse:
                    if isinstance(day_info, dict):
                        day = day_info['date']
                        current_count = day_info['count']
                    else:
                        day = day_info
                        current_count = 0
                    
                    api_result = self._check_ohlcv_api_for_date(symbol, day)
                    
                    # NUEVO: Detectar discrepancias significativas
                    has_significant_discrepancy = False
                    if api_result['record_count'] and current_count > 0:
                        api_count = api_result['record_count']
                        discrepancy_ratio = api_count / current_count if current_count > 0 else float('inf')
                        
                        # Si API tiene >2x los datos de BD, es discrepancia significativa
                        if discrepancy_ratio >= 2.0:
                            has_significant_discrepancy = True
                            api_verification['significant_discrepancies'] += 1
                            log.warning(f"    🚨 DISCREPANCIA SIGNIFICATIVA {day}: BD={current_count}, API={api_count} (ratio: {discrepancy_ratio:.1f}x)")
                    
                    api_verification['sparse_days_checked'].append({
                        'date': day,
                        'current_records': current_count,
                        'api_records': api_result['record_count'],
                        'api_has_data': api_result['has_data'],
                        'api_error': api_result.get('error'),
                        'has_significant_discrepancy': has_significant_discrepancy  # NUEVO
                    })
                    
                    if api_result['record_count'] and api_result['record_count'] <= current_count * 1.1:
                        api_verification['api_confirms_sparse'] += 1
                    if api_result.get('error'):
                        api_verification['api_call_failures'] += 1
                    
                    time_module.sleep(self.api_call_delay)
            
            # Combinar diagnóstico básico con verificación de API
            enhanced_diagnosis = basic_diagnosis.copy()
            enhanced_diagnosis['api_verification'] = api_verification
            enhanced_diagnosis['data_quality_assessment'] = self._assess_ohlcv_quality(basic_diagnosis, api_verification)
            
            self._log_ohlcv_api_results(symbol, api_verification)
            
            return enhanced_diagnosis
            
        except Exception as e:
            log.error(f"❌ Error en diagnóstico OHLCV enhanced: {e}")
            import traceback
            log.debug(traceback.format_exc())
            return {'error': str(e)}
    
    def _diagnose_orderbook_enhanced(self, symbol: str, start_date: datetime, end_date: datetime, verify_api: bool) -> Dict:
        """Diagnóstico orderbook con verificación de API - MEJORADO CON EXCLUSIÓN DEL DÍA ACTUAL"""
        log.info(f"\n📊 ENHANCED ORDERBOOK DIAGNOSIS")
        log.info(f"{'='*50}")
        
        try:
            # NUEVO: Excluir el día de hoy para orderbook (la API no puede obtener datos del día actual)
            today = datetime.now().date()
            if end_date.date() >= today:
                original_end_date = end_date
                end_date = datetime.combine(today - timedelta(days=1), datetime.max.time())
                log.info(f"📅 Excluyendo día actual para orderbook: {original_end_date.date()} → {end_date.date()}")
            
            # Diagnóstico básico primero
            basic_diagnosis = self._diagnose_orderbook_basic(symbol, start_date, end_date)
            
            if not verify_api or basic_diagnosis.get('status') == 'NO_DATA':
                return basic_diagnosis
            
            # Verificación de API para días problemáticos
            log.info(f"🔍 Verificando API para días orderbook problemáticos...")
            
            # Obtener días faltantes y días con pocos snapshots
            missing_days, sparse_days = self._get_problematic_orderbook_days(symbol, start_date, end_date)
            
            # Samplear días para verificar
            sample_missing = self._sample_days(missing_days, max_samples=3)  # Menos para orderbook (más costoso)
            sample_sparse = self._sample_days(sparse_days, max_samples=3)
            
            api_verification = {
                'missing_days_checked': [],
                'sparse_days_checked': [],
                'api_has_data_for_missing': 0,
                'api_confirms_sparse': 0,
                'api_call_failures': 0,
                'significant_discrepancies': 0  # NUEVO
            }
            
            # Verificar días faltantes
            if sample_missing:
                log.info(f"  Verificando {len(sample_missing)} días faltantes en API orderbook...")
                for day in sample_missing:
                    # NUEVO: Saltar si es hoy o futuro
                    if isinstance(day, date) and day >= today:
                        log.info(f"  ⏭️ Saltando {day} (día actual o futuro)")
                        continue
                    
                    api_result = self._check_orderbook_api_for_date(symbol, day)
                    api_verification['missing_days_checked'].append({
                        'date': day,
                        'api_has_data': api_result['has_data'],
                        'api_snapshots': api_result['snapshot_count'],
                        'api_error': api_result.get('error')
                    })
                    
                    if api_result['has_data']:
                        api_verification['api_has_data_for_missing'] += 1
                    if api_result.get('error'):
                        api_verification['api_call_failures'] += 1
                    
                    time_module.sleep(self.api_call_delay * 2)  # Más delay para orderbook
            
            # MEJORADO: Verificar días escasos con detección de discrepancias
            if sample_sparse:
                log.info(f"  Verificando {len(sample_sparse)} días escasos en API orderbook...")
                for day_info in sample_sparse:
                    if isinstance(day_info, dict):
                        day = day_info['date']
                        current_count = day_info['count']
                    else:
                        day = day_info
                        current_count = 0
                    
                    # NUEVO: Saltar si es hoy o futuro
                    if isinstance(day, date) and day >= today:
                        log.info(f"  ⏭️ Saltando {day} (día actual o futuro)")
                        continue
                    
                    api_result = self._check_orderbook_api_for_date(symbol, day)
                    
                    # NUEVO: Detectar discrepancias significativas para orderbook
                    has_significant_discrepancy = False
                    if api_result['snapshot_count'] and current_count > 0:
                        api_count = api_result['snapshot_count']
                        discrepancy_ratio = api_count / current_count if current_count > 0 else float('inf')
                        
                        # Para orderbook, si API tiene >3x los datos, es significativo (threshold más alto)
                        if discrepancy_ratio >= 3.0:
                            has_significant_discrepancy = True
                            api_verification['significant_discrepancies'] += 1
                            log.warning(f"    🚨 DISCREPANCIA ORDERBOOK {day}: BD={current_count}, API={api_count} (ratio: {discrepancy_ratio:.1f}x)")
                    
                    api_verification['sparse_days_checked'].append({
                        'date': day,
                        'current_snapshots': current_count,
                        'api_snapshots': api_result['snapshot_count'],
                        'api_has_data': api_result['has_data'],
                        'api_error': api_result.get('error'),
                        'has_significant_discrepancy': has_significant_discrepancy  # NUEVO
                    })
                    
                    if (api_result['snapshot_count'] and 
                        api_result['snapshot_count'] <= current_count * 1.1):
                        api_verification['api_confirms_sparse'] += 1
                    if api_result.get('error'):
                        api_verification['api_call_failures'] += 1
                    
                    time_module.sleep(self.api_call_delay * 2)
            
            # Combinar diagnóstico básico con verificación de API
            enhanced_diagnosis = basic_diagnosis.copy()
            enhanced_diagnosis['api_verification'] = api_verification
            enhanced_diagnosis['data_quality_assessment'] = self._assess_orderbook_quality(basic_diagnosis, api_verification)
            
            self._log_orderbook_api_results(symbol, api_verification)
            
            return enhanced_diagnosis
            
        except Exception as e:
            log.error(f"❌ Error en diagnóstico orderbook enhanced: {e}")
            import traceback
            log.error(traceback.format_exc())
            return {'error': str(e)}
    
    def _execute_surgical_repair(self, symbol: str, diagnosis_results: Dict) -> Dict:
        """Ejecuta surgical repair basado en los resultados del diagnóstico - MEJORADO"""
        log.info(f"\n🔧 INICIANDO SURGICAL REPAIR PARA {symbol}")
        log.info(f"{'='*60}")
        
        repair_results = {
            'ohlcv_repair': {'attempted': False, 'success': False, 'records_added': 0},
            'orderbook_repair': {'attempted': False, 'success': False, 'records_added': 0},
            'pre_repair_stats': {},
            'post_repair_stats': {}
        }
        
        # Capturar estadísticas pre-repair
        repair_results['pre_repair_stats'] = self._get_data_stats(symbol)
        
        # MEJORADO: OHLCV Repair - incluye días con discrepancias significativas
        ohlcv_results = diagnosis_results.get('ohlcv', {})
        ohlcv_api = ohlcv_results.get('api_verification', {})
        
        # Días completamente faltantes
        missing_ohlcv_days = [d for d in ohlcv_api.get('missing_days_checked', []) if d['api_has_data']]
        
        # NUEVO: Días con discrepancias significativas
        discrepancy_ohlcv_days = [d for d in ohlcv_api.get('sparse_days_checked', []) 
                                 if d.get('has_significant_discrepancy', False)]
        
        # Combinar ambos tipos para repair
        all_ohlcv_repair_days = missing_ohlcv_days + discrepancy_ohlcv_days
        
        if all_ohlcv_repair_days:
            log.info(f"🔧 OHLCV REPAIR: {len(missing_ohlcv_days)} días faltantes + {len(discrepancy_ohlcv_days)} con discrepancias")
            repair_results['ohlcv_repair']['attempted'] = True
            
            for day_info in all_ohlcv_repair_days:
                try:
                    day = day_info['date']
                    
                    # Determinar tipo de repair
                    if day_info in missing_ohlcv_days:
                        log.info(f"  📅 Reparando día faltante: {day}")
                    else:
                        current_records = day_info.get('current_records', 0)
                        api_records = day_info.get('api_records', 0)
                        log.info(f"  📅 Reparando discrepancia: {day} (BD: {current_records}, API: {api_records})")
                    
                    records_added = self._repair_ohlcv_day(symbol, day)
                    repair_results['ohlcv_repair']['records_added'] += records_added
                    
                    if records_added > 0:
                        log.info(f"  ✅ {day}: +{records_added} registros OHLCV")
                    else:
                        log.warning(f"  ⚠️ {day}: No se pudieron obtener datos OHLCV")
                        
                except Exception as e:
                    log.error(f"  ❌ Error reparando OHLCV {day}: {e}")
                    
                time_module.sleep(0.3)  # Rate limiting
            
            if repair_results['ohlcv_repair']['records_added'] > 0:
                repair_results['ohlcv_repair']['success'] = True
                self.repair_stats['total_ohlcv_repaired'] += repair_results['ohlcv_repair']['records_added']
        
        # MEJORADO: Orderbook Repair - incluye días con discrepancias significativas
        orderbook_results = diagnosis_results.get('orderbook', {})
        orderbook_api = orderbook_results.get('api_verification', {})
        
        # Días completamente faltantes
        missing_orderbook_days = [d for d in orderbook_api.get('missing_days_checked', []) if d['api_has_data']]
        
        # NUEVO: Días con discrepancias significativas
        discrepancy_orderbook_days = [d for d in orderbook_api.get('sparse_days_checked', []) 
                                     if d.get('has_significant_discrepancy', False)]
        
        # Combinar ambos tipos para repair
        all_orderbook_repair_days = missing_orderbook_days + discrepancy_orderbook_days
        
        if all_orderbook_repair_days:
            log.info(f"🔧 ORDERBOOK REPAIR: {len(missing_orderbook_days)} días faltantes + {len(discrepancy_orderbook_days)} con discrepancias")
            repair_results['orderbook_repair']['attempted'] = True
            
            for day_info in all_orderbook_repair_days:
                try:
                    day = day_info['date']
                    
                    # NUEVO: Saltar si es día actual o futuro (no se puede reparar)
                    today = datetime.now().date()
                    if isinstance(day, date) and day >= today:
                        log.info(f"  ⏭️ Saltando {day} (día actual o futuro - no disponible en API)")
                        continue
                    
                    # Determinar tipo de repair
                    if day_info in missing_orderbook_days:
                        log.info(f"  📅 Reparando día orderbook faltante: {day}")
                    else:
                        current_snapshots = day_info.get('current_snapshots', 0)
                        api_snapshots = day_info.get('api_snapshots', 0)
                        log.info(f"  📅 Reparando discrepancia orderbook: {day} (BD: {current_snapshots}, API: {api_snapshots})")
                    
                    records_added = self._repair_orderbook_day(symbol, day)
                    repair_results['orderbook_repair']['records_added'] += records_added
                    
                    if records_added > 0:
                        log.info(f"  ✅ {day}: +{records_added} snapshots orderbook")
                    else:
                        log.warning(f"  ⚠️ {day}: No se pudieron obtener datos orderbook")
                        
                except Exception as e:
                    log.error(f"  ❌ Error reparando orderbook {day}: {e}")
                    
                time_module.sleep(0.8)  # Rate limiting más agresivo para orderbook
            
            if repair_results['orderbook_repair']['records_added'] > 0:
                repair_results['orderbook_repair']['success'] = True
                self.repair_stats['total_orderbook_repaired'] += repair_results['orderbook_repair']['records_added']
        
        # Capturar estadísticas post-repair
        repair_results['post_repair_stats'] = self._get_data_stats(symbol)
        
        # Log resultados del repair
        self._log_repair_results(symbol, repair_results)
        
        # Actualizar contador global
        if repair_results['ohlcv_repair']['success'] or repair_results['orderbook_repair']['success']:
            self.repair_stats['symbols_repaired'] += 1
        
        return repair_results
    
    def _assess_ohlcv_quality(self, basic: Dict, api_verification: Dict) -> Dict:
        """Evaluar calidad de datos OHLCV - MEJORADO para incluir discrepancias"""
        assessment = {
            'overall_grade': 'UNKNOWN',
            'data_completeness': 'UNKNOWN',
            'api_consistency': 'UNKNOWN',
            'ingestion_health': 'UNKNOWN',
            'recommendations': []
        }
        
        try:
            coverage_pct = basic.get('coverage_pct', 0)
            missing_days_with_api_data = api_verification.get('api_has_data_for_missing', 0)
            significant_discrepancies = api_verification.get('significant_discrepancies', 0)  # NUEVO
            total_missing_checked = len(api_verification.get('missing_days_checked', []))
            
            # Evaluar completitud de datos
            if coverage_pct >= 95:
                assessment['data_completeness'] = 'EXCELLENT'
            elif coverage_pct >= 85:
                assessment['data_completeness'] = 'GOOD'
            elif coverage_pct >= 70:
                assessment['data_completeness'] = 'FAIR'
            else:
                assessment['data_completeness'] = 'POOR'
            
            # MEJORADO: Evaluar consistencia con API (incluye discrepancias)
            total_issues = missing_days_with_api_data + significant_discrepancies
            if total_missing_checked > 0:
                api_consistency_pct = (total_missing_checked - missing_days_with_api_data) / total_missing_checked * 100
                if api_consistency_pct >= 80 and significant_discrepancies == 0:
                    assessment['api_consistency'] = 'GOOD'
                elif significant_discrepancies > 0:
                    assessment['api_consistency'] = 'POOR'  # Hay discrepancias significativas
                else:
                    assessment['api_consistency'] = 'FAIR'
            
            # MEJORADO: Evaluar salud de ingesta (incluye discrepancias)
            if total_issues == 0:
                assessment['ingestion_health'] = 'GOOD'
            elif total_issues <= 2:
                assessment['ingestion_health'] = 'FAIR'
            else:
                assessment['ingestion_health'] = 'POOR'
            
            # Grado general
            grades = [assessment['data_completeness'], assessment['api_consistency'], assessment['ingestion_health']]
            if 'POOR' in grades:
                assessment['overall_grade'] = 'POOR'
            elif 'FAIR' in grades:
                assessment['overall_grade'] = 'FAIR'
            elif 'GOOD' in grades:
                assessment['overall_grade'] = 'GOOD'
            else:
                assessment['overall_grade'] = 'EXCELLENT'
            
            # MEJORADO: Recomendaciones que incluyen discrepancias
            if missing_days_with_api_data > 0:
                assessment['recommendations'].append(f"Re-ingestar {missing_days_with_api_data} días con datos disponibles en API")
            
            if significant_discrepancies > 0:
                assessment['recommendations'].append(f"Reparar {significant_discrepancies} días con discrepancias significativas (API tiene mucho más datos)")
            
            if coverage_pct < 90:
                assessment['recommendations'].append("Mejorar cobertura de datos OHLCV")
                
            if api_verification.get('api_call_failures', 0) > 0:
                assessment['recommendations'].append("Revisar conectividad con CoinAPI")
            
        except Exception as e:
            log.warning(f"Error evaluando calidad OHLCV: {e}")
        
        return assessment
    
    def _assess_orderbook_quality(self, basic: Dict, api_verification: Dict) -> Dict:
        """Evaluar calidad de datos orderbook - MEJORADO"""
        assessment = {
            'overall_grade': 'UNKNOWN',
            'data_density': 'UNKNOWN',
            'api_consistency': 'UNKNOWN',
            'recommendations': []
        }
        
        try:
            avg_per_day = basic.get('avg_per_day', 0)
            missing_days_with_api_data = api_verification.get('api_has_data_for_missing', 0)
            significant_discrepancies = api_verification.get('significant_discrepancies', 0)  # NUEVO
            
            # Evaluar densidad de datos
            if avg_per_day >= 50000:
                assessment['data_density'] = 'EXCELLENT'
            elif avg_per_day >= 20000:
                assessment['data_density'] = 'GOOD'
            elif avg_per_day >= 5000:
                assessment['data_density'] = 'FAIR'
            else:
                assessment['data_density'] = 'POOR'
            
            # MEJORADO: Consistencia con API (incluye discrepancias)
            total_issues = missing_days_with_api_data + significant_discrepancies
            total_missing_checked = len(api_verification.get('missing_days_checked', []))
            
            if total_missing_checked > 0:
                api_consistency_pct = (total_missing_checked - missing_days_with_api_data) / total_missing_checked * 100
                if api_consistency_pct >= 80 and significant_discrepancies == 0:
                    assessment['api_consistency'] = 'GOOD'
                elif significant_discrepancies > 0:
                    assessment['api_consistency'] = 'POOR'
                else:
                    assessment['api_consistency'] = 'FAIR'
            
            # Grado general
            if assessment['data_density'] == 'POOR' or assessment['api_consistency'] == 'POOR':
                assessment['overall_grade'] = 'POOR'
            elif assessment['data_density'] == 'FAIR':
                assessment['overall_grade'] = 'FAIR'
            else:
                assessment['overall_grade'] = 'GOOD'
            
            # MEJORADO: Recomendaciones
            if missing_days_with_api_data > 0:
                assessment['recommendations'].append(f"Re-ingestar {missing_days_with_api_data} días de orderbook disponibles en API")
            
            if significant_discrepancies > 0:
                assessment['recommendations'].append(f"Reparar {significant_discrepancies} días orderbook con discrepancias significativas")
            
            if avg_per_day < 20000:
                assessment['recommendations'].append("Mejorar densidad de snapshots de orderbook")
                
        except Exception as e:
            log.warning(f"Error evaluando calidad orderbook: {e}")
        
        return assessment
    
    def _log_ohlcv_api_results(self, symbol: str, api_verification: Dict):
        """Log resultados de verificación OHLCV API - MEJORADO"""
        log.info(f"\n🔍 RESULTADOS VERIFICACIÓN API OHLCV:")
        
        missing_checked = api_verification.get('missing_days_checked', [])
        if missing_checked:
            log.info(f"  Días faltantes verificados: {len(missing_checked)}")
            api_has_data = sum(1 for d in missing_checked if d['api_has_data'])
            log.info(f"  API tiene datos para: {api_has_data}/{len(missing_checked)} días faltantes")
            
            if api_has_data > 0:
                log.warning(f"  ⚠️ HAY {api_has_data} días con datos disponibles en API que no tenemos")
                for day_info in missing_checked:
                    if day_info['api_has_data']:
                        log.warning(f"    • {day_info['date']}: API tiene {day_info['api_records']} registros")
        
        sparse_checked = api_verification.get('sparse_days_checked', [])
        significant_discrepancies = api_verification.get('significant_discrepancies', 0)
        
        if sparse_checked:
            log.info(f"  Días escasos verificados: {len(sparse_checked)}")
            if significant_discrepancies > 0:
                log.warning(f"  🚨 DISCREPANCIAS SIGNIFICATIVAS: {significant_discrepancies}")
            
            for day_info in sparse_checked:
                marker = "🚨" if day_info.get('has_significant_discrepancy', False) else "ℹ️"
                log.info(f"    {marker} {day_info['date']}: BD={day_info['current_records']}, API={day_info['api_records']}")
    
    def _log_orderbook_api_results(self, symbol: str, api_verification: Dict):
        """Log resultados de verificación orderbook API - MEJORADO"""
        log.info(f"\n🔍 RESULTADOS VERIFICACIÓN API ORDERBOOK:")
        
        missing_checked = api_verification.get('missing_days_checked', [])
        if missing_checked:
            api_has_data = sum(1 for d in missing_checked if d['api_has_data'])
            log.info(f"  Días orderbook escasos verificados: {len(missing_checked)}")
            
            if api_has_data > 0:
                log.warning(f"  ⚠️ HAY {api_has_data} días con orderbook disponible en API que no tenemos")
        
        sparse_checked = api_verification.get('sparse_days_checked', [])
        significant_discrepancies = api_verification.get('significant_discrepancies', 0)
        
        if sparse_checked:
            log.info(f"  Días orderbook escasos verificados: {len(sparse_checked)}")
            if significant_discrepancies > 0:
                log.warning(f"  🚨 DISCREPANCIAS ORDERBOOK SIGNIFICATIVAS: {significant_discrepancies}")
            
            for day_info in sparse_checked:
                marker = "🚨" if day_info.get('has_significant_discrepancy', False) else "ℹ️"
                log.info(f"    {marker} {day_info['date']}: BD={day_info['current_snapshots']}, API={day_info['api_snapshots']}")

    def _diagnose_funding_enhanced(self, symbol: str, start_date: datetime, end_date: datetime, verify_api: bool) -> Dict:
        """Diagnóstico funding rates con verificación de MEXC API"""
        log.info(f"\n📊 ENHANCED FUNDING DIAGNOSIS")
        log.info(f"{'='*50}")
        
        try:
            # Diagnóstico básico primero
            basic_diagnosis = self._diagnose_funding_basic(symbol, start_date, end_date)
            
            if not verify_api or basic_diagnosis.get('status') == 'NO_DATA':
                return basic_diagnosis
            
            # Verificación con MEXC API
            log.info(f"🔍 Verificando MEXC API para funding rates...")
            
            api_verification = self._verify_funding_with_mexc(symbol, start_date, end_date)
            
            # Combinar diagnóstico básico con verificación de API
            enhanced_diagnosis = basic_diagnosis.copy()
            enhanced_diagnosis['api_verification'] = api_verification
            enhanced_diagnosis['data_quality_assessment'] = self._assess_funding_quality(basic_diagnosis, api_verification)
            
            self._log_funding_api_results(symbol, api_verification)
            
            return enhanced_diagnosis
            
        except Exception as e:
            log.error(f"❌ Error en diagnóstico funding enhanced: {e}")
            return {'error': str(e)}
    
    def _diagnose_funding_basic(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Diagnóstico funding básico - FIXED: LÓGICA ADAPTATIVA"""
        with db_manager.get_session() as session:
            stats_query = text("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record,
                    COUNT(DISTINCT DATE(timestamp)) as days_with_data,
                    AVG(funding_rate) as avg_funding_rate
                FROM funding_rates
                WHERE symbol = :symbol
                AND timestamp BETWEEN :start_date AND :end_date
            """)
            
            stats = session.execute(stats_query, {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date
            }).fetchone()
            
            if not stats or not stats.total_records:
                return {
                    'total_records': 0,
                    'expected_records': 0,
                    'coverage_pct': 0,
                    'missing_records': 0,
                    'days_with_data': 0,
                    'status': 'NO_DATA'
                }
            
            # NUEVO: Detectar patrón de frecuencia real
            frequency_pattern = self._detect_funding_frequency_pattern(symbol, session, start_date, end_date)
            
            total_days = (end_date - start_date).days + 1
            
            # FIXED: Usar frecuencia detectada en lugar de asumir 3/día
            if frequency_pattern['avg_per_day'] > 0:
                expected_per_day = frequency_pattern['avg_per_day']
                total_expected = int(total_days * expected_per_day)
            else:
                # Fallback: usar promedio real de días con datos
                expected_per_day = stats.total_records / max(1, stats.days_with_data)
                total_expected = stats.total_records  # Si no hay patrón claro, asumir que lo que hay es correcto
            
            coverage_pct = (stats.total_records / total_expected * 100) if total_expected > 0 else 0
            
            log.info(f"📈 FUNDING BÁSICO: {stats.total_records:,} registros ({coverage_pct:.2f}%)")
            log.info(f"📊 Patrón detectado: {frequency_pattern['description']}")
            
            return {
                'total_records': stats.total_records,
                'expected_records': total_expected,
                'coverage_pct': coverage_pct,
                'missing_records': max(0, total_expected - stats.total_records),
                'days_with_data': stats.days_with_data,
                'avg_funding_rate': stats.avg_funding_rate,
                'frequency_pattern': frequency_pattern
            }
    
    def _detect_funding_frequency_pattern(self, symbol: str, session, start_date: datetime, end_date: datetime) -> Dict:
        """NUEVO: Detectar patrón de frecuencia real de funding rates"""
        
        # Analizar registros por día para detectar patrón
        daily_analysis = session.execute(text("""
            SELECT 
                DATE(timestamp) as day,
                COUNT(*) as records_per_day,
                STRING_AGG(EXTRACT(HOUR FROM timestamp)::text, ',' ORDER BY timestamp) as hours
            FROM funding_rates 
            WHERE symbol = :symbol
            AND timestamp BETWEEN :start_date AND :end_date
            GROUP BY DATE(timestamp)
            ORDER BY day
        """), {'symbol': symbol, 'start_date': start_date, 'end_date': end_date}).fetchall()
        
        if not daily_analysis:
            return {'avg_per_day': 3, 'description': 'Sin datos - asumiendo 3/día (cada 8h)'}
        
        # Calcular estadísticas
        daily_counts = [row.records_per_day for row in daily_analysis]
        avg_per_day = sum(daily_counts) / len(daily_counts)
        
        # Detectar patrón predominante
        count_3_per_day = sum(1 for c in daily_counts if c == 3)
        count_6_per_day = sum(1 for c in daily_counts if c == 6)
        total_days = len(daily_counts)
        
        # Determinar descripción del patrón
        if count_6_per_day > count_3_per_day and count_6_per_day / total_days > 0.6:
            pattern_desc = f"Principalmente 6/día (cada 4h): {count_6_per_day}/{total_days} días ({count_6_per_day/total_days*100:.1f}%)"
            expected_avg = 6
        elif count_3_per_day > count_6_per_day and count_3_per_day / total_days > 0.6:
            pattern_desc = f"Principalmente 3/día (cada 8h): {count_3_per_day}/{total_days} días ({count_3_per_day/total_days*100:.1f}%)"
            expected_avg = 3
        else:
            pattern_desc = f"Patrón mixto: {count_3_per_day} días 3/día, {count_6_per_day} días 6/día, promedio {avg_per_day:.1f}/día"
            expected_avg = avg_per_day
        
        return {
            'avg_per_day': expected_avg,
            'description': pattern_desc,
            'days_with_3': count_3_per_day,
            'days_with_6': count_6_per_day,
            'total_days': total_days,
            'actual_avg': avg_per_day
        }
    
    def _verify_funding_with_mexc(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Verificar funding rates con MEXC API"""
        try:
            log.info(f"  📡 Consultando MEXC API para funding rates...")
            
            # Verificar funding actual
            current_funding = self.mexc.get_current_funding_rate(symbol)
            
            # Verificar muestra de historial (últimos 7 días para no sobrecargar)
            recent_start = max(start_date, end_date - timedelta(days=7))
            historical_funding = self.mexc.get_funding_rate_history_range(symbol, recent_start, end_date)
            
            return {
                'current_funding_available': current_funding is not None,
                'current_funding_rate': current_funding.get('fundingRate') if current_funding else None,
                'historical_sample_records': len(historical_funding) if not historical_funding.empty else 0,
                'historical_sample_period': f"{recent_start.date()} to {end_date.date()}",
                'api_responsive': True
            }
            
        except Exception as e:
            log.warning(f"  ❌ Error consultando MEXC API: {e}")
            return {
                'api_responsive': False,
                'api_error': str(e)
            }
    
    def _diagnose_ohlcv_basic(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Diagnóstico OHLCV básico"""
        with db_manager.get_session() as session:
            # Estadísticas generales
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
            
            # Calcular métricas
            total_days = (end_date - start_date).days + 1
            expected_minutes_per_day = 1440  # 24 * 60
            total_expected = total_days * expected_minutes_per_day
            total_actual = stats.total_records or 0
            coverage_pct = (total_actual / total_expected * 100) if total_expected > 0 else 0
            
            log.info(f"📈 OHLCV BÁSICO: {total_actual:,} registros ({coverage_pct:.2f}%)")
            log.info(f"  Días: {stats.days_with_data} con datos, {total_days - stats.days_with_data} faltantes")
            
            return {
                'total_records': total_actual,
                'expected_records': total_expected,
                'coverage_pct': coverage_pct,
                'missing_records': total_expected - total_actual,
                'days_with_data': stats.days_with_data,
                'missing_days': total_days - stats.days_with_data,
            }
    
    def _get_problematic_ohlcv_days(self, symbol: str, start_date: datetime, end_date: datetime) -> Tuple[List[date], List[Dict]]:
        """Obtener días faltantes y días con pocos datos OHLCV"""
        with db_manager.get_session() as session:
            # Días con datos
            daily_query = text("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as minute_count
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
            
            # Días faltantes
            all_dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
            dates_with_data = set(pd.to_datetime(daily_df['date']).dt.date) if not daily_df.empty else set()
            missing_days = sorted(list(set(all_dates.date) - dates_with_data))
            
            # Días escasos (menos de 1000 minutos)
            sparse_days = []
            if not daily_df.empty:
                sparse_df = daily_df[daily_df['minute_count'] < 1000]
                sparse_days = [
                    {'date': pd.to_datetime(row['date']).date(), 'count': row['minute_count']}
                    for _, row in sparse_df.iterrows()
                ]
            
            return missing_days, sparse_days
    
    def _check_ohlcv_api_for_date(self, symbol: str, day: Union[date, datetime]) -> Dict:
        """Verificar si la API tiene datos OHLCV para una fecha específica"""
        try:
            if isinstance(day, datetime):
                date_str = day.date().isoformat()
            elif isinstance(day, date):
                date_str = day.isoformat()
            else:
                date_str = str(day)
            
            log.debug(f"  📡 Verificando API OHLCV para {symbol} en {date_str}")
            
            df = self.coinapi.get_ohlcv_for_date(symbol, date_str)
            
            return {
                'has_data': not df.empty,
                'record_count': len(df) if not df.empty else 0,
                'date': day
            }
            
        except Exception as e:
            log.debug(f"  ❌ Error API OHLCV {symbol} {day}: {e}")
            return {
                'has_data': False,
                'record_count': 0,
                'date': day,
                'error': str(e)
            }
    
    def _diagnose_orderbook_basic(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Diagnóstico orderbook básico - FIXED: EXCLUYE DÍA ACTUAL"""
        with db_manager.get_session() as session:
            # NUEVO: Ajustar end_date para excluir el día actual
            today = datetime.now().date()
            if end_date.date() >= today:
                end_date = datetime.combine(today - timedelta(days=1), datetime.max.time())
                log.debug(f"📅 Ajustando end_date para orderbook (excluyendo hoy): {end_date.date()}")
            
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
            
            if not stats or not stats.total_records:
                return {
                    'total_records': 0,
                    'valid_quotes': 0,
                    'avg_per_day': 0,
                    'days_with_data': 0,
                    'missing_days': (end_date - start_date).days + 1,
                    'status': 'NO_DATA'
                }
            
            # Análisis diario
            daily_query = text("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as snapshot_count
                FROM orderbook
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
            
            # Detectar días sin datos (excluyendo hoy)
            all_dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
            dates_with_data = set(pd.to_datetime(daily_df['date']).dt.date) if not daily_df.empty else set()
            missing_dates = set(all_dates.date) - dates_with_data
            
            log.info(f"📈 ORDERBOOK BÁSICO: {stats.total_records:,} snapshots")
            log.info(f"  Días: {stats.days_with_data} con datos, {len(missing_dates)} faltantes")
            
            return {
                'total_records': stats.total_records or 0,
                'valid_quotes': stats.valid_quotes or 0,
                'avg_per_day': stats.total_records / max(1, stats.days_with_data),
                'days_with_data': stats.days_with_data,
                'missing_days': len(missing_dates),
                'avg_spread_pct': stats.avg_spread_pct,
                'missing_dates_list': sorted(list(missing_dates))[:10]  # Primeros 10 para API check
            }
    
    def _get_problematic_orderbook_days(self, symbol: str, start_date: datetime, end_date: datetime) -> Tuple[List[date], List[Dict]]:
        """Obtener días faltantes y días con pocos snapshots - FIXED: EXCLUYE HOY"""
        with db_manager.get_session() as session:
            # NUEVO: Ajustar end_date para excluir el día actual
            today = datetime.now().date()
            if end_date.date() >= today:
                end_date = datetime.combine(today - timedelta(days=1), datetime.max.time())
                log.debug(f"📅 Ajustando end_date para orderbook problemático (excluyendo hoy): {end_date.date()}")
            
            daily_query = text("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as snapshot_count
                FROM orderbook
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
            
            # Días faltantes (excluyendo hoy)
            all_dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
            dates_with_data = set(pd.to_datetime(daily_df['date']).dt.date) if not daily_df.empty else set()
            missing_days = sorted(list(set(all_dates.date) - dates_with_data))
            
            # Días escasos (menos de 10,000 snapshots)
            sparse_days = []
            if not daily_df.empty:
                sparse_df = daily_df[daily_df['snapshot_count'] < 10000]
                sparse_days = [
                    {'date': pd.to_datetime(row['date']).date(), 'count': int(row['snapshot_count'])}
                    for _, row in sparse_df.iterrows()
                ]
            
            return missing_days, sparse_days
    
    def _check_orderbook_api_for_date(self, symbol: str, day: Union[date, datetime]) -> Dict:
        """Verificar si la API tiene datos orderbook para una fecha específica"""
        try:
            # Manejar diferentes tipos de fecha
            if isinstance(day, datetime):
                date_str = day.date().isoformat()
            elif isinstance(day, date):
                date_str = day.isoformat()
            else:
                date_str = str(day)
            
            log.debug(f"  📡 Verificando API orderbook para {symbol} en {date_str}")
            
            # Usar la misma función que usa la ingesta
            df = self.coinapi.get_orderbook_for_date(symbol, date_str)
            
            return {
                'has_data': not df.empty,
                'snapshot_count': len(df) if not df.empty else 0,
                'date': day
            }
            
        except Exception as e:
            log.debug(f"  ❌ Error API orderbook {symbol} {day}: {e}")
            return {
                'has_data': False,
                'snapshot_count': 0,
                'date': day,
                'error': str(e)
            }
    
    def _repair_ohlcv_day(self, symbol: str, target_date) -> int:
        """Repara datos OHLCV para un día específico"""
        try:
            # Convertir fecha
            if isinstance(target_date, str):
                target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
            elif isinstance(target_date, date):
                target_datetime = datetime.combine(target_date, datetime.min.time())
            else:
                target_datetime = target_date
            
            log.debug(f"    📊 Reparando OHLCV: {symbol} - {target_date}")
            
            # Eliminar datos existentes para ese día (si los hay)
            with db_manager.get_session() as session:
                delete_query = text("""
                    DELETE FROM ohlcv 
                    WHERE symbol = :symbol 
                    AND DATE(timestamp) = :target_date
                """)
                
                deleted_count = session.execute(delete_query, {
                    'symbol': symbol,
                    'target_date': target_date if isinstance(target_date, date) else target_date.date()
                }).rowcount
                session.commit()
                
                if deleted_count > 0:
                    log.debug(f"    🗑️ Eliminados {deleted_count} registros OHLCV existentes")
            
            # Obtener datos frescos de la API
            date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
            df = self.coinapi.get_ohlcv_for_date(symbol, date_str)
            
            if not df.empty:
                # Insertar datos usando la función existente
                records_count = self.ingestion._insert_ohlcv_data(symbol, df)
                return records_count
            else:
                log.debug(f"    ❌ No se obtuvieron datos OHLCV de API para {target_date}")
                return 0
                
        except Exception as e:
            log.error(f"Error reparando OHLCV {target_date}: {e}")
            return 0
    
    def _repair_orderbook_day(self, symbol: str, target_date) -> int:
        """Repara datos orderbook para un día específico"""
        try:
            # Convertir fecha
            if isinstance(target_date, str):
                target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
            elif isinstance(target_date, date):
                target_datetime = datetime.combine(target_date, datetime.min.time())
            else:
                target_datetime = target_date
            
            log.debug(f"    📊 Reparando orderbook: {symbol} - {target_date}")
            
            # Eliminar datos existentes para ese día (si los hay)
            with db_manager.get_session() as session:
                delete_query = text("""
                    DELETE FROM orderbook 
                    WHERE symbol = :symbol 
                    AND DATE(timestamp) = :target_date
                """)
                
                deleted_count = session.execute(delete_query, {
                    'symbol': symbol,
                    'target_date': target_date if isinstance(target_date, date) else target_date.date()
                }).rowcount
                session.commit()
                
                if deleted_count > 0:
                    log.debug(f"    🗑️ Eliminados {deleted_count} snapshots orderbook existentes")
            
            # Obtener datos frescos de la API
            date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
            df = self.coinapi.get_orderbook_for_date(symbol, date_str)
            
            if not df.empty:
                # Insertar datos usando la función existente
                records_count = self.ingestion._insert_orderbook_data(symbol, df)
                return records_count
            else:
                log.debug(f"    ❌ No se obtuvieron datos orderbook de API para {target_date}")
                return 0
                
        except Exception as e:
            log.error(f"Error reparando orderbook {target_date}: {e}")
            return 0
    
    def _get_data_stats(self, symbol: str) -> Dict:
        """Obtiene estadísticas actuales de datos para un símbolo"""
        stats = {}
        
        with db_manager.get_session() as session:
            # OHLCV stats
            ohlcv_result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    COUNT(DISTINCT DATE(timestamp)) as days_with_data
                FROM ohlcv 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            stats['ohlcv'] = {
                'total_records': ohlcv_result.total_records or 0,
                'days_with_data': ohlcv_result.days_with_data or 0,
                'min_date': ohlcv_result.min_date,
                'max_date': ohlcv_result.max_date
            }
            
            # Orderbook stats
            orderbook_result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    COUNT(DISTINCT DATE(timestamp)) as days_with_data
                FROM orderbook 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            stats['orderbook'] = {
                'total_records': orderbook_result.total_records or 0,
                'days_with_data': orderbook_result.days_with_data or 0,
                'min_date': orderbook_result.min_date,
                'max_date': orderbook_result.max_date
            }
        
        return stats
    
    def _log_repair_results(self, symbol: str, repair_results: Dict):
        """Log resultados del surgical repair"""
        log.info(f"\n📊 SURGICAL REPAIR RESULTS - {symbol}")
        log.info(f"{'='*50}")
        
        pre_stats = repair_results['pre_repair_stats']
        post_stats = repair_results['post_repair_stats']
        
        # OHLCV repair results
        ohlcv_repair = repair_results['ohlcv_repair']
        if ohlcv_repair['attempted']:
            log.info(f"🔧 OHLCV REPAIR:")
            log.info(f"  Pre-repair: {pre_stats['ohlcv']['total_records']:,} registros")
            log.info(f"  Post-repair: {post_stats['ohlcv']['total_records']:,} registros")
            log.info(f"  Registros añadidos: +{ohlcv_repair['records_added']:,}")
            log.info(f"  Estado: {'✅ ÉXITO' if ohlcv_repair['success'] else '❌ FALLO'}")
        
        # Orderbook repair results
        orderbook_repair = repair_results['orderbook_repair']
        if orderbook_repair['attempted']:
            log.info(f"🔧 ORDERBOOK REPAIR:")
            log.info(f"  Pre-repair: {pre_stats['orderbook']['total_records']:,} snapshots")
            log.info(f"  Post-repair: {post_stats['orderbook']['total_records']:,} snapshots")
            log.info(f"  Snapshots añadidos: +{orderbook_repair['records_added']:,}")
            log.info(f"  Estado: {'✅ ÉXITO' if orderbook_repair['success'] else '❌ FALLO'}")
        
        # Overall improvement
        total_improvement = ohlcv_repair['records_added'] + orderbook_repair['records_added']
        if total_improvement > 0:
            log.info(f"✅ TOTAL IMPROVEMENT: +{total_improvement:,} registros/snapshots")
        else:
            log.info(f"ℹ️ No se encontraron datos adicionales para reparar")
    
    def _sample_days(self, days_list: Union[List[date], List[Dict]], max_samples: int) -> List:
        """Samplear días para verificación de API"""
        if len(days_list) <= max_samples:
            return days_list
        
        # NUEVO: Filtrar el día de hoy automáticamente
        today = datetime.now().date()
        
        # Manejar lista de dates o lista de dicts
        if days_list and isinstance(days_list[0], dict):
            # Lista de dicts (sparse days)
            filtered_days = [d for d in days_list if d['date'] < today]
            recent_days = [d for d in filtered_days if d['date'] >= (today - timedelta(days=30))]
            older_days = [d for d in filtered_days if d not in recent_days]
        else:
            # Lista de dates (missing days)
            filtered_days = [d for d in days_list if isinstance(d, date) and d < today]
            recent_days = [d for d in filtered_days if d >= (today - timedelta(days=30))]
            older_days = [d for d in filtered_days if d not in recent_days]
        
        # Priorizar días recientes
        sample = recent_days[:max_samples//2] if recent_days else []
        remaining = max_samples - len(sample)
        
        if remaining > 0 and older_days:
            sample.extend(random.sample(older_days, min(remaining, len(older_days))))
        
        return sample
    
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
    
    def _assess_funding_quality(self, basic: Dict, api_verification: Dict) -> Dict:
        """Evaluar calidad de funding rates"""
        assessment = {
            'overall_grade': 'UNKNOWN',
            'completeness': 'UNKNOWN',
            'api_health': 'UNKNOWN',
            'recommendations': []
        }
        
        try:
            coverage_pct = basic.get('coverage_pct', 0)
            api_responsive = api_verification.get('api_responsive', False)
            
            # Evaluar completitud
            if coverage_pct >= 95:
                assessment['completeness'] = 'EXCELLENT'
            elif coverage_pct >= 80:
                assessment['completeness'] = 'GOOD'
            else:
                assessment['completeness'] = 'POOR'
            
            # Evaluar salud de API
            assessment['api_health'] = 'GOOD' if api_responsive else 'POOR'
            
            # Grado general
            if assessment['completeness'] == 'POOR' or assessment['api_health'] == 'POOR':
                assessment['overall_grade'] = 'POOR'
            else:
                assessment['overall_grade'] = assessment['completeness']
            
            # Recomendaciones
            if coverage_pct < 90:
                assessment['recommendations'].append("Completar historial de funding rates desde MEXC")
            
            if not api_responsive:
                assessment['recommendations'].append("Verificar conectividad con MEXC API")
                
        except Exception as e:
            log.warning(f"Error evaluando calidad funding: {e}")
        
        return assessment
    
    def _log_funding_api_results(self, symbol: str, api_verification: Dict):
        """Log resultados de verificación funding API"""
        log.info(f"\n🔍 RESULTADOS VERIFICACIÓN MEXC API:")
        
        if api_verification.get('api_responsive'):
            log.info(f"  ✅ MEXC API respondiendo correctamente")
            if api_verification.get('current_funding_available'):
                log.info(f"  ✅ Funding rate actual disponible: {api_verification.get('current_funding_rate')}")
            
            sample_records = api_verification.get('historical_sample_records', 0)
            log.info(f"  📊 Muestra histórica: {sample_records} registros")
        else:
            log.error(f"  ❌ MEXC API no respondiendo: {api_verification.get('api_error')}")
    
    def _log_enhanced_summary(self, results: Dict):
        """Log resumen mejorado del diagnóstico"""
        symbol = results['symbol']
        
        log.info(f"\n{'='*80}")
        log.info(f"📊 ENHANCED SUMMARY: {symbol}")
        log.info(f"{'='*80}")
        
        # OHLCV Summary with API insights
        if 'ohlcv' in results and 'error' not in results['ohlcv']:
            ohlcv = results['ohlcv']
            if ohlcv.get('status') == 'NO_DATA':
                log.warning(f"\n📈 OHLCV: NO HAY DATOS")
            else:
                log.info(f"\n📈 OHLCV:")
                log.info(f"  Cobertura: {ohlcv.get('coverage_pct', 0):.2f}%")
                
                # Enhanced assessment
                assessment = ohlcv.get('data_quality_assessment', {})
                overall_grade = assessment.get('overall_grade', 'UNKNOWN')
                log.info(f"  Calidad general: {overall_grade}")
                
                # API insights
                api_verification = ohlcv.get('api_verification', {})
                missing_with_api = api_verification.get('api_has_data_for_missing', 0)
                significant_discrepancies = api_verification.get('significant_discrepancies', 0)
                
                if missing_with_api > 0:
                    log.warning(f"  ⚠️ {missing_with_api} días faltantes tienen datos en API")
                if significant_discrepancies > 0:
                    log.warning(f"  🚨 {significant_discrepancies} días con discrepancias significativas")
                
                # Recommendations
                recommendations = assessment.get('recommendations', [])
                if recommendations:
                    log.info(f"  Recomendaciones:")
                    for rec in recommendations[:3]:  # Top 3
                        log.info(f"    • {rec}")
        
        # Orderbook Summary with API insights
        if 'orderbook' in results and 'error' not in results['orderbook']:
            orderbook = results['orderbook']
            if orderbook.get('status') == 'NO_DATA':
                log.warning(f"\n📚 ORDERBOOK: NO HAY DATOS")
            else:
                log.info(f"\n📚 ORDERBOOK:")
                log.info(f"  Snapshots totales: {orderbook.get('total_records', 0):,}")
                log.info(f"  Promedio diario: {orderbook.get('avg_per_day', 0):.0f}")
                
                # Enhanced assessment
                assessment = orderbook.get('data_quality_assessment', {})
                overall_grade = assessment.get('overall_grade', 'UNKNOWN')
                log.info(f"  Calidad general: {overall_grade}")
                
                # API insights
                api_verification = orderbook.get('api_verification', {})
                missing_with_api = api_verification.get('api_has_data_for_missing', 0)
                significant_discrepancies = api_verification.get('significant_discrepancies', 0)
                
                if missing_with_api > 0:
                    log.warning(f"  ⚠️ {missing_with_api} días faltantes tienen orderbook en API")
                if significant_discrepancies > 0:
                    log.warning(f"  🚨 {significant_discrepancies} días orderbook con discrepancias significativas")
        
        # Funding Summary with API insights
        if 'funding_rates' in results and isinstance(results['funding_rates'], dict):
            funding = results['funding_rates']
            if funding.get('status') == 'NO_DATA':
                log.warning(f"\n💰 FUNDING RATES: NO HAY DATOS")
            elif 'coverage_pct' in funding:
                log.info(f"\n💰 FUNDING RATES:")
                log.info(f"  Cobertura: {funding.get('coverage_pct', 0):.2f}%")
                
                # Enhanced assessment
                assessment = funding.get('data_quality_assessment', {})
                if assessment:
                    overall_grade = assessment.get('overall_grade', 'UNKNOWN')
                    log.info(f"  Calidad general: {overall_grade}")
                
                # NUEVO: Mostrar patrón de frecuencia detectado
                frequency_pattern = funding.get('frequency_pattern', {})
                if frequency_pattern:
                    log.info(f"  Patrón detectado: {frequency_pattern.get('description', 'N/A')}")
                
                # API insights
                api_verification = funding.get('api_verification', {})
                if api_verification.get('api_responsive'):
                    log.info(f"  ✅ MEXC API operativa")
                else:
                    log.error(f"  ❌ MEXC API no disponible")
        
        # Surgical Repair Summary
        if 'surgical_repair' in results:
            repair = results['surgical_repair']
            log.info(f"\n🔧 SURGICAL REPAIR:")
            
            ohlcv_repair = repair.get('ohlcv_repair', {})
            orderbook_repair = repair.get('orderbook_repair', {})
            
            if ohlcv_repair.get('attempted'):
                status = '✅ ÉXITO' if ohlcv_repair.get('success') else '❌ FALLO'
                log.info(f"  OHLCV: +{ohlcv_repair.get('records_added', 0)} registros - {status}")
            
            if orderbook_repair.get('attempted'):
                status = '✅ ÉXITO' if orderbook_repair.get('success') else '❌ FALLO'
                log.info(f"  Orderbook: +{orderbook_repair.get('records_added', 0)} snapshots - {status}")
        
        # Overall recommendations
        log.info(f"\n💡 RECOMENDACIONES PRIORITARIAS:")
        
        all_recommendations = []
        for data_type in ['ohlcv', 'orderbook', 'funding_rates']:
            if data_type in results and 'data_quality_assessment' in results[data_type]:
                recommendations = results[data_type]['data_quality_assessment'].get('recommendations', [])
                all_recommendations.extend(recommendations)
        
        if all_recommendations:
            # Mostrar top 5 recomendaciones únicas
            unique_recommendations = list(dict.fromkeys(all_recommendations))[:5]
            for i, rec in enumerate(unique_recommendations, 1):
                log.info(f"  {i}. {rec}")
        else:
            log.info(f"  ✅ No se encontraron problemas críticos")
        
        log.info(f"\n✅ Enhanced diagnosis completado para {symbol}")
        log.info(f"{'='*80}\n")
    
    def log_global_surgical_summary(self):
        """Log resumen global del surgical repair"""
        if self.enable_surgical_repair:
            log.info(f"\n{'='*80}")
            log.info(f"🔧 SURGICAL REPAIR GLOBAL SUMMARY")
            log.info(f"{'='*80}")
            log.info(f"  Símbolos reparados: {self.repair_stats['symbols_repaired']}")
            log.info(f"  Total OHLCV records añadidos: {self.repair_stats['total_ohlcv_repaired']:,}")
            log.info(f"  Total orderbook snapshots añadidos: {self.repair_stats['total_orderbook_repaired']:,}")
            log.info(f"  Fallos de reparación: {self.repair_stats['repair_failures']}")
            
            total_repairs = self.repair_stats['total_ohlcv_repaired'] + self.repair_stats['total_orderbook_repaired']
            if total_repairs > 0:
                log.info(f"✅ SURGICAL REPAIR EXITOSO: +{total_repairs:,} registros/snapshots totales")
            else:
                log.info(f"ℹ️ No se encontraron datos adicionales que reparar")

def main():
    """Función principal del diagnóstico mejorado con surgical repair"""
    parser = argparse.ArgumentParser(description="🔍 Enhanced data integrity diagnosis with API verification + surgical repair")
    parser.add_argument("--symbol", type=str, help="Analizar símbolo específico")
    parser.add_argument("--no-api", action="store_true", help="Saltar verificación de API")
    parser.add_argument("--quick", action="store_true", help="Análisis rápido (menos verificaciones de API)")
    parser.add_argument("--surgical-repair", action="store_true", help="🔧 Ejecutar surgical repair automático para datos faltantes")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar qué se repararía, no ejecutar")
    
    args = parser.parse_args()
    
    log.info("🔍 INICIANDO ENHANCED DIAGNOSIS CON VERIFICACIÓN DE API")
    if args.surgical_repair:
        if args.dry_run:
            log.info("🔧 MODO: Surgical repair DRY RUN (solo simulación)")
        else:
            log.info("🔧 MODO: Surgical repair ACTIVO (reparará datos automáticamente)")
    log.info(f"Timestamp: {datetime.now()}")
    
    verify_api = not args.no_api
    enable_repair = args.surgical_repair and not args.dry_run
    
    if not verify_api:
        log.info("⚠️ Saltando verificación de API (modo --no-api)")
    elif args.quick:
        log.info("⚡ Modo rápido - verificaciones limitadas de API")
    
# UPDATED: Obtener símbolos desde symbol_info table
    if args.symbol:
        symbols = [args.symbol]
        log.info(f"Analizando símbolo específico: {args.symbol}")
    else:
        log.info("🔍 Getting symbols from symbol_info table...")
        
        # Simplificado: obtener todos los símbolos de symbol_info
        symbols = settings.get_symbols_from_db()
        
        if symbols:
            log.info(f"✅ Found {len(symbols)} symbols in symbol_info table")
            log.info(f"📋 Data source: Database (symbol_info table)")
        else:
            # Fallback: use YAML (for cases where DB is not populated yet)
            log.warning("⚠️ No symbols found in database, falling back to YAML configuration...")
            try:
                active_pairs = settings.get_active_pairs()
                symbols = []
                for pair in active_pairs:
                    symbols.extend([pair.symbol1, pair.symbol2])
                symbols = list(set(symbols))
                log.info(f"✅ Found {len(symbols)} symbols from YAML configuration")
                log.info(f"📋 Data source: YAML configuration (fallback)")
            except Exception as yaml_error:
                log.error(f"Failed to load symbols from YAML: {yaml_error}")
                # Ultimate fallback
                symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
                log.warning(f"Using default symbols: {symbols}")
                log.info(f"📋 Data source: Hardcoded defaults")        
        # Log data source
        if settings.get_active_symbols_from_db():
            log.info(f"📋 Data source: Database (active symbols from active pairs)")
        elif settings.get_symbols_from_db():
            log.info(f"📋 Data source: Database (all available symbols)")
        else:
            log.info(f"📋 Data source: YAML configuration (fallback)")
    
    if not symbols:
        log.error("No hay símbolos para analizar")
        return False
    
    # Crear diagnóstico mejorado con surgical repair
    diagnosis = EnhancedDataDiagnosis(enable_surgical_repair=enable_repair)
    all_results = {}
    
    # Analizar cada símbolo
    start_time = datetime.now()
    
    for i, symbol in enumerate(symbols):
        symbol_start = datetime.now()
        log.info(f"\n🔄 Procesando símbolo {i+1}/{len(symbols)}: {symbol}")
        
        try:
            results = diagnosis.diagnose_symbol_enhanced(symbol, verify_api=verify_api)
            all_results[symbol] = results
            
            symbol_duration = (datetime.now() - symbol_start).total_seconds()
            log.info(f"⏱️ Tiempo de análisis para {symbol}: {symbol_duration:.1f} segundos")
            
        except Exception as e:
            log.error(f"❌ Error analizando {symbol}: {e}")
            import traceback
            log.debug(traceback.format_exc())
            all_results[symbol] = {'error': str(e)}
    
    # Resumen final global mejorado
    total_duration = (datetime.now() - start_time).total_seconds()
    
    log.info(f"\n{'='*80}")
    log.info(f"🏁 ENHANCED DIAGNOSIS COMPLETADO")
    log.info(f"{'='*80}")
    log.info(f"Total símbolos analizados: {len(all_results)}")
    log.info(f"Tiempo total: {total_duration:.1f} segundos ({total_duration/60:.1f} minutos)")
    
    # Estadísticas globales mejoradas
    total_api_missing_days = 0
    total_discrepancies = 0
    symbols_with_api_issues = 0
    
    for symbol, results in all_results.items():
        if 'error' in results:
            continue
            
        # Contar días con datos disponibles en API que no tenemos + discrepancias
        for data_type in ['ohlcv', 'orderbook']:
            if data_type in results and 'api_verification' in results[data_type]:
                missing_with_api = results[data_type]['api_verification'].get('api_has_data_for_missing', 0)
                discrepancies = results[data_type]['api_verification'].get('significant_discrepancies', 0)
                
                if missing_with_api > 0 or discrepancies > 0:
                    total_api_missing_days += missing_with_api
                    total_discrepancies += discrepancies
                    symbols_with_api_issues += 1
                    break
    
    log.info(f"\n📊 ENHANCED STATISTICS:")
    log.info(f"  Días con datos disponibles en API no ingresados: {total_api_missing_days}")
    log.info(f"  Días con discrepancias significativas detectadas: {total_discrepancies}")
    log.info(f"  Símbolos con problemas de ingesta detectados: {symbols_with_api_issues}")
    
    # Log global surgical repair summary
    diagnosis.log_global_surgical_summary()
    
    if verify_api:
        if enable_repair:
            if diagnosis.repair_stats['symbols_repaired'] > 0:
                log.info(f"\n✅ SURGICAL REPAIR EXITOSO")
                log.info(f"  {diagnosis.repair_stats['symbols_repaired']} símbolos reparados")
                log.info(f"  +{diagnosis.repair_stats['total_ohlcv_repaired']:,} registros OHLCV")
                log.info(f"  +{diagnosis.repair_stats['total_orderbook_repaired']:,} snapshots orderbook")
            else:
                log.info(f"\n✅ NO SE DETECTARON PROBLEMAS DE INGESTA QUE REPARAR")
        elif total_api_missing_days > 0 or total_discrepancies > 0:
            log.error(f"\n❌ PROBLEMAS DE INGESTA DETECTADOS:")
            if total_api_missing_days > 0:
                log.error(f"  {total_api_missing_days} días tienen datos en API que no fueron ingresados")
            if total_discrepancies > 0:
                log.error(f"  {total_discrepancies} días tienen discrepancias significativas (API tiene mucho más datos)")
            log.error(f"  Ejecutar con --surgical-repair para reparar automáticamente")
        else:
            log.info(f"\n✅ NO SE DETECTARON PROBLEMAS DE INGESTA")
            log.info(f"  Los gaps en datos corresponden a falta de datos en las APIs")
    
    log.info(f"\n📁 Logs detallados en: logs/enhanced_data_diagnosis.log")
    log.info(f"🏁 Enhanced diagnosis finalizado: {datetime.now()}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)