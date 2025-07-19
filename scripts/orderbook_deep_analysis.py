#!/usr/bin/env python3
"""
📊 ORDERBOOK DATA DEEP ANALYSIS - Análisis exhaustivo de datos de orderbook
Version 1.0 - Analiza completitud, calidad, frecuencias, gaps, y todo lo necesario

Funcionalidades:
- Análisis temporal completo (días, frecuencias, gaps)
- Calidad de datos por niveles de liquidez
- Distribuciones de spreads y volúmenes
- Detección de anomalías y problemas
- Reportes detallados con visualizaciones
- Recomendaciones de limpieza
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

# Styling
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 9
sns.set_palette("husl")

class OrderbookDeepAnalysis:
    """Análisis exhaustivo de datos de orderbook"""
    
    def __init__(self):
        self.analysis_results = {}
        self.global_stats = {}
        
    def analyze_symbol(self, symbol: str) -> Dict:
        """Análisis completo para un símbolo específico"""
        log.info(f"🔍 Iniciando análisis profundo de orderbook para {symbol}")
        
        try:
            results = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'basic_stats': self._get_basic_stats(symbol),
                'temporal_analysis': self._analyze_temporal_coverage(symbol),
                'frequency_analysis': self._analyze_frequency_patterns(symbol),
                'quality_analysis': self._analyze_data_quality(symbol),
                'liquidity_analysis': self._analyze_liquidity_levels(symbol),
                'spread_analysis': self._analyze_spread_distributions(symbol),
                'anomaly_analysis': self._detect_anomalies(symbol),
                'completeness_score': 0,
                'recommendations': []
            }
            
            # Calcular score general de completitud
            results['completeness_score'] = self._calculate_completeness_score(results)
            
            # Generar recomendaciones
            results['recommendations'] = self._generate_recommendations(results)
            
            log.info(f"✅ Análisis completado para {symbol}")
            return results
            
        except Exception as e:
            log.error(f"❌ Error analizando {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
            return {'symbol': symbol, 'error': str(e)}
    
    def _get_basic_stats(self, symbol: str) -> Dict:
        """Estadísticas básicas del símbolo"""
        log.info(f"  📋 Obteniendo estadísticas básicas...")
        
        with db_manager.get_session() as session:
            # Estadísticas generales
            basic_query = text("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record,
                    COUNT(DISTINCT DATE(timestamp)) as unique_days,
                    COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL THEN 1 END) as valid_quotes,
                    COUNT(CASE WHEN bid1_price IS NULL OR ask1_price IS NULL THEN 1 END) as invalid_quotes
                FROM orderbook 
                WHERE symbol = :symbol
            """)
            
            basic_result = session.execute(basic_query, {'symbol': symbol}).fetchone()
            
            if not basic_result.first_record:
                return {'error': 'No data found'}
            
            # Análisis de días
            total_days_possible = (basic_result.last_record - basic_result.first_record).days + 1
            data_coverage_pct = (basic_result.unique_days / total_days_possible) * 100
            
            # Estadísticas de calidad
            records_per_day_avg = basic_result.total_records / basic_result.unique_days if basic_result.unique_days > 0 else 0
            valid_quotes_pct = (basic_result.valid_quotes / basic_result.total_records) * 100 if basic_result.total_records > 0 else 0
            
            return {
                'total_records': basic_result.total_records,
                'first_record': basic_result.first_record,
                'last_record': basic_result.last_record,
                'date_range_days': total_days_possible,
                'unique_days_with_data': basic_result.unique_days,
                'data_coverage_pct': data_coverage_pct,
                'records_per_day_avg': records_per_day_avg,
                'valid_quotes': basic_result.valid_quotes,
                'invalid_quotes': basic_result.invalid_quotes,
                'valid_quotes_pct': valid_quotes_pct
            }
    
    def _analyze_temporal_coverage(self, symbol: str) -> Dict:
        """Análisis detallado de cobertura temporal"""
        log.info(f"  🕐 Analizando cobertura temporal...")
        
        with db_manager.get_session() as session:
            # Obtener datos por día
            daily_stats_query = text("""
                SELECT 
                    DATE(timestamp) as day,
                    COUNT(*) as records_count,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp,
                    COUNT(DISTINCT EXTRACT(HOUR FROM timestamp)) as hours_with_data,
                    COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL THEN 1 END) as valid_records
                FROM orderbook 
                WHERE symbol = :symbol
                GROUP BY DATE(timestamp)
                ORDER BY day
            """)
            
            daily_df = pd.read_sql(daily_stats_query, session.bind, params={'symbol': symbol})
            
            if daily_df.empty:
                return {'error': 'No daily data found'}
            
            daily_df['day'] = pd.to_datetime(daily_df['day'])
            
            # Generar rango completo de fechas esperadas
            date_range = pd.date_range(
                start=daily_df['day'].min(),
                end=daily_df['day'].max(),
                freq='D'
            )
            
            # Crear DataFrame completo con días faltantes
            complete_df = pd.DataFrame({'day': date_range})
            merged_df = complete_df.merge(daily_df, on='day', how='left')
            merged_df['records_count'] = merged_df['records_count'].fillna(0)
            merged_df['has_data'] = merged_df['records_count'] > 0
            
            # Clasificar días por calidad
            merged_df['day_quality'] = merged_df['records_count'].apply(
                lambda x: 'excellent' if x >= 1000
                else 'good' if x >= 500
                else 'fair' if x >= 100
                else 'poor' if x > 0
                else 'missing'
            )
            
            # Identificar gaps (días consecutivos sin datos)
            gaps = self._find_data_gaps(merged_df)
            
            # Estadísticas de cobertura
            days_with_data = merged_df['has_data'].sum()
            total_possible_days = len(merged_df)
            coverage_pct = (days_with_data / total_possible_days) * 100
            
            # Estadísticas por calidad
            quality_stats = merged_df['day_quality'].value_counts().to_dict()
            
            # Análisis de horarios (para días con datos)
            hourly_coverage = daily_df['hours_with_data'].describe().to_dict()
            
            return {
                'total_possible_days': total_possible_days,
                'days_with_data': int(days_with_data),
                'days_missing': int(total_possible_days - days_with_data),
                'coverage_percentage': coverage_pct,
                'quality_breakdown': quality_stats,
                'gaps_found': gaps,
                'hourly_coverage_stats': hourly_coverage,
                'daily_data': merged_df,
                'avg_records_per_day': daily_df['records_count'].mean(),
                'median_records_per_day': daily_df['records_count'].median(),
                'std_records_per_day': daily_df['records_count'].std()
            }
    
    def _analyze_frequency_patterns(self, symbol: str) -> Dict:
        """Análisis de patrones de frecuencia"""
        log.info(f"  📊 Analizando patrones de frecuencia...")
        
        with db_manager.get_session() as session:
            # Analizar intervalos entre registros
            intervals_query = text("""
                WITH time_diffs AS (
                    SELECT 
                        timestamp,
                        LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                        EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)))/60 as minutes_diff
                    FROM orderbook 
                    WHERE symbol = :symbol 
                    AND bid1_price IS NOT NULL 
                    AND ask1_price IS NOT NULL
                    ORDER BY timestamp
                    LIMIT 50000  -- Limitamos para no sobrecargar
                )
                SELECT 
                    COUNT(*) as total_intervals,
                    AVG(minutes_diff) as avg_interval_minutes,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY minutes_diff) as median_interval,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY minutes_diff) as p95_interval,
                    MIN(minutes_diff) as min_interval,
                    MAX(minutes_diff) as max_interval,
                    COUNT(CASE WHEN minutes_diff <= 1.1 THEN 1 END) as regular_intervals,
                    COUNT(CASE WHEN minutes_diff > 1.1 AND minutes_diff <= 5 THEN 1 END) as minor_gaps,
                    COUNT(CASE WHEN minutes_diff > 5 AND minutes_diff <= 60 THEN 1 END) as medium_gaps,
                    COUNT(CASE WHEN minutes_diff > 60 THEN 1 END) as major_gaps
                FROM time_diffs 
                WHERE prev_timestamp IS NOT NULL
            """)
            
            intervals_result = session.execute(intervals_query, {'symbol': symbol}).fetchone()
            
            if not intervals_result.total_intervals:
                return {'error': 'No interval data found'}
            
            # Análisis por horas del día
            hourly_pattern_query = text("""
                SELECT 
                    EXTRACT(HOUR FROM timestamp) as hour,
                    COUNT(*) as records_count,
                    AVG(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL 
                        THEN (ask1_price - bid1_price) / bid1_price * 100 END) as avg_spread_pct
                FROM orderbook 
                WHERE symbol = :symbol
                GROUP BY EXTRACT(HOUR FROM timestamp)
                ORDER BY hour
            """)
            
            hourly_df = pd.read_sql(hourly_pattern_query, session.bind, params={'symbol': symbol})
            
            # Análisis por día de la semana
            weekday_pattern_query = text("""
                SELECT 
                    EXTRACT(DOW FROM timestamp) as weekday,
                    COUNT(*) as records_count,
                    COUNT(DISTINCT DATE(timestamp)) as unique_days
                FROM orderbook 
                WHERE symbol = :symbol
                GROUP BY EXTRACT(DOW FROM timestamp)
                ORDER BY weekday
            """)
            
            weekday_df = pd.read_sql(weekday_pattern_query, session.bind, params={'symbol': symbol})
            weekday_df['avg_records_per_day'] = weekday_df['records_count'] / weekday_df['unique_days']
            
            # Calcular regularidad
            regular_pct = (intervals_result.regular_intervals / intervals_result.total_intervals) * 100
            
            return {
                'interval_stats': {
                    'total_intervals_analyzed': intervals_result.total_intervals,
                    'avg_interval_minutes': intervals_result.avg_interval_minutes,
                    'median_interval_minutes': intervals_result.median_interval,
                    'p95_interval_minutes': intervals_result.p95_interval,
                    'min_interval_minutes': intervals_result.min_interval,
                    'max_interval_minutes': intervals_result.max_interval,
                    'regularity_percentage': regular_pct
                },
                'gap_analysis': {
                    'regular_intervals': intervals_result.regular_intervals,
                    'minor_gaps': intervals_result.minor_gaps,
                    'medium_gaps': intervals_result.medium_gaps,
                    'major_gaps': intervals_result.major_gaps
                },
                'hourly_patterns': hourly_df.to_dict('records'),
                'weekday_patterns': weekday_df.to_dict('records'),
                'consistency_score': min(100, regular_pct)
            }
    
    def _analyze_data_quality(self, symbol: str) -> Dict:
        """Análisis de calidad de datos"""
        log.info(f"  🔍 Analizando calidad de datos...")
        
        with db_manager.get_session() as session:
            # Análisis de campos nulos por nivel
            null_analysis_query = text("""
                SELECT 
                    'level_1' as level,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN bid1_price IS NULL THEN 1 END) as null_bid_price,
                    COUNT(CASE WHEN bid1_size IS NULL THEN 1 END) as null_bid_size,
                    COUNT(CASE WHEN ask1_price IS NULL THEN 1 END) as null_ask_price,
                    COUNT(CASE WHEN ask1_size IS NULL THEN 1 END) as null_ask_size
                FROM orderbook WHERE symbol = :symbol
                
                UNION ALL
                
                SELECT 
                    'level_2' as level,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN bid2_price IS NULL THEN 1 END) as null_bid_price,
                    COUNT(CASE WHEN bid2_size IS NULL THEN 1 END) as null_bid_size,
                    COUNT(CASE WHEN ask2_price IS NULL THEN 1 END) as null_ask_price,
                    COUNT(CASE WHEN ask2_size IS NULL THEN 1 END) as null_ask_size
                FROM orderbook WHERE symbol = :symbol
                
                UNION ALL
                
                SELECT 
                    'level_5' as level,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN bid5_price IS NULL THEN 1 END) as null_bid_price,
                    COUNT(CASE WHEN bid5_size IS NULL THEN 1 END) as null_bid_size,
                    COUNT(CASE WHEN ask5_price IS NULL THEN 1 END) as null_ask_price,
                    COUNT(CASE WHEN ask5_size IS NULL THEN 1 END) as null_ask_size
                FROM orderbook WHERE symbol = :symbol
            """)
            
            null_df = pd.read_sql(null_analysis_query, session.bind, params={'symbol': symbol})
            
            # Análisis de spreads válidos vs inválidos
            spread_validity_query = text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL 
                          AND bid1_price > 0 AND ask1_price > 0 
                          AND bid1_price < ask1_price THEN 1 END) as valid_spreads,
                    COUNT(CASE WHEN bid1_price >= ask1_price THEN 1 END) as crossed_spreads,
                    COUNT(CASE WHEN bid1_price <= 0 OR ask1_price <= 0 THEN 1 END) as invalid_prices,
                    AVG(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL 
                        AND bid1_price > 0 AND ask1_price > 0 AND bid1_price < ask1_price
                        THEN (ask1_price - bid1_price) / bid1_price * 100 END) as avg_valid_spread_pct
                FROM orderbook 
                WHERE symbol = :symbol
            """)
            
            spread_result = session.execute(spread_validity_query, {'symbol': symbol}).fetchone()
            
            # Análisis de profundidad promedio
            depth_analysis_query = text("""
                SELECT 
                    AVG(CASE WHEN bid1_price IS NOT NULL AND bid1_size IS NOT NULL AND bid1_size > 0 THEN 1 ELSE 0 END +
                        CASE WHEN bid2_price IS NOT NULL AND bid2_size IS NOT NULL AND bid2_size > 0 THEN 1 ELSE 0 END +
                        CASE WHEN bid3_price IS NOT NULL AND bid3_size IS NOT NULL AND bid3_size > 0 THEN 1 ELSE 0 END +
                        CASE WHEN bid4_price IS NOT NULL AND bid4_size IS NOT NULL AND bid4_size > 0 THEN 1 ELSE 0 END +
                        CASE WHEN bid5_price IS NOT NULL AND bid5_size IS NOT NULL AND bid5_size > 0 THEN 1 ELSE 0 END) as avg_bid_levels,
                    AVG(CASE WHEN ask1_price IS NOT NULL AND ask1_size IS NOT NULL AND ask1_size > 0 THEN 1 ELSE 0 END +
                        CASE WHEN ask2_price IS NOT NULL AND ask2_size IS NOT NULL AND ask2_size > 0 THEN 1 ELSE 0 END +
                        CASE WHEN ask3_price IS NOT NULL AND ask3_size IS NOT NULL AND ask3_size > 0 THEN 1 ELSE 0 END +
                        CASE WHEN ask4_price IS NOT NULL AND ask4_size IS NOT NULL AND ask4_size > 0 THEN 1 ELSE 0 END +
                        CASE WHEN ask5_price IS NOT NULL AND ask5_size IS NOT NULL AND ask5_size > 0 THEN 1 ELSE 0 END) as avg_ask_levels,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 
                        CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL 
                        AND bid1_price > 0 AND ask1_price > 0 AND bid1_price < ask1_price
                        THEN (ask1_price - bid1_price) / bid1_price * 100 END) as median_spread_pct
                FROM orderbook 
                WHERE symbol = :symbol
            """)
            
            depth_result = session.execute(depth_analysis_query, {'symbol': symbol}).fetchone()
            
            # Calcular scores de calidad
            valid_spread_pct = (spread_result.valid_spreads / spread_result.total_records) * 100
            quality_score = min(100, valid_spread_pct)
            
            return {
                'null_analysis': null_df.to_dict('records'),
                'spread_validity': {
                    'total_records': spread_result.total_records,
                    'valid_spreads': spread_result.valid_spreads,
                    'crossed_spreads': spread_result.crossed_spreads,
                    'invalid_prices': spread_result.invalid_prices,
                    'valid_spread_percentage': valid_spread_pct,
                    'avg_valid_spread_pct': spread_result.avg_valid_spread_pct,
                    'median_spread_pct': depth_result.median_spread_pct
                },
                'depth_stats': {
                    'avg_bid_levels': depth_result.avg_bid_levels,
                    'avg_ask_levels': depth_result.avg_ask_levels,
                    'avg_total_levels': (depth_result.avg_bid_levels + depth_result.avg_ask_levels) / 2
                },
                'quality_score': quality_score
            }
    
    def _analyze_liquidity_levels(self, symbol: str) -> Dict:
        """Análisis detallado de niveles de liquidez"""
        log.info(f"  💧 Analizando niveles de liquidez...")
        
        with db_manager.get_session() as session:
            # Análisis de volúmenes por nivel
            liquidity_query = text("""
                SELECT 
                    AVG(bid1_size * bid1_price) as avg_bid1_value,
                    AVG(bid2_size * bid2_price) as avg_bid2_value,
                    AVG(bid3_size * bid3_price) as avg_bid3_value,
                    AVG(ask1_size * ask1_price) as avg_ask1_value,
                    AVG(ask2_size * ask2_price) as avg_ask2_value,
                    AVG(ask3_size * ask3_price) as avg_ask3_value,
                    AVG((bid1_size * bid1_price) + (bid2_size * bid2_price) + (bid3_size * bid3_price) +
                        (ask1_size * ask1_price) + (ask2_size * ask2_price) + (ask3_size * ask3_price)) as avg_total_liquidity,
                    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY 
                        (bid1_size * bid1_price) + (ask1_size * ask1_price)) as p10_level1_liquidity,
                    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY 
                        (bid1_size * bid1_price) + (ask1_size * ask1_price)) as p90_level1_liquidity
                FROM orderbook 
                WHERE symbol = :symbol
                AND bid1_price IS NOT NULL AND ask1_price IS NOT NULL
                AND bid1_size IS NOT NULL AND ask1_size IS NOT NULL
                AND bid1_price > 0 AND ask1_price > 0
                AND bid1_size > 0 AND ask1_size > 0
            """)
            
            liquidity_result = session.execute(liquidity_query, {'symbol': symbol}).fetchone()
            
            # Distribución de profundidad por snapshot
            depth_distribution_query = text("""
                SELECT 
                    depth_levels,
                    COUNT(*) as snapshots_count,
                    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orderbook WHERE symbol = :symbol) as percentage
                FROM (
                    SELECT 
                        (CASE WHEN bid1_size > 0 AND ask1_size > 0 THEN 1 ELSE 0 END +
                         CASE WHEN bid2_size > 0 AND ask2_size > 0 THEN 1 ELSE 0 END +
                         CASE WHEN bid3_size > 0 AND ask3_size > 0 THEN 1 ELSE 0 END +
                         CASE WHEN bid4_size > 0 AND ask4_size > 0 THEN 1 ELSE 0 END +
                         CASE WHEN bid5_size > 0 AND ask5_size > 0 THEN 1 ELSE 0 END) as depth_levels
                    FROM orderbook 
                    WHERE symbol = :symbol
                ) depth_calc
                GROUP BY depth_levels
                ORDER BY depth_levels
            """)
            
            depth_dist_df = pd.read_sql(depth_distribution_query, session.bind, params={'symbol': symbol})
            
            return {
                'avg_liquidity_by_level': {
                    'bid1_value_usd': liquidity_result.avg_bid1_value or 0,
                    'bid2_value_usd': liquidity_result.avg_bid2_value or 0,
                    'bid3_value_usd': liquidity_result.avg_bid3_value or 0,
                    'ask1_value_usd': liquidity_result.avg_ask1_value or 0,
                    'ask2_value_usd': liquidity_result.avg_ask2_value or 0,
                    'ask3_value_usd': liquidity_result.avg_ask3_value or 0,
                    'total_top3_liquidity': liquidity_result.avg_total_liquidity or 0
                },
                'level1_liquidity_range': {
                    'p10': liquidity_result.p10_level1_liquidity or 0,
                    'p90': liquidity_result.p90_level1_liquidity or 0
                },
                'depth_distribution': depth_dist_df.to_dict('records')
            }
    
    def _analyze_spread_distributions(self, symbol: str) -> Dict:
        """Análisis de distribuciones de spreads"""
        log.info(f"  📈 Analizando distribuciones de spreads...")
        
        with db_manager.get_session() as session:
            spread_stats_query = text("""
                WITH spread_calc AS (
                    SELECT 
                        timestamp,
                        (ask1_price - bid1_price) / bid1_price * 100 as spread_pct
                    FROM orderbook 
                    WHERE symbol = :symbol
                    AND bid1_price IS NOT NULL AND ask1_price IS NOT NULL
                    AND bid1_price > 0 AND ask1_price > 0
                    AND bid1_price < ask1_price
                    LIMIT 10000  -- Limitamos para performance
                )
                SELECT 
                    COUNT(*) as total_spreads,
                    AVG(spread_pct) as avg_spread,
                    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY spread_pct) as p10_spread,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY spread_pct) as p25_spread,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY spread_pct) as median_spread,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY spread_pct) as p75_spread,
                    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY spread_pct) as p90_spread,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY spread_pct) as p95_spread,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY spread_pct) as p99_spread,
                    MIN(spread_pct) as min_spread,
                    MAX(spread_pct) as max_spread,
                    STDDEV(spread_pct) as std_spread
                FROM spread_calc
            """)
            
            spread_result = session.execute(spread_stats_query, {'symbol': symbol}).fetchone()
            
            # Clasificación de spreads
            spread_classification_query = text("""
                WITH spread_calc AS (
                    SELECT (ask1_price - bid1_price) / bid1_price * 100 as spread_pct
                    FROM orderbook 
                    WHERE symbol = :symbol
                    AND bid1_price IS NOT NULL AND ask1_price IS NOT NULL
                    AND bid1_price > 0 AND ask1_price > 0
                    AND bid1_price < ask1_price
                )
                SELECT 
                    COUNT(CASE WHEN spread_pct <= 0.1 THEN 1 END) as tight_spreads,
                    COUNT(CASE WHEN spread_pct > 0.1 AND spread_pct <= 0.5 THEN 1 END) as normal_spreads,
                    COUNT(CASE WHEN spread_pct > 0.5 AND spread_pct <= 1.0 THEN 1 END) as wide_spreads,
                    COUNT(CASE WHEN spread_pct > 1.0 THEN 1 END) as very_wide_spreads,
                    COUNT(*) as total_spreads
                FROM spread_calc
            """)
            
            classification_result = session.execute(spread_classification_query, {'symbol': symbol}).fetchone()
            
            if not spread_result.total_spreads:
                return {'error': 'No valid spreads found'}
            
            return {
                'spread_statistics': {
                    'count': spread_result.total_spreads,
                    'mean': spread_result.avg_spread,
                    'median': spread_result.median_spread,
                    'std': spread_result.std_spread,
                    'min': spread_result.min_spread,
                    'max': spread_result.max_spread,
                    'percentiles': {
                        'p10': spread_result.p10_spread,
                        'p25': spread_result.p25_spread,
                        'p75': spread_result.p75_spread,
                        'p90': spread_result.p90_spread,
                        'p95': spread_result.p95_spread,
                        'p99': spread_result.p99_spread
                    }
                },
                'spread_classification': {
                    'tight_spreads_pct': (classification_result.tight_spreads / classification_result.total_spreads) * 100,
                    'normal_spreads_pct': (classification_result.normal_spreads / classification_result.total_spreads) * 100,
                    'wide_spreads_pct': (classification_result.wide_spreads / classification_result.total_spreads) * 100,
                    'very_wide_spreads_pct': (classification_result.very_wide_spreads / classification_result.total_spreads) * 100
                }
            }
    
    def _detect_anomalies(self, symbol: str) -> Dict:
        """Detección de anomalías en los datos"""
        log.info(f"  🚨 Detectando anomalías...")
        
        with db_manager.get_session() as session:
            # Anomalías de precios
            price_anomalies_query = text("""
                WITH price_stats AS (
                    SELECT 
                        timestamp,
                        bid1_price,
                        ask1_price,
                        (ask1_price - bid1_price) / bid1_price * 100 as spread_pct,
                        LAG(bid1_price) OVER (ORDER BY timestamp) as prev_bid,
                        LAG(ask1_price) OVER (ORDER BY timestamp) as prev_ask
                    FROM orderbook 
                    WHERE symbol = :symbol
                    AND bid1_price IS NOT NULL AND ask1_price IS NOT NULL
                    AND bid1_price > 0 AND ask1_price > 0
                    ORDER BY timestamp
                    LIMIT 20000
                )
                SELECT 
                    COUNT(CASE WHEN spread_pct > 5.0 THEN 1 END) as extreme_spreads,
                    COUNT(CASE WHEN ABS(bid1_price - prev_bid) / prev_bid > 0.1 THEN 1 END) as price_jumps,
                    COUNT(CASE WHEN bid1_price >= ask1_price THEN 1 END) as crossed_quotes,
                    COUNT(*) as total_analyzed
                FROM price_stats
                WHERE prev_bid IS NOT NULL
            """)
            
            anomaly_result = session.execute(price_anomalies_query, {'symbol': symbol}).fetchone()
            
            # Anomalías de volumen
            volume_anomalies_query = text("""
                WITH volume_stats AS (
                    SELECT 
                        bid1_size, ask1_size,
                        AVG(bid1_size) OVER (ROWS BETWEEN 100 PRECEDING AND CURRENT ROW) as avg_bid_size,
                        AVG(ask1_size) OVER (ROWS BETWEEN 100 PRECEDING AND CURRENT ROW) as avg_ask_size
                    FROM orderbook 
                    WHERE symbol = :symbol
                    AND bid1_size IS NOT NULL AND ask1_size IS NOT NULL
                    AND bid1_size > 0 AND ask1_size > 0
                    ORDER BY timestamp
                    LIMIT 10000
                )
                SELECT 
                    COUNT(CASE WHEN bid1_size > avg_bid_size * 10 THEN 1 END) as extreme_bid_volumes,
                    COUNT(CASE WHEN ask1_size > avg_ask_size * 10 THEN 1 END) as extreme_ask_volumes,
                    COUNT(*) as total_analyzed
                FROM volume_stats
                WHERE avg_bid_size > 0 AND avg_ask_size > 0
            """)
            
            volume_result = session.execute(volume_anomalies_query, {'symbol': symbol}).fetchone()
            
            # Calcular tasa de anomalías
            total_anomalies = (anomaly_result.extreme_spreads + anomaly_result.price_jumps + 
                             anomaly_result.crossed_quotes + volume_result.extreme_bid_volumes + 
                             volume_result.extreme_ask_volumes)
            
            anomaly_rate = (total_anomalies / (anomaly_result.total_analyzed + volume_result.total_analyzed)) * 100
            
            return {
                'price_anomalies': {
                    'extreme_spreads': anomaly_result.extreme_spreads,
                    'price_jumps': anomaly_result.price_jumps,
                    'crossed_quotes': anomaly_result.crossed_quotes
                },
                'volume_anomalies': {
                    'extreme_bid_volumes': volume_result.extreme_bid_volumes,
                    'extreme_ask_volumes': volume_result.extreme_ask_volumes
                },
                'total_anomalies': total_anomalies,
                'anomaly_rate_pct': anomaly_rate,
                'data_health': 'excellent' if anomaly_rate < 1 else 'good' if anomaly_rate < 5 else 'poor'
            }
    
    def _find_data_gaps(self, daily_df: pd.DataFrame) -> List[Dict]:
        """Encuentra gaps de datos consecutivos"""
        missing_days = daily_df[daily_df['records_count'] == 0]
        
        if missing_days.empty:
            return []
        
        gaps = []
        gap_start = None
        
        for idx, row in missing_days.iterrows():
            if gap_start is None:
                gap_start = row['day']
                gap_end = row['day']
            elif (row['day'] - gap_end).days == 1:
                gap_end = row['day']
            else:
                # Gap terminado, agregar a la lista
                gaps.append({
                    'start_date': gap_start.strftime('%Y-%m-%d'),
                    'end_date': gap_end.strftime('%Y-%m-%d'),
                    'duration_days': (gap_end - gap_start).days + 1
                })
                gap_start = row['day']
                gap_end = row['day']
        
        # Agregar último gap
        if gap_start is not None:
            gaps.append({
                'start_date': gap_start.strftime('%Y-%m-%d'),
                'end_date': gap_end.strftime('%Y-%m-%d'),
                'duration_days': (gap_end - gap_start).days + 1
            })
        
        return gaps
    
    def _calculate_completeness_score(self, results: Dict) -> float:
        """Calcula un score general de completitud (0-100)"""
        try:
            score = 0
            
            # Cobertura temporal (40% del score)
            if 'temporal_analysis' in results:
                temporal = results['temporal_analysis']
                if 'coverage_percentage' in temporal:
                    score += temporal['coverage_percentage'] * 0.4
            
            # Calidad de datos (30% del score)
            if 'quality_analysis' in results:
                quality = results['quality_analysis']
                if 'quality_score' in quality:
                    score += quality['quality_score'] * 0.3
            
            # Consistencia de frecuencia (20% del score)
            if 'frequency_analysis' in results:
                frequency = results['frequency_analysis']
                if 'consistency_score' in frequency:
                    score += frequency['consistency_score'] * 0.2
            
            # Ausencia de anomalías (10% del score)
            if 'anomaly_analysis' in results:
                anomaly = results['anomaly_analysis']
                if 'anomaly_rate_pct' in anomaly:
                    anomaly_score = max(0, 100 - anomaly['anomaly_rate_pct'] * 10)
                    score += anomaly_score * 0.1
            
            return min(100, max(0, score))
            
        except Exception as e:
            log.error(f"Error calculating completeness score: {e}")
            return 0
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        try:
            # Recomendaciones basadas en cobertura temporal
            if 'temporal_analysis' in results:
                temporal = results['temporal_analysis']
                if temporal.get('coverage_percentage', 0) < 90:
                    recommendations.append("⚠️ Cobertura temporal baja - verificar ingesta para días faltantes")
                
                if len(temporal.get('gaps_found', [])) > 0:
                    long_gaps = [g for g in temporal['gaps_found'] if g['duration_days'] > 3]
                    if long_gaps:
                        recommendations.append(f"🚨 Encontrados {len(long_gaps)} gaps largos (>3 días) - prioridad alta")
            
            # Recomendaciones basadas en calidad
            if 'quality_analysis' in results:
                quality = results['quality_analysis']
                spread_validity = quality.get('spread_validity', {})
                
                if spread_validity.get('valid_spread_percentage', 0) < 95:
                    recommendations.append("⚠️ Alto porcentaje de spreads inválidos - revisar datos fuente")
                
                if spread_validity.get('crossed_spreads', 0) > 0:
                    recommendations.append("🔴 Spreads cruzados detectados - requiere limpieza")
            
            # Recomendaciones basadas en frecuencia
            if 'frequency_analysis' in results:
                frequency = results['frequency_analysis']
                if frequency.get('consistency_score', 0) < 80:
                    recommendations.append("📊 Baja consistencia en frecuencia - verificar procesos de ingesta")
            
            # Recomendaciones basadas en anomalías
            if 'anomaly_analysis' in results:
                anomaly = results['anomaly_analysis']
                if anomaly.get('anomaly_rate_pct', 0) > 5:
                    recommendations.append("🚨 Alta tasa de anomalías - implementar filtros de calidad")
            
            # Recomendaciones generales
            completeness = results.get('completeness_score', 0)
            if completeness >= 90:
                recommendations.append("✅ Datos en excelente estado - listos para trading")
            elif completeness >= 75:
                recommendations.append("✅ Datos en buen estado - minor limpieza recomendada")
            elif completeness >= 50:
                recommendations.append("⚠️ Datos requieren mejoras significativas")
            else:
                recommendations.append("🚨 Datos en estado crítico - requiere intervención inmediata")
            
            if not recommendations:
                recommendations.append("✅ No se detectaron problemas importantes")
            
            return recommendations
            
        except Exception as e:
            log.error(f"Error generating recommendations: {e}")
            return ["❌ Error generando recomendaciones"]
    
    def create_comprehensive_visualization(self, symbol: str, results: Dict):
        """Crear visualización comprehensiva de todos los análisis"""
        if 'error' in results:
            log.warning(f"Skipping visualization for {symbol} due to errors")
            return
        
        log.info(f"📊 Creando visualización comprehensiva para {symbol}")
        
        symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
        
        # Crear figura grande con múltiples subplots
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        fig.suptitle(f'{symbol_short} - Análisis Profundo de Orderbook', fontsize=20, fontweight='bold')
        
        try:
            # 1. Cobertura temporal por día (row 0, cols 0-1)
            if 'temporal_analysis' in results and 'daily_data' in results['temporal_analysis']:
                ax1 = fig.add_subplot(gs[0, :2])
                self._plot_temporal_coverage(ax1, results['temporal_analysis'], symbol_short)
            
            # 2. Distribución de calidad de días (row 0, col 2)
            if 'temporal_analysis' in results:
                ax2 = fig.add_subplot(gs[0, 2])
                self._plot_quality_distribution(ax2, results['temporal_analysis'], symbol_short)
            
            # 3. Score de completitud (row 0, col 3)
            ax3 = fig.add_subplot(gs[0, 3])
            self._plot_completeness_score(ax3, results, symbol_short)
            
            # 4. Patrones de frecuencia por hora (row 1, col 0)
            if 'frequency_analysis' in results:
                ax4 = fig.add_subplot(gs[1, 0])
                self._plot_hourly_patterns(ax4, results['frequency_analysis'], symbol_short)
            
            # 5. Distribución de spreads (row 1, col 1)
            if 'spread_analysis' in results:
                ax5 = fig.add_subplot(gs[1, 1])
                self._plot_spread_distribution(ax5, results['spread_analysis'], symbol_short)
            
            # 6. Niveles de liquidez (row 1, col 2)
            if 'liquidity_analysis' in results:
                ax6 = fig.add_subplot(gs[1, 2])
                self._plot_liquidity_levels(ax6, results['liquidity_analysis'], symbol_short)
            
            # 7. Profundidad del orderbook (row 1, col 3)
            if 'liquidity_analysis' in results:
                ax7 = fig.add_subplot(gs[1, 3])
                self._plot_depth_distribution(ax7, results['liquidity_analysis'], symbol_short)
            
            # 8. Análisis de gaps (row 2, cols 0-1)
            if 'temporal_analysis' in results:
                ax8 = fig.add_subplot(gs[2, :2])
                self._plot_gaps_analysis(ax8, results['temporal_analysis'], symbol_short)
            
            # 9. Anomalías detectadas (row 2, col 2)
            if 'anomaly_analysis' in results:
                ax9 = fig.add_subplot(gs[2, 2])
                self._plot_anomalies(ax9, results['anomaly_analysis'], symbol_short)
            
            # 10. Patrones semanales (row 2, col 3)
            if 'frequency_analysis' in results:
                ax10 = fig.add_subplot(gs[2, 3])
                self._plot_weekday_patterns(ax10, results['frequency_analysis'], symbol_short)
            
            # 11. Resumen de estadísticas (row 3, all cols)
            ax11 = fig.add_subplot(gs[3, :])
            self._plot_statistics_summary(ax11, results, symbol_short)
            
            # Guardar visualización
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            
            filename = f'{symbol_short}_orderbook_deep_analysis.png'
            filepath = plots_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            log.info(f"✅ Visualización guardada: {filepath}")
            
        except Exception as e:
            log.error(f"Error creating visualization: {e}")
            plt.close()
    
    def _plot_temporal_coverage(self, ax, temporal_data, symbol_short):
        """Plot temporal coverage"""
        if 'daily_data' not in temporal_data:
            ax.text(0.5, 0.5, 'No daily data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        daily_df = temporal_data['daily_data']
        
        # Crear colores basados en calidad
        colors = daily_df['day_quality'].map({
            'excellent': 'green',
            'good': 'lightgreen', 
            'fair': 'yellow',
            'poor': 'orange',
            'missing': 'red'
        })
        
        ax.scatter(daily_df['day'], daily_df['records_count'], c=colors, alpha=0.7, s=20)
        ax.set_ylabel('Records per Day')
        ax.set_title(f'{symbol_short} - Temporal Coverage')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add coverage percentage as text
        coverage = temporal_data.get('coverage_percentage', 0)
        ax.text(0.02, 0.98, f'Coverage: {coverage:.1f}%', transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), verticalalignment='top')
    
    def _plot_quality_distribution(self, ax, temporal_data, symbol_short):
        """Plot quality distribution pie chart"""
        if 'quality_breakdown' not in temporal_data:
            ax.text(0.5, 0.5, 'No quality data', ha='center', va='center', transform=ax.transAxes)
            return
        
        quality_data = temporal_data['quality_breakdown']
        
        labels = list(quality_data.keys())
        sizes = list(quality_data.values())
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red'][:len(labels)]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{symbol_short} - Day Quality')
    
    def _plot_completeness_score(self, ax, results, symbol_short):
        """Plot completeness score gauge"""
        score = results.get('completeness_score', 0)
        
        # Simple bar representation of score
        ax.barh([0], [score], color='green' if score >= 80 else 'orange' if score >= 60 else 'red')
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Completeness Score')
        ax.set_title(f'{symbol_short} - Overall Score')
        ax.set_yticks([])
        
        # Add score text
        ax.text(score/2, 0, f'{score:.1f}%', ha='center', va='center', fontweight='bold', color='white')
    
    def _plot_hourly_patterns(self, ax, frequency_data, symbol_short):
        """Plot hourly patterns"""
        if 'hourly_patterns' not in frequency_data:
            ax.text(0.5, 0.5, 'No hourly data', ha='center', va='center', transform=ax.transAxes)
            return
        
        hourly_df = pd.DataFrame(frequency_data['hourly_patterns'])
        
        ax.bar(hourly_df['hour'], hourly_df['records_count'], alpha=0.7, color='skyblue')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Record Count')
        ax.set_title(f'{symbol_short} - Hourly Activity')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 3))
    
    def _plot_spread_distribution(self, ax, spread_data, symbol_short):
        """Plot spread distribution"""
        if 'spread_statistics' not in spread_data:
            ax.text(0.5, 0.5, 'No spread data', ha='center', va='center', transform=ax.transAxes)
            return
        
        stats = spread_data['spread_statistics']
        percentiles = stats['percentiles']
        
        # Box plot representation
        bp_data = [
            percentiles['p10'], percentiles['p25'], stats['median'], 
            percentiles['p75'], percentiles['p90']
        ]
        
        ax.boxplot([bp_data], labels=[symbol_short])
        ax.set_ylabel('Spread %')
        ax.set_title(f'{symbol_short} - Spread Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_liquidity_levels(self, ax, liquidity_data, symbol_short):
        """Plot liquidity levels"""
        if 'avg_liquidity_by_level' not in liquidity_data:
            ax.text(0.5, 0.5, 'No liquidity data', ha='center', va='center', transform=ax.transAxes)
            return
        
        levels_data = liquidity_data['avg_liquidity_by_level']
        
        bid_levels = [levels_data.get(f'bid{i}_value_usd', 0) for i in range(1, 4)]
        ask_levels = [levels_data.get(f'ask{i}_value_usd', 0) for i in range(1, 4)]
        
        x = np.arange(3)
        width = 0.35
        
        ax.bar(x - width/2, bid_levels, width, label='Bid', color='lightblue')
        ax.bar(x + width/2, ask_levels, width, label='Ask', color='lightcoral')
        
        ax.set_xlabel('Level')
        ax.set_ylabel('Avg Liquidity (USD)')
        ax.set_title(f'{symbol_short} - Liquidity by Level')
        ax.set_xticks(x)
        ax.set_xticklabels(['L1', 'L2', 'L3'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_depth_distribution(self, ax, liquidity_data, symbol_short):
        """Plot depth distribution"""
        if 'depth_distribution' not in liquidity_data:
            ax.text(0.5, 0.5, 'No depth data', ha='center', va='center', transform=ax.transAxes)
            return
        
        depth_df = pd.DataFrame(liquidity_data['depth_distribution'])
        
        ax.bar(depth_df['depth_levels'], depth_df['percentage'], alpha=0.7, color='lightgreen')
        ax.set_xlabel('Depth Levels Available')
        ax.set_ylabel('Percentage of Snapshots')
        ax.set_title(f'{symbol_short} - Depth Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_gaps_analysis(self, ax, temporal_data, symbol_short):
        """Plot gaps analysis"""
        gaps = temporal_data.get('gaps_found', [])
        
        if not gaps:
            ax.text(0.5, 0.5, '✅ No significant gaps found', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='green')
            ax.set_title(f'{symbol_short} - Data Gaps')
            return
        
        # Plot gaps as horizontal bars
        gap_starts = [pd.to_datetime(gap['start_date']) for gap in gaps]
        gap_durations = [gap['duration_days'] for gap in gaps]
        
        colors = ['red' if d > 7 else 'orange' if d > 3 else 'yellow' for d in gap_durations]
        
        y_pos = np.arange(len(gaps))
        bars = ax.barh(y_pos, gap_durations, color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([gap['start_date'] for gap in gaps])
        ax.set_xlabel('Gap Duration (days)')
        ax.set_title(f'{symbol_short} - Data Gaps Found')
        ax.grid(True, alpha=0.3)
        
        # Add duration labels
        for i, (bar, duration) in enumerate(zip(bars, gap_durations)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{duration}d', ha='left', va='center')
    
    def _plot_anomalies(self, ax, anomaly_data, symbol_short):
        """Plot anomalies summary"""
        anomaly_types = ['Extreme Spreads', 'Price Jumps', 'Crossed Quotes', 'Volume Spikes']
        
        price_anomalies = anomaly_data.get('price_anomalies', {})
        volume_anomalies = anomaly_data.get('volume_anomalies', {})
        
        counts = [
            price_anomalies.get('extreme_spreads', 0),
            price_anomalies.get('price_jumps', 0),
            price_anomalies.get('crossed_quotes', 0),
            volume_anomalies.get('extreme_bid_volumes', 0) + volume_anomalies.get('extreme_ask_volumes', 0)
        ]
        
        colors = ['red' if c > 100 else 'orange' if c > 10 else 'green' for c in counts]
        
        bars = ax.bar(range(len(anomaly_types)), counts, color=colors, alpha=0.7)
        ax.set_xticks(range(len(anomaly_types)))
        ax.set_xticklabels(anomaly_types, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(f'{symbol_short} - Anomalies Detected')
        ax.grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                       str(count), ha='center', va='bottom')
    
    def _plot_weekday_patterns(self, ax, frequency_data, symbol_short):
        """Plot weekday patterns"""
        if 'weekday_patterns' not in frequency_data:
            ax.text(0.5, 0.5, 'No weekday data', ha='center', va='center', transform=ax.transAxes)
            return
        
        weekday_df = pd.DataFrame(frequency_data['weekday_patterns'])
        weekday_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        
        ax.bar(weekday_df['weekday'], weekday_df['avg_records_per_day'], 
               color='lightsteelblue', alpha=0.7)
        ax.set_xticks(weekday_df['weekday'])
        ax.set_xticklabels([weekday_names[int(w)] for w in weekday_df['weekday']])
        ax.set_ylabel('Avg Records per Day')
        ax.set_title(f'{symbol_short} - Weekly Pattern')
        ax.grid(True, alpha=0.3)
    
    def _plot_statistics_summary(self, ax, results, symbol_short):
        """Plot statistics summary table"""
        ax.axis('off')
        
        # Recopilar estadísticas clave
        stats_data = []
        
        # Basic stats
        if 'basic_stats' in results:
            basic = results['basic_stats']
            stats_data.extend([
                ['Total Records', f"{basic.get('total_records', 0):,}"],
                ['Data Coverage', f"{basic.get('data_coverage_pct', 0):.1f}%"],
                ['Valid Quotes', f"{basic.get('valid_quotes_pct', 0):.1f}%"],
                ['Avg Records/Day', f"{basic.get('records_per_day_avg', 0):.0f}"]
            ])
        
        # Quality stats
        if 'quality_analysis' in results:
            quality = results['quality_analysis']
            spread_validity = quality.get('spread_validity', {})
            stats_data.extend([
                ['Avg Spread', f"{spread_validity.get('avg_valid_spread_pct', 0):.4f}%"],
                ['Crossed Spreads', f"{spread_validity.get('crossed_spreads', 0):,}"]
            ])
        
        # Completeness score
        stats_data.append(['Completeness Score', f"{results.get('completeness_score', 0):.1f}%"])
        
        # Anomalies
        if 'anomaly_analysis' in results:
            anomaly = results['anomaly_analysis']
            stats_data.extend([
                ['Anomaly Rate', f"{anomaly.get('anomaly_rate_pct', 0):.2f}%"],
                ['Data Health', anomaly.get('data_health', 'unknown').title()]
            ])
        
        # Create table
        table = ax.table(
            cellText=stats_data,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style table
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j == 0:  # Metric names
                        cell.set_facecolor('#E8F5E8')
                    else:  # Values
                        cell.set_facecolor('#F8F8F8')
        
        ax.set_title(f'{symbol_short} - Key Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    
    def export_detailed_report(self, symbol: str, results: Dict):
        """Exportar reporte detallado a CSV y texto"""
        if 'error' in results:
            return
        
        log.info(f"📄 Exportando reporte detallado para {symbol}")
        
        # Crear directorio de reportes
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Reporte en texto
        txt_file = reports_dir / f"{symbol_short}_orderbook_analysis_{timestamp}.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"ORDERBOOK DEEP ANALYSIS REPORT\n")
            f.write(f"{'='*60}\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Completeness Score: {results.get('completeness_score', 0):.2f}%\n\n")
            
            # Basic Statistics
            if 'basic_stats' in results:
                f.write("BASIC STATISTICS\n")
                f.write("-" * 20 + "\n")
                basic = results['basic_stats']
                for key, value in basic.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.2f}\n")
                    elif isinstance(value, datetime):
                        f.write(f"{key}: {value.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Recommendations
            if 'recommendations' in results:
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 20 + "\n")
                for rec in results['recommendations']:
                    f.write(f"• {rec}\n")
                f.write("\n")
            
            # Temporal Analysis
            if 'temporal_analysis' in results:
                f.write("TEMPORAL ANALYSIS\n")
                f.write("-" * 20 + "\n")
                temporal = results['temporal_analysis']
                f.write(f"Coverage: {temporal.get('coverage_percentage', 0):.2f}%\n")
                f.write(f"Days with data: {temporal.get('days_with_data', 0)}\n")
                f.write(f"Days missing: {temporal.get('days_missing', 0)}\n")
                
                gaps = temporal.get('gaps_found', [])
                if gaps:
                    f.write(f"Data gaps found: {len(gaps)}\n")
                    for gap in gaps:
                        f.write(f"  Gap: {gap['start_date']} to {gap['end_date']} ({gap['duration_days']} days)\n")
                f.write("\n")
            
            # Quality Analysis
            if 'quality_analysis' in results:
                f.write("QUALITY ANALYSIS\n")
                f.write("-" * 20 + "\n")
                quality = results['quality_analysis']
                spread_validity = quality.get('spread_validity', {})
                f.write(f"Valid spreads: {spread_validity.get('valid_spread_percentage', 0):.2f}%\n")
                f.write(f"Crossed spreads: {spread_validity.get('crossed_spreads', 0)}\n")
                f.write(f"Average spread: {spread_validity.get('avg_valid_spread_pct', 0):.4f}%\n")
                f.write("\n")
        
        log.info(f"✅ Reporte de texto guardado: {txt_file}")
        
        # 2. Datos diarios a CSV (si existen)
        if 'temporal_analysis' in results and 'daily_data' in results['temporal_analysis']:
            daily_df = results['temporal_analysis']['daily_data']
            csv_file = reports_dir / f"{symbol_short}_daily_data_{timestamp}.csv"
            daily_df.to_csv(csv_file, index=False)
            log.info(f"✅ Datos diarios guardados: {csv_file}")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="📊 Orderbook Deep Analysis - Análisis exhaustivo de datos")
    parser.add_argument("--symbol", type=str, help="Símbolo específico a analizar")
    parser.add_argument("--no-plots", action="store_true", help="Omitir generación de gráficos")
    parser.add_argument("--no-export", action="store_true", help="Omitir exportación de reportes")
    
    args = parser.parse_args()
    
    log.info("🔍 Iniciando análisis profundo de datos de orderbook")
    
    # Obtener símbolos a analizar
    if args.symbol:
        symbols = [args.symbol]
        log.info(f"🎯 Analizando símbolo específico: {args.symbol}")
    else:
        try:
            active_pairs = settings.get_active_pairs()
            symbols = []
            for pair in active_pairs:
                symbols.extend([pair.symbol1, pair.symbol2])
            symbols = list(set(symbols))
            log.info(f"🎯 Analizando todos los símbolos: {len(symbols)}")
        except Exception as e:
            log.error(f"Error cargando símbolos: {e}")
            symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
            log.info(f"🎯 Usando símbolos fallback: {symbols}")
    
    if not symbols:
        log.error("No hay símbolos para analizar")
        return False
    
    # Crear instancia del analizador
    analyzer = OrderbookDeepAnalysis()
    
    # Crear directorios necesarios
    Path("plots").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # Analizar cada símbolo
    all_results = {}
    successful_analyses = 0
    
    for i, symbol in enumerate(symbols):
        log.info(f"\n{'='*80}")
        log.info(f"ANALIZANDO SÍMBOLO {i+1}/{len(symbols)}: {symbol}")
        log.info(f"{'='*80}")
        
        try:
            # Realizar análisis completo
            results = analyzer.analyze_symbol(symbol)
            
            if 'error' not in results:
                all_results[symbol] = results
                successful_analyses += 1
                
                # Crear visualizaciones
                if not args.no_plots:
                    analyzer.create_comprehensive_visualization(symbol, results)
                
                # Exportar reportes
                if not args.no_export:
                    analyzer.export_detailed_report(symbol, results)
                
                # Log resumen del símbolo
                completeness = results.get('completeness_score', 0)
                log.info(f"✅ {symbol} completado - Score: {completeness:.1f}%")
                
                # Mostrar recomendaciones principales
                recommendations = results.get('recommendations', [])
                if recommendations:
                    log.info(f"📋 Recomendaciones principales:")
                    for rec in recommendations[:3]:  # Top 3
                        log.info(f"   {rec}")
            else:
                log.error(f"❌ {symbol} falló: {results['error']}")
                
        except Exception as e:
            log.error(f"💥 Error procesando {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
            continue
    
    # Resumen final
    log.info(f"\n{'='*80}")
    log.info(f"RESUMEN FINAL DEL ANÁLISIS")
    log.info(f"{'='*80}")
    log.info(f"✅ Símbolos analizados exitosamente: {successful_analyses}/{len(symbols)}")
    
    if successful_analyses > 0:
        # Estadísticas globales
        avg_completeness = np.mean([r.get('completeness_score', 0) for r in all_results.values()])
        log.info(f"📊 Score promedio de completitud: {avg_completeness:.1f}%")
        
        # Mejores y peores símbolos
        sorted_results = sorted(all_results.items(), 
                              key=lambda x: x[1].get('completeness_score', 0), 
                              reverse=True)
        
        if sorted_results:
            best_symbol, best_score = sorted_results[0][0], sorted_results[0][1].get('completeness_score', 0)
            worst_symbol, worst_score = sorted_results[-1][0], sorted_results[-1][1].get('completeness_score', 0)
            
            log.info(f"🏆 Mejor símbolo: {best_symbol.split('_')[-2]} ({best_score:.1f}%)")
            log.info(f"⚠️ Símbolo que requiere atención: {worst_symbol.split('_')[-2]} ({worst_score:.1f}%)")
        
        # Recomendaciones globales
        all_recommendations = []
        for results in all_results.values():
            all_recommendations.extend(results.get('recommendations', []))
        
        # Contar recomendaciones más frecuentes
        rec_counts = {}
        for rec in all_recommendations:
            key = rec.split('-')[0].strip()  # Tomar la parte principal
            rec_counts[key] = rec_counts.get(key, 0) + 1
        
        if rec_counts:
            log.info(f"\n📋 RECOMENDACIONES GLOBALES MÁS FRECUENTES:")
            for rec, count in sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                log.info(f"   {rec} (afecta {count} símbolos)")
        
        log.info(f"\n📁 ARCHIVOS GENERADOS:")
        if not args.no_plots:
            log.info(f"   📊 Gráficos: plots/*_orderbook_deep_analysis.png")
        if not args.no_export:
            log.info(f"   📄 Reportes: reports/*_orderbook_analysis_*.txt")
            log.info(f"   📄 Datos CSV: reports/*_daily_data_*.csv")
        
        log.info(f"\n🎉 Análisis profundo completado exitosamente!")
        return True
    else:
        log.error(f"❌ No se pudo analizar ningún símbolo exitosamente")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)