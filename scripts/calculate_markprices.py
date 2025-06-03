#!/usr/bin/env python3
"""
Mark price calculation script - ADVANCED VWAP VERSION
Calcula mark prices usando VWAP del orderbook y valida con barras OHLCV
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
from src.database.migrations import create_mark_prices_table, check_mark_prices_schema
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

def ensure_mark_prices_schema():
    """Ensure mark prices table exists"""
    log.info("Checking mark prices schema...")
    
    if not check_mark_prices_schema():
        log.info("Creating mark prices table...")
        create_mark_prices_table()
    else:
        log.info("Mark prices schema is up to date")

def calculate_orderbook_vwap(row: pd.Series, volume_threshold: float = 1000.0) -> Dict:
    """
    Calcula VWAP del orderbook usando múltiples niveles
    
    Args:
        row: Fila del orderbook con todos los niveles
        volume_threshold: Volumen mínimo para incluir en VWAP (en quote currency)
    
    Returns:
        Dict con métricas de VWAP y calidad
    """
    
    # Extraer niveles de bid y ask
    bid_levels = []
    ask_levels = []
    
    for i in range(1, 11):  # 10 niveles
        bid_price = row.get(f'bid{i}_price')
        bid_size = row.get(f'bid{i}_size')
        ask_price = row.get(f'ask{i}_price')
        ask_size = row.get(f'ask{i}_size')
        
        # Validar bid
        if pd.notna(bid_price) and pd.notna(bid_size) and bid_price > 0 and bid_size > 0:
            volume_usd = bid_price * bid_size
            if volume_usd >= volume_threshold:
                bid_levels.append({
                    'price': bid_price,
                    'size': bid_size,
                    'volume_usd': volume_usd,
                    'level': i
                })
        
        # Validar ask
        if pd.notna(ask_price) and pd.notna(ask_size) and ask_price > 0 and ask_size > 0:
            volume_usd = ask_price * ask_size
            if volume_usd >= volume_threshold:
                ask_levels.append({
                    'price': ask_price,
                    'size': ask_size,
                    'volume_usd': volume_usd,
                    'level': i
                })
    
    if not bid_levels or not ask_levels:
        return {
            'vwap': None,
            'quality_score': 0.0,
            'bid_depth': 0,
            'ask_depth': 0,
            'total_bid_volume': 0,
            'total_ask_volume': 0,
            'weighted_spread_pct': None,
            'depth_imbalance': None
        }
    
    # Calcular VWAP del lado bid
    total_bid_value = sum(level['price'] * level['size'] for level in bid_levels)
    total_bid_size = sum(level['size'] for level in bid_levels)
    bid_vwap = total_bid_value / total_bid_size if total_bid_size > 0 else None
    
    # Calcular VWAP del lado ask
    total_ask_value = sum(level['price'] * level['size'] for level in ask_levels)
    total_ask_size = sum(level['size'] for level in ask_levels)
    ask_vwap = total_ask_value / total_ask_size if total_ask_size > 0 else None
    
    if bid_vwap is None or ask_vwap is None:
        return {
            'vwap': None,
            'quality_score': 0.0,
            'bid_depth': len(bid_levels),
            'ask_depth': len(ask_levels),
            'total_bid_volume': sum(level['volume_usd'] for level in bid_levels),
            'total_ask_volume': sum(level['volume_usd'] for level in ask_levels),
            'weighted_spread_pct': None,
            'depth_imbalance': None
        }
    
    # Mark price = promedio ponderado de VWAP bid y ask
    # Dar más peso al lado con más liquidez
    total_bid_volume_usd = sum(level['volume_usd'] for level in bid_levels)
    total_ask_volume_usd = sum(level['volume_usd'] for level in ask_levels)
    total_volume = total_bid_volume_usd + total_ask_volume_usd
    
    if total_volume > 0:
        bid_weight = total_bid_volume_usd / total_volume
        ask_weight = total_ask_volume_usd / total_volume
        vwap_mark_price = (bid_vwap * bid_weight) + (ask_vwap * ask_weight)
    else:
        # Fallback a simple promedio
        vwap_mark_price = (bid_vwap + ask_vwap) / 2
    
    # Calcular métricas de calidad
    weighted_spread_pct = (ask_vwap - bid_vwap) / vwap_mark_price * 100
    
    # Depth imbalance (positivo = más liquidez en asks, negativo = más en bids)
    depth_imbalance = (total_ask_volume_usd - total_bid_volume_usd) / total_volume * 100 if total_volume > 0 else 0
    
    # Quality score basado en:
    # - Número de niveles activos
    # - Volumen total
    # - Spread
    # - Balance de liquidez
    depth_score = min(1.0, (len(bid_levels) + len(ask_levels)) / 10)  # Max 10 niveles cada lado
    volume_score = min(1.0, total_volume / 50000)  # Normalizado a $50K
    spread_score = max(0.0, 1.0 - (weighted_spread_pct / 2.0))  # Penalizar spreads >2%
    balance_score = max(0.0, 1.0 - (abs(depth_imbalance) / 50))  # Penalizar desbalances >50%
    
    quality_score = (depth_score * 0.3 + volume_score * 0.3 + spread_score * 0.3 + balance_score * 0.1)
    
    return {
        'vwap': vwap_mark_price,
        'quality_score': quality_score,
        'bid_depth': len(bid_levels),
        'ask_depth': len(ask_levels),
        'total_bid_volume': total_bid_volume_usd,
        'total_ask_volume': total_ask_volume_usd,
        'weighted_spread_pct': weighted_spread_pct,
        'depth_imbalance': depth_imbalance,
        'bid_vwap': bid_vwap,
        'ask_vwap': ask_vwap
    }

def validate_against_ohlcv(vwap_price: float, ohlcv_data: Dict) -> Dict:
    """
    Valida el VWAP calculado contra la barra OHLCV
    
    Args:
        vwap_price: Precio VWAP calculado del orderbook
        ohlcv_data: Dict con datos OHLCV (open, high, low, close)
    
    Returns:
        Dict con validación y métricas
    """
    
    if not ohlcv_data or vwap_price is None:
        return {
            'is_within_range': False,
            'price_vs_close_pct': None,
            'price_vs_typical_pct': None,
            'ohlcv_validation_score': 0.0,
            'validation_status': 'no_ohlcv_data'
        }
    
    open_price = ohlcv_data.get('open')
    high_price = ohlcv_data.get('high')
    low_price = ohlcv_data.get('low')
    close_price = ohlcv_data.get('close')
    
    # Verificar que tenemos datos válidos
    if any(pd.isna(x) or x <= 0 for x in [open_price, high_price, low_price, close_price]):
        return {
            'is_within_range': False,
            'price_vs_close_pct': None,
            'price_vs_typical_pct': None,
            'ohlcv_validation_score': 0.0,
            'validation_status': 'invalid_ohlcv'
        }
    
    # Validar que VWAP está dentro del rango OHLCV
    is_within_range = low_price <= vwap_price <= high_price
    
    # Calcular desviaciones
    price_vs_close_pct = (vwap_price - close_price) / close_price * 100
    
    # Typical price = (high + low + close) / 3
    typical_price = (high_price + low_price + close_price) / 3
    price_vs_typical_pct = (vwap_price - typical_price) / typical_price * 100
    
    # Score de validación
    range_score = 1.0 if is_within_range else 0.0
    
    # Penalizar desviaciones grandes vs close
    close_deviation_score = max(0.0, 1.0 - (abs(price_vs_close_pct) / 2.0))  # Penalizar >2%
    
    # Penalizar desviaciones vs typical price
    typical_deviation_score = max(0.0, 1.0 - (abs(price_vs_typical_pct) / 1.5))  # Penalizar >1.5%
    
    ohlcv_validation_score = (range_score * 0.5 + close_deviation_score * 0.3 + typical_deviation_score * 0.2)
    
    # Determinar status
    if not is_within_range:
        validation_status = 'outside_ohlcv_range'
    elif abs(price_vs_close_pct) > 2.0:
        validation_status = 'high_deviation_close'
    elif abs(price_vs_typical_pct) > 1.5:
        validation_status = 'high_deviation_typical'
    else:
        validation_status = 'validated'
    
    return {
        'is_within_range': is_within_range,
        'price_vs_close_pct': price_vs_close_pct,
        'price_vs_typical_pct': price_vs_typical_pct,
        'ohlcv_validation_score': ohlcv_validation_score,
        'validation_status': validation_status
    }

def get_existing_mark_price_range(symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get existing mark price data range for symbol"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
            FROM mark_prices 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        return result.min_ts, result.max_ts

def calculate_mark_prices_for_period(symbol: str, start_date: datetime, end_date: datetime) -> int:
    """Calculate mark prices for a specific period using VWAP + OHLCV validation"""
    log.info(f"Calculating VWAP mark prices for {symbol} from {start_date} to {end_date}")
    
    with db_manager.get_session() as session:
        # Get orderbook data with all levels
        orderbook_query = text("""
            SELECT 
                timestamp,
                bid1_price, bid1_size, bid2_price, bid2_size, bid3_price, bid3_size,
                bid4_price, bid4_size, bid5_price, bid5_size, bid6_price, bid6_size,
                bid7_price, bid7_size, bid8_price, bid8_size, bid9_price, bid9_size,
                bid10_price, bid10_size,
                ask1_price, ask1_size, ask2_price, ask2_size, ask3_price, ask3_size,
                ask4_price, ask4_size, ask5_price, ask5_size, ask6_price, ask6_size,
                ask7_price, ask7_size, ask8_price, ask8_size, ask9_price, ask9_size,
                ask10_price, ask10_size
            FROM orderbook 
            WHERE symbol = :symbol 
            AND timestamp >= :start_date 
            AND timestamp <= :end_date
            AND bid1_price IS NOT NULL 
            AND ask1_price IS NOT NULL
            ORDER BY timestamp
        """)
        
        orderbook_df = pd.read_sql(orderbook_query, session.bind, params={
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        
        if orderbook_df.empty:
            log.warning(f"No orderbook data found for {symbol} in period")
            return 0
        
        # Get OHLCV data for validation
        ohlcv_query = text("""
            SELECT timestamp, open, high, low, close, volume
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
        
        # Convert to dict for faster lookup
        ohlcv_dict = {}
        if not ohlcv_df.empty:
            ohlcv_df.set_index('timestamp', inplace=True)
            ohlcv_dict = ohlcv_df.to_dict('index')
        
        log.info(f"Processing {len(orderbook_df):,} orderbook snapshots with {len(ohlcv_dict):,} OHLCV bars")
        
        # Calculate mark prices
        mark_prices = []
        processed_count = 0
        
        for idx, row in orderbook_df.iterrows():
            # Calculate VWAP from orderbook
            vwap_result = calculate_orderbook_vwap(row)
            
            if vwap_result['vwap'] is None:
                continue  # Skip if no valid VWAP
            
            # Get corresponding OHLCV data
            timestamp = row['timestamp']
            ohlcv_data = ohlcv_dict.get(timestamp, {})
            
            # Validate against OHLCV
            validation_result = validate_against_ohlcv(vwap_result['vwap'], ohlcv_data)
            
            # Combined quality score
            orderbook_quality = vwap_result['quality_score']
            ohlcv_quality = validation_result['ohlcv_validation_score']
            combined_quality_score = (orderbook_quality * 0.7 + ohlcv_quality * 0.3)
            
            # Determine if mark price is valid
            is_valid = (
                vwap_result['quality_score'] >= 0.3 and  # Minimum orderbook quality
                validation_result['ohlcv_validation_score'] >= 0.5 and  # Reasonable OHLCV agreement
                vwap_result['weighted_spread_pct'] is not None and
                vwap_result['weighted_spread_pct'] <= 5.0  # Maximum 5% spread
            )
            
            # Determine validation source
            if validation_result['validation_status'] == 'validated':
                validation_source = 'vwap_validated'
            elif validation_result['validation_status'] in ['high_deviation_close', 'high_deviation_typical']:
                validation_source = 'vwap_minor_deviation'
            elif validation_result['validation_status'] == 'outside_ohlcv_range':
                validation_source = 'vwap_ohlcv_conflict'
                is_valid = False  # Mark as invalid if outside OHLCV range
            else:
                validation_source = 'vwap_no_ohlcv'
            
            mark_prices.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'mark_price': float(vwap_result['vwap']),
                'orderbook_mid': (row['bid1_price'] + row['ask1_price']) / 2 if pd.notna(row['bid1_price']) and pd.notna(row['ask1_price']) else None,
                'ohlcv_close': float(ohlcv_data.get('close')) if ohlcv_data.get('close') and pd.notna(ohlcv_data.get('close')) else None,
                'bid_ask_spread_pct': float(vwap_result['weighted_spread_pct']) if vwap_result['weighted_spread_pct'] is not None else None,
                'price_deviation_pct': float(validation_result['price_vs_close_pct']) if validation_result['price_vs_close_pct'] is not None else None,
                'liquidity_score': float(combined_quality_score),
                'is_valid': is_valid,
                'validation_source': validation_source,
            })
            
            processed_count += 1
            if processed_count % 10000 == 0:
                log.info(f"  Processed {processed_count:,} / {len(orderbook_df):,} snapshots...")
        
        if not mark_prices:
            log.warning(f"No valid mark prices calculated for {symbol}")
            return 0
        
        # Insert mark prices in batches
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(mark_prices), batch_size):
            batch = mark_prices[i:i + batch_size]
            
            insert_query = text("""
                INSERT INTO mark_prices (
                    symbol, timestamp, mark_price, orderbook_mid, ohlcv_close,
                    bid_ask_spread_pct, price_deviation_pct, liquidity_score,
                    is_valid, validation_source
                ) VALUES (
                    :symbol, :timestamp, :mark_price, :orderbook_mid, :ohlcv_close,
                    :bid_ask_spread_pct, :price_deviation_pct, :liquidity_score,
                    :is_valid, :validation_source
                ) ON CONFLICT (symbol, timestamp) 
                DO UPDATE SET
                    mark_price = EXCLUDED.mark_price,
                    orderbook_mid = EXCLUDED.orderbook_mid,
                    ohlcv_close = EXCLUDED.ohlcv_close,
                    bid_ask_spread_pct = EXCLUDED.bid_ask_spread_pct,
                    price_deviation_pct = EXCLUDED.price_deviation_pct,
                    liquidity_score = EXCLUDED.liquidity_score,
                    is_valid = EXCLUDED.is_valid,
                    validation_source = EXCLUDED.validation_source
            """)
            
            session.execute(insert_query, batch)
            total_inserted += len(batch)
            
            if i % (batch_size * 10) == 0:  # Log progress every 10 batches
                log.info(f"  Inserted {i + len(batch):,} / {len(mark_prices):,} mark prices...")
        
        # Log quality summary
        valid_count = sum(1 for mp in mark_prices if mp['is_valid'])
        avg_quality = sum(mp['liquidity_score'] for mp in mark_prices) / len(mark_prices)
        spreads_with_data = [mp['bid_ask_spread_pct'] for mp in mark_prices if mp['bid_ask_spread_pct'] is not None]
        avg_spread = sum(spreads_with_data) / len(spreads_with_data) if spreads_with_data else 0
        
        log.info(f"Mark price calculation completed for {symbol}:")
        log.info(f"  Total calculated: {total_inserted:,}")
        log.info(f"  Valid prices: {valid_count:,} ({valid_count/total_inserted*100:.1f}%)")
        log.info(f"  Average quality score: {avg_quality:.3f}")
        log.info(f"  Average VWAP spread: {avg_spread:.3f}%")
        
        return total_inserted

def calculate_mark_prices_incremental(symbol: str) -> int:
    """Calculate mark prices incrementally (only missing periods)"""
    log.info(f"Starting incremental VWAP mark price calculation for {symbol}")
    
    # Get existing data range
    existing_min, existing_max = get_existing_mark_price_range(symbol)
    
    with db_manager.get_session() as session:
        # Get available data range from orderbook
        available_result = session.execute(text("""
            SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
            FROM orderbook 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        if not available_result.min_ts:
            log.warning(f"No orderbook data available for {symbol}")
            return 0
        
        available_min = available_result.min_ts
        available_max = available_result.max_ts
        
        periods_to_process = []
        
        if existing_min is None:
            # No existing data - process everything
            periods_to_process.append((available_min, available_max))
            log.info(f"No existing mark prices - processing full range: {available_min} to {available_max}")
        else:
            # Check for gaps
            if available_min < existing_min:
                # Gap at beginning
                periods_to_process.append((available_min, existing_min - timedelta(minutes=1)))
                log.info(f"Gap at beginning: {available_min} to {existing_min}")
            
            if available_max > existing_max:
                # Gap at end (new data)
                periods_to_process.append((existing_max + timedelta(minutes=1), available_max))
                log.info(f"New data at end: {existing_max} to {available_max}")
        
        if not periods_to_process:
            log.info(f"No missing mark price data for {symbol}")
            return 0
        
        total_processed = 0
        for start_date, end_date in periods_to_process:
            processed = calculate_mark_prices_for_period(symbol, start_date, end_date)
            total_processed += processed
        
        return total_processed

def get_mark_price_summary(symbol: str) -> Dict:
    """Get summary statistics for calculated mark prices"""
    with db_manager.get_session() as session:
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN is_valid = TRUE THEN 1 END) as valid_records,
                COUNT(CASE WHEN validation_source = 'vwap_validated' THEN 1 END) as fully_validated,
                COUNT(CASE WHEN validation_source = 'vwap_minor_deviation' THEN 1 END) as minor_deviation,
                COUNT(CASE WHEN validation_source = 'vwap_ohlcv_conflict' THEN 1 END) as ohlcv_conflict,
                COUNT(CASE WHEN validation_source = 'vwap_no_ohlcv' THEN 1 END) as no_ohlcv,
                MIN(timestamp) as min_timestamp,
                MAX(timestamp) as max_timestamp,
                AVG(CASE WHEN price_deviation_pct IS NOT NULL THEN ABS(price_deviation_pct) END) as avg_abs_deviation_pct,
                AVG(CASE WHEN bid_ask_spread_pct IS NOT NULL THEN bid_ask_spread_pct END) as avg_spread_pct,
                AVG(CASE WHEN liquidity_score IS NOT NULL THEN liquidity_score END) as avg_quality_score,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mark_price) as median_mark_price,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY bid_ask_spread_pct) as p95_spread
            FROM mark_prices 
            WHERE symbol = :symbol
        """), {'symbol': symbol}).fetchone()
        
        return {
            'symbol': symbol,
            'total_records': result.total_records or 0,
            'valid_records': result.valid_records or 0,
            'valid_percentage': (result.valid_records or 0) / max(1, result.total_records or 1) * 100,
            'validation_breakdown': {
                'fully_validated': result.fully_validated or 0,
                'minor_deviation': result.minor_deviation or 0,
                'ohlcv_conflict': result.ohlcv_conflict or 0,
                'no_ohlcv': result.no_ohlcv or 0
            },
            'time_range': {
                'start': result.min_timestamp,
                'end': result.max_timestamp
            },
            'quality_metrics': {
                'avg_abs_deviation_pct': result.avg_abs_deviation_pct or 0,
                'avg_spread_pct': result.avg_spread_pct or 0,
                'avg_quality_score': result.avg_quality_score or 0,
                'median_mark_price': result.median_mark_price or 0,
                'p95_spread': result.p95_spread or 0
            }
        }

def generate_mark_price_report(summaries: Dict):
    """Generate comprehensive mark price calculation report"""
    log.info("\n" + "="*80)
    log.info("VWAP MARK PRICE CALCULATION REPORT")
    log.info("="*80)
    
    total_records = sum(s['total_records'] for s in summaries.values())
    total_valid = sum(s['valid_records'] for s in summaries.values())
    
    log.info(f"Processed {len(summaries)} symbols")
    log.info(f"Total VWAP mark prices calculated: {total_records:,}")
    log.info(f"Total valid mark prices: {total_valid:,} ({total_valid/max(1,total_records)*100:.1f}%)")
    
    for symbol, summary in summaries.items():
        symbol_short = symbol.split('_')[-2] if '_' in symbol else symbol
        log.info(f"\n{symbol_short} VWAP Mark Price Summary:")
        log.info(f"  Records: {summary['total_records']:,}")
        log.info(f"  Valid: {summary['valid_records']:,} ({summary['valid_percentage']:.1f}%)")
        log.info(f"  Time Range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        
        log.info(f"  Validation Quality:")
        validation = summary['validation_breakdown']
        log.info(f"    Fully Validated: {validation['fully_validated']:,}")
        log.info(f"    Minor Deviation: {validation['minor_deviation']:,}")
        log.info(f"    OHLCV Conflict: {validation['ohlcv_conflict']:,}")
        log.info(f"    No OHLCV Data: {validation['no_ohlcv']:,}")
        
        log.info(f"  Quality Metrics:")
        quality = summary['quality_metrics']
        log.info(f"    Avg VWAP Spread: {quality['avg_spread_pct']:.3f}%")
        log.info(f"    P95 VWAP Spread: {quality['p95_spread']:.3f}%")
        log.info(f"    Avg |Deviation| from OHLCV: {quality['avg_abs_deviation_pct']:.3f}%")
        log.info(f"    Avg Quality Score: {quality['avg_quality_score']:.3f}")
        log.info(f"    Median Mark Price: ${quality['median_mark_price']:.6f}")
    
    log.info(f"\nVWAP Calculation Notes:")
    log.info(f"- Mark prices calculated using volume-weighted average of orderbook levels")
    log.info(f"- Validation against OHLCV bars to detect anomalies")
    log.info(f"- Quality scoring based on depth, volume, spread, and OHLCV agreement")
    log.info(f"- Use 'WHERE is_valid = TRUE' for highest quality data in backtesting")

def main():
    """Main VWAP mark price calculation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate VWAP mark prices from orderbook and validate with OHLCV")
    parser.add_argument("--symbol", type=str, help="Specific symbol to process")
    parser.add_argument("--force", action="store_true", help="Force recalculation of existing data")
    
    args = parser.parse_args()
    
    log.info("Starting VWAP mark price calculation...")
    
    # Ensure schema is ready
    ensure_mark_prices_schema()
    
    # Get symbols to process
    if args.symbol:
        symbols = [args.symbol]
    else:
        try:
            active_pairs = settings.get_active_pairs()
            symbols = []
            for pair in active_pairs:
                symbols.extend([pair.symbol1, pair.symbol2])
            symbols = list(set(symbols))  # Remove duplicates
        except Exception as e:
            log.error(f"Could not load symbols from config: {e}")
            symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']  # Fallback
    
    if not symbols:
        log.error("No symbols to process")
        return False
    
    log.info(f"Processing VWAP mark prices for {len(symbols)} symbols: {', '.join([s.split('_')[-2] for s in symbols])}")
    
    summaries = {}
    
    try:
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"PROCESSING VWAP MARK PRICES FOR {symbol}")
            log.info(f"{'='*60}")
            
            if args.force:
                # Delete existing data and recalculate
                with db_manager.get_session() as session:
                    session.execute(text("DELETE FROM mark_prices WHERE symbol = :symbol"), {'symbol': symbol})
                    log.info(f"Deleted existing mark prices for {symbol}")
            
            # Calculate mark prices
            processed_count = calculate_mark_prices_incremental(symbol)
            
            if processed_count > 0:
                log.info(f"Successfully processed {processed_count:,} VWAP mark prices for {symbol}")
                summaries[symbol] = get_mark_price_summary(symbol)
            else:
                log.warning(f"No VWAP mark prices processed for {symbol}")
        
        # Generate final report
        if summaries:
            generate_mark_price_report(summaries)
        
        log.info(f"\nVWAP mark price calculation completed!")
        log.info(f"Processed: {len(summaries)}/{len(symbols)} symbols")
        log.info("Next steps:")
        log.info("  - Review VWAP quality in the report above")
        log.info("  - Run backtester with mark_prices, orderbook, and funding_rates")
        log.info("  - Use 'WHERE is_valid = TRUE' for highest quality mark prices")
        log.info("  - VWAP provides more realistic execution prices than simple mid-point")
        
        return True
        
    except Exception as e:
        log.error(f"VWAP mark price calculation failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)