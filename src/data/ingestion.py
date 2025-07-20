#!/usr/bin/env python3
"""
Data ingestion module - FIXED FUNDING RATES LOGIC
FIXED: Funding rates con detección inteligente y rangos realistas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from sqlalchemy import text
import time

from src.database.connection import db_manager
from src.data.coinapi_client import coinapi_client
from src.data.mexc_client import mexc_client
from src.utils.logger import get_ingestion_logger
from src.utils.exceptions import DataValidationError
from config.settings import settings

log = get_ingestion_logger()

class DataIngestion:
    """Main data ingestion class - CON FUNDING RATES FIXED"""
    
    def __init__(self):
        self.coinapi = coinapi_client
        self.mexc = mexc_client
    
    def ingest_data(self, symbols: List[str], data_types: List[str] = None, 
                   overwrite: bool = False, days_back: int = None) -> Dict[str, Dict[str, bool]]:
        """
        Función principal de ingesta con parámetros selectivos - FIXED
        """
        
        # Tipos de datos disponibles
        available_types = ['ohlcv', 'orderbook', 'funding']
        
        # Si no se especifican tipos, usar todos
        if data_types is None:
            data_types = available_types.copy()
        else:
            # Validar tipos especificados
            data_types = [dt for dt in data_types if dt in available_types]
        
        if not data_types:
            log.error("No valid data types specified")
            return {}
        
        # FIXED: En overwrite mode, si no se especifica days_back, usar todos los datos
        if overwrite and days_back is None:
            log.info(f"🚀 Starting data ingestion for {len(symbols)} symbols")
            log.info(f"📊 Data types: {data_types}")
            log.info(f"🔄 Mode: OVERWRITE (ALL AVAILABLE DATA)")
        elif overwrite:
            log.info(f"🚀 Starting data ingestion for {len(symbols)} symbols")
            log.info(f"📊 Data types: {data_types}")
            log.info(f"🔄 Mode: OVERWRITE ({days_back} days back)")
        else:
            log.info(f"🚀 Starting data ingestion for {len(symbols)} symbols")
            log.info(f"📊 Data types: {data_types}")
            log.info(f"🔄 Mode: INCREMENTAL")
            if days_back is None:
                days_back = 7  # Default for incremental mode
        
        results = {}
        
        for symbol in symbols:
            log.info(f"\n{'='*60}")
            log.info(f"PROCESSING {symbol}")
            log.info(f"{'='*60}")
            
            symbol_results = {}
            
            # 1. Update symbol info (lightweight operation)
            log.info(f"📋 Updating symbol info for {symbol}...")
            self.update_symbol_info(symbol)
            
            # 2. Process each requested data type
            for data_type in data_types:
                log.info(f"\n📊 Processing {data_type.upper()} for {symbol}...")
                
                try:
                    if data_type == 'ohlcv':
                        success = self.ingest_ohlcv_data(symbol, overwrite=overwrite, days_back=days_back)
                        symbol_results['ohlcv'] = success
                        
                    elif data_type == 'orderbook':
                        success = self.ingest_orderbook_data(symbol, overwrite=overwrite, days_back=days_back)
                        symbol_results['orderbook'] = success
                        
                    elif data_type == 'funding':
                        if "PERP_" in symbol:
                            # FIXED: Usar función específica para funding con nueva lógica
                            success = self.ingest_funding_rates_single(symbol, overwrite=overwrite)
                            symbol_results['funding'] = success
                        else:
                            log.info(f"⏭️ Skipping funding rates for {symbol} (not a perpetual contract)")
                            symbol_results['funding'] = True  # Not applicable, so mark as successful
                    
                    if success:
                        log.info(f"✅ {data_type.upper()} completed for {symbol}")
                    else:
                        log.error(f"❌ {data_type.upper()} failed for {symbol}")
                        
                except Exception as e:
                    log.error(f"💥 Error processing {data_type} for {symbol}: {e}")
                    symbol_results[data_type] = False
            
            results[symbol] = symbol_results
            
            # Log symbol summary
            successful_types = [dt for dt, success in symbol_results.items() if success]
            log.info(f"\n📊 {symbol} SUMMARY: {len(successful_types)}/{len(data_types)} data types successful")
            log.info(f"✅ Successful: {successful_types}")
            if len(successful_types) < len(data_types):
                failed_types = [dt for dt, success in symbol_results.items() if not success]
                log.warning(f"❌ Failed: {failed_types}")
        
        # Final summary
        log.info(f"\n🎉 DATA INGESTION COMPLETED!")
        self._log_final_summary(results, data_types)
        
        return results
    
    def ingest_funding_rates_single(self, symbol: str, overwrite: bool = False) -> bool:
        """
        NUEVA FUNCIÓN: Ingesta funding rates para un símbolo con lógica inteligente
        """
        log.info(f"Starting SMART funding rates ingestion for {symbol} (overwrite={overwrite})")
        
        try:
            if overwrite:
                # Modo overwrite: eliminar todo y fetch todo disponible
                deleted_count = self.delete_existing_data(symbol, 'funding')
                log.info(f"Deleted {deleted_count:,} existing funding records for overwrite")
                
                # Fetch ALL available data
                funding_data = self.mexc.get_funding_rate_history(symbol)
                
                if funding_data.empty:
                    log.warning(f"No funding rate data obtained for {symbol} in overwrite mode")
                    return False
                
                # Insert data
                success = self._ingest_symbol_funding_rates(symbol, funding_data)
                if success:
                    log.info(f"✅ Successfully ingested {len(funding_data)} funding records for {symbol}")
                return success
                
            else:
                # NUEVA LÓGICA INCREMENTAL INTELIGENTE
                return self._ingest_funding_incremental_smart(symbol)
                
        except Exception as e:
            log.error(f"Error in funding rates ingestion for {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False
    
    def _ingest_funding_incremental_smart(self, symbol: str) -> bool:
        """
        NUEVA FUNCIÓN: Lógica incremental inteligente para funding rates
        """
        log.info(f"🧠 Smart incremental funding ingestion for {symbol}")
        
        # 1. Verificar datos existentes
        min_date, max_date = self.get_funding_data_range(symbol)
        
        with db_manager.get_session() as session:
            # Obtener estadísticas de datos existentes
            stats_query = text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT DATE(timestamp)) as total_days,
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts
                FROM funding_rates 
                WHERE symbol = :symbol
            """)
            
            stats = session.execute(stats_query, {'symbol': symbol}).fetchone()
            
            log.info(f"📊 Current funding data for {symbol}:")
            log.info(f"  Records: {stats.total_records}")
            log.info(f"  Days covered: {stats.total_days}")
            if stats.min_ts and stats.max_ts:
                log.info(f"  Range: {stats.min_ts.date()} to {stats.max_ts.date()}")
        
        # 2. Si hay datos recientes y abundantes, probablemente está completo
        if stats.total_records > 0 and stats.max_ts:
            days_since_last = (datetime.now() - stats.max_ts).days
            
            # CRITERIO INTELIGENTE: Si tiene datos abundantes y recientes, probablemente está completo
            if stats.total_records >= 1000 and days_since_last <= 3:
                log.info(f"✅ {symbol} funding appears complete: {stats.total_records} records, last update {days_since_last} days ago")
                
                # Solo verificar si necesita actualización reciente
                recent_data = self._fetch_recent_funding_only(symbol)
                if not recent_data.empty:
                    inserted = self._ingest_symbol_funding_rates(symbol, recent_data)
                    if inserted:
                        log.info(f"✅ Added {len(recent_data)} recent funding records for {symbol}")
                    return True
                else:
                    log.info(f"✅ No recent funding updates needed for {symbol}")
                    return True
            
            # Si hay datos pero no son tan recientes, fetch solo datos recientes
            elif stats.total_records > 100:
                log.info(f"📅 {symbol} has {stats.total_records} records, fetching recent updates only")
                recent_data = self._fetch_recent_funding_only(symbol)
                
                if not recent_data.empty:
                    inserted = self._ingest_symbol_funding_rates(symbol, recent_data)
                    if inserted:
                        log.info(f"✅ Added {len(recent_data)} recent funding records for {symbol}")
                    return True
                else:
                    log.info(f"✅ No recent updates available for {symbol}")
                    return True
        
        # 3. Si hay pocos datos o ninguno, fetch historial completo
        log.info(f"📥 {symbol} needs complete funding history - fetching all available data")
        funding_data = self.mexc.get_funding_rate_history(symbol)
        
        if funding_data.empty:
            log.warning(f"No funding rate data obtained for {symbol}")
            # Si ya había algunos datos, no es un fallo total
            return stats.total_records > 0
        
        # Insert data
        success = self._ingest_symbol_funding_rates(symbol, funding_data)
        if success:
            log.info(f"✅ Successfully ingested {len(funding_data)} funding records for {symbol}")
        
        return success
    
    def _fetch_recent_funding_only(self, symbol: str, days_back: int = 7) -> pd.DataFrame:
        """
        NUEVA FUNCIÓN: Fetch solo funding rates recientes
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            log.info(f"🔍 Fetching recent funding for {symbol}: last {days_back} days")
            
            recent_data = self.mexc.get_funding_rate_history_range(symbol, start_date, end_date)
            
            if not recent_data.empty:
                log.info(f"📥 Found {len(recent_data)} recent funding records for {symbol}")
            else:
                log.info(f"📭 No recent funding data for {symbol}")
            
            return recent_data
            
        except Exception as e:
            log.warning(f"Error fetching recent funding for {symbol}: {e}")
            return pd.DataFrame()
    
    def _detect_missing_days(self, available_start: datetime, available_end: datetime, 
                            existing_min: datetime, existing_max: datetime, 
                            symbol: str, data_type: str) -> List[Tuple[datetime, datetime]]:
        """
        NUEVA FUNCIÓN: Detecta todos los días faltantes en el rango completo
        Retorna lista de rangos (start_date, end_date) que necesitan ser fetched
        """
        log.info(f"🔍 Detecting missing days for {symbol} {data_type}")
        log.info(f"  Available range: {available_start.date()} to {available_end.date()}")
        
        if existing_min and existing_max:
            log.info(f"  Existing range: {existing_min.date()} to {existing_max.date()}")
        else:
            log.info(f"  No existing data")
        
        ranges_to_fetch = []
        
        # Si no hay datos existentes, fetch todo el rango disponible
        if not existing_min or not existing_max:
            ranges_to_fetch.append((available_start.date(), available_end.date()))
            log.info(f"  No existing data - will fetch: {available_start.date()} to {available_end.date()}")
            return ranges_to_fetch
        
        # Verificar qué días existen realmente en la DB
        with db_manager.get_session() as session:
            table_name = 'ohlcv' if data_type == 'ohlcv' else 'orderbook'
            
            existing_days_query = f"""
                SELECT DISTINCT DATE(timestamp) as day
                FROM {table_name}
                WHERE symbol = :symbol
                AND timestamp >= :start_date
                AND timestamp <= :end_date
                ORDER BY day
            """
            
            result = session.execute(text(existing_days_query), {
                'symbol': symbol,
                'start_date': available_start,
                'end_date': available_end
            }).fetchall()
            
            existing_days = set(row.day for row in result)
            log.info(f"  Found {len(existing_days)} existing days in DB")
        
        # Generar set completo de días que deberían existir
        current_date = available_start.date()
        should_exist_days = set()
        
        while current_date <= available_end.date():
            should_exist_days.add(current_date)
            current_date += timedelta(days=1)
        
        log.info(f"  Should exist: {len(should_exist_days)} days")
        
        # Encontrar días faltantes
        missing_days = should_exist_days - existing_days
        
        if not missing_days:
            log.info(f"  ✅ No missing days for {symbol} {data_type}")
            return ranges_to_fetch
        
        log.info(f"  ❌ Missing {len(missing_days)} days for {symbol} {data_type}")
        
        # Agrupar días consecutivos en rangos
        missing_days_sorted = sorted(missing_days)
        
        if missing_days_sorted:
            range_start = missing_days_sorted[0]
            range_end = missing_days_sorted[0]
            
            for day in missing_days_sorted[1:]:
                if day == range_end + timedelta(days=1):
                    # Día consecutivo, extender rango
                    range_end = day
                else:
                    # Gap encontrado, cerrar rango actual y empezar nuevo
                    ranges_to_fetch.append((range_start, range_end))
                    range_start = day
                    range_end = day
            
            # Agregar último rango
            ranges_to_fetch.append((range_start, range_end))
        
        # Log de rangos a fetch
        log.info(f"  📋 Will fetch {len(ranges_to_fetch)} ranges:")
        for i, (start, end) in enumerate(ranges_to_fetch):
            days_in_range = (end - start).days + 1
            log.info(f"    Range {i+1}: {start} to {end} ({days_in_range} days)")
        
        return ranges_to_fetch
    
    def get_symbol_data_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get existing data range for symbol"""
        with db_manager.get_session() as session:
            result = session.execute(text("""
                SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date
                FROM ohlcv 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            return result.min_date, result.max_date
    
    def get_orderbook_data_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get existing orderbook data range for symbol"""
        with db_manager.get_session() as session:
            result = session.execute(text("""
                SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date
                FROM orderbook 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            return result.min_date, result.max_date
    
    def get_funding_data_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get existing funding data range for symbol"""
        with db_manager.get_session() as session:
            result = session.execute(text("""
                SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date
                FROM funding_rates 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            return result.min_date, result.max_date
    
    def delete_existing_data(self, symbol: str, data_type: str, start_date: str = None, end_date: str = None) -> int:
        """Eliminar datos existentes para sobreescritura"""
        
        table_map = {
            'ohlcv': 'ohlcv',
            'orderbook': 'orderbook', 
            'funding': 'funding_rates'
        }
        
        if data_type not in table_map:
            log.error(f"Tipo de datos no válido: {data_type}")
            return 0
        
        table_name = table_map[data_type]
        
        try:
            with db_manager.get_session() as session:
                if start_date and end_date:
                    # Eliminar solo el rango específico
                    delete_query = text(f"""
                        DELETE FROM {table_name} 
                        WHERE symbol = :symbol 
                        AND timestamp >= :start_date 
                        AND timestamp <= :end_date
                    """)
                    result = session.execute(delete_query, {
                        'symbol': symbol,
                        'start_date': start_date,
                        'end_date': end_date
                    })
                    log.info(f"🗑️ Eliminando {data_type} de {symbol} entre {start_date} y {end_date}")
                else:
                    # Eliminar todos los datos del símbolo
                    delete_query = text(f"DELETE FROM {table_name} WHERE symbol = :symbol")
                    result = session.execute(delete_query, {'symbol': symbol})
                    log.info(f"🗑️ Eliminando TODOS los datos {data_type} para {symbol}")
                
                session.commit()
                deleted_count = result.rowcount
                log.info(f"✅ Eliminados {deleted_count:,} registros {data_type} para {symbol}")
                return deleted_count
                
        except Exception as e:
            log.error(f"Error eliminando datos {data_type} de {symbol}: {e}")
            return 0
    
    def update_symbol_info(self, symbol: str) -> bool:
        """Update symbol information with robust error handling"""
        log.debug(f"Updating symbol info for {symbol}")
        
        try:
            symbol_info = self.coinapi.get_symbol_info(symbol)
            
            with db_manager.get_session() as session:
                # Upsert symbol info
                session.execute(text("""
                    INSERT INTO symbol_info (
                        symbol_id, exchange_id, symbol_type, asset_id_base, asset_id_quote,
                        data_start, data_end, data_quote_start, data_quote_end,
                        data_orderbook_start, data_orderbook_end,
                        data_trade_start, data_trade_end
                    ) VALUES (
                        :symbol_id, :exchange_id, :symbol_type, :asset_id_base, :asset_id_quote,
                        :data_start, :data_end, :data_quote_start, :data_quote_end,
                        :data_orderbook_start, :data_orderbook_end,
                        :data_trade_start, :data_trade_end
                    ) ON CONFLICT (symbol_id) 
                    DO UPDATE SET
                        exchange_id = EXCLUDED.exchange_id,
                        symbol_type = EXCLUDED.symbol_type,
                        asset_id_base = EXCLUDED.asset_id_base,
                        asset_id_quote = EXCLUDED.asset_id_quote,
                        data_start = EXCLUDED.data_start,
                        data_end = EXCLUDED.data_end,
                        data_quote_start = EXCLUDED.data_quote_start,
                        data_quote_end = EXCLUDED.data_quote_end,
                        data_orderbook_start = EXCLUDED.data_orderbook_start,
                        data_orderbook_end = EXCLUDED.data_orderbook_end,
                        data_trade_start = EXCLUDED.data_trade_start,
                        data_trade_end = EXCLUDED.data_trade_end,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'symbol_id': symbol,
                    'exchange_id': symbol_info.get('exchange_id'),
                    'symbol_type': symbol_info.get('symbol_type'),
                    'asset_id_base': symbol_info.get('asset_id_base'),
                    'asset_id_quote': symbol_info.get('asset_id_quote'),
                    'data_start': symbol_info.get('data_start'),
                    'data_end': symbol_info.get('data_end'),
                    'data_quote_start': symbol_info.get('data_quote_start'),
                    'data_quote_end': symbol_info.get('data_quote_end'),
                    'data_orderbook_start': symbol_info.get('data_orderbook_start'),
                    'data_orderbook_end': symbol_info.get('data_orderbook_end'),
                    'data_trade_start': symbol_info.get('data_trade_start'),
                    'data_trade_end': symbol_info.get('data_trade_end')
                })
                session.commit()
            
            log.debug(f"✅ Updated symbol info for {symbol}")
            return True
            
        except Exception as e:
            log.warning(f"Failed to update symbol info for {symbol}: {e}")
            return False
    
    def ingest_ohlcv_data(self, symbol: str, overwrite: bool = False, days_back: int = None) -> bool:
        """Ingest OHLCV data - FIXED: Lógica de éxito corregida para modo incremental"""
        log.info(f"Starting OHLCV ingestion for {symbol} (overwrite={overwrite}, days_back={days_back})")
        
        try:
            # Get available data range from API
            available_start, available_end = self.coinapi.get_available_date_range(symbol)
            
            if not available_start or not available_end:
                log.warning(f"No available data range for {symbol}")
                return False
            
            if overwrite:
                if days_back is None:
                    # FIXED: Usar TODO el rango disponible
                    start_fetch = available_start
                    end_fetch = available_end
                    log.info(f"OVERWRITE MODE: Fetching ALL AVAILABLE OHLCV data {start_fetch.date()} to {end_fetch.date()}")
                else:
                    # Usar días específicos
                    start_fetch = available_end - timedelta(days=days_back)
                    end_fetch = available_end
                    log.info(f"OVERWRITE MODE: Fetching OHLCV {start_fetch.date()} to {end_fetch.date()}")
                
                # Eliminar datos existentes en el rango
                self.delete_existing_data(symbol, 'ohlcv', start_fetch.isoformat(), end_fetch.isoformat())
                
                # Fetch data para el rango especificado
                total_records = self._fetch_ohlcv_range(symbol, start_fetch, end_fetch)
                
                # En overwrite mode, success basado en si se obtuvieron datos
                success = total_records > 0
                log.info(f"✅ Successfully fetched {total_records} OHLCV records for {symbol}")
                
            else:
                # Modo incremental - NUEVA LÓGICA MEJORADA
                min_date, max_date = self.get_symbol_data_range(symbol)
                
                # Usar nueva función de detección de gaps
                ranges_to_fetch = self._detect_missing_days(
                    available_start, available_end, min_date, max_date, symbol, 'ohlcv'
                )
                
                if not ranges_to_fetch:
                    log.info(f"✅ No missing OHLCV data for {symbol} - already complete")
                    return True  # ← FIXED: Ya completo = éxito
                
                # Fetch data para cada rango faltante
                total_records = 0
                for start_date, end_date in ranges_to_fetch:
                    # Convertir dates a datetime
                    start_datetime = datetime.combine(start_date, datetime.min.time())
                    end_datetime = datetime.combine(end_date, datetime.max.time())
                    
                    records = self._fetch_ohlcv_range(symbol, start_datetime, end_datetime)
                    total_records += records
                
                # FIXED: En modo incremental, success si ya había datos O se agregaron nuevos
                if total_records > 0:
                    success = True
                    log.info(f"✅ Successfully fetched {total_records} new OHLCV records for {symbol}")
                else:
                    # Verificar si ya había datos suficientes
                    min_date, max_date = self.get_symbol_data_range(symbol)
                    if min_date and max_date:
                        # Ya hay datos en DB, los días "faltantes" simplemente están vacíos en la API
                        success = True
                        log.info(f"✅ OHLCV for {symbol} already complete - missing days have no data in API")
                    else:
                        # No hay datos en DB y no se pudieron obtener nuevos
                        success = False
                        log.warning(f"❌ No OHLCV data exists and no new data could be fetched for {symbol}")
            
            return success
            
        except Exception as e:
            log.error(f"OHLCV ingestion failed for {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False
    
    def ingest_orderbook_data(self, symbol: str, overwrite: bool = False, days_back: int = None) -> bool:
        """Ingest orderbook data - FIXED: Lógica de éxito corregida para modo incremental"""
        log.info(f"Starting orderbook ingestion for {symbol} (overwrite={overwrite}, days_back={days_back})")
        
        try:
            # Get available range
            available_start, available_end = self.coinapi.get_available_date_range(symbol)
            
            if not available_start or not available_end:
                log.warning(f"No available orderbook data range for {symbol}")
                return False
            
            if overwrite:
                if days_back is None:
                    # FIXED: Usar TODO el rango disponible
                    start_fetch = available_start.date()
                    end_fetch = available_end.date()
                    log.info(f"OVERWRITE MODE: Fetching ALL AVAILABLE orderbook data {start_fetch} to {end_fetch}")
                else:
                    # Usar días específicos
                    start_fetch = available_end.date() - timedelta(days=days_back)
                    end_fetch = available_end.date()
                    log.info(f"OVERWRITE MODE: Fetching orderbook {start_fetch} to {end_fetch}")
                
                # Eliminar datos existentes en el rango con fechas precisas
                start_datetime = datetime.combine(start_fetch, datetime.min.time())
                end_datetime = datetime.combine(end_fetch, datetime.max.time())
                
                deleted_count = self.delete_existing_data(
                    symbol, 'orderbook', 
                    start_datetime.isoformat(),
                    end_datetime.isoformat()
                )
                
                log.info(f"Deleted {deleted_count:,} existing orderbook records for overwrite")
                
                # Fetch data para el rango especificado
                total_records = self._fetch_orderbook_range(symbol, start_fetch, end_fetch)
                success = total_records > 0
                log.info(f"✅ Successfully fetched {total_records} orderbook snapshots for {symbol}")
                
            else:
                # Modo incremental - NUEVA LÓGICA MEJORADA
                min_date, max_date = self.get_orderbook_data_range(symbol)
                
                # Usar nueva función de detección de gaps
                ranges_to_fetch = self._detect_missing_days(
                    available_start, available_end, min_date, max_date, symbol, 'orderbook'
                )
                
                if not ranges_to_fetch:
                    log.info(f"✅ No missing orderbook data for {symbol} - already complete")
                    return True  # ← FIXED: Ya completo = éxito
                
                # Fetch data para cada rango faltante
                total_records = 0
                for start_date, end_date in ranges_to_fetch:
                    records = self._fetch_orderbook_range(symbol, start_date, end_date)
                    total_records += records
                
                # FIXED: En modo incremental, success si ya había datos O se agregaron nuevos
                if total_records > 0:
                    success = True
                    log.info(f"✅ Successfully fetched {total_records} new orderbook snapshots for {symbol}")
                else:
                    # Verificar si ya había datos suficientes
                    min_date, max_date = self.get_orderbook_data_range(symbol)
                    if min_date and max_date:
                        # Ya hay datos en DB, los días "faltantes" simplemente están vacíos en la API
                        success = True
                        log.info(f"✅ Orderbook for {symbol} already complete - missing days have no data in API")
                    else:
                        # No hay datos en DB y no se pudieron obtener nuevos
                        success = False
                        log.warning(f"❌ No orderbook data exists and no new data could be fetched for {symbol}")
            
            return success
            
        except Exception as e:
            log.error(f"Orderbook ingestion failed for {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False
    
    def ingest_funding_rates(self, symbols: List[str], overwrite: bool = False) -> Dict[str, bool]:
        """
        LEGACY FUNCTION: Mantener compatibilidad pero usar nueva lógica
        """
        log.info(f"Starting funding rates ingestion for {len(symbols)} symbols (overwrite={overwrite})")
        
        results = {}
        
        for symbol in symbols:
            if "PERP_" not in symbol:
                log.warning(f"Skipping {symbol} - not a perpetual contract")
                results[symbol] = False
                continue
            
            log.info(f"Ingesting funding rates for {symbol}")
            
            try:
                # Usar nueva función inteligente
                success = self.ingest_funding_rates_single(symbol, overwrite=overwrite)
                results[symbol] = success
                
                # Small delay between symbols
                time.sleep(0.1)
                
            except Exception as e:
                log.error(f"Error ingesting funding rates for {symbol}: {e}")
                results[symbol] = False
        
        successful = sum(1 for success in results.values() if success)
        log.info(f"Funding rates ingestion completed: {successful}/{len(symbols)} symbols successful")
        
        return results
    
    def _fetch_ohlcv_range(self, symbol: str, start_date: datetime, end_date: datetime) -> int:
        """Fetch OHLCV data for a specific range with progress tracking"""
        log.info(f"Fetching OHLCV {symbol}: {start_date.date()} to {end_date.date()}")
        
        try:
            current_date = start_date.date()
            end_date_only = end_date.date()
            total_records = 0
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
                           desc=f"OHLCV {symbol.split('_')[-2] if '_' in symbol else symbol}", 
                           unit="days")
            
            while current_date <= end_date_only:
                try:
                    df = self.coinapi.get_ohlcv_for_date(symbol, current_date.isoformat())
                    
                    if not df.empty:
                        records_count = self._insert_ohlcv_data(symbol, df)
                        total_records += records_count
                        if records_count > 0:
                            successful_days += 1
                    
                    if use_tqdm:
                        pbar.set_postfix(Records=total_records, Days=successful_days)
                        pbar.update(1)
                    else:
                        if (current_date.day % 5 == 0):  # Log every 5 days
                            log.info(f"Progress: {successful_days} days, {total_records} records")
                    
                    # Rate limiting
                    time.sleep(0.2)
                    
                except Exception as e:
                    log.warning(f"Failed to fetch OHLCV for {symbol} on {current_date}: {e}")
                    if use_tqdm:
                        pbar.update(1)
                
                current_date += timedelta(days=1)
            
            if use_tqdm:
                pbar.close()
            
            log.info(f"OHLCV fetch completed: {total_records} records from {successful_days}/{total_days} days")
            return total_records
            
        except Exception as e:
            log.error(f"Error fetching OHLCV range for {symbol}: {e}")
            return 0
    
    def _fetch_orderbook_range(self, symbol: str, start_date, end_date) -> int:
        """Fetch orderbook data for a specific range - ROBUST VERSION"""
        log.info(f"Fetching orderbook {symbol}: {start_date} to {end_date}")
        
        try:
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False
            
            current_date = start_date
            total_records = 0
            successful_days = 0
            
            total_days = (end_date - start_date).days + 1
            
            if use_tqdm:
                pbar = tqdm(total=total_days, 
                           desc=f"Orderbook {symbol.split('_')[-2] if '_' in symbol else symbol}", 
                           unit="days")
            
            while current_date <= end_date:
                try:
                    df = self.coinapi.get_orderbook_for_date(symbol, current_date.isoformat())
                    
                    if not df.empty:
                        records_count = self._insert_orderbook_data(symbol, df)
                        total_records += records_count
                        if records_count > 0:
                            successful_days += 1
                    
                    if use_tqdm:
                        pbar.set_postfix(Records=total_records, Days=successful_days)
                        pbar.update(1)
                    else:
                        if (current_date.day % 3 == 0):  # Log every 3 days for orderbook
                            log.info(f"Orderbook progress: {successful_days} days, {total_records} records")
                    
                    # Rate limiting más agresivo para orderbook
                    time.sleep(0.5)
                    
                except Exception as e:
                    log.warning(f"Failed to fetch orderbook for {symbol} on {current_date}: {e}")
                    if use_tqdm:
                        pbar.update(1)
                
                current_date += timedelta(days=1)
            
            if use_tqdm:
                pbar.close()
            
            log.info(f"Orderbook fetch completed: {total_records} records from {successful_days}/{total_days} days")
            return total_records
            
        except Exception as e:
            log.error(f"Error fetching orderbook range for {symbol}: {e}")
            return 0
    
    def _insert_ohlcv_data(self, symbol: str, df: pd.DataFrame) -> int:
        """Insert OHLCV data into database - ROBUST VERSION"""
        if df.empty:
            return 0
        
        try:
            with db_manager.get_session() as session:
                records = []
                for timestamp, row in df.iterrows():
                   try:
                       record = {
                           'symbol': symbol,
                           'timestamp': timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp,
                           'open': float(row['open']),
                           'high': float(row['high']),
                           'low': float(row['low']),
                           'close': float(row['close']),
                           'volume': float(row['volume'])
                       }
                       
                       # Validar datos básicos
                       if (record['open'] > 0 and record['high'] > 0 and 
                           record['low'] > 0 and record['close'] > 0 and
                           record['high'] >= record['low'] and
                           record['volume'] >= 0):
                           records.append(record)
                       
                   except (ValueError, KeyError) as e:
                       log.debug(f"Skipping invalid OHLCV record for {symbol} at {timestamp}: {e}")
                       continue
               
                if records:
                   session.execute(text("""
                       INSERT INTO ohlcv (symbol, timestamp, open, high, low, close, volume)
                       VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
                       ON CONFLICT (symbol, timestamp) DO UPDATE SET
                           open = EXCLUDED.open,
                           high = EXCLUDED.high,
                           low = EXCLUDED.low,
                           close = EXCLUDED.close,
                           volume = EXCLUDED.volume
                   """), records)
                   session.commit()
               
            return len(records)
               
        except Exception as e:
            log.error(f"Error inserting OHLCV data for {symbol}: {e}")
            return 0
   
    def _insert_orderbook_data(self, symbol: str, df: pd.DataFrame) -> int:
       """Insert orderbook data ROBUSTO - Version 3.0"""
       if df.empty:
           return 0
       
       try:
           with db_manager.get_session() as session:
               records = []
               
               for timestamp, row in df.iterrows():
                   try:
                       record = {
                           'symbol': symbol,
                           'timestamp': timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
                       }
                       
                       # Agregar niveles bid/ask (1-10)
                       valid_level1 = False
                       
                       for i in range(1, 11):
                           for side in ['bid', 'ask']:
                               price_col = f'{side}{i}_price'
                               size_col = f'{side}{i}_size'
                               
                               price = row.get(price_col)
                               size = row.get(size_col)
                               
                               # Validar y convertir
                               try:
                                   if pd.notna(price) and price > 0:
                                       record[price_col] = float(price)
                                       if i == 1:  # Nivel 1
                                           valid_level1 = True
                                   else:
                                       record[price_col] = None
                                       
                                   if pd.notna(size) and size > 0:
                                       record[size_col] = float(size)
                                   else:
                                       record[size_col] = None
                               except (ValueError, TypeError):
                                   record[price_col] = None
                                   record[size_col] = None
                       
                       # Solo agregar si tiene al menos nivel 1 válido y spread correcto
                       if valid_level1:
                           bid1 = record.get('bid1_price')
                           ask1 = record.get('ask1_price')
                           if bid1 and ask1 and bid1 < ask1:
                               records.append(record)
                       
                   except Exception as e:
                       log.debug(f"Error procesando record {symbol} {timestamp}: {e}")
                       continue
               
               if records:
                   # Construir query dinámica
                   columns = list(records[0].keys())
                   placeholders = ', '.join([f':{col}' for col in columns])
                   columns_str = ', '.join(columns)
                   
                   # Para sobreescritura usamos UPSERT
                   update_clauses = []
                   for col in columns:
                       if col not in ['symbol', 'timestamp']:
                           update_clauses.append(f"{col} = EXCLUDED.{col}")
                   
                   insert_query = f"""
                       INSERT INTO orderbook ({columns_str})
                       VALUES ({placeholders})
                       ON CONFLICT (symbol, timestamp) DO UPDATE SET
                           {', '.join(update_clauses)}
                   """
                   
                   result = session.execute(text(insert_query), records)
                   session.commit()
                   
                   log.debug(f"✅ Insertados {len(records)} orderbook records para {symbol}")
                   return len(records)
               else:
                   log.debug(f"No hay records válidos para {symbol}")
                   return 0
                   
       except Exception as e:
           log.error(f"Error insertando orderbook {symbol}: {e}")
           import traceback
           log.debug(traceback.format_exc())
           return 0
   
    def _ingest_symbol_funding_rates(self, symbol: str, funding_df: pd.DataFrame) -> bool:
       """Insert funding rates for a specific symbol - ROBUST VERSION"""
       try:
           with db_manager.get_session() as session:
               records = []
               
               for _, row in funding_df.iterrows():
                   try:
                       timestamp = row['timestamp']
                       
                       # Handle timestamp conversion
                       if isinstance(timestamp, str):
                           timestamp = pd.to_datetime(timestamp, utc=True)
                       elif isinstance(timestamp, (int, float)):
                           timestamp = pd.to_datetime(timestamp, unit='s', utc=True)
                       elif not isinstance(timestamp, datetime):
                           timestamp = pd.to_datetime(timestamp, utc=True)
                       
                       # Convert to naive UTC datetime
                       if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                           timestamp = timestamp.tz_convert('UTC').tz_localize(None)
                       
                       if hasattr(timestamp, 'to_pydatetime'):
                           timestamp = timestamp.to_pydatetime()
                       
                       records.append({
                           'symbol': symbol,
                           'timestamp': timestamp,
                           'funding_rate': float(row['funding_rate']),
                           'collect_cycle': int(row.get('collect_cycle', 28800))
                       })
                       
                   except Exception as e:
                       log.debug(f"Skipping invalid funding rate record for {symbol}: {e}")
                       continue
               
               if records:
                   session.execute(text("""
                       INSERT INTO funding_rates (symbol, timestamp, funding_rate, collect_cycle)
                       VALUES (:symbol, :timestamp, :funding_rate, :collect_cycle)
                       ON CONFLICT (symbol, timestamp) DO UPDATE SET
                           funding_rate = EXCLUDED.funding_rate,
                           collect_cycle = EXCLUDED.collect_cycle
                   """), records)
                   session.commit()
                   
                   log.info(f"✅ Inserted {len(records)} funding rate records for {symbol}")
               
               return True
               
       except Exception as e:
           log.error(f"Error inserting funding rates for {symbol}: {e}")
           return False
    
    def _log_final_summary(self, results: Dict[str, Dict[str, bool]], data_types: List[str]):
        """Log final ingestion summary"""
        total_symbols = len(results)
        total_operations = total_symbols * len(data_types)
        successful_operations = sum(
            sum(1 for success in symbol_results.values() if success)
            for symbol_results in results.values()
        )
        
        log.info(f"📊 FINAL SUMMARY:")
        log.info(f"  Symbols processed: {total_symbols}")
        log.info(f"  Data types: {data_types}")
        log.info(f"  Total operations: {successful_operations}/{total_operations}")
        log.info(f"  Success rate: {successful_operations/total_operations*100:.1f}%")
        
        # Per data type summary
        for data_type in data_types:
            type_successes = sum(
                1 for symbol_results in results.values() 
                if symbol_results.get(data_type, False)
            )
            log.info(f"  {data_type.upper()}: {type_successes}/{total_symbols} symbols successful")
        
        # Show quick stats if available
        self._log_quick_stats(list(results.keys()), data_types)
    
    def _log_quick_stats(self, symbols: List[str], data_types: List[str]):
        """Log quick statistics from database"""
        try:
            with db_manager.get_session() as session:
                for data_type in data_types:
                    if data_type == 'ohlcv':
                        for symbol in symbols:
                            result = session.execute(text("""
                                SELECT COUNT(*) as count 
                                FROM ohlcv WHERE symbol = :symbol
                            """), {'symbol': symbol}).fetchone()
                            log.info(f"    {symbol}: {result.count:,} OHLCV records")
                    
                    elif data_type == 'orderbook':
                        for symbol in symbols:
                            result = session.execute(text("""
                                SELECT COUNT(*) as count 
                                FROM orderbook WHERE symbol = :symbol
                            """), {'symbol': symbol}).fetchone()
                            log.info(f"    {symbol}: {result.count:,} orderbook records")
                    
                    elif data_type == 'funding':
                        for symbol in symbols:
                            if "PERP_" in symbol:
                                result = session.execute(text("""
                                    SELECT COUNT(*) as count 
                                    FROM funding_rates WHERE symbol = :symbol
                                """), {'symbol': symbol}).fetchone()
                                log.info(f"    {symbol}: {result.count:,} funding records")
        except Exception as e:
            log.debug(f"Could not fetch quick stats: {e}")

# Global instance
data_ingestion = DataIngestion()