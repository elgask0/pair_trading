#!/usr/bin/env python3
"""
Data ingestion module - VERSION CON PARÁMETROS Y SOBREESCRITURA
Permite seleccionar qué tipo de datos ingestar y si sobreescribir
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
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
    """Main data ingestion class - CON PARÁMETROS Y SOBREESCRITURA"""
    
    def __init__(self):
        self.coinapi = coinapi_client
        self.mexc = mexc_client
    
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
        log.info(f"Updating symbol info for {symbol}")
        
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
            
            log.info(f"✅ Updated symbol info for {symbol}")
            return True
            
        except Exception as e:
            log.error(f"Failed to update symbol info for {symbol}: {e}")
            return False
    
    def ingest_ohlcv_data(self, symbol: str, overwrite: bool = False, days_back: int = 30) -> bool:
        """Ingest OHLCV data - CON OPCIÓN DE SOBREESCRITURA"""
        log.info(f"Starting OHLCV ingestion for {symbol} (overwrite={overwrite}, days_back={days_back})")
        
        try:
            # Get available data range from API
            available_start, available_end = self.coinapi.get_available_date_range(symbol)
            
            if not available_start or not available_end:
                log.warning(f"No available data range for {symbol}")
                return False
            
            if overwrite:
                # Calcular rango a eliminar y reemplazar
                start_fetch = available_end - timedelta(days=days_back)
                end_fetch = available_end
                
                log.info(f"OVERWRITE MODE: Fetching OHLCV {start_fetch.date()} to {end_fetch.date()}")
                
                # Eliminar datos existentes en el rango
                self.delete_existing_data(symbol, 'ohlcv', start_fetch.isoformat(), end_fetch.isoformat())
                
                # Fetch data para el rango especificado
                total_records = self._fetch_ohlcv_range(symbol, start_fetch, end_fetch)
                
            else:
                # Modo incremental (como antes)
                min_date, max_date = self.get_symbol_data_range(symbol)
                
                if min_date and max_date:
                    log.info(f"Existing OHLCV data for {symbol}: {min_date.date()} to {max_date.date()}")
                else:
                    log.info(f"No existing OHLCV data for {symbol}")
                
                # Determine what data to fetch
                ranges_to_fetch = []
                
                if not min_date:
                    # No existing data, fetch last days_back days
                    start_fetch = max(available_start, available_end - timedelta(days=days_back))
                    ranges_to_fetch.append((start_fetch, available_end))
                    log.info(f"No existing data - fetching last {days_back} days: {start_fetch.date()} to {available_end.date()}")
                else:
                    # Check for new data at the end
                    if available_end > max_date:
                        gap_start = max_date + timedelta(hours=1)  # Small overlap
                        ranges_to_fetch.append((gap_start, available_end))
                        log.info(f"New data available: {gap_start.date()} to {available_end.date()}")
                
                if not ranges_to_fetch:
                    log.info(f"No new OHLCV data to fetch for {symbol}")
                    return True
                
                # Fetch data for each range
                total_records = 0
                for start_date, end_date in ranges_to_fetch:
                    records = self._fetch_ohlcv_range(symbol, start_date, end_date)
                    total_records += records
            
            log.info(f"✅ Successfully fetched {total_records} OHLCV records for {symbol}")
            return total_records > 0
            
        except Exception as e:
            log.error(f"OHLCV ingestion failed for {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False
    
    def ingest_orderbook_data(self, symbol: str, overwrite: bool = False, days_back: int = 7) -> bool:
        """Ingest orderbook data - CON OPCIÓN DE SOBREESCRITURA"""
        log.info(f"Starting orderbook ingestion for {symbol} (overwrite={overwrite}, days_back={days_back})")
        
        try:
            # Get available range
            available_start, available_end = self.coinapi.get_available_date_range(symbol)
            
            if not available_start or not available_end:
                log.warning(f"No available orderbook data range for {symbol}")
                return False
            
            if overwrite:
                # Calcular rango a eliminar y reemplazar
                start_fetch = available_end.date() - timedelta(days=days_back)
                end_fetch = available_end.date()
                
                log.info(f"OVERWRITE MODE: Fetching orderbook {start_fetch} to {end_fetch}")
                
                # Eliminar datos existentes en el rango
                self.delete_existing_data(symbol, 'orderbook', 
                                        datetime.combine(start_fetch, datetime.min.time()).isoformat(),
                                        datetime.combine(end_fetch, datetime.max.time()).isoformat())
                
                # Fetch data para el rango especificado
                total_records = self._fetch_orderbook_range(symbol, start_fetch, end_fetch)
                
            else:
                # Modo incremental (como antes)
                min_date, max_date = self.get_orderbook_data_range(symbol)
                
                if min_date and max_date:
                    log.info(f"Existing orderbook data for {symbol}: {min_date.date()} to {max_date.date()}")
                else:
                    log.info(f"No existing orderbook data for {symbol}")
                
                # Determine new data range
                if max_date:
                    # Fetch only recent data
                    start_fetch = max(max_date.date(), available_end.date() - timedelta(days=3))
                    if start_fetch < available_end.date():
                        start_fetch = start_fetch + timedelta(days=1)  # Start from next day
                    else:
                        log.info(f"No new orderbook data for {symbol}")
                        return True
                else:
                    # No existing data, fetch last days_back days
                    start_fetch = available_end.date() - timedelta(days=days_back)
                
                end_fetch = available_end.date()
                
                if start_fetch > end_fetch:
                    log.info(f"No new orderbook data for {symbol}")
                    return True
                
                log.info(f"Fetching orderbook data: {start_fetch} to {end_fetch}")
                
                # Fetch orderbook data
                total_records = self._fetch_orderbook_range(symbol, start_fetch, end_fetch)
            
            log.info(f"✅ Successfully fetched {total_records} orderbook snapshots for {symbol}")
            return total_records > 0
            
        except Exception as e:
            log.error(f"Orderbook ingestion failed for {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False
    
    def ingest_funding_rates(self, symbols: List[str], overwrite: bool = False) -> Dict[str, bool]:
        """Ingest funding rates - CON OPCIÓN DE SOBREESCRITURA"""
        log.info(f"Starting funding rates ingestion for {len(symbols)} symbols (overwrite={overwrite})")
        
        results = {}
        
        for symbol in symbols:
            if "PERP_" not in symbol:
                log.warning(f"Skipping {symbol} - not a perpetual contract")
                results[symbol] = False
                continue
            
            log.info(f"Ingesting funding rates for {symbol}")
            
            try:
                if overwrite:
                    # Eliminar todos los funding rates existentes
                    self.delete_existing_data(symbol, 'funding')
                
                # Get funding rate history from MEXC
                funding_data = self.mexc.get_funding_rate_history(symbol)
                
                if funding_data.empty:
                    log.warning(f"No funding rate data for {symbol}")
                    results[symbol] = False
                    continue
                
                # Insert funding rates
                success = self._ingest_symbol_funding_rates(symbol, funding_data)
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
            from tqdm import tqdm
            
            total_days = (end_date_only - current_date).days + 1
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
                    
                    pbar.set_postfix(Records=total_records, Days=successful_days)
                    pbar.update(1)
                    
                    # Rate limiting
                    time.sleep(0.2)
                    
                except Exception as e:
                    log.warning(f"Failed to fetch OHLCV for {symbol} on {current_date}: {e}")
                    pbar.update(1)
                
                current_date += timedelta(days=1)
            
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
            from tqdm import tqdm
            
            current_date = start_date
            total_records = 0
            successful_days = 0
            
            total_days = (end_date - start_date).days + 1
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
                    
                    pbar.set_postfix(Records=total_records, Days=successful_days)
                    pbar.update(1)
                    
                    # Rate limiting más agresivo para orderbook
                    time.sleep(0.5)
                    
                except Exception as e:
                    log.warning(f"Failed to fetch orderbook for {symbol} on {current_date}: {e}")
                    pbar.update(1)
                
                current_date += timedelta(days=1)
            
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
                        log.warning(f"Skipping invalid OHLCV record for {symbol} at {timestamp}: {e}")
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
                        log.warning(f"Error procesando record {symbol} {timestamp}: {e}")
                        continue
                
                if records:
                    # Construir query dinámica
                    columns = list(records[0].keys())
                    placeholders = ', '.join([f':{col}' for col in columns])
                    columns_str = ', '.join(columns)
                    
                    # Para sobreescritura usamos UPSERT
                    insert_query = f"""
                        INSERT INTO orderbook ({columns_str})
                        VALUES ({placeholders})
                        ON CONFLICT (symbol, timestamp) DO UPDATE SET
                            bid1_price = EXCLUDED.bid1_price,
                            bid1_size = EXCLUDED.bid1_size,
                            ask1_price = EXCLUDED.ask1_price,
                            ask1_size = EXCLUDED.ask1_size,
                            bid2_price = EXCLUDED.bid2_price,
                            bid2_size = EXCLUDED.bid2_size,
                            ask2_price = EXCLUDED.ask2_price,
                            ask2_size = EXCLUDED.ask2_size,
                            bid3_price = EXCLUDED.bid3_price,
                            bid3_size = EXCLUDED.bid3_size,
                            ask3_price = EXCLUDED.ask3_price,
                            ask3_size = EXCLUDED.ask3_size,
                            bid4_price = EXCLUDED.bid4_price,
                            bid4_size = EXCLUDED.bid4_size,
                            ask4_price = EXCLUDED.ask4_price,
                            ask4_size = EXCLUDED.ask4_size,
                            bid5_price = EXCLUDED.bid5_price,
                            bid5_size = EXCLUDED.bid5_size,
                            ask5_price = EXCLUDED.ask5_price,
                            ask5_size = EXCLUDED.ask5_size,
                            bid6_price = EXCLUDED.bid6_price,
                            bid6_size = EXCLUDED.bid6_size,
                            ask6_price = EXCLUDED.ask6_price,
                            ask6_size = EXCLUDED.ask6_size,
                            bid7_price = EXCLUDED.bid7_price,
                            bid7_size = EXCLUDED.bid7_size,
                            ask7_price = EXCLUDED.ask7_price,
                            ask7_size = EXCLUDED.ask7_size,
                            bid8_price = EXCLUDED.bid8_price,
                            bid8_size = EXCLUDED.bid8_size,
                            ask8_price = EXCLUDED.ask8_price,
                            ask8_size = EXCLUDED.ask8_size,
                            bid9_price = EXCLUDED.bid9_price,
                            bid9_size = EXCLUDED.bid9_size,
                            ask9_price = EXCLUDED.ask9_price,
                            ask9_size = EXCLUDED.ask9_size,
                            bid10_price = EXCLUDED.bid10_price,
                            bid10_size = EXCLUDED.bid10_size,
                            ask10_price = EXCLUDED.ask10_price,
                            ask10_size = EXCLUDED.ask10_size
                    """
                    
                    result = session.execute(text(insert_query), records)
                    session.commit()
                    
                    log.info(f"✅ Insertados {len(records)} orderbook records para {symbol}")
                    return len(records)
                else:
                    log.warning(f"No hay records válidos para {symbol}")
                    return 0
                    
        except Exception as e:
            log.error(f"Error insertando orderbook {symbol}: {e}")
            import traceback
            log.error(traceback.format_exc())
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
                        log.warning(f"Skipping invalid funding rate record for {symbol}: {e}")
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
    
    def ingest_symbol_data(self, symbol: str, data_types: List[str] = None, overwrite: bool = False, 
                          days_back: int = 7) -> bool:
        """Ingest data for a symbol - CON PARÁMETROS DE CONTROL"""
        
        # Tipos de datos disponibles
        available_types = ['ohlcv', 'orderbook', 'funding']
        
        # Si no se especifican tipos, usar todos
        if data_types is None:
            data_types = available_types
        
        # Validar tipos
        data_types = [dt for dt in data_types if dt in available_types]
        
        if not data_types:
            log.error(f"No valid data types specified")
            return False
        
        log.info(f"Starting selective ingestion for {symbol}")
        log.info(f"Data types: {data_types}")
        