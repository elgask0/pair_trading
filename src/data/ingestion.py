#!/usr/bin/env python3
"""
Data ingestion module - FIXED VERSION
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
    """Main data ingestion class"""
    
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
    
    def get_funding_data_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get existing funding data range for symbol"""
        with db_manager.get_session() as session:
            result = session.execute(text("""
                SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date
                FROM funding_rates 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            return result.min_date, result.max_date
    
    def update_symbol_info(self, symbol: str) -> bool:
        """Update symbol information"""
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
            
            log.info(f"Updated symbol info for {symbol}")
            return True
            
        except Exception as e:
            log.error(f"Failed to update symbol info for {symbol}: {e}")
            return False
    
    def ingest_ohlcv_data(self, symbol: str) -> bool:
        """Ingest OHLCV data incrementally"""
        log.info(f"Starting incremental OHLCV ingestion for {symbol}")
        
        try:
            # Get existing data range
            min_date, max_date = self.get_symbol_data_range(symbol)
            
            if min_date and max_date:
                log.info(f"Existing OHLCV data for {symbol}: {min_date} to {max_date}")
            else:
                log.info(f"No existing OHLCV data for {symbol}")
            
            # Get available data range from API
            available_start, available_end = self.coinapi.get_available_date_range(symbol)
            
            if not available_start or not available_end:
                log.warning(f"No available data range for {symbol}")
                return False
            
            log.info(f"Available data for {symbol}: {available_start} to {available_end}")
            
            # Determine what data to fetch
            ranges_to_fetch = []
            
            if not min_date:
                # No existing data, fetch everything
                ranges_to_fetch.append((available_start, available_end))
            else:
                # Check for gaps at beginning and end
                if available_start < min_date:
                    gap_end = min_date - timedelta(minutes=1)
                    ranges_to_fetch.append((available_start, gap_end))
                    log.info(f"Gap at beginning: {available_start} to {gap_end}")
                
                if available_end > max_date:
                    gap_start = max_date + timedelta(minutes=1)
                    ranges_to_fetch.append((gap_start, available_end))
                    log.info(f"New data at end: {gap_start} to {available_end}")
            
            if not ranges_to_fetch:
                log.info(f"No new OHLCV data to fetch for {symbol}")
                return True
            
            # Fetch data for each range
            total_new_records = 0
            for start_date, end_date in ranges_to_fetch:
                records = self._fetch_ohlcv_range(symbol, start_date, end_date)
                total_new_records += records
            
            log.info(f"Successfully fetched {total_new_records} new OHLCV records for {symbol}")
            return True
            
        except Exception as e:
            log.error(f"OHLCV ingestion failed for {symbol}: {e}")
            return False
    
    def _fetch_ohlcv_range(self, symbol: str, start_date: datetime, end_date: datetime) -> int:
        """Fetch OHLCV data for a specific range"""
        log.info(f"Fetching OHLCV {symbol}: {start_date.date()} to {end_date.date()} ({(end_date - start_date).days} days)")
        
        try:
            # Fetch data day by day with progress
            current_date = start_date.date()
            end_date_only = end_date.date()
            total_records = 0
            
            from tqdm import tqdm
            
            pbar = tqdm(total=(end_date_only - current_date).days + 1, 
                       desc=f"OHLCV {symbol.split('_')[-2] if '_' in symbol else symbol}", 
                       unit="days")
            
            while current_date <= end_date_only:
                try:
                    df = self.coinapi.get_ohlcv_for_date(symbol, current_date.isoformat())
                    
                    if not df.empty:
                        records_count = self._insert_ohlcv_data(symbol, df)
                        total_records += records_count
                    
                    pbar.set_postfix(Records=total_records)
                    pbar.update(1)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    log.warning(f"Failed to fetch OHLCV for {symbol} on {current_date}: {e}")
                
                current_date += timedelta(days=1)
            
            pbar.close()
            return total_records
            
        except Exception as e:
            log.error(f"Error fetching OHLCV range for {symbol}: {e}")
            return 0
    
    def _insert_ohlcv_data(self, symbol: str, df: pd.DataFrame) -> int:
        """Insert OHLCV data into database - FIXED VERSION"""
        if df.empty:
            return 0
        
        try:
            with db_manager.get_session() as session:
                records = []
                for timestamp, row in df.iterrows():
                    # Ensure all values are properly converted
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
                        records.append(record)
                    except (ValueError, KeyError) as e:
                        log.warning(f"Skipping invalid OHLCV record for {symbol} at {timestamp}: {e}")
                        continue
                
                if records:
                    # Batch insert with conflict resolution
                    session.execute(text("""
                        INSERT INTO ohlcv (symbol, timestamp, open, high, low, close, volume)
                        VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
                        ON CONFLICT (symbol, timestamp) DO NOTHING
                    """), records)
                
                return len(records)
                
        except Exception as e:
            log.error(f"Error inserting OHLCV data for {symbol}: {e}")
            return 0
    
    def ingest_orderbook_data(self, symbol: str) -> bool:
        """Ingest orderbook data incrementally"""
        log.info(f"Starting incremental orderbook ingestion for {symbol}")
        
        try:
            # Get existing orderbook data range
            with db_manager.get_session() as session:
                result = session.execute(text("""
                    SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date
                    FROM orderbook 
                    WHERE symbol = :symbol
                """), {'symbol': symbol}).fetchone()
                
                min_date, max_date = result.min_date, result.max_date
            
            if min_date and max_date:
                log.info(f"Existing orderbook data for {symbol}: {min_date} to {max_date}")
            else:
                log.info(f"No existing orderbook data for {symbol}")
            
            # Get available range and determine what to fetch
            available_start, available_end = self.coinapi.get_available_date_range(symbol)
            
            if not available_start or not available_end:
                log.warning(f"No available orderbook data range for {symbol}")
                return False
            
            # Determine new data range
            if max_date:
                start_fetch = max_date.date() + timedelta(days=1)
            else:
                start_fetch = available_start.date()
            
            end_fetch = available_end.date()
            
            if start_fetch > end_fetch:
                log.info(f"No new orderbook data for {symbol}")
                return True
            
            log.info(f"New orderbook data: {start_fetch} to {end_fetch}")
            
            # Fetch orderbook data
            total_records = self._fetch_orderbook_range(symbol, start_fetch, end_fetch)
            log.info(f"Successfully fetched {total_records} new orderbook snapshots for {symbol}")
            return True
            
        except Exception as e:
            log.error(f"Orderbook ingestion failed for {symbol}: {e}")
            return False
    
    def _fetch_orderbook_range(self, symbol: str, start_date, end_date) -> int:
        """Fetch orderbook data for a specific range"""
        log.info(f"Fetching orderbook {symbol}: {start_date} to {end_date} ({(end_date - start_date).days} days)")
        
        try:
            from tqdm import tqdm
            
            current_date = start_date
            total_records = 0
            
            pbar = tqdm(total=(end_date - start_date).days + 1, 
                       desc=f"Orderbook {symbol.split('_')[-2] if '_' in symbol else symbol}", 
                       unit="days")
            
            while current_date <= end_date:
                try:
                    df = self.coinapi.get_orderbook_for_date(symbol, current_date.isoformat())
                    
                    if not df.empty:
                        records_count = self._insert_orderbook_data(symbol, df)
                        total_records += records_count
                    
                    pbar.set_postfix(Records=total_records)
                    pbar.update(1)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    log.error(f"Error fetching orderbook {symbol} for {current_date}: {e}")
                
                current_date += timedelta(days=1)
            
            pbar.close()
            return total_records
            
        except Exception as e:
            log.error(f"Error fetching orderbook range for {symbol}: {e}")
            return 0
    
    def _insert_orderbook_data(self, symbol: str, df: pd.DataFrame) -> int:
        """Insert orderbook data into database - FIXED VERSION"""
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
                        
                        # Add bid/ask levels (up to 10 levels)
                        for i in range(1, 11):
                            for side in ['bid', 'ask']:
                                price_col = f'{side}_{i}_price'
                                size_col = f'{side}_{i}_size'
                                
                                price = row.get(price_col)
                                size = row.get(size_col)
                                # Convert to float if valid, otherwise None
                                try:
                                    record[f'{side}{i}_price'] = float(price) if pd.notna(price) and price != 0 else None
                                    record[f'{side}{i}_size'] = float(size) if pd.notna(size) and size != 0 else None
                                except (ValueError, TypeError):
                                    record[f'{side}{i}_price'] = None
                                    record[f'{side}{i}_size'] = None
                        
                        records.append(record)
                        
                    except Exception as e:
                        log.warning(f"Skipping invalid orderbook record for {symbol} at {timestamp}: {e}")
                        continue
                
                if records:
                    # Build dynamic insert query
                    columns = list(records[0].keys())
                    placeholders = ', '.join([f':{col}' for col in columns])
                    columns_str = ', '.join(columns)
                    
                    insert_query = f"""
                        INSERT INTO orderbook ({columns_str})
                        VALUES ({placeholders})
                        ON CONFLICT (symbol, timestamp) DO NOTHING
                    """
                    
                    session.execute(text(insert_query), records)
                
                return len(records)
                
        except Exception as e:
            log.error(f"Error inserting orderbook data for {symbol}: {e}")
            return 0
    
    def ingest_funding_rates(self, symbols: List[str]) -> Dict[str, bool]:
        """Ingest funding rates for perpetual symbols"""
        log.info(f"Starting funding rates ingestion for {len(symbols)} symbols")
        
        results = {}
        
        for symbol in symbols:
            if "PERP_" not in symbol:
                log.warning(f"Skipping {symbol} - not a perpetual contract")
                results[symbol] = False
                continue
            
            log.info(f"Ingesting funding rates for {symbol}")
            
            try:
                # Get funding rate history from MEXC
                funding_data = self.mexc.get_funding_rate_history(symbol)
                
                if funding_data.empty:
                    log.warning(f"No funding rate data for {symbol}")
                    results[symbol] = False
                    continue
                
                # Insert funding rates
                success = self._ingest_symbol_funding_rates(symbol, funding_data)
                results[symbol] = success
                
            except Exception as e:
                log.error(f"Error ingesting funding rates for {symbol}: {e}")
                results[symbol] = False
        
        successful = sum(1 for success in results.values() if success)
        log.info(f"Funding rates ingestion completed: {successful}/{len(symbols)} symbols successful")
        
        return results
    
    def _ingest_symbol_funding_rates(self, symbol: str, funding_df: pd.DataFrame) -> bool:
        """Insert funding rates for a specific symbol - FIXED VERSION"""
        try:
            with db_manager.get_session() as session:
                records = []
                
                for _, row in funding_df.iterrows():
                    try:
                        # Handle timestamp conversion carefully
                        timestamp = row['timestamp']
                        
                        # Ensure we have a proper datetime object
                        if isinstance(timestamp, str):
                            timestamp = pd.to_datetime(timestamp, utc=True)
                        elif isinstance(timestamp, (int, float)):
                            timestamp = pd.to_datetime(timestamp, unit='s', utc=True)
                        elif not isinstance(timestamp, datetime):
                            timestamp = pd.to_datetime(timestamp, utc=True)
                        
                        # Convert to naive UTC datetime for database storage
                        if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                            timestamp = timestamp.tz_convert('UTC').tz_localize(None)
                        
                        # Convert pandas timestamp to python datetime
                        if hasattr(timestamp, 'to_pydatetime'):
                            timestamp = timestamp.to_pydatetime()
                        
                        records.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'funding_rate': float(row['funding_rate']),
                            'collect_cycle': int(row.get('collect_cycle', 28800))  # Default 8 hours
                        })
                        
                    except Exception as e:
                        log.warning(f"Skipping invalid funding rate record for {symbol}: {e}")
                        continue
                
                if records:
                    session.execute(text("""
                        INSERT INTO funding_rates (symbol, timestamp, funding_rate, collect_cycle)
                        VALUES (:symbol, :timestamp, :funding_rate, :collect_cycle)
                        ON CONFLICT (symbol, timestamp) DO NOTHING
                    """), records)
                    
                    log.info(f"Inserted {len(records)} funding rate records for {symbol}")
                
                return True
                
        except Exception as e:
            log.error(f"Error inserting funding rates for {symbol}: {e}")
            return False
    
    def ingest_symbol_data(self, symbol: str, funding_only: bool = False) -> bool:
        """Ingest complete data for a symbol"""
        log.info(f"Starting complete ingestion for {symbol}")
        
        success = True
        
        try:
            # Update symbol info
            if not self.update_symbol_info(symbol):
                log.warning(f"Failed to update symbol info for {symbol}")
            
            if not funding_only:
                # Ingest OHLCV data
                if not self.ingest_ohlcv_data(symbol):
                    log.error(f"OHLCV ingestion failed for {symbol}")
                    success = False
                
                # Ingest orderbook data
                if not self.ingest_orderbook_data(symbol):
                    log.error(f"Orderbook ingestion failed for {symbol}")
                    success = False
            
            # Ingest funding rates for perpetual contracts
            if "PERP_" in symbol:
                funding_results = self.ingest_funding_rates([symbol])
                if not funding_results.get(symbol, False):
                    log.warning(f"Funding rates ingestion failed for {symbol}")
                    # Don't mark as complete failure for funding rates alone
            
            if success:
                log.info(f"Successfully processed {symbol}")
            else:
                log.error(f"Some data types failed for {symbol}")
            
            return success
            
        except Exception as e:
            log.error(f"Complete ingestion failed for {symbol}: {e}")
            return False
    
    def ingest_all_symbols(self, funding_only: bool = False) -> bool:
        """Ingest data for all configured symbols"""
        log.info(f"Starting complete incremental data ingestion for all symbols")
        
        try:
            # Get active symbols
            active_pairs = settings.get_active_pairs()
            symbols = []
            for pair in active_pairs:
                symbols.extend([pair.symbol1, pair.symbol2])
            symbols = list(set(symbols))  # Remove duplicates
            
            if not symbols:
                log.error("No active symbols configured")
                return False
            
            success_count = 0
            
            for i, symbol in enumerate(symbols, 1):
                log.info(f"\n=== Processing {symbol} ===")
                
                if self.ingest_symbol_data(symbol, funding_only):
                    success_count += 1
                    log.info(f"Successfully processed {symbol} ({i}/{len(symbols)})")
                else:
                    log.error(f"Failed to process {symbol} ({i}/{len(symbols)})")
            
            log.info(f"Complete incremental ingestion finished: {success_count}/{len(symbols)} symbols successful")
            return success_count == len(symbols)
            
        except Exception as e:
            log.error(f"Bulk ingestion failed: {e}")
            return False


# Global data ingestion instance
data_ingestion = DataIngestion()