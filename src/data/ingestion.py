from datetime import datetime, timedelta, date  # Agregar 'date' aquí
from typing import List, Optional
import pandas as pd
from sqlalchemy import and_, func
from tqdm import tqdm
from src.data.coinapi_client import coinapi_client
from src.database.connection import db_manager
from src.database.models import OHLCV, Symbol, SymbolInfo, Orderbook
from src.utils.logger import log
from src.utils.exceptions import DataValidationError, APIError
from config.settings import settings

class DataIngestion:
    """Handle complete incremental data ingestion from CoinAPI"""
    
    def __init__(self):
        self.coinapi = coinapi_client
    
    def update_symbol_info(self, symbol: str) -> bool:
        """Update symbol metadata in database"""
        log.info(f"Updating symbol info for {symbol}")
        
        try:
            # Get symbol info from CoinAPI
            symbol_data = self.coinapi.get_symbol_info(symbol)
            
            with db_manager.get_session() as session:
                # Check if symbol info already exists
                existing = session.query(SymbolInfo).filter(SymbolInfo.symbol_id == symbol).first()
                
                if existing:
                    # Update existing
                    existing.exchange_id = symbol_data.get('exchange_id')
                    existing.symbol_type = symbol_data.get('symbol_type')
                    existing.asset_id_base = symbol_data.get('asset_id_base')
                    existing.asset_id_quote = symbol_data.get('asset_id_quote')
                    existing.data_start = self._parse_datetime(symbol_data.get('data_start'))
                    existing.data_end = self._parse_datetime(symbol_data.get('data_end'))
                    existing.data_quote_start = self._parse_datetime(symbol_data.get('data_quote_start'))
                    existing.data_quote_end = self._parse_datetime(symbol_data.get('data_quote_end'))
                    existing.data_orderbook_start = self._parse_datetime(symbol_data.get('data_orderbook_start'))
                    existing.data_orderbook_end = self._parse_datetime(symbol_data.get('data_orderbook_end'))
                    existing.data_trade_start = self._parse_datetime(symbol_data.get('data_trade_start'))
                    existing.data_trade_end = self._parse_datetime(symbol_data.get('data_trade_end'))
                    
                    log.info(f"Updated symbol info for {symbol}")
                else:
                    # Create new
                    new_info = SymbolInfo(
                        symbol_id=symbol,
                        exchange_id=symbol_data.get('exchange_id'),
                        symbol_type=symbol_data.get('symbol_type'),
                        asset_id_base=symbol_data.get('asset_id_base'),
                        asset_id_quote=symbol_data.get('asset_id_quote'),
                        data_start=self._parse_datetime(symbol_data.get('data_start')),
                        data_end=self._parse_datetime(symbol_data.get('data_end')),
                        data_quote_start=self._parse_datetime(symbol_data.get('data_quote_start')),
                        data_quote_end=self._parse_datetime(symbol_data.get('data_quote_end')),
                        data_orderbook_start=self._parse_datetime(symbol_data.get('data_orderbook_start')),
                        data_orderbook_end=self._parse_datetime(symbol_data.get('data_orderbook_end')),
                        data_trade_start=self._parse_datetime(symbol_data.get('data_trade_start')),
                        data_trade_end=self._parse_datetime(symbol_data.get('data_trade_end'))
                    )
                    session.add(new_info)
                    log.info(f"Created symbol info for {symbol}")
            
            return True
            
        except Exception as e:
            log.error(f"Failed to update symbol info for {symbol}: {e}")
            return False
    
    def ingest_ohlcv_data(self, symbol: str) -> bool:
        """Ingest OHLCV data incrementally (only missing data)"""
        log.info(f"Starting incremental OHLCV ingestion for {symbol}")
        
        try:
            # Check if symbol exists
            with db_manager.get_session() as session:
                db_symbol = session.query(Symbol).filter(Symbol.symbol == symbol).first()
                if not db_symbol:
                    log.error(f"Symbol {symbol} not found in database.")
                    return False
            
            # Get existing data range
            last_timestamp = self._get_last_ohlcv_timestamp(symbol)
            first_timestamp = self._get_first_ohlcv_timestamp(symbol)
            
            if last_timestamp:
                log.info(f"Existing OHLCV data for {symbol}: {first_timestamp} to {last_timestamp}")
            
            # Get available data range from CoinAPI
            available_start, available_end = self.coinapi.get_available_date_range(symbol)
            
            # Determine what data we need to fetch
            fetch_ranges = []
            
            if not last_timestamp:
                # No existing data, fetch everything
                fetch_ranges.append((available_start, available_end))
                log.info(f"No existing data, will fetch all: {available_start} to {available_end}")
            else:
                # Check for gaps at the beginning
                if first_timestamp > available_start:
                    gap_end = first_timestamp - timedelta(days=1)
                    fetch_ranges.append((available_start, gap_end))
                    log.info(f"Gap at beginning: {available_start} to {gap_end}")
                
                # Check for new data at the end
                if last_timestamp < available_end:
                    new_start = last_timestamp + timedelta(days=1)
                    fetch_ranges.append((new_start, available_end))
                    log.info(f"New data at end: {new_start} to {available_end}")
            
            if not fetch_ranges:
                log.info(f"No new OHLCV data needed for {symbol}")
                return True
            
            # Fetch missing data
            total_records = 0
            for start_date, end_date in fetch_ranges:
                records = self._fetch_ohlcv_range(symbol, start_date, end_date)
                total_records += records
            
            log.info(f"Successfully fetched {total_records} new OHLCV records for {symbol}")
            return True
            
        except Exception as e:
            log.error(f"Failed to ingest OHLCV data for {symbol}: {e}")
            return False
    
    def ingest_orderbook_data(self, symbol: str) -> bool:
        """Ingest orderbook data incrementally (all available data)"""
        log.info(f"Starting incremental orderbook ingestion for {symbol}")
        
        try:
            # Check if symbol exists
            with db_manager.get_session() as session:
                db_symbol = session.query(Symbol).filter(Symbol.symbol == symbol).first()
                if not db_symbol:
                    log.error(f"Symbol {symbol} not found in database.")
                    return False
            
            # Get existing orderbook data range
            last_ob_timestamp = self._get_last_orderbook_timestamp(symbol)
            first_ob_timestamp = self._get_first_orderbook_timestamp(symbol)
            
            if last_ob_timestamp:
                log.info(f"Existing orderbook data for {symbol}: {first_ob_timestamp} to {last_ob_timestamp}")
            
            # Get symbol info to know orderbook availability
            with db_manager.get_session() as session:
                symbol_info = session.query(SymbolInfo).filter(SymbolInfo.symbol_id == symbol).first()
                if not symbol_info or not symbol_info.data_orderbook_start:
                    log.warning(f"No orderbook availability info for {symbol}")
                    return True
                
                available_start = symbol_info.data_orderbook_start.date()
                available_end = min(
                    symbol_info.data_orderbook_end.date() if symbol_info.data_orderbook_end else datetime.now().date(),
                    datetime.now().date()
                )
            
            # Determine what data we need to fetch
            fetch_ranges = []
            
            if not last_ob_timestamp:
                # No existing data, fetch everything available
                fetch_ranges.append((available_start, available_end))
                log.info(f"No existing orderbook data, will fetch all: {available_start} to {available_end}")
            else:
                last_date = last_ob_timestamp.date()
                
                # Check for new data
                if last_date < available_end:
                    new_start = last_date + timedelta(days=1)
                    fetch_ranges.append((new_start, available_end))
                    log.info(f"New orderbook data: {new_start} to {available_end}")
            
            if not fetch_ranges:
                log.info(f"No new orderbook data needed for {symbol}")
                return True
            
            # Fetch missing orderbook data
            total_records = 0
            for start_date, end_date in fetch_ranges:
                records = self._fetch_orderbook_range(symbol, start_date, end_date)
                total_records += records
            
            log.info(f"Successfully fetched {total_records} new orderbook snapshots for {symbol}")
            return True
            
        except Exception as e:
            log.error(f"Failed to ingest orderbook data for {symbol}: {e}")
            return False
    
    def _fetch_ohlcv_range(self, symbol: str, start_date: datetime, end_date: datetime) -> int:
        """Fetch OHLCV data for a specific date range"""
        current_date = start_date.date()
        end_date = end_date.date()
        total_days = (end_date - current_date).days + 1
        total_records = 0
        
        log.info(f"Fetching OHLCV {symbol}: {current_date} to {end_date} ({total_days} days)")
        
        with tqdm(total=total_days, desc=f"OHLCV {symbol}", unit="days") as pbar:
            while current_date <= end_date:
                date_str = current_date.isoformat()
                
                try:
                    # Check if we already have data for this date
                    if self._has_ohlcv_for_date(symbol, current_date):
                        current_date += timedelta(days=1)
                        pbar.update(1)
                        continue
                    
                    # Fetch data
                    daily_df = self.coinapi.get_ohlcv_for_date(symbol, date_str)
                    
                    if not daily_df.empty:
                        # Process and insert
                        df = self._process_ohlcv_dataframe(daily_df)
                        records = self._insert_ohlcv_data(df, symbol)
                        total_records += records
                    
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    pbar.set_postfix({'Records': total_records})
                    
                except Exception as e:
                    log.error(f"Error fetching OHLCV {symbol} for {date_str}: {e}")
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    continue
        
        return total_records
    
    def _fetch_orderbook_range(self, symbol: str, start_date: date, end_date: date) -> int:
        """Fetch orderbook data for a specific date range"""
        current_date = start_date
        total_days = (end_date - current_date).days + 1
        total_records = 0
        
        log.info(f"Fetching orderbook {symbol}: {current_date} to {end_date} ({total_days} days)")
        
        with tqdm(total=total_days, desc=f"Orderbook {symbol}", unit="days") as pbar:
            while current_date <= end_date:
                date_str = current_date.isoformat()
                
                try:
                    # Check if we already have data for this date
                    if self._has_orderbook_for_date(symbol, current_date):
                        current_date += timedelta(days=1)
                        pbar.update(1)
                        continue
                    
                    # Fetch data
                    daily_df = self.coinapi.get_orderbook_for_date(symbol, date_str)
                    
                    if not daily_df.empty:
                        records = self._insert_orderbook_data(daily_df, symbol)
                        total_records += records
                    
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    pbar.set_postfix({'Records': total_records})
                    
                except Exception as e:
                    log.error(f"Error fetching orderbook {symbol} for {date_str}: {e}")
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    continue
        
        return total_records
    
    def ingest_symbol_data(self, symbol: str) -> bool:
        """Ingest all data for a single symbol"""
        log.info(f"Starting complete ingestion for {symbol}")
        
        # Step 1: Update symbol info
        if not self.update_symbol_info(symbol):
            log.error(f"Failed to update symbol info for {symbol}")
            return False
        
        # Step 2: Ingest OHLCV data (incremental)
        if not self.ingest_ohlcv_data(symbol):
            log.error(f"Failed to ingest OHLCV data for {symbol}")
            return False
        
        # Step 3: Ingest orderbook data (incremental)
        try:
            self.ingest_orderbook_data(symbol)
        except Exception as e:
            log.warning(f"Orderbook ingestion failed for {symbol}: {e}")
        
        log.info(f"Successfully processed {symbol}")
        return True
    
    def ingest_all_symbols(self) -> bool:
        """Complete incremental ingestion for all symbols"""
        log.info("Starting complete incremental data ingestion for all symbols")
        
        symbols_to_process = []
        for pair_config in settings.get_active_pairs():
            symbols_to_process.extend([pair_config.symbol1, pair_config.symbol2])
        
        symbols_to_process = list(set(symbols_to_process))
        
        success_count = 0
        total_symbols = len(symbols_to_process)
        
        for symbol in symbols_to_process:
            log.info(f"\n=== Processing {symbol} ===")
            
            if self.ingest_symbol_data(symbol):
                success_count += 1
                log.info(f"Successfully processed {symbol} ({success_count}/{total_symbols})")
            else:
                log.error(f"Failed to process {symbol}")
        
        log.info(f"Complete incremental ingestion finished: {success_count}/{total_symbols} symbols successful")
        return success_count == total_symbols
    
    # Helper methods
    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse datetime string from CoinAPI"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).replace(tzinfo=None)
        except:
            return None
    
    def _get_last_ohlcv_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get the last OHLCV timestamp"""
        with db_manager.get_session() as session:
            result = session.query(OHLCV.timestamp)\
                          .filter(OHLCV.symbol == symbol)\
                          .order_by(OHLCV.timestamp.desc())\
                          .first()
            return result[0] if result else None
    
    def _get_first_ohlcv_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get the first OHLCV timestamp"""
        with db_manager.get_session() as session:
            result = session.query(OHLCV.timestamp)\
                          .filter(OHLCV.symbol == symbol)\
                          .order_by(OHLCV.timestamp.asc())\
                          .first()
            return result[0] if result else None
    
    def _get_last_orderbook_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get the last orderbook timestamp"""
        with db_manager.get_session() as session:
            result = session.query(Orderbook.timestamp)\
                          .filter(Orderbook.symbol == symbol)\
                          .order_by(Orderbook.timestamp.desc())\
                          .first()
            return result[0] if result else None
    
    def _get_first_orderbook_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get the first orderbook timestamp"""
        with db_manager.get_session() as session:
            result = session.query(Orderbook.timestamp)\
                          .filter(Orderbook.symbol == symbol)\
                          .order_by(Orderbook.timestamp.asc())\
                          .first()
            return result[0] if result else None
    
    def _has_ohlcv_for_date(self, symbol: str, target_date: date) -> bool:
        """Check if we have OHLCV data for a specific date"""
        with db_manager.get_session() as session:
            start_of_day = datetime.combine(target_date, datetime.min.time())
            end_of_day = datetime.combine(target_date, datetime.max.time())
            
            count = session.query(OHLCV)\
                         .filter(OHLCV.symbol == symbol)\
                         .filter(OHLCV.timestamp >= start_of_day)\
                         .filter(OHLCV.timestamp <= end_of_day)\
                         .count()
            return count > 0
    
    def _has_orderbook_for_date(self, symbol: str, target_date: date) -> bool:
        """Check if we have orderbook data for a specific date"""
        with db_manager.get_session() as session:
            start_of_day = datetime.combine(target_date, datetime.min.time())
            end_of_day = datetime.combine(target_date, datetime.max.time())
            
            count = session.query(Orderbook)\
                         .filter(Orderbook.symbol == symbol)\
                         .filter(Orderbook.timestamp >= start_of_day)\
                         .filter(Orderbook.timestamp <= end_of_day)\
                         .count()
            return count > 0
    
    def _process_ohlcv_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process OHLCV dataframe from CoinAPI format to our format"""
        df_processed = df.reset_index()
        df_processed = df_processed.rename(columns={
            'time_period_start': 'timestamp',
            'price_open': 'open',
            'price_high': 'high',
            'price_low': 'low',
            'price_close': 'close',
            'volume_traded': 'volume'
        })
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df_processed.columns]
        df_processed = df_processed[available_cols].copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        return df_processed.dropna(subset=['timestamp'])
    
    def _insert_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> int:
        """Insert OHLCV data into database"""
        records = []
        
        for _, row in df.iterrows():
            record = OHLCV(
                symbol=symbol,
                timestamp=row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            records.append(record)
        
        if records:
            with db_manager.get_session() as session:
                session.bulk_save_objects(records)
            
        return len(records)
    
    def _insert_orderbook_data(self, df: pd.DataFrame, symbol: str) -> int:
        """Insert orderbook data into database - FIXED VERSION"""
        records = []
        
        for timestamp, row in df.iterrows():
            record_data = {
                'symbol': symbol,
                'timestamp': timestamp
            }
            
            # Add bid and ask levels - CONVERTIR A TIPOS PYTHON NATIVOS
            for i in range(1, 11):
                # Obtener valores y convertir np.float64 a float nativo de Python
                bid_price = row.get(f'bid{i}_price')
                bid_size = row.get(f'bid{i}_size')
                ask_price = row.get(f'ask{i}_price')
                ask_size = row.get(f'ask{i}_size')
                
                # Conversión segura a tipos Python nativos
                record_data[f'bid{i}_price'] = float(bid_price) if pd.notna(bid_price) else None
                record_data[f'bid{i}_size'] = float(bid_size) if pd.notna(bid_size) else None
                record_data[f'ask{i}_price'] = float(ask_price) if pd.notna(ask_price) else None
                record_data[f'ask{i}_size'] = float(ask_size) if pd.notna(ask_size) else None
            
            records.append(Orderbook(**record_data))
        
        if records:
            with db_manager.get_session() as session:
                session.bulk_save_objects(records)
            
        return len(records)
    
    def get_symbol_data_range(self, symbol: str) -> tuple:
        """Get the date range of available OHLCV data"""
        with db_manager.get_session() as session:
            min_date = session.query(OHLCV.timestamp).filter(OHLCV.symbol == symbol).order_by(OHLCV.timestamp.asc()).first()
            max_date = session.query(OHLCV.timestamp).filter(OHLCV.symbol == symbol).order_by(OHLCV.timestamp.desc()).first()
            
            if min_date and max_date:
                return min_date[0], max_date[0]
            return None, None

# Global instance
data_ingestion = DataIngestion()