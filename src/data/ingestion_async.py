#!/usr/bin/env python3
"""
Ultra-fast async data ingestion - VERSION OPTIMIZADA
Paraleliza llamadas API y optimiza inserciones en DB
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from sqlalchemy import text
from sqlalchemy.pool import QueuePool

from src.database.connection import db_manager
from src.utils.logger import get_ingestion_logger
from config.settings import settings

log = get_ingestion_logger()

class UltraFastIngestion:
    """Ingesta ultra-rápida con paralelización masiva"""
    
    def __init__(self):
        self.api_key = settings.COINAPI_KEY
        self.base_url = settings.COINAPI_BASE_URL
        
        # Configuración optimizada
        self.max_concurrent_api_calls = 10  # Llamadas API simultáneas
        self.max_concurrent_db_ops = 4      # Operaciones DB simultáneas
        self.batch_size = 10000             # Registros por batch
        self.chunk_days = 5                 # Días por chunk de API
        
        # Semáforos para control de concurrencia
        self.api_semaphore = asyncio.Semaphore(self.max_concurrent_api_calls)
        self.db_semaphore = asyncio.Semaphore(self.max_concurrent_db_ops)
        
        # Thread pool para operaciones DB
        self.db_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_db_ops)
        
        # Session compartida para todas las requests
        self.session = None
        
        # Buffer para acumular datos antes de insertar
        self.data_buffer = {
            'ohlcv': [],
            'orderbook': []
        }
    
    async def create_session(self):
        """Crear session HTTP reutilizable con configuración optimizada"""
        timeout = aiohttp.ClientTimeout(total=30, connect=5, sock_read=25)
        connector = aiohttp.TCPConnector(
            limit=100,  # Conexiones totales
            limit_per_host=30,  # Por host
            force_close=True,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'X-CoinAPI-Key': self.api_key,
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
        )
    
    async def close_session(self):
        """Cerrar session HTTP"""
        if self.session:
            await self.session.close()
    
    async def ingest_symbols_fast(self, symbols: List[str], data_types: List[str], 
                                 start_date: datetime, end_date: datetime) -> Dict:
        """Ingesta ultra-rápida de múltiples símbolos"""
        
        log.info(f"🚀 ULTRA-FAST ingestion starting...")
        log.info(f"  Symbols: {len(symbols)}")
        log.info(f"  Period: {start_date.date()} to {end_date.date()}")
        log.info(f"  Concurrency: {self.max_concurrent_api_calls} API calls")
        
        await self.create_session()
        
        try:
            # Crear todas las tareas de ingesta
            tasks = []
            
            for symbol in symbols:
                if 'ohlcv' in data_types:
                    tasks.append(self.ingest_ohlcv_async(symbol, start_date, end_date))
                
                if 'orderbook' in data_types:
                    tasks.append(self.ingest_orderbook_async(symbol, start_date, end_date))
            
            # Ejecutar todas las tareas en paralelo
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flush final de buffers
            await self.flush_all_buffers()
            
            elapsed = time.time() - start_time
            log.info(f"✅ ULTRA-FAST ingestion completed in {elapsed:.1f} seconds!")
            
            # Calcular estadísticas
            total_records = sum(r for r in results if isinstance(r, int))
            log.info(f"📊 Total records: {total_records:,}")
            log.info(f"⚡ Speed: {total_records/elapsed:.0f} records/second")
            
            return {'success': True, 'total_records': total_records, 'elapsed': elapsed}
            
        finally:
            await self.close_session()
            self.db_executor.shutdown(wait=True)
    
    async def ingest_ohlcv_async(self, symbol: str, start_date: datetime, end_date: datetime) -> int:
        """Ingesta OHLCV asíncrona para un símbolo"""
        log.info(f"📊 Starting async OHLCV ingestion for {symbol}")
        
        # Dividir en chunks de días
        date_chunks = self.create_date_chunks(start_date, end_date, self.chunk_days)
        
        # Crear tareas para cada chunk
        tasks = []
        for chunk_start, chunk_end in date_chunks:
            task = self.fetch_ohlcv_chunk(symbol, chunk_start, chunk_end)
            tasks.append(task)
        
        # Ejecutar chunks en paralelo
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Contar registros totales
        total_records = 0
        for result in chunk_results:
            if isinstance(result, int):
                total_records += result
            elif isinstance(result, Exception):
                log.error(f"Chunk failed for {symbol}: {result}")
        
        log.info(f"✅ OHLCV {symbol}: {total_records:,} records")
        return total_records
    
    async def fetch_ohlcv_chunk(self, symbol: str, start_date: datetime, end_date: datetime) -> int:
        """Fetch un chunk de datos OHLCV"""
        async with self.api_semaphore:
            current_date = start_date
            all_data = []
            
            while current_date <= end_date:
                try:
                    # Construir URL y params
                    url = f"{self.base_url}/ohlcv/{symbol}/history"
                    params = {
                        'period_id': '1MIN',
                        'date': current_date.strftime('%Y-%m-%d'),
                        'limit': 100000
                    }
                    
                    # Hacer request asíncrona
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data:
                                # Procesar datos inline
                                for item in data:
                                    try:
                                        timestamp = pd.to_datetime(item.get('time_period_start'))
                                        all_data.append({
                                            'symbol': symbol,
                                            'timestamp': timestamp,
                                            'open': float(item.get('price_open', 0)),
                                            'high': float(item.get('price_high', 0)),
                                            'low': float(item.get('price_low', 0)),
                                            'close': float(item.get('price_close', 0)),
                                            'volume': float(item.get('volume_traded', 0))
                                        })
                                    except:
                                        continue
                        
                        elif response.status == 429:
                            # Rate limit - esperar un poco
                            await asyncio.sleep(2)
                            continue
                
                except Exception as e:
                    log.warning(f"Error fetching {symbol} {current_date}: {e}")
                
                current_date += timedelta(days=1)
            
            # Agregar al buffer con validación mínima
            if all_data:
                valid_data = [d for d in all_data if d['open'] > 0 and d['volume'] >= 0]
                await self.add_to_buffer('ohlcv', valid_data)
                return len(valid_data)
            
            return 0
    
    async def ingest_orderbook_async(self, symbol: str, start_date: datetime, end_date: datetime) -> int:
        """Ingesta orderbook asíncrona optimizada"""
        log.info(f"📊 Starting async orderbook ingestion for {symbol}")
        
        # Para orderbook, usar chunks más pequeños (más pesado)
        date_chunks = self.create_date_chunks(start_date, end_date, 2)  # 2 días por chunk
        
        tasks = []
        for chunk_start, chunk_end in date_chunks:
            task = self.fetch_orderbook_chunk(symbol, chunk_start, chunk_end)
            tasks.append(task)
        
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_records = sum(r for r in chunk_results if isinstance(r, int))
        log.info(f"✅ Orderbook {symbol}: {total_records:,} records")
        return total_records
    
    async def fetch_orderbook_chunk(self, symbol: str, start_date: datetime, end_date: datetime) -> int:
        """Fetch chunk de orderbook con procesamiento optimizado"""
        async with self.api_semaphore:
            current_date = start_date
            all_data = []
            
            while current_date <= end_date:
                try:
                    url = f"{self.base_url}/orderbooks/{symbol}/history"
                    params = {
                        'date': current_date.strftime('%Y-%m-%d'),
                        'limit': 1000,
                        'limit_levels': 10
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data and isinstance(data, list):
                                # Procesar inline con validación mínima
                                for snapshot in data:
                                    processed = self.process_orderbook_snapshot_fast(snapshot, symbol)
                                    if processed:
                                        all_data.append(processed)
                        
                        elif response.status == 429:
                            await asyncio.sleep(3)
                            continue
                
                except Exception as e:
                    log.debug(f"Orderbook error {symbol} {current_date}: {e}")
                
                current_date += timedelta(days=1)
            
            if all_data:
                await self.add_to_buffer('orderbook', all_data)
                return len(all_data)
            
            return 0
    
    def process_orderbook_snapshot_fast(self, snapshot: dict, symbol: str) -> Optional[dict]:
        """Procesamiento ultra-rápido de snapshot"""
        try:
            timestamp = snapshot.get('time_exchange')
            if not timestamp:
                return None
            
            timestamp = pd.to_datetime(timestamp)
            
            bids = snapshot.get('bids', [])
            asks = snapshot.get('asks', [])
            
            if not bids or not asks:
                return None
            
            # Crear record con validación mínima
            record = {
                'symbol': symbol,
                'timestamp': timestamp
            }
            
            # Procesar niveles (solo validar nivel 1)
            for i in range(10):
                if i < len(bids):
                    record[f'bid{i+1}_price'] = float(bids[i].get('price', 0)) or None
                    record[f'bid{i+1}_size'] = float(bids[i].get('size', 0)) or None
                else:
                    record[f'bid{i+1}_price'] = None
                    record[f'bid{i+1}_size'] = None
                
                if i < len(asks):
                    record[f'ask{i+1}_price'] = float(asks[i].get('price', 0)) or None
                    record[f'ask{i+1}_size'] = float(asks[i].get('size', 0)) or None
                else:
                    record[f'ask{i+1}_price'] = None
                    record[f'ask{i+1}_size'] = None
            
            # Validación mínima: solo verificar nivel 1
            if (record.get('bid1_price') and record.get('ask1_price') and 
                record['bid1_price'] < record['ask1_price']):
                return record
            
            return None
            
        except:
            return None
    
    async def add_to_buffer(self, data_type: str, data: List[dict]):
        """Agregar datos al buffer y flush si es necesario"""
        self.data_buffer[data_type].extend(data)
        
        if len(self.data_buffer[data_type]) >= self.batch_size:
            await self.flush_buffer(data_type)
    
    async def flush_buffer(self, data_type: str):
        """Flush buffer a base de datos"""
        if not self.data_buffer[data_type]:
            return
        
        data_to_insert = self.data_buffer[data_type]
        self.data_buffer[data_type] = []
        
        # Insertar en thread separado para no bloquear
        loop = asyncio.get_event_loop()
        async with self.db_semaphore:
            await loop.run_in_executor(
                self.db_executor,
                self.bulk_insert_data,
                data_type,
                data_to_insert
            )
    
    def bulk_insert_data(self, data_type: str, data: List[dict]):
        """Inserción bulk ultra-rápida"""
        if not data:
            return
        
        try:
            # Usar COPY para máxima velocidad (PostgreSQL)
            if data_type == 'ohlcv':
                self.bulk_insert_ohlcv(data)
            elif data_type == 'orderbook':
                self.bulk_insert_orderbook(data)
                
        except Exception as e:
            log.error(f"Bulk insert error: {e}")
    
    def bulk_insert_ohlcv(self, data: List[dict]):
        """Bulk insert OHLCV usando COPY"""
        with db_manager.get_session() as session:
            # Convertir a formato para COPY
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            for record in data:
                writer.writerow([
                    record['symbol'],
                    record['timestamp'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume']
                ])
            
            output.seek(0)
            
            # Usar COPY para inserción masiva
            conn = session.connection().connection
            cursor = conn.cursor()
            
            cursor.copy_expert("""
                COPY ohlcv (symbol, timestamp, open, high, low, close, volume)
                FROM STDIN WITH CSV
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """, output)
            
            conn.commit()
            cursor.close()
            
            log.debug(f"Bulk inserted {len(data)} OHLCV records")
    
    def bulk_insert_orderbook(self, data: List[dict]):
        """Bulk insert orderbook con prepared statement"""
        with db_manager.get_session() as session:
            # Usar INSERT múltiple más eficiente
            if data:
                # Construir values para insert múltiple
                columns = list(data[0].keys())
                placeholders = ', '.join([f':{col}' for col in columns])
                columns_str = ', '.join(columns)
                
                # Para orderbook, usar chunks más pequeños
                chunk_size = 1000
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i+chunk_size]
                    
                    # Ejecutar insert múltiple
                    session.execute(text(f"""
                        INSERT INTO orderbook ({columns_str})
                        VALUES ({placeholders})
                        ON CONFLICT (symbol, timestamp) DO UPDATE SET
                            bid1_price = EXCLUDED.bid1_price,
                            bid1_size = EXCLUDED.bid1_size,
                            ask1_price = EXCLUDED.ask1_price,
                            ask1_size = EXCLUDED.ask1_size
                    """), chunk)
                
                session.commit()
                log.debug(f"Bulk inserted {len(data)} orderbook records")
    
    async def flush_all_buffers(self):
        """Flush todos los buffers pendientes"""
        for data_type in self.data_buffer:
            await self.flush_buffer(data_type)
    
    def create_date_chunks(self, start_date: datetime, end_date: datetime, chunk_days: int) -> List[Tuple[datetime, datetime]]:
        """Crear chunks de fechas para procesamiento paralelo"""
        chunks = []
        current = start_date
        
        while current <= end_date:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
        
        return chunks


async def run_ultra_fast_ingestion(symbols: List[str], days_back: int = 30):
    """Función principal para ejecutar ingesta ultra-rápida"""
    ingestion = UltraFastIngestion()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    data_types = ['ohlcv', 'orderbook']  # Tipos a ingestar
    
    result = await ingestion.ingest_symbols_fast(
        symbols=symbols,
        data_types=data_types,
        start_date=start_date,
        end_date=end_date
    )
    
    return result