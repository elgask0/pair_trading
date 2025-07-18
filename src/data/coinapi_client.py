#!/usr/bin/env python3
"""
CoinAPI client ROBUSTO - Solo datos reales de API
Version 3.1 - FIXED: Solicita 10 niveles de orderbook
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from config.settings import settings
from src.utils.logger import get_ingestion_logger
from src.utils.exceptions import APIError

log = get_ingestion_logger()

class CoinAPIClient:
    """CoinAPI client ROBUSTO - Solo datos reales"""
    
    def __init__(self):
        self.base_url = getattr(settings, 'COINAPI_BASE_URL', 'https://rest.coinapi.io/v1')
        self.api_key = getattr(settings, 'COINAPI_KEY', '')
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                'X-CoinAPI-Key': self.api_key,
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            })
        
        # Configuración de reintentos
        self.max_retries = 3
        self.base_timeout = 15
        self.retry_delay = 2
    
    def _make_request_with_retries(self, url: str, params: dict, context: str = "") -> Optional[dict]:
        """Hacer petición con reintentos automáticos"""
        
        for attempt in range(self.max_retries):
            timeout = self.base_timeout + (attempt * 5)  # Timeout incremental
            
            try:
                log.debug(f"Intento {attempt + 1}/{self.max_retries} para {context}")
                log.debug(f"URL: {url}")
                log.debug(f"Params: {params}")
                log.debug(f"Timeout: {timeout}s")
                
                response = self.session.get(url, params=params, timeout=timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    log.debug(f"✅ {context} exitoso en intento {attempt + 1}")
                    return data
                    
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 30))
                    log.warning(f"Rate limit en {context} - esperando {retry_after}s")
                    time.sleep(retry_after)
                    continue
                    
                else:
                    log.warning(f"Error {response.status_code} en {context}: {response.text[:200]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        return None
                        
            except requests.exceptions.Timeout:
                log.warning(f"Timeout en {context} (intento {attempt + 1}) - {timeout}s")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    log.error(f"Todos los intentos de {context} fallaron por timeout")
                    return None
                    
            except Exception as e:
                log.warning(f"Error en {context} (intento {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    log.error(f"Todos los intentos de {context} fallaron: {e}")
                    return None
        
        return None
    
    def get_orderbook_for_date(self, symbol: str, date_str: str) -> pd.DataFrame:
        """Get orderbook data - FIXED: Solicita 10 niveles"""
        
        log.info(f"Obteniendo orderbook REAL para {symbol} en {date_str}")
        
        url = f"{self.base_url}/orderbooks/{symbol}/history"
        
        # FIXED: Configuraciones optimizadas para 10 niveles
        configs = [
            {'date': date_str, 'limit': 500, 'limit_levels': 10},   # FIXED: 10 niveles
            {'date': date_str, 'limit': 1000, 'limit_levels': 10},  # FIXED: 10 niveles
            {'date': date_str, 'limit': 200, 'limit_levels': 8},    # Fallback con 8 niveles
            {'date': date_str, 'limit': 100, 'limit_levels': 5}     # Fallback mínimo
        ]
        
        for i, params in enumerate(configs):
            context = f"orderbook {symbol} {date_str} (config {i+1}: {params['limit_levels']} levels)"
            data = self._make_request_with_retries(url, params, context)
            
            if data and isinstance(data, list) and len(data) > 0:
                # Procesar datos reales
                processed_df = self._process_orderbook_response(data, symbol, date_str)
                if not processed_df.empty:
                    # Verificar cuántos niveles realmente obtuvimos
                    levels_obtained = self._count_levels_in_df(processed_df)
                    log.info(f"✅ Datos reales obtenidos para {symbol} en {date_str}: {len(processed_df)} snapshots, {levels_obtained} niveles promedio")
                    return processed_df
            
            # Pausa entre configuraciones
            if i < len(configs) - 1:
                time.sleep(2)
        
        # Si todo falla, retornar DataFrame vacío
        log.warning(f"❌ No se pudieron obtener datos reales para {symbol} en {date_str}")
        return pd.DataFrame()
    
    def _count_levels_in_df(self, df: pd.DataFrame) -> float:
        """Contar niveles promedio en el DataFrame"""
        if df.empty:
            return 0
        
        total_levels = 0
        total_snapshots = 0
        
        for _, row in df.iterrows():
            snapshot_levels = 0
            for i in range(1, 11):
                if (pd.notna(row.get(f'bid{i}_price')) and pd.notna(row.get(f'ask{i}_price')) and
                    row.get(f'bid{i}_price', 0) > 0 and row.get(f'ask{i}_price', 0) > 0):
                    snapshot_levels += 1
                else:
                    break
            
            total_levels += snapshot_levels
            total_snapshots += 1
        
        return total_levels / total_snapshots if total_snapshots > 0 else 0
    
    def _process_orderbook_response(self, data: List[dict], symbol: str, date_str: str) -> pd.DataFrame:
        """Procesar respuesta de orderbook de CoinAPI - MEJORADO para 10 niveles"""
        
        processed_data = []
        valid_snapshots = 0
        
        for snapshot in data:
            if not isinstance(snapshot, dict):
                continue
            
            # Extraer timestamp - CoinAPI usa 'time_exchange'
            timestamp = snapshot.get('time_exchange')
            if not timestamp:
                timestamp = snapshot.get('timestamp') or snapshot.get('time')
            
            if not timestamp:
                continue
            
            try:
                timestamp = pd.to_datetime(timestamp, utc=True)
            except:
                continue
            
            processed_snapshot = {'timestamp': timestamp}
            
            # Extraer bids y asks
            bids = snapshot.get('bids', [])
            asks = snapshot.get('asks', [])
            
            if not bids or not asks:
                continue
            
            # FIXED: Procesar hasta 10 niveles con validación mejorada
            has_level1 = False
            levels_processed = 0
            
            for i in range(10):
                # Procesar bids
                if i < len(bids) and isinstance(bids[i], dict):
                    try:
                        price = float(bids[i].get('price', 0))
                        size = float(bids[i].get('size', 0))
                        
                        if price > 0 and size > 0:
                            processed_snapshot[f'bid{i+1}_price'] = price
                            processed_snapshot[f'bid{i+1}_size'] = size
                            if i == 0:
                                has_level1 = True
                            levels_processed = max(levels_processed, i + 1)
                        else:
                            processed_snapshot[f'bid{i+1}_price'] = None
                            processed_snapshot[f'bid{i+1}_size'] = None
                    except (ValueError, TypeError):
                        processed_snapshot[f'bid{i+1}_price'] = None
                        processed_snapshot[f'bid{i+1}_size'] = None
                else:
                    processed_snapshot[f'bid{i+1}_price'] = None
                    processed_snapshot[f'bid{i+1}_size'] = None
                
                # Procesar asks
                if i < len(asks) and isinstance(asks[i], dict):
                    try:
                        price = float(asks[i].get('price', 0))
                        size = float(asks[i].get('size', 0))
                        
                        if price > 0 and size > 0:
                            processed_snapshot[f'ask{i+1}_price'] = price
                            processed_snapshot[f'ask{i+1}_size'] = size
                            if i == 0:
                                has_level1 = True
                            levels_processed = max(levels_processed, i + 1)
                        else:
                            processed_snapshot[f'ask{i+1}_price'] = None
                            processed_snapshot[f'ask{i+1}_size'] = None
                    except (ValueError, TypeError):
                        processed_snapshot[f'ask{i+1}_price'] = None
                        processed_snapshot[f'ask{i+1}_size'] = None
                else:
                    processed_snapshot[f'ask{i+1}_price'] = None
                    processed_snapshot[f'ask{i+1}_size'] = None
            
            # Validar spread
            if has_level1:
                bid1 = processed_snapshot.get('bid1_price')
                ask1 = processed_snapshot.get('ask1_price')
                
                if bid1 and ask1 and bid1 < ask1:
                    processed_data.append(processed_snapshot)
                    valid_snapshots += 1
                    
                    # Log ocasional para debugging
                    if valid_snapshots % 100 == 0:
                        log.debug(f"Snapshot {valid_snapshots}: {levels_processed} niveles procesados")
        
        if processed_data:
            df = pd.DataFrame(processed_data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            return df
        else:
            return pd.DataFrame()
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information - ROBUSTO"""
        log.info(f"Getting symbol info for {symbol}")
        
        try:
            # Intentar endpoint directo
            url = f"{self.base_url}/symbols/{symbol}"
            data = self._make_request_with_retries(url, {}, f"symbol_info {symbol}")
            
            if data:
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif isinstance(data, dict):
                    return data
            
            # Fallback a búsqueda
            url = f"{self.base_url}/symbols"
            params = {'filter_symbol_id': symbol}
            data = self._make_request_with_retries(url, params, f"symbol_search {symbol}")
            
            if data and isinstance(data, list):
                for sym in data:
                    if isinstance(sym, dict) and sym.get('symbol_id') == symbol:
                        return sym
            
            # Fallback final a datos básicos
            return self._create_basic_symbol_info(symbol)
            
        except Exception as e:
            log.error(f"Failed to get symbol info for {symbol}: {e}")
            return self._create_basic_symbol_info(symbol)
    
    def _create_basic_symbol_info(self, symbol: str) -> Dict:
        """Create basic symbol info when API fails"""
        log.info(f"Creating basic symbol info for {symbol}")
        
        parts = symbol.split('_')
        if len(parts) >= 4:
            exchange_id = parts[0]
            asset_base = parts[2]
            asset_quote = parts[3]
        else:
            exchange_id = 'UNKNOWN'
            asset_base = symbol
            asset_quote = 'USDT'
        
        now = datetime.now()
        return {
            'symbol_id': symbol,
            'exchange_id': exchange_id,
            'symbol_type': 'FUTURES' if 'PERP' in symbol else 'SPOT',
            'asset_id_base': asset_base,
            'asset_id_quote': asset_quote,
            'data_start': now - timedelta(days=365),
            'data_end': now,
            'data_quote_start': now - timedelta(days=365),
            'data_quote_end': now,
            'data_orderbook_start': now - timedelta(days=365),
            'data_orderbook_end': now,
            'data_trade_start': now - timedelta(days=365),
            'data_trade_end': now
        }
    
    def get_available_date_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get available data date range for symbol"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            
            data_start = symbol_info.get('data_start')
            data_end = symbol_info.get('data_end')
            
            if data_start:
                if isinstance(data_start, str):
                    data_start = pd.to_datetime(data_start)
                elif isinstance(data_start, datetime):
                    data_start = data_start
                else:
                    data_start = None
            
            if data_end:
                if isinstance(data_end, str):
                    data_end = pd.to_datetime(data_end)
                elif isinstance(data_end, datetime):
                    data_end = data_end
                else:
                    data_end = None
            
            # Fallback a rangos razonables
            if not data_start:
                data_start = datetime.now() - timedelta(days=365)
            if not data_end:
                data_end = datetime.now()
            
            log.info(f"Available data for {symbol}: {data_start.date()} to {data_end.date()}")
            return data_start, data_end
            
        except Exception as e:
            log.error(f"Failed to get date range for {symbol}: {e}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            return start_date, end_date
    
    def get_ohlcv_for_date(self, symbol: str, date_str: str) -> pd.DataFrame:
        """Get OHLCV data ROBUSTO"""
        
        url = f"{self.base_url}/ohlcv/{symbol}/history"
        params = {
            'period_id': '1MIN',
            'date': date_str,
            'limit': 100000
        }
        
        context = f"ohlcv {symbol} {date_str}"
        data = self._make_request_with_retries(url, params, context)
        
        if data and isinstance(data, list) and len(data) > 0:
            try:
                df = pd.DataFrame(data)
                
                # Buscar columna de tiempo
                time_col = None
                for col in ['time_period_start', 'timestamp', 'time']:
                    if col in df.columns:
                        time_col = col
                        break
                
                if not time_col:
                    log.warning(f"No time column found in OHLCV response for {symbol}")
                    return pd.DataFrame()
                
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                
                # Mapear columnas
                column_mapping = {
                    'price_open': 'open', 'open_price': 'open', 'o': 'open',
                    'price_high': 'high', 'high_price': 'high', 'h': 'high',
                    'price_low': 'low', 'low_price': 'low', 'l': 'low',
                    'price_close': 'close', 'close_price': 'close', 'c': 'close',
                    'volume_traded': 'volume', 'vol': 'volume', 'v': 'volume'
                }
                
                df = df.rename(columns=column_mapping)
                
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    log.warning(f"Missing OHLCV columns {missing_cols} for {symbol}")
                    return pd.DataFrame()
                
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=required_cols)
                
                log.info(f"Successfully fetched {len(df)} OHLCV records for {symbol} on {date_str}")
                return df[required_cols]
                
            except Exception as e:
                log.error(f"Error processing OHLCV for {symbol} on {date_str}: {e}")
                return pd.DataFrame()
        else:
            log.warning(f"No OHLCV data for {symbol} on {date_str}")
            return pd.DataFrame()

# Instancia global
coinapi_client = CoinAPIClient()