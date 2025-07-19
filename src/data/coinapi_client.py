#!/usr/bin/env python3
"""
CoinAPI client ULTRA OPTIMIZADO - Version 4.0 INCREMENTAL
MEJORAS: Timeouts adaptativos, circuit breaker, skip inteligente
MANTIENE: Toda la funcionalidad existente
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import random
from config.settings import settings
from src.utils.logger import get_ingestion_logger
from src.utils.exceptions import APIError

log = get_ingestion_logger()

class CoinAPIClient:
    """CoinAPI client ULTRA OPTIMIZADO - BACKWARD COMPATIBLE"""
    
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
        
        # 🚀 NUEVOS: Timeouts adaptativos
        self.base_connect_timeout = 3
        self.base_read_timeout = 10  # Reducido de 30 a 10
        self.max_read_timeout = 45
        self.current_timeout_multiplier = 1.0
        
        # 🔄 NUEVOS: Configuraciones inteligentes
        self.max_retries = 2  # Reducido de 3 a 2
        self.base_retry_delay = 1  # Reducido de 2 a 1
        self.max_retry_delay = 10
        
        # 🚨 NUEVO: Circuit breaker
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.circuit_breaker_delay = 30
        self.last_circuit_break = None
        
        # 📊 NUEVOS: Estadísticas
        self.total_requests = 0
        self.successful_requests = 0
        self.timeout_requests = 0
        self.rate_limit_hits = 0
        
        # ✅ MANTENER: Configuración legacy para compatibilidad
        self.timeout = (self.base_connect_timeout, self.base_read_timeout)
        self.retry_delay = self.base_retry_delay
    
    def _get_adaptive_timeout(self) -> Tuple[int, int]:
        """NUEVO: Calcular timeout adaptativo"""
        connect_timeout = self.base_connect_timeout
        read_timeout = min(
            self.base_read_timeout * self.current_timeout_multiplier,
            self.max_read_timeout
        )
        return (int(connect_timeout), int(read_timeout))
    
    def _update_timeout_strategy(self, success: bool, was_timeout: bool):
        """NUEVO: Actualizar estrategia de timeout"""
        if success:
            self.current_timeout_multiplier = max(0.8, self.current_timeout_multiplier * 0.95)
            self.consecutive_failures = 0
        elif was_timeout:
            self.current_timeout_multiplier = min(3.0, self.current_timeout_multiplier * 1.3)
            self.consecutive_failures += 1
        else:
            self.consecutive_failures += 1
    
    def _should_circuit_break(self) -> bool:
        """NUEVO: Circuit breaker logic"""
        if self.consecutive_failures >= self.max_consecutive_failures:
            now = datetime.now()
            
            if (self.last_circuit_break is None or 
                (now - self.last_circuit_break).total_seconds() > self.circuit_breaker_delay):
                
                log.warning(f"🚨 CIRCUIT BREAKER: {self.consecutive_failures} fallos - pausa {self.circuit_breaker_delay}s")
                
                self.last_circuit_break = now
                time.sleep(self.circuit_breaker_delay)
                self.consecutive_failures = 0
                return True
        return False
    
    def _get_jittered_delay(self, base_delay: float) -> float:
        """NUEVO: Añadir jitter para evitar thundering herd"""
        jitter = random.uniform(0.5, 1.5)
        return min(base_delay * jitter, self.max_retry_delay)
    
    def _make_request_with_retries(self, url: str, params: dict, context: str = "") -> Optional[dict]:
        """MEJORADO: Con circuit breaker y timeouts adaptativos"""
        
        # NUEVO: Verificar circuit breaker
        if self._should_circuit_break():
            return None
        
        self.total_requests += 1
        
        for attempt in range(self.max_retries):
            try:
                # NUEVO: Usar timeout adaptativo
                adaptive_timeout = self._get_adaptive_timeout()
                
                log.debug(f"🔄 {context} - intento {attempt + 1}/{self.max_retries} - timeout: {adaptive_timeout}")
                
                response = self.session.get(url, params=params, timeout=adaptive_timeout)
                
                self._log_rate_limit_headers(response, context)
                
                if response.status_code == 200:
                    data = response.json()
                    self.successful_requests += 1
                    self._update_timeout_strategy(success=True, was_timeout=False)
                    
                    # NUEVO: Log ocasional de stats
                    if self.total_requests % 20 == 0:
                        success_rate = (self.successful_requests / self.total_requests) * 100
                        log.info(f"📊 Success: {success_rate:.1f}%, timeout_mult: {self.current_timeout_multiplier:.2f}")
                    
                    return data
                    
                elif response.status_code == 429:
                    # MEJORADO: Rate limit con jitter
                    retry_after = int(response.headers.get('Retry-After', 15))
                    retry_after = min(retry_after, 60)
                    
                    self.rate_limit_hits += 1
                    
                    # NUEVO: Detection de rate limit abuse
                    if self.rate_limit_hits > 5:
                        log.warning(f"⚠️ Muchos rate limits - activando circuit breaker")
                        self.consecutive_failures = self.max_consecutive_failures
                        return None
                    
                    jittered_delay = self._get_jittered_delay(retry_after)
                    log.warning(f"🕐 Rate limit - esperando {jittered_delay:.1f}s")
                    time.sleep(jittered_delay)
                    continue
                    
                elif response.status_code == 503:
                    # MANTENER: Backoff exponencial para 503
                    wait_time = self._get_jittered_delay(2 ** attempt + 1)
                    log.warning(f"🔧 Error 503 - esperando {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    log.warning(f"❌ Error {response.status_code} en {context}: {response.text[:100]}")
                    if attempt < self.max_retries - 1:
                        delay = self._get_jittered_delay(self.base_retry_delay * (attempt + 1))
                        time.sleep(delay)
                        continue
                    else:
                        return None
                        
            except requests.exceptions.Timeout:
                self.timeout_requests += 1
                self._update_timeout_strategy(success=False, was_timeout=True)
                
                timeout_rate = (self.timeout_requests / self.total_requests) * 100
                log.warning(f"⏱️ Timeout en {context} - rate: {timeout_rate:.1f}%")
                
                # NUEVO: Skip agresivo si muchos timeouts
                if timeout_rate > 50 and attempt == 0:
                    log.warning(f"⚡ Alto rate de timeouts - skip")
                    break
                    
                if attempt < self.max_retries - 1:
                    delay = self._get_jittered_delay(self.base_retry_delay * (attempt + 1))
                    time.sleep(delay)
                    continue
                else:
                    log.error(f"💥 Todos los timeouts fallaron: {context}")
                    break
                    
            except Exception as e:
                log.warning(f"💥 Error en {context} (intento {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self._get_jittered_delay(self.base_retry_delay * (attempt + 1))
                    time.sleep(delay)
                    continue
                else:
                    break
        
        # NUEVO: Update failure stats
        self._update_timeout_strategy(success=False, was_timeout=False)
        return None
    
    def _log_rate_limit_headers(self, response: requests.Response, context: str):
        """MANTENER: Log rate limits (sin cambios)"""
        headers = response.headers
        
        limit = headers.get('X-RateLimit-Limit')
        remaining = headers.get('X-RateLimit-Remaining')
        cost = headers.get('X-RateLimit-Request-Cost')
        
        if remaining and limit:
            usage_pct = ((int(limit) - int(remaining)) / int(limit)) * 100
            
            # MEJORADO: Solo log relevante
            if int(remaining) < 500:
                log.warning(f"⚠️ Rate limit bajo: {remaining}/{limit} ({usage_pct:.1f}%)")
            elif self.total_requests % 50 == 0:
                log.debug(f"📊 Rate limit: {remaining}/{limit} ({usage_pct:.1f}%)")
        
        if cost:
            log.debug(f"Coste: {cost} créditos")
    
    def get_orderbook_for_date(self, symbol: str, date_str: str) -> pd.DataFrame:
        """MEJORADO: Con configuraciones más inteligentes y skip rápido"""
        
        log.info(f"📊 Obteniendo orderbook para {symbol} en {date_str}")
        
        url = f"{self.base_url}/orderbooks/{symbol}/history"
        
        # 🚀 MEJORADO: Configuraciones más diferenciadas
        configs = [
            {'date': date_str, 'limit': 800, 'limit_levels': 10, 'name': 'completo'},
            {'date': date_str, 'limit': 300, 'limit_levels': 6, 'name': 'medio'}, 
            {'date': date_str, 'limit': 100, 'limit_levels': 3, 'name': 'básico'}
        ]
        
        start_time = time.time()
        
        for i, config in enumerate(configs):
            config_name = config.pop('name')  # Remove name from params
            context = f"orderbook {symbol} {date_str} ({config_name})"
            
            try:
                data = self._make_request_with_retries(url, config, context)
                
                if data and isinstance(data, list) and len(data) > 0:
                    processed_df = self._process_orderbook_response(data, symbol, date_str)
                    if not processed_df.empty:
                        elapsed = time.time() - start_time
                        levels_obtained = self._count_levels_in_df(processed_df)
                        log.info(f"✅ {symbol} {date_str}: {len(processed_df)} válidos, "
                               f"{levels_obtained:.1f} niveles, {elapsed:.1f}s")
                        return processed_df
                else:
                    log.debug(f"Config {config_name}: Sin datos")
                
            except Exception as e:
                log.warning(f"Error config {config_name}: {e}")
                continue
            
            # MEJORADO: Pausas inteligentes
            if i < len(configs) - 1:
                base_pause = 1.0 if self.consecutive_failures < 2 else 3.0
                pause = self._get_jittered_delay(base_pause)
                time.sleep(pause)
        
        elapsed = time.time() - start_time
        log.warning(f"❌ No datos para {symbol} en {date_str} ({elapsed:.1f}s)")
        return pd.DataFrame()
    
    def _count_levels_in_df(self, df: pd.DataFrame) -> float:
        """MANTENER: Sin cambios (ya optimizado)"""
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
        """MANTENER: Sin cambios (funciona bien)"""
        
        processed_data = []
        valid_snapshots = 0
        target_date = pd.to_datetime(date_str).date()
        
        for snapshot in data:
            if not isinstance(snapshot, dict):
                continue
            
            timestamp = snapshot.get('time_exchange')
            if not timestamp:
                timestamp = snapshot.get('timestamp') or snapshot.get('time')
            
            if not timestamp:
                continue
            
            try:
                timestamp = pd.to_datetime(timestamp, utc=True)
                
                if timestamp.date() != target_date:
                    continue
                    
            except:
                continue
            
            processed_snapshot = {'timestamp': timestamp}
            
            bids = snapshot.get('bids', [])
            asks = snapshot.get('asks', [])
            
            if not bids or not asks:
                continue
            
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
        
        if processed_data:
            df = pd.DataFrame(processed_data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            return df
        else:
            return pd.DataFrame()
    
    def get_ohlcv_for_date(self, symbol: str, date_str: str) -> pd.DataFrame:
        """MANTENER: Sin cambios (funciona bien)"""
        
        url = f"{self.base_url}/ohlcv/{symbol}/history"
        
        target_date = pd.to_datetime(date_str).date()
        time_start = f"{target_date}T00:00:00"
        time_end = f"{target_date}T23:59:59"
        
        params = {
            'period_id': '1MIN',
            'time_start': time_start,
            'time_end': time_end,
            'limit': 1500
        }
        
        context = f"ohlcv {symbol} {date_str}"
        data = self._make_request_with_retries(url, params, context)
        
        if data and isinstance(data, list) and len(data) > 0:
            try:
                df = pd.DataFrame(data)
                
                time_col = None
                for col in ['time_period_start', 'timestamp', 'time']:
                    if col in df.columns:
                        time_col = col
                        break
                
                if not time_col:
                    log.warning(f"No time column found for {symbol}")
                    return pd.DataFrame()
                
                df[time_col] = pd.to_datetime(df[time_col])
                df = df[df[time_col].dt.date == target_date]
                
                if df.empty:
                    return pd.DataFrame()
                
                df.set_index(time_col, inplace=True)
                
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
                    return pd.DataFrame()
                
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=required_cols)
                
                if len(df) > 1500:
                    df = df.tail(1440)
                
                return df[required_cols]
                
            except Exception as e:
                log.error(f"Error processing OHLCV for {symbol} on {date_str}: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """MANTENER: Sin cambios (funciona bien)"""
        log.info(f"Getting symbol info for {symbol}")
        
        try:
            url = f"{self.base_url}/symbols/{symbol}"
            data = self._make_request_with_retries(url, {}, f"symbol_info {symbol}")
            
            if data:
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif isinstance(data, dict):
                    return data
            
            url = f"{self.base_url}/symbols"
            params = {'filter_symbol_id': symbol}
            data = self._make_request_with_retries(url, params, f"symbol_search {symbol}")
            
            if data and isinstance(data, list):
                for sym in data:
                    if isinstance(sym, dict) and sym.get('symbol_id') == symbol:
                        return sym
            
            return self._create_basic_symbol_info(symbol)
            
        except Exception as e:
            log.error(f"Failed to get symbol info for {symbol}: {e}")
            return self._create_basic_symbol_info(symbol)
    
    def _create_basic_symbol_info(self, symbol: str) -> Dict:
        """MANTENER: Sin cambios"""
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
        """MANTENER: Sin cambios"""
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
    
    def print_performance_stats(self):
        """NUEVO: Imprimir estadísticas de rendimiento"""
        if self.total_requests > 0:
            success_rate = (self.successful_requests / self.total_requests) * 100
            timeout_rate = (self.timeout_requests / self.total_requests) * 100
            
            log.info(f"\n📊 PERFORMANCE STATS:")
            log.info(f"  Total requests: {self.total_requests}")
            log.info(f"  Success rate: {success_rate:.1f}%")
            log.info(f"  Timeout rate: {timeout_rate:.1f}%")
            log.info(f"  Rate limit hits: {self.rate_limit_hits}")
            log.info(f"  Current timeout multiplier: {self.current_timeout_multiplier:.2f}")

# Instancia global
coinapi_client = CoinAPIClient()