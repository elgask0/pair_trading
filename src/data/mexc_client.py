#!/usr/bin/env python3
"""
MEXC client FIXED - USA ENDPOINTS CORRECTOS + MISSING DAYS LOGIC
Basado en el informe oficial de MEXC API endpoints
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from config.settings import settings
from src.utils.logger import get_ingestion_logger
from src.utils.exceptions import APIError
import time

log = get_ingestion_logger()

class MEXCClient:
    """MEXC client FIXED - USA ENDPOINTS CORRECTOS del informe oficial"""
    
    def __init__(self):
        # CORRECTO: Endpoint del informe que funciona
        self.base_url = 'https://contract.mexc.com/api/v1'
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Rate limiting según informe: 20 req/2s = 10 req/s
        self.rate_limit_delay = 0.11  # 110ms entre requests
        self.last_request_time = 0
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert internal symbol to MEXC format"""
        # MEXCFTS_PERP_GIGA_USDT -> GIGA_USDT (confirmado por diagnóstico)
        if "MEXCFTS_PERP_" in symbol:
            return symbol.replace("MEXCFTS_PERP_", "")
        return symbol
    
    def _rate_limit(self):
        """Respect rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: dict = None) -> dict:
        """Make rate-limited request"""
        self._rate_limit()
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for API error in response
                if isinstance(data, dict) and data.get('code') != 0:
                    raise APIError(f"MEXC API error: {data.get('msg', 'Unknown error')}")
                
                return data
            else:
                log.warning(f"MEXC API returned {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            log.error(f"MEXC API request failed: {e}")
            return None
    
    def list_available_symbols(self, quote: str = "USDT") -> List[str]:
        """NUEVO: Lista símbolos disponibles usando endpoint correcto del informe"""
        log.info(f"Listing available MEXC symbols for {quote}...")
        
        # CORRECTO: Endpoint del informe
        url = f"{self.base_url}/contract/detail"
        
        data = self._make_request(url)
        if not data or not data.get('data'):
            log.error("Failed to get contract details from MEXC")
            return []
        
        symbols = []
        for contract in data['data']:
            # Filtrar por settleCoin y apiAllowed según informe
            if (contract.get('settleCoin') == quote and 
                contract.get('apiAllowed', True)):
                symbols.append(contract['symbol'])
        
        log.info(f"Found {len(symbols)} available {quote} symbols on MEXC")
        return symbols
    
    def get_current_funding_rate(self, symbol: str) -> Optional[dict]:
        """NUEVO: Obtener funding rate actual usando endpoint correcto"""
        mexc_symbol = self._convert_symbol(symbol)
        
        # CORRECTO: Endpoint del informe
        url = f"{self.base_url}/contract/funding_rate/{mexc_symbol}"
        
        data = self._make_request(url)
        if data and data.get('data'):
            return data['data']
        
        return None
    
    def get_funding_rate_history_range(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """NUEVO: Obtener funding rates para un rango específico"""
        mexc_symbol = self._convert_symbol(symbol)
        log.info(f"Fetching funding rate history for {mexc_symbol} from {start_date.date()} to {end_date.date()}")
        
        # CORRECTO: Endpoint histórico del informe
        url = f"{self.base_url}/contract/funding_rate/history"
        
        all_records = []
        page = 1
        page_size = 1000  # Máximo según informe
        
        while True:
            params = {
                'symbol': mexc_symbol,
                'page_num': page,
                'page_size': page_size
            }
            
            data = self._make_request(url, params)
            
            if not data or not data.get('data') or not data['data'].get('resultList'):
                break
            
            records = data['data']['resultList']
            
            # Filtrar por rango de fechas
            filtered_records = []
            for record in records:
                # Convert timestamp (ms) to datetime
                settle_time = datetime.utcfromtimestamp(record['settleTime'] / 1000)
                
                if start_date <= settle_time <= end_date:
                    filtered_records.append({
                        'timestamp': settle_time,
                        'funding_rate': float(record['fundingRate']),
                        'collect_cycle': 28800  # 8 hours según informe
                    })
                elif settle_time < start_date:
                    # Ya pasamos el rango, podemos parar
                    break
            
            all_records.extend(filtered_records)
            
            # Si no hay más páginas o ya cubrimos el rango
            if len(records) < page_size or (filtered_records and filtered_records[-1]['timestamp'] < start_date):
                break
            
            page += 1
            
            # Safety check
            if page > 100:  # Máximo 100 páginas
                log.warning(f"Reached maximum pages for {mexc_symbol}")
                break
        
        if all_records:
            df = pd.DataFrame(all_records)
            df = df.sort_values('timestamp')
            df = df.drop_duplicates(subset=['timestamp'])
            
            log.info(f"Fetched {len(df)} funding rate records for {mexc_symbol}")
            return df
        
        return pd.DataFrame()
    
    def get_funding_rate_history(self, symbol: str) -> pd.DataFrame:
        """CORREGIDO: Usar endpoints correctos en lugar de generar mock data"""
        mexc_symbol = self._convert_symbol(symbol)
        log.info(f"Fetching ALL funding rate history for {mexc_symbol}")
        
        try:
            # CORRECTO: Endpoint histórico del informe
            url = f"{self.base_url}/contract/funding_rate/history"
            
            all_records = []
            page = 1
            page_size = 1000
            
            while True:
                params = {
                    'symbol': mexc_symbol,
                    'page_num': page,
                    'page_size': page_size
                }
                
                data = self._make_request(url, params)
                
                if not data or not data.get('data') or not data['data'].get('resultList'):
                    break
                
                records = data['data']['resultList']
                
                # Convert records
                for record in records:
                    try:
                        all_records.append({
                            'timestamp': datetime.utcfromtimestamp(record['settleTime'] / 1000),
                            'funding_rate': float(record['fundingRate']),
                            'collect_cycle': 28800
                        })
                    except Exception as e:
                        log.debug(f"Skipping invalid record: {e}")
                        continue
                
                # Check if we got all data
                if len(records) < page_size:
                    break
                
                page += 1
                
                # Safety check
                if page > 200:
                    log.warning(f"Reached maximum pages for {mexc_symbol}")
                    break
            
            if all_records:
                df = pd.DataFrame(all_records)
                df = df.sort_values('timestamp')
                df = df.drop_duplicates(subset=['timestamp'])
                
                log.info(f"Successfully fetched {len(df)} REAL funding rate records for {mexc_symbol}")
                return df
            else:
                log.warning(f"No funding rate data found for {mexc_symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            log.error(f"Failed to get REAL funding rates for {mexc_symbol}: {e}")
            # NO crear mock data - retornar vacío
            return pd.DataFrame()
    
    def _create_mock_funding_data(self, symbol: str) -> pd.DataFrame:
        """ELIMINADO: No más mock data - solo datos reales"""
        log.error(f"Mock data creation disabled - no real data available for {symbol}")
        return pd.DataFrame()

# Global client instance
mexc_client = MEXCClient()