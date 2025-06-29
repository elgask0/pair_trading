#!/usr/bin/env python3
"""
CoinAPI client for historical data - FIXED VERSION
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
    """CoinAPI client for market data"""
    
    def __init__(self):
        self.base_url = getattr(settings, 'COINAPI_BASE_URL', 'https://rest.coinapi.io/v1')
        self.api_key = getattr(settings, 'COINAPI_KEY', '')
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'X-CoinAPI-Key': self.api_key,
                'Accept': 'application/json'
            })
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information - FIXED VERSION"""
        log.info(f"Getting symbol info for {symbol}")
        
        try:
            # Try direct symbol lookup first
            url = f"{self.base_url}/symbols/{symbol}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                # Handle both single symbol and list responses
                if isinstance(data, list) and len(data) > 0:
                    return data[0]  # Take first result
                elif isinstance(data, dict):
                    return data
                else:
                    log.warning(f"Unexpected response format for {symbol}")
                    return self._search_symbol_info(symbol)
            elif response.status_code == 404:
                log.warning(f"Direct symbol lookup failed for {symbol}, trying search...")
                return self._search_symbol_info(symbol)
            else:
                log.warning(f"CoinAPI error {response.status_code} for {symbol}, trying search...")
                return self._search_symbol_info(symbol)
                
        except Exception as e:
            log.error(f"Failed to get symbol info for {symbol}: {e}")
            # Fallback to search
            try:
                return self._search_symbol_info(symbol)
            except:
                # Ultimate fallback - create mock symbol info
                return self._create_mock_symbol_info(symbol)
    
    def _search_symbol_info(self, symbol: str) -> Dict:
        """Search for symbol info using symbols endpoint - FIXED VERSION"""
        try:
            url = f"{self.base_url}/symbols"
            params = {'filter_symbol_id': symbol}
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response formats
                symbols_list = data if isinstance(data, list) else data.get('data', [])
                
                for sym in symbols_list:
                    if isinstance(sym, dict) and sym.get('symbol_id') == symbol:
                        log.info(f"Found symbol info for {symbol} via search")
                        return sym
                
                # If not found, try partial match
                for sym in symbols_list:
                    if isinstance(sym, dict) and symbol in sym.get('symbol_id', ''):
                        log.info(f"Found partial match for {symbol}: {sym.get('symbol_id')}")
                        return sym
                
                log.warning(f"Symbol {symbol} not found in search results")
                return self._create_mock_symbol_info(symbol)
            else:
                log.warning(f"CoinAPI search error {response.status_code}: {response.text}")
                return self._create_mock_symbol_info(symbol)
                
        except Exception as e:
            log.error(f"Symbol search failed for {symbol}: {e}")
            return self._create_mock_symbol_info(symbol)
    
    def _create_mock_symbol_info(self, symbol: str) -> Dict:
        """Create mock symbol info when API fails"""
        log.info(f"Creating mock symbol info for {symbol}")
        
        # Extract exchange and base/quote from symbol name
        parts = symbol.split('_')
        if len(parts) >= 4:  # MEXCFTS_PERP_SPX_USDT
            exchange_id = parts[0]
            asset_base = parts[2] if len(parts) > 2 else 'UNK'
            asset_quote = parts[3] if len(parts) > 3 else 'USDT'
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
            'data_start': now - timedelta(days=365),  # 1 year ago
            'data_end': now,
            'data_quote_start': now - timedelta(days=365),
            'data_quote_end': now,
            'data_orderbook_start': now - timedelta(days=365),
            'data_orderbook_end': now,
            'data_trade_start': now - timedelta(days=365),
            'data_trade_end': now
        }
    
    def get_available_date_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get available data date range for symbol - FIXED VERSION"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            
            # Handle missing or None values
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
            
            # Fallback to reasonable defaults if missing
            if not data_start:
                data_start = datetime.now() - timedelta(days=365)
            if not data_end:
                data_end = datetime.now()
            
            log.info(f"Available data for {symbol}: {data_start} to {data_end}")
            return data_start, data_end
            
        except Exception as e:
            log.error(f"Failed to get date range for {symbol}: {e}")
            # Return reasonable defaults
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            log.info(f"Using fallback date range for {symbol}: {start_date} to {end_date}")
            return start_date, end_date
    
    def get_ohlcv_for_date(self, symbol: str, date_str: str) -> pd.DataFrame:
        """Get OHLCV data for a specific date - IMPROVED ERROR HANDLING"""
        try:
            url = f"{self.base_url}/ohlcv/{symbol}/history"
            params = {
                'period_id': '1MIN',
                'time_start': f"{date_str}T00:00:00Z",
                'time_end': f"{date_str}T23:59:59Z",
                'limit': 2000
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if not data:
                    return pd.DataFrame()
                
                # Handle different response formats
                if isinstance(data, dict):
                    if 'data' in data:
                        data = data['data']
                    elif 'result' in data:
                        data = data['result']
                    else:
                        log.warning(f"Unexpected OHLCV response format for {symbol} on {date_str}")
                        return pd.DataFrame()
                
                if not isinstance(data, list) or not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                
                # Check if required columns exist
                time_col = None
                for col in ['time_period_start', 'timestamp', 'time']:
                    if col in df.columns:
                        time_col = col
                        break
                
                if not time_col:
                    log.warning(f"No time column found in OHLCV response for {symbol} on {date_str}")
                    return pd.DataFrame()
                
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                
                # Map columns to standard names
                column_mapping = {
                    'price_open': 'open', 'open_price': 'open', 'o': 'open',
                    'price_high': 'high', 'high_price': 'high', 'h': 'high',
                    'price_low': 'low', 'low_price': 'low', 'l': 'low',
                    'price_close': 'close', 'close_price': 'close', 'c': 'close',
                    'volume_traded': 'volume', 'vol': 'volume', 'v': 'volume'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Ensure we have all required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    log.warning(f"Missing OHLCV columns {missing_cols} for {symbol} on {date_str}")
                    return pd.DataFrame()
                
                # Convert to float and handle any string/null values
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with NaN values
                df = df.dropna(subset=required_cols)
                
                return df[required_cols]
            else:
                log.warning(f"No OHLCV data for {symbol} on {date_str}: HTTP {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            log.error(f"Failed to get OHLCV for {symbol} on {date_str}: {e}")
            return pd.DataFrame()
    
    def get_orderbook_for_date(self, symbol: str, date_str: str) -> pd.DataFrame:
        """Get orderbook data for a specific date - IMPROVED ERROR HANDLING"""
        try:
            url = f"{self.base_url}/orderbooks/{symbol}/history"
            params = {
                'time_start': f"{date_str}T00:00:00Z",
                'time_end': f"{date_str}T23:59:59Z",
                'limit': 1000
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if not data:
                    return pd.DataFrame()
                
                # Handle different response formats
                if isinstance(data, dict):
                    if 'data' in data:
                        data = data['data']
                    elif 'result' in data:
                        data = data['result']
                    else:
                        log.warning(f"Unexpected orderbook response format for {symbol} on {date_str}")
                        return pd.DataFrame()
                
                if not isinstance(data, list) or not data:
                    return pd.DataFrame()
                
                # Process orderbook data
                processed_data = []
                for snapshot in data:
                    if not isinstance(snapshot, dict):
                        continue
                    
                    # Find timestamp field
                    timestamp = None
                    for time_field in ['time_exchange', 'timestamp', 'time']:
                        if time_field in snapshot:
                            timestamp = snapshot[time_field]
                            break
                    
                    if not timestamp:
                        continue
                    
                    processed_snapshot = {
                        'timestamp': pd.to_datetime(timestamp)
                    }
                    
                    # Extract bids and asks (up to 10 levels)
                    bids = snapshot.get('bids', [])
                    asks = snapshot.get('asks', [])
                    
                    for i in range(10):
                        # Process bids
                        if i < len(bids) and isinstance(bids[i], dict):
                            try:
                                processed_snapshot[f'bid_{i+1}_price'] = float(bids[i].get('price', 0))
                                processed_snapshot[f'bid_{i+1}_size'] = float(bids[i].get('size', 0))
                            except (ValueError, TypeError):
                                processed_snapshot[f'bid_{i+1}_price'] = None
                                processed_snapshot[f'bid_{i+1}_size'] = None
                        else:
                            processed_snapshot[f'bid_{i+1}_price'] = None
                            processed_snapshot[f'bid_{i+1}_size'] = None
                        
                        # Process asks
                        if i < len(asks) and isinstance(asks[i], dict):
                            try:
                                processed_snapshot[f'ask_{i+1}_price'] = float(asks[i].get('price', 0))
                                processed_snapshot[f'ask_{i+1}_size'] = float(asks[i].get('size', 0))
                            except (ValueError, TypeError):
                                processed_snapshot[f'ask_{i+1}_price'] = None
                                processed_snapshot[f'ask_{i+1}_size'] = None
                        else:
                            processed_snapshot[f'ask_{i+1}_price'] = None
                            processed_snapshot[f'ask_{i+1}_size'] = None
                    
                    processed_data.append(processed_snapshot)
                
                if processed_data:
                    df = pd.DataFrame(processed_data)
                    df.set_index('timestamp', inplace=True)
                    return df
                else:
                    return pd.DataFrame()
            else:
                log.warning(f"No orderbook data for {symbol} on {date_str}: HTTP {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            log.error(f"Failed to get orderbook for {symbol} on {date_str}: {e}")
            return pd.DataFrame()

# Global client instance
coinapi_client = CoinAPIClient()