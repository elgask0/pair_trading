import requests
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from src.utils.logger import log
from src.utils.exceptions import APIError
from config.settings import settings

class MEXCClient:
    """MEXC API client for fetching market data"""
    
    def __init__(self):
        self.base_url = settings.MEXC_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make HTTP request with error handling and rate limiting"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Rate limiting - MEXC allows ~10 requests per second
            time.sleep(0.1)
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check MEXC API response format
            if not data.get('success', True):
                raise APIError(f"MEXC API error: {data.get('errorMsg', 'Unknown error')}")
                
            return data.get('data', data)
            
        except requests.RequestException as e:
            log.error(f"HTTP request failed for {endpoint}: {e}")
            raise APIError(f"Request failed: {e}")
        except Exception as e:
            log.error(f"Unexpected error for {endpoint}: {e}")
            raise APIError(f"Unexpected error: {e}")
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information"""
        try:
            data = self._make_request("contract/detail", {"symbol": symbol})
            log.debug(f"Retrieved symbol info for {symbol}")
            return data
        except Exception as e:
            log.error(f"Failed to get symbol info for {symbol}: {e}")
            raise
    
    def get_klines(self, symbol: str, interval: str = "1m", limit: int = 1000, 
                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Get historical kline/candlestick data
        
        Args:
            symbol: Trading symbol (e.g., "GIGA_USDT")
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Number of records to return (max 1000)
            start_time: Start time for historical data
            end_time: End time for historical data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        try:
            data = self._make_request("contract/kline", params)
            log.debug(f"Retrieved {len(data)} klines for {symbol}")
            return data
        except Exception as e:
            log.error(f"Failed to get klines for {symbol}: {e}")
            raise
    
    def get_historical_data(self, symbol: str, days_back: Optional[int] = None) -> pd.DataFrame:
        """
        Get all available historical data for a symbol
        
        Args:
            symbol: Trading symbol
            days_back: If specified, only get this many days back. If None, get all available data.
        """
        log.info(f"Fetching historical data for {symbol}")
        
        all_data = []
        current_end_time = datetime.now()
        
        # If days_back is specified, set start time
        if days_back:
            start_time = current_end_time - timedelta(days=days_back)
        else:
            start_time = None
        
        # Fetch data in chunks (MEXC limits to 1000 records per request)
        while True:
            try:
                # Calculate start time for this chunk (1000 minutes back)
                chunk_start_time = current_end_time - timedelta(minutes=1000)
                
                # Don't go before our desired start time
                if start_time and chunk_start_time < start_time:
                    chunk_start_time = start_time
                
                klines = self.get_klines(
                    symbol=symbol,
                    interval="1m",
                    limit=1000,
                    start_time=chunk_start_time,
                    end_time=current_end_time
                )
                
                if not klines:
                    log.info(f"No more data available for {symbol}")
                    break
                
                all_data.extend(klines)
                log.debug(f"Fetched {len(klines)} records, total: {len(all_data)}")
                
                # Update end time for next chunk
                oldest_timestamp = min(float(k[0]) for k in klines)
                current_end_time = datetime.fromtimestamp(oldest_timestamp / 1000)
                
                # If we've reached our start time, stop
                if start_time and current_end_time <= start_time:
                    break
                
                # If we got less than 1000 records, we've reached the beginning
                if len(klines) < 1000:
                    break
                    
            except Exception as e:
                log.error(f"Error fetching historical data chunk for {symbol}: {e}")
                break
        
        if not all_data:
            log.warning(f"No historical data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Convert price and volume columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df = df.reset_index(drop=True)
        
        log.info(f"Successfully fetched {len(df)} records for {symbol} from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol"""
        try:
            data = self._make_request("contract/ticker", {"symbol": symbol})
            price = float(data['lastPrice'])
            log.debug(f"Latest price for {symbol}: {price}")
            return price
        except Exception as e:
            log.error(f"Failed to get latest price for {symbol}: {e}")
            raise

# Global MEXC client instance
mexc_client = MEXCClient()