import requests
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from src.utils.logger import log
from src.utils.exceptions import APIError
from config.settings import settings

class CoinAPIClient:
    """CoinAPI client with symbol info and complete data fetching"""
    
    def __init__(self):
        self.base_url = settings.COINAPI_BASE_URL
        self.api_key = settings.COINAPI_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'X-CoinAPI-Key': self.api_key,
            'Accept': 'application/json'
        })
        self.rate_limit_delay = 1.0
    
    def _safe_request(self, url: str, params: Dict = None, retries: int = 3) -> requests.Response:
        """Safe request with retries and optimized error handling"""
        for attempt in range(retries):
            # Rate limiting more conservative
            time.sleep(self.rate_limit_delay)
            
            try:
                # Intelligent timeout based on endpoint type
                timeout = 60 if 'orderbook' in url else 30
                response = self.session.get(url, params=params, timeout=timeout)
                
                # Enhanced credit monitoring
                remaining = response.headers.get('X-RateLimit-Remaining-Day')
                used = response.headers.get('X-RateLimit-Request-Count-Day')
                limit = response.headers.get('X-RateLimit-Limit-Day')
                
                if remaining and used:
                    log.info(f"API Usage: {used}/{limit} ({remaining} remaining)")
                    
                    # Early warning when approaching limit
                    if int(remaining) < 2000:
                        log.warning(f"APPROACHING LIMIT: Only {remaining} API calls remaining!")
                        # Slow down even more when approaching limit
                        time.sleep(2.0)
                        
                    # Critical stop to preserve remaining calls
                    if int(remaining) < 500:
                        log.error(f"CRITICAL: Stopping ingestion - only {remaining} calls left")
                        log.error(f"Wait for daily reset or upgrade plan")
                        raise APIError(f"API limit protection: {remaining} calls remaining")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 429:
                    # Intelligent rate limiting handling
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait_time = int(retry_after)
                        log.warning(f"Rate limited (429). Waiting {wait_time}s as requested by server")
                    else:
                        # More aggressive exponential backoff for 429s
                        wait_time = min(300, 30 * (2 ** attempt))  # Max 5 minutes
                        log.warning(f"Rate limited (429). Waiting {wait_time}s (exponential backoff)")
                    
                    time.sleep(wait_time)
                    continue
                    
                elif e.response and e.response.status_code == 400:
                    log.error(f"Bad request (400): {e.response.text}")
                    raise APIError(f"Bad request: {e.response.text}")
                raise
                
            except requests.exceptions.Timeout as e:
                if attempt < retries - 1:
                    # Smarter backoff for timeouts
                    wait_time = min(60, 10 * (2 ** attempt))  # 10s, 20s, 40s (max 60s)
                    log.warning(f"Timeout on attempt {attempt + 1}/{retries}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    log.error(f"Request timed out after {retries} attempts: {url}")
                    raise APIError(f"Request timed out after {retries} attempts: {e}")
                    
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s
                    log.warning(f"Request failed ({e}). Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    log.error(f"Request failed after {retries} attempts: {e}")
                    raise APIError(f"Request failed after {retries} attempts: {e}")
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol metadata including data availability dates"""
        # Try individual symbol endpoint first
        url = f"{self.base_url}/symbols/{symbol}"
        
        try:
            response = self._safe_request(url)
            data = response.json()
            
            # If it's a list, find our symbol
            if isinstance(data, list):
                for item in data:
                    if item.get('symbol_id') == symbol:
                        log.info(f"Retrieved symbol info for {symbol}")
                        return item
                raise APIError(f"Symbol {symbol} not found in response")
            
            # If it's a dict, return directly
            elif isinstance(data, dict):
                log.info(f"Retrieved symbol info for {symbol}")
                return data
            
            else:
                raise APIError(f"Unexpected response format for {symbol}")
                
        except APIError:
            # Try searching in symbols list as fallback
            log.warning(f"Direct symbol lookup failed for {symbol}, trying search...")
            return self._search_symbol_info(symbol)
        except Exception as e:
            log.error(f"Failed to get symbol info for {symbol}: {e}")
            raise
    
    def _search_symbol_info(self, symbol: str) -> Dict:
        """Search for symbol in symbols list"""
        url = f"{self.base_url}/symbols"
        params = {"filter_symbol_id": symbol}
        
        try:
            response = self._safe_request(url, params)
            data = response.json()
            
            if isinstance(data, list):
                for item in data:
                    if item.get('symbol_id') == symbol:
                        log.info(f"Found symbol info for {symbol} via search")
                        return item
            
            raise APIError(f"Symbol {symbol} not found in search results")
            
        except Exception as e:
            log.error(f"Failed to search symbol info for {symbol}: {e}")
            raise
    
    def get_available_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get the available date range for a symbol"""
        symbol_info = self.get_symbol_info(symbol)
        
        # Parse dates from symbol info
        data_start_str = symbol_info.get('data_start')
        data_end_str = symbol_info.get('data_end')
        
        if not data_start_str or not data_end_str:
            raise APIError(f"No date range information for {symbol}")
        
        data_start = datetime.fromisoformat(data_start_str.replace('Z', '+00:00')).replace(tzinfo=None)
        data_end = datetime.fromisoformat(data_end_str.replace('Z', '+00:00')).replace(tzinfo=None)
        
        log.info(f"Available data for {symbol}: {data_start} to {data_end}")
        return data_start, data_end
    
    def get_ohlcv_for_date(self, symbol: str, date_str: str) -> pd.DataFrame:
        """Get 1MIN OHLCV data for a specific date"""
        start_time = f"{date_str}T00:00:00"
        end_dt = datetime.fromisoformat(date_str) + timedelta(days=1)
        end_time = end_dt.isoformat()
        
        url = f"{self.base_url}/ohlcv/{symbol}/history"
        params = {
            "period_id": "1MIN",
            "time_start": start_time,
            "time_end": end_time,
            "limit": 1500  # Maximum per day (1440 min + buffer)
        }
        
        try:
            response = self._safe_request(url, params)
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["time_period_start"] = pd.to_datetime(df["time_period_start"])
            df = df.set_index("time_period_start").sort_index()
            
            return df
            
        except Exception as e:
            log.error(f"Failed to get OHLCV for {symbol} on {date_str}: {e}")
            raise
    
    def get_orderbook_for_date(self, symbol: str, date_str: str) -> pd.DataFrame:
        """Get orderbook snapshots for a specific date"""
        url = f"{self.base_url}/orderbooks/{symbol}/history"
        params = {
            "date": date_str,
            "limit_levels": 10
        }
        
        try:
            response = self._safe_request(url, params)
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Process orderbook data
            records = []
            for snapshot in data:
                record = {
                    'timestamp': pd.to_datetime(snapshot['time_exchange']),
                }
                
                # Extract bids (up to 10 levels)
                bids = snapshot.get('bids', [])[:10]
                for i, bid in enumerate(bids, 1):
                    record[f'bid{i}_price'] = bid['price']
                    record[f'bid{i}_size'] = bid['size']
                
                # Fill remaining bid levels with None
                for i in range(len(bids) + 1, 11):
                    record[f'bid{i}_price'] = None
                    record[f'bid{i}_size'] = None
                
                # Extract asks (up to 10 levels)
                asks = snapshot.get('asks', [])[:10]
                for i, ask in enumerate(asks, 1):
                    record[f'ask{i}_price'] = ask['price']
                    record[f'ask{i}_size'] = ask['size']
                
                # Fill remaining ask levels with None
                for i in range(len(asks) + 1, 11):
                    record[f'ask{i}_price'] = None
                    record[f'ask{i}_size'] = None
                
                records.append(record)
            
            if records:
                df = pd.DataFrame(records)
                df = df.set_index('timestamp').sort_index()
                return df
            else:
                return pd.DataFrame()
            
        except Exception as e:
            log.error(f"Failed to get orderbook for {symbol} on {date_str}: {e}")
            raise
    
    def get_all_historical_ohlcv(self, symbol: str, existing_data_end: datetime = None) -> pd.DataFrame:
        """Get ALL available OHLCV data for a symbol"""
        log.info(f"Fetching ALL historical OHLCV data for {symbol}")
        
        # Get available date range
        data_start, data_end = self.get_available_date_range(symbol)
        
        # If we have existing data, start from where we left off
        if existing_data_end:
            data_start = max(data_start, existing_data_end + timedelta(days=1))
            log.info(f"Resuming from {data_start} (existing data until {existing_data_end})")
        
        # Ensure we don't go beyond today
        data_end = min(data_end, datetime.now())
        
        if data_start >= data_end:
            log.info(f"No new data to fetch for {symbol}")
            return pd.DataFrame()
        
        total_days = (data_end.date() - data_start.date()).days + 1
        log.info(f"Will fetch {total_days} days of 1MIN data from {data_start.date()} to {data_end.date()}")
        
        all_dataframes = []
        current_date = data_start.date()
        successful_days = 0
        failed_days = 0
        
        with tqdm(total=total_days, desc=f"OHLCV {symbol}", unit="days") as pbar:
            while current_date <= data_end.date():
                date_str = current_date.isoformat()
                
                try:
                    daily_df = self.get_ohlcv_for_date(symbol, date_str)
                    
                    if not daily_df.empty:
                        all_dataframes.append(daily_df)
                        successful_days += 1
                        
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': successful_days,
                        'Failed': failed_days,
                        'Records': sum(len(df) for df in all_dataframes)
                    })
                    
                except Exception as e:
                    failed_days += 1
                    log.error(f"Error fetching OHLCV {symbol} for {date_str}: {e}")
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    continue
        
        if not all_dataframes:
            log.warning(f"No OHLCV data found for {symbol}")
            return pd.DataFrame()
        
        # Combine and process
        combined_df = pd.concat(all_dataframes).sort_index().reset_index()
        combined_df = combined_df.rename(columns={
            'time_period_start': 'timestamp',
            'price_open': 'open',
            'price_high': 'high',
            'price_low': 'low',
            'price_close': 'close',
            'volume_traded': 'volume'
        })
        
        # Clean and deduplicate
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in combined_df.columns]
        combined_df = combined_df[available_cols].copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        log.info(f"Successfully fetched {len(combined_df)} OHLCV records for {symbol}")
        return combined_df

# Global instance
coinapi_client = CoinAPIClient()