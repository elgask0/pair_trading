#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE API & SYMBOL INFO DIAGNOSIS
Diagnóstico completo de API responses, symbol_info table y data processing
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.data.coinapi_client import coinapi_client
from src.data.mexc_client import mexc_client
from src.utils.logger import get_validation_logger
from config.settings import settings

log = get_validation_logger()

def check_symbol_info_table_structure():
    """Verificar estructura de la tabla symbol_info"""
    log.info("🔍 CHECKING SYMBOL_INFO TABLE STRUCTURE")
    log.info("="*60)
    
    with db_manager.get_session() as session:
        try:
            # Verificar si la tabla existe
            table_exists = session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'symbol_info'
                );
            """)).fetchone()[0]
            
            if not table_exists:
                log.error("❌ symbol_info table does NOT exist!")
                return False
            
            log.info("✅ symbol_info table exists")
            
            # Obtener estructura de columnas
            columns_query = text("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_name = 'symbol_info'
                ORDER BY ordinal_position;
            """)
            
            columns = session.execute(columns_query).fetchall()
            
            log.info(f"\n📋 SYMBOL_INFO TABLE STRUCTURE ({len(columns)} columns):")
            log.info("-" * 60)
            for col in columns:
                nullable = "NULL" if col.is_nullable == "YES" else "NOT NULL"
                default = f"DEFAULT {col.column_default}" if col.column_default else ""
                log.info(f"  {col.column_name:<25} {col.data_type:<20} {nullable:<10} {default}")
            
            # Contar registros existentes
            count_result = session.execute(text("SELECT COUNT(*) FROM symbol_info")).fetchone()
            log.info(f"\n📊 Total records in symbol_info: {count_result[0]}")
            
            return True
            
        except Exception as e:
            log.error(f"❌ Error checking symbol_info structure: {e}")
            return False

def check_existing_symbol_info_data():
    """Verificar datos existentes en symbol_info"""
    log.info("\n🔍 CHECKING EXISTING SYMBOL_INFO DATA")
    log.info("="*60)
    
    with db_manager.get_session() as session:
        try:
            # Obtener todos los registros existentes
            existing_query = text("""
                SELECT 
                    symbol_id,
                    exchange_id,
                    symbol_type,
                    data_start,
                    data_end,
                    data_orderbook_start,
                    data_orderbook_end,
                    created_at,
                    updated_at
                FROM symbol_info
                ORDER BY symbol_id;
            """)
            
            existing_data = session.execute(existing_query).fetchall()
            
            if not existing_data:
                log.warning("⚠️ No existing data in symbol_info table")
                return {}
            
            log.info(f"📊 Found {len(existing_data)} existing symbol_info records:")
            log.info("-" * 100)
            
            data_dict = {}
            for row in existing_data:
                symbol = row.symbol_id
                data_dict[symbol] = {
                    'exchange_id': row.exchange_id,
                    'symbol_type': row.symbol_type,
                    'data_start': row.data_start,
                    'data_end': row.data_end,
                    'data_orderbook_start': row.data_orderbook_start,
                    'data_orderbook_end': row.data_orderbook_end,
                    'created_at': row.created_at,
                    'updated_at': row.updated_at
                }
                
                log.info(f"🔸 {symbol}:")
                log.info(f"   Exchange: {row.exchange_id}")
                log.info(f"   Type: {row.symbol_type}")
                log.info(f"   Data range: {row.data_start} → {row.data_end}")
                log.info(f"   Orderbook range: {row.data_orderbook_start} → {row.data_orderbook_end}")
                log.info(f"   Last updated: {row.updated_at}")
                log.info("")
            
            return data_dict
            
        except Exception as e:
            log.error(f"❌ Error checking existing symbol_info data: {e}")
            return {}

def test_coinapi_symbol_info(symbol: str):
    """Testear respuesta real de CoinAPI para symbol info"""
    log.info(f"\n🔍 TESTING COINAPI SYMBOL_INFO FOR {symbol}")
    log.info("="*60)
    
    try:
        # Test del método actual
        log.info(f"📡 Calling coinapi.get_symbol_info('{symbol}')...")
        symbol_info = coinapi_client.get_symbol_info(symbol)
        
        if symbol_info:
            log.info("✅ CoinAPI returned data:")
            log.info("-" * 40)
            
            # Mostrar todos los campos recibidos
            for key, value in symbol_info.items():
                if isinstance(value, datetime):
                    value_str = value.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    value_str = str(value)
                log.info(f"  {key:<25}: {value_str}")
            
            # Verificar campos críticos
            critical_fields = ['symbol_id', 'exchange_id', 'data_start', 'data_end']
            missing_critical = [field for field in critical_fields if not symbol_info.get(field)]
            
            if missing_critical:
                log.warning(f"⚠️ Missing critical fields: {missing_critical}")
            else:
                log.info("✅ All critical fields present")
            
            # Verificar rangos de fechas
            data_start = symbol_info.get('data_start')
            data_end = symbol_info.get('data_end')
            
            if data_start and data_end:
                if isinstance(data_start, str):
                    data_start = pd.to_datetime(data_start)
                if isinstance(data_end, str):
                    data_end = pd.to_datetime(data_end)
                
                days_range = (data_end - data_start).days
                log.info(f"📅 Date range analysis:")
                log.info(f"   Start: {data_start}")
                log.info(f"   End: {data_end}")
                log.info(f"   Total days: {days_range}")
                
                now = datetime.now()
                if data_start < now - timedelta(days=1000):
                    log.warning(f"⚠️ Start date is very old ({data_start.date()})")
                if data_end > now + timedelta(days=1):
                    log.warning(f"⚠️ End date is in the future ({data_end.date()})")
            
            return symbol_info
        else:
            log.error("❌ CoinAPI returned no data")
            return None
            
    except Exception as e:
        log.error(f"❌ Error testing CoinAPI symbol_info: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None

def test_coinapi_date_range_method(symbol: str):
    """Testear el método get_available_date_range"""
    log.info(f"\n🔍 TESTING COINAPI DATE RANGE METHOD FOR {symbol}")
    log.info("="*60)
    
    try:
        log.info(f"📡 Calling coinapi.get_available_date_range('{symbol}')...")
        start_date, end_date = coinapi_client.get_available_date_range(symbol)
        
        if start_date and end_date:
            log.info("✅ Date range method returned:")
            log.info(f"   Start: {start_date}")
            log.info(f"   End: {end_date}")
            
            days_range = (end_date - start_date).days
            log.info(f"   Total days: {days_range}")
            
            # Verificar si el rango es razonable
            now = datetime.now()
            if days_range > 1000:
                log.warning(f"⚠️ Very large date range: {days_range} days")
            if start_date < now - timedelta(days=1095):  # 3 años
                log.warning(f"⚠️ Start date very old: {start_date.date()}")
            
            return start_date, end_date
        else:
            log.error("❌ Date range method returned None")
            return None, None
            
    except Exception as e:
        log.error(f"❌ Error testing date range method: {e}")
        return None, None

def check_actual_data_in_tables(symbol: str):
    """Verificar datos reales en las tablas principales"""
    log.info(f"\n🔍 CHECKING ACTUAL DATA IN TABLES FOR {symbol}")
    log.info("="*60)
    
    with db_manager.get_session() as session:
        try:
            # OHLCV data
            ohlcv_query = text("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    COUNT(DISTINCT DATE(timestamp)) as unique_days
                FROM ohlcv 
                WHERE symbol = :symbol
            """)
            
            ohlcv_result = session.execute(ohlcv_query, {'symbol': symbol}).fetchone()
            
            log.info(f"📊 OHLCV DATA:")
            log.info(f"   Total records: {ohlcv_result.total_records:,}")
            log.info(f"   Date range: {ohlcv_result.min_date} → {ohlcv_result.max_date}")
            log.info(f"   Unique days: {ohlcv_result.unique_days}")
            
            if ohlcv_result.min_date and ohlcv_result.max_date:
                total_possible_days = (ohlcv_result.max_date.date() - ohlcv_result.min_date.date()).days + 1
                coverage_pct = (ohlcv_result.unique_days / total_possible_days) * 100
                log.info(f"   Coverage: {coverage_pct:.1f}%")
            
            # Orderbook data
            orderbook_query = text("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    COUNT(DISTINCT DATE(timestamp)) as unique_days,
                    COUNT(CASE WHEN bid1_price IS NOT NULL AND ask1_price IS NOT NULL THEN 1 END) as valid_quotes
                FROM orderbook 
                WHERE symbol = :symbol
            """)
            
            orderbook_result = session.execute(orderbook_query, {'symbol': symbol}).fetchone()
            
            log.info(f"\n📊 ORDERBOOK DATA:")
            log.info(f"   Total records: {orderbook_result.total_records:,}")
            log.info(f"   Valid quotes: {orderbook_result.valid_quotes:,}")
            log.info(f"   Date range: {orderbook_result.min_date} → {orderbook_result.max_date}")
            log.info(f"   Unique days: {orderbook_result.unique_days}")
            
            if orderbook_result.min_date and orderbook_result.max_date:
                total_possible_days = (orderbook_result.max_date.date() - orderbook_result.min_date.date()).days + 1
                coverage_pct = (orderbook_result.unique_days / total_possible_days) * 100
                log.info(f"   Coverage: {coverage_pct:.1f}%")
            
            # Funding data (if perpetual)
            if "PERP_" in symbol:
                funding_query = text("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(timestamp) as min_date,
                        MAX(timestamp) as max_date,
                        COUNT(DISTINCT DATE(timestamp)) as unique_days
                    FROM funding_rates 
                    WHERE symbol = :symbol
                """)
                
                funding_result = session.execute(funding_query, {'symbol': symbol}).fetchone()
                
                log.info(f"\n📊 FUNDING DATA:")
                log.info(f"   Total records: {funding_result.total_records:,}")
                log.info(f"   Date range: {funding_result.min_date} → {funding_result.max_date}")
                log.info(f"   Unique days: {funding_result.unique_days}")
                
                if funding_result.min_date and funding_result.max_date:
                    total_possible_days = (funding_result.max_date.date() - funding_result.min_date.date()).days + 1
                    # Funding rates are 3 times per day (every 8 hours)
                    expected_records = total_possible_days * 3
                    coverage_pct = (funding_result.total_records / expected_records) * 100
                    log.info(f"   Expected records: {expected_records} (3 per day)")
                    log.info(f"   Coverage: {coverage_pct:.1f}%")
            
            return {
                'ohlcv': ohlcv_result,
                'orderbook': orderbook_result,
                'funding': funding_result if "PERP_" in symbol else None
            }
            
        except Exception as e:
            log.error(f"❌ Error checking actual data: {e}")
            return None

def test_api_endpoints_directly(symbol: str):
    """Testear endpoints de API directamente"""
    log.info(f"\n🔍 TESTING API ENDPOINTS DIRECTLY FOR {symbol}")
    log.info("="*60)
    
    try:
        # Test endpoint directo de symbol info
        api_key = getattr(settings, 'COINAPI_KEY', '')
        base_url = 'https://rest.coinapi.io/v1'
        
        if not api_key:
            log.warning("⚠️ No COINAPI_KEY found in settings")
            return
        
        import requests
        
        headers = {
            'X-CoinAPI-Key': api_key,
            'Accept': 'application/json'
        }
        
        # Test 1: Direct symbol info
        log.info(f"🌐 Testing direct API call: GET /symbols/{symbol}")
        
        try:
            response = requests.get(f"{base_url}/symbols/{symbol}", headers=headers, timeout=10)
            log.info(f"   Status Code: {response.status_code}")
            log.info(f"   Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                data = response.json()
                log.info(f"   Response Type: {type(data)}")
                log.info(f"   Response Length: {len(data) if isinstance(data, list) else 'N/A'}")
                
                if isinstance(data, list) and len(data) > 0:
                    log.info("   First item fields:")
                    for key, value in data[0].items():
                        log.info(f"     {key}: {value}")
                elif isinstance(data, dict):
                    log.info("   Response fields:")
                    for key, value in data.items():
                        log.info(f"     {key}: {value}")
            else:
                log.warning(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            log.error(f"   Error: {e}")
        
        # Test 2: Symbol search
        log.info(f"\n🌐 Testing symbol search: GET /symbols?filter_symbol_id={symbol}")
        
        try:
            response = requests.get(f"{base_url}/symbols", 
                                  headers=headers, 
                                  params={'filter_symbol_id': symbol}, 
                                  timeout=10)
            log.info(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                log.info(f"   Found {len(data)} symbols")
                
                for item in data[:3]:  # Show first 3
                    if item.get('symbol_id') == symbol:
                        log.info(f"   ✅ Exact match found: {item.get('symbol_id')}")
                        log.info(f"      Exchange: {item.get('exchange_id')}")
                        log.info(f"      Type: {item.get('symbol_type')}")
                        break
            else:
                log.warning(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            log.error(f"   Error: {e}")
            
    except Exception as e:
        log.error(f"❌ Error testing API endpoints: {e}")

def test_mexc_api(symbol: str):
    """Testear MEXC API para funding rates"""
    log.info(f"\n🔍 TESTING MEXC API FOR {symbol}")
    log.info("="*60)
    
    if "PERP_" not in symbol:
        log.info("⏭️ Skipping MEXC test - not a perpetual contract")
        return
    
    try:
        # Test current funding rate
        log.info(f"📡 Testing MEXC current funding rate...")
        current_funding = mexc_client.get_current_funding_rate(symbol)
        
        if current_funding:
            log.info("✅ Current funding rate obtained:")
            log.info(f"   Data: {current_funding}")
        else:
            log.warning("⚠️ No current funding rate data")
        
        # Test funding history (recent)
        log.info(f"\n📡 Testing MEXC funding history (last 7 days)...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        recent_funding = mexc_client.get_funding_rate_history_range(symbol, start_date, end_date)
        
        if not recent_funding.empty:
            log.info(f"✅ Recent funding history obtained: {len(recent_funding)} records")
            log.info(f"   Date range: {recent_funding['timestamp'].min()} → {recent_funding['timestamp'].max()}")
            log.info(f"   Sample rates: {recent_funding['funding_rate'].head(3).tolist()}")
        else:
            log.warning("⚠️ No recent funding history data")
            
    except Exception as e:
        log.error(f"❌ Error testing MEXC API: {e}")

def analyze_date_range_issues(symbol: str, existing_symbol_info: dict):
    """Analizar problemas de rangos de fechas"""
    log.info(f"\n🔍 ANALYZING DATE RANGE ISSUES FOR {symbol}")
    log.info("="*60)
    
    try:
        # Comparar symbol_info vs datos reales
        existing_info = existing_symbol_info.get(symbol, {})
        
        if existing_info:
            log.info("📊 SYMBOL_INFO vs ACTUAL DATA COMPARISON:")
            log.info("-" * 50)
            
            # Get actual data ranges
            actual_data = check_actual_data_in_tables(symbol)
            
            if actual_data:
                ohlcv = actual_data['ohlcv']
                orderbook = actual_data['orderbook']
                
                log.info(f"Symbol_info data_start: {existing_info.get('data_start')}")
                log.info(f"Actual OHLCV min:      {ohlcv.min_date}")
                log.info(f"Actual Orderbook min:  {orderbook.min_date}")
                log.info("")
                log.info(f"Symbol_info data_end:   {existing_info.get('data_end')}")
                log.info(f"Actual OHLCV max:       {ohlcv.max_date}")
                log.info(f"Actual Orderbook max:   {orderbook.max_date}")
                
                # Calcular diferencias
                symbol_start = existing_info.get('data_start')
                symbol_end = existing_info.get('data_end')
                
                if symbol_start and ohlcv.min_date:
                    start_diff = (ohlcv.min_date - symbol_start).days
                    log.info(f"\n📅 Start date difference: {start_diff} days")
                    if abs(start_diff) > 30:
                        log.warning(f"⚠️ Large start date difference: {start_diff} days")
                
                if symbol_end and ohlcv.max_date:
                    end_diff = (symbol_end - ohlcv.max_date).days
                    log.info(f"📅 End date difference: {end_diff} days")
                    if abs(end_diff) > 7:
                        log.warning(f"⚠️ Large end date difference: {end_diff} days")
        
        # Testear el rango que se está usando actualmente
        log.info(f"\n🧠 TESTING CURRENT DATE RANGE LOGIC:")
        log.info("-" * 50)
        
        api_start, api_end = test_coinapi_date_range_method(symbol)
        
        if api_start and api_end:
            days_range = (api_end - api_start).days
            log.info(f"Current logic would search: {days_range} days")
            
            if days_range > 365:
                log.warning(f"⚠️ PROBLEM: Searching {days_range} days is too much!")
                log.warning(f"   This could cause massive API calls")
                log.warning(f"   Recommended: Limit to last 365 days maximum")
            
            now = datetime.now()
            if api_start < now - timedelta(days=730):
                log.warning(f"⚠️ PROBLEM: Start date too old: {api_start.date()}")
                log.warning(f"   This could search for non-existent historical data")
        
    except Exception as e:
        log.error(f"❌ Error analyzing date range issues: {e}")

def generate_recommendations():
    """Generar recomendaciones basadas en el diagnóstico"""
    log.info(f"\n💡 RECOMMENDATIONS BASED ON DIAGNOSIS")
    log.info("="*60)
    
    log.info("Based on the diagnosis above, here are the recommended fixes:")
    log.info("")
    
    log.info("1. 🔧 SYMBOL_INFO UPDATE:")
    log.info("   - Validate and sanitize date ranges from API")
    log.info("   - Limit historical searches to max 365 days")
    log.info("   - Use conservative fallbacks for missing data")
    log.info("")
    
    log.info("2. 🧠 SMART DATE RANGE DETECTION:")
    log.info("   - Don't search for data older than 2 years")
    log.info("   - Limit incremental searches to reasonable gaps (<90 days)")
    log.info("   - Use actual data ranges, not API-reported ranges")
    log.info("")
    
    log.info("3. ⚡ FUNDING RATES OPTIMIZATION:")
    log.info("   - Detect when funding data is already complete (>1000 records)")
    log.info("   - Only fetch recent updates if data exists")
    log.info("   - Don't attempt massive historical fetches")
    log.info("")
    
    log.info("4. 🔍 MISSING DAYS LOGIC:")
    log.info("   - Use day-level aggregation to detect real gaps")
    log.info("   - Don't re-detect the same gaps on multiple runs")
    log.info("   - Prioritize recent gaps over old gaps")

def main():
    """Función principal de diagnóstico"""
    import argparse
    
    parser = argparse.ArgumentParser(description="🔍 Comprehensive API & Symbol Info Diagnosis")
    parser.add_argument("--symbol", type=str, help="Specific symbol to diagnose")
    
    args = parser.parse_args()
    
    log.info("🔍 STARTING COMPREHENSIVE DIAGNOSIS")
    log.info("="*80)
    
    # Get symbols to test
    if args.symbol:
        symbols = [args.symbol]
    else:
        try:
            active_pairs = settings.get_active_pairs()
            symbols = []
            for pair in active_pairs:
                symbols.extend([pair.symbol1, pair.symbol2])
            symbols = list(set(symbols))[:2]  # Limit to 2 symbols for diagnosis
        except:
            symbols = ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
    
    log.info(f"🎯 Diagnosing symbols: {symbols}")
    
    try:
        # 1. Check table structure
        table_ok = check_symbol_info_table_structure()
        
        if not table_ok:
            log.error("❌ Cannot proceed - symbol_info table issues")
            return False
        
        # 2. Check existing data
        existing_data = check_existing_symbol_info_data()
        
        # 3. Test each symbol
        for symbol in symbols:
            log.info(f"\n{'='*80}")
            log.info(f"🔬 DETAILED DIAGNOSIS FOR {symbol}")
            log.info(f"{'='*80}")
            
            # Test CoinAPI
            api_symbol_info = test_coinapi_symbol_info(symbol)
            
            # Test date range method
            test_coinapi_date_range_method(symbol)
            
            # Check actual data in tables
            check_actual_data_in_tables(symbol)
            
            # Test API endpoints directly
            test_api_endpoints_directly(symbol)
            
            # Test MEXC API if perpetual
            test_mexc_api(symbol)
            
            # Analyze date range issues
            analyze_date_range_issues(symbol, existing_data)
        
        # 4. Generate recommendations
        generate_recommendations()
        
        log.info(f"\n🎉 DIAGNOSIS COMPLETED!")
        log.info("Check the detailed output above to understand the issues.")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Diagnosis failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)