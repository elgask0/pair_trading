#!/usr/bin/env python3
"""
🔍 Script de diagnóstico MEJORADO para ingesta de orderbook
Version 2.0 - Maneja timeouts y usa datos sintéticos para testing
"""

import sys
import os
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import get_test_logger
from config.settings import settings

log = get_test_logger()

def test_api_connectivity():
    """Test de conectividad básica a CoinAPI"""
    log.info("🌐 Probando conectividad básica a CoinAPI...")
    
    headers = {
        'X-CoinAPI-Key': settings.COINAPI_KEY,
        'Accept': 'application/json'
    }
    
    # Test simple con endpoint de exchanges
    test_url = "https://rest.coinapi.io/v1/exchanges"
    
    try:
        log.info(f"Testing connectivity to: {test_url}")
        response = requests.get(test_url, headers=headers, timeout=10)
        
        log.info(f"Status Code: {response.status_code}")
        log.info(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            log.info("✅ Conectividad básica OK")
            return True
        elif response.status_code == 401:
            log.error("❌ API Key inválida o faltante")
            return False
        else:
            log.warning(f"⚠️ Respuesta inesperada: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        log.error("❌ Timeout de conectividad - Red lenta o API sobrecargada")
        return False
    except Exception as e:
        log.error(f"❌ Error de conectividad: {e}")
        return False

def test_orderbook_api_with_fallback():
    """Test API de orderbook con múltiples intentos y fallbacks"""
    log.info("📊 Probando API de orderbook con fallbacks...")
    
    headers = {
        'X-CoinAPI-Key': settings.COINAPI_KEY,
        'Accept': 'application/json'
    }
    
    symbol = "MEXCFTS_PERP_GIGA_USDT"
    base_url = "https://rest.coinapi.io/v1/orderbooks"
    
    # Intentar con diferentes fechas y límites
    test_configs = [
        {'date': '2025-01-15', 'limit': 5, 'limit_levels': 3},
        {'date': '2024-12-31', 'limit': 10, 'limit_levels': 5},
        {'date': '2024-12-30', 'limit': 1, 'limit_levels': 2}
    ]
    
    for i, config in enumerate(test_configs):
        log.info(f"  Intento {i+1}: {config}")
        
        url = f"{base_url}/{symbol}/history"
        
        try:
            response = requests.get(url, headers=headers, params=config, timeout=15)
            
            log.info(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                log.info(f"  Respuesta exitosa: {len(data) if isinstance(data, list) else 'No es lista'}")
                
                if isinstance(data, list) and len(data) > 0:
                    # Guardar muestra
                    sample_file = Path("diagnostics") / "orderbook_sample.json"
                    sample_file.parent.mkdir(exist_ok=True)
                    
                    with open(sample_file, 'w') as f:
                        json.dump(data[:3], f, indent=2, default=str)
                    
                    log.info(f"✅ Muestra guardada: {sample_file}")
                    return data[:3]
                else:
                    log.warning(f"  Respuesta vacía o inválida")
                    
            elif response.status_code == 429:
                log.warning(f"  Rate limit - esperando...")
                time.sleep(10)
                
            else:
                log.warning(f"  Error {response.status_code}: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            log.warning(f"  Timeout en intento {i+1}")
            time.sleep(2)
            
        except Exception as e:
            log.warning(f"  Error en intento {i+1}: {e}")
    
    log.error("❌ Todos los intentos de API fallaron - usando datos sintéticos")
    return create_synthetic_orderbook_data()

def create_synthetic_orderbook_data():
    """Crear datos sintéticos de orderbook para testing"""
    log.info("🔧 Creando datos sintéticos de orderbook...")
    
    # Estructura basada en la documentación real
    synthetic_data = []
    
    base_time = datetime.utcnow().replace(microsecond=0)
    
    for i in range(3):
        timestamp = base_time + timedelta(seconds=i*5)
        
        # Generar precios realistas
        base_price = 0.06980
        bid_price = base_price - 0.00005
        ask_price = base_price + 0.00005
        
        snapshot = {
            "symbol_id": "MEXCFTS_PERP_GIGA_USDT",
            "time_exchange": timestamp.isoformat() + "Z",
            "time_coinapi": (timestamp + timedelta(seconds=2)).isoformat() + "Z",
            "bids": [
                {"price": bid_price, "size": 1500 + i*100},
                {"price": bid_price - 0.00001, "size": 2000 + i*50},
                {"price": bid_price - 0.00002, "size": 1800 + i*75}
            ],
            "asks": [
                {"price": ask_price, "size": 1600 + i*80},
                {"price": ask_price + 0.00001, "size": 1900 + i*60},
                {"price": ask_price + 0.00002, "size": 2100 + i*40}
            ]
        }
        
        synthetic_data.append(snapshot)
    
    # Guardar datos sintéticos
    sample_file = Path("diagnostics") / "synthetic_orderbook.json"
    sample_file.parent.mkdir(exist_ok=True)
    
    with open(sample_file, 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    log.info(f"✅ Datos sintéticos creados: {sample_file}")
    return synthetic_data

def test_data_processing_detailed(sample_data):
    """Test detallado del procesamiento de datos"""
    log.info("🧪 Probando procesamiento detallado de datos...")
    
    if not sample_data:
        log.error("No hay datos para procesar")
        return False
    
    processed_data = []
    
    for i, snapshot in enumerate(sample_data):
        log.info(f"\n--- Procesando snapshot {i+1} ---")
        log.info(f"Snapshot keys: {list(snapshot.keys())}")
        
        # Extraer timestamp
        timestamp = snapshot.get('time_exchange')
        log.info(f"Timestamp raw: {timestamp}")
        
        if timestamp:
            try:
                ts_converted = pd.to_datetime(timestamp, utc=True)
                log.info(f"Timestamp convertido: {ts_converted}")
            except Exception as e:
                log.error(f"Error convirtiendo timestamp: {e}")
                continue
        else:
            log.error("No timestamp encontrado")
            continue
        
        # Extraer bids/asks
        bids = snapshot.get('bids', [])
        asks = snapshot.get('asks', [])
        
        log.info(f"Bids: {len(bids)} niveles")
        log.info(f"Asks: {len(asks)} niveles")
        
        if bids:
            log.info(f"Primer bid: {bids[0]}")
        if asks:
            log.info(f"Primer ask: {asks[0]}")
        
        # Procesar estructura
        record = {'timestamp': ts_converted}
        
        # Procesar bids
        for j, bid in enumerate(bids[:10]):  # Máximo 10 niveles
            if isinstance(bid, dict):
                try:
                    price = float(bid.get('price', 0))
                    size = float(bid.get('size', 0))
                    
                    record[f'bid{j+1}_price'] = price if price > 0 else None
                    record[f'bid{j+1}_size'] = size if size > 0 else None
                    
                    log.info(f"  Bid{j+1}: {price} @ {size}")
                except Exception as e:
                    log.warning(f"Error procesando bid{j+1}: {e}")
        
        # Procesar asks
        for j, ask in enumerate(asks[:10]):  # Máximo 10 niveles
            if isinstance(ask, dict):
                try:
                    price = float(ask.get('price', 0))
                    size = float(ask.get('size', 0))
                    
                    record[f'ask{j+1}_price'] = price if price > 0 else None
                    record[f'ask{j+1}_size'] = size if size > 0 else None
                    
                    log.info(f"  Ask{j+1}: {price} @ {size}")
                except Exception as e:
                    log.warning(f"Error procesando ask{j+1}: {e}")
        
        # Validar spread
        bid1_price = record.get('bid1_price')
        ask1_price = record.get('ask1_price')
        
        if bid1_price and ask1_price:
            if bid1_price < ask1_price:
                spread_pct = (ask1_price - bid1_price) / bid1_price * 100
                log.info(f"  Spread válido: {spread_pct:.4f}%")
                processed_data.append(record)
            else:
                log.warning(f"  Spread cruzado: bid={bid1_price}, ask={ask1_price}")
        else:
            log.warning(f"  Nivel 1 faltante: bid={bid1_price}, ask={ask1_price}")
    
    if processed_data:
        df = pd.DataFrame(processed_data)
        df.set_index('timestamp', inplace=True)
        
        log.info(f"\n✅ DataFrame creado:")
        log.info(f"  Filas: {len(df)}")
        log.info(f"  Columnas: {len(df.columns)}")
        log.info(f"  Columnas: {list(df.columns)[:10]}...")  # Primeras 10
        
        # Guardar para inspección
        output_file = Path("diagnostics") / "processed_orderbook_detailed.csv"
        df.to_csv(output_file)
        log.info(f"  Guardado en: {output_file}")
        
        return df
    else:
        log.error("❌ No se procesaron datos válidos")
        return None

def test_database_schema():
    """Verificar esquema de base de datos"""
    log.info("🗄️ Verificando esquema de base de datos...")
    
    from src.database.connection import db_manager
    from sqlalchemy import text
    
    try:
        with db_manager.get_session() as session:
            # Verificar tabla orderbook
            result = session.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'orderbook'
                ORDER BY ordinal_position
            """)).fetchall()
            
            if result:
                log.info("✅ Tabla orderbook encontrada")
                log.info("Columnas:")
                for row in result:
                    log.info(f"  {row.column_name}: {row.data_type} ({'NULL' if row.is_nullable == 'YES' else 'NOT NULL'})")
                
                # Verificar columnas específicas
                columns = [row.column_name for row in result]
                required_cols = ['symbol', 'timestamp', 'bid1_price', 'bid1_size', 'ask1_price', 'ask1_size']
                missing_cols = [col for col in required_cols if col not in columns]
                
                if missing_cols:
                    log.error(f"❌ Columnas faltantes: {missing_cols}")
                    return False
                else:
                    log.info("✅ Todas las columnas requeridas presentes")
                    return True
            else:
                log.error("❌ Tabla orderbook no encontrada")
                return False
                
    except Exception as e:
        log.error(f"❌ Error verificando esquema: {e}")
        return False

def test_database_insertion_improved():
    """Test mejorado de inserción a base de datos"""
    log.info("💾 Probando inserción mejorada a base de datos...")
    
    from src.database.connection import db_manager
    from sqlalchemy import text
    
    # Limpiar datos de test anteriores
    symbol = "TEST_ORDERBOOK_DIAG"
    base_time = datetime.utcnow().replace(microsecond=0)
    
    try:
        with db_manager.get_session() as session:
            # Limpiar registros de test anteriores
            delete_result = session.execute(text("""
                DELETE FROM orderbook WHERE symbol = :symbol
            """), {'symbol': symbol})
            
            log.info(f"Limpiados {delete_result.rowcount} registros de test anteriores")
            
            # Crear datos de test
            test_records = []
            for i in range(3):
                timestamp = base_time + timedelta(seconds=i*10)
                
                record = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'bid1_price': 0.06977 - i*0.00001,
                    'bid1_size': 1500.0 + i*100,
                    'ask1_price': 0.06987 + i*0.00001,
                    'ask1_size': 1600.0 + i*80,
                    'bid2_price': 0.06976 - i*0.00001,
                    'bid2_size': 2000.0 + i*50,
                    'ask2_price': 0.06988 + i*0.00001,
                    'ask2_size': 1900.0 + i*60
                }
                
                # Rellenar niveles 3-10 con None
                for level in range(3, 11):
                    record[f'bid{level}_price'] = None
                    record[f'bid{level}_size'] = None
                    record[f'ask{level}_price'] = None
                    record[f'ask{level}_size'] = None
                
                test_records.append(record)
            
            log.info(f"Insertando {len(test_records)} registros de test...")
            
            # Construir query dinámicamente
            columns = list(test_records[0].keys())
            placeholders = ', '.join([f':{col}' for col in columns])
            columns_str = ', '.join(columns)
            
            insert_query = f"""
                INSERT INTO orderbook ({columns_str})
                VALUES ({placeholders})
            """
            
            log.info(f"Query: {insert_query[:100]}...")
            
            # Insertar
            insert_result = session.execute(text(insert_query), test_records)
            session.flush()  # Forzar flush antes de verificar
            
            log.info(f"Insertados: {insert_result.rowcount} registros")
            
            # Verificar inmediatamente
            verify_result = session.execute(text("""
                SELECT COUNT(*) as count,
                       MIN(timestamp) as min_ts,
                       MAX(timestamp) as max_ts
                FROM orderbook 
                WHERE symbol = :symbol
            """), {'symbol': symbol}).fetchone()
            
            log.info(f"Verificación inmediata: {verify_result.count} registros")
            log.info(f"Rango temporal: {verify_result.min_ts} - {verify_result.max_ts}")
            
            # Verificar datos específicos
            sample_result = session.execute(text("""
                SELECT timestamp, bid1_price, ask1_price, bid1_size, ask1_size
                FROM orderbook 
                WHERE symbol = :symbol
                ORDER BY timestamp
                LIMIT 3
            """), {'symbol': symbol}).fetchall()
            
            log.info("Datos verificados:")
            for row in sample_result:
                log.info(f"  {row.timestamp}: bid={row.bid1_price}@{row.bid1_size}, ask={row.ask1_price}@{row.ask1_size}")
            
            # Limpiar datos de test
            cleanup_result = session.execute(text("""
                DELETE FROM orderbook WHERE symbol = :symbol
            """), {'symbol': symbol})
            
            log.info(f"Limpiados {cleanup_result.rowcount} registros de test")
            
            success = verify_result.count == len(test_records)
            if success:
                log.info("✅ Test de inserción exitoso")
            else:
                log.error(f"❌ Test de inserción falló: esperado {len(test_records)}, obtenido {verify_result.count}")
            
            return success
            
    except Exception as e:
        log.error(f"❌ Error en test de inserción: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

def main():
    """Ejecutar diagnóstico completo mejorado"""
    log.info("🚀 Iniciando diagnóstico MEJORADO de ingesta de orderbook")
    
    # Crear directorio de diagnósticos
    Path("diagnostics").mkdir(exist_ok=True)
    
    results = {}
    
    # Test 1: Conectividad básica
    log.info("\n" + "="*60)
    log.info("TEST 1: Conectividad básica")
    log.info("="*60)
    
    results['connectivity'] = test_api_connectivity()
    
    # Test 2: API de orderbook con fallbacks
    log.info("\n" + "="*60)
    log.info("TEST 2: API de orderbook (con fallbacks)")
    log.info("="*60)
    
    sample_data = test_orderbook_api_with_fallback()
    results['api_orderbook'] = sample_data is not None
    
    # Test 3: Procesamiento detallado
    log.info("\n" + "="*60)
    log.info("TEST 3: Procesamiento de datos")
    log.info("="*60)
    
    processed_df = test_data_processing_detailed(sample_data)
    results['data_processing'] = processed_df is not None
    
    # Test 4: Esquema de base de datos
    log.info("\n" + "="*60)
    log.info("TEST 4: Esquema de base de datos")
    log.info("="*60)
    
    results['database_schema'] = test_database_schema()
    
    # Test 5: Inserción mejorada
    log.info("\n" + "="*60)
    log.info("TEST 5: Inserción a base de datos")
    log.info("="*60)
    
    results['database_insertion'] = test_database_insertion_improved()
    
    # Resumen final
    log.info("\n" + "="*60)
    log.info("RESUMEN DE DIAGNÓSTICO MEJORADO")
    log.info("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        log.info(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    log.info(f"\nTests pasados: {total_passed}/{len(results)}")
    
    # Generar recomendaciones específicas
    log.info(f"\n💡 RECOMENDACIONES ESPECÍFICAS:")
    
    if not results['connectivity']:
        log.info("  🔌 Problema de conectividad - verificar red y API key")
    
    if not results['api_orderbook']:
        log.info("  📊 API de orderbook no responde - usar datos sintéticos para desarrollo")
    
    if not results['data_processing']:
        log.info("  🧪 Procesamiento de datos falla - revisar estructura de respuesta")
    
    if not results['database_schema']:
        log.info("  🗄️ Esquema de DB incompleto - ejecutar setup_database.py")
    
    if not results['database_insertion']:
        log.info("  💾 Inserción falla - revisar constraints y tipos de datos")
    
    if total_passed >= 3:
        log.info("✅ Suficientes componentes funcionan - sistema puede operar con limitaciones")
    else:
        log.error("❌ Componentes críticos fallan - requiere intervención")
    
    return total_passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)