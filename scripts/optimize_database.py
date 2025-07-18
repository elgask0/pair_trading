#!/usr/bin/env python3
"""
Optimizaciones de base de datos para ingesta rápida
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from sqlalchemy import text
from src.utils.logger import get_setup_logger

log = get_setup_logger()

def optimize_for_bulk_insert():
    """Optimizar DB para inserciones masivas"""
    
    with db_manager.get_session() as session:
        log.info("🔧 Optimizing database for bulk inserts...")
        
        # 1. Desactivar autovacuum temporalmente durante ingesta masiva
        session.execute(text("ALTER TABLE ohlcv SET (autovacuum_enabled = false)"))
        session.execute(text("ALTER TABLE orderbook SET (autovacuum_enabled = false)"))
        
        # 2. Aumentar work_mem para operaciones grandes
        session.execute(text("SET work_mem = '256MB'"))
        
        # 3. Aumentar maintenance_work_mem
        session.execute(text("SET maintenance_work_mem = '1GB'"))
        
        # 4. Configurar para bulk operations
        session.execute(text("SET synchronous_commit = OFF"))
        session.execute(text("SET commit_delay = 100000"))  # microseconds
        
        # 5. Drop índices no críticos antes de bulk insert
        try:
            session.execute(text("DROP INDEX IF EXISTS idx_ohlcv_analysis"))
            session.execute(text("DROP INDEX IF EXISTS idx_orderbook_analysis"))
        except:
            pass
        
        session.commit()
        log.info("✅ Database optimized for bulk operations")

def restore_normal_settings():
    """Restaurar configuración normal después de bulk insert"""
    
    with db_manager.get_session() as session:
        log.info("🔧 Restoring normal database settings...")
        
        # Reactivar autovacuum
        session.execute(text("ALTER TABLE ohlcv SET (autovacuum_enabled = true)"))
        session.execute(text("ALTER TABLE orderbook SET (autovacuum_enabled = true)"))
        
        # Restaurar configuración
        session.execute(text("SET synchronous_commit = ON"))
        
        # Recrear índices
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_analysis 
            ON ohlcv(symbol, timestamp) 
            WHERE volume > 0
        """))
        
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_orderbook_analysis 
            ON orderbook(symbol, timestamp) 
            WHERE valid_for_trading = true
        """))
        
        # Ejecutar VACUUM ANALYZE
        session.execute(text("VACUUM ANALYZE ohlcv"))
        session.execute(text("VACUUM ANALYZE orderbook"))
        
        session.commit()
        log.info("✅ Database settings restored")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Optimize for bulk insert")
    parser.add_argument("--restore", action="store_true", help="Restore normal settings")
    
    args = parser.parse_args()
    
    if args.optimize:
        optimize_for_bulk_insert()
    elif args.restore:
        restore_normal_settings()
    else:
        print("Use --optimize or --restore")