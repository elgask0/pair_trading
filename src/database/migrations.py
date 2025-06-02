from sqlalchemy import create_engine, text
from src.database.models import Base
from src.database.connection import db_manager
from src.utils.logger import log
from src.utils.exceptions import DatabaseError

def create_all_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=db_manager.engine)
        log.info("All database tables created successfully")
        return True
    except Exception as e:
        log.error(f"Failed to create tables: {e}")
        raise DatabaseError(f"Table creation failed: {e}")

def drop_all_tables():
    """Drop all database tables - USE WITH CAUTION"""
    try:
        Base.metadata.drop_all(bind=db_manager.engine)
        log.warning("All database tables dropped")
        return True
    except Exception as e:
        log.error(f"Failed to drop tables: {e}")
        raise DatabaseError(f"Table dropping failed: {e}")

def reset_database():
    """Reset database - drop and recreate all tables"""
    log.warning("Resetting database - all data will be lost!")
    drop_all_tables()
    create_all_tables()
    log.info("Database reset completed")

def add_data_quality_columns():
    """Add data quality columns to existing tables - NON-DESTRUCTIVE"""
    log.info("Adding data quality columns to existing tables...")
    
    with db_manager.get_session() as session:
        try:
            # Add columns to orderbook table if they don't exist
            session.execute(text("""
                ALTER TABLE orderbook 
                ADD COLUMN IF NOT EXISTS liquidity_quality VARCHAR(20),
                ADD COLUMN IF NOT EXISTS valid_for_trading BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS spread_pct FLOAT,
                ADD COLUMN IF NOT EXISTS threshold_p80 FLOAT
            """))
            log.info("Added data quality columns to orderbook table")
            
            # Add indexes for performance if they don't exist
            try:
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_orderbook_quality 
                    ON orderbook(symbol, valid_for_trading, timestamp)
                """))
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_orderbook_spread 
                    ON orderbook(symbol, spread_pct) WHERE spread_pct IS NOT NULL
                """))
                log.info("Added performance indexes for data quality columns")
            except Exception as idx_error:
                log.warning(f"Index creation warning (may already exist): {idx_error}")
            
            return True
            
        except Exception as e:
            log.error(f"Failed to add data quality columns: {e}")
            raise DatabaseError(f"Column addition failed: {e}")

def check_data_quality_schema():
    """Check if data quality columns exist"""
    with db_manager.get_session() as session:
        try:
            result = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'orderbook' 
                AND column_name IN ('liquidity_quality', 'valid_for_trading', 'spread_pct', 'threshold_p80')
            """)).fetchall()
            
            existing_columns = [row[0] for row in result]
            required_columns = ['liquidity_quality', 'valid_for_trading', 'spread_pct', 'threshold_p80']
            missing_columns = [col for col in required_columns if col not in existing_columns]
            
            if missing_columns:
                log.info(f"Missing data quality columns: {missing_columns}")
                return False
            else:
                log.info("All data quality columns exist")
                return True
                
        except Exception as e:
            log.error(f"Failed to check schema: {e}")
            return False