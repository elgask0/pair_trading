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

def create_funding_rates_table():
    """Create funding rates table if it doesn't exist"""
    try:
        with db_manager.get_session() as session:
            # Check if table already exists
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'funding_rates'
            """)).fetchone()
            
            if result:
                log.info("Funding rates table already exists")
                return True
            
            # Create the table
            session.execute(text("""
                CREATE TABLE funding_rates (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    funding_rate FLOAT NOT NULL,
                    collect_cycle INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                );
            """))
            
            # Create indexes
            session.execute(text("""
                CREATE INDEX idx_funding_symbol_timestamp 
                ON funding_rates(symbol, timestamp);
            """))
            session.execute(text("""
                CREATE INDEX idx_funding_timestamp 
                ON funding_rates(timestamp);
            """))
            
            log.info("Funding rates table created successfully")
            return True
            
    except Exception as e:
        log.error(f"Failed to create funding rates table: {e}")
        raise DatabaseError(f"Funding rates table creation failed: {e}")

def check_funding_rates_schema():
    """Check if funding rates table exists and has correct schema"""
    with db_manager.get_session() as session:
        try:
            # Check if table exists
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'funding_rates'
            """)).fetchone()
            
            if not result:
                log.info("Funding rates table does not exist")
                return False
            
            # Check if all required columns exist
            columns_result = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'funding_rates'
                AND column_name IN ('symbol', 'timestamp', 'funding_rate', 'collect_cycle')
            """)).fetchall()
            
            existing_columns = [row[0] for row in columns_result]
            required_columns = ['symbol', 'timestamp', 'funding_rate', 'collect_cycle']
            missing_columns = [col for col in required_columns if col not in existing_columns]
            
            if missing_columns:
                log.warning(f"Funding rates table exists but missing columns: {missing_columns}")
                return False
            
            log.info("Funding rates table schema is correct")
            return True
                
        except Exception as e:
            log.error(f"Failed to check funding rates schema: {e}")
            return False

def update_schema_for_funding():
    """Update database schema to include funding rates functionality"""
    log.info("Updating schema for funding rates...")
    
    try:
        # Create funding rates table if it doesn't exist
        if not check_funding_rates_schema():
            log.info("Creating funding rates table...")
            create_funding_rates_table()
        else:
            log.info("Funding rates schema is up to date")
        
        # Also ensure data quality columns exist
        if not check_data_quality_schema():
            log.info("Adding missing data quality columns...")
            add_data_quality_columns()
        else:
            log.info("Data quality schema is up to date")
        
        log.info("Schema update for funding rates completed successfully")
        return True
        
    except Exception as e:
        log.error(f"Failed to update schema for funding rates: {e}")
        return False

def migrate_funding_rates_data():
    """Migrate existing funding rates data if needed - placeholder for future migrations"""
    log.info("Checking if funding rates data migration is needed...")
    
    # This is a placeholder for future data migrations
    # For example, if we need to change data formats or add computed columns
    
    try:
        with db_manager.get_session() as session:
            # Example: Check if we need to migrate data format
            # For now, just log that no migration is needed
            log.info("No funding rates data migration needed")
            return True
            
    except Exception as e:
        log.error(f"Funding rates data migration failed: {e}")
        return False

# REEMPLAZAR la función run_all_migrations() existente por esta:

def run_all_migrations():
    """Run all necessary migrations to bring schema up to date - UPDATED"""
    log.info("Running all database migrations...")
    
    try:
        # 1. Create base tables
        create_all_tables()
        
        # 2. Add data quality columns if needed
        if not check_data_quality_schema():
            add_data_quality_columns()
        
        # 3. Create funding rates table if needed  
        if not check_funding_rates_schema():
            create_funding_rates_table()
        
        # 4. Create mark prices table if needed (NEW)
        if not check_mark_prices_schema():
            create_mark_prices_table()
        
        # 5. Run any data migrations
        migrate_funding_rates_data()
        
        log.info("All migrations completed successfully")
        
        # Log final schema state
        schema_info = get_schema_version()
        log.info(f"Final schema state: {schema_info}")
        
        return True
        
    except Exception as e:
        log.error(f"Migration failed: {e}")
        return False

# REEMPLAZAR la función get_schema_version() existente por esta:

def get_schema_version():
    """Get current schema version - UPDATED"""
    try:
        with db_manager.get_session() as session:
            # Check what tables exist to determine schema version
            tables_result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)).fetchall()
            
            existing_tables = [row[0] for row in tables_result]
            
            schema_info = {
                'has_base_tables': 'symbols' in existing_tables and 'ohlcv' in existing_tables,
                'has_orderbook': 'orderbook' in existing_tables,
                'has_funding_rates': 'funding_rates' in existing_tables,
                'has_mark_prices': 'mark_prices' in existing_tables,  # NEW
                'has_data_quality': check_data_quality_schema(),
                'total_tables': len(existing_tables),
                'table_list': existing_tables
            }
            
            return schema_info
            
    except Exception as e:
        log.error(f"Failed to get schema version: {e}")
        return {'error': str(e)}
        
def create_mark_prices_table():
    """Create mark prices table if it doesn't exist"""
    try:
        with db_manager.get_session() as session:
            # Check if table already exists
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'mark_prices'
            """)).fetchone()
            
            if result:
                log.info("Mark prices table already exists")
                return True
            
            # Create the table
            session.execute(text("""
                CREATE TABLE mark_prices (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    mark_price FLOAT NOT NULL,
                    orderbook_mid FLOAT,
                    ohlcv_close FLOAT,
                    bid_ask_spread_pct FLOAT,
                    price_deviation_pct FLOAT,
                    liquidity_score FLOAT,
                    is_valid BOOLEAN DEFAULT TRUE,
                    validation_source VARCHAR(30),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                );
            """))
            
            # Create indexes
            session.execute(text("""
                CREATE INDEX idx_markprice_symbol_timestamp 
                ON mark_prices(symbol, timestamp);
            """))
            session.execute(text("""
                CREATE INDEX idx_markprice_timestamp 
                ON mark_prices(timestamp);
            """))
            session.execute(text("""
                CREATE INDEX idx_markprice_valid 
                ON mark_prices(symbol, is_valid, timestamp);
            """))
            session.execute(text("""
                CREATE INDEX idx_markprice_quality 
                ON mark_prices(symbol, liquidity_score, timestamp);
            """))
            
            log.info("Mark prices table created successfully")
            return True
            
    except Exception as e:
        log.error(f"Failed to create mark prices table: {e}")
        raise DatabaseError(f"Mark prices table creation failed: {e}")

def check_mark_prices_schema():
    """Check if mark prices table exists and has correct schema"""
    with db_manager.get_session() as session:
        try:
            # Check if table exists
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'mark_prices'
            """)).fetchone()
            
            if not result:
                log.info("Mark prices table does not exist")
                return False
            
            # Check if all required columns exist
            columns_result = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'mark_prices'
                AND column_name IN ('symbol', 'timestamp', 'mark_price', 'is_valid')
            """)).fetchall()
            
            existing_columns = [row[0] for row in columns_result]
            required_columns = ['symbol', 'timestamp', 'mark_price', 'is_valid']
            missing_columns = [col for col in required_columns if col not in existing_columns]
            
            if missing_columns:
                log.warning(f"Mark prices table exists but missing columns: {missing_columns}")
                return False
            
            log.info("Mark prices table schema is correct")
            return True
                
        except Exception as e:
            log.error(f"Failed to check mark prices schema: {e}")
            return False