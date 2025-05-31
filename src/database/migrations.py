from sqlalchemy import create_engine
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