from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
from config.settings import settings
from src.utils.logger import log
from src.utils.exceptions import DatabaseError

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(
                settings.database_url,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False
            )
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
            log.info("Database connection initialized")
        except Exception as e:
            log.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            log.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            log.info("Database connection test successful")
            return True
        except Exception as e:
            log.error(f"Database connection test failed: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()