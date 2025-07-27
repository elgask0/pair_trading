from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
import yaml
import os
from pathlib import Path

class TradingPairConfig:
    """Configuration for a specific trading pair"""
    def __init__(self, **kwargs):
        self.symbol1 = kwargs.get('symbol1')
        self.symbol2 = kwargs.get('symbol2')
        self.pair_name = f"{self.symbol1}_{self.symbol2}"
        
        # Trading signals
        self.entry_zscore = kwargs.get('entry_zscore', 2.0)
        self.exit_zscore = kwargs.get('exit_zscore', 0.5)
        self.stop_loss_zscore = kwargs.get('stop_loss_zscore', 3.0)
        
        # Position sizing
        self.position_size_pct = kwargs.get('position_size_pct', 0.1)
        
        # Technical analysis windows (in minutes)
        self.correlation_window = kwargs.get('correlation_window', 60)
        self.cointegration_window = kwargs.get('cointegration_window', 1440)
        self.zscore_window = kwargs.get('zscore_window', 60)
        
        # Validation criteria
        self.min_correlation = kwargs.get('min_correlation', 0.7)
        
        # Status and metadata
        self.is_active = kwargs.get('is_active', True)
        self.notes = kwargs.get('notes', '')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol1': self.symbol1,
            'symbol2': self.symbol2,
            'pair_name': self.pair_name,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'stop_loss_zscore': self.stop_loss_zscore,
            'position_size_pct': self.position_size_pct,
            'correlation_window': self.correlation_window,
            'cointegration_window': self.cointegration_window,
            'zscore_window': self.zscore_window,
            'min_correlation': self.min_correlation,
            'is_active': self.is_active,
            'notes': self.notes
        }

class Settings(BaseSettings):
    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "pair_trading"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "password"
    
    # CoinAPI
    COINAPI_KEY: str = ""
    COINAPI_BASE_URL: str = "https://rest.coinapi.io/v1"
    
    # Capital
    INITIAL_CAPITAL: float = 10000.0
    
    # Data Configuration
    FREQUENCY: str = "1min"
    
    # Monitoring
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    LOG_LEVEL: str = "INFO"
    
    # File paths
    SYMBOLS_CONFIG_FILE: str = "config/symbols.yaml"
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # ========================================
    # YAML-BASED FUNCTIONS (for setup_database.py only)
    # ========================================
    
    def _load_symbols_config(self) -> Dict:
        """Load symbols configuration from YAML file"""
        config_path = Path(self.SYMBOLS_CONFIG_FILE)
        if not config_path.exists():
            raise FileNotFoundError(f"Symbols config file not found: {self.SYMBOLS_CONFIG_FILE}")
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get_trading_pairs(self) -> List[TradingPairConfig]:
        """Get all configured trading pairs from YAML file (for initial setup only)"""
        config = self._load_symbols_config()
        pairs = []
        
        for pair_data in config.get('trading_pairs', []):
            pairs.append(TradingPairConfig(**pair_data))
        
        return pairs
    
    def get_active_pairs(self) -> List[TradingPairConfig]:
        """Get only active trading pairs from YAML (for initial setup only)"""
        return [pair for pair in self.get_trading_pairs() if pair.is_active]
    
    def get_all_symbols(self) -> List[str]:
        """Get all unique symbols from all pairs in YAML (for initial setup only)"""
        symbols = set()
        for pair in self.get_trading_pairs():
            symbols.add(pair.symbol1)
            symbols.add(pair.symbol2)
        return list(symbols)
    
    def get_active_symbols(self) -> List[str]:
        """Get all unique symbols from active pairs only in YAML (for initial setup only)"""
        symbols = set()
        for pair in self.get_active_pairs():
            symbols.add(pair.symbol1)
            symbols.add(pair.symbol2)
        return list(symbols)
    
    # ========================================
    # DATABASE-BASED FUNCTIONS (for production scripts)
    # ========================================
    
    def get_symbols_from_db(self) -> List[str]:
        """Get all symbols from symbol_info table"""
        try:
            from src.database.connection import db_manager
            from sqlalchemy import text
            
            with db_manager.get_session() as session:
                result = session.execute(text("""
                    SELECT DISTINCT symbol_id 
                    FROM symbol_info 
                    ORDER BY symbol_id
                """)).fetchall()
                
                symbols = [row.symbol_id for row in result]
                return symbols
                
        except Exception as e:
            print(f"Warning: Could not get symbols from DB: {e}")
            return []
    
    def get_active_symbols_from_db(self) -> List[str]:
        """Get all symbols from symbol_info table (for diagnosis, all symbols are active)"""
        return self.get_symbols_from_db()
    
    def symbol_exists_in_db(self, symbol: str) -> bool:
        """Check if symbol exists in symbol_info table"""
        try:
            from src.database.connection import db_manager
            from sqlalchemy import text
            
            with db_manager.get_session() as session:
                result = session.execute(text("""
                    SELECT 1 FROM symbol_info 
                    WHERE symbol_id = :symbol 
                    LIMIT 1
                """), {'symbol': symbol}).fetchone()
                
                return result is not None
                
        except Exception as e:
            print(f"Warning: Could not check symbol in DB: {e}")
            return False
    
    def get_symbol_info_from_db(self, symbol: str) -> Optional[Dict]:
        """Get symbol information from database"""
        try:
            from src.database.connection import db_manager
            from sqlalchemy import text
            
            with db_manager.get_session() as session:
                result = session.execute(text("""
                    SELECT symbol_id, exchange_id, symbol_type, asset_id_base, 
                           asset_id_quote, data_start
                    FROM symbol_info 
                    WHERE symbol_id = :symbol
                """), {'symbol': symbol}).fetchone()
                
                if result:
                    return {
                        'symbol_id': result.symbol_id,
                        'exchange_id': result.exchange_id,
                        'symbol_type': result.symbol_type,
                        'asset_id_base': result.asset_id_base,
                        'asset_id_quote': result.asset_id_quote,
                        'data_start': result.data_start
                    }
                
                return None
                
        except Exception as e:
            print(f"Warning: Could not get symbol info from DB: {e}")
            return None
    
    # ========================================
    # UTILITY FUNCTIONS
    # ========================================
    
    def use_database_mode(self) -> bool:
        """Check if database is available and should be used"""
        try:
            symbols = self.get_symbols_from_db()
            return len(symbols) > 0
        except:
            return False
    
    def get_best_available_symbols(self) -> List[str]:
        """Get symbols from best available source (DB first, then YAML fallback)"""
        # Try database first
        symbols = self.get_symbols_from_db()
        if symbols:
            return symbols
        
        # Fallback to YAML
        try:
            return self.get_active_symbols()
        except:
            # Ultimate fallback
            return ['MEXCFTS_PERP_GIGA_USDT', 'MEXCFTS_PERP_SPX_USDT']
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()