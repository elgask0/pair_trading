#!/usr/bin/env python3
"""
🎯 INTERACTIVE SYMBOL CHOOSER - Version 3.0
Permite búsqueda manual de assets para insertar en symbol_info
"""

import sys
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
from sqlalchemy import text

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.data.coinapi_client import coinapi_client
from config.settings import settings
from src.utils.logger import setup_logger

log = setup_logger("choose_symbols")

class InteractiveSymbolChooser:
    """Selector interactivo de símbolos usando CoinAPI"""
    
    def __init__(self, default_exchange: str = "MEXCFTS"):
        self.coinapi = coinapi_client
        self.default_exchange = default_exchange
        
    def search_candidates(self, asset_base: str, exchange_filter: str = None) -> List[Dict]:
        """Busca candidatos en CoinAPI para un asset base"""
        exchange_filter = exchange_filter or self.default_exchange
        log.info(f"🔍 Searching CoinAPI for asset: {asset_base} on {exchange_filter}")
        
        try:
            url = f"{self.coinapi.base_url}/symbols"
            params = {
                'filter_asset_id': asset_base.upper(),
                'filter_exchange_id': exchange_filter
            }
            
            data = self.coinapi._make_request_with_retries(
                url, params, f"search_candidates_{asset_base}"
            )
            
            if not data:
                log.warning(f"No data returned for {asset_base} on {exchange_filter}")
                return []
            
            candidates = []
            
            for symbol_info in data:
                if not isinstance(symbol_info, dict):
                    continue
                
                # Solo necesitamos los campos de la tabla actual
                candidate = {
                    'symbol_id': symbol_info.get('symbol_id', ''),
                    'exchange_id': symbol_info.get('exchange_id', ''),
                    'symbol_type': symbol_info.get('symbol_type', ''),
                    'asset_id_base': symbol_info.get('asset_id_base', ''),
                    'asset_id_quote': symbol_info.get('asset_id_quote', ''),
                    'data_start': symbol_info.get('data_start', '')
                }
                
                # Filtrar que tenga los campos mínimos requeridos
                if (candidate['symbol_id'] and 
                    candidate['exchange_id'] and 
                    candidate['asset_id_base']):
                    candidates.append(candidate)
            
            log.info(f"✅ Found {len(candidates)} candidates for {asset_base} on {exchange_filter}")
            return candidates
            
        except Exception as e:
            log.error(f"Error searching candidates for {asset_base}: {e}")
            return []
    
    def filter_and_sort_candidates(self, candidates: List[Dict], symbol_type: str = None) -> List[Dict]:
        """Filtra y ordena candidatos por calidad"""
        if symbol_type:
            # Mapear tipos de símbolo
            type_mapping = {
                'PERPETUAL': 'PERPETUAL',
                'FUTURES': 'PERPETUAL',  # Some APIs return FUTURES for perpetuals
                'SPOT': 'SPOT'
            }
            
            candidates = [c for c in candidates if type_mapping.get(c.get('symbol_type'), c.get('symbol_type')) == symbol_type]
        
        # Scoring system simple
        def score_candidate(candidate):
            score = 0
            
            # Preferir PERPETUAL sobre SPOT para trading de futuros
            if candidate.get('symbol_type') in ['PERPETUAL', 'FUTURES']:
                score += 100
            elif candidate.get('symbol_type') == 'SPOT':
                score += 50
            
            # Preferir símbolos con data_start disponible
            if candidate.get('data_start'):
                score += 20
                # Preferir datos más antiguos (más historial)
                try:
                    start_date = datetime.fromisoformat(candidate['data_start'].replace('Z', ''))
                    days_old = (datetime.now() - start_date).days
                    score += min(days_old // 10, 50)  # Hasta 50 puntos por antigüedad
                except:
                    pass
            
            return score
        
        candidates.sort(key=score_candidate, reverse=True)
        return candidates
    
    def interactive_selection_for_asset(self, asset_base: str, exchange_filter: str = None) -> Optional[str]:
        """Selección interactiva para un asset específico"""
        exchange_filter = exchange_filter or self.default_exchange
        
        print(f"\n{'='*80}")
        print(f"🎯 SEARCHING SYMBOLS FOR ASSET: {asset_base} (Exchange: {exchange_filter})")
        print(f"{'='*80}")
        
        # Check if already exists in database
        existing = self._check_existing_symbols(asset_base)
        if existing:
            print(f"\n📋 EXISTING SYMBOLS IN DATABASE:")
            for i, sym in enumerate(existing):
                data_start = sym['data_start'].strftime('%Y-%m-%d') if sym['data_start'] else 'N/A'
                print(f"  [{i}] {sym['symbol_id']} ({sym['exchange_id']}, {sym['symbol_type']}) - from {data_start}")
            
            use_existing = input(f"\nUse existing symbols? (y/n/s=skip/c=change exchange): ").strip().lower()
            if use_existing == 'y':
                print("Using existing symbols - no changes made")
                return 'existing'
            elif use_existing == 's':
                return None
            elif use_existing == 'c':
                new_exchange = input(f"Enter exchange (current: {exchange_filter}): ").strip().upper()
                if new_exchange:
                    exchange_filter = new_exchange
                    print(f"Changed to exchange: {exchange_filter}")
        
        # Search for candidates
        candidates = self.search_candidates(asset_base, exchange_filter)
        if not candidates:
            print(f"❌ No candidates found for {asset_base} on {exchange_filter}")
            
            # Offer to try other exchanges
            retry = input(f"Try other exchanges? (y/n): ").strip().lower()
            if retry == 'y':
                other_exchanges = ['BINANCE', 'COINBASE', 'KRAKEN', 'BYBIT', 'OKX']
                for exchange in other_exchanges:
                    if exchange != exchange_filter:
                        print(f"Trying {exchange}...")
                        candidates = self.search_candidates(asset_base, exchange)
                        if candidates:
                            exchange_filter = exchange
                            break
            
            if not candidates:
                return None
        
        # Group by symbol type
        spot_candidates = self.filter_and_sort_candidates(candidates, 'SPOT')
        perpetual_candidates = self.filter_and_sort_candidates(candidates, 'PERPETUAL')
        
        print(f"\n📊 SEARCH RESULTS on {exchange_filter}: {len(candidates)} total candidates")
        
        # Show SPOT options
        if spot_candidates:
            print(f"\n💰 SPOT CANDIDATES ({len(spot_candidates)}):")
            for i, candidate in enumerate(spot_candidates[:10]):  # Top 10
                data_info = self._format_data_info(candidate)
                print(f"  [S{i}] {candidate['symbol_id']} {data_info}")
        
        # Show PERPETUAL options
        if perpetual_candidates:
            print(f"\n🚀 PERPETUAL CANDIDATES ({len(perpetual_candidates)}):")
            for i, candidate in enumerate(perpetual_candidates[:10]):  # Top 10
                data_info = self._format_data_info(candidate)
                print(f"  [P{i}] {candidate['symbol_id']} {data_info}")
        
        # Interactive selection
        while True:
            print(f"\nOptions:")
            if spot_candidates:
                print(f"  S<n> - Select SPOT candidate number n")
            if perpetual_candidates:
                print(f"  P<n> - Select PERPETUAL candidate number n")
            print(f"  change - Change exchange filter")
            print(f"  cancel - Cancel selection")
            
            choice = input(f"\nYour choice for {asset_base}: ").strip().lower()
            
            if choice == 'cancel':
                return None
            elif choice == 'change':
                new_exchange = input(f"Enter exchange (current: {exchange_filter}): ").strip().upper()
                if new_exchange:
                    return self.interactive_selection_for_asset(asset_base, new_exchange)
            elif choice.startswith('s') and choice[1:].isdigit():
                idx = int(choice[1:])
                if 0 <= idx < len(spot_candidates):
                    selected = spot_candidates[idx]
                    return self._confirm_and_insert(selected, asset_base)
                else:
                    print(f"Invalid index. Use 0-{len(spot_candidates)-1}")
            elif choice.startswith('p') and choice[1:].isdigit():
                idx = int(choice[1:])
                if 0 <= idx < len(perpetual_candidates):
                    selected = perpetual_candidates[idx]
                    return self._confirm_and_insert(selected, asset_base)
                else:
                    print(f"Invalid index. Use 0-{len(perpetual_candidates)-1}")
            else:
                print("Invalid choice. Try again.")
    
    def _format_data_info(self, candidate: Dict) -> str:
        """Formatea información de disponibilidad de datos"""
        data_start = candidate.get('data_start', '')
        if data_start:
            try:
                # Extraer solo la fecha (YYYY-MM-DD)
                if 'T' in data_start:
                    date_str = data_start.split('T')[0]
                else:
                    date_str = data_start[:10]
                return f"[from {date_str}]"
            except:
                return "[data available]"
        return "[no data info]"
    
    def _check_existing_symbols(self, asset_base: str) -> List[Dict]:
        """Verifica si ya existen símbolos para este asset en la BD"""
        try:
            with db_manager.get_session() as session:
                result = session.execute(text("""
                    SELECT symbol_id, exchange_id, symbol_type, asset_id_base, data_start
                    FROM symbol_info 
                    WHERE asset_id_base = :asset_base
                    ORDER BY symbol_type, exchange_id
                """), {'asset_base': asset_base}).fetchall()
                
                return [
                    {
                        'symbol_id': row.symbol_id,
                        'exchange_id': row.exchange_id,
                        'symbol_type': row.symbol_type,
                        'asset_id_base': row.asset_id_base,
                        'data_start': row.data_start
                    }
                    for row in result
                ]
        except Exception as e:
            log.warning(f"Error checking existing symbols: {e}")
            return []
    
    def _confirm_and_insert(self, candidate: Dict, asset_base: str) -> str:
        """Confirma selección e inserta en BD"""
        print(f"\n✅ SELECTED: {candidate['symbol_id']}")
        print(f"   Exchange: {candidate['exchange_id']}")
        print(f"   Type: {candidate['symbol_type']}")
        print(f"   Base/Quote: {candidate['asset_id_base']}/{candidate['asset_id_quote']}")
        if candidate.get('data_start'):
            print(f"   Data from: {candidate['data_start']}")
        
        confirm = input(f"\nConfirm insertion? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Selection cancelled")
            return None
        
        # Insert into database using only the fields that exist in the table
        try:
            with db_manager.get_session() as session:
                # Parse data_start date
                data_start = self._parse_date(candidate.get('data_start'))
                
                session.execute(text("""
                    INSERT INTO symbol_info (
                        symbol_id, exchange_id, symbol_type, 
                        asset_id_base, asset_id_quote, data_start
                    ) VALUES (
                        :symbol_id, :exchange_id, :symbol_type,
                        :asset_id_base, :asset_id_quote, :data_start
                    ) ON CONFLICT (symbol_id) DO UPDATE SET
                        exchange_id = EXCLUDED.exchange_id,
                        symbol_type = EXCLUDED.symbol_type,
                        asset_id_base = EXCLUDED.asset_id_base,
                        asset_id_quote = EXCLUDED.asset_id_quote,
                        data_start = EXCLUDED.data_start
                """), {
                    'symbol_id': candidate['symbol_id'],
                    'exchange_id': candidate['exchange_id'],
                    'symbol_type': candidate['symbol_type'],
                    'asset_id_base': candidate['asset_id_base'],
                    'asset_id_quote': candidate['asset_id_quote'],
                    'data_start': data_start
                })
                session.commit()
                
                log.info(f"✅ Inserted {candidate['symbol_id']} into symbol_info table")
                print(f"✅ Successfully added to database!")
                return candidate['symbol_id']
                
        except Exception as e:
            log.error(f"Error inserting symbol: {e}")
            print(f"❌ Error inserting into database: {e}")
            return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime - simplified for data_start only"""
        if not date_str:
            return None
        
        try:
            # Handle different date formats from API
            if 'T' in date_str:
                # "2024-11-12T00:00:00.0000000Z" -> "2024-11-12"
                date_only = date_str.split('T')[0]
            else:
                # "2024-11-12" -> "2024-11-12"
                date_only = date_str[:10]
            
            return datetime.strptime(date_only, '%Y-%m-%d')
        except Exception as e:
            log.debug(f"Could not parse date: {date_str}, error: {e}")
            return None
    
    def run_manual_selection(self):
        """Ejecuta selección manual - el usuario ingresa el asset"""
        print("🎯 INTERACTIVE SYMBOL CHOOSER - MANUAL MODE")
        print("=" * 60)
        print(f"Default exchange: {self.default_exchange}")
        
        # Test database connection
        if not db_manager.test_connection():
            log.error("❌ Database connection failed")
            return False
        
        while True:
            print(f"\n{'='*50}")
            asset_input = input("Enter asset to search (or 'quit' to exit): ").strip().upper()
            
            if asset_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if not asset_input:
                print("Please enter a valid asset name")
                continue
            
            print(f"Searching for asset: {asset_input}")
            result = self.interactive_selection_for_asset(asset_input)
            
            if result == 'existing':
                print(f"✅ Asset {asset_input} already exists in database")
            elif result:
                print(f"✅ Successfully added {result} for asset {asset_input}")
            else:
                print(f"❌ No symbol selected for asset {asset_input}")
            
            # Ask if want to continue
            continue_search = input(f"\nSearch for another asset? (y/n): ").strip().lower()
            if continue_search != 'y':
                break
        
        return True

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="🎯 Interactive Symbol Chooser")
    parser.add_argument("--asset", type=str, help="Search for specific asset directly (e.g., GIGA)")
    parser.add_argument("--exchange", type=str, default="MEXCFTS", help="Default exchange filter (default: MEXCFTS)")
    parser.add_argument("--list-existing", action="store_true", help="List existing symbols in database")
    
    args = parser.parse_args()
    
    chooser = InteractiveSymbolChooser(default_exchange=args.exchange)
    
    if args.list_existing:
        # List existing symbols
        try:
            with db_manager.get_session() as session:
                result = session.execute(text("""
                    SELECT symbol_id, exchange_id, symbol_type, asset_id_base, asset_id_quote, data_start
                    FROM symbol_info 
                    ORDER BY asset_id_base, symbol_type, exchange_id
                """)).fetchall()
                
                if result:
                    print("📋 EXISTING SYMBOLS IN DATABASE:")
                    print("-" * 80)
                    current_asset = None
                    for row in result:
                        if row.asset_id_base != current_asset:
                            if current_asset is not None:
                                print()
                            current_asset = row.asset_id_base
                            print(f"Asset: {current_asset}")
                        
                        data_start = row.data_start.strftime('%Y-%m-%d') if row.data_start else 'N/A'
                        print(f"  {row.symbol_id} ({row.exchange_id}, {row.symbol_type}) - from {data_start}")
                else:
                    print("No symbols found in database")
                    
        except Exception as e:
            log.error(f"Error listing symbols: {e}")
        
        return True
    
    elif args.asset:
        # Search for specific asset directly
        result = chooser.interactive_selection_for_asset(args.asset)
        if result == 'existing':
            print(f"✅ Asset {args.asset} already exists in database")
            return True
        elif result:
            print(f"✅ Successfully added {result} for asset {args.asset}")
            return True
        else:
            print(f"❌ No symbol selected for asset {args.asset}")
            return False
    
    else:
        # Run manual selection mode
        return chooser.run_manual_selection()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)