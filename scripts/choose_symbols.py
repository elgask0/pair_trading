#!/usr/bin/env python3
"""
🎯 INTERACTIVE SYMBOL CHOOSER - Version 3.1 WITH DATE EDITOR - FIXED
Permite búsqueda manual de assets para insertar en symbol_info
NUEVO: Editor interactivo de fechas de inicio (data_start) - FIXED UPDATED_AT
"""

import sys
import os
import argparse
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
from sqlalchemy import text

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import db_manager
from src.data.coinapi_client import coinapi_client
from config.settings import settings
from src.utils.logger import get_logger

log = get_logger()

class InteractiveSymbolChooser:
    """Selector interactivo de símbolos usando CoinAPI + Editor de fechas"""
    
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
    
    def list_all_symbols_with_dates(self) -> List[Dict]:
        """NUEVO: Lista todos los símbolos con sus fechas de inicio"""
        try:
            with db_manager.get_session() as session:
                result = session.execute(text("""
                    SELECT symbol_id, exchange_id, symbol_type, asset_id_base, asset_id_quote, data_start
                    FROM symbol_info 
                    ORDER BY asset_id_base, symbol_type, exchange_id
                """)).fetchall()
                
                return [
                    {
                        'symbol_id': row.symbol_id,
                        'exchange_id': row.exchange_id,
                        'symbol_type': row.symbol_type,
                        'asset_id_base': row.asset_id_base,
                        'asset_id_quote': row.asset_id_quote,
                        'data_start': row.data_start
                    }
                    for row in result
                ]
        except Exception as e:
            log.error(f"Error listing symbols: {e}")
            return []
    
    def interactive_date_editor(self) -> bool:
        """NUEVO: Editor interactivo de fechas de inicio"""
        print(f"\n{'='*80}")
        print(f"📅 INTERACTIVE DATE EDITOR - Change data_start dates")
        print(f"{'='*80}")
        
        # Listar todos los símbolos
        symbols = self.list_all_symbols_with_dates()
        
        if not symbols:
            print("❌ No symbols found in database")
            return False
        
        # Mostrar símbolos con índices
        print(f"\n📋 SYMBOLS IN DATABASE:")
        print(f"{'Index':<6} {'Symbol':<25} {'Type':<12} {'Exchange':<10} {'Current Start Date':<20}")
        print(f"{'-'*6} {'-'*25} {'-'*12} {'-'*10} {'-'*20}")
        
        for i, symbol in enumerate(symbols):
            current_date = symbol['data_start'].strftime('%Y-%m-%d') if symbol['data_start'] else 'N/A'
            print(f"{i:<6} {symbol['symbol_id']:<25} {symbol['symbol_type']:<12} {symbol['exchange_id']:<10} {current_date:<20}")
        
        while True:
            print(f"\nOptions:")
            print(f"  <number> - Select symbol by index to edit date")
            print(f"  all - Show all symbols again")
            print(f"  quit - Exit date editor")
            
            choice = input(f"\nEnter your choice: ").strip().lower()
            
            if choice == 'quit':
                return True
            elif choice == 'all':
                # Mostrar lista de nuevo
                symbols = self.list_all_symbols_with_dates()  # Refresh list
                print(f"\n📋 SYMBOLS IN DATABASE:")
                print(f"{'Index':<6} {'Symbol':<25} {'Type':<12} {'Exchange':<10} {'Current Start Date':<20}")
                print(f"{'-'*6} {'-'*25} {'-'*12} {'-'*10} {'-'*20}")
                
                for i, symbol in enumerate(symbols):
                    current_date = symbol['data_start'].strftime('%Y-%m-%d') if symbol['data_start'] else 'N/A'
                    print(f"{i:<6} {symbol['symbol_id']:<25} {symbol['symbol_type']:<12} {symbol['exchange_id']:<10} {current_date:<20}")
                continue
            elif choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(symbols):
                    success = self._edit_symbol_start_date(symbols[idx])
                    if success:
                        print(f"✅ Date updated successfully!")
                        # Refresh the symbols list
                        symbols = self.list_all_symbols_with_dates()
                    else:
                        print(f"❌ Date update failed")
                else:
                    print(f"Invalid index. Use 0-{len(symbols)-1}")
            else:
                print("Invalid choice. Try again.")
    
    def _edit_symbol_start_date(self, symbol: Dict) -> bool:
        """NUEVO: Edita la fecha de inicio de un símbolo específico"""
        current_date = symbol['data_start'].strftime('%Y-%m-%d') if symbol['data_start'] else 'N/A'
        
        print(f"\n📅 EDITING DATE FOR: {symbol['symbol_id']}")
        print(f"Current start date: {current_date}")
        print(f"Type: {symbol['symbol_type']}, Exchange: {symbol['exchange_id']}")
        
        # Sugerir algunas fechas comunes
        print(f"\nCommon start dates:")
        print(f"  1. 2024-10-29 (SPX typical start)")
        print(f"  2. 2024-11-12 (GIGA typical start)")
        print(f"  3. 2024-01-01 (Start of 2024)")
        print(f"  4. 2023-01-01 (Start of 2023)")
        print(f"  5. Custom date")
        print(f"  6. Cancel")
        
        while True:
            choice = input(f"\nSelect option (1-6) or enter date directly (YYYY-MM-DD): ").strip()
            
            new_date = None
            
            if choice == '1':
                new_date = datetime(2024, 10, 29)
            elif choice == '2':
                new_date = datetime(2024, 11, 12)
            elif choice == '3':
                new_date = datetime(2024, 1, 1)
            elif choice == '4':
                new_date = datetime(2023, 1, 1)
            elif choice == '5':
                date_input = input("Enter custom date (YYYY-MM-DD): ").strip()
                new_date = self._parse_user_date(date_input)
                if not new_date:
                    print("❌ Invalid date format. Use YYYY-MM-DD")
                    continue
            elif choice == '6':
                print("Date edit cancelled")
                return False
            else:
                # Try to parse as direct date input
                new_date = self._parse_user_date(choice)
                if not new_date:
                    print("❌ Invalid choice or date format. Use options 1-6 or YYYY-MM-DD format")
                    continue
            
            if new_date:
                # Confirm the change
                new_date_str = new_date.strftime('%Y-%m-%d')
                print(f"\n📅 CONFIRM DATE CHANGE:")
                print(f"Symbol: {symbol['symbol_id']}")
                print(f"Current date: {current_date}")
                print(f"New date: {new_date_str}")
                
                confirm = input(f"\nConfirm this change? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    return self._update_symbol_start_date(symbol['symbol_id'], new_date)
                else:
                    print("Change cancelled")
                    return False
    
    def _parse_user_date(self, date_str: str) -> Optional[datetime]:
        """NUEVO: Parse fecha introducida por el usuario"""
        if not date_str:
            return None
        
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            try:
                # Try other common formats
                return datetime.strptime(date_str, '%Y/%m/%d')
            except ValueError:
                try:
                    return datetime.strptime(date_str, '%d-%m-%Y')
                except ValueError:
                    return None
    
    def _update_symbol_start_date(self, symbol_id: str, new_date: datetime) -> bool:
        """NUEVO: Actualiza la fecha de inicio en la base de datos - FIXED"""
        try:
            with db_manager.get_session() as session:
                # FIXED: Solo actualizar data_start, sin updated_at
                result = session.execute(text("""
                    UPDATE symbol_info 
                    SET data_start = :new_date
                    WHERE symbol_id = :symbol_id
                """), {
                    'symbol_id': symbol_id,
                    'new_date': new_date
                })
                
                session.commit()
                
                if result.rowcount > 0:
                    log.info(f"✅ Updated start date for {symbol_id} to {new_date.date()}")
                    return True
                else:
                    log.error(f"❌ No rows updated for {symbol_id}")
                    return False
                    
        except Exception as e:
            log.error(f"Error updating start date for {symbol_id}: {e}")
            return False
    
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
            print(f"MAIN MENU:")
            print(f"  1. Search and add new symbol")
            print(f"  2. Edit start dates of existing symbols")
            print(f"  3. List all existing symbols")
            print(f"  4. Quit")
            
            main_choice = input("Select option (1-4): ").strip()
            
            if main_choice == '1':
                # Original functionality - search and add symbols
                asset_input = input("Enter asset to search (or 'back' to return): ").strip().upper()
                
                if asset_input.lower() == 'back':
                    continue
                
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
            
            elif main_choice == '2':
                # NEW functionality - edit start dates
                self.interactive_date_editor()
            
            elif main_choice == '3':
                # List existing symbols
                symbols = self.list_all_symbols_with_dates()
                if symbols:
                    print("📋 EXISTING SYMBOLS IN DATABASE:")
                    print("-" * 80)
                    current_asset = None
                    for symbol in symbols:
                        if symbol['asset_id_base'] != current_asset:
                            if current_asset is not None:
                                print()
                            current_asset = symbol['asset_id_base']
                            print(f"Asset: {current_asset}")
                        
                        data_start = symbol['data_start'].strftime('%Y-%m-%d') if symbol['data_start'] else 'N/A'
                        print(f"  {symbol['symbol_id']} ({symbol['exchange_id']}, {symbol['symbol_type']}) - from {data_start}")
                else:
                    print("No symbols found in database")
            
            elif main_choice == '4':
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid choice. Please select 1-4.")
        
        return True

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="🎯 Interactive Symbol Chooser with Date Editor")
    parser.add_argument("--asset", type=str, help="Search for specific asset directly (e.g., GIGA)")
    parser.add_argument("--exchange", type=str, default="MEXCFTS", help="Default exchange filter (default: MEXCFTS)")
    parser.add_argument("--list-existing", action="store_true", help="List existing symbols in database")
    parser.add_argument("--edit-dates", action="store_true", help="🆕 Go directly to date editor")
    
    args = parser.parse_args()
    
    chooser = InteractiveSymbolChooser(default_exchange=args.exchange)
    
    if args.edit_dates:
        # NEW: Go directly to date editor
        return chooser.interactive_date_editor()
    
    elif args.list_existing:
        # List existing symbols
        symbols = chooser.list_all_symbols_with_dates()
        
        if symbols:
            print("📋 EXISTING SYMBOLS IN DATABASE:")
            print("-" * 80)
            current_asset = None
            for symbol in symbols:
                if symbol['asset_id_base'] != current_asset:
                    if current_asset is not None:
                        print()
                    current_asset = symbol['asset_id_base']
                    print(f"Asset: {current_asset}")
                
                data_start = symbol['data_start'].strftime('%Y-%m-%d') if symbol['data_start'] else 'N/A'
                print(f"  {symbol['symbol_id']} ({symbol['exchange_id']}, {symbol['symbol_type']}) - from {data_start}")
        else:
            print("No symbols found in database")
        
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
        # Run manual selection mode with new menu
        return chooser.run_manual_selection()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)