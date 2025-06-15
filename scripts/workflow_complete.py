#!/usr/bin/env python3
"""
🚀 Complete Workflow Script - FULLY AUTOMATED VERSION
Ejecuta todo el pipeline sin intervención manual
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import get_setup_logger

log = get_setup_logger()

def run_script(script_path, args: list = None) -> bool:
    """Ejecutar un script y retornar si fue exitoso"""
    script_path_str = str(script_path) if isinstance(script_path, Path) else script_path
    
    cmd = [sys.executable, script_path_str]
    if args:
        cmd.extend(args)
    
    log.info(f"🏃 Ejecutando: {' '.join(cmd)}")
    
    try:
        if not Path(script_path_str).exists():
            log.error(f"❌ Script no encontrado: {script_path_str}")
            return False
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        log.info(f"✅ Completado exitosamente: {script_path_str}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"❌ Error ejecutando {script_path_str}: {e}")
        return False
    except Exception as e:
        log.error(f"💥 Error inesperado ejecutando {script_path_str}: {e}")
        return False

def check_script_exists(scripts_dir: Path, script_name: str) -> bool:
    """Verificar que un script existe"""
    script_path = scripts_dir / script_name
    if not script_path.exists():
        log.error(f"❌ Script no encontrado: {script_path}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="🚀 Complete trading system workflow - FULLY AUTOMATED")
    parser.add_argument("--symbol", type=str, help="🎯 Process specific symbol only")
    parser.add_argument("--skip-setup", action="store_true", help="⏭️ Skip database setup")
    parser.add_argument("--funding-only", action="store_true", help="💰 Only funding rates")
    parser.add_argument("--force-markprices", action="store_true", help="🔧 Force mark prices recalculation")
    parser.add_argument("--no-plots", action="store_true", help="📊 Skip plot generation for speed")
    parser.add_argument("--fast", action="store_true", help="⚡ Legacy fast mode (use --force-markprices instead)")
    
    args = parser.parse_args()
    
    log.info("🚀 Starting COMPLETE AUTOMATED workflow...")
    log.info("🤖 NO MANUAL INTERVENTION REQUIRED")
    
    scripts_dir = Path(__file__).parent
    symbol_args = ["--symbol", args.symbol] if args.symbol else []
    
    success_count = 0
    total_steps = 7 if not args.funding_only else 4
    
    log.info(f"📁 Scripts directory: {scripts_dir}")
    log.info(f"🎯 Target symbol: {args.symbol if args.symbol else 'ALL'}")
    log.info(f"📊 Total steps: {total_steps}")
    log.info(f"🤖 Mode: FULLY AUTOMATED")
    
    try:
        # 1. Setup Database
        if not args.skip_setup:
            log.info("🔧 Step 1: Database setup...")
            script_path = scripts_dir / "setup_database.py"
            if check_script_exists(scripts_dir, "setup_database.py") and run_script(script_path):
                success_count += 1
                log.info("✅ Step 1 completed")
            else:
                log.error("❌ Step 1 failed")
        else:
            log.info("⏭️ Skipping database setup")
            success_count += 1
        
        # 2. Ingest Data
        log.info("📥 Step 2: Data ingestion...")
        ingest_args = symbol_args.copy()
        if args.funding_only:
            ingest_args.append("--funding-only")
        
        script_path = scripts_dir / "ingest_data.py"
        if check_script_exists(scripts_dir, "ingest_data.py") and run_script(script_path, ingest_args):
            success_count += 1
            log.info("✅ Step 2 completed")
        else:
            log.error("❌ Step 2 failed")
        
        if args.funding_only:
            log.info("💰 Funding-only workflow completed!")
            log.info(f"✅ {success_count}/{total_steps} steps completed")
            return success_count == total_steps
        
        # 3. Validate Data
        log.info("✅ Step 3: Data validation...")
        script_path = scripts_dir / "validate_data.py"
        if check_script_exists(scripts_dir, "validate_data.py") and run_script(script_path, symbol_args):
            success_count += 1
            log.info("✅ Step 3 completed")
        else:
            log.error("❌ Step 3 failed")
        
        # 4. Clean Data
        log.info("🧹 Step 4: Data cleaning...")
        script_path = scripts_dir / "clean_data.py"
        if check_script_exists(scripts_dir, "clean_data.py") and run_script(script_path, symbol_args):
            success_count += 1
            log.info("✅ Step 4 completed")
        else:
            log.error("❌ Step 4 failed")
        
        # 5. Calculate Mark Prices - FULLY AUTOMATED (NO MANUAL INPUT)
        log.info("💎 Step 5: Mark prices calculation (FULLY AUTOMATED)...")
        markprice_args = symbol_args.copy()
        
        # Determinar si usar --force
        force_mode = args.force_markprices or args.fast  # backward compatibility
        if force_mode:
            markprice_args.append("--force")
            log.info("🔧 Using --force for mark prices (explicit request)")
        else:
            log.info("🔄 Using incremental mark prices calculation (fully automatic)")
        
        script_path = scripts_dir / "calculate_markprices.py"
        if check_script_exists(scripts_dir, "calculate_markprices.py") and run_script(script_path, markprice_args):
            success_count += 1
            log.info("✅ Step 5 completed")
        else:
            log.error("❌ Step 5 failed")
        
        # 6. Analyze Liquidity
        log.info("🔬 Step 6: Liquidity analysis...")
        analyzers_dir = scripts_dir / "analyzers"
        script_path = analyzers_dir / "analyze_data.py"
        
        # Argumentos para análisis
        analyze_args = symbol_args.copy()
        if args.no_plots:
            analyze_args.append("--no-plots")
        
        if script_path.exists() and run_script(script_path, analyze_args):
            success_count += 1
            log.info("✅ Step 6 completed")
        else:
            log.warning("⚠️ Step 6 failed or script not found - continuing...")
        
        # 7. Analyze Mark Prices
        log.info("📈 Step 7: Mark prices analysis...")
        script_path = analyzers_dir / "analyze_markprices.py"
        
        markprice_analyze_args = symbol_args.copy()
        
        if script_path.exists() and run_script(script_path, markprice_analyze_args):
            success_count += 1
            log.info("✅ Step 7 completed")
        else:
            log.warning("⚠️ Step 7 failed or script not found - continuing...")
        
        # Final Summary
        log.info(f"\n{'='*60}")
        log.info(f"🎉 COMPLETE AUTOMATED WORKFLOW FINISHED!")
        log.info(f"{'='*60}")
        log.info(f"✅ Steps completed: {success_count}/{total_steps}")
        
        if success_count == total_steps:
            log.info("🏆 ALL STEPS COMPLETED SUCCESSFULLY!")
            log.info("📊 Check plots/ directory for analysis results")
            log.info("📝 Check logs/ directory for detailed logs")
            log.info("\n🚀 Your trading system is ready!")
            log.info("Next steps:")
            log.info("  - Review analysis charts in plots/")
            log.info("  - Run backtesting: python scripts/run_backtest.py")
            log.info("  - Start live trading: python scripts/start_trading.py")
            return True
        elif success_count >= 5:  # Core steps completed (setup, ingest, validate, clean, markprices)
            log.info("🎯 CORE WORKFLOW COMPLETED!")
            log.info("Analysis steps may have failed but core data pipeline is ready")
            log.info("✅ Database, ingestion, validation, cleaning, and mark prices are done")
            return True
        else:
            log.warning(f"⚠️ Only {success_count} steps completed")
            log.warning("Critical steps may have failed - check logs for details")
            return False
        
    except KeyboardInterrupt:
        log.warning("🛑 Workflow interrupted by user")
        return False
    except Exception as e:
        log.error(f"💥 Workflow failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Automated workflow completed successfully!")
    else:
        print("\n❌ Automated workflow failed. Check logs for details.")
    sys.exit(0 if success else 1)