#!/usr/bin/env python3
"""
🚀 Complete Workflow Script
Ejecuta todo el pipeline desde setup hasta análisis final
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import get_setup_logger

log = get_setup_logger()

def run_script(script_path: str, args: list = None) -> bool:
    """Ejecutar un script y retornar si fue exitoso"""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    log.info(f"🏃 Ejecutando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        log.info(f"✅ Completado exitosamente: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"❌ Error ejecutando {script_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="🚀 Complete trading system workflow")
    parser.add_argument("--symbol", type=str, help="🎯 Process specific symbol only")
    parser.add_argument("--skip-setup", action="store_true", help="⏭️ Skip database setup")
    parser.add_argument("--funding-only", action="store_true", help="💰 Only funding rates")
    
    args = parser.parse_args()
    
    log.info("🚀 Starting COMPLETE workflow...")
    
    scripts_dir = Path(__file__).parent
    symbol_args = ["--symbol", args.symbol] if args.symbol else []
    
    success_count = 0
    total_steps = 7 if not args.funding_only else 4
    
    try:
        # 1. Setup Database
        if not args.skip_setup:
            log.info("🔧 Step 1: Database setup...")
            if run_script(scripts_dir / "setup_database.py"):
                success_count += 1
        else:
            log.info("⏭️ Skipping database setup")
            success_count += 1
        
        # 2. Ingest Data
        log.info("📥 Step 2: Data ingestion...")
        ingest_args = symbol_args.copy()
        if args.funding_only:
            ingest_args.append("--funding-only")
        
        if run_script(scripts_dir / "ingest_data.py", ingest_args):
            success_count += 1
        
        if args.funding_only:
            log.info("💰 Funding-only workflow completed!")
            log.info(f"✅ {success_count}/{total_steps} steps completed")
            return success_count == total_steps
        
        # 3. Validate Data
        log.info("✅ Step 3: Data validation...")
        if run_script(scripts_dir / "validate_data.py", symbol_args):
            success_count += 1
        
        # 4. Clean Data
        log.info("🧹 Step 4: Data cleaning...")
        if run_script(scripts_dir / "clean_data.py", symbol_args):
            success_count += 1
        
        # 5. Calculate Mark Prices
        log.info("💎 Step 5: Mark prices calculation...")
        if run_script(scripts_dir / "calculate_markprices.py", symbol_args):
            success_count += 1
        
        # 6. Analyze Liquidity
        log.info("🔬 Step 6: Liquidity analysis...")
        if run_script(scripts_dir / "analyzers" / "analyze_data.py", symbol_args):
            success_count += 1
        
        # 7. Analyze Mark Prices
        log.info("📈 Step 7: Mark prices analysis...")
        if run_script(scripts_dir / "analyzers" / "analyze_markprices.py", symbol_args):
            success_count += 1
        
        log.info(f"\n🎉 COMPLETE WORKFLOW FINISHED!")
        log.info(f"✅ {success_count}/{total_steps} steps completed successfully")
        
        if success_count == total_steps:
            log.info("🏆 All steps completed successfully!")
            log.info("📊 Check plots/ directory for analysis results")
            return True
        else:
            log.warning(f"⚠️ {total_steps - success_count} steps failed")
            return False
        
    except Exception as e:
        log.error(f"💥 Workflow failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)