#!/usr/bin/env python3
"""
📊 Analysis Workflow Script
Ejecuta solo los scripts de análisis
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
        log.info(f"✅ Completado: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"❌ Error: {script_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="📊 Analysis workflow")
    parser.add_argument("--symbol", type=str, help="🎯 Analyze specific symbol only")
    parser.add_argument("--no-plots", action="store_true", help="🚫 Skip plot generation")
    
    args = parser.parse_args()
    
    log.info("📊 Starting ANALYSIS workflow...")
    
    scripts_dir = Path(__file__).parent
    symbol_args = ["--symbol", args.symbol] if args.symbol else []
    
    if args.no_plots:
        symbol_args.append("--no-plots")
    
    success_count = 0
    total_steps = 3
    
    try:
        # 1. Test Mark Prices
        log.info("🧪 Step 1: Testing mark prices...")
        if run_script(scripts_dir / "test_markprice.py"):
            success_count += 1
        
        # 2. Analyze Liquidity
        log.info("🔬 Step 2: Liquidity analysis...")
        if run_script(scripts_dir / "analyzers" / "analyze_data.py", symbol_args):
            success_count += 1
        
        # 3. Analyze Mark Prices
        log.info("📈 Step 3: Mark prices analysis...")
        if run_script(scripts_dir / "analyzers" / "analyze_markprices.py", symbol_args):
            success_count += 1
        
        log.info(f"\n🎉 ANALYSIS WORKFLOW FINISHED!")
        log.info(f"✅ {success_count}/{total_steps} analysis steps completed")
        
        if success_count == total_steps:
            log.info("🏆 All analysis completed successfully!")
            log.info("📊 Check plots/ directory for results")
            return True
        else:
            log.warning(f"⚠️ {total_steps - success_count} analysis steps failed")
            return False
        
    except Exception as e:
        log.error(f"💥 Analysis workflow failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)