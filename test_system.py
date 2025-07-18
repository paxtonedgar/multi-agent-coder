#!/usr/bin/env python3
"""
Legacy test script for LangGraph Multi-Agent Coding System
This file is kept for backward compatibility. Use tests/run_tests.py for comprehensive testing.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run comprehensive test suite"""
    print("🧪 Multi-Agent Coding System Test Suite")
    print("=" * 50)
    print("📝 Note: This is the legacy test script.")
    print("🚀 For comprehensive testing, use: python tests/run_tests.py")
    print("=" * 50)
    
    # Check if comprehensive test suite exists
    test_runner = Path("tests/run_tests.py")
    if test_runner.exists():
        print("✅ Comprehensive test suite found!")
        print("🔄 Running comprehensive tests...")
        
        # Run the comprehensive test suite
        result = subprocess.run([sys.executable, "tests/run_tests.py"], 
                              capture_output=False, text=True)
        
        return result.returncode
    else:
        print("❌ Comprehensive test suite not found.")
        print("📁 Expected location: tests/run_tests.py")
        print("🔧 Please run: python tests/run_tests.py")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 