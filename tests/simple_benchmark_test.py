#!/usr/bin/env python3
"""
Simple Benchmark Test - Quick test with proper imports
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

@dataclass
class SimpleBenchmarkResult:
    """Simple benchmark result"""
    problem_id: str
    description: str
    success: bool
    execution_time: float
    output_length: int
    return_code: int
    error_message: str = None
    output_preview: str = None

def test_simple_problems():
    """Test with simple problems"""
    print("🧪 Simple Benchmark Test")
    print("=" * 50)
    
    # Simple test problems
    problems = [
        {
            'id': 'simple_001',
            'description': 'Write a function that returns "hello world"'
        },
        {
            'id': 'simple_002', 
            'description': 'Create a function that adds two numbers'
        },
        {
            'id': 'simple_003',
            'description': 'Write a function that checks if a number is even'
        }
    ]
    
    results = []
    
    for problem in problems:
        print(f"\n🔍 Testing: {problem['id']}")
        print(f"📝 Description: {problem['description']}")
        
        start_time = time.time()
        
        try:
            # Run with a short timeout
            result = subprocess.run([
                sys.executable, 'main.py', problem['description']
            ], capture_output=True, text=True, timeout=15)  # 15 second timeout
            
            execution_time = time.time() - start_time
            
            # Analyze result
            success = result.returncode == 0
            output_length = len(result.stdout)
            output_preview = result.stdout[:200] if result.stdout else ""
            error_message = result.stderr[:200] if result.stderr else None
            
            print(f"⏱️  Time: {execution_time:.2f}s")
            print(f"📊 Return code: {result.returncode}")
            print(f"📝 Output length: {output_length} chars")
            print(f"✅ Success: {success}")
            
            if output_preview:
                print(f"📄 Output preview: {output_preview}")
            if error_message:
                print(f"❌ Error: {error_message}")
            
            results.append(SimpleBenchmarkResult(
                problem_id=problem['id'],
                description=problem['description'],
                success=success,
                execution_time=execution_time,
                output_length=output_length,
                return_code=result.returncode,
                error_message=error_message,
                output_preview=output_preview
            ))
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"⏰ TIMEOUT after {execution_time:.2f}s")
            
            results.append(SimpleBenchmarkResult(
                problem_id=problem['id'],
                description=problem['description'],
                success=False,
                execution_time=execution_time,
                output_length=0,
                return_code=-1,
                error_message="Timeout after 15 seconds"
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"💥 EXCEPTION: {e}")
            
            results.append(SimpleBenchmarkResult(
                problem_id=problem['id'],
                description=problem['description'],
                success=False,
                execution_time=execution_time,
                output_length=0,
                return_code=-1,
                error_message=str(e)
            ))
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 SIMPLE BENCHMARK RESULTS")
    print(f"{'='*60}")
    
    total_problems = len(results)
    successful_problems = sum(1 for r in results if r.success)
    success_rate = successful_problems / total_problems if total_problems > 0 else 0
    avg_time = sum(r.execution_time for r in results) / total_problems if total_problems > 0 else 0
    avg_length = sum(r.output_length for r in results) / total_problems if total_problems > 0 else 0
    
    print(f"🎯 Success Rate: {success_rate:.1%} ({successful_problems}/{total_problems})")
    print(f"⏱️  Average Time: {avg_time:.2f}s")
    print(f"📝 Average Output Length: {avg_length:.0f} chars")
    
    print(f"\n📈 Per-Problem Results:")
    print("-" * 60)
    
    for result in results:
        print(f"{result.problem_id:12} | "
              f"Success: {'✅' if result.success else '❌'} | "
              f"Time: {result.execution_time:6.2f}s | "
              f"Length: {result.output_length:6d} chars | "
              f"Code: {result.return_code:3d}")
        if result.error_message:
            print(f"             | Error: {result.error_message}")
    
    # Save results
    results_file = "simple_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_problems': total_problems,
                'successful_problems': successful_problems,
                'success_rate': success_rate,
                'avg_execution_time': avg_time,
                'avg_output_length': avg_length
            },
            'results': [asdict(r) for r in results]
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")
    
    return success_rate > 0

def test_imports():
    """Test if we can import the required modules"""
    print("\n🔧 Testing Imports")
    print("=" * 30)
    
    modules = ['memory', 'agents', 'tools', 'graph']
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module} imports successfully")
        except Exception as e:
            print(f"❌ {module} import failed: {e}")
    
    # Test if main.py can be imported
    try:
        import main
        print("✅ main.py imports successfully")
    except Exception as e:
        print(f"❌ main.py import failed: {e}")

def main():
    """Main function"""
    print("🚀 Simple Benchmark Test Suite")
    print("=" * 60)
    
    # Test imports first
    test_imports()
    
    # Test simple problems
    success = test_simple_problems()
    
    if success:
        print("\n🎉 Simple benchmark test completed successfully!")
        return 0
    else:
        print("\n❌ Simple benchmark test found issues!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 