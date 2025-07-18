#!/usr/bin/env python3
"""
Simplified Benchmark Runner for Multi-Agent Coding System

This module provides a clean interface for running benchmarks against the system.
It can be used independently or integrated into the test suite.

Usage:
    python tests/benchmark_runner.py                    # Run all benchmarks
    python tests/benchmark_runner.py --quick            # Run quick test
    python tests/benchmark_runner.py --benchmark swe    # Run specific benchmark
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.integration.test_benchmarks import BenchmarkTestHarness

def run_benchmarks(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run benchmarks with the given configuration"""
    print("🚀 Multi-Agent Coding System Benchmark Runner")
    print("=" * 50)
    
    # Create harness with configuration
    harness = BenchmarkTestHarness(
        max_problems_per_benchmark=config.get('max_problems', 3)
    )
    
    # Filter benchmarks if specified
    if config.get('benchmark'):
        benchmark_name = config['benchmark']
        if benchmark_name in harness.sample_problems:
            # Keep only the specified benchmark
            other_benchmarks = list(harness.sample_problems.keys())
            for other in other_benchmarks:
                if other != benchmark_name:
                    del harness.sample_problems[other]
            print(f"🎯 Running only {benchmark_name} benchmark")
        else:
            print(f"⚠️  Benchmark '{benchmark_name}' not found. Available: {list(harness.sample_problems.keys())}")
            return {"error": f"Benchmark '{benchmark_name}' not found"}
    
    # Run benchmarks
    results = harness.run_benchmarks()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    
    summary = results['summary']
    print(f"Total Problems: {summary['total_problems']}")
    print(f"Successful: {summary['successful_problems']}")
    print(f"Success Rate: {summary['overall_success_rate']:.2%}")
    print(f"Avg Quality Score: {summary['avg_quality_score']:.2f}")
    print(f"Avg Performance Score: {summary['avg_performance_score']:.2f}")
    print(f"Avg Execution Time: {summary['avg_execution_time']:.1f}s")
    
    # Print benchmark-specific results
    print("\n📈 BENCHMARK BREAKDOWN:")
    for benchmark, metrics in results['benchmark_metrics'].items():
        print(f"\n{benchmark.upper()}:")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Avg Quality: {metrics['avg_quality_score']:.2f}")
        print(f"  Avg Performance: {metrics['avg_performance_score']:.2f}")
        print(f"  Avg Time: {metrics['avg_execution_time']:.1f}s")
    
    # Save results
    if config.get('save_results', True):
        output_file = Path("tests/reports/benchmark_results.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    return results

def main():
    """Main entry point for the benchmark runner"""
    parser = argparse.ArgumentParser(description="Run benchmarks for multi-agent coding system")
    parser.add_argument('--quick', action='store_true', help='Run quick test with minimal problems')
    parser.add_argument('--benchmark', type=str, help='Run specific benchmark (swe_bench, human_eval, mbpp)')
    parser.add_argument('--max-problems', type=int, default=3, help='Max problems per benchmark')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'max_problems': 1 if args.quick else args.max_problems,
        'benchmark': args.benchmark,
        'save_results': not args.no_save
    }
    
    # Run benchmarks
    try:
        results = run_benchmarks(config)
        
        # Exit with appropriate code
        if 'error' in results:
            print(f"\n❌ Error: {results['error']}")
            sys.exit(1)
        
        success_rate = results['summary']['overall_success_rate']
        if success_rate >= 0.5:
            print(f"\n✅ Benchmark run completed successfully (Success rate: {success_rate:.2%})")
            sys.exit(0)
        else:
            print(f"\n⚠️  Benchmark run completed with low success rate ({success_rate:.2%})")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmark run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 