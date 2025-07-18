#!/usr/bin/env python3
"""
Comprehensive test runner for multi-agent coder system
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_pytest_with_coverage(test_paths: List[str], coverage_target: float = 80.0) -> Dict[str, Any]:
    """Run pytest with coverage and return results"""
    print(f"🧪 Running tests with coverage target: {coverage_target}%")
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=.",
        "--cov-report=html:tests/reports/coverage_html",
        "--cov-report=json:tests/reports/coverage.json",
        "--cov-report=term-missing",
        "--cov-fail-under", str(coverage_target),
        "--junitxml=tests/reports/junit.xml",
        "--html=tests/reports/report.html",
        "--self-contained-html",
        "-v",
        "--tb=short"
    ]
    
    # Add test paths
    cmd.extend(test_paths)
    
    # Run tests
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    # Parse results
    test_results = {
        'exit_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'duration': end_time - start_time,
        'success': result.returncode == 0
    }
    
    return test_results

def run_performance_tests() -> Dict[str, Any]:
    """Run performance benchmarks"""
    print("⚡ Running performance tests...")
    
    performance_results = {}
    
    # Test CLI startup time
    start_time = time.time()
    result = subprocess.run([sys.executable, "main.py", "--help"], 
                          capture_output=True, text=True)
    end_time = time.time()
    
    performance_results['cli_startup_time'] = end_time - start_time
    performance_results['cli_startup_success'] = result.returncode == 0
    
    # Test memory usage
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        performance_results['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
    except ImportError:
        performance_results['memory_usage_mb'] = 0
        performance_results['psutil_missing'] = True
    
    return performance_results

def run_quality_analysis() -> Dict[str, Any]:
    """Run code quality analysis"""
    print("🔍 Running quality analysis...")
    
    quality_results = {}
    
    # Run ruff for linting
    try:
        result = subprocess.run([sys.executable, "-m", "ruff", "check", "."], 
                              capture_output=True, text=True)
        quality_results['ruff_issues'] = len(result.stdout.splitlines()) if result.stdout else 0
        quality_results['ruff_success'] = result.returncode == 0
    except Exception as e:
        quality_results['ruff_error'] = str(e)
    
    # Run mypy for type checking
    try:
        result = subprocess.run([sys.executable, "-m", "mypy", "."], 
                              capture_output=True, text=True)
        quality_results['mypy_issues'] = len(result.stdout.splitlines()) if result.stdout else 0
        quality_results['mypy_success'] = result.returncode == 0
    except Exception as e:
        quality_results['mypy_error'] = str(e)
    
    return quality_results

def run_benchmark_tests() -> Dict[str, Any]:
    """Run benchmark tests"""
    print("🏆 Running benchmark tests...")
    
    try:
        # Import and run benchmark tests
        from tests.integration.test_benchmarks import BenchmarkTestHarness
        
        harness = BenchmarkTestHarness(max_problems_per_benchmark=1)  # Quick test
        results = harness.run_benchmarks()
        
        return {
            'success': True,
            'results': results,
            'summary': results.get('summary', {}),
            'benchmark_metrics': results.get('benchmark_metrics', {})
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def generate_test_report(test_results: Dict[str, Any], 
                        performance_results: Dict[str, Any],
                        quality_results: Dict[str, Any],
                        benchmark_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    
    # Parse test output for statistics
    stdout = test_results.get('stdout', '')
    lines = stdout.split('\n')
    
    # Extract test statistics
    test_stats = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': 0,
        'duration': test_results.get('duration', 0)
    }
    
    for line in lines:
        if 'collected' in line and 'items' in line:
            # Extract total tests
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'collected':
                    test_stats['total_tests'] = int(parts[i-1])
                    break
        elif 'passed' in line and 'failed' in line:
            # Extract passed/failed counts
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'passed':
                    test_stats['passed'] = int(parts[i-1])
                elif part == 'failed':
                    test_stats['failed'] = int(parts[i-1])
                elif part == 'skipped':
                    test_stats['skipped'] = int(parts[i-1])
                elif part == 'error':
                    test_stats['errors'] = int(parts[i-1])
    
    # Calculate success rate
    if test_stats['total_tests'] > 0:
        test_stats['success_rate'] = (test_stats['passed'] / test_stats['total_tests']) * 100
    else:
        test_stats['success_rate'] = 0
    
    # Create comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_results': test_results,
        'test_statistics': test_stats,
        'performance_results': performance_results,
        'quality_results': quality_results,
        'benchmark_results': benchmark_results,
        'summary': {
            'overall_success': test_results.get('success', False),
            'test_success_rate': test_stats['success_rate'],
            'performance_acceptable': performance_results.get('cli_startup_time', 0) < 2.0,
            'quality_acceptable': quality_results.get('ruff_success', False) and quality_results.get('mypy_success', False),
            'benchmark_success': benchmark_results.get('success', False) if benchmark_results else False
        }
    }
    
    return report

def save_report(report: Dict[str, Any], output_file: str):
    """Save report to file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Report saved to: {output_file}")

def print_summary(report: Dict[str, Any]):
    """Print test summary"""
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    stats = report['test_statistics']
    summary = report['summary']
    
    print(f"✅ Overall Success: {'YES' if summary['overall_success'] else 'NO'}")
    print(f"📊 Test Success Rate: {stats['success_rate']:.1f}%")
    print(f"🧪 Total Tests: {stats['total_tests']}")
    print(f"✅ Passed: {stats['passed']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⏭️  Skipped: {stats['skipped']}")
    print(f"💥 Errors: {stats['errors']}")
    print(f"⏱️  Duration: {stats['duration']:.2f}s")
    
    perf = report['performance_results']
    print(f"\n⚡ Performance:")
    print(f"   CLI Startup: {perf.get('cli_startup_time', 0):.3f}s")
    print(f"   Memory Usage: {perf.get('memory_usage_mb', 0):.1f}MB")
    
    # Print benchmark results if available
    if report.get('benchmark_results'):
        bench = report['benchmark_results']
        if bench.get('success'):
            summary = bench.get('summary', {})
            print(f"\n🏆 Benchmark Results:")
            print(f"   Success Rate: {summary.get('overall_success_rate', 0):.2%}")
            print(f"   Avg Quality: {summary.get('avg_quality_score', 0):.2f}")
            print(f"   Avg Performance: {summary.get('avg_performance_score', 0):.2f}")
        else:
            print(f"\n🏆 Benchmark Results: Failed - {bench.get('error', 'Unknown error')}")
    
    qual = report['quality_results']
    print(f"\n🔍 Quality:")
    print(f"   Ruff Issues: {qual.get('ruff_issues', 0)}")
    print(f"   MyPy Issues: {qual.get('mypy_issues', 0)}")
    
    print("\n" + "="*60)

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run comprehensive tests for multi-agent coder")
    parser.add_argument("--test-paths", nargs="+", 
                       default=["tests/"],
                       help="Test paths to run")
    parser.add_argument("--coverage-target", type=float, default=80.0,
                       help="Coverage target percentage")
    parser.add_argument("--output", default="tests/reports/test_report.json",
                       help="Output report file")
    parser.add_argument("--skip-performance", action="store_true",
                       help="Skip performance tests")
    parser.add_argument("--skip-quality", action="store_true",
                       help="Skip quality analysis")
    
    args = parser.parse_args()
    
    print("🚀 Starting comprehensive test suite for multi-agent coder")
    print(f"📁 Test paths: {args.test_paths}")
    print(f"🎯 Coverage target: {args.coverage_target}%")
    
    # Ensure reports directory exists
    os.makedirs("tests/reports", exist_ok=True)
    
    # Run tests
    test_results = run_pytest_with_coverage(args.test_paths, args.coverage_target)
    
    # Run performance tests
    performance_results = {}
    if not args.skip_performance:
        performance_results = run_performance_tests()
    
    # Run quality analysis
    quality_results = {}
    if not args.skip_quality:
        quality_results = run_quality_analysis()
    
    # Run benchmark tests
    benchmark_results = run_benchmark_tests()
    
    # Generate report
    report = generate_test_report(test_results, performance_results, quality_results, benchmark_results)
    
    # Save report
    save_report(report, args.output)
    
    # Print summary
    print_summary(report)
    
    # Exit with appropriate code
    if report['summary']['overall_success']:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the report for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 