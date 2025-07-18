#!/usr/bin/env python3
"""
Quick benchmark test using sample problems
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_harness import BenchmarkHarness, BenchmarkProblem
from sample_benchmark_problems import create_sample_benchmark_data, create_mock_benchmark_repos

def run_quick_benchmark_test():
    """Run a quick benchmark test with sample problems"""
    print("🚀 Quick Benchmark Test")
    print("=" * 50)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temp directory: {temp_dir}")
        
        # Create mock benchmark repositories
        print("🔧 Setting up mock benchmark repositories...")
        create_mock_benchmark_repos(temp_dir)
        
        # Initialize harness
        harness = BenchmarkHarness(temp_dir=temp_dir, max_problems_per_benchmark=2)
        
        # Extract problems
        print("📋 Extracting problems...")
        problems = harness.extract_problems()
        
        if not problems:
            print("❌ No problems extracted")
            return 1
        
        total_problems = sum(len(probs) for probs in problems.values())
        print(f"📊 Extracted {total_problems} problems total")
        
        # Mock agent execution
        with patch('subprocess.run') as mock_run:
            # Mock successful agent execution
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """
def fibonacci(n):
    if n < 0:
        raise ValueError("Input must be non-negative")
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55
"""
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            # Evaluate benchmarks
            print("🚀 Starting evaluation...")
            harness.evaluate_benchmarks(problems)
            
            # Calculate metrics
            print("📈 Calculating metrics...")
            harness.calculate_metrics()
            
            # Generate report
            report_file = Path(temp_dir) / "quick_benchmark_results.json"
            harness.generate_report(str(report_file))
            
            print(f"\n✅ Quick benchmark test completed!")
            print(f"📊 Results saved to: {report_file}")
            
            # Print summary
            if harness.metrics:
                print("\n📊 Summary:")
                for metrics in harness.metrics:
                    print(f"  {metrics.benchmark}: {metrics.success_rate:.1%} success rate "
                          f"({metrics.successful_problems}/{metrics.total_problems} problems)")
            
            return 0

if __name__ == "__main__":
    sys.exit(run_quick_benchmark_test()) 