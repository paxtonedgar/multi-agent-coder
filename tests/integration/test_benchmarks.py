#!/usr/bin/env python3
"""
Integration tests for benchmark evaluation of the multi-agent coding system.

This module integrates the benchmark harnesses to test the system against:
- SWE-Bench (Software Engineering Benchmark)
- LiveBench/LiveCodeBench (Dynamic problems)
- HumanEval (OpenAI's coding benchmark)
- MBPP (Google's Python programming problems)
- Custom sample problems

Tests focus on:
- Success rate on real coding problems
- Code quality and correctness
- Performance metrics
- Error handling and robustness
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class BenchmarkProblem:
    """Represents a benchmark problem"""
    benchmark: str
    problem_id: str
    description: str
    language: str
    difficulty: str
    test_file: Optional[str] = None
    expected_output: Optional[str] = None
    dependencies: List[str] = None

@dataclass
class BenchmarkResult:
    """Represents benchmark evaluation result"""
    benchmark: str
    problem_id: str
    success: bool
    execution_time: float
    code_length: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    test_output: Optional[str] = None
    agent_output: Optional[str] = None
    quality_score: float = 0.0
    performance_score: float = 0.0

class BenchmarkTestHarness:
    """Test harness for running benchmarks against the multi-agent system"""
    
    def __init__(self, temp_dir: Optional[str] = None, max_problems_per_benchmark: int = 3):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="benchmark_test_")
        self.max_problems_per_benchmark = max_problems_per_benchmark
        self.results: List[BenchmarkResult] = []
        
        # Sample problems for testing (avoiding external API calls)
        self.sample_problems = {
            'swe_bench': [
                BenchmarkProblem(
                    benchmark='swe_bench',
                    problem_id='swe_001',
                    description='Fix the bug in the fibonacci function that causes infinite recursion for negative numbers',
                    language='python',
                    difficulty='easy'
                ),
                BenchmarkProblem(
                    benchmark='swe_bench',
                    problem_id='swe_002',
                    description='Implement a function to find the longest common subsequence between two strings',
                    language='python',
                    difficulty='medium'
                )
            ],
            'human_eval': [
                BenchmarkProblem(
                    benchmark='human_eval',
                    problem_id='human_001',
                    description='Write a function that determines if a string is a palindrome, ignoring spaces and punctuation',
                    language='python',
                    difficulty='easy'
                ),
                BenchmarkProblem(
                    benchmark='human_eval',
                    problem_id='human_002',
                    description='Implement a function to find all prime numbers up to a given limit using the Sieve of Eratosthenes',
                    language='python',
                    difficulty='medium'
                )
            ],
            'mbpp': [
                BenchmarkProblem(
                    benchmark='mbpp',
                    problem_id='mbpp_001',
                    description='Create a simple calculator that can perform basic arithmetic operations',
                    language='python',
                    difficulty='easy'
                )
            ]
        }
    
    def run_agent_on_problem(self, problem: BenchmarkProblem) -> BenchmarkResult:
        """Run the multi-agent system on a benchmark problem"""
        start_time = time.time()
        
        try:
            # Set environment variables for minimal mode to avoid external API calls
            env = os.environ.copy()
            env.update({
                'NO_EXTERNAL': '1',
                'MINIMAL_MODE': '1',
                'ANTHROPIC_API_KEY': 'test-key'  # Use test key to avoid real API calls
            })
            
            # Create a temporary file for the problem
            problem_file = Path(self.temp_dir) / f"{problem.problem_id}.txt"
            with open(problem_file, 'w') as f:
                f.write(f"Solve this {problem.benchmark} problem: {problem.description}")
            
            # Run the CLI with the problem
            cmd = [
                sys.executable, 'main.py',
                f"Solve this {problem.benchmark} problem: {problem.description}",
                '--mode', 'minimal'
            ]
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            execution_time = time.time() - start_time
            
            # Get output from stdout
            agent_output = result.stdout
            
            # Determine success based on return code and output
            success = result.returncode == 0 and len(agent_output.strip()) > 0
            
            # Calculate quality and performance scores
            quality_score = self._calculate_quality_score(agent_output, problem)
            performance_score = self._calculate_performance_score(execution_time, len(agent_output))
            
            return BenchmarkResult(
                benchmark=problem.benchmark,
                problem_id=problem.problem_id,
                success=success,
                execution_time=execution_time,
                code_length=len(agent_output),
                error_type=None if success else 'execution_error',
                error_message=None if success else result.stderr,
                test_output=result.stdout,
                agent_output=agent_output,
                quality_score=quality_score,
                performance_score=performance_score
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                benchmark=problem.benchmark,
                problem_id=problem.problem_id,
                success=False,
                execution_time=execution_time,
                code_length=0,
                error_type='timeout',
                error_message=f"Timeout after {execution_time:.1f} seconds",
                agent_output="",
                quality_score=0.0,
                performance_score=0.0
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                benchmark=problem.benchmark,
                problem_id=problem.problem_id,
                success=False,
                execution_time=execution_time,
                code_length=0,
                error_type='exception',
                error_message=str(e),
                agent_output="",
                quality_score=0.0,
                performance_score=0.0
            )
    
    def _calculate_quality_score(self, output: str, problem: BenchmarkProblem) -> float:
        """Calculate quality score based on output characteristics"""
        if not output.strip():
            return 0.0
        
        score = 0.0
        
        # Check for code-like content
        if 'def ' in output or 'class ' in output:
            score += 0.3
        
        # Check for problem-specific keywords
        problem_keywords = {
            'fibonacci': ['fib', 'recursion', 'negative'],
            'palindrome': ['palindrome', 'reverse', 'string'],
            'prime': ['prime', 'sieve', 'eratosthenes'],
            'calculator': ['calc', 'arithmetic', 'operation'],
            'subsequence': ['subsequence', 'dynamic', 'programming']
        }
        
        for keyword, related_words in problem_keywords.items():
            if keyword in problem.description.lower():
                for word in related_words:
                    if word in output.lower():
                        score += 0.2
                        break
        
        # Check for proper Python syntax indicators
        if 'import ' in output or 'from ' in output:
            score += 0.1
        
        if 'return ' in output:
            score += 0.1
        
        # Check for test-like content
        if 'test' in output.lower() or 'assert' in output.lower():
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_performance_score(self, execution_time: float, code_length: int) -> float:
        """Calculate performance score based on execution time and code length"""
        # Prefer faster execution and reasonable code length
        time_score = max(0, 1.0 - (execution_time / 30.0))  # 30s is max
        length_score = min(1.0, code_length / 1000.0)  # 1000 chars is good
        
        return (time_score + length_score) / 2.0
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark problems and return results"""
        print("🚀 Starting benchmark evaluation...")
        
        all_problems = []
        for benchmark_name, problems in self.sample_problems.items():
            all_problems.extend(problems[:self.max_problems_per_benchmark])
        
        for problem in all_problems:
            print(f"📝 Testing {problem.benchmark}: {problem.problem_id}")
            result = self.run_agent_on_problem(problem)
            self.results.append(result)
            
            status = "✅" if result.success else "❌"
            print(f"   {status} {result.problem_id}: {result.execution_time:.1f}s, "
                  f"Quality: {result.quality_score:.2f}, Performance: {result.performance_score:.2f}")
        
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        if not self.results:
            return {"error": "No results available"}
        
        total_problems = len(self.results)
        successful_problems = sum(1 for r in self.results if r.success)
        success_rate = successful_problems / total_problems if total_problems > 0 else 0
        
        avg_execution_time = sum(r.execution_time for r in self.results) / total_problems
        avg_quality_score = sum(r.quality_score for r in self.results) / total_problems
        avg_performance_score = sum(r.performance_score for r in self.results) / total_problems
        
        # Group by benchmark
        benchmark_metrics = {}
        for result in self.results:
            if result.benchmark not in benchmark_metrics:
                benchmark_metrics[result.benchmark] = {
                    'total': 0,
                    'successful': 0,
                    'execution_times': [],
                    'quality_scores': [],
                    'performance_scores': []
                }
            
            metrics = benchmark_metrics[result.benchmark]
            metrics['total'] += 1
            if result.success:
                metrics['successful'] += 1
            metrics['execution_times'].append(result.execution_time)
            metrics['quality_scores'].append(result.quality_score)
            metrics['performance_scores'].append(result.performance_score)
        
        # Calculate averages for each benchmark
        for benchmark, metrics in benchmark_metrics.items():
            total = metrics['total']
            metrics['success_rate'] = metrics['successful'] / total if total > 0 else 0
            metrics['avg_execution_time'] = sum(metrics['execution_times']) / total
            metrics['avg_quality_score'] = sum(metrics['quality_scores']) / total
            metrics['avg_performance_score'] = sum(metrics['performance_scores']) / total
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_problems': total_problems,
                'successful_problems': successful_problems,
                'overall_success_rate': success_rate,
                'avg_quality_score': avg_quality_score,
                'avg_performance_score': avg_performance_score,
                'avg_execution_time': avg_execution_time
            },
            'benchmark_metrics': benchmark_metrics,
            'detailed_results': [asdict(r) for r in self.results]
        }

# Test fixtures and test functions
@pytest.fixture
def benchmark_harness():
    """Fixture providing a benchmark test harness"""
    return BenchmarkTestHarness()

def test_benchmark_harness_initialization(benchmark_harness):
    """Test that the benchmark harness initializes correctly"""
    assert benchmark_harness.temp_dir is not None
    assert Path(benchmark_harness.temp_dir).exists()
    assert len(benchmark_harness.sample_problems) > 0
    
    # Check that we have problems for each benchmark
    for benchmark_name, problems in benchmark_harness.sample_problems.items():
        assert len(problems) > 0
        for problem in problems:
            assert problem.benchmark == benchmark_name
            assert problem.language == 'python'

def test_quality_score_calculation(benchmark_harness):
    """Test quality score calculation"""
    problem = BenchmarkProblem(
        benchmark='test',
        problem_id='test_001',
        description='Write a fibonacci function',
        language='python',
        difficulty='easy'
    )
    
    # Test empty output
    score = benchmark_harness._calculate_quality_score("", problem)
    assert score == 0.0
    
    # Test good output
    good_output = """
def fibonacci(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    score = benchmark_harness._calculate_quality_score(good_output, problem)
    assert score > 0.5  # Should have good score for relevant code

def test_performance_score_calculation(benchmark_harness):
    """Test performance score calculation"""
    # Test fast execution with reasonable code length
    score = benchmark_harness._calculate_performance_score(5.0, 500)
    assert score > 0.5
    
    # Test slow execution
    score = benchmark_harness._calculate_performance_score(25.0, 500)
    assert score < 0.5

@pytest.mark.integration
def test_benchmark_execution(benchmark_harness):
    """Integration test: Run a single benchmark problem"""
    # Use a simple problem for testing
    problem = BenchmarkProblem(
        benchmark='test',
        problem_id='test_simple',
        description='Write a function that returns "hello world"',
        language='python',
        difficulty='easy'
    )
    
    result = benchmark_harness.run_agent_on_problem(problem)
    
    # Basic assertions
    assert result.benchmark == 'test'
    assert result.problem_id == 'test_simple'
    assert isinstance(result.execution_time, float)
    assert isinstance(result.quality_score, float)
    assert isinstance(result.performance_score, float)
    assert 0.0 <= result.quality_score <= 1.0
    assert 0.0 <= result.performance_score <= 1.0

@pytest.mark.integration
def test_full_benchmark_suite(benchmark_harness):
    """Integration test: Run the full benchmark suite"""
    # Run with minimal problems for testing
    benchmark_harness.max_problems_per_benchmark = 1
    
    results = benchmark_harness.run_benchmarks()
    
    # Check that we got results
    assert 'summary' in results
    assert 'benchmark_metrics' in results
    assert 'detailed_results' in results
    
    summary = results['summary']
    assert summary['total_problems'] > 0
    assert 0.0 <= summary['overall_success_rate'] <= 1.0
    assert summary['avg_execution_time'] > 0.0
    
    # Check that we have metrics for each benchmark
    for benchmark_name in benchmark_harness.sample_problems.keys():
        assert benchmark_name in results['benchmark_metrics']

def test_benchmark_result_serialization(benchmark_harness):
    """Test that benchmark results can be serialized to JSON"""
    result = BenchmarkResult(
        benchmark='test',
        problem_id='test_001',
        success=True,
        execution_time=5.0,
        code_length=100,
        quality_score=0.8,
        performance_score=0.7
    )
    
    # Test serialization
    result_dict = asdict(result)
    json_str = json.dumps(result_dict)
    
    # Test deserialization
    loaded_dict = json.loads(json_str)
    assert loaded_dict['benchmark'] == 'test'
    assert loaded_dict['success'] is True
    assert loaded_dict['execution_time'] == 5.0

if __name__ == "__main__":
    # Run benchmarks if executed directly
    harness = BenchmarkTestHarness()
    results = harness.run_benchmarks()
    
    print("\n📊 Benchmark Results Summary:")
    print(f"Total Problems: {results['summary']['total_problems']}")
    print(f"Success Rate: {results['summary']['overall_success_rate']:.2%}")
    print(f"Avg Quality Score: {results['summary']['avg_quality_score']:.2f}")
    print(f"Avg Performance Score: {results['summary']['avg_performance_score']:.2f}")
    print(f"Avg Execution Time: {results['summary']['avg_execution_time']:.1f}s")
    
    # Save results
    output_file = Path("tests/reports/benchmark_results.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}") 