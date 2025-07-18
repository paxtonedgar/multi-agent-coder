#!/usr/bin/env python3
"""
Test the benchmark harness with sample problems
"""

import pytest
import tempfile
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from tests.benchmark_harness import BenchmarkHarness, BenchmarkProblem, BenchmarkResult, BenchmarkMetrics

class TestBenchmarkHarness:
    """Test the benchmark harness functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_harness_initialization(self, temp_dir):
        """Test harness initialization"""
        harness = BenchmarkHarness(temp_dir=temp_dir, max_problems_per_benchmark=3)
        
        assert harness.temp_dir == temp_dir
        assert harness.max_problems_per_benchmark == 3
        assert len(harness.benchmarks) == 5  # All 5 benchmarks
        assert 'swe_bench' in harness.benchmarks
        assert 'livebench' in harness.benchmarks
    
    def test_extract_description_from_comments(self, temp_dir):
        """Test extracting problem descriptions from Python comments"""
        harness = BenchmarkHarness(temp_dir=temp_dir)
        
        # Test with comments
        content = """
# This is a test problem
# Write a function to sort a list
def some_function():
    pass
"""
        description = harness._extract_description_from_comments(content)
        assert "test problem" in description
        assert "sort a list" in description
        
        # Test with docstring
        content = '''
"""
This is a docstring problem
"""
def some_function():
    pass
'''
        description = harness._extract_description_from_comments(content)
        assert description == ""  # Should stop at docstring
        
        # Test with no comments
        content = """
def some_function():
    pass
"""
        description = harness._extract_description_from_comments(content)
        assert description == ""
    
    @patch('subprocess.run')
    def test_run_agent_on_problem_success(self, mock_run, temp_dir):
        """Test running agent on problem with success"""
        harness = BenchmarkHarness(temp_dir=temp_dir)
        
        # Mock successful agent execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        problem = BenchmarkProblem(
            benchmark='swe_bench',
            problem_id='test_1',
            description='Write a fibonacci function',
            language='python',
            difficulty='easy'
        )
        
        result = harness.run_agent_on_problem(problem)
        
        assert result.benchmark == 'swe_bench'
        assert result.problem_id == 'test_1'
        assert result.success is True
        assert result.execution_time > 0
        assert result.code_length > 0
        assert result.error_type is None
    
    @patch('subprocess.run')
    def test_run_agent_on_problem_timeout(self, mock_run, temp_dir):
        """Test running agent on problem with timeout"""
        harness = BenchmarkHarness(temp_dir=temp_dir)
        
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired(['python', 'main.py'], 300)
        
        problem = BenchmarkProblem(
            benchmark='swe_bench',
            problem_id='test_1',
            description='Write a fibonacci function',
            language='python',
            difficulty='easy'
        )
        
        result = harness.run_agent_on_problem(problem)
        
        assert result.success is False
        assert result.error_type == 'timeout'
        assert 'timed out' in result.error_message
    
    def test_calculate_metrics(self, temp_dir):
        """Test metrics calculation"""
        harness = BenchmarkHarness(temp_dir=temp_dir)
        
        # Add sample results
        harness.results = [
            BenchmarkResult(
                benchmark='swe_bench',
                problem_id='test_1',
                success=True,
                execution_time=10.0,
                code_length=100,
                error_type=None
            ),
            BenchmarkResult(
                benchmark='swe_bench',
                problem_id='test_2',
                success=False,
                execution_time=5.0,
                code_length=50,
                error_type='timeout',
                error_message='Timed out'
            ),
            BenchmarkResult(
                benchmark='livebench',
                problem_id='live_1',
                success=True,
                execution_time=15.0,
                code_length=200,
                error_type=None
            )
        ]
        
        harness.calculate_metrics()
        
        assert len(harness.metrics) == 2  # Two benchmarks with results
        
        # Check SWE-Bench metrics
        swe_metrics = next(m for m in harness.metrics if m.benchmark == 'swe_bench')
        assert swe_metrics.total_problems == 2
        assert swe_metrics.successful_problems == 1
        assert swe_metrics.success_rate == 0.5
        assert swe_metrics.avg_execution_time == 7.5
        assert swe_metrics.avg_code_length == 75
        assert swe_metrics.error_distribution['timeout'] == 1
        
        # Check LiveBench metrics
        live_metrics = next(m for m in harness.metrics if m.benchmark == 'livebench')
        assert live_metrics.total_problems == 1
        assert live_metrics.successful_problems == 1
        assert live_metrics.success_rate == 1.0
    
    def test_generate_report(self, temp_dir):
        """Test report generation"""
        harness = BenchmarkHarness(temp_dir=temp_dir)
        
        # Add sample results and metrics
        harness.results = [
            BenchmarkResult(
                benchmark='swe_bench',
                problem_id='test_1',
                success=True,
                execution_time=10.0,
                code_length=100,
                error_type=None
            )
        ]
        
        harness.metrics = [
            BenchmarkMetrics(
                benchmark='swe_bench',
                total_problems=1,
                successful_problems=1,
                success_rate=1.0,
                avg_execution_time=10.0,
                avg_code_length=100,
                error_distribution={}
            )
        ]
        
        report_file = Path(temp_dir) / "test_report.json"
        harness.generate_report(str(report_file))
        
        assert report_file.exists()
        
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        assert 'timestamp' in report
        assert 'summary' in report
        assert 'benchmark_metrics' in report
        assert 'detailed_results' in report
        
        assert report['summary']['total_problems'] == 1
        assert report['summary']['total_successful'] == 1
        assert report['summary']['overall_success_rate'] == 1.0

@pytest.mark.integration
class TestBenchmarkHarnessIntegration:
    """Integration tests for benchmark harness"""
    
    @pytest.fixture
    def harness(self):
        """Create harness instance"""
        return BenchmarkHarness(max_problems_per_benchmark=2)
    
    def test_full_workflow_with_mock_agent(self, harness, temp_dir):
        """Test full workflow with mocked agent"""
        with patch('subprocess.run') as mock_run:
            # Mock successful agent execution
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "def solution(): return 'success'"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            # Create sample problems
            problems = {
                'swe_bench': [
                    BenchmarkProblem(
                        benchmark='swe_bench',
                        problem_id='test_1',
                        description='Write a simple function',
                        language='python',
                        difficulty='easy'
                    )
                ]
            }
            
            # Run evaluation
            harness.evaluate_benchmarks(problems)
            
            assert len(harness.results) == 1
            assert harness.results[0].success is True
            
            # Calculate metrics
            harness.calculate_metrics()
            
            assert len(harness.metrics) == 1
            assert harness.metrics[0].success_rate == 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 