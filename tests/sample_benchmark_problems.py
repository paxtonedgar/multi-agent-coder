#!/usr/bin/env python3
"""
Sample benchmark problems for testing the harness
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_harness import BenchmarkProblem

def create_sample_benchmark_data(temp_dir: str) -> Dict[str, List[BenchmarkProblem]]:
    """Create sample benchmark data for testing"""
    
    # Create sample SWE-Bench problems
    swe_bench_problems = [
        BenchmarkProblem(
            benchmark='swe_bench',
            problem_id='swe_001',
            description='Fix the bug in the fibonacci function that causes infinite recursion for negative numbers',
            language='python',
            difficulty='easy',
            test_file=None
        ),
        BenchmarkProblem(
            benchmark='swe_bench',
            problem_id='swe_002',
            description='Implement a function to find the longest common subsequence between two strings',
            language='python',
            difficulty='medium',
            test_file=None
        ),
        BenchmarkProblem(
            benchmark='swe_bench',
            problem_id='swe_003',
            description='Create a class to represent a binary tree with methods for insertion, deletion, and traversal',
            language='python',
            difficulty='hard',
            test_file=None
        )
    ]
    
    # Create sample LiveBench problems
    livebench_problems = [
        BenchmarkProblem(
            benchmark='livebench',
            problem_id='live_001',
            description='Write a function to implement the quicksort algorithm with proper pivot selection',
            language='python',
            difficulty='medium',
            test_file=None
        ),
        BenchmarkProblem(
            benchmark='livebench',
            problem_id='live_002',
            description='Create a REST API endpoint that accepts JSON data and returns a sorted version',
            language='python',
            difficulty='hard',
            test_file=None
        )
    ]
    
    # Create sample RExBench problems
    rexbench_problems = [
        BenchmarkProblem(
            benchmark='rexbench',
            problem_id='rex_001',
            description='Implement a neural network layer from scratch using only numpy, following the architecture described in the research paper',
            language='python',
            difficulty='hard',
            test_file=None
        )
    ]
    
    # Create sample GAIA problems
    gaia_problems = [
        BenchmarkProblem(
            benchmark='gaia',
            problem_id='gaia_001',
            description='Create a simple game of Tic-Tac-Toe with a command-line interface',
            language='python',
            difficulty='medium',
            test_file=None
        ),
        BenchmarkProblem(
            benchmark='gaia',
            problem_id='gaia_002',
            description='Write a data processing script that reads a CSV file and generates summary statistics',
            language='python',
            difficulty='easy',
            test_file=None
        )
    ]
    
    # Create sample LMC-Eval problems
    lmc_eval_problems = [
        BenchmarkProblem(
            benchmark='lmc_eval',
            problem_id='lmc_001',
            description='Write a function that determines if a string is a palindrome, ignoring spaces and punctuation',
            language='python',
            difficulty='easy',
            test_file=None
        ),
        BenchmarkProblem(
            benchmark='lmc_eval',
            problem_id='lmc_002',
            description='Implement a function to find all prime numbers up to a given limit using the Sieve of Eratosthenes',
            language='python',
            difficulty='medium',
            test_file=None
        )
    ]
    
    return {
        'swe_bench': swe_bench_problems,
        'livebench': livebench_problems,
        'rexbench': rexbench_problems,
        'gaia': gaia_problems,
        'lmc_eval': lmc_eval_problems
    }

def create_mock_benchmark_repos(temp_dir: str) -> None:
    """Create mock benchmark repository structure for testing"""
    
    temp_path = Path(temp_dir)
    
    # Create SWE-Bench mock structure
    swe_bench_dir = temp_path / 'swe_bench'
    swe_bench_dir.mkdir()
    data_dir = swe_bench_dir / 'data'
    data_dir.mkdir()
    
    swe_bench_data = [
        {
            "id": "swe_001",
            "description": "Fix the bug in the fibonacci function that causes infinite recursion for negative numbers",
            "language": "python",
            "difficulty": "easy"
        },
        {
            "id": "swe_002", 
            "description": "Implement a function to find the longest common subsequence between two strings",
            "language": "python",
            "difficulty": "medium"
        }
    ]
    
    with open(data_dir / 'problems.json', 'w') as f:
        json.dump(swe_bench_data, f, indent=2)
    
    # Create LiveBench mock structure
    livebench_dir = temp_path / 'livebench'
    livebench_dir.mkdir()
    problems_dir = livebench_dir / 'problems'
    problems_dir.mkdir()
    
    # Create sample Python problem files
    quicksort_problem = '''
# Implement quicksort algorithm
# Write a function that sorts a list using the quicksort algorithm
# The function should handle edge cases like empty lists and lists with duplicates

def quicksort(arr):
    # TODO: Implement quicksort
    pass
'''
    
    with open(problems_dir / 'quicksort.py', 'w') as f:
        f.write(quicksort_problem)
    
    # Create RExBench mock structure
    rexbench_dir = temp_path / 'rexbench'
    rexbench_dir.mkdir()
    tasks_dir = rexbench_dir / 'tasks'
    tasks_dir.mkdir()
    
    neural_task_dir = tasks_dir / 'neural_network'
    neural_task_dir.mkdir()
    
    with open(neural_task_dir / 'README.md', 'w') as f:
        f.write("""
# Neural Network Implementation

Implement a neural network layer from scratch using only numpy.
Follow the architecture described in the research paper.

Requirements:
- Use only numpy for calculations
- Implement forward and backward pass
- Handle batch processing
- Include proper initialization
        """)
    
    # Create GAIA mock structure
    gaia_dir = temp_path / 'gaia'
    gaia_dir.mkdir()
    coding_tasks_dir = gaia_dir / 'coding_tasks'
    coding_tasks_dir.mkdir()
    
    gaia_data = [
        {
            "id": "gaia_001",
            "description": "Create a simple game of Tic-Tac-Toe with a command-line interface",
            "language": "python",
            "difficulty": "medium"
        },
        {
            "id": "gaia_002",
            "description": "Write a data processing script that reads a CSV file and generates summary statistics",
            "language": "python", 
            "difficulty": "easy"
        }
    ]
    
    with open(coding_tasks_dir / 'tasks.json', 'w') as f:
        json.dump(gaia_data, f, indent=2)
    
    # Create LMC-Eval mock structure
    lmc_eval_dir = temp_path / 'lmc_eval'
    lmc_eval_dir.mkdir()
    problems_dir = lmc_eval_dir / 'problems'
    problems_dir.mkdir()
    
    lmc_data = [
        {
            "id": "lmc_001",
            "question": "Write a function that determines if a string is a palindrome, ignoring spaces and punctuation",
            "language": "python",
            "difficulty": "easy"
        },
        {
            "id": "lmc_002",
            "question": "Implement a function to find all prime numbers up to a given limit using the Sieve of Eratosthenes",
            "language": "python",
            "difficulty": "medium"
        }
    ]
    
    with open(problems_dir / 'problems.json', 'w') as f:
        json.dump(lmc_data, f, indent=2)

def run_sample_benchmark_test():
    """Run a quick test with sample benchmark problems"""
    print("🧪 Running sample benchmark test...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock benchmark repositories
        create_mock_benchmark_repos(temp_dir)
        
        # Create sample problems
        problems = create_sample_benchmark_data(temp_dir)
        
        # Initialize harness
        from benchmark_harness import BenchmarkHarness
        harness = BenchmarkHarness(temp_dir=temp_dir, max_problems_per_benchmark=2)
        
        # Test problem extraction
        extracted_problems = harness.extract_problems()
        
        print(f"📊 Extracted problems:")
        for benchmark, probs in extracted_problems.items():
            print(f"  {benchmark}: {len(probs)} problems")
        
        # Test with a single problem
        if extracted_problems:
            first_benchmark = list(extracted_problems.keys())[0]
            first_problem = extracted_problems[first_benchmark][0]
            
            print(f"\n🔍 Testing with problem: {first_problem.problem_id}")
            print(f"   Description: {first_problem.description[:100]}...")
            
            # Mock the agent execution
            from unittest.mock import patch, MagicMock
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "def solution(): return 'test solution'"
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                result = harness.run_agent_on_problem(first_problem)
                
                print(f"   Result: {'✅ Success' if result.success else '❌ Failed'}")
                print(f"   Execution time: {result.execution_time:.2f}s")
                print(f"   Code length: {result.code_length} characters")
        
        print("\n✅ Sample benchmark test completed!")

if __name__ == "__main__":
    run_sample_benchmark_test() 