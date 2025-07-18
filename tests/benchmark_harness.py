#!/usr/bin/env python3
"""
Comprehensive Benchmark Testing Harness for Multi-Agent Coding System

Evaluates the system against specialized AI coding benchmarks:
- SWE-Bench (Software Engineering Benchmark)
- LiveBench/LiveCodeBench (Dynamic problems)
- RExBench (Research code extension)
- GAIA (General AI Assistants)
- LMC-Eval (Logical coding questions)

Focuses exclusively on Python problems for quick evaluation.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import git
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics"""
    benchmark: str
    total_problems: int
    successful_problems: int
    success_rate: float
    avg_execution_time: float
    avg_code_length: int
    error_distribution: Dict[str, int]

class BenchmarkHarness:
    """Main benchmark testing harness"""
    
    def __init__(self, temp_dir: Optional[str] = None, max_problems_per_benchmark: int = 5):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="benchmark_harness_")
        self.max_problems_per_benchmark = max_problems_per_benchmark
        self.results: List[BenchmarkResult] = []
        self.metrics: List[BenchmarkMetrics] = []
        
        # Benchmark configurations
        self.benchmarks = {
            'swe_bench': {
                'repo': 'https://github.com/swe-bench/swe-bench',
                'problems_dir': 'data',
                'test_script': 'eval.py',
                'language_filter': ['python']
            },
            'livebench': {
                'repo': 'https://github.com/livebench/livebench',
                'problems_dir': 'problems',
                'test_script': 'evaluate.py',
                'language_filter': ['python']
            },
            'rexbench': {
                'repo': 'https://github.com/rexbench/rexbench',
                'problems_dir': 'tasks',
                'test_script': 'eval.py',
                'language_filter': ['python']
            },
            'gaia': {
                'repo': 'https://github.com/gaia-benchmark/GAIA',
                'problems_dir': 'coding_tasks',
                'test_script': 'evaluate.py',
                'language_filter': ['python']
            },
            'lmc_eval': {
                'repo': 'https://github.com/lmc-eval/lmc-eval',
                'problems_dir': 'problems',
                'test_script': 'eval.py',
                'language_filter': ['python']
            }
        }
    
    def setup_benchmarks(self) -> bool:
        """Clone and setup benchmark repositories"""
        print("🔧 Setting up benchmark repositories...")
        
        for benchmark_name, config in self.benchmarks.items():
            benchmark_dir = Path(self.temp_dir) / benchmark_name
            
            try:
                if not benchmark_dir.exists():
                    print(f"📥 Cloning {benchmark_name}...")
                    git.Repo.clone_from(config['repo'], benchmark_dir)
                
                # Install dependencies if requirements.txt exists
                requirements_file = benchmark_dir / 'requirements.txt'
                if requirements_file.exists():
                    print(f"📦 Installing dependencies for {benchmark_name}...")
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
                    ], check=True, capture_output=True)
                
                print(f"✅ {benchmark_name} setup complete")
                
            except Exception as e:
                print(f"❌ Failed to setup {benchmark_name}: {e}")
                return False
        
        return True
    
    def extract_problems(self) -> Dict[str, List[BenchmarkProblem]]:
        """Extract problems from benchmark repositories"""
        print("📋 Extracting benchmark problems...")
        
        problems = {}
        
        for benchmark_name, config in self.benchmarks.items():
            benchmark_dir = Path(self.temp_dir) / benchmark_name
            problems_dir = benchmark_dir / config['problems_dir']
            
            if not problems_dir.exists():
                print(f"⚠️  Problems directory not found for {benchmark_name}")
                continue
            
            benchmark_problems = self._extract_benchmark_problems(
                benchmark_name, problems_dir, config
            )
            problems[benchmark_name] = benchmark_problems[:self.max_problems_per_benchmark]
            
            print(f"📊 Extracted {len(problems[benchmark_name])} problems from {benchmark_name}")
        
        return problems
    
    def _extract_benchmark_problems(self, benchmark_name: str, problems_dir: Path, config: Dict) -> List[BenchmarkProblem]:
        """Extract problems for a specific benchmark"""
        problems = []
        
        try:
            if benchmark_name == 'swe_bench':
                problems = self._extract_swe_bench_problems(problems_dir)
            elif benchmark_name == 'livebench':
                problems = self._extract_livebench_problems(problems_dir)
            elif benchmark_name == 'rexbench':
                problems = self._extract_rexbench_problems(problems_dir)
            elif benchmark_name == 'gaia':
                problems = self._extract_gaia_problems(problems_dir)
            elif benchmark_name == 'lmc_eval':
                problems = self._extract_lmc_eval_problems(problems_dir)
            
            # Filter by language
            problems = [p for p in problems if p.language in config['language_filter']]
            
        except Exception as e:
            print(f"❌ Error extracting {benchmark_name} problems: {e}")
        
        return problems
    
    def _extract_swe_bench_problems(self, problems_dir: Path) -> List[BenchmarkProblem]:
        """Extract SWE-Bench problems"""
        problems = []
        
        # Look for JSON files with problem data
        for json_file in problems_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if 'python' in item.get('language', '').lower():
                            problems.append(BenchmarkProblem(
                                benchmark='swe_bench',
                                problem_id=item.get('id', str(len(problems))),
                                description=item.get('description', ''),
                                language='python',
                                difficulty=item.get('difficulty', 'medium'),
                                test_file=item.get('test_file'),
                                expected_output=item.get('expected_output')
                            ))
                elif isinstance(data, dict):
                    if 'python' in data.get('language', '').lower():
                        problems.append(BenchmarkProblem(
                            benchmark='swe_bench',
                            problem_id=data.get('id', str(len(problems))),
                            description=data.get('description', ''),
                            language='python',
                            difficulty=data.get('difficulty', 'medium'),
                            test_file=data.get('test_file'),
                            expected_output=data.get('expected_output')
                        ))
                        
            except Exception as e:
                print(f"⚠️  Error reading {json_file}: {e}")
        
        return problems
    
    def _extract_livebench_problems(self, problems_dir: Path) -> List[BenchmarkProblem]:
        """Extract LiveBench problems"""
        problems = []
        
        # Look for Python problem files
        for py_file in problems_dir.glob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Extract problem description from comments
                description = self._extract_description_from_comments(content)
                
                problems.append(BenchmarkProblem(
                    benchmark='livebench',
                    problem_id=py_file.stem,
                    description=description or f"Solve the problem in {py_file.name}",
                    language='python',
                    difficulty='medium',
                    test_file=str(py_file)
                ))
                
            except Exception as e:
                print(f"⚠️  Error reading {py_file}: {e}")
        
        return problems
    
    def _extract_rexbench_problems(self, problems_dir: Path) -> List[BenchmarkProblem]:
        """Extract RExBench problems"""
        problems = []
        
        # Look for task directories
        for task_dir in problems_dir.iterdir():
            if task_dir.is_dir():
                try:
                    # Look for README or description file
                    readme_file = task_dir / 'README.md'
                    if readme_file.exists():
                        with open(readme_file, 'r') as f:
                            description = f.read()
                    else:
                        description = f"Implement the research task in {task_dir.name}"
                    
                    problems.append(BenchmarkProblem(
                        benchmark='rexbench',
                        problem_id=task_dir.name,
                        description=description,
                        language='python',
                        difficulty='hard',
                        test_file=str(task_dir / 'test.py') if (task_dir / 'test.py').exists() else None
                    ))
                    
                except Exception as e:
                    print(f"⚠️  Error reading {task_dir}: {e}")
        
        return problems
    
    def _extract_gaia_problems(self, problems_dir: Path) -> List[BenchmarkProblem]:
        """Extract GAIA problems"""
        problems = []
        
        # Look for JSON or YAML files with task definitions
        for file_path in problems_dir.glob("*"):
            if file_path.suffix in ['.json', '.yaml', '.yml']:
                try:
                    with open(file_path, 'r') as f:
                        if file_path.suffix == '.json':
                            data = json.load(f)
                        else:
                            import yaml
                            data = yaml.safe_load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if 'python' in item.get('language', '').lower():
                                problems.append(BenchmarkProblem(
                                    benchmark='gaia',
                                    problem_id=item.get('id', str(len(problems))),
                                    description=item.get('description', ''),
                                    language='python',
                                    difficulty=item.get('difficulty', 'medium')
                                ))
                    elif isinstance(data, dict):
                        if 'python' in data.get('language', '').lower():
                            problems.append(BenchmarkProblem(
                                benchmark='gaia',
                                problem_id=data.get('id', str(len(problems))),
                                description=data.get('description', ''),
                                language='python',
                                difficulty=data.get('difficulty', 'medium')
                            ))
                            
                except Exception as e:
                    print(f"⚠️  Error reading {file_path}: {e}")
        
        return problems
    
    def _extract_lmc_eval_problems(self, problems_dir: Path) -> List[BenchmarkProblem]:
        """Extract LMC-Eval problems"""
        problems = []
        
        # Look for problem files
        for file_path in problems_dir.glob("*"):
            if file_path.suffix in ['.json', '.txt', '.md']:
                try:
                    if file_path.suffix == '.json':
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        if isinstance(data, list):
                            for item in data:
                                if 'python' in item.get('language', '').lower():
                                    problems.append(BenchmarkProblem(
                                        benchmark='lmc_eval',
                                        problem_id=item.get('id', str(len(problems))),
                                        description=item.get('question', ''),
                                        language='python',
                                        difficulty=item.get('difficulty', 'medium')
                                    ))
                    else:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        problems.append(BenchmarkProblem(
                            benchmark='lmc_eval',
                            problem_id=file_path.stem,
                            description=content[:500] + "..." if len(content) > 500 else content,
                            language='python',
                            difficulty='medium'
                        ))
                        
                except Exception as e:
                    print(f"⚠️  Error reading {file_path}: {e}")
        
        return problems
    
    def _extract_description_from_comments(self, content: str) -> str:
        """Extract problem description from Python file comments"""
        lines = content.split('\n')
        description_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') and not line.startswith('#!'):
                description_lines.append(line[1:].strip())
            elif line.startswith('"""') or line.startswith("'''"):
                # Found docstring
                break
            elif line and not line.startswith('#'):
                # Found code, stop looking
                break
        
        return ' '.join(description_lines) if description_lines else ""
    
    def run_agent_on_problem(self, problem: BenchmarkProblem) -> BenchmarkResult:
        """Run our agent on a benchmark problem"""
        start_time = time.time()
        
        try:
            # Prepare the prompt
            prompt = f"Solve this {problem.benchmark} problem: {problem.description}"
            
            # Run the agent
            result = subprocess.run([
                sys.executable, 'main.py', prompt
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            execution_time = time.time() - start_time
            agent_output = result.stdout
            
            # Extract code from output
            code_length = len(agent_output)
            
            # Test the solution if test file exists
            success = False
            error_type = None
            error_message = None
            test_output = None
            
            if problem.test_file and os.path.exists(problem.test_file):
                try:
                    # Write agent output to temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(agent_output)
                        temp_file = f.name
                    
                    # Run tests
                    test_result = subprocess.run([
                        sys.executable, '-m', 'pytest', temp_file, '-v'
                    ], capture_output=True, text=True, timeout=60)
                    
                    test_output = test_result.stdout + test_result.stderr
                    success = test_result.returncode == 0
                    
                    if not success:
                        error_type = 'test_failure'
                        error_message = test_result.stderr
                    
                    # Clean up
                    os.unlink(temp_file)
                    
                except subprocess.TimeoutExpired:
                    error_type = 'timeout'
                    error_message = 'Test execution timed out'
                except Exception as e:
                    error_type = 'test_error'
                    error_message = str(e)
            else:
                # No test file, assume success if no obvious errors
                success = result.returncode == 0 and 'error' not in agent_output.lower()
                if not success:
                    error_type = 'execution_error'
                    error_message = result.stderr
            
            return BenchmarkResult(
                benchmark=problem.benchmark,
                problem_id=problem.problem_id,
                success=success,
                execution_time=execution_time,
                code_length=code_length,
                error_type=error_type,
                error_message=error_message,
                test_output=test_output,
                agent_output=agent_output
            )
            
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                benchmark=problem.benchmark,
                problem_id=problem.problem_id,
                success=False,
                execution_time=time.time() - start_time,
                code_length=0,
                error_type='timeout',
                error_message='Agent execution timed out'
            )
        except Exception as e:
            return BenchmarkResult(
                benchmark=problem.benchmark,
                problem_id=problem.problem_id,
                success=False,
                execution_time=time.time() - start_time,
                code_length=0,
                error_type='system_error',
                error_message=str(e)
            )
    
    def evaluate_benchmarks(self, problems: Dict[str, List[BenchmarkProblem]]) -> None:
        """Evaluate all benchmark problems"""
        print("🚀 Starting benchmark evaluation...")
        
        total_problems = sum(len(probs) for probs in problems.values())
        completed = 0
        
        for benchmark_name, benchmark_problems in problems.items():
            print(f"\n📊 Evaluating {benchmark_name} ({len(benchmark_problems)} problems)...")
            
            for problem in benchmark_problems:
                print(f"  🔍 Problem {problem.problem_id}: {problem.description[:50]}...")
                
                result = self.run_agent_on_problem(problem)
                self.results.append(result)
                
                completed += 1
                print(f"    {'✅' if result.success else '❌'} {result.success} "
                      f"({result.execution_time:.2f}s, {result.code_length} chars)")
        
        print(f"\n✅ Completed {completed}/{total_problems} problems")
    
    def calculate_metrics(self) -> None:
        """Calculate aggregated metrics"""
        print("📈 Calculating metrics...")
        
        for benchmark_name in self.benchmarks.keys():
            benchmark_results = [r for r in self.results if r.benchmark == benchmark_name]
            
            if not benchmark_results:
                continue
            
            total_problems = len(benchmark_results)
            successful_problems = sum(1 for r in benchmark_results if r.success)
            success_rate = successful_problems / total_problems if total_problems > 0 else 0
            avg_execution_time = sum(r.execution_time for r in benchmark_results) / total_problems
            avg_code_length = sum(r.code_length for r in benchmark_results) / total_problems
            
            # Error distribution
            error_distribution = {}
            for result in benchmark_results:
                if result.error_type:
                    error_distribution[result.error_type] = error_distribution.get(result.error_type, 0) + 1
            
            metrics = BenchmarkMetrics(
                benchmark=benchmark_name,
                total_problems=total_problems,
                successful_problems=successful_problems,
                success_rate=success_rate,
                avg_execution_time=avg_execution_time,
                avg_code_length=avg_code_length,
                error_distribution=error_distribution
            )
            
            self.metrics.append(metrics)
    
    def generate_report(self, output_file: str = "benchmark_results.json") -> None:
        """Generate comprehensive benchmark report"""
        print(f"📊 Generating report: {output_file}")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_problems': len(self.results),
                'total_successful': sum(1 for r in self.results if r.success),
                'overall_success_rate': sum(1 for r in self.results if r.success) / len(self.results) if self.results else 0,
                'avg_execution_time': sum(r.execution_time for r in self.results) / len(self.results) if self.results else 0,
                'avg_code_length': sum(r.code_length for r in self.results) / len(self.results) if self.results else 0
            },
            'benchmark_metrics': [asdict(m) for m in self.metrics],
            'detailed_results': [asdict(r) for r in self.results]
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("📊 BENCHMARK EVALUATION SUMMARY")
        print("="*80)
        
        if not self.metrics:
            print("❌ No metrics available")
            return
        
        # Overall summary
        total_problems = sum(m.total_problems for m in self.metrics)
        total_successful = sum(m.successful_problems for m in self.metrics)
        overall_success_rate = total_successful / total_problems if total_problems > 0 else 0
        
        print(f"🎯 Overall Success Rate: {overall_success_rate:.1%} ({total_successful}/{total_problems})")
        print(f"⏱️  Average Execution Time: {sum(m.avg_execution_time for m in self.metrics) / len(self.metrics):.2f}s")
        print(f"📝 Average Code Length: {sum(m.avg_code_length for m in self.metrics) / len(self.metrics):.0f} characters")
        
        print("\n📈 Per-Benchmark Results:")
        print("-" * 60)
        
        for metrics in self.metrics:
            print(f"{metrics.benchmark:15} | "
                  f"Success: {metrics.success_rate:6.1%} | "
                  f"Problems: {metrics.total_problems:3d} | "
                  f"Time: {metrics.avg_execution_time:5.2f}s | "
                  f"Length: {metrics.avg_code_length:6.0f} chars")
        
        print("\n🚨 Error Distribution:")
        print("-" * 40)
        
        all_errors = {}
        for result in self.results:
            if result.error_type:
                all_errors[result.error_type] = all_errors.get(result.error_type, 0) + 1
        
        for error_type, count in sorted(all_errors.items(), key=lambda x: x[1], reverse=True):
            print(f"{error_type:20} | {count:3d} occurrences")
        
        print("="*80)

def main():
    """Main benchmark harness execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Testing Harness")
    parser.add_argument("--temp-dir", help="Temporary directory for benchmarks")
    parser.add_argument("--max-problems", type=int, default=5, 
                       help="Maximum problems per benchmark")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output report file")
    parser.add_argument("--skip-setup", action="store_true",
                       help="Skip benchmark setup (use existing)")
    
    args = parser.parse_args()
    
    print("🚀 Multi-Agent Coding System Benchmark Harness")
    print("=" * 60)
    
    # Initialize harness
    harness = BenchmarkHarness(
        temp_dir=args.temp_dir,
        max_problems_per_benchmark=args.max_problems
    )
    
    try:
        # Setup benchmarks
        if not args.skip_setup:
            if not harness.setup_benchmarks():
                print("❌ Benchmark setup failed")
                return 1
        
        # Extract problems
        problems = harness.extract_problems()
        
        if not problems:
            print("❌ No problems extracted from benchmarks")
            return 1
        
        total_problems = sum(len(probs) for probs in problems.values())
        print(f"📋 Extracted {total_problems} problems total")
        
        # Evaluate benchmarks
        harness.evaluate_benchmarks(problems)
        
        # Calculate metrics
        harness.calculate_metrics()
        
        # Generate report
        harness.generate_report(args.output)
        
        print(f"\n✅ Benchmark evaluation complete!")
        print(f"📊 Results saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark evaluation interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 