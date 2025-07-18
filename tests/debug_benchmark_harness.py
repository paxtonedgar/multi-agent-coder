#!/usr/bin/env python3
"""
Debug Benchmark Harness - Shows detailed output and error information
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class DebugBenchmarkProblem:
    """Debug benchmark problem"""
    benchmark: str
    problem_id: str
    description: str
    language: str
    difficulty: str

@dataclass
class DebugBenchmarkResult:
    """Debug benchmark result with detailed logging"""
    benchmark: str
    problem_id: str
    success: bool
    execution_time: float
    code_length: int
    error_type: str = None
    error_message: str = None
    agent_output: str = None
    agent_stderr: str = None
    return_code: int = None
    quality_score: float = 0.0
    performance_score: float = 0.0

class DebugBenchmarkHarness:
    """Debug benchmark harness with detailed logging"""
    
    def __init__(self):
        self.results: List[DebugBenchmarkResult] = []
        
        # Sample problems for debugging
        self.sample_problems = [
            DebugBenchmarkProblem(
                benchmark='swe_bench',
                problem_id='swe_001',
                description='Fix the bug in the fibonacci function that causes infinite recursion for negative numbers. The function should raise a ValueError for negative inputs.',
                language='python',
                difficulty='easy'
            ),
            DebugBenchmarkProblem(
                benchmark='swe_bench',
                problem_id='swe_002',
                description='Write a simple function that returns the sum of two numbers.',
                language='python',
                difficulty='easy'
            ),
            DebugBenchmarkProblem(
                benchmark='swe_bench',
                problem_id='swe_003',
                description='Create a function that checks if a string is a palindrome.',
                language='python',
                difficulty='easy'
            )
        ]
    
    def run_agent_with_debug(self, problem: DebugBenchmarkProblem) -> DebugBenchmarkResult:
        """Run agent with detailed debugging information"""
        print(f"\n🔍 DEBUG: Running problem {problem.problem_id}")
        print(f"📝 Description: {problem.description}")
        
        start_time = time.time()
        
        try:
            # Prepare the prompt
            prompt = f"Solve this {problem.benchmark} problem: {problem.description}"
            print(f"🤖 Prompt: {prompt[:100]}...")
            
            # Run the agent with timeout
            print("⏱️  Starting agent execution...")
            result = subprocess.run([
                sys.executable, 'main.py', prompt
            ], capture_output=True, text=True, timeout=60)  # 1 minute timeout
            
            execution_time = time.time() - start_time
            agent_output = result.stdout
            agent_stderr = result.stderr
            
            print(f"⏱️  Execution time: {execution_time:.2f}s")
            print(f"📊 Return code: {result.returncode}")
            print(f"📝 Output length: {len(agent_output)} characters")
            print(f"⚠️  Error length: {len(agent_stderr)} characters")
            
            # Show first 500 characters of output
            print(f"📄 Output preview:")
            print("-" * 50)
            print(agent_output[:500])
            if len(agent_output) > 500:
                print("... (truncated)")
            print("-" * 50)
            
            # Show stderr if any
            if agent_stderr:
                print(f"❌ Error output:")
                print("-" * 50)
                print(agent_stderr[:500])
                if len(agent_stderr) > 500:
                    print("... (truncated)")
                print("-" * 50)
            
            # Analyze the output
            success = result.returncode == 0 and 'error' not in agent_output.lower()
            quality_score = self._calculate_quality_score(agent_output, problem)
            performance_score = self._calculate_performance_score(execution_time, len(agent_output))
            
            error_type = None
            error_message = None
            
            if result.returncode != 0:
                error_type = 'execution_error'
                error_message = f"Return code: {result.returncode}"
            elif 'error' in agent_output.lower():
                error_type = 'output_error'
                error_message = 'Output contains error'
            elif len(agent_output) > 5000:
                error_type = 'verbose_output'
                error_message = f'Output too verbose: {len(agent_output)} characters'
            elif execution_time > 30:
                error_type = 'slow_execution'
                error_message = f'Execution too slow: {execution_time:.2f}s'
            
            print(f"✅ Success: {success}")
            print(f"⭐ Quality Score: {quality_score:.2f}")
            print(f"⚡ Performance Score: {performance_score:.2f}")
            if error_type:
                print(f"❌ Error Type: {error_type}")
                print(f"❌ Error Message: {error_message}")
            
            return DebugBenchmarkResult(
                benchmark=problem.benchmark,
                problem_id=problem.problem_id,
                success=success,
                execution_time=execution_time,
                code_length=len(agent_output),
                error_type=error_type,
                error_message=error_message,
                agent_output=agent_output,
                agent_stderr=agent_stderr,
                return_code=result.returncode,
                quality_score=quality_score,
                performance_score=performance_score
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"⏰ TIMEOUT after {execution_time:.2f}s")
            
            return DebugBenchmarkResult(
                benchmark=problem.benchmark,
                problem_id=problem.problem_id,
                success=False,
                execution_time=execution_time,
                code_length=0,
                error_type='timeout',
                error_message=f'Agent execution timed out after {execution_time:.2f}s',
                quality_score=0.0,
                performance_score=0.0
            )
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"💥 EXCEPTION: {e}")
            
            return DebugBenchmarkResult(
                benchmark=problem.benchmark,
                problem_id=problem.problem_id,
                success=False,
                execution_time=execution_time,
                code_length=0,
                error_type='system_error',
                error_message=str(e),
                quality_score=0.0,
                performance_score=0.0
            )
    
    def _calculate_quality_score(self, output: str, problem: DebugBenchmarkProblem) -> float:
        """Calculate quality score with detailed analysis"""
        score = 0.0
        analysis = []
        
        # Check for code presence
        if 'def ' in output or 'class ' in output:
            score += 0.3
            analysis.append("✅ Contains function/class definition")
        else:
            analysis.append("❌ No function/class definition")
        
        # Check for proper structure
        if 'import ' in output or 'from ' in output:
            score += 0.2
            analysis.append("✅ Contains imports")
        else:
            analysis.append("❌ No imports")
        
        # Check for documentation
        if '"""' in output or "'''" in output or '# ' in output:
            score += 0.2
            analysis.append("✅ Contains documentation")
        else:
            analysis.append("❌ No documentation")
        
        # Check for error handling
        if 'try:' in output or 'except' in output or 'if ' in output:
            score += 0.2
            analysis.append("✅ Contains error handling/conditionals")
        else:
            analysis.append("❌ No error handling/conditionals")
        
        # Check for reasonable length
        if 100 <= len(output) <= 2000:
            score += 0.1
            analysis.append("✅ Reasonable output length")
        else:
            analysis.append(f"❌ Output length: {len(output)} chars (expected 100-2000)")
        
        print(f"📊 Quality Analysis:")
        for item in analysis:
            print(f"   {item}")
        
        return min(score, 1.0)
    
    def _calculate_performance_score(self, execution_time: float, code_length: int) -> float:
        """Calculate performance score with detailed analysis"""
        print(f"⚡ Performance Analysis:")
        
        # Time score (faster is better, max 60s)
        time_score = max(0, 1 - (execution_time / 60))
        print(f"   ⏱️  Execution time: {execution_time:.2f}s (score: {time_score:.2f})")
        
        # Length score (reasonable length is better)
        if 100 <= code_length <= 1000:
            length_score = 1.0
            print(f"   📝 Code length: {code_length} chars (score: {length_score:.2f}) - Good length")
        elif code_length < 100:
            length_score = code_length / 100
            print(f"   📝 Code length: {code_length} chars (score: {length_score:.2f}) - Too short")
        else:
            length_score = max(0, 1 - ((code_length - 1000) / 1000))
            print(f"   📝 Code length: {code_length} chars (score: {length_score:.2f}) - Too verbose")
        
        overall_score = (time_score + length_score) / 2
        print(f"   🎯 Overall performance score: {overall_score:.2f}")
        
        return overall_score
    
    def run_debug_evaluation(self) -> None:
        """Run debug evaluation on sample problems"""
        print("🚀 Debug Benchmark Evaluation")
        print("=" * 60)
        
        for problem in self.sample_problems:
            print(f"\n{'='*60}")
            print(f"🧪 Testing Problem: {problem.problem_id}")
            print(f"{'='*60}")
            
            result = self.run_agent_with_debug(problem)
            self.results.append(result)
            
            print(f"\n📊 Problem {problem.problem_id} Summary:")
            print(f"   Success: {'✅ YES' if result.success else '❌ NO'}")
            print(f"   Quality: {result.quality_score:.2f}")
            print(f"   Performance: {result.performance_score:.2f}")
            print(f"   Time: {result.execution_time:.2f}s")
            print(f"   Length: {result.code_length} chars")
            if result.error_type:
                print(f"   Error: {result.error_type} - {result.error_message}")
        
        self.print_debug_summary()
    
    def print_debug_summary(self) -> None:
        """Print detailed debug summary"""
        print(f"\n{'='*80}")
        print("📊 DEBUG EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        if not self.results:
            print("❌ No results to report")
            return
        
        total_problems = len(self.results)
        successful_problems = sum(1 for r in self.results if r.success)
        overall_success_rate = successful_problems / total_problems
        
        avg_quality = sum(r.quality_score for r in self.results) / total_problems
        avg_performance = sum(r.performance_score for r in self.results) / total_problems
        avg_time = sum(r.execution_time for r in self.results) / total_problems
        avg_length = sum(r.code_length for r in self.results) / total_problems
        
        print(f"🎯 Overall Success Rate: {overall_success_rate:.1%} ({successful_problems}/{total_problems})")
        print(f"⭐ Average Quality Score: {avg_quality:.2f}")
        print(f"⚡ Average Performance Score: {avg_performance:.2f}")
        print(f"⏱️  Average Execution Time: {avg_time:.2f}s")
        print(f"📝 Average Code Length: {avg_length:.0f} characters")
        
        print(f"\n📈 Per-Problem Results:")
        print("-" * 80)
        
        for result in self.results:
            print(f"{result.problem_id:10} | "
                  f"Success: {'✅' if result.success else '❌'} | "
                  f"Quality: {result.quality_score:5.2f} | "
                  f"Performance: {result.performance_score:5.2f} | "
                  f"Time: {result.execution_time:6.2f}s | "
                  f"Length: {result.code_length:6d} chars")
            if result.error_type:
                print(f"           | Error: {result.error_type} - {result.error_message}")
        
        print(f"\n🚨 Error Distribution:")
        print("-" * 40)
        
        error_counts = {}
        for result in self.results:
            if result.error_type:
                error_counts[result.error_type] = error_counts.get(result.error_type, 0) + 1
        
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{error_type:20} | {count:3d} occurrences")
        
        print("="*80)
        
        # Save detailed results
        debug_file = "debug_benchmark_results.json"
        with open(debug_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_problems': total_problems,
                    'successful_problems': successful_problems,
                    'overall_success_rate': overall_success_rate,
                    'avg_quality_score': avg_quality,
                    'avg_performance_score': avg_performance,
                    'avg_execution_time': avg_time,
                    'avg_code_length': avg_length
                },
                'detailed_results': [asdict(r) for r in self.results]
            }, f, indent=2)
        
        print(f"\n📊 Detailed results saved to: {debug_file}")

def main():
    """Main debug benchmark execution"""
    print("🔍 Debug Benchmark Harness")
    print("=" * 60)
    
    harness = DebugBenchmarkHarness()
    
    try:
        harness.run_debug_evaluation()
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Debug evaluation interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Debug evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 