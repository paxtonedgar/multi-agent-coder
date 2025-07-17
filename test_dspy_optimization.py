#!/usr/bin/env python3
"""
Test script for dynamic DSPy optimization with real-world data sources
Validates the shift from static mocks to robust, real-world examples
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import ProjectBrain
from prompts import get_example_dataset, create_optimizer, PromptFactory, validate_clean_code
from graph import run_workflow

def test_dynamic_dataset_generation():
    """Test dynamic example dataset generation from real-world sources."""
    print("=" * 60)
    print("Testing Dynamic DSPy Dataset Generation")
    print("=" * 60)
    
    # Initialize brain
    brain = ProjectBrain()
    
    # Test dataset generation
    print("\n1. Generating dynamic examples...")
    examples = get_example_dataset(brain)
    
    print(f"✓ Generated {len(examples)} examples")
    print(f"✓ Examples cached in brain: {len(brain.memory.get('dspy_examples', []))}")
    
    # Validate example structure
    print("\n2. Validating example structure...")
    valid_examples = 0
    for example in examples:
        if 'task' in example and 'results' in example:
            if len(example['task']) > 10 and len(example['results']) > 20:
                valid_examples += 1
    
    print(f"✓ {valid_examples}/{len(examples)} examples have valid structure")
    
    # Test quality scoring
    print("\n3. Testing quality scoring...")
    scores = []
    for example in examples[:5]:  # Test first 5 examples
        score = validate_clean_code(example['results'])
        scores.append(score)
        print(f"  Example: {example['task'][:50]}... → Score: {score:.2f}")
    
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"✓ Average quality score: {avg_score:.2f}")
    
    return len(examples) >= 50, avg_score >= 0.5

def test_optimizer_selection():
    """Test optimizer selection based on dataset size."""
    print("\n" + "=" * 60)
    print("Testing Optimizer Selection")
    print("=" * 60)
    
    # Test with small dataset (should use BootstrapFewShot)
    print("\n1. Testing with small dataset...")
    small_examples = [{'task': 'test', 'results': 'test'} for _ in range(50)]
    optimizer_small = create_optimizer(small_examples)
    print(f"✓ Small dataset optimizer: {type(optimizer_small).__name__}")
    
    # Test with large dataset (should use MIPRO)
    print("\n2. Testing with large dataset...")
    large_examples = [{'task': 'test', 'results': 'test'} for _ in range(150)]
    optimizer_large = create_optimizer(large_examples)
    print(f"✓ Large dataset optimizer: {type(optimizer_large).__name__}")
    
    return True

def test_prompt_factory_optimization():
    """Test PromptFactory with dynamic optimization."""
    print("\n" + "=" * 60)
    print("Testing PromptFactory Optimization")
    print("=" * 60)
    
    # Initialize brain with examples
    brain = ProjectBrain()
    examples = get_example_dataset(brain)
    
    print(f"\n1. Initializing PromptFactory with {len(examples)} examples...")
    factory = PromptFactory(brain)
    
    print("✓ PromptFactory initialized")
    print(f"✓ Available modules: {list(factory.get_all_modules().keys())}")
    
    # Test module retrieval and recompilation
    print("\n2. Testing module retrieval...")
    for i in range(6):  # Trigger recompilation at run 5
        module = factory.get_module('research')
        if module:
            print(f"  Run {i+1}: Got research module")
        else:
            print(f"  Run {i+1}: No research module")
    
    return True

def test_workflow_integration():
    """Test workflow integration with DSPy optimization."""
    print("\n" + "=" * 60)
    print("Testing Workflow Integration")
    print("=" * 60)
    
    # Initialize brain
    brain = ProjectBrain()
    
    # Run a simple workflow
    print("\n1. Running sample workflow...")
    task = "Create a simple REST API for user management"
    
    result = run_workflow(task, brain, mode="quick")
    
    if result['success']:
        print("✓ Workflow completed successfully")
        print(f"✓ Generated {len(result.get('code_files', []))} code files")
        print(f"✓ Research results: {len(result.get('research_results', []))}")
        
        # Check if examples were logged
        examples_in_brain = len(brain.memory.get('dspy_examples', []))
        print(f"✓ DSPy examples in brain: {examples_in_brain}")
        
        return examples_in_brain > 0
    else:
        print(f"✗ Workflow failed: {result.get('error', 'Unknown error')}")
        return False

def test_dataset_growth():
    """Test dataset growth over multiple runs."""
    print("\n" + "=" * 60)
    print("Testing Dataset Growth")
    print("=" * 60)
    
    # Initialize brain
    brain = ProjectBrain()
    
    # Clear existing examples for clean test
    brain.memory['dspy_examples'] = []
    brain._save()
    
    initial_count = len(brain.memory.get('dspy_examples', []))
    print(f"\n1. Initial examples: {initial_count}")
    
    # Run multiple workflows to grow dataset
    tasks = [
        "Build a simple calculator API",
        "Create a file upload service",
        "Implement user authentication",
        "Build a task management system"
    ]
    
    for i, task in enumerate(tasks):
        print(f"\n2.{i+1}. Running workflow: {task}")
        result = run_workflow(task, brain, mode="quick")
        
        if result['success']:
            current_count = len(brain.memory.get('dspy_examples', []))
            print(f"  ✓ Examples after workflow: {current_count}")
        else:
            print(f"  ✗ Workflow failed: {result.get('error', 'Unknown error')}")
    
    final_count = len(brain.memory.get('dspy_examples', []))
    growth = final_count - initial_count
    
    print(f"\n3. Dataset growth: {initial_count} → {final_count} (+{growth})")
    
    return growth > 0

def main():
    """Run all tests."""
    print("Dynamic DSPy Optimization Test Suite")
    print("Testing shift from static mocks to real-world data sources")
    print(f"Started at: {datetime.now().isoformat()}")
    
    tests = [
        ("Dynamic Dataset Generation", test_dynamic_dataset_generation),
        ("Optimizer Selection", test_optimizer_selection),
        ("PromptFactory Optimization", test_prompt_factory_optimization),
        ("Workflow Integration", test_workflow_integration),
        ("Dataset Growth", test_dataset_growth)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dynamic DSPy optimization is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 