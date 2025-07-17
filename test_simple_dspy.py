#!/usr/bin/env python3
"""
Simple test for DSPy optimization core functionality
"""

import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import ProjectBrain
from prompts import get_example_dataset, create_optimizer, PromptFactory, validate_clean_code

def test_basic_functionality():
    """Test basic DSPy functionality."""
    print("=" * 60)
    print("Testing Basic DSPy Functionality")
    print("=" * 60)
    
    # Initialize brain
    brain = ProjectBrain()
    
    # Test dataset generation
    print("\n1. Testing dataset generation...")
    examples = get_example_dataset(brain)
    print(f"✓ Generated {len(examples)} examples")
    
    # Check format
    if examples:
        first_example = examples[0]
        print(f"✓ Example format: {list(first_example.keys())}")
        if 'inputs' in first_example and 'outputs' in first_example:
            print("✓ Correct DSPy format")
        else:
            print("✗ Incorrect format")
            return False
    
    # Test optimizer creation
    print("\n2. Testing optimizer creation...")
    optimizer = create_optimizer(examples)
    if optimizer:
        print(f"✓ Optimizer created: {type(optimizer).__name__}")
    else:
        print("✗ Failed to create optimizer")
        return False
    
    # Test quality scoring
    print("\n3. Testing quality scoring...")
    if examples:
        score = validate_clean_code(examples[0])
        print(f"✓ Quality score: {score:.2f}")
    
    # Test PromptFactory
    print("\n4. Testing PromptFactory...")
    factory = PromptFactory(brain)
    modules = factory.get_all_modules()
    print(f"✓ Created {len(modules)} modules")
    print(f"✓ Module names: {list(modules.keys())}")
    
    return True

def test_example_caching():
    """Test example caching in brain."""
    print("\n" + "=" * 60)
    print("Testing Example Caching")
    print("=" * 60)
    
    # Initialize brain
    brain = ProjectBrain()
    
    # Clear existing examples
    brain.memory['dspy_examples'] = []
    brain._save()
    
    print(f"\n1. Initial examples: {len(brain.memory.get('dspy_examples', []))}")
    
    # Generate examples
    examples = get_example_dataset(brain)
    cached_examples = brain.memory.get('dspy_examples', [])
    
    print(f"2. Generated examples: {len(examples)}")
    print(f"3. Cached examples: {len(cached_examples)}")
    
    # Test caching
    if len(cached_examples) > 0:
        print("✓ Examples successfully cached")
        return True
    else:
        print("✗ Examples not cached")
        return False

def main():
    """Run simple tests."""
    print("Simple DSPy Optimization Test")
    print(f"Started at: {datetime.now().isoformat()}")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Example Caching", test_example_caching)
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
        print("🎉 All tests passed! Core DSPy functionality is working.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 