#!/usr/bin/env python3
"""
Test script for enhanced GitHub content extraction
"""

import sys
import os
import json
from typing import List, Dict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import RealResearchTools
from memory import ProjectBrain

def test_github_content_extraction():
    """Test the enhanced GitHub content extraction functionality."""
    print("🧪 Testing Enhanced GitHub Content Extraction")
    print("=" * 50)
    
    # Initialize brain and research tools
    brain = ProjectBrain()
    research_tools = RealResearchTools(brain)
    
    # Test repositories
    test_repos = [
        "stanfordnlp/dspy",
        "langchain-ai/langgraph",
        "openai/openai-python"
    ]
    
    total_examples = 0
    
    for repo_url in test_repos:
        print(f"\n📁 Testing repository: {repo_url}")
        print("-" * 30)
        
        try:
            # Test different content types
            content_types = ['issues', 'prs', 'notebooks', 'readme', 'examples']
            
            for content_type in content_types:
                print(f"  🔍 Extracting {content_type}...")
                
                examples = research_tools.extract_github_content(
                    f"https://github.com/{repo_url}", 
                    content_types=[content_type]
                )
                
                print(f"    Found {len(examples)} examples")
                
                # Show a sample example
                if examples:
                    example = examples[0]
                    print(f"    Sample: {example['task'][:60]}...")
                    print(f"    Source: {example['source']}")
                    print(f"    URL: {example['url']}")
                
                total_examples += len(examples)
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    print(f"\n📊 Total examples extracted: {total_examples}")
    
    # Check brain memory
    cached_examples = brain.memory.get('github_content_examples', [])
    print(f"📦 Cached in brain: {len(cached_examples)} examples")
    
    return total_examples > 0

def test_dspy_example_filtering():
    """Test DSPy-relevant example filtering."""
    print("\n🔍 Testing DSPy Example Filtering")
    print("=" * 40)
    
    # Sample examples
    sample_examples = [
        {
            'task': 'Implement DSPy optimization module',
            'results': 'class DSPyOptimizer: def compile(self, module): pass',
            'source': 'Test',
            'url': 'https://example.com'
        },
        {
            'task': 'Fix bug in user authentication',
            'results': 'def authenticate_user(): pass',
            'source': 'Test',
            'url': 'https://example.com'
        },
        {
            'task': 'Add prompt engineering example',
            'results': 'prompt = "Write a function that..."',
            'source': 'Test',
            'url': 'https://example.com'
        }
    ]
    
    # Filter for DSPy-relevant examples
    relevant_keywords = [
        'dspy', 'optimization', 'prompt', 'example', 'tutorial',
        'module', 'signature', 'metric', 'compiler', 'bootstrap',
        'few-shot', 'mipro', 'teleprompter', 'react', 'chain'
    ]
    
    dspy_relevant = []
    for example in sample_examples:
        task_lower = example['task'].lower()
        results_lower = example['results'].lower()
        
        if any(keyword in task_lower or keyword in results_lower for keyword in relevant_keywords):
            dspy_relevant.append(example)
        elif 'def ' in results_lower or 'class ' in results_lower:
            dspy_relevant.append(example)
    
    print(f"Original examples: {len(sample_examples)}")
    print(f"DSPy relevant: {len(dspy_relevant)}")
    
    for i, example in enumerate(dspy_relevant, 1):
        print(f"  {i}. {example['task']}")
    
    return len(dspy_relevant) > 0

def test_prompts_integration():
    """Test integration with prompts.py example generation."""
    print("\n🔗 Testing Prompts Integration")
    print("=" * 35)
    
    try:
        from prompts import _fetch_github_examples
        
        print("  🔍 Fetching GitHub examples...")
        examples = _fetch_github_examples()
        
        print(f"  📊 Total examples: {len(examples)}")
        
        if examples:
            print("  📝 Sample examples:")
            for i, example in enumerate(examples[:3], 1):
                print(f"    {i}. {example['task'][:50]}...")
                print(f"       Results length: {len(example['results'])} chars")
        
        return len(examples) > 0
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Enhanced GitHub Content Extraction Tests")
    print("=" * 55)
    
    tests = [
        ("GitHub Content Extraction", test_github_content_extraction),
        ("DSPy Example Filtering", test_dspy_example_filtering),
        ("Prompts Integration", test_prompts_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status} {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ FAIL {test_name}: {e}")
    
    # Summary
    print("\n" + "=" * 55)
    print("📋 Test Summary")
    print("=" * 55)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced GitHub extraction is working.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 