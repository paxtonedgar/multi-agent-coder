#!/usr/bin/env python3
"""
Quick CLI Test - Test CLI with simplified workflow
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def test_quick_mode():
    """Test CLI with quick mode"""
    print("🚀 Quick CLI Test")
    print("=" * 50)
    
    simple_prompt = "Write a function that returns 'hello world'"
    print(f"🤖 Testing prompt: {simple_prompt}")
    
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, 'main.py', simple_prompt, '--mode', 'quick'
        ], capture_output=True, text=True, timeout=30)  # 30 second timeout
        
        execution_time = time.time() - start_time
        
        print(f"⏱️  Execution time: {execution_time:.2f}s")
        print(f"📊 Return code: {result.returncode}")
        print(f"📝 Output length: {len(result.stdout)} characters")
        print(f"⚠️  Error length: {len(result.stderr)} characters")
        
        if result.stdout:
            print("📄 Output preview:")
            print("-" * 50)
            print(result.stdout[:500])
            if len(result.stdout) > 500:
                print("... (truncated)")
            print("-" * 50)
        
        if result.stderr:
            print("❌ Error output:")
            print("-" * 50)
            print(result.stderr[:500])
            if len(result.stderr) > 500:
                print("... (truncated)")
            print("-" * 50)
        
        if result.returncode == 0:
            print("✅ Quick mode test successful!")
            return True
        else:
            print("❌ Quick mode test failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Quick mode timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Quick mode test failed: {e}")
        return False

def test_research_mode():
    """Test CLI with research-only mode"""
    print("\n🔬 Research Mode Test")
    print("=" * 50)
    
    simple_prompt = "Research Python web frameworks"
    print(f"🤖 Testing prompt: {simple_prompt}")
    
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, 'main.py', simple_prompt, '--mode', 'research_only'
        ], capture_output=True, text=True, timeout=30)  # 30 second timeout
        
        execution_time = time.time() - start_time
        
        print(f"⏱️  Execution time: {execution_time:.2f}s")
        print(f"📊 Return code: {result.returncode}")
        print(f"📝 Output length: {len(result.stdout)} characters")
        
        if result.stdout:
            print("📄 Output preview:")
            print("-" * 50)
            print(result.stdout[:500])
            if len(result.stdout) > 500:
                print("... (truncated)")
            print("-" * 50)
        
        if result.returncode == 0:
            print("✅ Research mode test successful!")
            return True
        else:
            print("❌ Research mode test failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Research mode timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Research mode test failed: {e}")
        return False

def test_direct_workflow():
    """Test workflow directly without CLI"""
    print("\n🔄 Direct Workflow Test")
    print("=" * 50)
    
    try:
        from memory import ProjectBrain
        from graph import run_workflow
        
        print("✅ Imports successful")
        
        # Initialize brain
        brain = ProjectBrain()
        print("✅ Brain initialized")
        
        # Test quick workflow
        task = "Write a function that returns 'hello world'"
        print(f"📋 Task: {task}")
        
        start_time = time.time()
        result = run_workflow(task, brain, mode="quick")
        execution_time = time.time() - start_time
        
        print(f"⏱️  Execution time: {execution_time:.2f}s")
        print(f"📊 Success: {result.get('success', False)}")
        
        if result.get('success'):
            print("✅ Direct workflow test successful!")
            print(f"📝 Messages: {len(result.get('messages', []))}")
            print(f"🔬 Research: {len(result.get('research_results', []))}")
            print(f"📋 Plan: {len(str(result.get('plan', {})))} chars")
            return True
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ Direct workflow failed: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Direct workflow test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Quick CLI Test Suite")
    print("=" * 60)
    
    # Test quick mode
    quick_success = test_quick_mode()
    
    # Test research mode
    research_success = test_research_mode()
    
    # Test direct workflow
    workflow_success = test_direct_workflow()
    
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Quick Mode: {'✅ PASS' if quick_success else '❌ FAIL'}")
    print(f"Research Mode: {'✅ PASS' if research_success else '❌ FAIL'}")
    print(f"Direct Workflow: {'✅ PASS' if workflow_success else '❌ FAIL'}")
    
    if quick_success or research_success or workflow_success:
        print("\n🎉 At least one test passed!")
        return 0
    else:
        print("\n❌ All tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 