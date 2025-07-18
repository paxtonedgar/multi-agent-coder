#!/usr/bin/env python3
"""
Quick Debug Test - Test CLI interface without full agent execution
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_cli_interface():
    """Test the CLI interface quickly"""
    print("🔍 Quick CLI Debug Test")
    print("=" * 50)
    
    # Test 1: Check if main.py exists and is executable
    print("1️⃣  Testing main.py existence...")
    if not os.path.exists('main.py'):
        print("❌ main.py not found!")
        return False
    print("✅ main.py found")
    
    # Test 2: Check help output
    print("\n2️⃣  Testing help output...")
    try:
        result = subprocess.run([sys.executable, 'main.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        print(f"✅ Help command successful (return code: {result.returncode})")
        print(f"📝 Help output length: {len(result.stdout)} characters")
        if result.stdout:
            print("📄 Help preview:")
            print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
    except subprocess.TimeoutExpired:
        print("❌ Help command timed out")
        return False
    except Exception as e:
        print(f"❌ Help command failed: {e}")
        return False
    
    # Test 3: Test with a very simple prompt
    print("\n3️⃣  Testing with simple prompt...")
    simple_prompt = "Write a function that returns 'hello world'"
    print(f"🤖 Testing prompt: {simple_prompt}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, 'main.py', simple_prompt], 
                              capture_output=True, text=True, timeout=30)  # 30 second timeout
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
        
        # Analyze the output
        if result.returncode == 0:
            print("✅ Command executed successfully")
            if 'def ' in result.stdout or 'function' in result.stdout.lower():
                print("✅ Output contains function definition")
            else:
                print("⚠️  Output doesn't contain function definition")
        else:
            print("❌ Command failed")
            
    except subprocess.TimeoutExpired:
        print("❌ Simple prompt timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Simple prompt failed: {e}")
        return False
    
    # Test 4: Check what's in main.py
    print("\n4️⃣  Analyzing main.py...")
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        print(f"📄 main.py size: {len(content)} characters")
        print(f"📄 main.py lines: {len(content.splitlines())}")
        
        # Look for key components
        if 'def main()' in content:
            print("✅ Contains main() function")
        else:
            print("❌ No main() function found")
        
        if 'argparse' in content:
            print("✅ Uses argparse for CLI")
        else:
            print("❌ No argparse found")
        
        if 'langchain' in content or 'langgraph' in content:
            print("✅ Contains LangChain/LangGraph imports")
        else:
            print("❌ No LangChain/LangGraph imports found")
        
        # Show first few lines
        print("📄 First 10 lines of main.py:")
        lines = content.splitlines()[:10]
        for i, line in enumerate(lines, 1):
            print(f"   {i:2d}: {line}")
        
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False
    
    print("\n✅ Quick debug test completed!")
    return True

def test_agent_components():
    """Test individual agent components"""
    print("\n🔧 Testing Agent Components")
    print("=" * 50)
    
    # Check if key files exist
    key_files = ['agents.py', 'memory.py', 'tools.py', 'graph.py']
    for file in key_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
    
    # Test importing key modules
    print("\n📦 Testing imports...")
    try:
        import agents
        print("✅ agents.py imports successfully")
    except Exception as e:
        print(f"❌ agents.py import failed: {e}")
    
    try:
        import memory
        print("✅ memory.py imports successfully")
    except Exception as e:
        print(f"❌ memory.py import failed: {e}")
    
    try:
        import tools
        print("✅ tools.py imports successfully")
    except Exception as e:
        print(f"❌ tools.py import failed: {e}")

def main():
    """Main debug function"""
    print("🚀 Quick Debug Test Suite")
    print("=" * 60)
    
    # Test CLI interface
    cli_success = test_cli_interface()
    
    # Test agent components
    test_agent_components()
    
    if cli_success:
        print("\n🎉 Debug test completed successfully!")
        return 0
    else:
        print("\n❌ Debug test found issues!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 