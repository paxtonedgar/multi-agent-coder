#!/usr/bin/env python3
"""
Test script for LangGraph Multi-Agent Coding System
"""

import os
import sys
from memory import ProjectBrain
from tools import create_research_tools
from agents import create_all_agents

def test_memory():
    """Test ProjectBrain functionality"""
    print("🧠 Testing ProjectBrain...")
    
    brain = ProjectBrain(".")
    
    # Test basic functionality
    print(f"✅ Brain initialized: {brain.brain_file}")
    print(f"✅ Memory loaded: {len(brain.memory)} keys")
    
    # Test embedding creation
    test_text = "def hello_world(): return 'Hello, World!'"
    embedding = brain._create_embedding(test_text)
    print(f"✅ Embedding created: {len(embedding)} dimensions")
    
    # Test file finding
    files = brain.find_relevant_files("test function", top_k=3)
    print(f"✅ File search: {len(files)} results")
    
    return True

def test_tools():
    """Test research tools"""
    print("\n🔧 Testing research tools...")
    
    brain = ProjectBrain(".")
    tools = create_research_tools(brain)
    
    print(f"✅ Tools created: {len(tools)} tools")
    
    # Test tool names
    tool_names = [tool.name for tool in tools]
    expected_tools = ['web_search', 'x_search', 'github_search', 'analyze_github_repo', 
                     'remember_decision', 'find_relevant_files', 'analyze_dependencies', 
                     'run_linter', 'git_commit']
    
    for expected in expected_tools:
        if expected in tool_names:
            print(f"✅ {expected} tool found")
        else:
            print(f"❌ {expected} tool missing")
    
    return True

def test_agents():
    """Test agent creation"""
    print("\n🤖 Testing agent creation...")
    
    brain = ProjectBrain(".")
    
    try:
        agents = create_all_agents(brain)
        print(f"✅ Agents created: {len(agents)} agents")
        
        expected_agents = ['planner', 'coder1', 'coder2', 'reviewer', 'integrator', 'architect', 'coordinator']
        
        for expected in expected_agents:
            if expected in agents:
                print(f"✅ {expected} agent created")
            else:
                print(f"❌ {expected} agent missing")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_api_keys():
    """Test API key configuration"""
    print("\n🔑 Testing API key configuration...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        print("✅ OpenAI API key found")
    else:
        print("❌ OpenAI API key not found")
    
    if anthropic_key:
        print("✅ Anthropic API key found")
    else:
        print("❌ Anthropic API key not found")
    
    if not openai_key and not anthropic_key:
        print("⚠️  No API keys found - some functionality will be limited")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Testing LangGraph Multi-Agent Coding System")
    print("=" * 50)
    
    tests = [
        ("Memory System", test_memory),
        ("Research Tools", test_tools),
        ("Agent Creation", test_agents),
        ("API Keys", test_api_keys),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 