#!/usr/bin/env python3
"""
Minimal Agent Test - Test agents directly without complex workflow
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def test_minimal_agent():
    """Test a minimal agent setup"""
    print("🧪 Minimal Agent Test")
    print("=" * 50)
    
    try:
        # Import required modules
        from memory import ProjectBrain
        from agents import create_all_agents
        
        print("✅ Imports successful")
        
        # Initialize brain
        print("🧠 Initializing brain...")
        brain = ProjectBrain()
        print("✅ Brain initialized")
        
        # Create agents
        print("🤖 Creating agents...")
        agents = create_all_agents(brain)
        print(f"✅ Created {len(agents)} agents: {list(agents.keys())}")
        
        # Test with a simple prompt
        simple_prompt = "Write a function that returns 'hello world'"
        print(f"\n🔍 Testing with prompt: {simple_prompt}")
        
        # Try the coder agent
        if 'coder1' in agents:
            print("👨‍💻 Testing coder1 agent...")
            start_time = time.time()
            
            try:
                result = agents['coder1'].invoke({
                    "input": simple_prompt,
                    "chat_history": []
                })
                
                execution_time = time.time() - start_time
                print(f"⏱️  Execution time: {execution_time:.2f}s")
                
                if result and 'output' in result:
                    output = result['output']
                    print(f"📝 Output length: {len(output)} characters")
                    print(f"📄 Output preview: {output[:200]}...")
                    print("✅ Coder agent test successful!")
                    return True
                else:
                    print("❌ No output from coder agent")
                    return False
                    
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"❌ Coder agent failed after {execution_time:.2f}s: {e}")
                return False
        
        else:
            print("❌ Coder agent not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_simple_workflow():
    """Test a simple workflow without complex dependencies"""
    print("\n🔄 Simple Workflow Test")
    print("=" * 50)
    
    try:
        from memory import ProjectBrain
        from agents import create_all_agents
        
        brain = ProjectBrain()
        agents = create_all_agents(brain)
        
        # Simple workflow: research -> plan -> code
        task = "Write a function that adds two numbers"
        print(f"📋 Task: {task}")
        
        # Step 1: Research
        print("\n1️⃣  Research step...")
        if 'architect' in agents:
            research_result = agents['architect'].invoke({
                "input": f"Research best practices for: {task}",
                "chat_history": []
            })
            print(f"✅ Research completed: {len(research_result.get('output', ''))} chars")
        else:
            print("⚠️  Architect not available, skipping research")
        
        # Step 2: Planning
        print("\n2️⃣  Planning step...")
        if 'planner' in agents:
            plan_result = agents['planner'].invoke({
                "input": f"Create a plan for: {task}",
                "chat_history": []
            })
            print(f"✅ Planning completed: {len(plan_result.get('output', ''))} chars")
        else:
            print("⚠️  Planner not available, skipping planning")
        
        # Step 3: Coding
        print("\n3️⃣  Coding step...")
        if 'coder1' in agents:
            code_result = agents['coder1'].invoke({
                "input": task,
                "chat_history": []
            })
            print(f"✅ Coding completed: {len(code_result.get('output', ''))} chars")
            print(f"📄 Code preview: {code_result.get('output', '')[:200]}...")
        else:
            print("❌ Coder not available")
            return False
        
        print("✅ Simple workflow test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Simple workflow test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Minimal Agent Test Suite")
    print("=" * 60)
    
    # Test minimal agent
    agent_success = test_minimal_agent()
    
    # Test simple workflow
    workflow_success = test_simple_workflow()
    
    if agent_success and workflow_success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 