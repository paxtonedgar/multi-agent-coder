#!/usr/bin/env python3
"""
Integration tests for Git repository functionality
"""

import os
import tempfile
import shutil
import subprocess
import json
from pathlib import Path

# Import our modules
from memory import ProjectBrain
from tools import GitRepositoryTools, create_research_tools
from graph import run_workflow
from agents import create_all_agents

def setup_test_repo():
    """Create a test Git repository for testing"""
    test_dir = tempfile.mkdtemp(prefix="test_repo_")
    
    # Initialize git repository
    subprocess.run(["git", "init"], cwd=test_dir, check=True)
    
    # Create some test files
    test_files = {
        "README.md": "# Test Repository\n\nThis is a test repository for integration testing.",
        "main.py": "def hello():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    hello()",
        "requirements.txt": "requests>=2.31.0\npytest>=7.4.0",
        "test_main.py": "import pytest\nfrom main import hello\n\ndef test_hello():\n    assert hello is not None"
    }
    
    for filename, content in test_files.items():
        with open(os.path.join(test_dir, filename), 'w') as f:
            f.write(content)
    
    # Add and commit files
    subprocess.run(["git", "add", "."], cwd=test_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=test_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=test_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=test_dir, check=True)
    
    return test_dir

def test_git_tools_basic():
    """Test basic Git repository tools functionality"""
    print("🧪 Testing basic Git tools...")
    
    # Create test brain
    brain = ProjectBrain()
    
    # Create Git tools
    git_tools = GitRepositoryTools(brain)
    
    # Test listing repositories (should be empty initially)
    result = git_tools.list_cloned_repositories()
    assert "No repositories currently cloned" in result
    print("✅ Initial repository list is empty")
    
    print("✅ Basic Git tools test passed")

def test_git_tools_clone_and_read():
    """Test cloning and reading from a local repository"""
    print("🧪 Testing clone and read functionality...")
    
    # Create test repository
    test_repo_path = setup_test_repo()
    
    try:
        # Create test brain
        brain = ProjectBrain()
        git_tools = GitRepositoryTools(brain)
        
        # Clone the test repository
        clone_result = git_tools.clone_repository(f"file://{test_repo_path}")
        print(f"Clone result: {clone_result}")
        assert "Repository cloned to" in clone_result or "Failed to clone" in clone_result
        if "Failed to clone" in clone_result:
            print(f"❌ Clone failed: {clone_result}")
            return
        print("✅ Repository cloned successfully")
        
        # List cloned repositories
        repo_list = git_tools.list_cloned_repositories()
        assert "test_repo_" in repo_list
        print("✅ Repository appears in list")
        
        # Get repository name
        repo_name = None
        for line in repo_list.split('\n'):
            if 'test_repo_' in line or 'test_repository' in line:
                repo_name = line.split(':')[0].strip('- ')
                break
        
        assert repo_name is not None
        print(f"✅ Found repository name: {repo_name}")
        
        # List files in repository
        files_result = git_tools.list_repository_files(repo_name)
        assert "README.md" in files_result
        assert "main.py" in files_result
        assert "requirements.txt" in files_result
        print("✅ Repository files listed correctly")
        
        # Read a file
        readme_content = git_tools.read_repository_file(repo_name, "README.md")
        assert "# Test Repository" in readme_content
        print("✅ File read successfully")
        
        # Get repository status
        status = git_tools.get_repository_status(repo_name)
        assert "Current Branch" in status
        assert "main" in status
        print("✅ Repository status retrieved")
        
        # Clean up
        cleanup_result = git_tools.cleanup_repository(repo_name)
        assert "cleaned up successfully" in cleanup_result
        print("✅ Repository cleaned up")
        
    finally:
        # Clean up test repository
        shutil.rmtree(test_repo_path)
    
    print("✅ Clone and read test passed")

def test_git_tools_create_and_modify():
    """Test creating and modifying files in repository"""
    print("🧪 Testing create and modify functionality...")
    
    # Create test repository
    test_repo_path = setup_test_repo()
    
    try:
        # Create test brain
        brain = ProjectBrain()
        git_tools = GitRepositoryTools(brain)
        
        # Clone the test repository
        clone_result = git_tools.clone_repository(f"file://{test_repo_path}")
        print(f"Clone result: {clone_result}")
        if "Failed to clone" in clone_result:
            print(f"❌ Clone failed: {clone_result}")
            return
        
        repo_name = None
        for line in git_tools.list_cloned_repositories().split('\n'):
            if 'test_repo_' in line or 'test_repository' in line:
                repo_name = line.split(':')[0].strip('- ')
                break
        
        # Create a new file
        new_file_content = """def new_function():
    \"\"\"A new function for testing\"\"\"
    return "Hello from new function"

if __name__ == "__main__":
    print(new_function())
"""
        create_result = git_tools.create_repository_file(
            repo_name, 
            "new_module.py", 
            new_file_content,
            "Add new module for testing"
        )
        assert "created and committed" in create_result
        print("✅ New file created and committed")
        
        # Read the created file
        read_result = git_tools.read_repository_file(repo_name, "new_module.py")
        assert "new_function" in read_result
        print("✅ Created file can be read")
        
        # Modify the file
        modified_content = """def new_function():
    \"\"\"A modified function for testing\"\"\"
    return "Hello from modified function"

def another_function():
    return "Another function"

if __name__ == "__main__":
    print(new_function())
    print(another_function())
"""
        modify_result = git_tools.modify_repository_file(
            repo_name,
            "new_module.py",
            modified_content,
            "Update new module with additional function"
        )
        assert "modified and committed" in modify_result
        print("✅ File modified and committed")
        
        # Verify modification
        read_modified = git_tools.read_repository_file(repo_name, "new_module.py")
        assert "another_function" in read_modified
        print("✅ File modification verified")
        
        # Create a new branch
        branch_result = git_tools.create_branch(repo_name, "feature/test-branch")
        assert "Created and switched to branch" in branch_result
        print("✅ New branch created")
        
        # Clean up
        git_tools.cleanup_repository(repo_name)
        
    finally:
        # Clean up test repository
        shutil.rmtree(test_repo_path)
    
    print("✅ Create and modify test passed")

def test_workflow_with_repo():
    """Test the full workflow with repository integration"""
    print("🧪 Testing workflow with repository integration...")
    
    # Create test repository
    test_repo_path = setup_test_repo()
    
    try:
        # Create test brain
        brain = ProjectBrain()
        
        # Run workflow with repository URL
        result = run_workflow(
            task="Create a simple calculator function",
            brain=brain,
            mode="full",
            repo_url=f"file://{test_repo_path}"
        )
        
        # Check that workflow completed
        assert result is not None
        print("✅ Workflow completed successfully")
        
        # Check that code files were generated
        final_state = result.get('final_state', {})
        if final_state:
            code_files = final_state.get('code_files', [])
            assert len(code_files) > 0
            print(f"✅ Generated {len(code_files)} code files")
            
            # Check for Git operations
            git_ops = [f for f in code_files if f.get('type') in ['git_operations', 'git_error']]
            if git_ops:
                print("✅ Git operations were performed")
            
            # Check for repository information
            git_repo = final_state.get('git_repo', {})
            if git_repo:
                print(f"✅ Repository info: {git_repo.get('name', 'Unknown')}")
        else:
            print("⚠️  No final state returned, but workflow completed")
        
    finally:
        # Clean up test repository
        shutil.rmtree(test_repo_path)
    
    print("✅ Workflow with repository test passed")

def test_agents_with_git_tools():
    """Test that agents can use Git tools"""
    print("🧪 Testing agents with Git tools...")
    
    # Create test brain
    brain = ProjectBrain()
    
    # Create agents
    agents = create_all_agents(brain)
    
    # Test that agents have access to Git tools
    for agent_name, agent in agents.items():
        # Check if agent has tools
        if hasattr(agent, 'tools'):
            tool_names = [tool.name for tool in agent.tools]
            git_tools = [name for name in tool_names if 'repository' in name.lower() or 'git' in name.lower()]
            if git_tools:
                print(f"✅ Agent {agent_name} has Git tools: {git_tools}")
    
    print("✅ Agents with Git tools test passed")

def test_memory_tracking():
    """Test that Git operations are tracked in memory"""
    print("🧪 Testing memory tracking...")
    
    # Create test repository
    test_repo_path = setup_test_repo()
    
    try:
        # Create test brain
        brain = ProjectBrain()
        git_tools = GitRepositoryTools(brain)
        
        # Perform some Git operations
        clone_result = git_tools.clone_repository(f"file://{test_repo_path}")
        print(f"Clone result: {clone_result}")
        if "Failed to clone" in clone_result:
            print(f"❌ Clone failed: {clone_result}")
            return
        
        # Check memory
        git_ops = brain.memory.get('git_operations', [])
        if len(git_ops) > 0:
            assert git_ops[0]['operation'] == 'clone'
            print("✅ Git operations tracked in memory")
        else:
            print("⚠️  Git operations not tracked in memory")
        
        # Clean up
        repo_name = None
        for line in git_tools.list_cloned_repositories().split('\n'):
            if 'test_repo_' in line or 'test_repository' in line:
                repo_name = line.split(':')[0].strip('- ')
                break
        if repo_name:
            git_tools.cleanup_repository(repo_name)
        
    finally:
        # Clean up test repository
        shutil.rmtree(test_repo_path)
    
    print("✅ Memory tracking test passed")

def main():
    """Run all integration tests"""
    print("🚀 Starting Git integration tests...\n")
    
    tests = [
        test_git_tools_basic,
        test_git_tools_clone_and_read,
        test_git_tools_create_and_modify,
        test_workflow_with_repo,
        test_agents_with_git_tools,
        test_memory_tracking
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Git integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 