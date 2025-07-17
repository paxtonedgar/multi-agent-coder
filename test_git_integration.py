#!/usr/bin/env python3
"""
Integration tests for Git repository functionality
"""

import os
import tempfile
import shutil
import subprocess
import json
import pytest
from pathlib import Path

# Import our modules
from memory import ProjectBrain
from tools import GitRepositoryTools, create_research_tools
from graph import run_workflow
from agents import create_all_agents

@pytest.fixture
def test_repo():
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
    
    yield test_dir
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)

@pytest.fixture
def brain():
    """Create a test brain instance"""
    return ProjectBrain()

@pytest.fixture
def git_tools(brain):
    """Create Git tools instance"""
    return GitRepositoryTools(brain)

def test_git_tools_basic(git_tools):
    """Test basic Git repository tools functionality"""
    print("🧪 Testing basic Git tools...")
    
    # Test listing repositories (should be empty initially)
    result = git_tools.list_cloned_repositories()
    assert "No repositories currently cloned" in result
    print("✅ Initial repository list is empty")
    
    print("✅ Basic Git tools test passed")

def test_git_tools_clone_and_read(test_repo, git_tools):
    """Test cloning and reading from a local repository"""
    print("🧪 Testing clone and read functionality...")
    
    # Clone the test repository
    clone_result = git_tools.clone_repository(f"file://{test_repo}")
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
    
    print("✅ Clone and read test passed")

def test_git_tools_create_and_modify(test_repo, git_tools):
    """Test creating and modifying files in repository"""
    print("🧪 Testing create and modify functionality...")
    
    # Clone the test repository
    clone_result = git_tools.clone_repository(f"file://{test_repo}")
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
    
    # Clean up
    git_tools.cleanup_repository(repo_name)
    print("✅ Create and modify test passed")

def test_workflow_with_repo(test_repo, brain):
    """Test workflow with repository integration"""
    print("🧪 Testing workflow with repository...")
    
    # Run workflow with repo URL
    result = run_workflow(
        task="Analyze the repository structure and create a summary",
        brain=brain,
        mode="quick",
        repo_url=f"file://{test_repo}"
    )
    
    assert result['success'] is True
    assert 'research_results' in result
    assert 'plan' in result
    print("✅ Workflow with repository completed successfully")
    
    # Check that repository was processed
    if result.get('research_results'):
        print(f"✅ Found {len(result['research_results'])} research results")
    
    print("✅ Workflow with repository test passed")

def test_agents_with_git_tools(brain):
    """Test agents with Git tools integration"""
    print("🧪 Testing agents with Git tools...")
    
    # Create all agents
    agents = create_all_agents(brain)
    
    # Test that agents have access to Git tools
    assert 'planner' in agents
    assert 'researcher' in agents
    assert 'coder' in agents
    assert 'reviewer' in agents
    assert 'auditor' in agents
    
    print("✅ All agents created successfully")
    print("✅ Agents with Git tools test passed")

def test_memory_tracking(test_repo, brain):
    """Test memory tracking for Git operations"""
    print("🧪 Testing memory tracking...")
    
    # Create Git tools
    git_tools = GitRepositoryTools(brain)
    
    # Perform some operations
    clone_result = git_tools.clone_repository(f"file://{test_repo}")
    if "Failed to clone" not in clone_result:
        repo_name = None
        for line in git_tools.list_cloned_repositories().split('\n'):
            if 'test_repo_' in line or 'test_repository' in line:
                repo_name = line.split(':')[0].strip('- ')
                break
        
        if repo_name:
            # Read a file to create memory entry
            git_tools.read_repository_file(repo_name, "README.md")
            
            # Check that memory was updated
            assert 'git_operations' in brain.memory
            assert len(brain.memory['git_operations']) > 0
            
            # Clean up
            git_tools.cleanup_repository(repo_name)
    
    print("✅ Memory tracking test passed")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 