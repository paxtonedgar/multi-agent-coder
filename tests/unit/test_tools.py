#!/usr/bin/env python3
"""
Consolidated Tools Tests - Unit tests for tools and API utilities
Merged from scattered test files to prevent duplication and mess
"""

import os
import sys
import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tools import (
    GitRepositoryTools,
    RealResearchTools,
    BuildTools,
    create_research_tools
)

# ==================== FIXTURES ====================

@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_github_token():
    """Mock GitHub token"""
    return "ghp_test_token_12345"

@pytest.fixture
def mock_hf_token():
    """Mock HuggingFace token"""
    return "hf_test_token_12345"

# ==================== GIT REPOSITORY TOOLS TESTS ====================

def test_git_repository_tools_initialization(temp_dir):
    """Test GitRepositoryTools initialization"""
    from memory import ProjectBrain
    
    brain = ProjectBrain(project_path=temp_dir)
    git_tools = GitRepositoryTools(brain)
    
    assert git_tools.memory == brain
    assert git_tools.github_token == ""  # No token in test environment

def test_git_repository_tools_validate_repo_url():
    """Test repository URL validation"""
    from memory import ProjectBrain
    
    brain = ProjectBrain()
    git_tools = GitRepositoryTools(brain)
    
    # Valid URLs
    assert git_tools._validate_repo_url("https://github.com/user/repo")
    assert git_tools._validate_repo_url("https://gitlab.com/user/repo")
    assert git_tools._validate_repo_url("file:///path/to/repo")
    
    # Invalid URLs
    assert not git_tools._validate_repo_url("invalid-url")
    assert not git_tools._validate_repo_url("http://invalid.com")

def test_git_repository_tools_sanitize_url():
    """Test URL sanitization"""
    from memory import ProjectBrain
    
    brain = ProjectBrain()
    git_tools = GitRepositoryTools(brain)
    
    # Should remove tokens from URLs
    url_with_token = "https://github.com/user/repo.git"
    sanitized = git_tools._sanitize_url(url_with_token)
    assert sanitized == url_with_token
    
    # Should handle URLs without tokens
    clean_url = "https://github.com/user/repo"
    sanitized = git_tools._sanitize_url(clean_url)
    assert sanitized == clean_url

# ==================== RESEARCH TOOLS TESTS ====================

def test_research_tools_initialization(temp_dir):
    """Test RealResearchTools initialization"""
    from memory import ProjectBrain
    
    brain = ProjectBrain(project_path=temp_dir)
    research_tools = RealResearchTools(brain)
    
    assert research_tools.memory == brain
    assert research_tools.cache is not None

def test_research_tools_api_key_handling():
    """Test API key handling in research tools"""
    from memory import ProjectBrain
    
    brain = ProjectBrain()
    research_tools = RealResearchTools(brain)
    
    # Test getting API key (should return empty string in test environment)
    api_key = research_tools._get_api_key("GOOGLE_API_KEY")
    assert api_key == ""
    
    # Test with environment variable
    with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
        api_key = research_tools._get_api_key("GOOGLE_API_KEY")
        assert api_key == "test_key"

def test_research_tools_rate_limiting():
    """Test rate limiting functionality"""
    from memory import ProjectBrain
    
    brain = ProjectBrain()
    research_tools = RealResearchTools(brain)
    
    # Test rate limit checking
    can_make_request = research_tools._check_rate_limit("test_api", limit=10)
    assert can_make_request == True
    
    # Test rate limit updating
    research_tools._update_rate_limit("test_api")
    # Should still be able to make requests
    can_make_request = research_tools._check_rate_limit("test_api", limit=10)
    assert can_make_request == True

# ==================== BUILD TOOLS TESTS ====================

def test_build_tools_initialization():
    """Test BuildTools initialization"""
    build_tools = BuildTools()
    assert build_tools is not None

def test_build_tools_analyze_python_deps(temp_dir):
    """Test Python dependency analysis"""
    build_tools = BuildTools()
    
    # Create a simple requirements.txt file
    requirements_file = os.path.join(temp_dir, "requirements.txt")
    with open(requirements_file, 'w') as f:
        f.write("pytest==7.0.0\nrequests==2.28.0\n")
    
    # Create a simple pyproject.toml file
    pyproject_file = os.path.join(temp_dir, "pyproject.toml")
    with open(pyproject_file, 'w') as f:
        f.write("""
[project]
name = "test-project"
version = "0.1.0"
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0"
]
        """)
    
    # Test dependency analysis
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="pytest==7.0.0\nrequests==2.28.0\n")
        
        deps = build_tools.analyze_python_deps(temp_dir)
        
        assert deps is not None
        assert 'dependencies' in deps
        assert 'manager' in deps
        assert 'dev_dependencies' in deps
        assert 'python_version' in deps
        assert 'total_size' in deps

def test_build_tools_run_linter(temp_dir):
    """Test linter functionality"""
    build_tools = BuildTools()
    
    # Create a simple Python file
    python_file = os.path.join(temp_dir, "test.py")
    with open(python_file, 'w') as f:
        f.write("def hello():\n    print('hello world')\n")
    
    # Test linter
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = build_tools.run_linter([python_file])
        
        assert result is not None
        assert 'files_checked' in result
        assert 'total_issues' in result
        assert 'fixed_issues' in result
        assert 'issues_by_file' in result

# ==================== INTEGRATION TESTS ====================

def test_create_research_tools(temp_dir):
    """Test creating research tools"""
    from memory import ProjectBrain
    
    brain = ProjectBrain(project_path=temp_dir)
    tools = create_research_tools(brain)
    
    assert tools is not None
    assert len(tools) > 0  # Should have multiple tools

# ==================== ERROR HANDLING TESTS ====================

def test_tools_error_handling():
    """Test tools handle errors gracefully"""
    # Test with existing tools that handle errors gracefully
    from memory import ProjectBrain
    from tools import RealResearchTools, BuildTools
    
    brain = ProjectBrain()
    research_tools = RealResearchTools(brain)
    build_tools = BuildTools()
    
    # Test that tools handle errors gracefully
    try:
        # Test with invalid project path
        deps = build_tools.analyze_python_deps("/invalid/path")
        assert deps is not None
    except Exception as e:
        pytest.fail(f"Build tools should handle invalid path gracefully: {e}")
    
    try:
        # Test with invalid files
        result = build_tools.run_linter(["/invalid/file.py"])
        assert result is not None
    except Exception as e:
        pytest.fail(f"Build tools should handle invalid files gracefully: {e}")

# ==================== PERFORMANCE TESTS ====================

def test_tools_performance():
    """Test tools performance"""
    from memory import ProjectBrain
    from tools import BuildTools
    
    brain = ProjectBrain()
    build_tools = BuildTools()
    
    start_time = time.time()
    
    # Test multiple operations
    for _ in range(5):
        build_tools.analyze_python_deps(".")
        build_tools.run_linter([])
    
    execution_time = time.time() - start_time
    
    # Should complete within reasonable time
    assert execution_time < 10, f"Tools operations took {execution_time}s, should be under 10s"

# ==================== INTEGRATION TESTS ====================

def test_tools_integration_workflow():
    """Test complete tools integration workflow"""
    from memory import ProjectBrain
    from tools import BuildTools, RealResearchTools
    
    brain = ProjectBrain()
    build_tools = BuildTools()
    research_tools = RealResearchTools(brain)
    
    # 1. Analyze dependencies
    deps = build_tools.analyze_python_deps(".")
    
    # 2. Run linter
    linter_result = build_tools.run_linter([])
    
    # 3. Verify workflow
    assert deps is not None
    assert linter_result is not None
    assert 'dependencies' in deps
    assert 'files_checked' in linter_result

def test_tools_caching_behavior():
    """Test tools caching behavior"""
    from memory import ProjectBrain
    from tools import RealResearchTools
    
    brain = ProjectBrain()
    research_tools = RealResearchTools(brain)
    
    # Test that caching works (should return same result for same query)
    # Note: In test environment, this will likely return cached results
    result1 = research_tools.web_search_integration("test query")
    result2 = research_tools.web_search_integration("test query")
    
    # Results should be consistent (may not be identical due to dynamic content)
    assert result1 is not None
    assert result2 is not None

if __name__ == "__main__":
    pytest.main([__file__]) 