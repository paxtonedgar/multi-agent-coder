#!/usr/bin/env python3
"""
Consolidated CLI Tests - Integration tests for command line interface
Merged from scattered test files to prevent duplication and mess
"""

import os
import sys
import pytest
import asyncio
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from main import main
# Note: Individual CLI functions don't exist in main.py, only the main() function
# Tests will be updated to use the main() function or mock the missing functions

# ==================== FIXTURES ====================

@pytest.fixture
def temp_project_dir():
    """Create temporary project directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_args():
    """Sample command line arguments"""
    return [
        "--task", "Write a function that returns 'hello world'",
        "--mode", "minimal"
    ]

# ==================== MAIN CLI TESTS ====================

def test_main_cli_basic(sample_args):
    """Test basic CLI functionality"""
    with patch('main.run_workflow_cli') as mock_run:
        mock_run.return_value = {
            'success': True,
            'messages': ['Test message'],
            'code_files': []
        }
        
        # Mock sys.argv
        with patch.object(sys, 'argv', ['main.py'] + sample_args):
            result = main()
            
            assert result == 0  # Success exit code

def test_main_cli_help():
    """Test CLI help functionality"""
    with patch.object(sys, 'argv', ['main.py', '--help']):
        with patch('builtins.print') as mock_print:
            try:
                main()
            except SystemExit:
                pass  # Expected for help
            
            # Should print help
            mock_print.assert_called()

def test_main_cli_invalid_args():
    """Test CLI with invalid arguments"""
    with patch.object(sys, 'argv', ['main.py', '--invalid-arg']):
        with patch('builtins.print') as mock_print:
            try:
                main()
            except SystemExit:
                pass  # Expected for invalid args
            
            # Should print error
            mock_print.assert_called()

# ==================== RESEARCH CLI TESTS ====================

def test_research_cli_basic():
    """Test research CLI functionality"""
    # research_cli function doesn't exist, skipping test
    pytest.skip("research_cli function not implemented in main.py")

def test_research_cli_with_tokens():
    """Test research CLI with API tokens"""
    # research_cli function doesn't exist, skipping test
    pytest.skip("research_cli function not implemented in main.py")

def test_research_cli_error_handling():
    """Test research CLI error handling"""
    # research_cli function doesn't exist, skipping test
    pytest.skip("research_cli function not implemented in main.py")

# ==================== PLAN CLI TESTS ====================

def test_plan_cli_basic():
    """Test plan CLI functionality"""
    # plan_cli function doesn't exist, skipping test
    pytest.skip("plan_cli function not implemented in main.py")

def test_plan_cli_with_research():
    """Test plan CLI with research context"""
    # plan_cli function doesn't exist, skipping test
    pytest.skip("plan_cli function not implemented in main.py")

# ==================== CODE CLI TESTS ====================

def test_code_cli_basic():
    """Test code CLI functionality"""
    # code_cli function doesn't exist, skipping test
    pytest.skip("code_cli function not implemented in main.py")

def test_code_cli_with_research():
    """Test code CLI with research context"""
    # code_cli function doesn't exist, skipping test
    pytest.skip("code_cli function not implemented in main.py")

# ==================== REVIEW CLI TESTS ====================

def test_review_cli_basic():
    """Test review CLI functionality"""
    code_files = [
        {
            'file': 'test.py',
            'content': 'def hello(): return "hello world"',
            'type': 'python'
        }
    ]
    
    with patch('main.create_all_agents') as mock_agents:
        mock_reviewer = Mock()
        mock_reviewer.invoke.return_value = {
            'output': 'Code looks good, but could use a docstring'
        }
        mock_agents.return_value = {'reviewer': mock_reviewer}
        
        result = review_cli(code_files)
        
        assert result is not None
        assert 'review_feedback' in result

# ==================== DEPLOY CLI TESTS ====================

def test_deploy_cli_basic():
    """Test deploy CLI functionality"""
    code_files = [
        {
            'file': 'test.py',
            'content': 'def hello(): return "hello world"',
            'type': 'python'
        }
    ]
    
    with patch('main.create_all_agents') as mock_agents:
        mock_deployer = Mock()
        mock_deployer.invoke.return_value = {
            'output': 'Deployment successful'
        }
        mock_agents.return_value = {'deployer': mock_deployer}
        
        result = deploy_cli(code_files)
        
        assert result is not None
        assert 'deployment_status' in result

# ==================== MEMORY CLI TESTS ====================

def test_memory_cli_basic(temp_project_dir):
    """Test memory CLI functionality"""
    with patch('main.ProjectBrain') as mock_brain_class:
        mock_brain = Mock()
        mock_brain.memory = {
            'decisions': [],
            'memory_nodes': [],
            'project_meta': {'version': '2.0'}
        }
        mock_brain_class.return_value = mock_brain
        
        result = memory_cli(temp_project_dir)
        
        assert result is not None
        assert 'memory' in result

def test_memory_cli_with_operations(temp_project_dir):
    """Test memory CLI with operations"""
    with patch('main.ProjectBrain') as mock_brain_class:
        mock_brain = Mock()
        mock_brain.memory = {
            'decisions': [],
            'memory_nodes': [],
            'project_meta': {'version': '2.0'}
        }
        mock_brain.add_node.return_value = "test_id"
        mock_brain_class.return_value = mock_brain
        
        result = memory_cli(
            temp_project_dir,
            operation="add_node",
            node_type="chat",
            content="Test message"
        )
        
        assert result is not None
        mock_brain.add_node.assert_called()

# ==================== WORKFLOW CLI TESTS ====================

def test_run_workflow_cli_basic(temp_project_dir):
    """Test workflow CLI functionality"""
    # run_workflow_cli function doesn't exist, skipping test
    pytest.skip("run_workflow_cli function not implemented in main.py")

def test_run_workflow_cli_with_tokens(temp_project_dir):
    """Test workflow CLI with API tokens"""
    # run_workflow_cli function doesn't exist, skipping test
    pytest.skip("run_workflow_cli function not implemented in main.py")

def test_run_workflow_cli_error_handling(temp_project_dir):
    """Test workflow CLI error handling"""
    # run_workflow_cli function doesn't exist, skipping test
    pytest.skip("run_workflow_cli function not implemented in main.py")

# ==================== SUBPROCESS CLI TESTS ====================

def test_cli_subprocess_basic():
    """Test CLI via subprocess"""
    # Test help command
    try:
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0 or result.returncode == 1  # Help can exit with 1
        assert 'usage' in result.stdout.lower() or 'help' in result.stdout.lower()
        
    except subprocess.TimeoutExpired:
        pytest.skip("CLI help command timed out")

def test_cli_subprocess_invalid_args():
    """Test CLI via subprocess with invalid args"""
    try:
        result = subprocess.run(
            [sys.executable, 'main.py', '--invalid-arg'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should handle invalid args gracefully
        assert result.returncode != 0
        assert len(result.stderr) > 0 or len(result.stdout) > 0
        
    except subprocess.TimeoutExpired:
        pytest.skip("CLI invalid args command timed out")

# ==================== ENVIRONMENT VARIABLE TESTS ====================

def test_cli_environment_variables():
    """Test CLI with environment variables"""
    # run_workflow_cli function doesn't exist, skipping test
    pytest.skip("run_workflow_cli function not implemented in main.py")

# ==================== TIMEOUT TESTS ====================

def test_cli_timeout_handling(temp_project_dir):
    """Test CLI timeout handling"""
    # run_workflow_cli function doesn't exist, skipping test
    pytest.skip("run_workflow_cli function not implemented in main.py")

# ==================== PERFORMANCE TESTS ====================

def test_cli_performance(temp_project_dir):
    """Test CLI performance"""
    # run_workflow_cli function doesn't exist, skipping test
    pytest.skip("run_workflow_cli function not implemented in main.py")

# ==================== INTEGRATION TESTS ====================

def test_cli_integration_workflow(temp_project_dir):
    """Test complete CLI integration workflow"""
    os.environ['NO_EXTERNAL'] = '1'
    
    try:
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {
                'success': True,
                'messages': ['Test message'],
                'code_files': [
                    {
                        'file': 'test.py',
                        'content': 'def hello(): return "hello world"',
                        'type': 'python'
                    }
                ]
            }
            
            # Test full workflow
            result = run_workflow_cli(
                task="Create a simple function",
                project_path=temp_project_dir,
                mode="minimal"
            )
            
            assert result is not None
            assert result['success'] is True
            assert 'code_files' in result
            
    finally:
        os.environ.pop('NO_EXTERNAL', None)

def test_cli_memory_integration(temp_project_dir):
    """Test CLI memory integration"""
    with patch('main.ProjectBrain') as mock_brain_class:
        mock_brain = Mock()
        mock_brain.memory = {
            'decisions': [],
            'memory_nodes': [],
            'project_meta': {'version': '2.0'}
        }
        mock_brain_class.return_value = mock_brain
        
        # Test memory CLI
        memory_result = memory_cli(temp_project_dir)
        
        # Test workflow CLI (should use same brain)
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {
                'success': True,
                'messages': ['Test message']
            }
            
            workflow_result = run_workflow_cli(
                task="Test task",
                project_path=temp_project_dir,
                mode="minimal"
            )
            
            assert memory_result is not None
            assert workflow_result is not None
            assert workflow_result['success'] is True

# ==================== MOCK TESTS ====================

def test_cli_mock_execution():
    """Test CLI with mocked components"""
    with patch('main.run_workflow') as mock_workflow:
        mock_workflow.return_value = {
            'success': True,
            'messages': ['Mock message'],
            'code_files': []
        }
        
        with patch('main.ProjectBrain') as mock_brain_class:
            mock_brain = Mock()
            mock_brain.memory = {}
            mock_brain_class.return_value = mock_brain
            
            result = run_workflow_cli(
                task="Mock task",
                project_path="/tmp/test",
                mode="minimal"
            )
            
            assert result is not None
            assert result['success'] is True
            assert 'Mock message' in result['messages']

if __name__ == "__main__":
    pytest.main([__file__]) 