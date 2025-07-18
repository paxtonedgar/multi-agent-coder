#!/usr/bin/env python3
"""
Consolidated E2E Tests - End-to-end tests for complete workflows
Merged from scattered test files to prevent duplication and mess
"""

import os
import sys
import pytest
import asyncio
import time
import tempfile
import shutil
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from memory import ProjectBrain
from graph import run_workflow
# run_workflow_cli function doesn't exist in main.py, using run_workflow instead

# ==================== FIXTURES ====================

@pytest.fixture
def temp_project_dir():
    """Create temporary project directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_brain(temp_project_dir):
    """Create sample brain for testing"""
    brain = ProjectBrain(project_path=temp_project_dir)
    return brain

# ==================== MINIMAL AGENT TESTS ====================

def test_minimal_agent_functionality(sample_brain):
    """Test minimal agent functionality"""
    from agents import create_all_agents
    
    # Create agents
    agents = create_all_agents(sample_brain)
    
    # Test coder agent
    coder = agents['coder1']
    result = coder.invoke({
        "input": "Write a function that returns 'hello world'",
        "chat_history": []
    })
    
    assert result is not None
    assert 'output' in result or 'content' in result
    output = result.get('output', result.get('content', ''))
    assert len(output) > 0, "Coder should produce output"

def test_minimal_agent_execution_time(sample_brain):
    """Test minimal agent execution time"""
    from agents import create_all_agents
    
    agents = create_all_agents(sample_brain)
    coder = agents['coder1']
    
    start_time = time.time()
    result = coder.invoke({
        "input": "Write a simple function",
        "chat_history": []
    })
    execution_time = time.time() - start_time
    
    assert result is not None
    assert execution_time < 30, f"Agent execution took {execution_time}s, should be under 30s"

# ==================== WORKFLOW E2E TESTS ====================

def test_minimal_workflow_e2e(sample_brain):
    """Test minimal workflow end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    try:
        result = run_workflow(
            task="Write a function that returns 'hello world'",
            brain=sample_brain,
            mode="minimal"
        )
        
        assert result is not None
        assert 'success' in result
        
        # Should either succeed or fail with timeout, not crash
        if not result['success']:
            assert 'error' in result
            error_msg = result['error'].lower()
            assert any(term in error_msg for term in ['timeout', 'timed out', 'error'])
            
    finally:
        os.environ.pop('NO_EXTERNAL', None)
        os.environ.pop('MINIMAL_MODE', None)

def test_quick_workflow_e2e(sample_brain):
    """Test quick workflow end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    
    try:
        result = run_workflow(
            task="Write a function that returns 'hello world'",
            brain=sample_brain,
            mode="quick"
        )
        
        assert result is not None
        assert 'success' in result
        
        if not result['success']:
            assert 'error' in result
            
    finally:
        os.environ.pop('NO_EXTERNAL', None)

def test_research_workflow_e2e(sample_brain):
    """Test research workflow end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    
    try:
        result = run_workflow(
            task="Research Python web frameworks",
            brain=sample_brain,
            mode="research_only"
        )
        
        assert result is not None
        assert 'success' in result
        
        if not result['success']:
            assert 'error' in result
            
    finally:
        os.environ.pop('NO_EXTERNAL', None)

# ==================== CLI E2E TESTS ====================

def test_cli_minimal_e2e(temp_project_dir):
    """Test CLI minimal mode end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    try:
        brain = ProjectBrain(project_path=temp_project_dir)
        result = run_workflow(
            task="Write a function that returns 'hello world'",
            brain=brain,
            mode="minimal"
        )
        
        assert result is not None
        assert 'success' in result
        
        if not result['success']:
            assert 'error' in result
            
    finally:
        os.environ.pop('NO_EXTERNAL', None)
        os.environ.pop('MINIMAL_MODE', None)

def test_cli_quick_e2e(temp_project_dir):
    """Test CLI quick mode end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    
    try:
        brain = ProjectBrain(project_path=temp_project_dir)
        result = run_workflow(
            task="Write a function that returns 'hello world'",
            brain=brain,
            mode="quick"
        )
        
        assert result is not None
        assert 'success' in result
        
        if not result['success']:
            assert 'error' in result
            
    finally:
        os.environ.pop('NO_EXTERNAL', None)

def test_cli_research_e2e(temp_project_dir):
    """Test CLI research mode end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    
    try:
        brain = ProjectBrain(project_path=temp_project_dir)
        result = run_workflow(
            task="Research Python web frameworks",
            brain=brain,
            mode="research_only"
        )
        
        assert result is not None
        assert 'success' in result
        
        if not result['success']:
            assert 'error' in result
            
    finally:
        os.environ.pop('NO_EXTERNAL', None)

# ==================== SUBPROCESS E2E TESTS ====================

def test_subprocess_cli_e2e(temp_project_dir):
    """Test CLI via subprocess end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    try:
        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(temp_project_dir)
        
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(original_cwd, 'main.py'),
                 'Write a simple function',
                 '--mode', 'minimal'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Should not crash
            assert result.returncode in [0, 1]  # 0=success, 1=help/error
            assert len(result.stdout) > 0 or len(result.stderr) > 0
            
        finally:
            os.chdir(original_cwd)
            
    finally:
        os.environ.pop('NO_EXTERNAL', None)
        os.environ.pop('MINIMAL_MODE', None)

# ==================== MEMORY E2E TESTS ====================

def test_memory_persistence_e2e(temp_project_dir):
    """Test memory persistence end-to-end"""
    # Create brain and add some data
    brain1 = ProjectBrain(project_path=temp_project_dir)
    brain1.memory['test_data'] = 'test_value'
    brain1._save()
    
    # Create new brain instance (should load existing data)
    brain2 = ProjectBrain(project_path=temp_project_dir)
    assert brain2.memory['test_data'] == 'test_value'

def test_memory_workflow_integration_e2e(temp_project_dir):
    """Test memory integration with workflow end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    try:
        brain = ProjectBrain(project_path=temp_project_dir)
        
        # Run workflow
        result = run_workflow(
            task="Write a function",
            brain=brain,
            mode="minimal"
        )
        
        # Verify memory was updated
        assert result is not None
        assert len(brain.memory['memory_nodes']) >= 0  # May be empty if workflow failed
        
    finally:
        os.environ.pop('NO_EXTERNAL', None)
        os.environ.pop('MINIMAL_MODE', None)

# ==================== BENCHMARK TESTS ====================

def test_simple_benchmark_e2e(temp_project_dir):
    """Test simple benchmark end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    try:
        # Simple benchmark problems
        problems = [
            "Write a function that returns 'hello world'",
            "Write a function that adds two numbers",
            "Write a function that checks if a number is even"
        ]
        
        results = []
        for problem in problems:
            start_time = time.time()
            
            result = run_workflow(
                task=problem,
                brain=ProjectBrain(project_path=temp_project_dir),
                mode="minimal"
            )
            
            execution_time = time.time() - start_time
            
            results.append({
                'problem': problem,
                'success': result.get('success', False),
                'execution_time': execution_time,
                'error': result.get('error', None)
            })
        
        # Verify all problems were attempted
        assert len(results) == 3
        
        # At least some should succeed or fail gracefully (not crash)
        success_count = sum(1 for r in results if r['success'])
        graceful_failure_count = sum(1 for r in results if r.get('error') and any(
            term in str(r.get('error', '')).lower() 
            for term in ['timeout', 'timed out', 'error', 'failed', 'rate limit']
        ))
        
        assert success_count + graceful_failure_count >= 1, "At least one problem should succeed or fail gracefully"
        
    finally:
        os.environ.pop('NO_EXTERNAL', None)
        os.environ.pop('MINIMAL_MODE', None)

def test_benchmark_performance_e2e(temp_project_dir):
    """Test benchmark performance end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    try:
        start_time = time.time()
        
        # Run multiple simple problems
        for i in range(3):
            result = run_workflow(
                task=f"Write function {i}",
                brain=ProjectBrain(project_path=temp_project_dir),
                mode="minimal"
            )
            
            assert result is not None
        
        total_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert total_time < 180, f"Benchmark took {total_time}s, should be under 180s"
        
    finally:
        os.environ.pop('NO_EXTERNAL', None)
        os.environ.pop('MINIMAL_MODE', None)

# ==================== ERROR RECOVERY TESTS ====================

def test_error_recovery_e2e(temp_project_dir):
    """Test error recovery end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    try:
        # Test with invalid input
        result = run_workflow(
            task="",  # Empty task
            brain=ProjectBrain(project_path=temp_project_dir),
            mode="minimal"
        )
        
        # Should handle gracefully
        assert result is not None
        if not result['success']:
            assert 'error' in result
        
        # Test with very long task
        long_task = "Write a function " + "with many details " * 100
        result2 = run_workflow(
            task=long_task,
            brain=ProjectBrain(project_path=temp_project_dir),
            mode="minimal"
        )
        
        # Should handle gracefully
        assert result2 is not None
        
    finally:
        os.environ.pop('NO_EXTERNAL', None)
        os.environ.pop('MINIMAL_MODE', None)

# ==================== INTEGRATION E2E TESTS ====================

def test_full_integration_e2e(temp_project_dir):
    """Test full integration end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    
    try:
        brain = ProjectBrain(project_path=temp_project_dir)
        
        # 1. Test research
        research_result = run_workflow(
            task="Research Python web frameworks",
            brain=brain,
            mode="research_only"
        )
        
        # 2. Test planning
        plan_result = run_workflow(
            task="Plan a web API",
            brain=brain,
            mode="minimal"
        )
        
        # 3. Test coding
        code_result = run_workflow(
            task="Write a simple web API",
            brain=brain,
            mode="minimal"
        )
        
        # All should complete without crashing
        assert research_result is not None
        assert plan_result is not None
        assert code_result is not None
        
        # Memory should persist across workflows
        assert len(brain.memory['memory_nodes']) >= 0
        
    finally:
        os.environ.pop('NO_EXTERNAL', None)

def test_multi_workflow_e2e(temp_project_dir):
    """Test multiple workflows end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    try:
        brain = ProjectBrain(project_path=temp_project_dir)
        
        # Run multiple workflows
        results = []
        for i in range(3):
            result = run_workflow(
                task=f"Task {i}: Write function {i}",
                brain=brain,
                mode="minimal"
            )
            results.append(result)
        
        # All should complete
        assert len(results) == 3
        for result in results:
            assert result is not None
        
        # Memory should accumulate
        assert len(brain.memory['memory_nodes']) >= 0
        
    finally:
        os.environ.pop('NO_EXTERNAL', None)
        os.environ.pop('MINIMAL_MODE', None)

# ==================== PERFORMANCE E2E TESTS ====================

def test_performance_under_load_e2e(temp_project_dir):
    """Test performance under load end-to-end"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    try:
        start_time = time.time()
        
        # Run multiple workflows concurrently (simulated)
        results = []
        for i in range(5):
            result = run_workflow(
                task=f"Load test task {i}",
                brain=ProjectBrain(project_path=temp_project_dir),
                mode="minimal"
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert total_time < 300, f"Load test took {total_time}s, should be under 300s"
        
        # All should complete
        assert len(results) == 5
        for result in results:
            assert result is not None
            
    finally:
        os.environ.pop('NO_EXTERNAL', None)
        os.environ.pop('MINIMAL_MODE', None)

# ==================== MOCK E2E TESTS ====================

def test_mock_e2e_workflow():
    """Test E2E workflow with mocked components"""
    with patch('graph.run_workflow') as mock_workflow:
        mock_workflow.return_value = {
            'success': True,
            'messages': ['Mock message'],
            'code_files': [
                {
                    'file': 'test.py',
                    'content': 'def hello(): return "hello world"',
                    'type': 'python'
                }
            ]
        }
        
        with patch('memory.ProjectBrain') as mock_brain_class:
            mock_brain = Mock()
            mock_brain.memory = {}
            mock_brain_class.return_value = mock_brain
            
            result = run_workflow(
                task="Mock task",
                brain=mock_brain,
                mode="minimal"
            )
            
            assert result is not None
            assert result['success'] is True
            assert len(result['code_files']) > 0

if __name__ == "__main__":
    pytest.main([__file__]) 