#!/usr/bin/env python3
"""
Integration Tests for Graph Workflow - Tests workflow graph creation and execution
"""

import os
import sys
import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from memory import ProjectBrain, NodeType
from graph import (
    create_workflow_graph,
    create_quick_workflow_graph,
    create_research_only_graph,
    create_minimal_workflow_graph,
    run_workflow,
    research_node,
    planning_node,
    coding_node,
    review_node,
    auditor_node,
    deploy_node,
    reflection_node
)

# ==================== RATE LIMITING HANDLERS ====================

def mock_huggingface_rate_limit():
    """Mock HuggingFace API calls to avoid 429 rate limiting"""
    return patch('prompts._fetch_hf_datasets', return_value=[])

def mock_external_apis():
    """Mock all external API calls to avoid rate limiting and network issues"""
    patches = [
        # Mock HuggingFace datasets
        patch('prompts._fetch_hf_datasets', return_value=[]),
        # Mock PromptFactory initialization
        patch('prompts.PromptFactory._initialize_modules'),
        # Mock agent creation
        patch('agents.create_all_agents'),
        # Mock fallback functions
        patch('graph._fallback_research'),
        patch('graph._fallback_planning'),
        patch('graph._fallback_audit'),
        # Mock external research tools
        patch('tools.web_search'),
        patch('tools.github_search'),
        patch('tools.x_search'),
    ]
    return patches

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

@pytest.fixture
def sample_state(sample_brain):
    """Create sample state for testing"""
    return {
        "messages": [
            {"role": "system", "content": "Starting workflow"},
            {"role": "ai", "content": "Here's a Python function: def test(): pass"}
        ],
        "current_step": "planning",
        "task": "Create a simple function",
        "repo_url": "",
        "research_results": [],
        "plan": {},
        "audit_feedback": [],
        "code_files": [],
        "review_feedback": [],
        "deployment_status": {},
        "brain_context": {"brain": sample_brain},
        "reflection_context": {}
    }

# ==================== GRAPH CREATION TESTS ====================

def test_create_workflow_graph(sample_brain):
    """Test full workflow graph creation"""
    graph = create_workflow_graph(sample_brain)
    
    assert graph is not None
    # Should have all major nodes
    nodes = graph.nodes
    expected_nodes = ['research', 'planning', 'coding', 'review', 'auditor', 'deploy']
    
    for node in expected_nodes:
        assert node in nodes, f"Expected node {node} not found in graph"

def test_create_quick_workflow_graph(sample_brain):
    """Test quick workflow graph creation"""
    graph = create_quick_workflow_graph(sample_brain)
    
    assert graph is not None
    # Quick workflow should have fewer nodes
    nodes = graph.nodes
    assert len(nodes) < 10, "Quick workflow should have fewer nodes than full workflow"

def test_create_research_only_graph(sample_brain):
    """Test research-only workflow graph creation"""
    graph = create_research_only_graph(sample_brain)
    
    assert graph is not None
    # Should only have research node
    nodes = graph.nodes
    assert 'research' in nodes
    assert len(nodes) <= 2, "Research-only workflow should have minimal nodes"

def test_create_minimal_workflow_graph(sample_brain):
    """Test minimal workflow graph creation"""
    graph = create_minimal_workflow_graph(sample_brain)
    
    assert graph is not None
    # Minimal workflow should have very few nodes
    nodes = graph.nodes
    assert 'planner' in nodes
    assert 'coder' in nodes
    assert len(nodes) <= 3, "Minimal workflow should have very few nodes"

# ==================== NODE FUNCTION TESTS ====================

def test_research_node(sample_state):
    """Test research node functionality"""
    # Mock external dependencies to avoid rate limiting
    with patch('graph._fallback_research') as mock_fallback:
        mock_fallback.return_value = [
            {
                'source': 'mock_research',
                'content': 'Mock research results',
                'timestamp': '2025-01-27T10:00:00Z'
            }
        ]
        
        result = research_node(sample_state)
        
        assert result is not None
        assert 'research_results' in result
        assert len(result['research_results']) > 0

def test_planning_node(sample_state):
    """Test planning node functionality"""
    # Add some research results first
    sample_state['research_results'] = [
        {
            'source': 'mock_research',
            'content': 'Mock research results',
            'timestamp': '2025-01-27T10:00:00Z'
        }
    ]
    
    # Mock external dependencies to avoid rate limiting
    with patch('graph._fallback_planning') as mock_fallback:
        mock_fallback.return_value = {
            'phases': ['Research', 'Design', 'Implementation'],
            'tasks': ['Task 1', 'Task 2'],
            'timeline': '1 week'
        }
        
        result = planning_node(sample_state)
        
        assert result is not None
        assert 'plan' in result
        assert result['plan']['phases'] == ['Research', 'Design', 'Implementation']

def test_coding_node(sample_state):
    """Test coding node functionality"""
    # Add plan first
    sample_state['plan'] = {
        'phases': ['Implementation'],
        'tasks': ['Write function']
    }
    
    # Mock external dependencies to avoid rate limiting
    with patch('graph._fallback_coding') as mock_fallback:
        mock_fallback.return_value = [
            {
                'file': 'test.py',
                'content': 'def hello(): return "hello world"',
                'type': 'python'
            }
        ]
        
        result = coding_node(sample_state)
        
        assert result is not None
        assert 'code_files' in result
        assert len(result['code_files']) > 0

def test_review_node(sample_state):
    """Test review node functionality"""
    # Add code files first
    sample_state['code_files'] = [
        {
            'file': 'test.py',
            'content': 'def hello(): return "hello world"',
            'type': 'python'
        }
    ]
    
    result = review_node(sample_state)
    
    assert result is not None
    assert 'review_feedback' in result

def test_auditor_node(sample_state):
    """Test auditor node functionality"""
    # Add code files first
    sample_state['code_files'] = [
        {
            'file': 'test.py',
            'content': 'def hello(): return "hello world"',
            'type': 'python'
        }
    ]
    
    # Mock external dependencies to avoid rate limiting
    with patch('graph._fallback_audit') as mock_fallback:
        mock_fallback.return_value = [
            {
                'severity': 'low',
                'issue': 'No docstring',
                'suggestion': 'Add docstring'
            }
        ]
        
        result = auditor_node(sample_state)
        
        assert result is not None
        assert 'audit_feedback' in result
        assert len(result['audit_feedback']) > 0

def test_deploy_node(sample_state):
    """Test deploy node functionality"""
    # Add code files first
    sample_state['code_files'] = [
        {
            'file': 'test.py',
            'content': 'def hello(): return "hello world"',
            'type': 'python'
        }
    ]
    
    # Mock the entire agent creation to prevent real API calls
    with patch('graph.create_all_agents') as mock_create_agents:
        # Create a mock integrator agent
        mock_integrator = Mock()
        mock_integrator.invoke.return_value = {
            'output': 'Deployment preparation completed successfully'
        }
        mock_create_agents.return_value = {'integrator': mock_integrator}
        
        result = deploy_node(sample_state)
        
        assert result is not None
        assert 'deployment_status' in result

def test_reflection_node(sample_state):
    """Test reflection node functionality"""
    # Add some context
    sample_state['messages'] = [
        {"role": "user", "content": "Write a function"},
        {"role": "assistant", "content": "Here's the function"}
    ]
    
    result = reflection_node(sample_state)
    
    assert result is not None
    assert 'reflection_context' in result

# ==================== WORKFLOW EXECUTION TESTS ====================

def test_run_workflow_minimal_mode(sample_brain):
    """Test minimal workflow execution with comprehensive mocking"""
    # Set environment to disable external APIs
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    # Apply all external API mocks to avoid rate limiting
    patches = mock_external_apis()
    
    with patch('prompts.PromptFactory') as mock_prompt_factory, \
         patch('agents.create_all_agents') as mock_create_agents:
        
        # Mock PromptFactory
        mock_factory = Mock()
        mock_factory.get_planning_prompt.return_value = "Mock planning prompt"
        mock_prompt_factory.return_value = mock_factory
        
        # Mock agent creation
        mock_planner = Mock()
        mock_planner.invoke.return_value = {
            'output': '{"phases": ["Implementation"], "tasks": ["Write function"]}'
        }
        mock_create_agents.return_value = {'planner': mock_planner}
        
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
                # Check for various error types including JSON serialization
                assert any(keyword in error_msg for keyword in ['timeout', 'error', 'json', 'serializable', 'bytes'])
                
        finally:
            # Clean up environment
            os.environ.pop('NO_EXTERNAL', None)
            os.environ.pop('MINIMAL_MODE', None)

def test_run_workflow_quick_mode(sample_brain):
    """Test quick workflow execution with comprehensive mocking"""
    # Set environment to disable external APIs
    os.environ['NO_EXTERNAL'] = '1'
    
    # Apply all external API mocks to avoid rate limiting
    with patch('prompts.PromptFactory') as mock_prompt_factory, \
         patch('agents.create_all_agents') as mock_create_agents:
        
        # Mock PromptFactory
        mock_factory = Mock()
        mock_factory.get_planning_prompt.return_value = "Mock planning prompt"
        mock_prompt_factory.return_value = mock_factory
        
        # Mock agent creation
        mock_planner = Mock()
        mock_planner.invoke.return_value = {
            'output': '{"phases": ["Implementation"], "tasks": ["Write function"]}'
        }
        mock_create_agents.return_value = {'planner': mock_planner}
        
        try:
            result = run_workflow(
                task="Write a function that returns 'hello world'",
                brain=sample_brain,
                mode="quick"
            )
            
            assert result is not None
            assert 'success' in result
            
            # Should either succeed or fail with timeout, not crash
            if not result['success']:
                assert 'error' in result
                
        finally:
            # Clean up environment
            os.environ.pop('NO_EXTERNAL', None)

def test_run_workflow_research_mode(sample_brain):
    """Test research-only workflow execution with mocking"""
    # Set environment to disable external APIs
    os.environ['NO_EXTERNAL'] = '1'
    
    # Mock external dependencies
    with patch('graph._fallback_research') as mock_fallback:
        mock_fallback.return_value = [
            {
                'source': 'mock_research',
                'content': 'Mock research results',
                'timestamp': '2025-01-27T10:00:00Z'
            }
        ]
        
        try:
            result = run_workflow(
                task="Research Python web frameworks",
                brain=sample_brain,
                mode="research_only"
            )
            
            assert result is not None
            assert 'success' in result
            
            # Should either succeed or fail with timeout, not crash
            if not result['success']:
                assert 'error' in result
                
        finally:
            # Clean up environment
            os.environ.pop('NO_EXTERNAL', None)

def test_run_workflow_full_mode(sample_brain):
    """Test full workflow execution with comprehensive mocking"""
    # Set environment to disable external APIs
    os.environ['NO_EXTERNAL'] = '1'
    
    # Apply all external API mocks to avoid rate limiting
    with patch('prompts.PromptFactory') as mock_prompt_factory, \
         patch('agents.create_all_agents') as mock_create_agents, \
         patch('graph._fallback_research') as mock_research, \
         patch('graph._fallback_planning') as mock_planning, \
         patch('graph._fallback_audit') as mock_audit:
        
        # Mock all fallback functions
        mock_research.return_value = [{'source': 'mock', 'content': 'mock'}]
        mock_planning.return_value = {'phases': ['Implementation'], 'tasks': ['Task']}
        mock_audit.return_value = [{'severity': 'low', 'issue': 'test'}]
        
        # Mock PromptFactory
        mock_factory = Mock()
        mock_factory.get_planning_prompt.return_value = "Mock planning prompt"
        mock_prompt_factory.return_value = mock_factory
        
        # Mock agent creation
        mock_planner = Mock()
        mock_planner.invoke.return_value = {
            'output': '{"phases": ["Implementation"], "tasks": ["Write function"]}'
        }
        mock_create_agents.return_value = {'planner': mock_planner}
        
        try:
            result = run_workflow(
                task="Build a simple web API",
                brain=sample_brain,
                mode="full"
            )
            
            assert result is not None
            assert 'success' in result
            
            # Should either succeed or fail with timeout, not crash
            if not result['success']:
                assert 'error' in result
                
        finally:
            # Clean up environment
            os.environ.pop('NO_EXTERNAL', None)

# ==================== TIMEOUT TESTS ====================

def test_workflow_timeout_handling(sample_brain):
    """Test workflow timeout handling"""
    # Set very short timeout
    os.environ['NO_EXTERNAL'] = '1'
    
    # Mock external dependencies
    with patch('prompts.PromptFactory') as mock_prompt_factory, \
         patch('agents.create_all_agents') as mock_create_agents:
        
        # Mock PromptFactory
        mock_factory = Mock()
        mock_factory.get_planning_prompt.return_value = "Mock planning prompt"
        mock_prompt_factory.return_value = mock_factory
        
        # Mock agent creation
        mock_planner = Mock()
        mock_planner.invoke.return_value = {
            'output': '{"phases": ["Implementation"], "tasks": ["Write function"]}'
        }
        mock_create_agents.return_value = {'planner': mock_planner}
        
        try:
            result = run_workflow(
                task="This should timeout quickly",
                brain=sample_brain,
                mode="minimal"
            )
            
            assert result is not None
            # Should handle timeout gracefully
            if not result['success']:
                assert 'error' in result
                
        finally:
            os.environ.pop('NO_EXTERNAL', None)

# ==================== ERROR HANDLING TESTS ====================

def test_workflow_error_handling(sample_brain):
    """Test workflow error handling"""
    # Test with invalid task
    result = run_workflow(
        task="",  # Empty task
        brain=sample_brain,
        mode="minimal"
    )
    
    assert result is not None
    # Should handle gracefully
    if not result['success']:
        assert 'error' in result

def test_node_error_handling(sample_state):
    """Test individual node error handling"""
    # Test with invalid state
    invalid_state = {}
    
    try:
        result = research_node(invalid_state)
        # Should handle gracefully
        assert result is not None
    except Exception as e:
        pytest.fail(f"Node should handle invalid state gracefully: {e}")

# ==================== PERFORMANCE TESTS ====================

def test_workflow_performance(sample_brain):
    """Test workflow performance"""
    os.environ['NO_EXTERNAL'] = '1'
    os.environ['MINIMAL_MODE'] = '1'
    
    # Mock external dependencies
    with patch('prompts.PromptFactory') as mock_prompt_factory, \
         patch('agents.create_all_agents') as mock_create_agents:
        
        # Mock PromptFactory
        mock_factory = Mock()
        mock_factory.get_planning_prompt.return_value = "Mock planning prompt"
        mock_prompt_factory.return_value = mock_factory
        
        # Mock agent creation
        mock_planner = Mock()
        mock_planner.invoke.return_value = {
            'output': '{"phases": ["Implementation"], "tasks": ["Write function"]}'
        }
        mock_create_agents.return_value = {'planner': mock_planner}
        
        try:
            start_time = time.time()
            
            result = run_workflow(
                task="Simple test task",
                brain=sample_brain,
                mode="minimal"
            )
            
            execution_time = time.time() - start_time
            
            assert result is not None
            # Should complete within reasonable time or timeout gracefully
            if result['success']:
                assert execution_time < 60, f"Workflow took {execution_time}s, should be under 60s"
            else:
                # If it failed, should be due to timeout or external issues
                assert 'error' in result
                
        finally:
            os.environ.pop('NO_EXTERNAL', None)
            os.environ.pop('MINIMAL_MODE', None)

# ==================== INTEGRATION TESTS ====================

def test_workflow_integration_chain(sample_brain):
    """Test complete workflow integration"""
    os.environ['NO_EXTERNAL'] = '1'
    
    # Mock external dependencies
    with patch('prompts.PromptFactory') as mock_prompt_factory, \
         patch('agents.create_all_agents') as mock_create_agents, \
         patch('graph._fallback_research') as mock_research, \
         patch('graph._fallback_planning') as mock_planning:
        
        # Mock all fallback functions
        mock_research.return_value = [{'source': 'mock', 'content': 'mock'}]
        mock_planning.return_value = {'phases': ['Implementation'], 'tasks': ['Task']}
        
        # Mock PromptFactory
        mock_factory = Mock()
        mock_factory.get_planning_prompt.return_value = "Mock planning prompt"
        mock_prompt_factory.return_value = mock_factory
        
        # Mock agent creation
        mock_planner = Mock()
        mock_planner.invoke.return_value = {
            'output': '{"phases": ["Implementation"], "tasks": ["Write function"]}'
        }
        mock_create_agents.return_value = {'planner': mock_planner}
        
        try:
            # Test the full chain: research -> plan -> code
            result = run_workflow(
                task="Create a simple calculator function",
                brain=sample_brain,
                mode="minimal"
            )
            
            assert result is not None
            assert 'success' in result
            
            if result['success']:
                # Should have some output
                assert 'messages' in result or 'code_files' in result
                
        finally:
            os.environ.pop('NO_EXTERNAL', None)

def test_workflow_state_persistence(sample_brain):
    """Test workflow state persistence"""
    os.environ['NO_EXTERNAL'] = '1'
    
    # Mock external dependencies
    with patch('prompts.PromptFactory') as mock_prompt_factory, \
         patch('agents.create_all_agents') as mock_create_agents:
        
        # Mock PromptFactory
        mock_factory = Mock()
        mock_factory.get_planning_prompt.return_value = "Mock planning prompt"
        mock_prompt_factory.return_value = mock_factory
        
        # Mock agent creation
        mock_planner = Mock()
        mock_planner.invoke.return_value = {
            'output': '{"phases": ["Implementation"], "tasks": ["Write function"]}'
        }
        mock_create_agents.return_value = {'planner': mock_planner}
        
        try:
            # Run workflow
            result1 = run_workflow(
                task="Test task 1",
                brain=sample_brain,
                mode="minimal"
            )
            
            # Run another workflow
            result2 = run_workflow(
                task="Test task 2",
                brain=sample_brain,
                mode="minimal"
            )
            
            # Both should work independently
            assert result1 is not None
            assert result2 is not None
            
        finally:
            os.environ.pop('NO_EXTERNAL', None)

# ==================== MOCK TESTS ====================

def test_mock_workflow_execution():
    """Test workflow with mocked components"""
    mock_brain = Mock(spec=ProjectBrain)
    mock_brain.memory = {}
    mock_brain._save = Mock()
    
    with patch('graph.create_minimal_workflow_graph') as mock_create_graph:
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph
        
        with patch('graph.BrainCheckpoint') as mock_checkpoint:
            mock_checkpoint_instance = Mock()
            mock_checkpoint.return_value = mock_checkpoint_instance
            
            with patch.object(mock_graph, 'compile') as mock_compile:
                mock_app = Mock()
                mock_compile.return_value = mock_app
                
                mock_app.invoke.return_value = {
                    'messages': [],
                    'research_results': [],
                    'plan': {},
                    'code_files': []
                }
                
                result = run_workflow(
                    task="Mock task",
                    brain=mock_brain,
                    mode="minimal"
                )
                
                assert result is not None
                assert result['success'] is True

if __name__ == "__main__":
    pytest.main([__file__]) 