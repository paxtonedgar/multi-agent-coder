#!/usr/bin/env python3
"""
Tests for async improvements in workflow nodes
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from graph import (
    tree_of_thoughts_node, 
    debate_node, 
    async_manager, 
    AsyncErrorHandler,
    run_workflow
)
from memory import ProjectBrain, NodeType

# ==================== ASYNC MANAGER TESTS ====================

def test_async_manager_loop_creation():
    """Test async manager creates loops correctly"""
    manager = async_manager
    
    # Test loop creation
    loop, needs_cleanup = manager.get_or_create_loop()
    assert loop is not None
    assert isinstance(loop, asyncio.AbstractEventLoop)
    
    # Test cleanup
    manager.cleanup()

def test_async_manager_run_async():
    """Test async manager runs coroutines correctly"""
    manager = async_manager
    
    async def test_coro():
        await asyncio.sleep(0.01)  # Small delay
        return "test_result"
    
    # Run coroutine
    result = manager.run_async(test_coro())
    assert result == "test_result"
    
    # Cleanup
    manager.cleanup()

def test_async_manager_error_handling():
    """Test async manager handles errors correctly"""
    manager = async_manager
    
    async def failing_coro():
        raise ValueError("Test error")
    
    # Should raise the error
    with pytest.raises(ValueError, match="Test error"):
        manager.run_async(failing_coro())
    
    # Cleanup
    manager.cleanup()

# ==================== ASYNC ERROR HANDLER TESTS ====================

def test_async_error_handler():
    """Test async error handler creates consistent error results"""
    error = ValueError("Test error")
    
    result = AsyncErrorHandler.handle_async_error(
        "test_operation", 
        error, 
        {'fallback': 'value'}
    )
    
    assert result['success'] == False
    assert result['error']['operation'] == 'test_operation'
    assert result['error']['error_type'] == 'ValueError'
    assert result['error']['error_message'] == 'Test error'
    assert result['fallback_value'] == {'fallback': 'value'}

def test_async_error_handler_fallback():
    """Test async error handler fallback result creation"""
    result = AsyncErrorHandler.create_fallback_result(
        "test_operation", 
        {'fallback': 'value'}
    )
    
    assert result['success'] == False
    assert result['error']['operation'] == 'test_operation'
    assert result['error']['error_type'] == 'AsyncOperationFailed'
    assert result['fallback_value'] == {'fallback': 'value'}

# ==================== TREE OF THOUGHTS NODE TESTS ====================

def test_tree_of_thoughts_node_async_improvement(mock_brain):
    """Test tree of thoughts node uses improved async pattern"""
    with patch('agents.ReasoningAgent') as mock_reasoning_agent:
        # Mock reasoning agent
        mock_agent = Mock()
        mock_agent.reason_with_validation.return_value = asyncio.Future()
        mock_agent.reason_with_validation.return_value.set_result({
            'output': 'Best reasoning path found',
            'paths': ['path1', 'path2'],
            'best_path': 'path1',
            'best_path_score': 0.85
        })
        mock_reasoning_agent.return_value = mock_agent
        
        state = {
            "task": "Solve complex optimization problem",
            "brain_context": {"brain": mock_brain}
        }
        
        # Test that it uses the async manager
        with patch('graph.async_manager') as mock_async_manager:
            mock_async_manager.run_async.return_value = {
                'output': 'Best reasoning path found',
                'paths': ['path1', 'path2'],
                'best_path': 'path1',
                'best_path_score': 0.85
            }
            
            result = tree_of_thoughts_node(state)
            
            # Verify async manager was used
            mock_async_manager.run_async.assert_called_once()
            
            # Verify result
            assert "reasoning_results" in result
            assert result["reasoning_results"]["output"] == "Best reasoning path found"

def test_tree_of_thoughts_node_async_error_handling(mock_brain):
    """Test tree of thoughts node handles async errors gracefully"""
    with patch('agents.ReasoningAgent') as mock_reasoning_agent:
        # Mock reasoning agent that raises exception
        mock_agent = Mock()
        mock_agent.reason_with_validation.side_effect = Exception("Async error")
        mock_reasoning_agent.return_value = mock_agent
        
        state = {
            "task": "Solve complex optimization problem",
            "brain_context": {"brain": mock_brain}
        }
        
        # Test error handling
        with patch('graph.async_manager') as mock_async_manager:
            mock_async_manager.run_async.side_effect = Exception("Async error")
            
            result = tree_of_thoughts_node(state)
            
            # Verify error was handled
            assert "reasoning_results" in result
            assert "error" in result["reasoning_results"]

# ==================== DEBATE NODE TESTS ====================

def test_debate_node_async_improvement(mock_brain):
    """Test debate node uses improved async pattern"""
    with patch('debate_framework.MultiAgentDebateFramework') as mock_debate_framework:
        # Mock debate framework
        mock_debate = Mock()
        mock_debate.conduct_debate.return_value = asyncio.Future()
        mock_debate.conduct_debate.return_value.set_result(Mock(
            final_proposal='Agreed on approach A',
            consensus_score=0.85,
            confidence=0.85
        ))
        mock_debate_framework.return_value = mock_debate
        
        state = {
            "task": "Choose best architecture",
            "brain_context": {"brain": mock_brain}
        }
        
        # Test that it uses the async manager
        with patch('graph.async_manager') as mock_async_manager:
            mock_async_manager.run_async.return_value = Mock(
                final_proposal='Agreed on approach A',
                consensus_score=0.85,
                confidence=0.85
            )
            
            result = debate_node(state)
            
            # Verify async manager was used
            mock_async_manager.run_async.assert_called_once()
            
            # Verify result
            assert "debate_results" in result
            assert result["consensus"] == 'Agreed on approach A'

def test_debate_node_async_error_handling(mock_brain):
    """Test debate node handles async errors gracefully"""
    with patch('debate_framework.MultiAgentDebateFramework') as mock_debate_framework:
        # Mock debate framework that raises exception
        mock_debate = Mock()
        mock_debate.conduct_debate.side_effect = Exception("Async error")
        mock_debate_framework.return_value = mock_debate
        
        state = {
            "task": "Choose best architecture",
            "brain_context": {"brain": mock_brain}
        }
        
        # Test error handling
        with patch('graph.async_manager') as mock_async_manager:
            mock_async_manager.run_async.side_effect = Exception("Async error")
            
            result = debate_node(state)
            
            # Verify error was handled
            assert "debate_results" in result
            assert "error" in result["debate_results"]

# ==================== WORKFLOW INTEGRATION TESTS ====================

def test_workflow_async_improvement(mock_brain):
    """Test workflow uses improved async pattern"""
    with patch('graph.async_manager') as mock_async_manager:
        # Mock successful workflow execution
        mock_async_manager.run_async.return_value = {
            "messages": [{"role": "system", "content": "Workflow completed"}],
            "research_results": [],
            "plan": {},
            "code_files": [],
            "review_feedback": [],
            "deployment_status": {}
        }
        
        result = run_workflow(
            task="Test task",
            brain=mock_brain,
            mode="minimal"
        )
        
        # Verify async manager was used
        mock_async_manager.run_async.assert_called_once()
        mock_async_manager.cleanup.assert_called_once()
        
        # Verify result
        assert result["success"] == True

def test_workflow_async_timeout_handling(mock_brain):
    """Test workflow handles async timeouts correctly"""
    with patch('graph.async_manager') as mock_async_manager:
        # Mock timeout
        mock_async_manager.run_async.side_effect = asyncio.TimeoutError()
        
        result = run_workflow(
            task="Test task",
            brain=mock_brain,
            mode="minimal"
        )
        
        # Verify cleanup was called
        mock_async_manager.cleanup.assert_called_once()
        
        # Verify timeout error was handled
        assert result["success"] == False
        assert "timed out" in result["error"].lower()

# ==================== FIXTURES ====================

@pytest.fixture
def mock_brain():
    """Create a mock brain for testing"""
    brain = Mock(spec=ProjectBrain)
    brain.add_node = Mock()
    brain.memory = {}
    return brain 