#!/usr/bin/env python3
"""
Consolidated Agent Tests - Unit tests for all agent components
Merged from scattered test files to prevent duplication and mess
"""

import os
import sys
import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from memory import ProjectBrain
from agents import (
    create_all_agents, 
    ReasoningAgent, 
    ReflectionAgent,
    get_default_model,
    get_anthropic_model,
    get_openai_model
)

# ==================== FIXTURES ====================

@pytest.fixture
def mock_brain():
    """Mock brain for testing"""
    brain = Mock(spec=ProjectBrain)
    brain.memory = {
        'decisions': [],
        'deployment_history': [],
        'github_examples': [],
        'reflections': []
    }
    brain._save = Mock()
    return brain

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = Mock()
    mock.invoke.return_value = {
        'output': 'Mock response from LLM',
        'content': 'Mock content'
    }
    return mock

# ==================== AGENT CREATION TESTS ====================

def test_create_all_agents(mock_brain):
    """Test creation of all agents"""
    agents = create_all_agents(mock_brain)
    
    expected_agents = [
        'planner', 'coder1', 'coder2', 'reviewer', 'integrator',
        'architect', 'coordinator', 'auditor', 'meta', 'reasoning', 'reflection'
    ]
    
    for agent_name in expected_agents:
        assert agent_name in agents, f"Agent {agent_name} not found"
        assert agents[agent_name] is not None, f"Agent {agent_name} is None"

def test_agent_creation_with_real_brain():
    """Test agent creation with real brain"""
    brain = ProjectBrain()
    agents = create_all_agents(brain)
    
    assert len(agents) >= 10, f"Expected at least 10 agents, got {len(agents)}"
    assert all(agents.values()), "All agents should be non-None"

# ==================== INDIVIDUAL AGENT TESTS ====================

@pytest.mark.asyncio
async def test_coder_agent_basic_functionality(mock_brain):
    """Test basic coder agent functionality"""
    agents = create_all_agents(mock_brain)
    coder = agents['coder1']
    
    # Test simple prompt
    result = coder.invoke({
        "input": "Write a function that returns 'hello world'",
        "chat_history": []
    })
    
    assert result is not None
    assert 'output' in result or 'content' in result
    output = result.get('output', result.get('content', ''))
    assert len(output) > 0, "Coder should produce output"

def test_planner_agent_planning(mock_brain):
    """Test planner agent creates plans"""
    agents = create_all_agents(mock_brain)
    planner = agents['planner']
    
    result = planner.invoke({
        "input": "Create a plan for building a web API",
        "chat_history": []
    })
    
    assert result is not None
    output = result.get('output', result.get('content', ''))
    assert len(output) > 0, "Planner should produce output"
    assert any(word in output.lower() for word in ['plan', 'step', 'phase']), "Should contain planning language"

def test_architect_agent_research(mock_brain):
    """Test architect agent research capabilities"""
    agents = create_all_agents(mock_brain)
    architect = agents['architect']
    
    result = architect.invoke({
        "input": "Research best practices for Python web frameworks",
        "chat_history": []
    })
    
    assert result is not None
    output = result.get('output', result.get('content', ''))
    assert len(output) > 0, "Architect should produce output"

# ==================== REASONING AGENT TESTS ====================

@pytest.mark.asyncio
async def test_reasoning_agent_initialization(mock_brain):
    """Test reasoning agent initialization"""
    reasoning_agent = ReasoningAgent(mock_brain, max_depth=3, parallel_paths=2)
    
    assert reasoning_agent.brain == mock_brain
    assert reasoning_agent.max_depth == 3
    assert reasoning_agent.parallel_paths == 2

@pytest.mark.asyncio
async def test_reasoning_agent_validation(mock_brain):
    """Test reasoning agent validation"""
    reasoning_agent = ReasoningAgent(mock_brain)
    
    # Mock the LLM responses
    with patch.object(reasoning_agent, '_execute_reasoning_step') as mock_execute:
        mock_execute.return_value = "Valid reasoning step"
        
        result = await reasoning_agent._validate_reasoning_path(
            "Test reasoning path", "Test task"
        )
        
        assert result is not None
        assert 'score' in result

# ==================== REFLECTION AGENT TESTS ====================

def test_reflection_agent_initialization(mock_brain):
    """Test reflection agent initialization"""
    reflection_agent = ReflectionAgent(mock_brain)
    
    assert reflection_agent.brain == mock_brain

@pytest.mark.asyncio
async def test_reflection_agent_critique(mock_brain):
    """Test reflection agent critique functionality"""
    reflection_agent = ReflectionAgent(mock_brain)
    
    result = await reflection_agent.critique_output(
        "Sample output", "Sample task", "Sample context"
    )
    
    assert result is not None
    assert 'critical_issues' in result

# ==================== MODEL SELECTION TESTS ====================

def test_get_default_model():
    """Test default model selection"""
    # Test with no API keys (should use mock)
    with patch.dict(os.environ, {}, clear=True):
        model = get_default_model()
        assert model is not None

def test_get_anthropic_model():
    """Test Anthropic model creation"""
    with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}, clear=True):
        model = get_anthropic_model()
        assert model is not None

def test_get_openai_model():
    """Test OpenAI model creation"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
        model = get_openai_model()
        assert model is not None

# ==================== PERFORMANCE TESTS ====================

def test_agent_execution_time():
    """Test agent execution time is reasonable"""
    brain = ProjectBrain()
    agents = create_all_agents(brain)
    coder = agents['coder1']
    
    start_time = time.time()
    result = coder.invoke({
        "input": "Write a simple function",
        "chat_history": []
    })
    execution_time = time.time() - start_time
    
    assert execution_time < 30, f"Agent execution took {execution_time}s, should be under 30s"
    assert result is not None

# ==================== ERROR HANDLING TESTS ====================

def test_agent_error_handling(mock_brain):
    """Test agents handle errors gracefully"""
    agents = create_all_agents(mock_brain)
    coder = agents['coder1']
    
    # Test with invalid input
    try:
        result = coder.invoke({
            "input": "",  # Empty input
            "chat_history": []
        })
        # Should not crash, even with empty input
        assert result is not None
    except Exception as e:
        pytest.fail(f"Agent should handle empty input gracefully: {e}")

# ==================== INTEGRATION TESTS ====================

def test_agent_chain_execution(mock_brain):
    """Test multiple agents working together"""
    agents = create_all_agents(mock_brain)
    
    # Test planner -> coder chain
    planner = agents['planner']
    coder = agents['coder1']
    
    # Planner creates plan
    plan_result = planner.invoke({
        "input": "Plan a simple calculator function",
        "chat_history": []
    })
    
    # Coder implements based on plan
    code_result = coder.invoke({
        "input": f"Implement this plan: {plan_result.get('output', '')}",
        "chat_history": []
    })
    
    assert plan_result is not None
    assert code_result is not None
    assert len(plan_result.get('output', '')) > 0
    assert len(code_result.get('output', '')) > 0

# ==================== MOCK LLM TESTS ====================

def test_mock_llm_functionality(mock_llm):
    """Test mock LLM works correctly"""
    result = mock_llm.invoke({
        "input": "Test prompt",
        "chat_history": []
    })
    
    assert result['output'] == 'Mock response from LLM'
    assert result['content'] == 'Mock content'

if __name__ == "__main__":
    pytest.main([__file__]) 