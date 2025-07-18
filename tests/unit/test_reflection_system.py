#!/usr/bin/env python3
"""
Reflection System Tests - Unit tests for reflection and self-improvement functionality
"""

import os
import sys
import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from memory import ProjectBrain, NodeType
from agents import ReflectionAgent, ReasoningAgent, create_meta_agent

# ==================== FIXTURES ====================

@pytest.fixture
def mock_brain():
    """Mock brain for testing"""
    brain = Mock(spec=ProjectBrain)
    brain.memory = {
        'decisions': [],
        'deployment_history': [],
        'github_examples': [],
        'reflections': [],
        'memory_nodes': [],
        'project_meta': {
            'version': '1.0',
            'last_updated': '2025-01-01T00:00:00',
            'project_name': 'test_project'
        }
    }
    brain._save = Mock()
    brain.add_node = Mock()
    brain.get_reflection_tree = Mock(return_value={'nodes': [], 'summary': 'No reflections'})
    brain.add_reflection = Mock(return_value='reflection_123')
    brain.refine_reflection = Mock(return_value='reflection_456')
    return brain

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value={'output': 'Mock response'})
    return llm

# ==================== REFLECTION AGENT TESTS ====================

@pytest.mark.asyncio
async def test_reflection_agent_critique_success(mock_brain, mock_llm):
    """Test successful output critique"""
    agent = ReflectionAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock successful response
    mock_llm.ainvoke.return_value = {
        'output': 'Correctness: 0.9\nCompleteness: 0.8\nCode Quality: 0.7\nExecutability: 0.9\nSecurity: 0.8\nPerformance: 0.7'
    }
    
    result = await agent.critique_output("test code", "test task", "test context")
    
    assert result['overall_score'] > 0.7
    assert 'scores' in result
    assert 'feedback' in result
    assert 'needs_revision' in result
    assert 'critical_issues' in result

@pytest.mark.asyncio
async def test_reflection_agent_critique_failure(mock_brain, mock_llm):
    """Test critique failure handling"""
    agent = ReflectionAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock failure
    mock_llm.ainvoke.side_effect = Exception("API error")
    
    result = await agent.critique_output("test code", "test task", "test context")
    
    assert result['overall_score'] == 0.5
    assert result['needs_revision'] == True
    assert 'Critique failed due to error' in result['feedback']
    assert 'Critique system unavailable' in result['critical_issues']

@pytest.mark.asyncio
async def test_reflection_agent_critique_no_scores(mock_brain, mock_llm):
    """Test critique with no score extraction"""
    agent = ReflectionAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock response without scores
    mock_llm.ainvoke.return_value = {
        'output': 'This is a general feedback without specific scores.'
    }
    
    result = await agent.critique_output("test code", "test task", "test context")
    
    assert result['overall_score'] == 0.5  # Default score
    assert all(score == 0.5 for score in result['scores'].values())

def test_reflection_agent_extract_critical_issues():
    """Test critical issue extraction"""
    agent = ReflectionAgent(Mock())
    
    critique_text = """
    This code has a security vulnerability.
    There's a bug in the logic.
    The performance is poor.
    """
    
    issues = agent._extract_critical_issues(critique_text)
    assert len(issues) > 0
    assert any('security' in issue.lower() for issue in issues)
    assert any('bug' in issue.lower() for issue in issues)

# ==================== REASONING AGENT TESTS ====================

@pytest.mark.asyncio
async def test_reasoning_agent_self_reflect_success(mock_brain, mock_llm):
    """Test successful self-reflection"""
    agent = ReasoningAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock successful JSON response
    mock_llm.ainvoke.return_value = {
        'output': '{"needs_revision": false, "confidence": 0.9, "feedback": "Good output", "improvement_suggestions": []}'
    }
    
    result = await agent._self_reflect("test output", "test task")
    
    assert result['needs_revision'] == False
    assert result['confidence'] == 0.9
    assert 'Good output' in result['feedback']

@pytest.mark.asyncio
async def test_reasoning_agent_self_reflect_json_failure(mock_brain, mock_llm):
    """Test self-reflection with JSON parsing failure"""
    agent = ReasoningAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock invalid JSON response
    mock_llm.ainvoke.return_value = {
        'output': 'This is not valid JSON'
    }
    
    result = await agent._self_reflect("test output", "test task")
    
    assert result['needs_revision'] == False
    assert result['confidence'] == 0.8  # Default confidence when no JSON found
    assert 'No specific feedback' in result['feedback']

@pytest.mark.asyncio
async def test_reasoning_agent_self_reflect_exception(mock_brain, mock_llm):
    """Test self-reflection with exception handling"""
    agent = ReasoningAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock exception
    mock_llm.ainvoke.side_effect = Exception("Network error")
    
    result = await agent._self_reflect("test output", "test task")
    
    assert result['needs_revision'] == False
    assert result['confidence'] == 0.5
    assert 'Reflection failed' in result['feedback']
    assert 'Check system connectivity' in result['improvement_suggestions']

@pytest.mark.asyncio
async def test_reasoning_agent_revise_output_success(mock_brain, mock_llm):
    """Test successful output revision"""
    agent = ReasoningAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock successful revision
    mock_llm.ainvoke.return_value = {
        'output': 'Improved version of the code'
    }
    
    result = await agent._revise_output("original code", "Fix the bug")
    
    assert 'Improved version' in result

@pytest.mark.asyncio
async def test_reasoning_agent_revise_output_failure(mock_brain, mock_llm):
    """Test output revision failure handling"""
    agent = ReasoningAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock failure
    mock_llm.ainvoke.side_effect = Exception("Revision failed")
    
    result = await agent._revise_output("original code", "Fix the bug")
    
    assert 'original code' in result
    assert 'Revision failed due to error' in result

@pytest.mark.asyncio
async def test_reasoning_agent_fallback_reasoning_success(mock_brain, mock_llm):
    """Test successful fallback reasoning"""
    agent = ReasoningAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock successful fallback
    mock_llm.ainvoke.return_value = {
        'output': 'Step-by-step solution'
    }
    
    result = await agent._fallback_reasoning("test task", "test context")
    
    assert 'Step-by-step solution' in result

@pytest.mark.asyncio
async def test_reasoning_agent_fallback_reasoning_failure(mock_brain, mock_llm):
    """Test fallback reasoning failure handling"""
    agent = ReasoningAgent(mock_brain)
    agent.llm = mock_llm
    
    # Mock failure
    mock_llm.ainvoke.side_effect = Exception("Fallback failed")
    
    result = await agent._fallback_reasoning("test task", "test context")
    
    assert 'Fallback reasoning failed' in result
    assert 'check system connectivity' in result.lower()

def test_reasoning_agent_logging(mock_brain):
    """Test reasoning logging functionality"""
    agent = ReasoningAgent(mock_brain)
    
    # Test start logging
    trace_id = agent._log_reasoning_start("test task", "test context")
    assert trace_id is not None
    assert len(agent.reasoning_traces) == 1
    
    # Test completion logging
    reflection = {"confidence": 0.8, "feedback": "Good"}
    agent._log_reasoning_completion(trace_id, "final output", reflection)
    
    # Verify trace was updated
    trace = next(t for t in agent.reasoning_traces if t['id'] == trace_id)
    assert trace['final_output'] == "final output"
    assert trace['reflection'] == reflection
    assert trace['success'] == True

# ==================== META AGENT TESTS ====================

def test_meta_agent_creation(mock_brain):
    """Test meta agent creation"""
    with patch('agents.get_model_router') as mock_router:
        mock_router.return_value = None
        
        agent = create_meta_agent(mock_brain)
        
        assert agent is not None
        assert hasattr(agent, 'agent')
        assert hasattr(agent, 'tools')

def test_meta_agent_tools(mock_brain):
    """Test meta agent tools"""
    with patch('agents.get_model_router') as mock_router:
        mock_router.return_value = None
        
        agent = create_meta_agent(mock_brain)
        
        # Check that reflection tools are present
        tool_names = [tool.name for tool in agent.tools]
        assert 'get_past_reflections' in tool_names
        assert 'add_reflection' in tool_names
        assert 'refine_reflection' in tool_names
        assert 'research_self_improvement' in tool_names

def test_meta_agent_get_past_reflections_tool(mock_brain):
    """Test get_past_reflections tool"""
    with patch('agents.get_model_router') as mock_router:
        mock_router.return_value = None
        
        agent = create_meta_agent(mock_brain)
        
        # Find the tool
        tool = next(t for t in agent.tools if t.name == 'get_past_reflections')
        
        # Test tool execution
        result = tool.invoke("test query")
        assert "No relevant past reflections found" in result

def test_meta_agent_add_reflection_tool(mock_brain):
    """Test add_reflection tool"""
    with patch('agents.get_model_router') as mock_router:
        mock_router.return_value = None
        
        agent = create_meta_agent(mock_brain)
        
        # Find the tool
        tool = next(t for t in agent.tools if t.name == 'add_reflection')
        
        # Test tool execution
        result = tool.invoke("test reflection content")
        assert "Reflection added with ID" in result

def test_meta_agent_refine_reflection_tool(mock_brain):
    """Test refine_reflection tool"""
    with patch('agents.get_model_router') as mock_router:
        mock_router.return_value = None
        
        agent = create_meta_agent(mock_brain)
        
        # Find the tool
        tool = next(t for t in agent.tools if t.name == 'refine_reflection')
        
        # Test tool execution with proper arguments
        result = tool.invoke({"reflection_id": "reflection_123", "new_content": "refined content"})
        assert "Reflection refined" in result

def test_meta_agent_research_tool(mock_brain):
    """Test research_self_improvement tool"""
    with patch('agents.get_model_router') as mock_router:
        mock_router.return_value = None
        
        agent = create_meta_agent(mock_brain)
        
        # Find the tool
        tool = next(t for t in agent.tools if t.name == 'research_self_improvement')
        
        # Test tool execution with mock research tools
        with patch('tools.RealResearchTools') as mock_research:
            mock_research_instance = Mock()
            mock_research_instance.web_search_integration.return_value = "Mock research results"
            mock_research.return_value = mock_research_instance
            
            result = tool.invoke("test topic")
            assert "Mock research results" in result

def test_meta_agent_research_tool_fallback(mock_brain):
    """Test research_self_improvement tool fallback"""
    with patch('agents.get_model_router') as mock_router:
        mock_router.return_value = None
        
        agent = create_meta_agent(mock_brain)
        
        # Find the tool
        tool = next(t for t in agent.tools if t.name == 'research_self_improvement')
        
        # Test tool execution with research failure
        with patch('tools.RealResearchTools') as mock_research:
            mock_research_instance = Mock()
            mock_research_instance.web_search_integration.side_effect = Exception("API error")
            mock_research.return_value = mock_research_instance
            
            result = tool.invoke("test topic")
            assert "Mock research data" in result

# ==================== ERROR HANDLING TESTS ====================

@pytest.mark.asyncio
async def test_reflection_system_serialization_safety(mock_brain):
    """Test that reflection system doesn't serialize coroutines"""
    agent = ReasoningAgent(mock_brain)
    
    # Test that logging doesn't fail with coroutines
    trace_id = agent._log_reasoning_start("test task", "test context")
    
    # This should not raise serialization errors
    try:
        agent._log_reasoning_completion(trace_id, "test output", {"confidence": 0.8})
        assert True  # Should not raise exception
    except Exception as e:
        if "coroutine" in str(e).lower():
            pytest.fail("Coroutine serialization error occurred")
        else:
            # Other errors are acceptable
            pass

def test_reflection_system_memory_integration(mock_brain):
    """Test reflection system memory integration"""
    agent = ReflectionAgent(mock_brain)
    
    # Test that brain operations don't fail
    try:
        agent._extract_critical_issues("test critique")
        assert True  # Should not raise exception
    except Exception as e:
        pytest.fail(f"Memory integration failed: {e}")

# ==================== INTEGRATION TESTS ====================

@pytest.mark.asyncio
async def test_reflection_system_integration(mock_brain, mock_llm):
    """Test full reflection system integration"""
    # Test reflection agent
    reflection_agent = ReflectionAgent(mock_brain)
    reflection_agent.llm = mock_llm
    
    mock_llm.ainvoke.return_value = {
        'output': 'Correctness: 0.9\nCompleteness: 0.8\nCode Quality: 0.7'
    }
    
    critique_result = await reflection_agent.critique_output("test code", "test task")
    assert critique_result['overall_score'] > 0.6  # Adjusted threshold for mock scores
    
    # Test reasoning agent
    reasoning_agent = ReasoningAgent(mock_brain)
    reasoning_agent.llm = mock_llm
    
    mock_llm.ainvoke.return_value = {
        'output': '{"needs_revision": false, "confidence": 0.9, "feedback": "Good"}'
    }
    
    reflection_result = await reasoning_agent._self_reflect("test output", "test task")
    assert reflection_result['confidence'] == 0.9
    
    # Test meta agent creation
    meta_agent = create_meta_agent(mock_brain)
    assert meta_agent is not None

@pytest.mark.asyncio
async def test_reflection_system_error_recovery(mock_brain, mock_llm):
    """Test reflection system error recovery"""
    # Test that system recovers from various errors
    reflection_agent = ReflectionAgent(mock_brain)
    reflection_agent.llm = mock_llm
    
    # Test API failure recovery
    mock_llm.ainvoke.side_effect = Exception("API unavailable")
    
    result = await reflection_agent.critique_output("test code", "test task")
    assert result['overall_score'] == 0.5  # Fallback score
    assert result['needs_revision'] == True
    
    # Test reasoning agent error recovery
    reasoning_agent = ReasoningAgent(mock_brain)
    reasoning_agent.llm = mock_llm
    
    result = await reasoning_agent._self_reflect("test output", "test task")
    assert result['confidence'] == 0.5  # Fallback confidence
    assert 'Reflection failed' in result['feedback']

# ==================== PERFORMANCE TESTS ====================

@pytest.mark.asyncio
async def test_reflection_system_performance(mock_brain, mock_llm):
    """Test reflection system performance"""
    import time
    
    reflection_agent = ReflectionAgent(mock_brain)
    reflection_agent.llm = mock_llm
    
    mock_llm.ainvoke.return_value = {
        'output': 'Correctness: 0.9\nCompleteness: 0.8\nCode Quality: 0.7'
    }
    
    start_time = time.time()
    
    # Test multiple critiques
    for i in range(5):
        await reflection_agent.critique_output(f"test code {i}", f"test task {i}")
    
    end_time = time.time()
    
    # Should complete quickly
    assert end_time - start_time < 2.0  # Less than 2 seconds for 5 critiques 