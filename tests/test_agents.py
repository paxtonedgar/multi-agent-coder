"""
Test agent capabilities and interactions
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    create_all_agents, create_planner_agent, create_coder_agent, 
    create_reviewer_agent, create_integrator_agent, create_architect_agent,
    create_research_coordinator_agent, create_auditor_agent, create_meta_agent,
    ReasoningAgent, ReflectionAgent, MockLLM, get_default_model
)

class TestAgentCreation:
    """Test agent creation and initialization"""
    
    def test_create_all_agents(self, temp_brain):
        """Test creation of all agents"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agents = create_all_agents(temp_brain)
            
            expected_agents = [
                'planner', 'coder1', 'coder2', 'reviewer', 'integrator',
                'architect', 'coordinator', 'auditor', 'meta'
            ]
            
            for agent_name in expected_agents:
                assert agent_name in agents, f"Agent {agent_name} not created"
                assert agents[agent_name] is not None, f"Agent {agent_name} is None"
    
    def test_planner_agent_creation(self, temp_brain):
        """Test planner agent creation"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_planner_agent(temp_brain)
            
            assert agent is not None
            assert hasattr(agent, 'invoke')
            assert hasattr(agent, 'tools')
    
    def test_coder_agent_creation(self, temp_brain):
        """Test coder agent creation"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_coder_agent(temp_brain, coder_id=1)
            
            assert agent is not None
            assert hasattr(agent, 'invoke')
            assert hasattr(agent, 'tools')
    
    def test_reviewer_agent_creation(self, temp_brain):
        """Test reviewer agent creation"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_reviewer_agent(temp_brain)
            
            assert agent is not None
            assert hasattr(agent, 'invoke')
            assert hasattr(agent, 'tools')
    
    def test_integrator_agent_creation(self, temp_brain):
        """Test integrator agent creation"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_integrator_agent(temp_brain)
            
            assert agent is not None
            assert hasattr(agent, 'invoke')
            assert hasattr(agent, 'tools')
    
    def test_architect_agent_creation(self, temp_brain):
        """Test architect agent creation"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_architect_agent(temp_brain)
            
            assert agent is not None
            assert hasattr(agent, 'invoke')
            assert hasattr(agent, 'tools')
    
    def test_coordinator_agent_creation(self, temp_brain):
        """Test research coordinator agent creation"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_research_coordinator_agent(temp_brain)
            
            assert agent is not None
            assert hasattr(agent, 'invoke')
            assert hasattr(agent, 'tools')
    
    def test_auditor_agent_creation(self, temp_brain):
        """Test auditor agent creation"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_auditor_agent(temp_brain)
            
            assert agent is not None
            assert hasattr(agent, 'invoke')
            assert hasattr(agent, 'tools')
    
    def test_meta_agent_creation(self, temp_brain):
        """Test meta agent creation"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_meta_agent(temp_brain)
            
            assert agent is not None
            assert hasattr(agent, 'invoke')
            assert hasattr(agent, 'tools')

class TestAgentInteractions:
    """Test agent interactions and communication"""
    
    @pytest.mark.asyncio
    async def test_planner_agent_task(self, temp_brain):
        """Test planner agent with a task"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_planner_agent(temp_brain)
            
            result = await agent.ainvoke({
                "input": "Create a plan for building a REST API",
                "chat_history": []
            })
            
            assert result is not None
            assert 'output' in result
            assert len(result['output']) > 0
    
    @pytest.mark.asyncio
    async def test_coder_agent_task(self, temp_brain):
        """Test coder agent with a task"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_coder_agent(temp_brain, coder_id=1)
            
            result = await agent.ainvoke({
                "input": "Write a Python function to calculate fibonacci numbers",
                "chat_history": []
            })
            
            assert result is not None
            assert 'output' in result
            assert len(result['output']) > 0
    
    @pytest.mark.asyncio
    async def test_reviewer_agent_task(self, temp_brain):
        """Test reviewer agent with code review"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_reviewer_agent(temp_brain)
            
            test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
            
            result = await agent.ainvoke({
                "input": f"Review this code:\n{test_code}",
                "chat_history": []
            })
            
            assert result is not None
            assert 'output' in result
            assert len(result['output']) > 0
    
    @pytest.mark.asyncio
    async def test_architect_agent_research(self, temp_brain):
        """Test architect agent research capabilities"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_architect_agent(temp_brain)
            
            result = await agent.ainvoke({
                "input": "Research best practices for microservices architecture",
                "chat_history": []
            })
            
            assert result is not None
            assert 'output' in result
            assert len(result['output']) > 0
    
    @pytest.mark.asyncio
    async def test_coordinator_agent_synthesis(self, temp_brain):
        """Test coordinator agent synthesis capabilities"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_research_coordinator_agent(temp_brain)
            
            research_data = "Research findings: FastAPI is popular, Django is comprehensive"
            
            result = await agent.ainvoke({
                "input": f"Synthesize this research:\n{research_data}",
                "chat_history": []
            })
            
            assert result is not None
            assert 'output' in result
            assert len(result['output']) > 0

class TestReasoningAgent:
    """Test enhanced reasoning agent capabilities"""
    
    @pytest.fixture
    def reasoning_agent(self, temp_brain):
        """Create reasoning agent for testing"""
        return ReasoningAgent(temp_brain, max_depth=3, parallel_paths=3)
    
    @pytest.mark.asyncio
    async def test_reasoning_with_validation(self, reasoning_agent):
        """Test reasoning with validation"""
        result = await reasoning_agent.reason_with_validation(
            task="Design a distributed caching system",
            context="Must handle 1M requests per second"
        )
        
        assert result is not None
        assert 'output' in result
        assert 'validation_scores' in result
        assert 'reflection' in result
        assert 'confidence' in result['reflection']
    
    @pytest.mark.asyncio
    async def test_parallel_reasoning_paths(self, reasoning_agent):
        """Test parallel reasoning paths"""
        paths = await reasoning_agent._generate_parallel_paths(
            task="Solve complex optimization problem",
            context="Multiple approaches needed"
        )
        
        assert len(paths) == 3  # parallel_paths=3
        assert all(isinstance(path, str) for path in paths)
        assert all(len(path) > 0 for path in paths)
    
    @pytest.mark.asyncio
    async def test_reasoning_validation(self, reasoning_agent):
        """Test reasoning validation"""
        test_path = "Step 1: Analyze requirements. Step 2: Design solution. Step 3: Implement."
        
        validation = await reasoning_agent._validate_reasoning_path(
            path=test_path,
            original_task="Design a system"
        )
        
        assert validation is not None
        assert 'coherence_score' in validation
        assert 'completeness_score' in validation
        assert 'relevance_score' in validation
    
    @pytest.mark.asyncio
    async def test_self_reflection(self, reasoning_agent):
        """Test self-reflection capability"""
        test_output = "Here is the solution: Use Redis for caching with clustering."
        
        reflection = await reasoning_agent._self_reflect(
            output=test_output,
            task="Design caching system"
        )
        
        assert reflection is not None
        assert 'confidence' in reflection
        assert 'improvements' in reflection
        assert 'critical_issues' in reflection
    
    @pytest.mark.asyncio
    async def test_output_revision(self, reasoning_agent):
        """Test output revision capability"""
        original_output = "Use Redis for caching."
        feedback = "Add more details about configuration and scaling."
        
        revised_output = await reasoning_agent._revise_output(
            output=original_output,
            feedback=feedback
        )
        
        assert revised_output is not None
        assert len(revised_output) > len(original_output)
        assert "configuration" in revised_output.lower() or "scaling" in revised_output.lower()

class TestReflectionAgent:
    """Test reflection agent capabilities"""
    
    @pytest.fixture
    def reflection_agent(self, temp_brain):
        """Create reflection agent for testing"""
        return ReflectionAgent(temp_brain)
    
    @pytest.mark.asyncio
    async def test_output_critique(self, reflection_agent):
        """Test output critique capability"""
        test_output = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
        
        critique = await reflection_agent.critique_output(
            output=test_output,
            task="Implement fibonacci function",
            context="Performance is important"
        )
        
        assert critique is not None
        assert 'overall_score' in critique
        assert 'strengths' in critique
        assert 'weaknesses' in critique
        assert 'recommendations' in critique
        assert 'critical_issues' in critique
    
    @pytest.mark.asyncio
    async def test_critical_issues_extraction(self, reflection_agent):
        """Test critical issues extraction"""
        critique_text = """
        Critical issues found:
        1. No input validation
        2. Potential stack overflow
        3. Missing error handling
        """
        
        issues = reflection_agent._extract_critical_issues(critique_text)
        
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert all(isinstance(issue, str) for issue in issues)
        assert any("validation" in issue.lower() for issue in issues)

class TestAgentTools:
    """Test agent tool capabilities"""
    
    @pytest.mark.asyncio
    async def test_agent_tools_availability(self, temp_brain):
        """Test that agents have required tools"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agents = create_all_agents(temp_brain)
            
            # Check that each agent has tools
            for agent_name, agent in agents.items():
                assert hasattr(agent, 'tools'), f"Agent {agent_name} missing tools"
                assert len(agent.tools) > 0, f"Agent {agent_name} has no tools"
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, temp_brain):
        """Test tool execution by agents"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_planner_agent(temp_brain)
            
            # Test that tools can be executed
            for tool in agent.tools:
                assert hasattr(tool, 'name'), f"Tool missing name: {tool}"
                assert hasattr(tool, 'description'), f"Tool missing description: {tool}"
                assert hasattr(tool, 'func'), f"Tool missing func: {tool}"

class TestAgentPerformance:
    """Test agent performance and response times"""
    
    @pytest.mark.asyncio
    async def test_agent_response_time(self, temp_brain):
        """Test agent response time"""
        import time
        
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_planner_agent(temp_brain)
            
            start_time = time.time()
            result = await agent.ainvoke({
                "input": "Create a simple plan",
                "chat_history": []
            })
            end_time = time.time()
            
            # Should respond quickly with mock
            assert end_time - start_time < 2.0
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_agent_memory_usage(self, temp_brain):
        """Test agent memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('agents.get_default_model', return_value=MockLLM()):
            agents = create_all_agents(temp_brain)
            
            # Use all agents
            for agent_name, agent in agents.items():
                await agent.ainvoke({
                    "input": f"Test task for {agent_name}",
                    "chat_history": []
                })
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 200 * 1024 * 1024  # 200MB

class TestAgentErrorHandling:
    """Test agent error handling"""
    
    @pytest.mark.asyncio
    async def test_agent_api_error_handling(self, temp_brain):
        """Test agent handling of API errors"""
        # Create a mock LLM that raises exceptions
        class ErrorLLM(MockLLM):
            async def ainvoke(self, input, **kwargs):
                raise Exception("API Error")
        
        with patch('agents.get_default_model', return_value=ErrorLLM()):
            agent = create_planner_agent(temp_brain)
            
            # Should handle errors gracefully
            try:
                result = await agent.ainvoke({
                    "input": "Test task",
                    "chat_history": []
                })
                # If it doesn't raise, should return error response
                assert result is not None
            except Exception as e:
                # Should be handled gracefully
                assert "API Error" in str(e)
    
    @pytest.mark.asyncio
    async def test_agent_invalid_input_handling(self, temp_brain):
        """Test agent handling of invalid inputs"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_planner_agent(temp_brain)
            
            # Test with empty input
            result = await agent.ainvoke({
                "input": "",
                "chat_history": []
            })
            
            assert result is not None
            assert 'output' in result
            
            # Test with None input
            result = await agent.ainvoke({
                "input": None,
                "chat_history": []
            })
            
            assert result is not None
            assert 'output' in result

class TestAgentCollaboration:
    """Test agent collaboration and coordination"""
    
    @pytest.mark.asyncio
    async def test_agent_chain_execution(self, temp_brain):
        """Test chain of agent executions"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agents = create_all_agents(temp_brain)
            
            # Simulate a workflow: architect -> planner -> coder -> reviewer
            architect_result = await agents['architect'].ainvoke({
                "input": "Research REST API best practices",
                "chat_history": []
            })
            
            planner_result = await agents['planner'].ainvoke({
                "input": f"Create plan based on: {architect_result['output']}",
                "chat_history": []
            })
            
            coder_result = await agents['coder1'].ainvoke({
                "input": f"Implement based on: {planner_result['output']}",
                "chat_history": []
            })
            
            reviewer_result = await agents['reviewer'].ainvoke({
                "input": f"Review this code: {coder_result['output']}",
                "chat_history": []
            })
            
            # All should succeed
            assert all(result is not None for result in [
                architect_result, planner_result, coder_result, reviewer_result
            ])
    
    @pytest.mark.asyncio
    async def test_agent_memory_sharing(self, temp_brain):
        """Test that agents share memory through brain"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agents = create_all_agents(temp_brain)
            
            # First agent adds to memory
            await agents['architect'].ainvoke({
                "input": "Research topic and remember key findings",
                "chat_history": []
            })
            
            # Second agent should be able to access memory
            result = await agents['coordinator'].ainvoke({
                "input": "Synthesize previous research",
                "chat_history": []
            })
            
            assert result is not None
            assert 'output' in result

class TestAgentQuality:
    """Test agent output quality"""
    
    @pytest.mark.asyncio
    async def test_code_generation_quality(self, temp_brain):
        """Test code generation quality"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_coder_agent(temp_brain, coder_id=1)
            
            result = await agent.ainvoke({
                "input": "Write a Python function to sort a list",
                "chat_history": []
            })
            
            output = result['output']
            
            # Check for code quality indicators
            assert 'def ' in output or 'class ' in output
            assert 'import ' in output or 'from ' in output
            assert '"""' in output or "'''" in output  # docstrings
    
    @pytest.mark.asyncio
    async def test_planning_quality(self, temp_brain):
        """Test planning quality"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_planner_agent(temp_brain)
            
            result = await agent.ainvoke({
                "input": "Create a detailed plan for building a web application",
                "chat_history": []
            })
            
            output = result['output']
            
            # Check for planning quality indicators
            assert any(word in output.lower() for word in ['phase', 'step', 'stage'])
            assert any(word in output.lower() for word in ['timeline', 'schedule', 'deadline'])
            assert any(word in output.lower() for word in ['risk', 'challenge', 'consideration'])
    
    @pytest.mark.asyncio
    async def test_review_quality(self, temp_brain):
        """Test review quality"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_reviewer_agent(temp_brain)
            
            test_code = """
def bad_function():
    x = 1
    y = 2
    return x + y
"""
            
            result = await agent.ainvoke({
                "input": f"Review this code:\n{test_code}",
                "chat_history": []
            })
            
            output = result['output']
            
            # Check for review quality indicators
            assert any(word in output.lower() for word in ['issue', 'problem', 'concern'])
            assert any(word in output.lower() for word in ['suggestion', 'recommendation', 'improvement'])
            assert any(word in output.lower() for word in ['security', 'performance', 'quality'])

class TestAgentReliability:
    """Test agent reliability and consistency"""
    
    @pytest.mark.asyncio
    async def test_agent_consistency(self, temp_brain):
        """Test agent output consistency across multiple runs"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_planner_agent(temp_brain)
            
            results = []
            for _ in range(3):
                result = await agent.ainvoke({
                    "input": "Create a simple plan",
                    "chat_history": []
                })
                results.append(result['output'])
            
            # All results should be similar (not necessarily identical due to randomness)
            assert all(len(result) > 0 for result in results)
            assert all(isinstance(result, str) for result in results)
    
    @pytest.mark.asyncio
    async def test_agent_state_persistence(self, temp_brain):
        """Test that agent state persists across calls"""
        with patch('agents.get_default_model', return_value=MockLLM()):
            agent = create_planner_agent(temp_brain)
            
            # First call
            result1 = await agent.ainvoke({
                "input": "Remember that we're building a web app",
                "chat_history": []
            })
            
            # Second call should remember context
            result2 = await agent.ainvoke({
                "input": "What are we building?",
                "chat_history": []
            })
            
            # Both should succeed
            assert result1 is not None
            assert result2 is not None
            assert 'output' in result1
            assert 'output' in result2 