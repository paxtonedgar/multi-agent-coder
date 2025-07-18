
"""
Test workflow orchestration and state management
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

from graph import (
    run_workflow, create_workflow_graph, create_quick_workflow_graph,
    research_node, planning_node, coding_node, review_node, deploy_node,
    auditor_node, reflection_node
)
from memory import AgentState, ProjectBrain

class TestWorkflowCreation:
    """Test workflow graph creation"""
    
    def test_create_workflow_graph(self, temp_brain):
        """Test full workflow graph creation"""
        graph = create_workflow_graph(temp_brain)
        
        assert graph is not None
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
        
        # Check for expected nodes
        expected_nodes = ['research', 'planning', 'coding', 'audit', 'review', 'deploy', 'reflection']
        for node in expected_nodes:
            assert node in graph.nodes, f"Node {node} missing from workflow graph"
    
    def test_create_quick_workflow_graph(self, temp_brain):
        """Test quick workflow graph creation"""
        graph = create_quick_workflow_graph(temp_brain)
        
        assert graph is not None
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
        
        # Quick workflow should have fewer nodes
        expected_nodes = ['research', 'planning', 'coding', 'review', 'deploy']
        for node in expected_nodes:
            assert node in graph.nodes, f"Node {node} missing from quick workflow graph"
    
    def test_workflow_edges(self, temp_brain):
        """Test workflow edge connections"""
        graph = create_workflow_graph(temp_brain)
        
        # Check for expected edges
        expected_edges = [
            ('research', 'planning'),
            ('planning', 'coding'),
            ('coding', 'audit'),
            ('audit', 'review'),
            ('review', 'deploy'),
            ('deploy', 'reflection')
        ]
        
        for edge in expected_edges:
            assert edge in graph.edges, f"Edge {edge} missing from workflow graph"

class TestWorkflowNodes:
    """Test individual workflow node execution"""
    
    @pytest.fixture
    def sample_state(self, temp_brain):
        """Create sample state for testing"""
        return AgentState(
            messages=[],
            current_step="research",
            task="Create a simple calculator",
            repo_url="",
            research_results=[],
            plan={},
            audit_feedback=[],
            code_files=[],
            review_feedback=[],
            deployment_status={},
            brain_context={"brain": temp_brain},
            reflection_context={}
        )
    
    @pytest.mark.asyncio
    async def test_research_node(self, sample_state):
        """Test research node execution"""
        with patch('graph._fallback_research') as mock_fallback:
            mock_fallback.return_value = [
                {
                    'source': 'architect',
                    'content': 'Research findings: Use FastAPI for REST APIs',
                    'timestamp': '2025-01-27T10:00:00Z'
                }
            ]
            
            result_state = research_node(sample_state)
            
            assert result_state is not None
            assert 'research_results' in result_state
            assert len(result_state['research_results']) > 0
            assert result_state['research_results'][0]['source'] == 'architect'
    
    @pytest.mark.asyncio
    async def test_planning_node(self, sample_state):
        """Test planning node execution"""
        # Add research results to state
        sample_state['research_results'] = [
            {
                'source': 'architect',
                'content': 'Research findings: Use FastAPI for REST APIs',
                'timestamp': '2025-01-27T10:00:00Z'
            }
        ]
        
        with patch('graph._fallback_planning') as mock_fallback:
            mock_fallback.return_value = {
                'phases': ['Research', 'Design', 'Implementation', 'Testing', 'Deployment'],
                'tasks': ['Setup FastAPI', 'Create endpoints', 'Add tests'],
                'timeline': '2-3 weeks',
                'risks': ['Integration complexity']
            }
            
            result_state = planning_node(sample_state)
            
            assert result_state is not None
            assert 'plan' in result_state
            assert 'phases' in result_state['plan']
            assert 'tasks' in result_state['plan']
    
    @pytest.mark.asyncio
    async def test_coding_node(self, sample_state):
        """Test coding node execution"""
        # Add plan to state
        sample_state['plan'] = {
            'phases': ['Research', 'Design', 'Implementation'],
            'tasks': ['Create calculator class', 'Add arithmetic methods'],
            'timeline': '1 week'
        }
        
        with patch('graph._fallback_coding') as mock_fallback:
            mock_fallback.return_value = [
                {
                    'filename': 'calculator.py',
                    'content': 'class Calculator:\n    def add(self, a, b):\n        return a + b',
                    'language': 'python'
                }
            ]
            
            result_state = coding_node(sample_state)
            
            assert result_state is not None
            assert 'code_files' in result_state
            assert len(result_state['code_files']) > 0
            assert result_state['code_files'][0]['filename'] == 'calculator.py'
    
    @pytest.mark.asyncio
    async def test_auditor_node(self, sample_state):
        """Test auditor node execution"""
        # Add code files to state
        sample_state['code_files'] = [
            {
                'filename': 'calculator.py',
                'content': 'class Calculator:\n    def add(self, a, b):\n        return a + b',
                'language': 'python'
            }
        ]
        
        with patch('graph._fallback_audit') as mock_fallback:
            mock_fallback.return_value = [
                {
                    'type': 'security',
                    'severity': 'low',
                    'message': 'Consider adding input validation',
                    'suggestion': 'Add type hints and validation'
                }
            ]
            
            result_state = auditor_node(sample_state)
            
            assert result_state is not None
            assert 'audit_feedback' in result_state
            assert len(result_state['audit_feedback']) > 0
            assert result_state['audit_feedback'][0]['type'] == 'security'
    
    @pytest.mark.asyncio
    async def test_review_node(self, sample_state):
        """Test review node execution"""
        # Add code files to state
        sample_state['code_files'] = [
            {
                'filename': 'calculator.py',
                'content': 'class Calculator:\n    def add(self, a, b):\n        return a + b',
                'language': 'python'
            }
        ]
        
        with patch('agents.create_reviewer_agent') as mock_reviewer:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                'output': 'Code review completed. Quality score: 0.85. Suggestions: Add type hints.'
            }
            mock_reviewer.return_value = mock_agent
            
            result_state = review_node(sample_state)
            
            assert result_state is not None
            assert 'review_feedback' in result_state
            assert len(result_state['review_feedback']) > 0
    
    @pytest.mark.asyncio
    async def test_deploy_node(self, sample_state):
        """Test deploy node execution"""
        # Add code files to state
        sample_state['code_files'] = [
            {
                'filename': 'calculator.py',
                'content': 'class Calculator:\n    def add(self, a, b):\n        return a + b',
                'language': 'python'
            }
        ]
        
        with patch('agents.create_integrator_agent') as mock_integrator:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                'output': 'Deployment ready. CI/CD configured with GitHub Actions.'
            }
            mock_integrator.return_value = mock_agent
            
            result_state = deploy_node(sample_state)
            
            assert result_state is not None
            assert 'deployment_status' in result_state
            assert 'status' in result_state['deployment_status']
    
    @pytest.mark.asyncio
    async def test_reflection_node(self, sample_state):
        """Test reflection node execution"""
        # Add all previous results to state
        sample_state['research_results'] = [{'source': 'test', 'content': 'Research done'}]
        sample_state['plan'] = {'phases': ['test'], 'tasks': ['test']}
        sample_state['code_files'] = [{'filename': 'test.py', 'content': 'test'}]
        sample_state['review_feedback'] = [{'type': 'quality', 'message': 'Good code'}]
        sample_state['deployment_status'] = {'status': 'ready'}
        
        with patch('agents.ReflectionAgent') as mock_reflection:
            mock_agent = MagicMock()
            mock_agent.critique_output.return_value = {
                'overall_score': 0.85,
                'strengths': ['Good structure'],
                'weaknesses': ['Missing tests'],
                'recommendations': ['Add unit tests']
            }
            mock_reflection.return_value = mock_agent
            
            result_state = reflection_node(sample_state)
            
            assert result_state is not None
            assert 'reflection_context' in result_state
            assert 'overall_score' in result_state['reflection_context']

class TestWorkflowExecution:
    """Test complete workflow execution"""
    
    @pytest.mark.asyncio
    async def test_run_workflow_full_mode(self, temp_brain):
        """Test full workflow execution"""
        with patch('graph.create_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Workflow completed successfully',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            result = run_workflow(
                task="Create a simple calculator",
                brain=temp_brain,
                mode="full",
                repo_url=""
            )
            
            assert result is not None
            assert 'final_result' in result
            assert result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_run_workflow_quick_mode(self, temp_brain):
        """Test quick workflow execution"""
        with patch('graph.create_quick_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Quick workflow completed',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            result = run_workflow(
                task="Create a simple calculator",
                brain=temp_brain,
                mode="quick",
                repo_url=""
            )
            
            assert result is not None
            assert 'final_result' in result
            assert result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_run_workflow_research_mode(self, temp_brain):
        """Test research-only workflow execution"""
        with patch('graph.create_research_only_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Research completed',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            result = run_workflow(
                task="Research REST API frameworks",
                brain=temp_brain,
                mode="research",
                repo_url=""
            )
            
            assert result is not None
            assert 'final_result' in result
            assert result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_run_workflow_with_repo(self, temp_brain, test_repo):
        """Test workflow execution with repository"""
        with patch('graph.create_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Workflow with repo completed',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            repo_url = f"file://{test_repo}"
            result = run_workflow(
                task="Analyze repository structure",
                brain=temp_brain,
                mode="full",
                repo_url=repo_url
            )
            
            assert result is not None
            assert 'final_result' in result
            assert result['status'] == 'success'

class TestWorkflowStateManagement:
    """Test workflow state management and persistence"""
    
    @pytest.mark.asyncio
    async def test_state_persistence_across_nodes(self, temp_brain):
        """Test that state persists across workflow nodes"""
        # Create initial state
        initial_state = AgentState(
            messages=[],
            current_step="research",
            task="Create a calculator",
            repo_url="",
            research_results=[],
            plan={},
            audit_feedback=[],
            code_files=[],
            review_feedback=[],
            deployment_status={},
            brain_context={"brain": temp_brain},
            reflection_context={}
        )
        
        # Simulate workflow execution
        with patch('graph._fallback_research') as mock_research:
            mock_research.return_value = [{'source': 'test', 'content': 'Research done'}]
            
            # Execute research node
            state_after_research = research_node(initial_state)
            assert 'research_results' in state_after_research
            assert len(state_after_research['research_results']) > 0
            
            # Execute planning node
            with patch('graph._fallback_planning') as mock_planning:
                mock_planning.return_value = {'phases': ['test'], 'tasks': ['test']}
                
                state_after_planning = planning_node(state_after_research)
                assert 'plan' in state_after_planning
                assert 'research_results' in state_after_planning  # Should persist
    
    @pytest.mark.asyncio
    async def test_state_validation(self, temp_brain):
        """Test state validation and error handling"""
        # Test with invalid state
        invalid_state = {
            'task': 'Test task',
            'brain_context': {'brain': temp_brain}
        }
        
        # Should handle invalid state gracefully
        try:
            result = research_node(invalid_state)
            assert result is not None
        except Exception as e:
            # Should provide meaningful error message
            assert 'brain' in str(e) or 'context' in str(e)
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, temp_brain):
        """Test state transitions between workflow steps"""
        state = AgentState(
            messages=[],
            current_step="research",
            task="Test task",
            repo_url="",
            research_results=[],
            plan={},
            audit_feedback=[],
            code_files=[],
            review_feedback=[],
            deployment_status={},
            brain_context={"brain": temp_brain},
            reflection_context={}
        )
        
        # Test step transitions
        assert state['current_step'] == "research"
        
        # After research
        with patch('graph._fallback_research') as mock_research:
            mock_research.return_value = [{'source': 'test', 'content': 'Research done'}]
            state = research_node(state)
        
        # After planning
        with patch('graph._fallback_planning') as mock_planning:
            mock_planning.return_value = {'phases': ['test'], 'tasks': ['test']}
            state = planning_node(state)
        
        # State should contain all previous results
        assert 'research_results' in state
        assert 'plan' in state
        assert len(state['research_results']) > 0

class TestWorkflowErrorHandling:
    """Test workflow error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_node_execution_error(self, temp_brain):
        """Test handling of node execution errors"""
        state = AgentState(
            messages=[],
            current_step="research",
            task="Test task",
            repo_url="",
            research_results=[],
            plan={},
            audit_feedback=[],
            code_files=[],
            review_feedback=[],
            deployment_status={},
            brain_context={"brain": temp_brain},
            reflection_context={}
        )
        
        # Mock research to raise exception
        with patch('graph._fallback_research', side_effect=Exception("Research failed")):
            try:
                result = research_node(state)
                # Should handle error gracefully
                assert result is not None
                assert 'messages' in result
            except Exception as e:
                # Should provide meaningful error
                assert "Research failed" in str(e)
    
    @pytest.mark.asyncio
    async def test_workflow_recovery(self, temp_brain):
        """Test workflow recovery from failures"""
        with patch('graph.create_workflow_graph') as mock_graph:
            # First call fails, second succeeds
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.side_effect = [
                Exception("Workflow failed"),
                {'final_result': 'Recovered successfully', 'status': 'success'}
            ]
            mock_graph.return_value = mock_graph_instance
            
            # Should handle failure gracefully
            try:
                result = run_workflow(
                    task="Test task",
                    brain=temp_brain,
                    mode="full",
                    repo_url=""
                )
                assert result is not None
            except Exception as e:
                assert "Workflow failed" in str(e)

class TestWorkflowPerformance:
    """Test workflow performance and optimization"""
    
    @pytest.mark.asyncio
    async def test_workflow_execution_time(self, temp_brain):
        """Test workflow execution time"""
        import time
        
        with patch('graph.create_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Workflow completed',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            start_time = time.time()
            result = run_workflow(
                task="Simple task",
                brain=temp_brain,
                mode="quick",
                repo_url=""
            )
            end_time = time.time()
            
            # Should complete quickly with mocks
            assert end_time - start_time < 5.0
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_workflow_memory_efficiency(self, temp_brain):
        """Test workflow memory efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('graph.create_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Workflow completed',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            # Run multiple workflows
            for _ in range(5):
                result = run_workflow(
                    task=f"Task {_}",
                    brain=temp_brain,
                    mode="quick",
                    repo_url=""
                )
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 100 * 1024 * 1024  # 100MB

class TestWorkflowIntegration:
    """Test workflow integration with other components"""
    
    @pytest.mark.asyncio
    async def test_workflow_with_dspy(self, temp_brain):
        """Test workflow with DSPy integration"""
        with patch('graph.PromptFactory') as mock_factory:
            mock_module = MagicMock()
            mock_module.forward.return_value = "DSPy optimized output"
            mock_factory.return_value.get_module.return_value = mock_module
            
            with patch('graph.create_workflow_graph') as mock_graph:
                mock_graph_instance = MagicMock()
                mock_graph_instance.invoke.return_value = {
                    'final_result': 'DSPy workflow completed',
                    'status': 'success'
                }
                mock_graph.return_value = mock_graph_instance
                
                result = run_workflow(
                    task="Test DSPy integration",
                    brain=temp_brain,
                    mode="full",
                    repo_url=""
                )
                
                assert result is not None
                assert 'final_result' in result
    
    @pytest.mark.asyncio
    async def test_workflow_with_hf_routing(self, temp_brain):
        """Test workflow with HF routing integration"""
        with patch('graph.HFRoutingSystem') as mock_hf:
            mock_router = MagicMock()
            mock_router.route_task.return_value = ("Routed output", "HF_DeepSeek-V3")
            mock_hf.return_value = mock_router
            
            with patch('graph.create_workflow_graph') as mock_graph:
                mock_graph_instance = MagicMock()
                mock_graph_instance.invoke.return_value = {
                    'final_result': 'HF routed workflow completed',
                    'status': 'success'
                }
                mock_graph.return_value = mock_graph_instance
                
                result = run_workflow(
                    task="Test HF routing",
                    brain=temp_brain,
                    mode="full",
                    repo_url=""
                )
                
                assert result is not None
                assert 'final_result' in result
    
    @pytest.mark.asyncio
    async def test_workflow_with_git_integration(self, temp_brain, test_repo):
        """Test workflow with Git repository integration"""
        with patch('graph.create_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Git integrated workflow completed',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            repo_url = f"file://{test_repo}"
            result = run_workflow(
                task="Analyze Git repository",
                brain=temp_brain,
                mode="full",
                repo_url=repo_url
            )
            
            assert result is not None
            assert 'final_result' in result
            assert result['status'] == 'success'

class TestWorkflowQuality:
    """Test workflow output quality and validation"""
    
    @pytest.mark.asyncio
    async def test_workflow_output_completeness(self, temp_brain):
        """Test that workflow produces complete outputs"""
        with patch('graph.create_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Complete workflow result',
                'status': 'success',
                'research_results': [{'source': 'test', 'content': 'Research'}],
                'plan': {'phases': ['test'], 'tasks': ['test']},
                'code_files': [{'filename': 'test.py', 'content': 'test'}],
                'review_feedback': [{'type': 'quality', 'message': 'Good'}],
                'deployment_status': {'status': 'ready'}
            }
            mock_graph.return_value = mock_graph_instance
            
            result = run_workflow(
                task="Test complete workflow",
                brain=temp_brain,
                mode="full",
                repo_url=""
            )
            
            assert result is not None
            assert 'final_result' in result
            assert 'status' in result
            assert result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_workflow_output_consistency(self, temp_brain):
        """Test workflow output consistency"""
        with patch('graph.create_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Consistent result',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            results = []
            for _ in range(3):
                result = run_workflow(
                    task="Test consistency",
                    brain=temp_brain,
                    mode="quick",
                    repo_url=""
                )
                results.append(result)
            
            # All results should be consistent
            assert all(result['status'] == 'success' for result in results)
            assert all('final_result' in result for result in results) 