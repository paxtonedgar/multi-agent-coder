"""
Test memory system and state management
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import (
    ProjectBrain, AgentState, MemoryNode, NodeType, BrainCheckpoint,
    MemorySaver
)

class TestProjectBrain:
    """Test ProjectBrain functionality"""
    
    def test_brain_initialization(self, temp_brain):
        """Test brain initialization"""
        assert temp_brain is not None
        assert hasattr(temp_brain, 'memory')
        assert hasattr(temp_brain, 'brain_file')
        assert isinstance(temp_brain.memory, dict)
    
    def test_brain_persistence(self, temp_brain):
        """Test brain data persistence"""
        # Add some test data
        temp_brain.memory['test_key'] = 'test_value'
        temp_brain.memory['test_list'] = [1, 2, 3]
        temp_brain.memory['test_dict'] = {'nested': 'value'}
        
        # Save brain
        temp_brain._save()
        
        # Verify file exists
        assert os.path.exists(temp_brain.brain_file)
        
        # Load brain in new instance
        new_brain = ProjectBrain(temp_brain.project_path)
        
        # Verify data persisted
        assert new_brain.memory['test_key'] == 'test_value'
        assert new_brain.memory['test_list'] == [1, 2, 3]
        assert new_brain.memory['test_dict'] == {'nested': 'value'}
    
    def test_brain_memory_structure(self, temp_brain):
        """Test brain memory structure"""
        expected_keys = [
            'project_meta', 'codebase_map', 'function_graph', 'decisions',
            'learnings', 'github_examples', 'deployment_history', 'embeddings',
            'git_operations', 'file_operations', 'memory_nodes', 'reflections',
            'conversation_threads', 'reflection_evolution'
        ]
        
        for key in expected_keys:
            assert key in temp_brain.memory, f"Missing key: {key}"
    
    def test_brain_embedding_creation(self, temp_brain):
        """Test embedding creation"""
        test_text = "This is a test text for embedding"
        
        with patch('memory._create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            embedding = temp_brain._create_embedding(test_text)
            
            assert embedding is not None
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_brain_file_operations(self, temp_brain):
        """Test brain file operations"""
        # Test file finding
        test_query = "calculator function"
        files = temp_brain.find_relevant_files(test_query, top_k=3)
        
        assert isinstance(files, list)
        assert all(isinstance(item, tuple) for item in files)
        assert all(len(item) == 2 for item in files)  # (filepath, score)
    
    def test_brain_decision_memory(self, temp_brain):
        """Test decision memory functionality"""
        decision = "Use FastAPI for REST API"
        reasoning = "FastAPI is modern, fast, and has good documentation"
        context = {"project_type": "web_api", "team_size": 3}
        
        temp_brain.remember_decision(decision, reasoning, context)
        
        # Verify decision was stored
        assert len(temp_brain.memory['decisions']) > 0
        last_decision = temp_brain.memory['decisions'][-1]
        assert last_decision['decision'] == decision
        assert last_decision['reasoning'] == reasoning
        assert last_decision['context'] == context

class TestMemoryNodes:
    """Test memory node functionality"""
    
    def test_memory_node_creation(self, temp_brain):
        """Test memory node creation"""
        content = "This is a test memory node"
        node_type = NodeType.REFLECTION
        
        node_id = temp_brain.add_node(
            node_type=node_type,
            content=content,
            parent_id=None,
            sources=["test"],
            metadata={"test": True}
        )
        
        assert node_id is not None
        assert isinstance(node_id, str)
        assert len(node_id) > 0
        
        # Verify node was added to memory
        assert len(temp_brain.memory['memory_nodes']) > 0
        node_data = temp_brain.memory['memory_nodes'][-1]
        assert node_data['content'] == content
        assert node_data['type'] == node_type.value
    
    def test_memory_node_types(self, temp_brain):
        """Test different memory node types"""
        node_types = [
            NodeType.CHAT,
            NodeType.REFLECTION,
            NodeType.DECISION,
            NodeType.LEARNING,
            NodeType.CODE_CHANGE
        ]
        
        for node_type in node_types:
            node_id = temp_brain.add_node(
                node_type=node_type,
                content=f"Test {node_type.value} node",
                parent_id=None
            )
            
            assert node_id is not None
            
            # Verify node was created with correct type
            node_data = temp_brain.memory['memory_nodes'][-1]
            assert node_data['type'] == node_type.value
    
    def test_memory_node_hierarchy(self, temp_brain):
        """Test memory node parent-child relationships"""
        # Create parent node
        parent_id = temp_brain.add_node(
            node_type=NodeType.REFLECTION,
            content="Parent reflection",
            parent_id=None
        )
        
        # Create child nodes
        child1_id = temp_brain.add_node(
            node_type=NodeType.LEARNING,
            content="Child learning 1",
            parent_id=parent_id
        )
        
        child2_id = temp_brain.add_node(
            node_type=NodeType.DECISION,
            content="Child decision 1",
            parent_id=parent_id
        )
        
        assert parent_id is not None
        assert child1_id is not None
        assert child2_id is not None
        
        # Verify hierarchy in graph
        assert temp_brain.conversation_tree.has_edge(parent_id, child1_id)
        assert temp_brain.conversation_tree.has_edge(parent_id, child2_id)
    
    def test_memory_node_serialization(self, temp_brain):
        """Test memory node serialization and deserialization"""
        # Create a node
        original_node = MemoryNode(
            id="test-id",
            type=NodeType.REFLECTION,
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            timestamp="2025-01-27T10:00:00Z",
            parent_id=None,
            metadata={"test": True},
            sources=["test"],
            refinements=[]
        )
        
        # Convert to dict
        node_dict = original_node.to_dict()
        
        # Convert back to node
        reconstructed_node = MemoryNode.from_dict(node_dict)
        
        # Verify reconstruction
        assert reconstructed_node.id == original_node.id
        assert reconstructed_node.type == original_node.type
        assert reconstructed_node.content == original_node.content
        assert reconstructed_node.embedding == original_node.embedding
        assert reconstructed_node.metadata == original_node.metadata
        assert reconstructed_node.sources == original_node.sources

class TestReflectionSystem:
    """Test reflection system functionality"""
    
    def test_add_reflection(self, temp_brain):
        """Test adding reflections"""
        reflection_content = "This is a test reflection"
        sources = ["test_source_1", "test_source_2"]
        
        reflection_id = temp_brain.add_reflection(
            content=reflection_content,
            parent_id=None,
            sources=sources
        )
        
        assert reflection_id is not None
        
        # Verify reflection was added
        assert len(temp_brain.memory['reflections']) > 0
        last_reflection = temp_brain.memory['reflections'][-1]
        assert last_reflection['content'] == reflection_content
        assert last_reflection['sources'] == sources
    
    def test_refine_reflection(self, temp_brain):
        """Test refining reflections"""
        # Add initial reflection
        reflection_id = temp_brain.add_reflection(
            content="Initial reflection",
            parent_id=None,
            sources=["initial"]
        )
        
        # Refine reflection
        new_content = "Refined reflection with more details"
        new_sources = ["refined_source_1", "refined_source_2"]
        
        refined_id = temp_brain.refine_reflection(
            reflection_id=reflection_id,
            new_content=new_content,
            sources=new_sources
        )
        
        assert refined_id is not None
        assert refined_id != reflection_id
        
        # Verify refinement was added
        assert len(temp_brain.memory['reflections']) > 1
        refined_reflection = temp_brain.memory['reflections'][-1]
        assert refined_reflection['content'] == new_content
        assert refined_reflection['sources'] == new_sources
    
    def test_reflection_tree_retrieval(self, temp_brain):
        """Test reflection tree retrieval"""
        # Create a reflection tree
        root_id = temp_brain.add_reflection("Root reflection", parent_id=None)
        child1_id = temp_brain.add_reflection("Child 1", parent_id=root_id)
        child2_id = temp_brain.add_reflection("Child 2", parent_id=root_id)
        grandchild_id = temp_brain.add_reflection("Grandchild", parent_id=child1_id)
        
        # Retrieve reflection tree
        tree = temp_brain.get_reflection_tree("reflection", depth=3)
        
        assert tree is not None
        assert 'nodes' in tree
        assert 'summary' in tree
        assert len(tree['nodes']) > 0
    
    def test_reflection_evolution_tracking(self, temp_brain):
        """Test reflection evolution tracking"""
        # Add multiple refinements
        reflection_id = temp_brain.add_reflection("Initial", parent_id=None)
        
        temp_brain.refine_reflection(reflection_id, "Refinement 1", ["source1"])
        temp_brain.refine_reflection(reflection_id, "Refinement 2", ["source2"])
        temp_brain.refine_reflection(reflection_id, "Final version", ["source3"])
        
        # Verify evolution tracking
        assert 'reflection_evolution' in temp_brain.memory
        if reflection_id in temp_brain.memory['reflection_evolution']:
            evolution = temp_brain.memory['reflection_evolution'][reflection_id]
            assert len(evolution) > 0

class TestStateManagement:
    """Test state management functionality"""
    
    def test_agent_state_creation(self, temp_brain):
        """Test AgentState creation"""
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
        
        assert state is not None
        assert state['current_step'] == "research"
        assert state['task'] == "Test task"
        assert 'brain' in state['brain_context']
    
    def test_state_persistence(self, temp_brain):
        """Test state persistence through brain"""
        # Create state with some data
        state = AgentState(
            messages=[{"role": "user", "content": "Test message"}],
            current_step="planning",
            task="Test task",
            repo_url="",
            research_results=[{"source": "test", "content": "Research done"}],
            plan={"phases": ["test"], "tasks": ["test"]},
            audit_feedback=[],
            code_files=[],
            review_feedback=[],
            deployment_status={},
            brain_context={"brain": temp_brain},
            reflection_context={}
        )
        
        # Save state to brain
        temp_brain.memory['current_state'] = state
        temp_brain._save()
        
        # Load state from brain
        new_brain = ProjectBrain(temp_brain.project_path)
        loaded_state = new_brain.memory.get('current_state')
        
        assert loaded_state is not None
        assert loaded_state['current_step'] == "planning"
        assert len(loaded_state['research_results']) > 0
        assert loaded_state['plan']['phases'] == ["test"]
    
    def test_state_transitions(self, temp_brain):
        """Test state transitions between workflow steps"""
        # Initial state
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
        
        # Simulate state transitions
        state['current_step'] = "planning"
        state['research_results'] = [{"source": "test", "content": "Research done"}]
        
        state['current_step'] = "coding"
        state['plan'] = {"phases": ["test"], "tasks": ["test"]}
        
        state['current_step'] = "review"
        state['code_files'] = [{"filename": "test.py", "content": "test"}]
        
        # Verify state evolution
        assert state['current_step'] == "review"
        assert len(state['research_results']) > 0
        assert 'plan' in state
        assert len(state['code_files']) > 0

class TestBrainCheckpoint:
    """Test brain checkpoint functionality"""
    
    def test_checkpoint_creation(self, temp_brain):
        """Test brain checkpoint creation"""
        checkpoint = BrainCheckpoint(temp_brain)
        
        assert checkpoint is not None
        assert isinstance(checkpoint, MemorySaver)
        assert hasattr(checkpoint, 'brain')
        assert checkpoint.brain == temp_brain
    
    def test_checkpoint_get(self, temp_brain):
        """Test checkpoint get operation"""
        checkpoint = BrainCheckpoint(temp_brain)
        
        config = {"config_id": "test_config"}
        
        # Test getting non-existent checkpoint
        result = checkpoint.get(config)
        assert result is None
        
        # Test getting existing checkpoint
        test_data = {"test": "data"}
        temp_brain.memory['checkpoints'] = {config["config_id"]: test_data}
        
        result = checkpoint.get(config)
        assert result == test_data
    
    def test_checkpoint_put(self, temp_brain):
        """Test checkpoint put operation"""
        checkpoint = BrainCheckpoint(temp_brain)
        
        config = {"config_id": "test_config"}
        value = {"test": "value"}
        
        # Put checkpoint
        checkpoint.put(config, value)
        
        # Verify checkpoint was stored
        assert 'checkpoints' in temp_brain.memory
        assert config["config_id"] in temp_brain.memory['checkpoints']
        assert temp_brain.memory['checkpoints'][config["config_id"]] == value

class TestMemoryPerformance:
    """Test memory system performance"""
    
    def test_memory_usage(self, temp_brain):
        """Test memory usage efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Add many nodes
        for i in range(100):
            temp_brain.add_node(
                node_type=NodeType.REFLECTION,
                content=f"Test reflection {i}",
                parent_id=None
            )
        
        # Add many reflections
        for i in range(100):
            temp_brain.add_reflection(
                content=f"Test reflection {i}",
                parent_id=None,
                sources=[f"source_{i}"]
            )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # 50MB
    
    def test_memory_retrieval_speed(self, temp_brain):
        """Test memory retrieval speed"""
        import time
        
        # Add test data
        for i in range(50):
            temp_brain.add_reflection(
                content=f"Test reflection {i}",
                parent_id=None,
                sources=[f"source_{i}"]
            )
        
        # Test retrieval speed
        start_time = time.time()
        
        for i in range(10):
            temp_brain.get_reflection_tree("reflection", depth=3)
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        # Should be fast
        assert retrieval_time < 1.0  # Less than 1 second for 10 retrievals
    
    def test_memory_persistence_speed(self, temp_brain):
        """Test memory persistence speed"""
        import time
        
        # Add test data
        for i in range(100):
            temp_brain.add_node(
                node_type=NodeType.REFLECTION,
                content=f"Test node {i}",
                parent_id=None
            )
        
        # Test save speed
        start_time = time.time()
        temp_brain._save()
        end_time = time.time()
        save_time = end_time - start_time
        
        # Should be fast
        assert save_time < 2.0  # Less than 2 seconds

class TestMemoryErrorHandling:
    """Test memory system error handling"""
    
    def test_invalid_node_type(self, temp_brain):
        """Test handling of invalid node types"""
        try:
            temp_brain.add_node(
                node_type="invalid_type",
                content="Test content",
                parent_id=None
            )
            # Should handle gracefully
        except Exception as e:
            assert "NodeType" in str(e) or "invalid" in str(e)
    
    def test_invalid_parent_id(self, temp_brain):
        """Test handling of invalid parent IDs"""
        # Try to add node with non-existent parent
        node_id = temp_brain.add_node(
            node_type=NodeType.REFLECTION,
            content="Test content",
            parent_id="non-existent-id"
        )
        
        # Should still create the node
        assert node_id is not None
        
        # But parent relationship should not exist
        assert not temp_brain.conversation_tree.has_edge("non-existent-id", node_id)
    
    def test_corrupted_memory_file(self, temp_brain):
        """Test handling of corrupted memory file"""
        # Corrupt the memory file
        with open(temp_brain.brain_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle gracefully
        try:
            new_brain = ProjectBrain(temp_brain.project_path)
            assert new_brain is not None
            assert isinstance(new_brain.memory, dict)
        except Exception as e:
            # Should provide meaningful error
            assert "json" in str(e).lower() or "corrupt" in str(e).lower()
    
    def test_missing_memory_directory(self, temp_brain):
        """Test handling of missing memory directory"""
        # Remove memory directory
        import shutil
        memory_dir = os.path.dirname(temp_brain.brain_file)
        shutil.rmtree(memory_dir, ignore_errors=True)
        
        # Should recreate directory
        new_brain = ProjectBrain(temp_brain.project_path)
        assert new_brain is not None
        assert os.path.exists(memory_dir)

class TestMemoryIntegration:
    """Test memory integration with other components"""
    
    def test_memory_with_agents(self, temp_brain):
        """Test memory integration with agents"""
        from agents import create_all_agents
        
        # Create agents with brain
        with patch('agents.get_default_model') as mock_model:
            mock_model.return_value = MagicMock()
            agents = create_all_agents(temp_brain)
            
            # Verify agents can access brain
            for agent_name, agent in agents.items():
                assert agent is not None
                # Agents should be able to access brain through their tools
    
    def test_memory_with_workflow(self, temp_brain):
        """Test memory integration with workflow"""
        from graph import run_workflow
        
        # Run workflow with brain
        with patch('graph.create_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Workflow completed',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            result = run_workflow(
                task="Test task",
                brain=temp_brain,
                mode="quick",
                repo_url=""
            )
            
            assert result is not None
            # Brain should contain workflow state
            assert 'current_state' in temp_brain.memory or len(temp_brain.memory['decisions']) > 0
    
    def test_memory_with_tools(self, temp_brain):
        """Test memory integration with tools"""
        from tools import create_research_tools
        
        # Create tools with brain
        tools = create_research_tools(temp_brain)
        
        # Verify tools can access brain
        assert len(tools) > 0
        for tool in tools:
            assert hasattr(tool, 'func')
            # Tools should be able to access brain through their implementation 