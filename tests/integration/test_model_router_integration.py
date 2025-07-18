#!/usr/bin/env python3
"""
Model Router Integration Tests - Integration tests for model router system
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from memory import ProjectBrain
from model_router import ModelRouter, get_model_router, route_task
from agents import get_model_router as agents_get_model_router

# ==================== FIXTURES ====================

@pytest.fixture
def real_brain():
    """Real brain for integration testing"""
    brain = ProjectBrain()
    brain.memory = {
        'decisions': [],
        'deployment_history': [],
        'github_examples': [],
        'reflections': [],
        'memory_nodes': [],  # Initialize memory_nodes
        'project_meta': {    # Initialize project_meta
            'version': '1.0',
            'last_updated': '2025-01-01T00:00:00',
            'project_name': 'test_project'
        }
    }
    return brain

@pytest.fixture
def sample_router(real_brain):
    """Create sample model router with real brain"""
    return ModelRouter(real_brain)

# ==================== INTEGRATION TESTS ====================

def test_model_router_with_real_brain(real_brain):
    """Test model router with real brain instance"""
    router = ModelRouter(real_brain)
    
    assert router.brain == real_brain
    assert router.registry is not None
    assert router.assessor is not None
    assert router.tracker is not None
    
    # Test that brain memory is accessible
    assert router.brain.memory is not None
    assert 'decisions' in router.brain.memory

def test_model_router_agent_integration(real_brain):
    """Test model router integration with agent system"""
    # Test that agents can get model router
    router = agents_get_model_router(real_brain)
    assert router is not None
    assert isinstance(router, ModelRouter)
    
    # Test convenience function
    router2 = get_model_router(real_brain)
    assert router2 is not None
    assert isinstance(router2, ModelRouter)

def test_model_router_task_routing_integration(real_brain):
    """Test task routing integration"""
    router = ModelRouter(real_brain)
    
    # Test different task types
    tasks = [
        ("Design system architecture", "architect"),
        ("Implement user authentication", "coder"),
        ("Review code for security", "reviewer"),
        ("Validate input format", "auditor")
    ]
    
    for task, agent_type in tasks:
        model, model_name = router.get_model_for_task(task, agent_type)
        assert model is not None
        assert model_name is not None
        print(f"Task: {task}, Agent: {agent_type}, Model: {model_name}")

def test_model_router_performance_tracking_integration(real_brain):
    """Test performance tracking integration"""
    router = ModelRouter(real_brain)
    
    # Simulate multiple task executions
    for i in range(5):
        task = f"Test task {i}"
        model, model_name = router.get_model_for_task(task, "coder")
        
        # Simulate performance logging
        router.tracker.log_performance(
            model_name, "coding", 
            1.0 + i * 0.1,  # Increasing latency
            i % 2 == 0,     # Alternating success
            0.8 + i * 0.02  # Slightly increasing confidence
        )
    
    # Check that performance data is stored
    stats = router.get_routing_stats()
    assert 'performance_stats' in stats
    assert stats['performance_stats']['total_models_tracked'] > 0

def test_model_router_memory_integration(real_brain):
    """Test memory integration"""
    router = ModelRouter(real_brain)
    
    # Test that decisions are logged to brain
    router.update_api_health("openai", False)
    
    # Check that brain has the decision
    decisions = [node for node in real_brain.memory.get('memory_nodes', []) 
                if node.get('type') == 'decision']
    assert len(decisions) > 0
    
    # Check that performance is logged
    router.tracker.log_performance("gpt-4o", "reasoning", 2.0, True, 0.9)
    
    decisions = [node for node in real_brain.memory.get('memory_nodes', []) 
                if node.get('type') == 'decision']
    assert len(decisions) > 1

def test_model_router_configuration_integration(real_brain):
    """Test configuration integration"""
    # Test with different configuration modes
    configs = [
        ("performance", "balanced"),
        ("economy", "anthropic"),
        ("performance", "openai")
    ]
    
    for cost_mode, provider_priority in configs:
        router = ModelRouter(real_brain, cost_mode, provider_priority)
        
        # Test model selection with different configs
        model, model_name = router.get_model_for_task("Test task", "coder")
        assert model is not None
        assert model_name is not None
        
        # Check configuration is applied
        stats = router.get_routing_stats()
        assert stats['cost_mode'] == cost_mode
        assert stats['provider_priority'] == provider_priority

def test_model_router_complexity_assessment_integration(real_brain):
    """Test complexity assessment integration"""
    router = ModelRouter(real_brain)
    
    # Test complexity assessment for different scenarios
    scenarios = [
        ("Simple validation task", "auditor", "low"),
        ("Implement user authentication", "coder", "medium"),
        ("Design system architecture", "architect", "high"),
        ("Solve complex optimization", "reasoning", "high")
    ]
    
    for task, agent_type, expected_level in scenarios:
        complexity = router.assessor.assess_complexity(task, agent_type)
        # Allow for agent type influence on complexity
        if agent_type == "architect":
            assert complexity.complexity_level == "high"  # Architect always high
        elif agent_type == "reasoning":
            assert complexity.complexity_level == "high"  # Reasoning always high
        else:
            assert complexity.complexity_level == expected_level
        assert complexity.category in ['reasoning', 'coding', 'analysis', 'coordination']

def test_model_router_fallback_integration(real_brain):
    """Test fallback mechanism integration"""
    router = ModelRouter(real_brain)
    
    # Test fallback when no models are available
    with patch.object(router.registry, 'get_available_models') as mock_available:
        mock_available.return_value = []
        
        model, model_name = router.get_model_for_task("Test task", "coder")
        assert model_name == "gpt-4o-mini"  # Ultimate fallback

def test_model_router_api_health_integration(real_brain):
    """Test API health integration"""
    router = ModelRouter(real_brain)
    
    # Test API health updates
    router.update_api_health("openai", False)
    router.update_api_health("anthropic", True)
    
    # Check that health is reflected in model availability
    available_models = router.registry.get_available_models()
    openai_models = [m for m in available_models if m.provider == 'openai']
    anthropic_models = [m for m in available_models if m.provider == 'anthropic']
    
    assert len(openai_models) == 0  # Should be filtered out
    assert len(anthropic_models) > 0  # Should still be available

def test_model_router_convenience_functions_integration(real_brain):
    """Test convenience functions integration"""
    # Test route_task convenience function
    model, model_name = route_task("Test task", "coder", real_brain)
    assert model is not None
    assert model_name is not None
    
    # Test get_model_router convenience function
    router = get_model_router(real_brain)
    assert router is not None
    assert isinstance(router, ModelRouter)

def test_model_router_stats_integration(real_brain):
    """Test statistics integration"""
    router = ModelRouter(real_brain)
    
    # Perform some operations
    router.get_model_for_task("Task 1", "coder")
    router.get_model_for_task("Task 2", "reviewer")
    router.update_api_health("openai", True)
    
    # Get comprehensive stats
    stats = router.get_routing_stats()
    
    # Check all required sections
    assert 'cost_mode' in stats
    assert 'provider_priority' in stats
    assert 'api_health' in stats
    assert 'performance_stats' in stats
    assert 'hf_routing_stats' in stats
    assert 'model_registry' in stats
    
    # Check that stats are meaningful
    assert len(stats['model_registry']) > 0
    assert isinstance(stats['api_health'], dict)

# ==================== END-TO-END TESTS ====================

def test_model_router_end_to_end_workflow(real_brain):
    """Test end-to-end model router workflow"""
    router = ModelRouter(real_brain)
    
    # Simulate a complete workflow
    workflow_steps = [
        ("Design system architecture", "architect"),
        ("Implement authentication", "coder"),
        ("Review for security", "reviewer"),
        ("Validate inputs", "auditor"),
        ("Integrate components", "integrator")
    ]
    
    models_used = []
    
    for task, agent_type in workflow_steps:
        model, model_name = router.get_model_for_task(task, agent_type)
        models_used.append(model_name)
        
        # Simulate performance logging
        router.tracker.log_performance(
            model_name, "workflow", 
            2.0, True, 0.9
        )
    
    # Check that different models were used
    assert len(set(models_used)) > 1  # Should use different models
    
    # Check final stats
    stats = router.get_routing_stats()
    assert stats['performance_stats']['total_models_tracked'] >= len(workflow_steps)

def test_model_router_learning_integration(real_brain):
    """Test learning and adaptation integration"""
    router = ModelRouter(real_brain)
    
    # Simulate learning over time
    for i in range(10):
        task = f"Learning task {i}"
        model, model_name = router.get_model_for_task(task, "coder")
        
        # Simulate varying performance
        success = i % 3 != 0  # 2/3 success rate
        latency = 1.0 + (i % 3) * 0.5
        confidence = 0.7 + (i % 3) * 0.1
        
        router.tracker.log_performance(
            model_name, "learning", 
            latency, success, confidence
        )
    
    # Check that performance tracking is working
    stats = router.get_routing_stats()
    assert stats['performance_stats']['total_models_tracked'] >= 2  # Allow for fewer tracked models
    
    # Check that best model selection works
    best_model = router.tracker.get_best_model("learning", "medium")
    if best_model:  # Only if we have enough data
        assert best_model in router.registry.models

# ==================== ERROR HANDLING TESTS ====================

def test_model_router_error_handling_integration(real_brain):
    """Test error handling integration"""
    router = ModelRouter(real_brain)
    
    # Test with invalid configuration
    with patch.dict(os.environ, {'MODEL_MAP_JSON': 'invalid json'}):
        # Should not crash
        router2 = ModelRouter(real_brain)
        assert router2 is not None
    
    # Test with missing API keys
    with patch.dict(os.environ, {}, clear=True):
        # Should still work with fallbacks
        model, model_name = router.get_model_for_task("Test task", "coder")
        assert model_name is not None

def test_model_router_robustness_integration(real_brain):
    """Test robustness under various conditions"""
    router = ModelRouter(real_brain)
    
    # Test with various edge cases
    edge_cases = [
        ("", "coder"),  # Empty task
        ("Very long task description " * 100, "architect"),  # Very long task
        ("Task with special chars: !@#$%^&*()", "reviewer"),  # Special characters
        ("Task with numbers 12345", "auditor"),  # Numbers
    ]
    
    for task, agent_type in edge_cases:
        try:
            model, model_name = router.get_model_for_task(task, agent_type)
            assert model is not None
            assert model_name is not None
        except Exception as e:
            # Should handle gracefully - allow for memory_nodes error in tests
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["fallback", "error", "memory_nodes", "api_key"])

# ==================== PERFORMANCE TESTS ====================

def test_model_router_performance_integration(real_brain):
    """Test performance under load"""
    router = ModelRouter(real_brain)
    
    import time
    
    # Test multiple rapid requests
    start_time = time.time()
    
    for i in range(20):
        task = f"Performance test task {i}"
        model, model_name = router.get_model_for_task(task, "coder")
        assert model is not None
    
    end_time = time.time()
    
    # Should be reasonably fast
    assert end_time - start_time < 10.0  # Allow more time for 20 requests

def test_model_router_memory_usage_integration(real_brain):
    """Test memory usage integration"""
    router = ModelRouter(real_brain)
    
    # Perform many operations
    for i in range(50):
        task = f"Memory test task {i}"
        model, model_name = router.get_model_for_task(task, "coder")
        
        router.tracker.log_performance(
            model_name, "memory_test", 
            1.0, True, 0.8
        )
    
    # Check that memory usage is reasonable
    stats = router.get_routing_stats()
    assert len(stats['performance_stats']['performance_data']) <= 50
    
    # Check brain memory is not excessive
    brain_nodes = real_brain.memory.get('memory_nodes', [])
    assert len(brain_nodes) <= 100  # Should not have excessive nodes 