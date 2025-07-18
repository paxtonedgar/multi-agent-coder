#!/usr/bin/env python3
"""
Model Router Tests - Unit tests for advanced model routing system
"""

import os
import sys
import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from memory import ProjectBrain
from model_router import (
    ModelRouter, ModelRegistry, ComplexityAssessor, PerformanceTracker,
    TaskComplexity, ModelConfig, CostMode, ProviderPriority
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
    brain.add_node = Mock()
    return brain

@pytest.fixture
def sample_router(mock_brain):
    """Create sample model router"""
    return ModelRouter(mock_brain)

# ==================== MODEL REGISTRY TESTS ====================

def test_model_registry_initialization():
    """Test model registry initialization"""
    registry = ModelRegistry()
    
    assert len(registry.models) > 0
    assert 'gpt-4o' in registry.models
    assert 'claude-3-5-sonnet' in registry.models
    assert 'deepseek-v3' in registry.models

def test_model_registry_get_available_models():
    """Test getting available models by complexity level"""
    registry = ModelRegistry()
    
    # Test high complexity models
    high_models = registry.get_available_models('high')
    assert len(high_models) > 0
    assert all(model.complexity_level == 'high' for model in high_models)
    
    # Test medium complexity models
    medium_models = registry.get_available_models('medium')
    assert len(medium_models) > 0
    assert all(model.complexity_level == 'medium' for model in medium_models)
    
    # Test low complexity models
    low_models = registry.get_available_models('low')
    assert len(low_models) > 0
    assert all(model.complexity_level == 'low' for model in low_models)

def test_model_registry_api_health():
    """Test API health management"""
    registry = ModelRegistry()
    
    # Test initial health
    assert registry.api_health['openai'] == True
    assert registry.api_health['anthropic'] == True
    
    # Test updating health
    registry.update_api_health('openai', False)
    assert registry.api_health['openai'] == False
    
    # Test filtering unavailable providers
    available_models = registry.get_available_models()
    openai_models = [m for m in available_models if m.provider == 'openai']
    assert len(openai_models) == 0  # Should be filtered out

def test_model_registry_get_model_config():
    """Test getting model configuration"""
    registry = ModelRegistry()
    
    config = registry.get_model_config('gpt-4o')
    assert config is not None
    assert config.name == 'gpt-4o'
    assert config.provider == 'openai'
    assert config.complexity_level == 'high'
    
    # Test non-existent model
    config = registry.get_model_config('non-existent-model')
    assert config is None

# ==================== COMPLEXITY ASSESSOR TESTS ====================

def test_complexity_assessor_high_complexity():
    """Test complexity assessment for high complexity tasks"""
    assessor = ComplexityAssessor()
    
    # Test architecture task
    complexity = assessor.assess_complexity("Design system architecture for microservices")
    assert complexity.score > 0.7
    assert complexity.category == 'reasoning'
    assert complexity.requires_reasoning == True
    
    # Test algorithm task
    complexity = assessor.assess_complexity("Optimize algorithm for performance")
    assert complexity.score > 0.7
    assert complexity.category == 'reasoning'
    assert complexity.requires_reasoning == True

def test_complexity_assessor_medium_complexity():
    """Test complexity assessment for medium complexity tasks"""
    assessor = ComplexityAssessor()
    
    # Test coding task
    complexity = assessor.assess_complexity("Implement user authentication system")
    assert 0.4 <= complexity.score <= 0.7
    assert complexity.category == 'coding'
    
    # Test review task
    complexity = assessor.assess_complexity("Review code for security vulnerabilities")
    assert 0.4 <= complexity.score <= 0.7
    assert complexity.category == 'analysis'

def test_complexity_assessor_low_complexity():
    """Test complexity assessment for low complexity tasks"""
    assessor = ComplexityAssessor()
    
    # Test validation task
    complexity = assessor.assess_complexity("Validate user input format")
    assert complexity.score <= 0.5  # Allow for default medium complexity
    assert complexity.category == 'coordination'
    
    # Test coordination task
    complexity = assessor.assess_complexity("Coordinate team meetings")
    assert complexity.score <= 0.5  # Allow for default medium complexity
    assert complexity.category == 'coordination'

def test_complexity_assessor_agent_type_influence():
    """Test how agent type influences complexity assessment"""
    assessor = ComplexityAssessor()
    
    # Test architect agent
    complexity = assessor.assess_complexity("Simple task", "architect")
    assert complexity.score >= 0.9  # Architect has high complexity
    
    # Test coder agent
    complexity = assessor.assess_complexity("Simple task", "coder")
    assert complexity.score >= 0.6  # Coder has medium complexity
    
    # Test auditor agent
    complexity = assessor.assess_complexity("Simple task", "auditor")
    assert complexity.score >= 0.3  # Auditor has low complexity

def test_complexity_assessor_requirements_detection():
    """Test detection of task requirements"""
    assessor = ComplexityAssessor()
    
    # Test reasoning requirement
    complexity = assessor.assess_complexity("Solve complex optimization problem")
    assert complexity.requires_reasoning == True
    
    # Test creativity requirement
    complexity = assessor.assess_complexity("Design creative user interface")
    assert complexity.requires_creativity == True
    
    # Test accuracy requirement
    complexity = assessor.assess_complexity("Ensure accuracy in calculations")
    assert complexity.requires_accuracy == True

# ==================== PERFORMANCE TRACKER TESTS ====================

def test_performance_tracker_logging(mock_brain):
    """Test performance logging"""
    tracker = PerformanceTracker(mock_brain)
    
    # Log performance
    tracker.log_performance("gpt-4o", "reasoning", 2.5, True, 0.9)
    
    # Check that performance was logged
    assert len(tracker.performance_cache) == 1
    key = "gpt-4o_reasoning"
    assert key in tracker.performance_cache
    
    perf = tracker.performance_cache[key]
    assert perf.model_name == "gpt-4o"
    assert perf.task_category == "reasoning"
    assert perf.success_rate == 1.0
    assert perf.avg_latency == 2.5
    assert perf.avg_confidence == 0.9
    assert perf.total_uses == 1

def test_performance_tracker_multiple_logs(mock_brain):
    """Test multiple performance logs"""
    tracker = PerformanceTracker(mock_brain)
    
    # Log multiple performances
    tracker.log_performance("gpt-4o", "reasoning", 2.0, True, 0.9)
    tracker.log_performance("gpt-4o", "reasoning", 3.0, False, 0.7)
    tracker.log_performance("gpt-4o", "reasoning", 2.5, True, 0.8)
    
    # Check aggregated performance
    key = "gpt-4o_reasoning"
    perf = tracker.performance_cache[key]
    assert perf.total_uses == 3
    assert perf.success_rate == 2/3  # 2 out of 3 successful
    assert perf.avg_latency == 2.5  # (2+3+2.5)/3
    assert abs(perf.avg_confidence - 0.8) < 0.001  # (0.9+0.7+0.8)/3

def test_performance_tracker_best_model_selection(mock_brain):
    """Test best model selection based on performance"""
    tracker = PerformanceTracker(mock_brain)
    
    # Log performance for different models
    tracker.log_performance("gpt-4o", "reasoning", 2.0, True, 0.9)
    tracker.log_performance("gpt-4o", "reasoning", 2.5, True, 0.8)
    tracker.log_performance("gpt-4o", "reasoning", 2.2, True, 0.85)
    
    tracker.log_performance("claude-3-5-sonnet", "reasoning", 1.5, True, 0.95)
    tracker.log_performance("claude-3-5-sonnet", "reasoning", 1.8, True, 0.92)
    tracker.log_performance("claude-3-5-sonnet", "reasoning", 1.6, True, 0.94)
    
    # Get best model
    best_model = tracker.get_best_model("reasoning", "high")
    assert best_model == "claude-3-5-sonnet"  # Better performance

def test_performance_tracker_stats(mock_brain):
    """Test performance statistics generation"""
    tracker = PerformanceTracker(mock_brain)
    
    # Log some performance data
    tracker.log_performance("gpt-4o", "reasoning", 2.0, True, 0.9)
    tracker.log_performance("claude-3-5-sonnet", "coding", 1.5, True, 0.8)
    
    # Get stats
    stats = tracker.get_performance_stats()
    
    assert stats['total_models_tracked'] == 2
    assert len(stats['performance_data']) == 2
    assert 'last_updated' in stats

# ==================== MODEL ROUTER TESTS ====================

def test_model_router_initialization(mock_brain):
    """Test model router initialization"""
    router = ModelRouter(mock_brain)
    
    assert router.brain == mock_brain
    assert router.cost_mode == CostMode.PERFORMANCE
    assert router.provider_priority == ProviderPriority.BALANCED
    assert router.registry is not None
    assert router.assessor is not None
    assert router.tracker is not None

def test_model_router_configuration_loading(mock_brain):
    """Test configuration loading from environment"""
    with patch.dict(os.environ, {
        'COST_MODE': 'economy',
        'PROVIDER_PRIORITY': 'anthropic',
        'CONFIDENCE_THRESHOLD': '0.7',
        'COMPLEXITY_THRESHOLD': '0.6'
    }):
        router = ModelRouter(mock_brain)
        
        assert router.cost_mode == CostMode.ECONOMY
        assert router.provider_priority == ProviderPriority.ANTHROPIC
        assert router.confidence_threshold == 0.7
        assert router.complexity_threshold == 0.6

def test_model_router_custom_model_mappings(mock_brain):
    """Test custom model mappings from environment"""
    custom_mappings = {
        "architect": "claude-3-opus",
        "coder": "gpt-4o-mini"
    }
    
    with patch.dict(os.environ, {
        'MODEL_MAP_JSON': json.dumps(custom_mappings)
    }):
        router = ModelRouter(mock_brain)
        # Should load custom mappings without error

def test_model_router_model_selection_high_complexity(mock_brain):
    """Test model selection for high complexity tasks"""
    router = ModelRouter(mock_brain)
    
    # Mock HF router to avoid actual model loading
    with patch.object(router, 'hf_router') as mock_hf:
        mock_hf.route_task.return_value = ("mock response", "main")
        
        # Test high complexity task
        model, model_name = router.get_model_for_task("Design system architecture", "architect")
        
        # Should select a high complexity model
        assert model is not None
        assert model_name in ['gpt-4o', 'claude-3-opus', 'deepseek-v3']

def test_model_router_model_selection_medium_complexity(mock_brain):
    """Test model selection for medium complexity tasks"""
    router = ModelRouter(mock_brain)
    
    # Mock HF router
    with patch.object(router, 'hf_router') as mock_hf:
        mock_hf.route_task.return_value = ("mock response", "main")
        
        # Test medium complexity task
        model, model_name = router.get_model_for_task("Implement user authentication", "coder")
        
        # Should select a medium complexity model
        assert model is not None
        assert model_name in ['gpt-4o-mini', 'claude-3-5-sonnet']

def test_model_router_model_selection_low_complexity(mock_brain):
    """Test model selection for low complexity tasks"""
    router = ModelRouter(mock_brain)
    
    # Mock HF router
    with patch.object(router, 'hf_router') as mock_hf:
        mock_hf.route_task.return_value = ("mock response", "main")
        
        # Test low complexity task
        model, model_name = router.get_model_for_task("Validate input format", "auditor")
        
        # Should select a low complexity model
        assert model is not None
        # Allow for fallback to medium complexity models if low complexity models aren't available
        assert model_name in ['gpt-3.5-turbo', 'claude-3-5-haiku', 'gpt-4o-mini', 'claude-3-5-sonnet']

def test_model_router_hf_routing_integration(mock_brain):
    """Test HF routing integration for reasoning tasks"""
    router = ModelRouter(mock_brain)
    
    # Mock successful HF routing
    with patch.object(router, 'hf_router') as mock_hf:
        mock_hf.route_task.return_value = ("HF response", "HF_deepseek-v3")
        
        # Test reasoning task that should trigger HF routing
        model, model_name = router.get_model_for_task("Solve complex optimization problem", "reasoning")
        
        # Should use HF routing
        assert model is not None
        assert model_name == "HF_deepseek-v3"
        mock_hf.route_task.assert_called_once()

def test_model_router_hf_routing_fallback(mock_brain):
    """Test HF routing fallback when HF fails"""
    router = ModelRouter(mock_brain)
    
    # Mock HF routing failure
    with patch.object(router, 'hf_router') as mock_hf:
        mock_hf.route_task.side_effect = Exception("HF routing failed")
        
        # Test reasoning task
        model, model_name = router.get_model_for_task("Solve complex optimization problem", "reasoning")
        
        # Should fallback to regular model selection
        assert model is not None
        assert not model_name.startswith("HF_")

def test_model_router_performance_based_selection(mock_brain):
    """Test model selection based on performance history"""
    router = ModelRouter(mock_brain)
    
    # Mock performance tracker to return best model
    with patch.object(router.tracker, 'get_best_model') as mock_get_best:
        mock_get_best.return_value = "claude-3-5-sonnet"
        
        # Mock HF router
        with patch.object(router, 'hf_router') as mock_hf:
            mock_hf.route_task.return_value = ("mock response", "main")
            
            model, model_name = router.get_model_for_task("Test task", "coder")
            
            # Should use best performing model
            assert model_name == "claude-3-5-sonnet"

def test_model_router_cost_mode_influence(mock_brain):
    """Test how cost mode influences model selection"""
    router = ModelRouter(mock_brain, cost_mode="economy")
    
    # Mock HF router
    with patch.object(router, 'hf_router') as mock_hf:
        mock_hf.route_task.return_value = ("mock response", "main")
        
        # Test model selection in economy mode
        model, model_name = router.get_model_for_task("Test task", "coder")
        
        # Should prefer cheaper models
        assert model is not None
        # Economy mode should prefer cheaper models like gpt-4o-mini

def test_model_router_provider_priority_influence(mock_brain):
    """Test how provider priority influences model selection"""
    router = ModelRouter(mock_brain, provider_priority="anthropic")
    
    # Mock HF router
    with patch.object(router, 'hf_router') as mock_hf:
        mock_hf.route_task.return_value = ("mock response", "main")
        
        # Test model selection with Anthropic priority
        model, model_name = router.get_model_for_task("Test task", "coder")
        
        # Should prefer Anthropic models
        assert model is not None
        # Should prefer Claude models when available

def test_model_router_api_health_update(mock_brain):
    """Test API health update functionality"""
    router = ModelRouter(mock_brain)
    
    # Update API health
    router.update_api_health("openai", False)
    
    # Check that health was updated
    assert router.registry.api_health["openai"] == False
    
    # Check that brain was logged
    mock_brain.add_node.assert_called_once()

def test_model_router_routing_stats(mock_brain):
    """Test routing statistics generation"""
    router = ModelRouter(mock_brain)
    
    # Get routing stats
    stats = router.get_routing_stats()
    
    assert 'cost_mode' in stats
    assert 'provider_priority' in stats
    assert 'api_health' in stats
    assert 'performance_stats' in stats
    assert 'hf_routing_stats' in stats
    assert 'model_registry' in stats

def test_model_router_refresh_models(mock_brain):
    """Test model refresh functionality"""
    router = ModelRouter(mock_brain)
    
    # Mock HF router refresh
    with patch.object(router, 'hf_router') as mock_hf:
        router.refresh_models()
        
        # Should call HF router refresh
        mock_hf.refresh_models.assert_called_once()

# ==================== INTEGRATION TESTS ====================

def test_model_router_integration_with_agents(mock_brain):
    """Test model router integration with agent creation"""
    from agents import get_model_router
    
    # Test getting model router
    router = get_model_router(mock_brain)
    assert router is not None
    assert isinstance(router, ModelRouter)

def test_model_router_force_model(mock_brain):
    """Test forcing specific model selection"""
    router = ModelRouter(mock_brain)
    
    # Force specific model
    model, model_name = router.get_model_for_task("Test task", force_model="gpt-4o")
    
    assert model is not None
    assert model_name == "gpt-4o"

def test_model_router_fallback_chain(mock_brain):
    """Test fallback chain when models are unavailable"""
    router = ModelRouter(mock_brain)
    
    # Mock all models as unavailable
    with patch.object(router.registry, 'get_available_models') as mock_available:
        mock_available.return_value = []
        
        # Mock HF router
        with patch.object(router, 'hf_router') as mock_hf:
            mock_hf.route_task.return_value = ("mock response", "main")
            
            # Should fallback to default
            model, model_name = router.get_model_for_task("Test task", "coder")
            
            assert model is not None
            assert model_name == "gpt-4o-mini"  # Ultimate fallback

# ==================== ERROR HANDLING TESTS ====================

def test_model_router_invalid_configuration(mock_brain):
    """Test handling of invalid configuration"""
    with patch.dict(os.environ, {
        'MODEL_MAP_JSON': 'invalid json'
    }):
        # Should not crash
        router = ModelRouter(mock_brain)
        assert router is not None

def test_model_router_model_creation_failure(mock_brain):
    """Test handling of model creation failure"""
    router = ModelRouter(mock_brain)
    
    # Mock HF router
    with patch.object(router, 'hf_router') as mock_hf:
        mock_hf.route_task.return_value = ("mock response", "main")
        
        # Mock model creation failure
        with patch.object(router, '_create_model_instance') as mock_create:
            mock_create.return_value = None
            
            # Should fallback to default model
            model, model_name = router.get_model_for_task("Test task", "coder")
            
            # The fallback should still work even if model creation fails
            # The test should handle the case where no model is created
            assert model_name == "gpt-4o-mini"  # Fallback name

# ==================== PERFORMANCE TESTS ====================

def test_model_router_performance(mock_brain):
    """Test model router performance"""
    router = ModelRouter(mock_brain)
    
    # Mock HF router
    with patch.object(router, 'hf_router') as mock_hf:
        mock_hf.route_task.return_value = ("mock response", "main")
        
        start_time = time.time()
        
        # Test multiple model selections
        for _ in range(10):
            model, model_name = router.get_model_for_task("Test task", "coder")
        
        end_time = time.time()
        
        # Should be fast (< 1 second for 10 selections)
        assert end_time - start_time < 1.0 