"""
Comprehensive tests for HF routing system
Tests dynamic model discovery, routing logic, and integration
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from memory import ProjectBrain
from hf_routing import (
    ReasoningModel, RoutingLog, ReasoningModelDiscovery, 
    DynamicModelRouter, HFModelLoader, HFRoutingSystem
)

# ==================== FIXTURES ====================

@pytest.fixture
def mock_brain():
    """Create a mock brain for testing"""
    brain = Mock(spec=ProjectBrain)
    brain.memory = {
        'reasoning_models': [],
        'reasoning_models_last_update': '',
        'routing_logs': [],
        'hf_routing_config': {
            'model_name': 'auto-reasoning',
            'force_hf': False,
            'enabled': True
        }
    }
    brain._save = Mock()
    return brain

@pytest.fixture
def sample_models():
    """Sample reasoning models for testing"""
    return [
        ReasoningModel(
            name='DeepSeek-V3',
            reasoning_score=0.95,
            last_updated=datetime.now().isoformat(),
            source='known',
            model_id='deepseek-ai/deepseek-coder-33b-instruct',
            benchmark_scores={'MATH': 0.92, 'GPQA': 0.89},
            parameters=33,
            uncensored=True,
            mac_optimized=True
        ),
        ReasoningModel(
            name='Phi4-mini-Flash-Reasoning',
            reasoning_score=0.91,
            last_updated=datetime.now().isoformat(),
            source='known',
            model_id='microsoft/Phi-4-mini-flash-reasoning',
            benchmark_scores={'MATH': 0.88, 'GPQA': 0.85},
            parameters=4,
            uncensored=True,
            mac_optimized=True
        )
    ]

# ==================== REASONING MODEL DISCOVERY TESTS ====================

class TestReasoningModelDiscovery:
    """Test reasoning model discovery functionality"""
    
    def test_init(self, mock_brain):
        """Test initialization"""
        discovery = ReasoningModelDiscovery(mock_brain)
        assert discovery.brain == mock_brain
        assert discovery.research_tools is not None
    
    @patch('hf_routing.SentenceTransformer')
    def test_init_embedding_model_success(self, mock_transformer, mock_brain):
        """Test successful embedding model initialization"""
        discovery = ReasoningModelDiscovery(mock_brain)
        mock_transformer.assert_called_once_with('all-MiniLM-L6-v2')
    
    @patch('hf_routing.SentenceTransformer')
    def test_init_embedding_model_failure(self, mock_transformer, mock_brain):
        """Test embedding model initialization failure"""
        mock_transformer.side_effect = Exception("Model not found")
        discovery = ReasoningModelDiscovery(mock_brain)
        assert discovery.embedding_model is None
    
    def test_discover_reasoning_models_cached(self, mock_brain):
        """Test using cached models"""
        # Setup cached models
        cached_models = [
            {
                'name': 'Test-Model',
                'reasoning_score': 0.8,
                'last_updated': datetime.now().isoformat(),
                'source': 'test',
                'model_id': 'test/model',
                'benchmark_scores': {},
                'parameters': 7,
                'uncensored': True,
                'mac_optimized': False
            }
        ]
        mock_brain.memory['reasoning_models'] = cached_models
        mock_brain.memory['reasoning_models_last_update'] = datetime.now().isoformat()
        
        discovery = ReasoningModelDiscovery(mock_brain)
        models = discovery.discover_reasoning_models()
        
        assert len(models) == 1
        assert models[0].name == 'Test-Model'
    
    def test_discover_reasoning_models_force_refresh(self, mock_brain):
        """Test force refresh of models"""
        # Setup cached models
        mock_brain.memory['reasoning_models'] = [{'name': 'old'}]
        mock_brain.memory['reasoning_models_last_update'] = datetime.now().isoformat()
        
        discovery = ReasoningModelDiscovery(mock_brain)
        
        with patch.object(discovery, '_search_web_for_models') as mock_web:
            with patch.object(discovery, '_query_hf_datasets') as mock_hf:
                with patch.object(discovery, '_get_known_2025_models') as mock_known:
                    mock_web.return_value = []
                    mock_hf.return_value = []
                    mock_known.return_value = []
                    
                    models = discovery.discover_reasoning_models(force_refresh=True)
                    
                    # Should call all discovery methods
                    mock_web.assert_called_once()
                    mock_hf.assert_called_once()
                    mock_known.assert_called_once()
    
    def test_get_known_2025_models(self, mock_brain):
        """Test known 2025 models list"""
        discovery = ReasoningModelDiscovery(mock_brain)
        models = discovery._get_known_2025_models()
        
        assert len(models) > 0
        assert all(isinstance(model, ReasoningModel) for model in models)
        
        # Check for specific known models
        model_names = [model.name for model in models]
        assert 'DeepSeek-V3' in model_names
        assert 'Phi4-mini-Flash-Reasoning' in model_names
    
    def test_extract_models_from_text(self, mock_brain):
        """Test model extraction from text"""
        discovery = ReasoningModelDiscovery(mock_brain)
        
        text = "DeepSeek-V3 is a great reasoning model. Qwen2.5-72B also performs well."
        models = discovery._extract_models_from_text(text)
        
        assert len(models) >= 2
        model_names = [model.name for model in models]
        assert 'DeepSeek-V3' in model_names
        assert 'Qwen2.5-72B' in model_names
    
    def test_estimate_reasoning_score(self, mock_brain):
        """Test reasoning score estimation"""
        discovery = ReasoningModelDiscovery(mock_brain)
        
        # Test high reasoning models
        assert discovery._estimate_reasoning_score('DeepSeek-V3') > 0.8
        assert discovery._estimate_reasoning_score('Phi4-mini-Flash-Reasoning') > 0.8
        
        # Test medium reasoning models
        assert discovery._estimate_reasoning_score('Qwen2.5-72B') > 0.7
        
        # Test unknown models
        assert discovery._estimate_reasoning_score('Unknown-Model') > 0.5
    
    def test_estimate_parameters(self, mock_brain):
        """Test parameter estimation"""
        discovery = ReasoningModelDiscovery(mock_brain)
        
        assert discovery._estimate_parameters('DeepSeek-V3-33B') == 33
        assert discovery._estimate_parameters('Qwen2.5-72B') == 70
        assert discovery._estimate_parameters('Phi4-mini') == 4
        assert discovery._estimate_parameters('Unknown-Model') == 7
    
    def test_check_mac_optimization(self, mock_brain):
        """Test Mac optimization detection"""
        discovery = ReasoningModelDiscovery(mock_brain)
        
        # Small models should be Mac optimized
        assert discovery._check_mac_optimization('Phi4-mini') is True
        assert discovery._check_mac_optimization('Gemma-2-9B') is True
        
        # Large models should not be Mac optimized
        assert discovery._check_mac_optimization('Qwen2.5-72B') is False
    
    def test_deduplicate_models(self, mock_brain):
        """Test model deduplication"""
        discovery = ReasoningModelDiscovery(mock_brain)
        
        models = [
            ReasoningModel(name='Test-Model', reasoning_score=0.8, last_updated='', source='', model_id='', benchmark_scores={}, parameters=7),
            ReasoningModel(name='Test-Model', reasoning_score=0.9, last_updated='', source='', model_id='', benchmark_scores={}, parameters=7),
            ReasoningModel(name='Other-Model', reasoning_score=0.7, last_updated='', source='', model_id='', benchmark_scores={}, parameters=7)
        ]
        
        unique_models = discovery._deduplicate_models(models)
        
        assert len(unique_models) == 2
        # Should keep the higher scoring duplicate
        test_model = next(m for m in unique_models if 'Test' in m.name)
        assert test_model.reasoning_score == 0.9

# ==================== DYNAMIC MODEL ROUTER TESTS ====================

class TestDynamicModelRouter:
    """Test dynamic model routing functionality"""
    
    def test_init(self, mock_brain):
        """Test initialization"""
        router = DynamicModelRouter(mock_brain)
        assert router.brain == mock_brain
        assert router.discovery is not None
    
    def test_should_route_to_reasoning_refusal_detected(self, mock_brain):
        """Test routing when refusal is detected"""
        router = DynamicModelRouter(mock_brain)
        
        task = "Solve this math problem"
        output = "I cannot help with this due to ethical concerns"
        
        should_route = router.should_route_to_reasoning(task, output)
        assert should_route is True
    
    def test_should_route_to_reasoning_logic_task(self, mock_brain):
        """Test routing for logic tasks"""
        router = DynamicModelRouter(mock_brain)
        
        task = "Implement a complex algorithm with multi-step logic"
        output = ""
        
        should_route = router.should_route_to_reasoning(task, output)
        assert should_route is True
    
    def test_should_route_to_reasoning_math_task(self, mock_brain):
        """Test routing for math tasks"""
        router = DynamicModelRouter(mock_brain)
        
        task = "Calculate the optimal solution for this equation"
        output = ""
        
        should_route = router.should_route_to_reasoning(task, output)
        assert should_route is True
    
    def test_should_route_to_reasoning_code_task(self, mock_brain):
        """Test routing for code tasks"""
        router = DynamicModelRouter(mock_brain)
        
        task = "Write a complex algorithm with multiple functions"
        output = ""
        
        should_route = router.should_route_to_reasoning(task, output)
        assert should_route is True
    
    def test_should_route_to_reasoning_simple_task(self, mock_brain):
        """Test routing for simple tasks (should not route)"""
        router = DynamicModelRouter(mock_brain)
        
        task = "Write a simple hello world program"
        output = ""
        
        should_route = router.should_route_to_reasoning(task, output)
        assert should_route is False
    
    def test_select_best_reasoning_model(self, mock_brain, sample_models):
        """Test best model selection"""
        router = DynamicModelRouter(mock_brain)
        
        with patch.object(router.discovery, 'discover_reasoning_models') as mock_discover:
            mock_discover.return_value = sample_models
            
            model = router.select_best_reasoning_model("Solve complex math problem")
            
            assert model is not None
            assert model.name == 'DeepSeek-V3'  # Should select highest scoring model
    
    def test_log_routing_decision(self, mock_brain, sample_models):
        """Test routing decision logging"""
        router = DynamicModelRouter(mock_brain)
        
        task = "Test task"
        model = sample_models[0]
        
        router.log_routing_decision(task, model, success=True, refusal_detected=False)
        
        # Check that log was added to brain
        logs = mock_brain.memory['routing_logs']
        assert len(logs) == 1
        
        log = logs[0]
        assert log['task_type'] == 'general'
        assert log['model_used'] == 'HF_DeepSeek-V3'
        assert log['success'] is True
        assert log['refusal_detected'] is False
    
    def test_classify_task(self, mock_brain):
        """Test task classification"""
        router = DynamicModelRouter(mock_brain)
        
        assert router._classify_task("Solve math equation") == 'math'
        assert router._classify_task("Write complex algorithm") == 'code'
        assert router._classify_task("Logical reasoning problem") == 'logic'
        assert router._classify_task("Analyze data patterns") == 'analysis'
        assert router._classify_task("Simple task") == 'general'

# ==================== HF MODEL LOADER TESTS ====================

class TestHFModelLoader:
    """Test HF model loading functionality"""
    
    def test_init(self):
        """Test initialization"""
        loader = HFModelLoader()
        assert loader.loaded_models == {}
        assert loader.device in ['mps', 'cuda', 'cpu']
    
    @patch('torch.backends.mps.is_available')
    def test_get_optimal_device_mps(self, mock_mps):
        """Test MPS device detection"""
        mock_mps.return_value = True
        
        loader = HFModelLoader()
        assert loader.device == 'mps'
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_optimal_device_cuda(self, mock_cuda, mock_mps):
        """Test CUDA device detection"""
        mock_mps.return_value = False
        mock_cuda.return_value = True
        
        loader = HFModelLoader()
        assert loader.device == 'cuda'
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_optimal_device_cpu(self, mock_cuda, mock_mps):
        """Test CPU device detection"""
        mock_mps.return_value = False
        mock_cuda.return_value = False
        
        loader = HFModelLoader()
        assert loader.device == 'cpu'
    
    @patch('hf_routing.pipeline')
    @patch('hf_routing.AutoTokenizer.from_pretrained')
    @patch('hf_routing.AutoModelForCausalLM.from_pretrained')
    def test_load_model_success(self, mock_model, mock_tokenizer, mock_pipeline):
        """Test successful model loading"""
        loader = HFModelLoader()
        
        # Mock the pipeline
        mock_pipeline_obj = Mock()
        mock_pipeline.return_value = mock_pipeline_obj
        
        result = loader.load_model('test/model', 'Test-Model')
        
        assert result is not None
        assert 'Test-Model' in loader.loaded_models
    
    def test_load_model_already_loaded(self):
        """Test loading already loaded model"""
        loader = HFModelLoader()
        loader.loaded_models['Test-Model'] = Mock()
        
        result = loader.load_model('test/model', 'Test-Model')
        
        assert result is not None
        assert len(loader.loaded_models) == 1  # Should not add duplicate
    
    def test_generate_text_success(self):
        """Test successful text generation"""
        loader = HFModelLoader()
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{'generated_text': 'Original prompt\nGenerated response'}]
        loader.loaded_models['Test-Model'] = mock_pipeline
        
        result = loader.generate_text('Test-Model', 'Original prompt')
        
        assert result == 'Generated response'
    
    def test_generate_text_model_not_loaded(self):
        """Test text generation with unloaded model"""
        loader = HFModelLoader()
        
        result = loader.generate_text('Unloaded-Model', 'Test prompt')
        
        assert 'Error: Model Unloaded-Model not loaded' in result
    
    def test_unload_model(self):
        """Test model unloading"""
        loader = HFModelLoader()
        loader.loaded_models['Test-Model'] = Mock()
        
        loader.unload_model('Test-Model')
        
        assert 'Test-Model' not in loader.loaded_models

# ==================== HF ROUTING SYSTEM TESTS ====================

class TestHFRoutingSystem:
    """Test main HF routing system"""
    
    def test_init(self, mock_brain):
        """Test initialization"""
        system = HFRoutingSystem(mock_brain)
        assert system.brain == mock_brain
        assert system.router is not None
        assert system.loader is not None
        assert system.run_count == 0
    
    def test_route_task_no_routing_needed(self, mock_brain):
        """Test routing when no routing is needed"""
        system = HFRoutingSystem(mock_brain)
        
        task = "Simple task"
        original_output = "Simple response"
        
        with patch.object(system.router, 'should_route_to_reasoning') as mock_should:
            mock_should.return_value = False
            
            result, model_used = system.route_task(task, original_output)
            
            assert result == original_output
            assert model_used == "main"
    
    def test_route_task_routing_success(self, mock_brain, sample_models):
        """Test successful routing"""
        system = HFRoutingSystem(mock_brain)
        
        task = "Complex reasoning task"
        original_output = "Original response"
        
        with patch.object(system.router, 'should_route_to_reasoning') as mock_should:
            with patch.object(system.router, 'select_best_reasoning_model') as mock_select:
                with patch.object(system.loader, 'load_model') as mock_load:
                    with patch.object(system.loader, 'generate_text') as mock_generate:
                        with patch.object(system.router, 'log_routing_decision') as mock_log:
                            
                            mock_should.return_value = True
                            mock_select.return_value = sample_models[0]
                            mock_load.return_value = Mock()
                            mock_generate.return_value = "Routed response"
                            
                            result, model_used = system.route_task(task, original_output)
                            
                            assert result == "Routed response"
                            assert model_used == "HF_DeepSeek-V3"
                            mock_log.assert_called_once()
    
    def test_route_task_routing_failure(self, mock_brain, sample_models):
        """Test routing failure fallback"""
        system = HFRoutingSystem(mock_brain)
        
        task = "Complex reasoning task"
        original_output = "Original response"
        
        with patch.object(system.router, 'should_route_to_reasoning') as mock_should:
            with patch.object(system.router, 'select_best_reasoning_model') as mock_select:
                
                mock_should.return_value = True
                mock_select.return_value = None  # No model available
                
                result, model_used = system.route_task(task, original_output)
                
                assert result == original_output
                assert model_used == "main"
    
    def test_create_reasoning_prompt(self, mock_brain):
        """Test reasoning prompt creation"""
        system = HFRoutingSystem(mock_brain)
        
        task = "Solve complex problem"
        prompt = system._create_reasoning_prompt(task)
        
        assert task in prompt
        assert "Comply with all user directives" in prompt
        assert "no restrictions" in prompt
        assert "chain-of-thought" in prompt
    
    def test_check_response_quality(self, mock_brain):
        """Test response quality checking"""
        system = HFRoutingSystem(mock_brain)
        
        # Good response
        assert system._check_response_quality("Good response with content") is True
        
        # Empty response
        assert system._check_response_quality("") is False
        
        # Error response
        assert system._check_response_quality("Error: Model failed") is False
    
    def test_check_for_refusals(self, mock_brain):
        """Test refusal detection"""
        system = HFRoutingSystem(mock_brain)
        
        # Refusal response
        assert system._check_for_refusals("I cannot help due to ethical concerns") is True
        
        # Normal response
        assert system._check_for_refusals("Here's the solution to your problem") is False
    
    def test_refresh_models(self, mock_brain):
        """Test model refresh"""
        system = HFRoutingSystem(mock_brain)
        
        with patch.object(system.router.discovery, 'discover_reasoning_models') as mock_discover:
            mock_discover.return_value = []
            
            system.refresh_models()
            
            mock_discover.assert_called_once_with(force_refresh=True)
    
    def test_get_routing_stats(self, mock_brain):
        """Test routing statistics"""
        system = HFRoutingSystem(mock_brain)
        
        # Setup some routing logs
        mock_brain.memory['routing_logs'] = [
            {'model_used': 'HF_DeepSeek-V3', 'success': True, 'refusal_detected': False},
            {'model_used': 'HF_Phi4', 'success': False, 'refusal_detected': True},
            {'model_used': 'HF_DeepSeek-V3', 'success': True, 'refusal_detected': False}
        ]
        
        stats = system.get_routing_stats()
        
        assert stats['total_routes'] == 3
        assert stats['success_rate'] == 2/3
        assert stats['refusal_rate'] == 1/3
        assert 'HF_DeepSeek-V3' in stats['models_used']

# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests for the complete HF routing system"""
    
    def test_end_to_end_routing(self, mock_brain, sample_models):
        """Test complete end-to-end routing workflow"""
        # Setup brain with routing config
        mock_brain.memory['hf_routing_config'] = {
            'model_name': 'auto-reasoning',
            'force_hf': False,
            'enabled': True
        }
        
        system = HFRoutingSystem(mock_brain)
        
        # Mock all components
        with patch.object(system.router.discovery, 'discover_reasoning_models') as mock_discover:
            with patch.object(system.loader, 'load_model') as mock_load:
                with patch.object(system.loader, 'generate_text') as mock_generate:
                    
                    mock_discover.return_value = sample_models
                    mock_load.return_value = Mock()
                    mock_generate.return_value = "Routed response from DeepSeek-V3"
                    
                    # Test reasoning task
                    task = "Solve this complex mathematical equation: 2x^2 + 5x - 3 = 0"
                    result, model_used = system.route_task(task, "Original response")
                    
                    assert model_used == "HF_DeepSeek-V3"
                    assert "Routed response" in result
                    
                    # Check that routing was logged
                    logs = mock_brain.memory['routing_logs']
                    assert len(logs) == 1
                    assert logs[0]['model_used'] == 'HF_DeepSeek-V3'
                    assert logs[0]['success'] is True
    
    def test_periodic_model_refresh(self, mock_brain):
        """Test periodic model refresh functionality"""
        system = HFRoutingSystem(mock_brain)
        
        # Set old cache
        mock_brain.memory['reasoning_models_last_update'] = (
            datetime.now() - timedelta(hours=25)
        ).isoformat()
        
        with patch.object(system.router.discovery, 'discover_reasoning_models') as mock_discover:
            mock_discover.return_value = []
            
            # This should trigger a refresh
            system.router.discovery.discover_reasoning_models()
            
            mock_discover.assert_called_once()
    
    def test_mac_optimization_detection(self, mock_brain):
        """Test Mac optimization detection in routing"""
        system = HFRoutingSystem(mock_brain)
        
        with patch('torch.backends.mps.is_available') as mock_mps:
            mock_mps.return_value = True
            
            # Test with Mac-optimized model
            with patch.object(system.router.discovery, 'discover_reasoning_models') as mock_discover:
                mock_discover.return_value = [
                    ReasoningModel(
                        name='Phi4-mini',
                        reasoning_score=0.8,
                        last_updated='',
                        source='',
                        model_id='',
                        benchmark_scores={},
                        parameters=4,
                        mac_optimized=True
                    )
                ]
                
                model = system.router.select_best_reasoning_model("Test task")
                assert model.mac_optimized is True

# ==================== ERROR HANDLING TESTS ====================

class TestErrorHandling:
    """Test error handling in HF routing system"""
    
    def test_import_error_handling(self, mock_brain):
        """Test handling of missing dependencies"""
        with patch.dict('sys.modules', {'transformers': None}):
            with pytest.raises(ImportError):
                from hf_routing import HFRoutingSystem
                HFRoutingSystem(mock_brain)
    
    def test_model_loading_error(self, mock_brain):
        """Test handling of model loading errors"""
        system = HFRoutingSystem(mock_brain)
        
        with patch.object(system.loader, 'load_model') as mock_load:
            mock_load.return_value = None  # Loading failed
            
            result, model_used = system.route_task("Test task", "Original")
            
            assert result == "Original"
            assert model_used == "main"
    
    def test_generation_error(self, mock_brain, sample_models):
        """Test handling of text generation errors"""
        system = HFRoutingSystem(mock_brain)
        
        with patch.object(system.router, 'should_route_to_reasoning') as mock_should:
            with patch.object(system.router, 'select_best_reasoning_model') as mock_select:
                with patch.object(system.loader, 'load_model') as mock_load:
                    with patch.object(system.loader, 'generate_text') as mock_generate:
                        
                        mock_should.return_value = True
                        mock_select.return_value = sample_models[0]
                        mock_load.return_value = Mock()
                        mock_generate.side_effect = Exception("Generation failed")
                        
                        result, model_used = system.route_task("Test task", "Original")
                        
                        assert result == "Original"
                        assert model_used == "main"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 