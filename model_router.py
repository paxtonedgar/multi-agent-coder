"""
Advanced Model Router with HF Integration
Optimizes model selection based on task complexity, agent specialization, and performance tracking
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLLM

from memory import ProjectBrain, NodeType
try:
    from hf_routing import HFRoutingSystem
    HF_ROUTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HF routing not available: {e}")
    HF_ROUTING_AVAILABLE = False
    HFRoutingSystem = None

# ==================== CONFIGURATION ====================

class CostMode(Enum):
    PERFORMANCE = "performance"
    ECONOMY = "economy"

class ProviderPriority(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    BALANCED = "balanced"

@dataclass
class ModelConfig:
    """Model configuration with performance characteristics"""
    name: str
    provider: str
    complexity_level: str  # low, medium, high
    cost_per_1k_tokens: float
    reasoning_score: float
    coding_score: float
    analysis_score: float
    speed_score: float
    mac_optimized: bool = False
    available: bool = True

@dataclass
class TaskComplexity:
    """Task complexity assessment"""
    score: float  # 0.0 to 1.0
    category: str  # reasoning, coding, analysis, coordination
    complexity_level: str  # low, medium, high
    estimated_tokens: int
    requires_reasoning: bool
    requires_creativity: bool
    requires_accuracy: bool

@dataclass
class ModelPerformance:
    """Performance tracking for model selection"""
    model_name: str
    task_category: str
    success_rate: float
    avg_latency: float
    avg_confidence: float
    total_uses: int
    last_used: str

# ==================== MODEL REGISTRY ====================

class ModelRegistry:
    """Registry of available models with performance characteristics"""
    
    def __init__(self):
        self.models = self._initialize_models()
        self.performance_data = {}
        self.api_health = {
            'openai': True,
            'anthropic': True,
            'huggingface': True
        }
    
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations"""
        return {
            # OpenAI Models
            'gpt-4o': ModelConfig(
                name='gpt-4o',
                provider='openai',
                complexity_level='high',
                cost_per_1k_tokens=0.005,
                reasoning_score=0.95,
                coding_score=0.92,
                analysis_score=0.94,
                speed_score=0.85
            ),
            'gpt-4o-mini': ModelConfig(
                name='gpt-4o-mini',
                provider='openai',
                complexity_level='medium',
                cost_per_1k_tokens=0.00015,
                reasoning_score=0.85,
                coding_score=0.88,
                analysis_score=0.82,
                speed_score=0.95
            ),
            'gpt-3.5-turbo': ModelConfig(
                name='gpt-3.5-turbo',
                provider='openai',
                complexity_level='low',
                cost_per_1k_tokens=0.0005,
                reasoning_score=0.70,
                coding_score=0.75,
                analysis_score=0.68,
                speed_score=0.98
            ),
            
            # Anthropic Models
            'claude-3-opus': ModelConfig(
                name='claude-3-opus',
                provider='anthropic',
                complexity_level='high',
                cost_per_1k_tokens=0.015,
                reasoning_score=0.96,
                coding_score=0.90,
                analysis_score=0.95,
                speed_score=0.80
            ),
            'claude-3-5-sonnet': ModelConfig(
                name='claude-3-5-sonnet',
                provider='anthropic',
                complexity_level='medium',
                cost_per_1k_tokens=0.003,
                reasoning_score=0.92,
                coding_score=0.85,
                analysis_score=0.90,
                speed_score=0.88
            ),
            'claude-3-5-haiku': ModelConfig(
                name='claude-3-5-haiku',
                provider='anthropic',
                complexity_level='low',
                cost_per_1k_tokens=0.00025,
                reasoning_score=0.78,
                coding_score=0.80,
                analysis_score=0.75,
                speed_score=0.96
            ),
            
            # HF Models (for reasoning tasks)
            'deepseek-v3': ModelConfig(
                name='deepseek-v3',
                provider='huggingface',
                complexity_level='high',
                cost_per_1k_tokens=0.0,  # Free
                reasoning_score=0.95,
                coding_score=0.93,
                analysis_score=0.89,
                speed_score=0.70,
                mac_optimized=True
            ),
            'qwen2.5-72b': ModelConfig(
                name='qwen2.5-72b',
                provider='huggingface',
                complexity_level='high',
                cost_per_1k_tokens=0.0,
                reasoning_score=0.93,
                coding_score=0.88,
                analysis_score=0.91,
                speed_score=0.65
            )
        }
    
    def get_available_models(self, complexity_level: str = None) -> List[ModelConfig]:
        """Get available models filtered by complexity level"""
        available = []
        for model in self.models.values():
            if not model.available or not self.api_health.get(model.provider, True):
                continue
            if complexity_level and model.complexity_level != complexity_level:
                continue
            available.append(model)
        return available
    
    def update_api_health(self, provider: str, healthy: bool):
        """Update API health status"""
        self.api_health[provider] = healthy
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name"""
        return self.models.get(model_name)

# ==================== COMPLEXITY ASSESSOR ====================

class ComplexityAssessor:
    """Assesses task complexity for model selection"""
    
    def __init__(self):
        self.complexity_keywords = {
            'high': [
                'architecture', 'system design', 'algorithm', 'optimization',
                'reasoning', 'analysis', 'strategy', 'planning', 'research',
                'complex', 'advanced', 'sophisticated', 'multi-step'
            ],
            'medium': [
                'implementation', 'coding', 'development', 'integration',
                'review', 'testing', 'debugging', 'refactoring', 'documentation'
            ],
            'low': [
                'simple', 'basic', 'routine', 'validation', 'coordination',
                'scoring', 'checking', 'formatting', 'copying'
            ]
        }
        
        self.category_keywords = {
            'reasoning': ['reason', 'think', 'analyze', 'solve', 'optimize', 'strategy'],
            'analysis': ['analyze', 'review', 'evaluate', 'assess', 'examine', 'security'],
            'coding': ['code', 'implement', 'develop', 'write', 'create', 'build'],
            'coordination': ['coordinate', 'organize', 'manage', 'schedule']
        }
    
    def assess_complexity(self, task: str, agent_type: str = None) -> TaskComplexity:
        """Assess task complexity and requirements"""
        
        task_lower = task.lower()
        
        # Determine complexity level
        complexity_score = 0.5  # Default medium
        complexity_level = 'medium'
        
        for level, keywords in self.complexity_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                if level == 'high':
                    complexity_score = 0.8
                    complexity_level = 'high'
                elif level == 'low':
                    complexity_score = 0.2
                    complexity_level = 'low'
                break
        
        # Adjust based on agent type
        if agent_type:
            agent_complexity = {
                'architect': 0.9,
                'reasoning': 0.85,
                'debate': 0.8,
                'meta': 0.8,
                'planner': 0.7,
                'coder': 0.6,
                'reviewer': 0.6,
                'integrator': 0.5,
                'coordinator': 0.3,
                'auditor': 0.3,
                'confidence': 0.2
            }
            agent_score = agent_complexity.get(agent_type, 0.5)
            complexity_score = max(complexity_score, agent_score)
        
        # Determine category with priority
        category = 'coordination'  # Default
        
        # Check for analysis first (highest priority)
        if any(keyword in task_lower for keyword in self.category_keywords['analysis']):
            category = 'analysis'
        # Then reasoning
        elif any(keyword in task_lower for keyword in self.category_keywords['reasoning']):
            category = 'reasoning'
        # Then coding
        elif any(keyword in task_lower for keyword in self.category_keywords['coding']):
            category = 'coding'
        # Then coordination (default)
        
        # Override category based on complexity level for better accuracy
        if complexity_level == 'high':
            if 'architecture' in task_lower or 'design' in task_lower:
                category = 'reasoning'
            elif 'review' in task_lower or 'analyze' in task_lower:
                category = 'analysis'
        elif complexity_level == 'medium':
            if 'review' in task_lower or 'analyze' in task_lower or 'security' in task_lower:
                category = 'analysis'
            elif 'implement' in task_lower or 'code' in task_lower:
                category = 'coding'
        
        # Estimate token count
        estimated_tokens = len(task.split()) * 2  # Rough estimate
        
        # Determine requirements
        requires_reasoning = category == 'reasoning' or complexity_score > 0.7
        requires_creativity = 'creative' in task_lower or 'design' in task_lower
        requires_accuracy = 'accuracy' in task_lower or 'precision' in task_lower
        
        return TaskComplexity(
            score=complexity_score,
            category=category,
            complexity_level=complexity_level,
            estimated_tokens=estimated_tokens,
            requires_reasoning=requires_reasoning,
            requires_creativity=requires_creativity,
            requires_accuracy=requires_accuracy
        )

# ==================== PERFORMANCE TRACKER ====================

class PerformanceTracker:
    """Tracks model performance for optimization"""
    
    def __init__(self, brain: ProjectBrain):
        self.brain = brain
        self.performance_cache = {}
    
    def log_performance(self, model_name: str, task_category: str, 
                       latency: float, success: bool, confidence: float = None):
        """Log model performance"""
        
        key = f"{model_name}_{task_category}"
        
        if key not in self.performance_cache:
            self.performance_cache[key] = ModelPerformance(
                model_name=model_name,
                task_category=task_category,
                success_rate=0.0,
                avg_latency=0.0,
                avg_confidence=0.0,
                total_uses=0,
                last_used=datetime.now().isoformat()
            )
        
        perf = self.performance_cache[key]
        perf.total_uses += 1
        
        # Update success rate
        if success:
            perf.success_rate = (perf.success_rate * (perf.total_uses - 1) + 1.0) / perf.total_uses
        else:
            perf.success_rate = (perf.success_rate * (perf.total_uses - 1)) / perf.total_uses
        
        # Update average latency
        perf.avg_latency = (perf.avg_latency * (perf.total_uses - 1) + latency) / perf.total_uses
        
        # Update average confidence
        if confidence is not None:
            perf.avg_confidence = (perf.avg_confidence * (perf.total_uses - 1) + confidence) / perf.total_uses
        
        perf.last_used = datetime.now().isoformat()
        
        # Log to brain
        self.brain.add_node(
            NodeType.DECISION,
            f"Model performance logged: {model_name} for {task_category}",
            metadata={
                'model': model_name,
                'task_category': task_category,
                'latency': latency,
                'success': success,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def get_best_model(self, task_category: str, complexity_level: str) -> Optional[str]:
        """Get best performing model for task category"""
        
        best_model = None
        best_score = 0.0
        
        for key, perf in self.performance_cache.items():
            if perf.task_category == task_category and perf.total_uses >= 3:
                # Calculate composite score
                score = (perf.success_rate * 0.4 + 
                        (1.0 - perf.avg_latency / 10.0) * 0.3 +  # Normalize latency
                        perf.avg_confidence * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_model = perf.model_name
        
        return best_model
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for MetaAgent analysis"""
        return {
            'total_models_tracked': len(self.performance_cache),
            'performance_data': [asdict(perf) for perf in self.performance_cache.values()],
            'last_updated': datetime.now().isoformat()
        }

# ==================== MAIN MODEL ROUTER ====================

class ModelRouter:
    """Advanced model router with HF integration"""
    
    def __init__(self, brain: ProjectBrain, cost_mode: str = "performance", 
                 provider_priority: str = "balanced"):
        self.brain = brain
        self.cost_mode = CostMode(cost_mode)
        self.provider_priority = ProviderPriority(provider_priority)
        
        # Initialize components
        self.registry = ModelRegistry()
        self.assessor = ComplexityAssessor()
        self.tracker = PerformanceTracker(brain)
        self.hf_router = HFRoutingSystem(brain) if HF_ROUTING_AVAILABLE else None
        
        # Load configuration from environment
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from environment variables"""
        
        # Cost mode
        if os.getenv("COST_MODE"):
            self.cost_mode = CostMode(os.getenv("COST_MODE"))
        
        # Provider priority
        if os.getenv("PROVIDER_PRIORITY"):
            self.provider_priority = ProviderPriority(os.getenv("PROVIDER_PRIORITY"))
        
        # Custom model mappings
        model_map_json = os.getenv("MODEL_MAP_JSON")
        if model_map_json:
            try:
                custom_mappings = json.loads(model_map_json)
                for agent_type, model_name in custom_mappings.items():
                    if model_name in self.registry.models:
                        # Update model availability based on custom mapping
                        pass
            except json.JSONDecodeError:
                print("Warning: Invalid MODEL_MAP_JSON format")
        
        # Performance thresholds
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
        self.complexity_threshold = float(os.getenv("COMPLEXITY_THRESHOLD", "0.7"))
    
    def get_model_for_task(self, task: str, agent_type: str = None, 
                          force_model: str = None) -> Tuple[BaseLLM, str]:
        """Get optimal model for task with fallback logic"""
        
        start_time = time.time()
        
        # Force specific model if requested
        if force_model:
            model = self._create_model_instance(force_model)
            if model:
                return model, force_model
        
        # Assess task complexity
        complexity = self.assessor.assess_complexity(task, agent_type)
        
        # Check if HF routing is needed for reasoning tasks
        if (complexity.requires_reasoning and complexity.score > self.complexity_threshold and 
            self.hf_router is not None):
            try:
                # Try HF routing first
                response, model_used = self.hf_router.route_task(task, force_hf=True)
                if model_used.startswith("HF_"):
                    # Create a wrapper for the HF response
                    class HFResponseWrapper(BaseLLM):
                        def invoke(self, input: dict, **kwargs) -> dict:
                            return {"output": response}
                        
                        def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
                            from langchain_core.outputs import Generation, LLMResult
                            generations = []
                            for prompt in prompts:
                                generations.append([Generation(text=response)])
                            return LLMResult(generations=generations)
                        
                        @property
                        def _llm_type(self) -> str:
                            return "hf_wrapper"
                    
                    self.tracker.log_performance(
                        model_used, complexity.category, 
                        time.time() - start_time, True, 0.9
                    )
                    return HFResponseWrapper(), model_used
            except Exception as e:
                print(f"HF routing failed: {e}")
        
        # Select optimal model based on complexity and performance
        model_name = self._select_optimal_model(complexity, agent_type)
        
        # Create model instance
        model = self._create_model_instance(model_name)
        if not model:
            # Fallback to default model
            model = self._create_model_instance("gpt-4o-mini")
            model_name = "gpt-4o-mini"
        
        # Log performance tracking
        self.tracker.log_performance(
            model_name, complexity.category,
            time.time() - start_time, True, 0.8
        )
        
        return model, model_name
    
    def _select_optimal_model(self, complexity: TaskComplexity, agent_type: str = None) -> str:
        """Select optimal model based on complexity and performance"""
        
        # Get best performing model from history
        best_performing = self.tracker.get_best_model(complexity.category, complexity.complexity_level)
        if best_performing:
            return best_performing
        
        # Select based on complexity level
        available_models = self.registry.get_available_models(complexity.complexity_level)
        
        if not available_models:
            # Fallback to any available model
            available_models = self.registry.get_available_models()
        
        if not available_models:
            return "gpt-4o-mini"  # Ultimate fallback
        
        # Score models based on requirements
        best_model = None
        best_score = 0.0
        
        for model_config in available_models:
            score = self._calculate_model_score(model_config, complexity)
            
            if score > best_score:
                best_score = score
                best_model = model_config.name
        
        return best_model or "gpt-4o-mini"
    
    def _calculate_model_score(self, model_config: ModelConfig, complexity: TaskComplexity) -> float:
        """Calculate model score based on task requirements"""
        
        score = 0.0
        
        # Base score from complexity level match
        complexity_scores = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        score += complexity_scores.get(model_config.complexity_level, 0.5)
        
        # Category-specific scoring
        if complexity.category == 'reasoning':
            score += model_config.reasoning_score * 0.4
        elif complexity.category == 'coding':
            score += model_config.coding_score * 0.4
        elif complexity.category == 'analysis':
            score += model_config.analysis_score * 0.4
        
        # Speed consideration
        score += model_config.speed_score * 0.2
        
        # Cost consideration based on mode
        if self.cost_mode == CostMode.ECONOMY:
            cost_penalty = min(model_config.cost_per_1k_tokens * 1000, 0.3)
            score -= cost_penalty
        
        # Provider priority
        if self.provider_priority == ProviderPriority.ANTHROPIC and model_config.provider == 'anthropic':
            score += 0.1
        elif self.provider_priority == ProviderPriority.OPENAI and model_config.provider == 'openai':
            score += 0.1
        
        return score
    
    def _create_model_instance(self, model_name: str) -> Optional[BaseLLM]:
        """Create model instance by name"""
        
        try:
            if model_name.startswith('gpt-'):
                return ChatOpenAI(
                    model=model_name,
                    temperature=0.3,
                    timeout=120,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            elif model_name.startswith('claude-'):
                return ChatAnthropic(
                    model=model_name,
                    temperature=0.3,
                    timeout=120,
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            else:
                # HF models are handled by HF router
                return None
                
        except Exception as e:
            print(f"Failed to create model instance for {model_name}: {e}")
            return None
    
    def update_api_health(self, provider: str, healthy: bool):
        """Update API health status"""
        self.registry.update_api_health(provider, healthy)
        
        # Log to brain
        self.brain.add_node(
            NodeType.DECISION,
            f"API health update: {provider} {'healthy' if healthy else 'unhealthy'}",
            metadata={
                'provider': provider,
                'healthy': healthy,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        return {
            'cost_mode': self.cost_mode.value,
            'provider_priority': self.provider_priority.value,
            'api_health': self.registry.api_health,
            'performance_stats': self.tracker.get_performance_stats(),
            'hf_routing_stats': self.hf_router.get_routing_stats() if self.hf_router else {},
            'model_registry': {
                name: asdict(config) for name, config in self.registry.models.items()
            }
        }
    
    def refresh_models(self):
        """Refresh model discovery and performance data"""
        if self.hf_router:
            self.hf_router.refresh_models()
        print("✅ Model router refreshed")

# ==================== CONVENIENCE FUNCTIONS ====================

def get_model_router(brain: ProjectBrain = None) -> ModelRouter:
    """Get model router instance"""
    if brain is None:
        brain = ProjectBrain()
    
    return ModelRouter(brain)

def route_task(task: str, agent_type: str = None, brain: ProjectBrain = None) -> Tuple[BaseLLM, str]:
    """Convenience function to route a task"""
    router = get_model_router(brain)
    return router.get_model_for_task(task, agent_type) 