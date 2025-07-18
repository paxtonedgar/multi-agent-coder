"""
Pytest configuration and fixtures for multi-agent coder testing
"""

import os
import sys
import tempfile
import shutil
import subprocess
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

import pytest
import pytest_asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import ProjectBrain
from tools import GitRepositoryTools, create_research_tools
from agents import create_all_agents, MockLLM
from graph import run_workflow, create_workflow_graph

# ==================== TEST CONFIGURATION ====================

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        "mock_api_responses": True,
        "use_real_apis": False,
        "test_timeout": 30,
        "max_concurrent_tests": 5,
        "coverage_target": 80.0,
        "performance_threshold": 5.0,  # seconds
        "quality_threshold": 0.8,  # 0-1 scale
    }

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# ==================== MOCK API RESPONSES ====================

@pytest.fixture(scope="session")
def mock_api_responses():
    """Mock API responses for testing without real API calls"""
    return {
        "openai": {
            "gpt-4o": {
                "research": "Research completed successfully! Latest 2025 techniques include...",
                "planning": "Implementation Plan: 1. Setup environment 2. Create core functionality 3. Add tests 4. Deploy",
                "coding": "```python\ndef hello_world():\n    return 'Hello, World!'\n```",
                "review": "Code review completed. Quality score: 0.85. Suggestions: Add type hints.",
                "deployment": "Deployment ready. CI/CD configured with GitHub Actions."
            }
        },
        "anthropic": {
            "claude-3.5-sonnet": {
                "research": "Research findings: Modern approaches include...",
                "planning": "Strategic plan: Phase 1: Foundation, Phase 2: Features, Phase 3: Optimization",
                "coding": "```python\nclass HelloWorld:\n    def greet(self):\n        return 'Hello, World!'\n```",
                "review": "Security review passed. Code quality: 0.9. Recommendations: Add error handling.",
                "deployment": "Production deployment configured. Monitoring enabled."
            }
        }
    }

# ==================== BRAIN AND MEMORY FIXTURES ====================

@pytest.fixture
def temp_brain():
    """Create a temporary brain for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_brain_")
    brain = ProjectBrain(temp_dir)
    
    yield brain
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def brain_with_memory(temp_brain):
    """Create brain with pre-populated memory"""
    # Add some test memory
    temp_brain.add_reflection("Test reflection 1", sources=["test"])
    temp_brain.add_reflection("Test reflection 2", sources=["test"])
    temp_brain.remember_decision("Use pytest for testing", "Best practice for Python testing")
    
    return temp_brain

# ==================== GIT REPOSITORY FIXTURES ====================

@pytest.fixture
def test_repo():
    """Create a test Git repository"""
    test_dir = tempfile.mkdtemp(prefix="test_repo_")
    
    # Initialize git repository
    subprocess.run(["git", "init"], cwd=test_dir, check=True)
    
    # Create test files
    test_files = {
        "README.md": "# Test Repository\n\nThis is a test repository for integration testing.",
        "main.py": "def hello():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    hello()",
        "requirements.txt": "requests>=2.31.0\npytest>=7.4.0",
        "test_main.py": "import pytest\nfrom main import hello\n\ndef test_hello():\n    assert hello is not None",
        "config.json": '{"name": "test-project", "version": "1.0.0"}'
    }
    
    for filename, content in test_files.items():
        with open(os.path.join(test_dir, filename), 'w') as f:
            f.write(content)
    
    # Add and commit files
    subprocess.run(["git", "add", "."], cwd=test_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=test_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=test_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=test_dir, check=True)
    
    yield test_dir
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)

@pytest.fixture
def git_tools(temp_brain):
    """Create Git tools instance"""
    return GitRepositoryTools(temp_brain)

# ==================== AGENT FIXTURES ====================

@pytest.fixture
def mock_agents(temp_brain):
    """Create mock agents for testing"""
    with patch('agents.get_default_model', return_value=MockLLM()):
        agents = create_all_agents(temp_brain)
        return agents

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    return MockLLM()

# ==================== WORKFLOW FIXTURES ====================

@pytest.fixture
def workflow_graph(temp_brain):
    """Create workflow graph for testing"""
    return create_workflow_graph(temp_brain)

@pytest.fixture
def quick_workflow_graph(temp_brain):
    """Create quick workflow graph for testing"""
    from graph import create_quick_workflow_graph
    return create_quick_workflow_graph(temp_brain)

# ==================== TOOL FIXTURES ====================

@pytest.fixture
def research_tools(temp_brain):
    """Create research tools for testing"""
    return create_research_tools(temp_brain)

# ==================== USER INTERACTION FIXTURES ====================

@pytest.fixture
def user_scenarios():
    """Common user interaction scenarios"""
    return {
        "simple_code_generation": {
            "initial_prompt": "Create a simple Hello World script with tests and docs",
            "follow_up": "Add error handling to the function",
            "refinement": "Make it more robust with logging",
            "expected_phases": ["research", "planning", "coding", "review", "deploy"]
        },
        "complex_project": {
            "initial_prompt": "Build a REST API with FastAPI for user management",
            "follow_up": "Add authentication and authorization",
            "refinement": "Implement rate limiting and caching",
            "expected_phases": ["research", "planning", "coding", "audit", "review", "deploy"]
        },
        "bug_fix": {
            "initial_prompt": "Fix the bug in the login function",
            "follow_up": "Add unit tests for the fix",
            "refinement": "Update documentation",
            "expected_phases": ["research", "planning", "coding", "review"]
        }
    }

@pytest.fixture
def cli_inputs():
    """Mock CLI inputs for testing"""
    return {
        "basic_task": ["Create a simple calculator"],
        "task_with_options": ["Build a web app", "--mode", "quick", "--use-hf", "auto-reasoning"],
        "task_with_repo": ["Analyze repository", "--repo-url", "https://github.com/test/repo"],
        "help_request": ["--help"],
        "stats_request": ["--show-stats"],
        "models_request": ["--list-models"]
    }

# ==================== PERFORMANCE FIXTURES ====================

@pytest.fixture
def performance_metrics():
    """Performance metrics tracking"""
    return {
        "start_time": None,
        "end_time": None,
        "api_calls": 0,
        "response_times": [],
        "memory_usage": [],
        "errors": []
    }

# ==================== QUALITY ASSESSMENT FIXTURES ====================

@pytest.fixture
def quality_metrics():
    """Quality assessment metrics"""
    return {
        "code_syntax": 0.0,
        "code_quality": 0.0,
        "documentation": 0.0,
        "test_coverage": 0.0,
        "security_score": 0.0,
        "overall_score": 0.0
    }

# ==================== TEST DATA FIXTURES ====================

@pytest.fixture
def sample_code():
    """Sample code for testing"""
    return {
        "python_function": """
def calculate_fibonacci(n: int) -> int:
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
""",
        "python_class": """
class Calculator:
    \"\"\"Simple calculator class.\"\"\"
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        \"\"\"Add two numbers.\"\"\"
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        \"\"\"Get calculation history.\"\"\"
        return self.history.copy()
""",
        "test_code": """
import pytest
from calculator import Calculator

def test_calculator_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0

def test_calculator_history():
    calc = Calculator()
    calc.add(1, 2)
    calc.add(3, 4)
    history = calc.get_history()
    assert len(history) == 2
    assert "1 + 2 = 3" in history[0]
"""
    }

@pytest.fixture
def sample_docs():
    """Sample documentation for testing"""
    return {
        "readme": """# Calculator Project

A simple calculator implementation in Python.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from calculator import Calculator

calc = Calculator()
result = calc.add(5, 3)
print(result)  # 8
```

## Testing

```bash
pytest tests/
```
""",
        "api_docs": """# API Documentation

## Endpoints

### POST /calculate
Perform a calculation.

**Request Body:**
```json
{
  "operation": "add",
  "a": 5,
  "b": 3
}
```

**Response:**
```json
{
  "result": 8,
  "operation": "add"
}
```
"""
    }

# ==================== ERROR SCENARIOS ====================

@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing"""
    return {
        "api_rate_limit": {
            "error_type": "rate_limit",
            "error_message": "Rate limit exceeded",
            "expected_handling": "retry_with_backoff"
        },
        "api_authentication": {
            "error_type": "authentication",
            "error_message": "Invalid API key",
            "expected_handling": "fallback_to_mock"
        },
        "network_timeout": {
            "error_type": "timeout",
            "error_message": "Request timeout",
            "expected_handling": "retry_with_timeout"
        },
        "invalid_input": {
            "error_type": "validation",
            "error_message": "Invalid input format",
            "expected_handling": "user_feedback"
        }
    }

# ==================== UTILITY FUNCTIONS ====================

def assert_code_quality(code: str, min_score: float = 0.7):
    """Assert code meets quality standards"""
    # Basic syntax check
    try:
        compile(code, '<string>', 'exec')
        syntax_score = 1.0
    except SyntaxError:
        syntax_score = 0.0
    
    # Basic quality checks
    quality_score = 0.0
    if 'def ' in code or 'class ' in code:
        quality_score += 0.3
    if '"""' in code or "'''" in code:
        quality_score += 0.2
    if 'import ' in code or 'from ' in code:
        quality_score += 0.2
    if 'test' in code.lower():
        quality_score += 0.3
    
    overall_score = (syntax_score + quality_score) / 2
    assert overall_score >= min_score, f"Code quality score {overall_score} below threshold {min_score}"

def assert_response_time(start_time: float, max_time: float = 5.0):
    """Assert response time is within acceptable limits"""
    import time
    elapsed = time.time() - start_time
    assert elapsed <= max_time, f"Response time {elapsed:.2f}s exceeds limit {max_time}s"

def assert_api_call_count(metrics: Dict, max_calls: int = 10):
    """Assert API call count is within limits"""
    assert metrics.get('api_calls', 0) <= max_calls, f"API calls {metrics.get('api_calls', 0)} exceed limit {max_calls}"

def create_mock_api_response(content: str, model: str = "gpt-4o"):
    """Create a mock API response"""
    return {
        "choices": [{
            "message": {
                "content": content,
                "role": "assistant"
            }
        }],
        "model": model,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    } 