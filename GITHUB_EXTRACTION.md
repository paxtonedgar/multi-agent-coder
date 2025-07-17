# Enhanced GitHub Content Extraction

## Overview

The enhanced GitHub content extraction system provides comprehensive parsing of GitHub repositories to extract prompt-output pairs for DSPy optimization. This system goes beyond simple code search to analyze issues, pull requests, notebooks, README files, and example code.

## Features

### 🔍 Multi-Source Content Extraction

- **Issues**: Extract examples from GitHub issues with code blocks and descriptions
- **Pull Requests**: Parse PR descriptions and file changes for code examples
- **Notebooks**: Extract code cells and outputs from Jupyter notebooks
- **README**: Parse documentation and code examples from README files
- **Examples**: Find example files and demo code throughout repositories

### 🎯 DSPy-Relevant Filtering

The system intelligently filters content for DSPy-relevant examples using keywords:
- `dspy`, `optimization`, `prompt`, `example`, `tutorial`
- `module`, `signature`, `metric`, `compiler`, `bootstrap`
- `few-shot`, `mipro`, `teleprompter`, `react`, `chain`

### 💾 Smart Caching

- Examples are cached in the project brain for reuse
- Prevents redundant API calls
- Enables offline access to previously extracted examples

## Usage

### Basic Usage

```python
from tools import RealResearchTools
from memory import ProjectBrain

# Initialize
brain = ProjectBrain()
research_tools = RealResearchTools(brain)

# Extract all content types
examples = research_tools.extract_github_content(
    "https://github.com/stanfordnlp/dspy",
    content_types=['issues', 'prs', 'notebooks', 'readme', 'examples']
)
```

### Selective Content Extraction

```python
# Extract only issues and PRs
examples = research_tools.extract_github_content(
    "https://github.com/langchain-ai/langgraph",
    content_types=['issues', 'prs']
)

# Extract only notebooks
examples = research_tools.extract_github_content(
    "https://github.com/openai/openai-python",
    content_types=['notebooks']
)
```

### LangChain Tool Integration

```python
from tools import create_research_tools

# Create tools with brain
brain = ProjectBrain()
tools = create_research_tools(brain)

# Use the extract_github_content tool
result = tools[4]("https://github.com/stanfordnlp/dspy", "issues,prs,notebooks")
```

## Content Types

### Issues Extraction

**What it extracts:**
- Issue titles as tasks
- Code blocks from issue descriptions
- Issue descriptions as context

**Example output:**
```json
{
    "task": "Issue: Tutorial Including SignatureOptimizer",
    "results": "```python\nclass MySignature(dspy.Signature):\n    input = dspy.InputField()\n    output = dspy.OutputField()\n```",
    "source": "GitHub Issue #485",
    "url": "https://github.com/stanfordnlp/dspy/issues/485"
}
```

### Pull Requests Extraction

**What it extracts:**
- PR titles as tasks
- Code blocks from PR descriptions
- File changes (patches) from PRs
- Code examples in PR discussions

**Example output:**
```json
{
    "task": "PR: Add async support for dspy.Evaluate",
    "results": "@@ -1,3 +1,5 @@\n+import asyncio\n+from typing import AsyncGenerator\n+\n async def evaluate_async(module, examples):\n     # Implementation\n```",
    "source": "GitHub PR #8504 file change",
    "url": "https://github.com/stanfordnlp/dspy/pull/8504"
}
```

### Notebooks Extraction

**What it extracts:**
- Code cells from Jupyter notebooks
- Cell outputs and results
- Markdown cells with code examples

**Supported formats:**
- `.ipynb` files
- `.jupyter` files
- `.notebook` files
- Notebooks in common directories (`notebooks/`, `examples/`, `tutorials/`, `docs/`)

**Example output:**
```json
{
    "task": "Notebook cell: dspy_tutorial.ipynb",
    "results": "import dspy\n\nclass SimpleQA(dspy.Signature):\n    question = dspy.InputField()\n    answer = dspy.OutputField()\n\n# Output:\n# SimpleQA(question='What is DSPy?', answer='DSPy is a framework...')",
    "source": "Jupyter Notebook: dspy_tutorial.ipynb",
    "url": "https://github.com/stanfordnlp/dspy/blob/main/examples/tutorial.ipynb"
}
```

### README Extraction

**What it extracts:**
- Code blocks from README files
- Headers as task descriptions
- Example files in the repository

**Example output:**
```json
{
    "task": "README: Quick Start Example",
    "results": "```python\nimport dspy\n\n# Define your signature\nclass QA(dspy.Signature):\n    question = dspy.InputField()\n    answer = dspy.OutputField()\n```",
    "source": "README: dspy",
    "url": "https://github.com/stanfordnlp/dspy/blob/main/README.md"
}
```

## Error Handling

### Rate Limiting

The system handles GitHub API rate limits gracefully:
- Detects rate limit errors
- Skips remaining content types when rate limited
- Continues with already extracted content
- Provides clear error messages

### Robust Parsing

- Handles different notebook formats
- Manages base64 encoded content
- Deals with malformed JSON
- Graceful fallbacks for parsing errors

### Network Resilience

- Timeout handling for API calls
- Retry logic for transient failures
- Graceful degradation when services are unavailable

## Integration with DSPy

### Automatic Example Generation

The extracted content is automatically integrated into DSPy optimization:

```python
from prompts import get_example_dataset

# Get examples including GitHub content
examples = get_example_dataset(brain)

# Use in DSPy optimization
optimizer = dspy.MIPRO(metric=validate_clean_code)
compiled_module = optimizer.compile(module, trainset=examples)
```

### Example Format

All extracted examples are converted to DSPy format:

```python
{
    'inputs': {'task': 'Issue: Tutorial Including SignatureOptimizer'},
    'outputs': {'results': 'class MySignature(dspy.Signature):...'}
}
```

## Configuration

### GitHub Token

Set your GitHub token for higher rate limits:

```bash
export GITHUB_TOKEN=your_github_token_here
```

### Content Type Selection

Choose which content types to extract:

```python
# All content types (default)
content_types = ['issues', 'prs', 'notebooks', 'readme', 'examples']

# Only code-related content
content_types = ['prs', 'notebooks', 'examples']

# Only documentation
content_types = ['issues', 'readme']
```

### Rate Limit Management

The system automatically manages API calls:
- Respects GitHub's rate limits
- Uses authenticated requests when token is available
- Implements exponential backoff for retries

## Performance

### Caching Strategy

- Examples are cached in the project brain
- Subsequent extractions use cached data
- Cache is persisted across sessions
- Automatic cache invalidation after 24 hours

### Optimization

- Parallel processing where possible
- Efficient API usage with pagination
- Smart filtering to reduce irrelevant content
- Memory-efficient processing of large repositories

## Testing

Run the test suite to validate functionality:

```bash
python test_github_extraction.py
```

The test suite validates:
- Content extraction from different repositories
- DSPy-relevant filtering
- Integration with prompts.py
- Error handling and rate limiting

## Troubleshooting

### Common Issues

1. **Rate Limiting**: Set a GitHub token for higher limits
2. **No Examples Found**: Check repository accessibility and content
3. **Parsing Errors**: Some repositories may have malformed content
4. **Network Issues**: Check internet connectivity and GitHub API status

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **Advanced Filtering**: ML-based relevance scoring
- **Content Clustering**: Group similar examples
- **Quality Scoring**: Rate example quality automatically
- **Cross-Repository Analysis**: Find patterns across multiple repos
- **Real-time Updates**: Webhook-based content updates

## Contributing

To enhance the GitHub extraction system:

1. Add new content type extractors
2. Improve filtering algorithms
3. Enhance error handling
4. Add new repository sources
5. Optimize performance

Follow the existing code patterns and add comprehensive tests for new features. 