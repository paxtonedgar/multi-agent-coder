# Multi-Agent Coder

A multi-agent AI coding system built with LangGraph where specialized agents collaborate through structured debate to solve programming tasks.

## Architecture

The system uses a graph-based workflow with five stages:

1. **Research** — Gathers context, searches documentation, analyzes requirements
2. **Planning** — Designs solution architecture and implementation strategy
3. **Coding** — Generates code with tool-calling agents (OpenAI, Anthropic, or HuggingFace models)
4. **Review** — Debate framework where agents critique and improve the solution
5. **Deploy** — Finalizes output with testing and validation

### Key Components

- **Debate Framework** (`debate_framework.py`) — Structured multi-agent debate for code review and decision-making
- **Creativity Engine** (`creativity_engine.py`) — Generates novel approaches and alternative solutions
- **HuggingFace Router** (`hf_routing.py`) — Dynamic model selection and routing across providers
- **Memory System** (`memory.py`) — Project-aware context with graph-based knowledge storage
- **Search Service** (`search_service.py`) — Web search integration for research phase
- **Monitoring API** (`main.py`) — FastAPI server exposing real-time workflow state

## Tech Stack

- **Orchestration**: LangGraph (StateGraph with checkpointing)
- **LLMs**: OpenAI, Anthropic, HuggingFace (configurable routing)
- **Framework**: LangChain (tool-calling agents, prompt templates)
- **API**: FastAPI + Uvicorn
- **Testing**: pytest

## Quick Start

```bash
git clone https://github.com/paxtonedgar/multi-agent-coder.git
cd multi-agent-coder

pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY=your_key
# or
export ANTHROPIC_API_KEY=your_key

# Run a task
python main.py --task "Build a REST API for user management"

# Or start the monitoring server
python main.py --serve
```

## Monitoring

The system exposes a FastAPI endpoint for real-time workflow monitoring at `http://localhost:8000`, showing current phase, progress, and agent activity logs. Pairs with [cursor-multi-agent-coding-ui-extension](https://github.com/paxtonedgar/cursor-multi-agent-coding-ui-extension-) for a VS Code dashboard.

## Testing

```bash
pytest
pytest test_enhanced_search.py -v
```
