# Multi-Agent Coder

A multi-agent AI coding system built with LangGraph where specialized agents collaborate through structured debate to solve programming tasks. Includes a VS Code extension for real-time monitoring.

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
- **VS Code Extension** (`vscode-extension/`) — Real-time dashboard with agent status, progress bars, and activity logs

## Tech Stack

- **Orchestration**: LangGraph (StateGraph with checkpointing)
- **LLMs**: OpenAI, Anthropic, HuggingFace (configurable routing)
- **Framework**: LangChain (tool-calling agents, prompt templates)
- **API**: FastAPI + Uvicorn
- **Monitoring UI**: VS Code extension (TypeScript) + Next.js dashboard
- **Testing**: pytest

## Project Structure

```
multi-agent-coder/
├── agents.py                # LangGraph agents with tool-calling
├── graph.py                 # StateGraph workflow (research → plan → code → review → deploy)
├── debate_framework.py      # Multi-agent debate for code review
├── creativity_engine.py     # Novel approach generation
├── hf_routing.py            # HuggingFace model router
├── memory.py                # Graph-based project memory
├── search_service.py        # Web search integration
├── main.py                  # FastAPI monitoring server + CLI entry point
├── tests/                   # Test suite
└── vscode-extension/        # VS Code monitoring extension
    ├── src/extension.ts      # Extension entry point
    ├── app/                  # Next.js dashboard UI
    └── components/           # UI components
```

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

### VS Code Extension

```bash
cd vscode-extension
pnpm install
# Press F5 in VS Code to launch extension dev host
```

Opens a real-time dashboard (`Ctrl+Shift+A` / `Cmd+Shift+A`) showing agent status, progress, and activity logs.

## Testing

```bash
pytest
pytest test_enhanced_search.py -v
```

## License

MIT
