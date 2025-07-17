# LangGraph Multi-Agent Coding System

A sophisticated multi-agent coding system built with LangGraph, featuring research, planning, coding, review, and deployment capabilities. Powered by OpenAI GPT-4o and Claude 3.5 Sonnet.

## 🚀 Features

- **Intelligent Research**: Web search, X (Twitter) integration, GitHub analysis
- **Strategic Planning**: AI-driven project planning with dependency analysis
- **Multi-Agent Coding**: Specialized agents for different coding tasks
- **Code Review**: Automated security and quality review
- **Deployment Ready**: CI/CD preparation and deployment automation
- **Persistent Memory**: Project brain with embeddings and decision tracking
- **2025 Technology Stack**: Latest models and deployment patterns

## 📋 Requirements

- Python 3.12+
- OpenAI API key or Anthropic API key
- Optional: GitHub token for enhanced research

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd multi-agent-coder
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   # OR
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Optional: GitHub token for enhanced research
   export GITHUB_TOKEN="your-github-token"
   ```

## 🎯 Usage

### Initialize Project Brain

First, initialize the project brain to analyze your codebase:

```bash
python main.py --init
```

### Full Workflow

Run a complete research → plan → code → review → deploy workflow:

```bash
python main.py "Create a FastAPI web service for user management"
```

### Quick Mode (No Research)

Skip research phase for faster execution:

```bash
python main.py --quick "Add authentication to existing API"
```

### Research Only

Get research insights without implementation:

```bash
python main.py --research-only "Best practices for edge deployment"
```

### Learn from GitHub Repository

Analyze and learn from existing repositories:

```bash
python main.py --learn-from github --reference "https://github.com/user/repo" "Implement similar features"
```

## 🏗️ Architecture

### Core Components

- **`memory.py`**: Project brain with embeddings and persistence
- **`tools.py`**: LangChain tools for research and build operations
- **`agents.py`**: Tool-calling agents with specialized roles
- **`graph.py`**: LangGraph workflow with conditional logic
- **`main.py`**: CLI interface and orchestration

### Agent Roles

1. **Planner**: Strategic planning and task breakdown
2. **Coder-1**: Core implementation and business logic
3. **Coder-2**: Tests, utilities, and documentation
4. **Reviewer**: Security and quality review
5. **Integrator**: Code merging and deployment preparation
6. **Architect**: Research and architectural decisions
7. **Coordinator**: Research synthesis and recommendations

### Workflow Stages

1. **Research**: Current solutions, technologies, best practices
2. **Planning**: Implementation plan with phases and dependencies
3. **Coding**: Multi-agent code generation with tests
4. **Review**: Automated code review and feedback
5. **Deploy**: Deployment preparation and CI/CD setup

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (preferred)
- `ANTHROPIC_API_KEY`: Anthropic API key (alternative)
- `GITHUB_TOKEN`: GitHub token for enhanced research

### Model Configuration

The system automatically selects the best available model:
- **OpenAI GPT-4o**: Primary choice for most tasks
- **Claude 3.5 Sonnet**: Alternative with excellent coding capabilities

### Project Brain

The project brain stores:
- Codebase analysis with AST parsing
- Function call graphs
- Architectural decisions
- Research findings
- Deployment history

## 📊 Example Output

```
╔══════════════════════════════════════════════════════════════╗
║                LangGraph Multi-Agent Coder                   ║
║                        v2025.1.0                             ║
║                                                              ║
║  Research • Plan • Code • Review • Deploy                   ║
║                                                              ║
║  Powered by LangGraph, OpenAI GPT-4o, Claude 3.5 Sonnet     ║
╚══════════════════════════════════════════════════════════════╝

📋 Project Context:
   Python: 3.12.0
   Directory: /path/to/project
   Git: ✅
   Virtual Env: ✅

🔑 API Keys:
   OpenAI: ✅
   Anthropic: ❌
   Github: ✅

🚀 Starting full workflow...
📝 Task: Create a FastAPI web service for user management

🎉 Workflow completed successfully!

📊 Summary:
   Research sources: 2
   Code files: 2
   Review feedback: 1
   Deployment: prepared
```

## 🛡️ Security Features

- **No Hardcoded Secrets**: All secrets via environment variables
- **Code Review**: Automated security vulnerability detection
- **Input Validation**: Comprehensive input sanitization
- **Error Handling**: Robust error handling and logging

## 🚀 Deployment Ready

The system prepares projects for modern deployment:

- **Edge Computing**: Cloudflare Workers, Vercel, Deno Deploy
- **Containerization**: Docker and Docker Compose support
- **CI/CD**: GitHub Actions, GitLab CI integration
- **Monitoring**: Built-in logging and performance tracking

## 🔍 Research Capabilities

- **Web Search**: DuckDuckGo integration for current information
- **X (Twitter)**: Latest discussions and trends
- **GitHub Analysis**: Deep repository analysis and pattern extraction
- **Technology Trends**: 2025 technology stack recommendations

## 📈 Performance

- **Async Processing**: Non-blocking operations where possible
- **Caching**: Intelligent caching of research results
- **Parallel Execution**: Multi-agent parallel processing
- **Memory Optimization**: Efficient state management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: This README and inline code comments
- **Examples**: See the `examples/` directory

## 🔮 Roadmap

- [ ] Enhanced web scraping capabilities
- [ ] Integration with more LLM providers
- [ ] Advanced deployment automation
- [ ] Real-time collaboration features
- [ ] Custom agent training capabilities

---

**Built with ❤️ using LangGraph, OpenAI, and Claude** 