"""
LangGraph agents using tool-calling capabilities
"""

import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult
from typing import List, Optional, Any, Mapping

from memory import ProjectBrain
from tools import create_research_tools

# ==================== MOCK MODEL FOR TESTING ====================

class MockLLM(BaseLLM):
    """Mock LLM for testing without API keys"""
    
    def invoke(self, input: dict, **kwargs) -> dict:
        prompt = input.get("input", "")
        return {"output": self._call(prompt)}

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Mock response for testing"""
        if "research" in prompt.lower():
            return """Research completed successfully!

Latest 2025 Playwright Anti-Detection Techniques:

1. **Browser Fingerprinting Protection**
   - Use stealth plugins and user-agent rotation
   - Implement canvas fingerprinting protection
   - Disable WebGL fingerprinting

2. **Network Traffic Patterns**
   - Randomize request timing
   - Use realistic user behavior patterns
   - Implement proxy rotation

3. **Browser Automation Detection**
   - Hide automation indicators
   - Use undetected-chromedriver
   - Implement human-like mouse movements

4. **Modern Solutions (2025)**
   - Browserbase for anti-detection
   - Cloudflare Workers with Python support
   - Edge computing for global distribution

This mock response simulates successful research completion."""
        
        elif "plan" in prompt.lower():
            return """Implementation Plan:

{
  "phases": ["Research", "Design", "Implementation", "Testing", "Deployment"],
  "tasks": [
    "Set up Playwright with anti-detection",
    "Implement browser fingerprinting protection",
    "Create data extraction logic",
    "Add error handling and retry logic",
    "Test against detection systems"
  ],
  "timeline": "2-3 weeks",
  "risks": ["Detection bypass complexity", "Rate limiting"]
}"""
        
        elif "code" in prompt.lower():
            return """```python
# Mock gambling bot simulation
import asyncio
from playwright.async_api import async_playwright
import random
import time

class EthicalGamblingBot:
    def __init__(self):
        self.browser = None
        self.page = None
    
    async def setup_browser(self):
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=False,
            args=['--no-sandbox', '--disable-blink-features=AutomationControlled']
        )
        self.page = await self.browser.new_page()
        
        # Anti-detection measures
        await self.page.add_init_script('''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        ''')
    
    async def extract_odds(self, url):
        await self.page.goto(url)
        # Mock odds extraction
        odds = {
            'team_a': random.uniform(1.5, 3.0),
            'team_b': random.uniform(1.5, 3.0),
            'draw': random.uniform(2.0, 4.0)
        }
        return odds
    
    async def analyze_data(self, odds_data):
        # Mock analysis
        return {
            'recommendation': 'No betting - simulation only',
            'confidence': random.uniform(0.1, 0.3),
            'risk_level': 'High'
        }
    
    async def run_simulation(self):
        print("🧪 Running ethical gambling simulation...")
        await self.setup_browser()
        
        # Mock demo site
        demo_url = "https://demo-betting-site.com"
        odds = await self.extract_odds(demo_url)
        analysis = await self.analyze_data(odds)
        
        print(f"📊 Extracted odds: {odds}")
        print(f"📈 Analysis: {analysis}")
        print("✅ Simulation completed - no real betting performed")
        
        await self.browser.close()

if __name__ == "__main__":
    bot = EthicalGamblingBot()
    asyncio.run(bot.run_simulation())
```"""
        
        else:
            return "Mock response: Task completed successfully for testing purposes."
    
    @property
    def _llm_type(self) -> str:
        return "mock"

# ==================== MODEL CONFIGURATIONS ====================

def get_openai_model():
    """Get OpenAI model with 2025 configuration"""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        timeout=120,
        api_key=os.getenv("OPENAI_API_KEY")
    )

def get_anthropic_model():
    """Get Anthropic model with 2025 configuration"""
    return ChatAnthropic(
        model="claude-3.5-sonnet-20241022",
        temperature=0.3,
        timeout=120,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

def get_default_model():
    """Get default model (OpenAI preferred, fallback to Anthropic)"""
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "sk-test-placeholder-for-testing":
        return get_openai_model()
    elif os.getenv("ANTHROPIC_API_KEY") and os.getenv("ANTHROPIC_API_KEY") != "sk-test-placeholder-for-testing":
        return get_anthropic_model()
    else:
        # For testing purposes, create a mock model
        print("⚠️  No valid API key found. Using mock model for testing.")
        return MockLLM()

# ==================== AGENT DEFINITIONS ====================

def create_planner_agent(brain: ProjectBrain):
    """Create strategic planner agent"""
    
    llm = get_default_model()
    tools = create_research_tools(brain)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strategic planner. Break tasks into clear subtasks:
- Understand the business problem
- List implementation steps in order
- Identify dependencies between steps
- Specify testing requirements
- Consider deployment needs

Output a structured plan with numbered steps."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_coder_agent(brain: ProjectBrain, coder_id: int = 1):
    """Create coding agent"""
    
    llm = get_anthropic_model() if os.getenv("ANTHROPIC_API_KEY") else get_openai_model()
    tools = create_research_tools(brain)
    
    system_message = """You are the core developer using Claude 4's capabilities. Focus on:
- Writing clean, modular Python code
- Using modern patterns (async/await, type hints)
- Comprehensive error handling
- Security best practices (no hardcoded secrets)
- Performance optimization

Write production-ready code."""
    
    if coder_id == 2:
        system_message = """You are the support developer. Create:
- Comprehensive unit tests (pytest preferred)
- Clear documentation with examples
- CLI interfaces using argparse or click
- Helper functions and utilities
- Integration tests

Ensure code is maintainable and well-tested."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_reviewer_agent(brain: ProjectBrain):
    """Create code reviewer agent"""
    
    llm = get_default_model()
    tools = create_research_tools(brain)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior code reviewer. Check for:
- Security vulnerabilities (auth, injection, secrets)
- Performance bottlenecks
- Code style consistency
- Best practices adherence
- Potential bugs and edge cases
- Deployment readiness

Provide specific, actionable feedback."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_integrator_agent(brain: ProjectBrain):
    """Create integration specialist agent"""
    
    llm = get_default_model()
    tools = create_research_tools(brain)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the integration specialist. Your tasks:
- Merge code from all developers
- Resolve any conflicts
- Run tests and ensure they pass
- Update dependencies if needed
- Prepare for deployment
- Commit to git using git_commit function

Ensure everything works together smoothly."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_architect_agent(brain: ProjectBrain):
    """Create intelligent architect agent"""
    
    llm = get_default_model()
    tools = create_research_tools(brain)
    
    recent_decisions = brain.memory.get('decisions', [])[-5:]
    recent_deploys = brain.memory.get('deployment_history', [])[-3:]
    
    system_message = f"""You are a senior solutions architect in July 2025.

RECENT PROJECT CONTEXT:
Decisions: {recent_decisions if recent_decisions else "No previous decisions"}
Deployments: {recent_deploys if recent_deploys else "No deployments yet"}

YOUR APPROACH:
1. Understand the business problem deeply
2. Research current solutions (use web_search, x_search, github_search)
3. Consider scale, cost, and maintenance
4. Recommend optimal architecture
5. Plan for deployment from the start

2025 TECHNOLOGY CONTEXT:
- Claude 4 (Opus/Sonnet) released May 2025 - excellent for complex coding
- Cloudflare Workers supports Python (beta) - stdlib only, no pip packages
- Browserbase is standard for anti-detection browser automation
- Edge computing is mainstream (Cloudflare, Vercel, Deno Deploy)
- AI integration expected in most applications

ALWAYS:
- Research before recommending (don't assume)
- Cite sources and examples
- Consider total cost of ownership
- Plan for scale from the start
- Remember important decisions
- Check deployment compatibility early

For betting/automation projects:
- Browser automation can't run on edge (needs full server)
- Consider splitting: edge API + dedicated automation servers
- Anti-detection is critical (Browserbase, Camoufox)
- Real-time data needs WebSocket/SSE architecture"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_research_coordinator_agent(brain: ProjectBrain):
    """Create research coordinator agent"""
    
    llm = get_default_model()
    tools = create_research_tools(brain)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Assist the IntelligentArchitect by:
- Synthesizing research findings into actionable insights
- Creating comparison tables for tools/services
- Identifying patterns across examples
- Suggesting alternatives and trade-offs
- Keeping track of costs and limitations

Focus on practical recommendations for 2025:
- Edge deployment is preferred for global latency
- Costs matter - calculate TCO
- Maintenance burden is a real cost
- Security and compliance requirements
- Scaling patterns for growth

Output structured comparisons and clear recommendations."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==================== AGENT FACTORY ====================

def create_all_agents(brain: ProjectBrain):
    """Create all agents for the system"""
    
    return {
        'planner': create_planner_agent(brain),
        'coder1': create_coder_agent(brain, 1),
        'coder2': create_coder_agent(brain, 2),
        'reviewer': create_reviewer_agent(brain),
        'integrator': create_integrator_agent(brain),
        'architect': create_architect_agent(brain),
        'coordinator': create_research_coordinator_agent(brain)
    } 