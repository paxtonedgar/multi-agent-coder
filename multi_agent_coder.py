#!/usr/bin/env python3
"""
Multi-Agent Coding System using AutoGen v0.2.35
Orchestrates specialized agents to build browser-based applications
"""

import os
import sys
import argparse
import subprocess
import asyncio
from typing import Dict, List, Optional
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Helper function to get API keys from macOS Keychain
def get_api_key(service: str, account: str) -> str:
    """Fetch API key from macOS Keychain"""
    try:
        result = subprocess.check_output(
            ['security', 'find-generic-password', '-s', service, '-a', account, '-w'],
            stderr=subprocess.DEVNULL
        )
        return result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        print(f"Warning: Could not fetch {service} key from Keychain")
        return ""

# Set up API keys
os.environ["OPENAI_API_KEY"] = get_api_key("OpenAI API key", "openAI")
anthropic_key = get_api_key("Anthropic API key", "anthropic")

# LLM Configurations
o1_config = {
    "config_list": [
        {
            "model": "o1-preview",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        },
        {
            "model": "gpt-4o",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    ],
    "temperature": 0.3,  # Lower for better planning precision
    "timeout": 120,
}

claude_config = {
    "config_list": [
        {
            "model": "claude-3-5-sonnet-20240620",  # Correct model ID
            "api_key": anthropic_key,
            "api_type": "anthropic",
            # Let AutoGen handle base_url internally
        }
    ],
    "temperature": 0.3,  # Lower for deterministic coding
    "timeout": 120,
}

default_config = {
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        },
        {
            "model": "gpt-4",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    ],
    "temperature": 0.3,  # Lower for code quality
    "timeout": 120,
}

# Git commit function for Integrator
def git_commit(message: str, branch: str = "feature-branch") -> str:
    """Git operations: checkout branch, add files, commit. Specify branch if needed."""
    try:
        # Create/checkout branch
        os.system(f"git checkout -b {branch} 2>/dev/null || git checkout {branch}")
        # Add all files
        os.system("git add .")
        # Commit with message
        result = os.system(f'git commit -m "{message}"')
        if result == 0:
            return f"Successfully committed to {branch}: {message}"
        else:
            return "No changes to commit or commit failed"
    except Exception as e:
        return f"Git error: {str(e)}"

# Agent creation function
def create_agents():
    """Create all specialized agents"""
    
    # Planner Agent - Uses O1/GPT-4o
    planner = AssistantAgent(
        name="Planner",
        llm_config=o1_config,
        system_message="""You are a strategic planner. Break tasks into subtasks:
- Browser automation (Selenium/Playwright/browserbase)
- API integrations (secure keys via env)
- CLI tools (argparse/Click)
- Testing strategies
Keep responses concise. Output structured task list."""
    )
    
    # Coder-1 Agent - Uses Claude
    coder1 = AssistantAgent(
        name="Coder-1",
        llm_config=claude_config,
        system_message="""You are the core developer. Focus on:
- Main application logic
- Browser automation (prefer Playwright)
- API integrations
- Error handling
Use env vars for secrets. Write clean, modular code."""
    )
    
    # Coder-2 Agent - Uses Claude
    coder2 = AssistantAgent(
        name="Coder-2",
        llm_config=claude_config,
        system_message="""You are the support developer. Handle:
- Unit/integration tests
- CLI interfaces (argparse)
- Documentation
- Helper functions
Ensure code is testable and maintainable."""
    )
    
    # Reviewer Agent - Uses GPT-4o
    reviewer = AssistantAgent(
        name="Reviewer",
        llm_config=default_config,
        system_message="""You are the code reviewer. Check for:
- Security issues (no hardcoded keys)
- Browser safety (XSS, injection)
- Code style and best practices
- Performance bottlenecks
Provide actionable feedback."""
    )
    
    # Integrator Agent - Uses GPT-4o
    integrator = AssistantAgent(
        name="Integrator",
        llm_config=default_config,
        system_message="""You are the integration specialist:
- Merge code from all agents
- Run tests (install deps with pip)
- Handle dependencies like pip install selenium or playwright
- Commit to git branch
- Ensure everything works together
Use git_commit function when ready."""
    )
    
    # Register git function for Integrator
    integrator.register_for_llm(
        name="git_commit",
        description="Commit code to git branch. Specify branch if needed."
    )(git_commit)
    
    # Admin Agent - UserProxyAgent
    admin = UserProxyAgent(
        name="Admin",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=5,
        code_execution_config={
            "work_dir": ".",
            "use_docker": False,
            "timeout": 100,
            "last_n_messages": 3,
        },
        system_message="Admin coordinating the team. Reply TERMINATE when task complete."
    )
    
    # Register git function for execution
    admin.register_for_execution(name="git_commit")(git_commit)
    
    return planner, coder1, coder2, reviewer, integrator, admin

# Async chat runner
async def run_chat_async(task: str, agents: tuple):
    """Run group chat asynchronously"""
    planner, coder1, coder2, reviewer, integrator, admin = agents
    
    # Create group chat
    groupchat = GroupChat(
        agents=[admin, planner, coder1, coder2, reviewer, integrator],
        messages=[],
        max_round=20,
        speaker_selection_method="auto",  # Dynamic routing based on context
    )
    
    # Create manager
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=default_config,
    )
    
    # Initiate chat
    await admin.a_initiate_chat(
        manager,
        message=f"Task: {task}\n\nPlease plan and implement this step by step.",
    )

# Sync chat runner (fallback)
def run_chat_sync(task: str, agents: tuple):
    """Run group chat synchronously"""
    planner, coder1, coder2, reviewer, integrator, admin = agents
    
    # Create group chat
    groupchat = GroupChat(
        agents=[admin, planner, coder1, coder2, reviewer, integrator],
        messages=[],
        max_round=20,
        speaker_selection_method="auto",  # Dynamic routing based on context
    )
    
    # Create manager
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=default_config,
    )
    
    # Initiate chat
    admin.initiate_chat(
        manager,
        message=f"Task: {task}\n\nPlease plan and implement this step by step.",
    )

# Main execution function
def main():
    """Main entry point with CLI interface"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Multi-agent coding system for browser automation projects"
    )
    parser.add_argument(
        "task",
        help="Task description for the agents to complete"
    )
    parser.add_argument(
        "--project-type",
        default="",
        help="Project type to append to task (e.g., 'selenium-bot', 'playwright-scraper')"
    )
    
    args = parser.parse_args()
    
    # Construct full task
    task = args.task
    if args.project_type:
        task += f" Project type: {args.project_type}"
    
    print(f"Starting multi-agent system...")
    print(f"Task: {task}")
    print("-" * 50)
    
    try:
        # Create agents
        agents = create_agents()
        # Human: Review agent prompts here if needed for task customization
        
        # Try async first, fallback to sync
        try:
            print("Attempting async mode for parallelism...")
            # Run async
            asyncio.run(run_chat_async(task, agents))
        except Exception as e:
            print(f"Async failed ({e}), using sync mode...")
            run_chat_sync(task, agents)
            
        print("\n" + "="*50)
        print("Task completed successfully!")
        # Human review point
        print("# Human review here - check generated code before deployment")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        # Symbiotic feedback point
        print("# Debug error and adjust agent prompts if needed")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Setup Requirements:

1. Install AutoGen 0.2.35:
   pip install pyautogen==0.2.35

2. Store API keys in macOS Keychain:
   security add-generic-password -s "OpenAI API Key" -a "openai" -w "your-openai-key"
   security add-generic-password -s "Anthropic API Key" -a "anthropic" -w "your-anthropic-key"

Usage Examples:

# Make executable
chmod +x multi_agent_coder.py

# Run with task
./multi_agent_coder.py "Test: Print hello world and commit it"

# Run with project type
./multi_agent_coder.py "Build a simulated automated gambler bot using Playwright for browser interaction, API for odds, and CLI controls" --project-type=gambling-bot
"""
