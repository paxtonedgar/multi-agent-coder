#!/usr/bin/env python3
"""
LangGraph Multi-Agent Coding System - Main Entry Point
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

from memory import ProjectBrain
from graph import run_workflow
from dotenv import load_dotenv
load_dotenv()

# ==================== UTILITY FUNCTIONS ====================

def analyze_project_context() -> Dict[str, Any]:
    """Analyze current project context"""
    context = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'working_directory': os.getcwd(),
        'has_git': os.path.exists('.git'),
        'has_requirements': os.path.exists('requirements.txt'),
        'has_pyproject': os.path.exists('pyproject.toml'),
        'has_venv': any(os.path.exists(venv) for venv in ['.venv', 'venv', 'env']),
        'api_keys': {
            'openai': bool(os.getenv("OPENAI_API_KEY")),
            'anthropic': bool(os.getenv("ANTHROPIC_API_KEY")),
            'github': bool(os.getenv("GITHUB_TOKEN"))
        }
    }
    
    # Check for common project files
    project_files = [
        'README.md', 'setup.py', 'Pipfile', 'poetry.lock',
        'Dockerfile', '.dockerignore', 'docker-compose.yml',
        '.github/workflows', '.gitignore', 'Makefile'
    ]
    
    context['project_files'] = {
        file: os.path.exists(file) for file in project_files
    }
    
    return context

def print_banner():
    """Print system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                LangGraph Multi-Agent Coder                   ║
║                        v2025.1.0                             ║
║                                                              ║
║  Research • Plan • Code • Review • Deploy                   ║
║                                                              ║
║  Powered by LangGraph, OpenAI GPT-4o, Claude 3.5 Sonnet     ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_context(context: Dict[str, Any]):
    """Print project context"""
    print("📋 Project Context:")
    print(f"   Python: {context['python_version']}")
    print(f"   Directory: {context['working_directory']}")
    print(f"   Git: {'✅' if context['has_git'] else '❌'}")
    print(f"   Virtual Env: {'✅' if context['has_venv'] else '❌'}")
    
    print("\n🔑 API Keys:")
    for provider, has_key in context['api_keys'].items():
        status = "✅" if has_key else "❌"
        print(f"   {provider.title()}: {status}")
    
    print("\n📁 Project Files:")
    for file, exists in context['project_files'].items():
        status = "✅" if exists else "❌"
        print(f"   {file}: {status}")

def print_workflow_result(result: Dict[str, Any]):
    """Print workflow execution result"""
    if result['success']:
        print("\n🎉 Workflow completed successfully!")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Research sources: {len(result.get('research_results', []))}")
        print(f"   Code files: {len(result.get('code_files', []))}")
        print(f"   Review feedback: {len(result.get('review_feedback', []))}")
        
        # Print deployment status
        deployment = result.get('deployment_status', {})
        if deployment:
            print(f"   Deployment: {deployment.get('status', 'unknown')}")
        
        # Print final messages
        messages = result.get('messages', [])
        if messages:
            print(f"\n💬 Final Messages:")
            for msg in messages[-3:]:  # Last 3 messages
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:200]
                    print(f"   {role}: {content}...")
        
    else:
        print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")

# ==================== MAIN FUNCTIONS ====================

def run_init_mode():
    """Initialize project brain"""
    print("🧠 Initializing project brain...")
    
    brain = ProjectBrain(".")
    result = brain.update_codebase_understanding()
    
    print(f"✅ {result}")
    print(f"📁 Brain saved to: {brain.brain_file}")
    
    # Print brain stats
    memory = brain.memory
    print(f"\n📊 Brain Statistics:")
    print(f"   Files analyzed: {len(memory.get('codebase_map', {}))}")
    print(f"   Functions mapped: {len(memory.get('function_graph', {}))}")
    print(f"   Decisions recorded: {len(memory.get('decisions', []))}")
    print(f"   GitHub examples: {len(memory.get('github_examples', []))}")

def run_workflow_mode(task: str, mode: str, reference: str = None):
    """Run the main workflow"""
    print(f"🚀 Starting {mode} workflow...")
    print(f"📝 Task: {task}")
    
    # Initialize brain
    brain = ProjectBrain(".")
    
    # Add learning context if requested
    if reference:
        print(f"📚 Learning from reference: {reference}")
        # This would integrate with the research tools
        task += f"\n\nReference repository: {reference}"
    
    # Run workflow
    result = run_workflow(task, brain, mode)
    
    # Print results
    print_workflow_result(result)
    
    return result

def run_research_mode(task: str, reference: str = None):
    """Run research-only mode"""
    print(f"🔍 Starting research mode...")
    print(f"📝 Research topic: {task}")
    
    # Initialize brain
    brain = ProjectBrain(".")
    
    # Add reference context
    if reference:
        task += f"\n\nReference repository: {reference}"
    
    # Run research workflow
    result = run_workflow(task, brain, "research_only")
    
    # Print research results
    if result['success']:
        research_results = result.get('research_results', [])
        print(f"\n📚 Research Results ({len(research_results)} sources):")
        
        for i, research in enumerate(research_results, 1):
            source = research.get('source', 'unknown')
            content = research.get('content', '')[:300]
            print(f"\n{i}. {source.upper()}:")
            print(f"   {content}...")
    
    return result

# ==================== CLI ENTRY ====================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LangGraph Multi-Agent Coding System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize project brain
  python main.py --init
  
  # Full workflow
  python main.py "Create a FastAPI web service for user management"
  
  # Quick workflow (no research)
  python main.py --quick "Add authentication to existing API"
  
  # Research only
  python main.py --research-only "Best practices for edge deployment"
  
  # Learn from GitHub repo
  python main.py --learn-from github --reference "https://github.com/user/repo" "Implement similar features"
        """
    )
    
    parser.add_argument("task", nargs="?", help="Task description")
    parser.add_argument("--init", action="store_true", help="Initialize project brain")
    parser.add_argument("--quick", action="store_true", help="Quick mode (no research)")
    parser.add_argument("--research-only", action="store_true", help="Research only mode")
    parser.add_argument("--learn-from", choices=["github", "web"], help="Learn from external sources")
    parser.add_argument("--reference", help="Reference GitHub repository or URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check Python version
    if sys.version_info < (3, 12):
        print("⚠️  Warning: Python 3.12+ recommended for optimal performance")
    
    # Analyze context
    context = analyze_project_context()
    if args.verbose:
        print_context(context)
    
    # Check API keys
    if not any(context['api_keys'].values()):
        print("❌ No API keys found!")
        print("Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY")
        print("Optional: GITHUB_TOKEN for enhanced research")
        return 1
    
    try:
        # Handle --init
        if args.init:
            run_init_mode()
            return 0
        
        # Check for task
        if not args.task:
            parser.print_help()
            return 1
        
        # Determine mode
        if args.research_only:
            mode = "research_only"
        elif args.quick:
            mode = "quick"
        else:
            mode = "full"
        
        # Run appropriate mode
        if mode == "research_only":
            result = run_research_mode(args.task, args.reference)
        else:
            result = run_workflow_mode(args.task, mode, args.reference)
        
        # Return appropriate exit code
        return 0 if result['success'] else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 