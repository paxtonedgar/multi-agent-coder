#!/usr/bin/env python3
"""
Main entry point for the multi-agent coding system
Enhanced with HF routing and dynamic model selection
"""

import os
import sys
import argparse
import asyncio
from typing import Optional

from memory import ProjectBrain
from graph import run_workflow

def main():
    """Main entry point with enhanced CLI options"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Coding System with HF Routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py "Build a REST API with FastAPI"
  
  # Use specific HF model
  python main.py "Solve complex math problem" --use-hf DeepSeek-V3
  
  # Auto-routing for reasoning tasks
  python main.py "Implement advanced algorithm" --use-hf auto-reasoning
  
  # Update model discovery
  python main.py "Research latest AI models" --update-models
  
  # Quick mode with HF routing
  python main.py "Create simple web app" --mode quick --use-hf auto-reasoning
        """
    )
    
    parser.add_argument(
        "task",
        help="The task to accomplish (e.g., 'Build a REST API')"
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "research_only", "minimal"],
        default="full",
        help="Workflow mode: full (default), quick, research_only, or minimal"
    )
    
    parser.add_argument(
        "--repo-url",
        help="GitHub repository URL for integration"
    )
    
    parser.add_argument(
        "--use-hf",
        help="Hugging Face model to use: 'auto-reasoning' (default), specific model name, or 'none'"
    )
    
    parser.add_argument(
        "--update-models",
        action="store_true",
        help="Force refresh of reasoning model discovery"
    )
    
    parser.add_argument(
        "--force-hf",
        action="store_true",
        help="Force routing to HF models for all tasks"
    )
    
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show HF routing statistics and exit"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available reasoning models and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize brain
    brain = ProjectBrain()
    
    # Handle special commands
    if args.show_stats:
        _show_routing_stats(brain)
        return
    
    if args.list_models:
        _list_available_models(brain)
        return
    
    # Update models if requested
    if args.update_models:
        _update_model_discovery(brain)
    
    # Configure HF routing
    if args.use_hf:
        _configure_hf_routing(brain, args.use_hf, args.force_hf)
    
    # Run workflow
    print(f"🚀 Starting {args.mode} workflow for: {args.task}")
    print(f"📊 HF Routing: {args.use_hf or 'disabled'}")
    
    try:
        result = run_workflow(
            task=args.task,
            brain=brain,
            mode=args.mode,
            repo_url=args.repo_url or ""
        )
        
        if result.get('success'):
            print("\n✅ Workflow completed successfully!")
            
            # Handle output truncation for research mode
            if args.mode == "research_only":
                research_results = result.get('research_results', [])
                if research_results:
                    print("\n📊 Research Results:")
                    for i, research in enumerate(research_results[:3], 1):  # Show first 3
                        content = research.get('content', '')
                        if len(content) > 1000:
                            content = content[:1000] + '\n[Truncated - Full results in logs]'
                        print(f"\n{i}. {research.get('source', 'Unknown')}:")
                        print(content)
                else:
                    print("📝 No research results available")
            else:
                # For other modes, show final result
                final_result = result.get('final_result', 'No result available')
                if len(final_result) > 1000:
                    final_result = final_result[:1000] + '\n[Truncated - Full result in logs]'
                print(f"📝 Final result: {final_result}")
        else:
            print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        sys.exit(1)

def _show_routing_stats(brain: ProjectBrain):
    """Show HF routing statistics"""
    try:
        from hf_routing import HFRoutingSystem
        router = HFRoutingSystem(brain)
        stats = router.get_routing_stats()
        
        print("📊 HF Routing Statistics:")
        print(f"  Total routes: {stats['total_routes']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Refusal rate: {stats['refusal_rate']:.2%}")
        print(f"  Models used: {stats['models_used']}")
        
    except ImportError:
        print("❌ HF routing not available")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

def _list_available_models(brain: ProjectBrain):
    """List available reasoning models"""
    try:
        from hf_routing import ReasoningModelDiscovery
        discovery = ReasoningModelDiscovery(brain)
        models = discovery.discover_reasoning_models()
        
        print("🤖 Available Reasoning Models:")
        for i, model in enumerate(models[:10], 1):  # Show top 10
            print(f"  {i}. {model.name}")
            print(f"     Score: {model.reasoning_score:.2f}")
            print(f"     Parameters: {model.parameters}B")
            print(f"     Mac optimized: {'✅' if model.mac_optimized else '❌'}")
            print(f"     Source: {model.source}")
            print()
        
    except ImportError:
        print("❌ HF routing not available")
    except Exception as e:
        print(f"❌ Error listing models: {e}")

def _update_model_discovery(brain: ProjectBrain):
    """Update model discovery cache"""
    try:
        from hf_routing import ReasoningModelDiscovery
        discovery = ReasoningModelDiscovery(brain)
        models = discovery.discover_reasoning_models(force_refresh=True)
        print(f"✅ Updated model discovery: {len(models)} models found")
        
    except ImportError:
        print("❌ HF routing not available")
    except Exception as e:
        print(f"❌ Error updating models: {e}")

def _configure_hf_routing(brain: ProjectBrain, model_name: str, force_hf: bool):
    """Configure HF routing settings"""
    try:
        # Store routing preferences in brain
        brain.memory['hf_routing_config'] = {
            'model_name': model_name,
            'force_hf': force_hf,
            'enabled': model_name != 'none'
        }
        brain._save()
        
        if model_name == 'none':
            print("🚫 HF routing disabled")
        elif model_name == 'auto-reasoning':
            print("🤖 HF routing enabled with auto-reasoning")
        else:
            print(f"🤖 HF routing enabled with model: {model_name}")
            
    except Exception as e:
        print(f"❌ Error configuring HF routing: {e}")

if __name__ == "__main__":
    main() 