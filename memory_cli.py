#!/usr/bin/env python3
"""
CLI interface for the enhanced memory system
Demonstrates persistent reflections, tree structures, and self-awareness
"""

import argparse
import json
import sys
from datetime import datetime
from memory import ProjectBrain, NodeType
from agents import create_meta_agent

def main():
    parser = argparse.ArgumentParser(description="Enhanced Memory System CLI")
    parser.add_argument("--project-path", default=".", help="Project path for brain")
    parser.add_argument("--command", required=True, choices=[
        "init", "add", "refine", "tree", "visualize", "reflect", "context", "prune", "stats"
    ], help="Command to execute")
    
    # Command-specific arguments
    parser.add_argument("--type", choices=["chat", "reflection", "decision", "learning", "code_change"], 
                       help="Node type for add command")
    parser.add_argument("--content", help="Content for add/refine commands")
    parser.add_argument("--parent-id", help="Parent node ID for hierarchical structure")
    parser.add_argument("--sources", nargs="*", help="Sources for reflections")
    parser.add_argument("--query", help="Query for tree/context commands")
    parser.add_argument("--depth", type=int, default=3, help="Tree depth for retrieval")
    parser.add_argument("--thread-id", help="Thread ID for context command")
    parser.add_argument("--output", help="Output path for visualization")
    parser.add_argument("--days-old", type=int, default=30, help="Days old for pruning")
    parser.add_argument("--reflection-id", help="Reflection ID for refine command")
    
    args = parser.parse_args()
    
    # Initialize brain
    brain = ProjectBrain(args.project_path)
    
    try:
        if args.command == "init":
            cmd_init(brain)
        elif args.command == "add":
            cmd_add(brain, args)
        elif args.command == "refine":
            cmd_refine(brain, args)
        elif args.command == "tree":
            cmd_tree(brain, args)
        elif args.command == "visualize":
            cmd_visualize(brain, args)
        elif args.command == "reflect":
            cmd_reflect(brain, args)
        elif args.command == "context":
            cmd_context(brain, args)
        elif args.command == "prune":
            cmd_prune(brain, args)
        elif args.command == "stats":
            cmd_stats(brain)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def cmd_init(brain):
    """Initialize the brain with sample data"""
    print("🧠 Initializing Enhanced Memory System...")
    
    # Add sample conversation thread
    chat1_id = brain.add_node(
        node_type=NodeType.CHAT,
        content="User: I need to build a web scraper for sports betting data",
        metadata={"thread_id": "sports_scraper_1"}
    )
    
    reflection1_id = brain.add_node(
        node_type=NodeType.REFLECTION,
        content="Need to research anti-detection techniques for sports betting sites",
        parent_id=chat1_id,
        sources=["https://example.com/anti-detection"]
    )
    
    decision1_id = brain.add_node(
        node_type=NodeType.DECISION,
        content="Will use Playwright with stealth plugins for anti-detection",
        parent_id=reflection1_id,
        metadata={"impact": "high", "reasoning": "Best practice for 2025"}
    )
    
    chat2_id = brain.add_node(
        node_type=NodeType.CHAT,
        content="User: What about deployment options?",
        metadata={"thread_id": "sports_scraper_1"}
    )
    
    decision2_id = brain.add_node(
        node_type=NodeType.DECISION,
        content="Deploy to Cloudflare Workers for edge performance and global distribution",
        parent_id=chat2_id,
        metadata={"impact": "medium", "reasoning": "Edge computing for low latency"}
    )
    
    print("✅ Sample data initialized:")
    print(f"  - Chat nodes: 2")
    print(f"  - Reflection nodes: 1")
    print(f"  - Decision nodes: 2")
    print(f"  - Thread: sports_scraper_1")

def cmd_add(brain, args):
    """Add a new node to the memory tree"""
    if not args.type or not args.content:
        print("Error: --type and --content are required for add command")
        sys.exit(1)
    
    node_type = NodeType(args.type)
    
    node_id = brain.add_node(
        node_type=node_type,
        content=args.content,
        parent_id=args.parent_id,
        sources=args.sources,
        metadata={"cli_created": True, "timestamp": datetime.now().isoformat()}
    )
    
    print(f"✅ Added {args.type} node: {node_id}")
    print(f"📝 Content: {args.content[:100]}...")
    if args.parent_id:
        print(f"🔗 Parent: {args.parent_id}")

def cmd_refine(brain, args):
    """Refine an existing reflection"""
    if not args.reflection_id or not args.content:
        print("Error: --reflection-id and --content are required for refine command")
        sys.exit(1)
    
    child_id = brain.refine_reflection(
        args.reflection_id,
        args.content,
        args.sources
    )
    
    print(f"✅ Refined reflection: {child_id}")
    print(f"🔗 Parent: {args.reflection_id}")
    print(f"📝 New content: {args.content[:100]}...")

def cmd_tree(brain, args):
    """Get reflection tree for a query"""
    if not args.query:
        print("Error: --query is required for tree command")
        sys.exit(1)
    
    tree = brain.get_reflection_tree(args.query, depth=args.depth)
    
    print(f"🌳 Reflection Tree for: '{args.query}'")
    print(f"📊 Found {len(tree['nodes'])} nodes")
    print(f"📝 Summary: {tree['summary']}")
    
    if tree['nodes']:
        print("\n📋 Nodes:")
        for i, node in enumerate(tree['nodes'][:5]):  # Show first 5
            print(f"  {i+1}. [{node['type']}] {node['content'][:80]}...")
            if node.get('sources'):
                print(f"     Sources: {', '.join(node['sources'])}")

def cmd_visualize(brain, args):
    """Visualize the memory tree"""
    result = brain.visualize_tree(output_path=args.output)
    print(f"📊 {result}")

def cmd_reflect(brain, args):
    """Generate reflection using MetaAgent"""
    if not args.content:
        print("Error: --content is required for reflect command")
        sys.exit(1)
    
    try:
        meta_agent = create_meta_agent(brain)
        
        reflection_prompt = f"""
        Reflect on this: {args.content}
        
        Consider:
        1. What problems does this reveal?
        2. What improvements could be made?
        3. How should we evolve the system?
        4. What research backs your insights?
        
        Generate actionable insights and store them as reflections.
        """
        
        result = meta_agent.invoke({
            "input": reflection_prompt,
            "chat_history": []
        })
        
        print("🤖 MetaAgent Reflection:")
        print("=" * 50)
        print(result.get('output', 'No output generated'))
        
    except Exception as e:
        print(f"⚠️  MetaAgent reflection failed: {e}")
        print("This is expected without API keys - using mock mode")

def cmd_context(brain, args):
    """Get conversation context for a thread"""
    if not args.thread_id:
        print("Error: --thread-id is required for context command")
        sys.exit(1)
    
    context = brain.get_conversation_context(args.thread_id)
    
    print(f"💬 Conversation Context for: {args.thread_id}")
    print(f"📊 Found {len(context['nodes'])} nodes")
    print(f"📝 Summary: {context['summary']}")
    
    if context['nodes']:
        print("\n📋 Thread Nodes:")
        for i, node in enumerate(context['nodes']):
            print(f"  {i+1}. [{node['type']}] {node['content'][:80]}...")

def cmd_prune(brain, args):
    """Prune old memory branches"""
    print(f"🧹 Pruning nodes older than {args.days_old} days...")
    
    # Count nodes before pruning
    nodes_before = len(brain.memory.get('memory_nodes', []))
    
    brain.prune_old_branches(days_old=args.days_old, similarity_threshold=0.3)
    
    # Count nodes after pruning
    nodes_after = len(brain.memory.get('memory_nodes', []))
    pruned = nodes_before - nodes_after
    
    print(f"✅ Pruning completed: {pruned} nodes removed")
    print(f"📊 Remaining nodes: {nodes_after}")

def cmd_stats(brain):
    """Show memory system statistics"""
    nodes = brain.memory.get('memory_nodes', [])
    
    # Count by type
    type_counts = {}
    for node in nodes:
        node_type = node.get('type', 'unknown')
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    # Count parent-child relationships
    parent_child_count = len([n for n in nodes if n.get('parent_id')])
    
    # Graph statistics
    graph_nodes = len(brain.conversation_tree.nodes())
    graph_edges = len(brain.conversation_tree.edges())
    
    print("📈 Memory System Statistics")
    print("=" * 40)
    print(f"📊 Total nodes: {len(nodes)}")
    print(f"🌳 Graph nodes: {graph_nodes}")
    print(f"🔗 Graph edges: {graph_edges}")
    print(f"🔗 Parent-child relationships: {parent_child_count}")
    
    print("\n📋 By Type:")
    for node_type, count in sorted(type_counts.items(), key=lambda x: str(x[0])):
        print(f"  {node_type}: {count}")
    
    # Reflection evolution stats
    reflection_nodes = [n for n in nodes if n.get('type') == 'reflection']
    if reflection_nodes:
        print(f"\n🤔 Reflection Evolution:")
        print(f"  Total reflections: {len(reflection_nodes)}")
        print(f"  With refinements: {len([n for n in reflection_nodes if n.get('refinements')])}")
        print(f"  With sources: {len([n for n in reflection_nodes if n.get('sources')])}")
    
    # Thread statistics
    thread_nodes = {}
    for node in nodes:
        thread_id = node.get('metadata', {}).get('thread_id')
        if thread_id:
            thread_nodes[thread_id] = thread_nodes.get(thread_id, 0) + 1
    
    if thread_nodes:
        print(f"\n💬 Conversation Threads:")
        for thread_id, count in sorted(thread_nodes.items()):
            print(f"  {thread_id}: {count} nodes")

if __name__ == "__main__":
    main() 