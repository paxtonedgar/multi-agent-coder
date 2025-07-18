#!/usr/bin/env python3
"""
Enhanced Search CLI Tool
Command-line interface for the multi-source search capabilities
"""

import asyncio
import argparse
import json
import sys
from typing import List, Dict, Any
from search_service import SearchService, create_search_config, SearchResult

class SearchCLI:
    """Command-line interface for enhanced search"""
    
    def __init__(self):
        self.config = create_search_config()
        self.search_service = SearchService(self.config)
    
    async def search(self, query: str, max_results: int = 10, semantic: bool = False, 
                    providers: List[str] = None, format_output: str = "text") -> str:
        """Perform search with specified options"""
        
        print(f"🔍 Searching for: '{query}'")
        print(f"📊 Max results: {max_results}")
        print(f"🧠 Semantic enhancement: {'✅' if semantic else '❌'}")
        
        if providers:
            print(f"🎯 Providers: {', '.join(providers)}")
        else:
            print(f"🎯 Providers: All available ({len(self.search_service.providers)})")
        
        print("-" * 50)
        
        try:
            if semantic:
                results = await self.search_service.search_with_semantic_enhancement(query, max_results)
            else:
                results = await self.search_service.search_aggregated(query, max_results)
            
            if not results:
                return "No relevant results found."
            
            if format_output == "json":
                return self._format_json(results)
            elif format_output == "markdown":
                return self._format_markdown(results)
            else:
                return self.search_service.format_results(results)
                
        except Exception as e:
            return f"Search failed: {e}"
    
    def _format_json(self, results: List[SearchResult]) -> str:
        """Format results as JSON"""
        json_results = []
        for result in results:
            json_results.append({
                "title": result.title,
                "description": result.description,
                "url": result.url,
                "source": result.source,
                "published_date": result.published_date,
                "relevance_score": result.relevance_score,
                "tags": result.tags
            })
        
        return json.dumps(json_results, indent=2)
    
    def _format_markdown(self, results: List[SearchResult]) -> str:
        """Format results as Markdown"""
        markdown = f"# Search Results\n\n"
        markdown += f"**Query:** {results[0].title if results else 'No results'}\n\n"
        
        for i, result in enumerate(results, 1):
            markdown += f"## {i}. {result.title}\n\n"
            markdown += f"{result.description}\n\n"
            markdown += f"**Source:** {result.source}\n"
            markdown += f"**URL:** {result.url}\n"
            markdown += f"**Relevance:** {result.relevance_score:.2f}\n"
            if result.tags:
                markdown += f"**Tags:** {', '.join(result.tags)}\n"
            markdown += f"**Date:** {result.published_date or 'Unknown'}\n\n"
            markdown += "---\n\n"
        
        return markdown
    
    def show_analytics(self) -> str:
        """Show search analytics"""
        insights = self.search_service.get_search_insights()
        
        if "message" in insights:
            return "No search history available yet."
        
        output = "📊 Search Analytics\n"
        output += "=" * 30 + "\n\n"
        output += f"Total searches: {insights['total_searches']}\n"
        output += f"Average results per search: {insights['avg_results_per_search']}\n"
        output += f"Average relevance score: {insights['avg_relevance_score']}\n\n"
        
        output += "Most common sources:\n"
        for source, count in insights['most_common_sources']:
            output += f"  {source}: {count}\n"
        
        output += "\nMost common queries:\n"
        for query, count in insights['most_common_queries']:
            output += f"  '{query}': {count}\n"
        
        return output
    
    def show_providers(self) -> str:
        """Show available search providers"""
        output = "🔍 Available Search Providers\n"
        output += "=" * 30 + "\n\n"
        
        for name, provider in self.search_service.providers.items():
            output += f"• {name}: {provider.__class__.__name__}\n"
        
        output += f"\nTotal providers: {len(self.search_service.providers)}"
        return output
    
    def show_config(self) -> str:
        """Show current configuration"""
        output = "⚙️  Search Configuration\n"
        output += "=" * 30 + "\n\n"
        
        for key, value in self.config.items():
            if "key" in key.lower() or "token" in key.lower():
                display_value = "✅ Set" if value else "❌ Not set"
            else:
                display_value = str(value)
            output += f"{key}: {display_value}\n"
        
        return output

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Enhanced Search CLI - Multi-source developer content search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search_cli.py "Python async programming"
  python search_cli.py "React hooks" --max-results 5 --semantic
  python search_cli.py "AI coding assistants" --format json
  python search_cli.py --analytics
  python search_cli.py --providers
  python search_cli.py --config
        """
    )
    
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--max-results", "-m", type=int, default=10, 
                       help="Maximum number of results (default: 10)")
    parser.add_argument("--semantic", "-s", action="store_true",
                       help="Enable semantic search enhancement")
    parser.add_argument("--providers", "-p", nargs="+", 
                       help="Specific providers to search (default: all)")
    parser.add_argument("--format", "-f", choices=["text", "json", "markdown"], 
                       default="text", help="Output format (default: text)")
    parser.add_argument("--analytics", "-a", action="store_true",
                       help="Show search analytics")
    parser.add_argument("--show-providers", action="store_true",
                       help="Show available search providers")
    parser.add_argument("--config", "-c", action="store_true",
                       help="Show current configuration")
    
    args = parser.parse_args()
    
    cli = SearchCLI()
    
    # Handle different modes
    if args.analytics:
        print(cli.show_analytics())
        return
    
    if args.show_providers:
        print(cli.show_providers())
        return
    
    if args.config:
        print(cli.show_config())
        return
    
    if not args.query:
        parser.print_help()
        return
    
    # Perform search
    try:
        result = asyncio.run(cli.search(
            query=args.query,
            max_results=args.max_results,
            semantic=args.semantic,
            providers=args.providers,
            format_output=args.format
        ))
        print(result)
        
    except KeyboardInterrupt:
        print("\n⏹️  Search interrupted by user")
    except Exception as e:
        print(f"❌ Search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 