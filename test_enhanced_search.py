#!/usr/bin/env python3
"""
Test script for enhanced search capabilities
Demonstrates Medium, Dev.to, Hashnode, and developer blog search
"""

import asyncio
import os
from search_service import SearchService, create_search_config, SearchResult

async def test_enhanced_search():
    """Test the enhanced search capabilities"""
    
    print("🔍 Testing Enhanced Search Capabilities")
    print("=" * 50)
    
    # Create search configuration
    config = create_search_config()
    search_service = SearchService(config)
    
    # Test queries
    test_queries = [
        "Python async programming",
        "React hooks best practices", 
        "AI coding assistants 2025",
        "Web development trends",
        "Machine learning tutorials"
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")
        print("-" * 30)
        
        try:
            # Search all providers
            all_results = await search_service.search_all(query)
            
            print(f"Found results from {len(all_results)} providers:")
            for provider, results in all_results.items():
                print(f"  {provider}: {len(results)} results")
            
            # Get aggregated results with advanced features
            aggregated = await search_service.search_aggregated(query, max_results=5)
            
            if aggregated:
                print(f"\nTop {len(aggregated)} aggregated results (with semantic enhancement):")
                print(search_service.format_results(aggregated))
            else:
                print("No relevant results found.")
                
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
        
        print("\n" + "="*50)

async def test_semantic_enhancement():
    """Test semantic search enhancement"""
    
    print("\n🧠 Testing Semantic Search Enhancement")
    print("=" * 50)
    
    config = create_search_config()
    search_service = SearchService(config)
    
    # Test semantic enhancement
    query = "Python web development"
    print(f"Original query: '{query}'")
    
    # Get semantic variations
    enhanced_queries = search_service.semantic_enhancer.enhance_query(query)
    print(f"Semantic variations: {enhanced_queries}")
    
    # Search with semantic enhancement
    results = await search_service.search_with_semantic_enhancement(query, max_results=5)
    
    if results:
        print(f"\nSemantic-enhanced results:")
        print(search_service.format_results(results))
    else:
        print("No semantic-enhanced results found.")

def test_analytics():
    """Test search analytics"""
    
    print("\n📊 Testing Search Analytics")
    print("=" * 50)
    
    config = create_search_config()
    search_service = SearchService(config)
    
    # Get search insights
    insights = search_service.get_search_insights()
    
    if "message" in insights:
        print("No search history available yet. Run some searches first.")
    else:
        print("Search Analytics Insights:")
        print(f"  Total searches: {insights['total_searches']}")
        print(f"  Avg results per search: {insights['avg_results_per_search']}")
        print(f"  Avg relevance score: {insights['avg_relevance_score']}")
        
        print("\nMost common sources:")
        for source, count in insights['most_common_sources']:
            print(f"  {source}: {count}")
        
        print("\nMost common queries:")
        for query, count in insights['most_common_queries']:
            print(f"  '{query}': {count}")

def test_content_categorization():
    """Test content categorization and summarization"""
    
    print("\n📝 Testing Content Categorization")
    print("=" * 50)
    
    config = create_search_config()
    search_service = SearchService(config)
    
    # Test different content types
    test_results = [
        SearchResult(
            title="How to Build a REST API with FastAPI",
            description="A comprehensive tutorial on building REST APIs using FastAPI framework",
            url="https://example.com/fastapi-tutorial",
            source="Tutorial Site",
            tags=["python", "fastapi", "tutorial"]
        ),
        SearchResult(
            title="React useState Hook Not Working",
            description="I'm having issues with the useState hook in my React component",
            url="https://example.com/react-question",
            source="Q&A Site",
            tags=["react", "javascript", "question"]
        ),
        SearchResult(
            title="New Python 3.12 Features Released",
            description="Latest announcement about Python 3.12 new features and improvements",
            url="https://example.com/python-news",
            source="News Site",
            tags=["python", "news", "release"]
        )
    ]
    
    for result in test_results:
        summary = search_service.content_summarizer.summarize_result(result)
        print(f"Original: {result.title}")
        print(f"Summary:  {summary}")
        print()

def test_individual_providers():
    """Test individual search providers"""
    
    print("\n🧪 Testing Individual Providers")
    print("=" * 50)
    
    config = create_search_config()
    search_service = SearchService(config)
    
    query = "Python web development"
    
    for provider_name, provider in search_service.providers.items():
        print(f"\n🔍 Testing {provider_name} provider...")
        try:
            results = asyncio.run(provider.search(query))
            print(f"  Found {len(results)} results")
            
            if results:
                print("  Top result:")
                top_result = results[0]
                print(f"    Title: {top_result.title}")
                print(f"    Source: {top_result.source}")
                print(f"    Relevance: {top_result.relevance_score:.2f}")
                print(f"    URL: {top_result.url}")
            
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Main test function"""
    
    print("🚀 Enhanced Search Test Suite")
    print("=" * 60)
    
    # Check configuration
    config = create_search_config()
    print("\n📋 Search Configuration:")
    print(f"  Medium API Key: {'✅' if config['medium_api_key'] else '❌'}")
    print(f"  Dev.to API Key: {'✅' if config['dev_to_api_key'] else '❌'}")
    print(f"  Hashnode API Key: {'✅' if config['hashnode_api_key'] else '❌'}")
    print(f"  Google CSE ID: {'✅' if config['google_cse_id'] else '❌'}")
    print(f"  Google API Key: {'✅' if config['google_api_key'] else '❌'}")
    print(f"  Bing API Key: {'✅' if config['bing_api_key'] else '❌'}")
    print(f"  arXiv Email: {'✅' if config['arxiv_email'] else '❌'}")
    
    print("\n💡 Note: RSS-based providers (Medium, Dev.to, Hashnode, Blogs) work without API keys!")
    
    # Run tests
    try:
        asyncio.run(test_enhanced_search())
        asyncio.run(test_semantic_enhancement())
        test_analytics()
        test_content_categorization()
        test_individual_providers()
        
        print("\n✅ All tests completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")

if __name__ == "__main__":
    main() 