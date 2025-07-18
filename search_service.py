#!/usr/bin/env python3
"""
Centralized Search Service for Multi-Agent Coder
Provides unified interface for multiple search providers with developer focus
"""

import os
import time
import asyncio
import requests
import feedparser
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from diskcache import Cache

# Initialize cache
cache = Cache('.cache')

@dataclass
class SearchResult:
    """Standardized search result"""
    title: str
    description: str
    url: str
    source: str
    published_date: Optional[str] = None
    relevance_score: float = 0.0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class SearchProvider(ABC):
    """Abstract base class for search providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)
        self.max_results = config.get('max_results', 5)
    
    @abstractmethod
    async def search(self, query: str) -> List[SearchResult]:
        """Search for content"""
        pass
    
    def _add_developer_context(self, query: str) -> str:
        """Add developer context to search query"""
        developer_terms = [
            "programming", "coding", "development", "software",
            "python", "javascript", "web development", "ai", "machine learning"
        ]
        
        # Add relevant developer terms if not already present
        query_lower = query.lower()
        missing_terms = [term for term in developer_terms if term not in query_lower]
        
        if missing_terms:
            # Add 1-2 most relevant terms
            relevant_terms = missing_terms[:2]
            return f"{query} {' '.join(relevant_terms)}"
        
        return query

class MediumSearchProvider(SearchProvider):
    """Medium search provider using RSS feeds"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feed_urls = [
            "https://medium.com/feed/tag/python",
            "https://medium.com/feed/tag/javascript",
            "https://medium.com/feed/tag/web-development",
            "https://medium.com/feed/tag/ai",
            "https://medium.com/feed/tag/machine-learning",
            "https://medium.com/feed/tag/react",
            "https://medium.com/feed/tag/nodejs",
            "https://medium.com/feed/tag/typescript"
        ]
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search Medium articles via RSS feeds"""
        results = []
        
        for feed_url in self.feed_urls[:3]:  # Limit to avoid rate limits
            try:
                # Use feedparser for better RSS handling
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:3]:
                    title = entry.get('title', '')
                    description = entry.get('summary', '')
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    
                    # Calculate relevance score
                    relevance = self._calculate_relevance(query, title, description)
                    
                    if relevance > 0.3:  # Only include relevant results
                        result = SearchResult(
                            title=title,
                            description=description[:200] + "..." if len(description) > 200 else description,
                            url=link,
                            source="Medium",
                            published_date=published,
                            relevance_score=relevance,
                            tags=self._extract_tags(entry)
                        )
                        results.append(result)
                
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"Medium RSS error for {feed_url}: {e}")
                continue
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:self.max_results]
    
    def _calculate_relevance(self, query: str, title: str, description: str) -> float:
        """Calculate relevance score for Medium article"""
        query_terms = query.lower().split()
        title_lower = title.lower()
        desc_lower = description.lower()
        
        score = 0.0
        
        for term in query_terms:
            if term in title_lower:
                score += 0.5  # Title matches are more important
            if term in desc_lower:
                score += 0.2
        
        # Bonus for developer-related terms
        dev_terms = ['python', 'javascript', 'react', 'node', 'api', 'web', 'development']
        for term in dev_terms:
            if term in title_lower or term in desc_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def _extract_tags(self, entry) -> List[str]:
        """Extract tags from Medium entry"""
        tags = []
        if hasattr(entry, 'tags'):
            tags = [tag.term for tag in entry.tags]
        return tags

class DevToSearchProvider(SearchProvider):
    """Dev.to search provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = "https://dev.to/api/articles"
        self.tags = ["python", "javascript", "webdev", "react", "node", "ai", "machine-learning"]
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search Dev.to articles"""
        results = []
        
        for tag in self.tags[:2]:  # Limit to avoid rate limits
            try:
                response = requests.get(
                    self.api_url,
                    params={
                        "tag": tag,
                        "top": 10,
                        "per_page": 5
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    articles = response.json()
                    
                    for article in articles:
                        title = article.get('title', '')
                        description = article.get('description', '')
                        url = article.get('url', '')
                        published = article.get('published_at', '')
                        tags = article.get('tags', [])
                        
                        # Calculate relevance
                        relevance = self._calculate_relevance(query, title, description)
                        
                        if relevance > 0.3:
                            result = SearchResult(
                                title=title,
                                description=description[:200] + "..." if len(description) > 200 else description,
                                url=url,
                                source="Dev.to",
                                published_date=published,
                                relevance_score=relevance,
                                tags=tags
                            )
                            results.append(result)
                
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"Dev.to search error for tag {tag}: {e}")
                continue
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:self.max_results]
    
    def _calculate_relevance(self, query: str, title: str, description: str) -> float:
        """Calculate relevance score for Dev.to article"""
        query_terms = query.lower().split()
        title_lower = title.lower()
        desc_lower = description.lower()
        
        score = 0.0
        
        for term in query_terms:
            if term in title_lower:
                score += 0.5
            if term in desc_lower:
                score += 0.2
        
        return min(score, 1.0)

class HashnodeSearchProvider(SearchProvider):
    """Hashnode search provider using GraphQL API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = "https://api.hashnode.com/"
        self.tags = ["python", "javascript", "web-development", "ai", "react", "nodejs"]
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search Hashnode articles"""
        results = []
        
        graphql_query = """
        query GetArticles($tag: String!) {
            articles(first: 5, filter: {tagSlug: $tag}) {
                edges {
                    node {
                        title
                        brief
                        url
                        author {
                            name
                        }
                        tags {
                            name
                        }
                        publishedAt
                    }
                }
            }
        }
        """
        
        for tag in self.tags[:2]:
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "query": graphql_query,
                        "variables": {"tag": tag}
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('data', {}).get('articles', {}).get('edges', [])
                    
                    for edge in articles:
                        article = edge['node']
                        title = article.get('title', '')
                        brief = article.get('brief', '')
                        url = article.get('url', '')
                        published = article.get('publishedAt', '')
                        tags = [tag['name'] for tag in article.get('tags', [])]
                        
                        relevance = self._calculate_relevance(query, title, brief)
                        
                        if relevance > 0.3:
                            result = SearchResult(
                                title=title,
                                description=brief[:200] + "..." if len(brief) > 200 else brief,
                                url=url,
                                source="Hashnode",
                                published_date=published,
                                relevance_score=relevance,
                                tags=tags
                            )
                            results.append(result)
                
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"Hashnode search error for tag {tag}: {e}")
                continue
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:self.max_results]
    
    def _calculate_relevance(self, query: str, title: str, brief: str) -> float:
        """Calculate relevance score for Hashnode article"""
        query_terms = query.lower().split()
        title_lower = title.lower()
        brief_lower = brief.lower()
        
        score = 0.0
        
        for term in query_terms:
            if term in title_lower:
                score += 0.5
            if term in brief_lower:
                score += 0.2
        
        return min(score, 1.0)

class DeveloperBlogSearchProvider(SearchProvider):
    """Search popular developer blogs and tech sites"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.blog_feeds = [
            ("LogRocket", "https://blog.logrocket.com/feed/"),
            ("CSS Tricks", "https://css-tricks.com/feed/"),
            ("Smashing Magazine", "https://www.smashingmagazine.com/feed/"),
            ("A List Apart", "https://alistapart.com/main/feed/"),
            ("Web.dev", "https://web.dev/feed.xml"),
            ("Google Developers", "https://developers.google.com/web/updates/rss.xml"),
            ("React Blog", "https://reactjs.org/feed.xml"),
            ("Vue.js Blog", "https://vuejs.org/feed.xml"),
            ("Angular Blog", "https://angular.io/feed.xml")
        ]
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search developer blogs via RSS feeds"""
        results = []
        
        for blog_name, feed_url in self.blog_feeds[:3]:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:2]:
                    title = entry.get('title', '')
                    description = entry.get('summary', '')
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    
                    relevance = self._calculate_relevance(query, title, description)
                    
                    if relevance > 0.3:
                        result = SearchResult(
                            title=title,
                            description=description[:200] + "..." if len(description) > 200 else description,
                            url=link,
                            source=blog_name,
                            published_date=published,
                            relevance_score=relevance,
                            tags=self._extract_tags(entry)
                        )
                        results.append(result)
                
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"Blog search error for {blog_name}: {e}")
                continue
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:self.max_results]
    
    def _calculate_relevance(self, query: str, title: str, description: str) -> float:
        """Calculate relevance score for blog article"""
        query_terms = query.lower().split()
        title_lower = title.lower()
        desc_lower = description.lower()
        
        score = 0.0
        
        for term in query_terms:
            if term in title_lower:
                score += 0.5
            if term in desc_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _extract_tags(self, entry) -> List[str]:
        """Extract tags from blog entry"""
        tags = []
        if hasattr(entry, 'tags'):
            tags = [tag.term for tag in entry.tags]
        return tags

class ArxivSearchProvider(SearchProvider):
    """arXiv search provider for academic papers"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.email = config.get('arxiv_email', '')
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search arXiv for academic papers"""
        if not self.email:
            return []
        
        try:
            import arxiv
            
            # Add developer context to query
            enhanced_query = self._add_developer_context(query)
            
            search = arxiv.Search(
                query=enhanced_query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            results = []
            for result in search.results():
                relevance = self._calculate_relevance(query, result.title, result.summary)
                
                if relevance > 0.3:
                    search_result = SearchResult(
                        title=result.title,
                        description=result.summary[:200] + "..." if len(result.summary) > 200 else result.summary,
                        url=result.pdf_url,
                        source="arXiv",
                        published_date=result.published.strftime("%Y-%m-%d"),
                        relevance_score=relevance,
                        tags=[cat for cat in result.categories]
                    )
                    results.append(search_result)
            
            return results
            
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []
    
    def _calculate_relevance(self, query: str, title: str, summary: str) -> float:
        """Calculate relevance score for arXiv paper"""
        query_terms = query.lower().split()
        title_lower = title.lower()
        summary_lower = summary.lower()
        
        score = 0.0
        
        for term in query_terms:
            if term in title_lower:
                score += 0.6  # Title matches are very important for papers
            if term in summary_lower:
                score += 0.3
        
        return min(score, 1.0)

# ==================== ADDITIONAL SEARCH PROVIDERS ====================

class StackOverflowSearchProvider(SearchProvider):
    """Stack Overflow search provider using their API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = "https://api.stackexchange.com/2.3/search"
        self.site = "stackoverflow"
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search Stack Overflow questions and answers"""
        try:
            # Add developer context
            enhanced_query = self._add_developer_context(query)
            
            response = requests.get(
                self.api_url,
                params={
                    "order": "desc",
                    "sort": "activity",
                    "intitle": enhanced_query,
                    "site": self.site,
                    "pagesize": self.max_results,
                    "filter": "withbody"  # Include body content
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    title = item.get('title', '')
                    body = item.get('body', '')
                    link = item.get('link', '')
                    score = item.get('score', 0)
                    answer_count = item.get('answer_count', 0)
                    tags = item.get('tags', [])
                    created_date = item.get('creation_date', 0)
                    
                    # Calculate relevance
                    relevance = self._calculate_relevance(query, title, body)
                    
                    if relevance > 0.3:
                        # Convert timestamp to readable date
                        from datetime import datetime
                        created_str = datetime.fromtimestamp(created_date).strftime("%Y-%m-%d")
                        
                        result = SearchResult(
                            title=title,
                            description=body[:200] + "..." if len(body) > 200 else body,
                            url=link,
                            source="Stack Overflow",
                            published_date=created_str,
                            relevance_score=relevance + (score * 0.01) + (answer_count * 0.02),  # Boost by score and answers
                            tags=tags
                        )
                        results.append(result)
                
                results.sort(key=lambda x: x.relevance_score, reverse=True)
                return results[:self.max_results]
            
        except Exception as e:
            print(f"Stack Overflow search error: {e}")
        
        return []
    
    def _calculate_relevance(self, query: str, title: str, body: str) -> float:
        """Calculate relevance score for Stack Overflow content"""
        query_terms = query.lower().split()
        title_lower = title.lower()
        body_lower = body.lower()
        
        score = 0.0
        
        for term in query_terms:
            if term in title_lower:
                score += 0.6  # Title matches are very important for SO
            if term in body_lower:
                score += 0.2
        
        return min(score, 1.0)

class RedditSearchProvider(SearchProvider):
    """Reddit search provider for developer communities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.subreddits = [
            "programming", "Python", "javascript", "webdev", 
            "reactjs", "node", "MachineLearning", "datascience"
        ]
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search Reddit developer communities"""
        try:
            # Use Reddit's JSON API
            results = []
            
            for subreddit in self.subreddits[:2]:  # Limit to avoid rate limits
                try:
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    response = requests.get(
                        url,
                        params={
                            "q": query,
                            "restrict_sr": "on",
                            "sort": "relevance",
                            "t": "month",  # Last month
                            "limit": 5
                        },
                        headers={"User-Agent": "MultiAgentCoder/1.0"},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for post in data.get('data', {}).get('children', []):
                            post_data = post.get('data', {})
                            title = post_data.get('title', '')
                            selftext = post_data.get('selftext', '')
                            url = post_data.get('url', '')
                            score = post_data.get('score', 0)
                            num_comments = post_data.get('num_comments', 0)
                            created_utc = post_data.get('created_utc', 0)
                            
                            relevance = self._calculate_relevance(query, title, selftext)
                            
                            if relevance > 0.3:
                                from datetime import datetime
                                created_str = datetime.fromtimestamp(created_utc).strftime("%Y-%m-%d")
                                
                                result = SearchResult(
                                    title=title,
                                    description=selftext[:200] + "..." if len(selftext) > 200 else selftext,
                                    url=url,
                                    source=f"Reddit r/{subreddit}",
                                    published_date=created_str,
                                    relevance_score=relevance + (score * 0.001) + (num_comments * 0.002),
                                    tags=[subreddit]
                                )
                                results.append(result)
                
                except Exception as e:
                    print(f"Reddit search error for r/{subreddit}: {e}")
                    continue
                
                await asyncio.sleep(self.rate_limit_delay)
            
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:self.max_results]
            
        except Exception as e:
            print(f"Reddit search error: {e}")
            return []
    
    def _calculate_relevance(self, query: str, title: str, selftext: str) -> float:
        """Calculate relevance score for Reddit content"""
        query_terms = query.lower().split()
        title_lower = title.lower()
        text_lower = selftext.lower()
        
        score = 0.0
        
        for term in query_terms:
            if term in title_lower:
                score += 0.5
            if term in text_lower:
                score += 0.2
        
        return min(score, 1.0)

class HackerNewsSearchProvider(SearchProvider):
    """Hacker News search provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = "https://hn.algolia.com/api/v1/search"
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search Hacker News stories and comments"""
        try:
            response = requests.get(
                self.api_url,
                params={
                    "query": query,
                    "tags": "story",
                    "numericFilters": "points>10",  # Only stories with >10 points
                    "hitsPerPage": self.max_results
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for hit in data.get('hits', []):
                    title = hit.get('title', '')
                    url = hit.get('url', '')
                    points = hit.get('points', 0)
                    num_comments = hit.get('num_comments', 0)
                    created_at = hit.get('created_at', '')
                    author = hit.get('author', '')
                    
                    relevance = self._calculate_relevance(query, title, '')
                    
                    if relevance > 0.3:
                        result = SearchResult(
                            title=title,
                            description=f"Posted by {author} • {points} points • {num_comments} comments",
                            url=url,
                            source="Hacker News",
                            published_date=created_at[:10] if created_at else None,
                            relevance_score=relevance + (points * 0.001) + (num_comments * 0.002),
                            tags=["hacker-news", "tech-news"]
                        )
                        results.append(result)
                
                return results
            
        except Exception as e:
            print(f"Hacker News search error: {e}")
        
        return []
    
    def _calculate_relevance(self, query: str, title: str, content: str) -> float:
        """Calculate relevance score for Hacker News content"""
        query_terms = query.lower().split()
        title_lower = title.lower()
        
        score = 0.0
        
        for term in query_terms:
            if term in title_lower:
                score += 0.6
        
        return min(score, 1.0)

class TrendAnalysisProvider(SearchProvider):
    """Provider for analyzing current trends in developer content"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trend_keywords = [
            "2025", "latest", "new", "trending", "popular", "hot",
            "announcement", "release", "update", "beta", "alpha"
        ]
    
    async def search(self, query: str) -> List[SearchResult]:
        """Analyze trends related to the query"""
        try:
            # Check if query is trend-related
            is_trend_query = any(keyword in query.lower() for keyword in self.trend_keywords)
            
            if not is_trend_query:
                return []
            
            # Get trending topics from multiple sources
            trends = []
            
            # GitHub trending
            try:
                response = requests.get(
                    "https://github.com/trending",
                    headers={"User-Agent": "MultiAgentCoder/1.0"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract trending repos (simplified)
                    repo_links = soup.find_all('h2', class_='h3')
                    for i, link in enumerate(repo_links[:3]):
                        if link.find('a'):
                            repo_name = link.find('a').get_text(strip=True)
                            trends.append(f"GitHub Trending: {repo_name}")
            
            except Exception as e:
                print(f"GitHub trending error: {e}")
            
            # Create trend analysis results
            results = []
            if trends:
                result = SearchResult(
                    title=f"Trending Topics for: {query}",
                    description="Current trending topics in developer community:\n" + "\n".join(trends),
                    url="https://github.com/trending",
                    source="Trend Analysis",
                    published_date="Current",
                    relevance_score=0.8,
                    tags=["trending", "2025", "latest"]
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Trend analysis error: {e}")
            return []
    
    def _calculate_relevance(self, query: str, title: str, content: str) -> float:
        """Calculate relevance for trend analysis"""
        return 0.8  # High relevance for trend queries

class CodeExampleSearchProvider(SearchProvider):
    """Search for code examples and tutorials"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.code_sites = [
            ("GitHub Gist", "https://gist.github.com"),
            ("CodePen", "https://codepen.io"),
            ("JSFiddle", "https://jsfiddle.net"),
            ("Replit", "https://replit.com")
        ]
    
    async def search(self, query: str) -> List[SearchResult]:
        """Search for code examples"""
        try:
            # For now, return curated code example suggestions
            # In a full implementation, this would search actual code repositories
            
            code_examples = {
                "python": [
                    "https://github.com/trending?l=python",
                    "https://gist.github.com/trending?l=python"
                ],
                "javascript": [
                    "https://github.com/trending?l=javascript", 
                    "https://codepen.io/trending"
                ],
                "react": [
                    "https://github.com/trending?l=javascript&q=react",
                    "https://codepen.io/trending?type=react"
                ]
            }
            
            results = []
            query_lower = query.lower()
            
            for lang, urls in code_examples.items():
                if lang in query_lower:
                    for url in urls:
                        result = SearchResult(
                            title=f"Code Examples: {lang.title()}",
                            description=f"Find {lang} code examples and tutorials",
                            url=url,
                            source="Code Examples",
                            published_date="Current",
                            relevance_score=0.7,
                            tags=[lang, "code-examples", "tutorials"]
                        )
                        results.append(result)
            
            return results[:self.max_results]
            
        except Exception as e:
            print(f"Code example search error: {e}")
            return []
    
    def _calculate_relevance(self, query: str, title: str, content: str) -> float:
        """Calculate relevance for code examples"""
        return 0.7

# ==================== ADVANCED SEARCH FEATURES ====================

class SemanticSearchEnhancer:
    """Enhances search results with semantic analysis"""
    
    def __init__(self):
        self.semantic_keywords = {
            "python": ["django", "flask", "fastapi", "pandas", "numpy", "async", "asyncio"],
            "javascript": ["react", "vue", "angular", "node", "express", "typescript"],
            "web": ["frontend", "backend", "fullstack", "api", "rest", "graphql"],
            "ai": ["machine learning", "deep learning", "neural networks", "tensorflow", "pytorch"],
            "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "serverless"]
        }
    
    def enhance_query(self, query: str) -> List[str]:
        """Generate semantic variations of the query"""
        enhanced_queries = [query]
        query_lower = query.lower()
        
        for category, keywords in self.semantic_keywords.items():
            if category in query_lower:
                for keyword in keywords[:2]:  # Limit variations
                    enhanced_queries.append(f"{query} {keyword}")
        
        return enhanced_queries
    
    def calculate_semantic_relevance(self, query: str, content: str) -> float:
        """Calculate semantic relevance using keyword matching"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Count semantic keyword matches
        semantic_score = 0.0
        for category, keywords in self.semantic_keywords.items():
            if category in query_lower:
                for keyword in keywords:
                    if keyword in content_lower:
                        semantic_score += 0.1
        
        return min(semantic_score, 0.5)

class ContentSummarizer:
    """Summarizes search results for better presentation"""
    
    def __init__(self):
        self.summary_templates = {
            "tutorial": "📚 Tutorial: {title} - {summary}",
            "article": "📝 Article: {title} - {summary}",
            "question": "❓ Q&A: {title} - {summary}",
            "news": "📰 News: {title} - {summary}",
            "code": "💻 Code: {title} - {summary}"
        }
    
    def categorize_content(self, title: str, description: str, tags: List[str]) -> str:
        """Categorize content type"""
        text = f"{title} {description} {' '.join(tags)}".lower()
        
        if any(word in text for word in ["tutorial", "guide", "how to", "step by step"]):
            return "tutorial"
        elif any(word in text for word in ["question", "problem", "error", "issue"]):
            return "question"
        elif any(word in text for word in ["news", "announcement", "release", "update"]):
            return "news"
        elif any(word in text for word in ["code", "example", "snippet", "implementation"]):
            return "code"
        else:
            return "article"
    
    def summarize_result(self, result: SearchResult) -> str:
        """Create a summarized version of the search result"""
        content_type = self.categorize_content(result.title, result.description, result.tags)
        template = self.summary_templates.get(content_type, self.summary_templates["article"])
        
        # Clean up description (remove HTML tags)
        import re
        clean_description = re.sub(r'<[^>]+>', '', result.description)
        clean_description = clean_description[:150] + "..." if len(clean_description) > 150 else clean_description
        
        return template.format(
            title=result.title,
            summary=clean_description
        )

class IntelligentFilter:
    """Intelligent filtering and ranking of search results"""
    
    def __init__(self):
        self.quality_indicators = {
            "high_quality": ["best practice", "production", "enterprise", "scalable", "optimized"],
            "recent": ["2025", "2024", "latest", "new", "recent"],
            "popular": ["popular", "trending", "hot", "viral", "top"],
            "comprehensive": ["complete", "comprehensive", "full", "detailed", "extensive"]
        }
        
        self.spam_indicators = [
            "click here", "buy now", "limited time", "act now", "exclusive offer",
            "make money fast", "work from home", "get rich quick"
        ]
    
    def calculate_quality_score(self, result: SearchResult) -> float:
        """Calculate quality score based on various indicators"""
        text = f"{result.title} {result.description}".lower()
        
        quality_score = 0.0
        
        # Check for quality indicators
        for category, indicators in self.quality_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    quality_score += 0.1
        
        # Check for spam indicators (negative score)
        for indicator in self.spam_indicators:
            if indicator in text:
                quality_score -= 0.2
        
        # Boost score based on source reputation
        reputable_sources = ["stack overflow", "github", "medium", "dev.to", "hashnode"]
        for source in reputable_sources:
            if source in result.source.lower():
                quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def filter_results(self, results: List[SearchResult], min_quality: float = 0.1) -> List[SearchResult]:
        """Filter results based on quality score"""
        filtered_results = []
        
        for result in results:
            quality_score = self.calculate_quality_score(result)
            # Only filter out very low quality results
            if quality_score >= min_quality:
                # Update relevance score with quality score (but don't penalize too much)
                result.relevance_score = max(result.relevance_score, quality_score * 0.3)
                filtered_results.append(result)
        
        # If no results pass the filter, return original results
        if not filtered_results and results:
            return results
        
        return filtered_results

class SearchAnalytics:
    """Analytics and insights for search queries"""
    
    def __init__(self):
        self.cache = cache
        self.search_history = self.cache.get('search_history', [])
        self.query_patterns = self.cache.get('query_patterns', {})
    
    def record_search(self, query: str, results: List[SearchResult]):
        """Record search query and results for analytics"""
        search_record = {
            "query": query,
            "timestamp": time.time(),
            "result_count": len(results),
            "sources": list(set(result.source for result in results)),
            "avg_relevance": sum(r.relevance_score for r in results) / len(results) if results else 0
        }
        
        self.search_history.append(search_record)
        
        # Update query patterns
        query_lower = query.lower()
        if query_lower not in self.query_patterns:
            self.query_patterns[query_lower] = 0
        self.query_patterns[query_lower] += 1
        
        # Persist to cache
        self.cache.set('search_history', self.search_history)
        self.cache.set('query_patterns', self.query_patterns)
    
    def record_search(self, query: str, results: List[SearchResult]):
        """Record search query and results for analytics"""
        search_record = {
            "query": query,
            "timestamp": time.time(),
            "result_count": len(results),
            "sources": list(set(result.source for result in results)),
            "avg_relevance": sum(r.relevance_score for r in results) / len(results) if results else 0
        }
        
        self.search_history.append(search_record)
        
        # Update query patterns
        query_lower = query.lower()
        if query_lower not in self.query_patterns:
            self.query_patterns[query_lower] = 0
        self.query_patterns[query_lower] += 1
    
    def get_search_insights(self) -> Dict[str, Any]:
        """Get insights from search history"""
        if not self.search_history:
            return {"message": "No search history available"}
        
        total_searches = len(self.search_history)
        avg_results = sum(record["result_count"] for record in self.search_history) / total_searches
        avg_relevance = sum(record["avg_relevance"] for record in self.search_history) / total_searches
        
        # Most common sources
        all_sources = []
        for record in self.search_history:
            all_sources.extend(record["sources"])
        
        source_counts = {}
        for source in all_sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        most_common_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Most common queries
        most_common_queries = sorted(self.query_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_searches": total_searches,
            "avg_results_per_search": round(avg_results, 2),
            "avg_relevance_score": round(avg_relevance, 2),
            "most_common_sources": most_common_sources,
            "most_common_queries": most_common_queries
        }

class SearchService:
    """Centralized search service coordinating multiple providers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.providers = self._initialize_providers()
        self.cache = cache
        
        # Initialize advanced features
        self.semantic_enhancer = SemanticSearchEnhancer()
        self.content_summarizer = ContentSummarizer()
        self.intelligent_filter = IntelligentFilter()
        self.analytics = SearchAnalytics()
    
    def _initialize_providers(self) -> Dict[str, SearchProvider]:
        """Initialize search providers based on configuration"""
        providers = {}
        
        # Always include RSS-based providers (no API keys needed)
        providers['medium'] = MediumSearchProvider(self.config)
        providers['dev_to'] = DevToSearchProvider(self.config)
        providers['hashnode'] = HashnodeSearchProvider(self.config)
        providers['blogs'] = DeveloperBlogSearchProvider(self.config)
        
        # Include arXiv if email is configured
        if self.config.get('arxiv_email'):
            providers['arxiv'] = ArxivSearchProvider(self.config)
        
        # Add new providers
        providers['stackoverflow'] = StackOverflowSearchProvider(self.config)
        providers['reddit'] = RedditSearchProvider(self.config)
        providers['hacker_news'] = HackerNewsSearchProvider(self.config)
        providers['trend_analysis'] = TrendAnalysisProvider(self.config)
        providers['code_examples'] = CodeExampleSearchProvider(self.config)
        
        return providers
    
    async def search_all(self, query: str) -> Dict[str, List[SearchResult]]:
        """Search across all available providers"""
        results = {}
        
        # Search all providers concurrently
        tasks = []
        for name, provider in self.providers.items():
            task = asyncio.create_task(provider.search(query))
            tasks.append((name, task))
        
        # Wait for all searches to complete
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                print(f"Search failed for {name}: {e}")
                results[name] = []
        
        return results
    
    async def search_aggregated(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search all providers and return aggregated, ranked results with advanced features"""
        all_results = await self.search_all(query)
        
        # Combine all results
        combined = []
        for provider_name, results in all_results.items():
            for result in results:
                # Add provider info to result
                result.source = f"{result.source} ({provider_name})"
                
                # Enhance with semantic relevance
                semantic_score = self.semantic_enhancer.calculate_semantic_relevance(
                    query, f"{result.title} {result.description}"
                )
                result.relevance_score = (result.relevance_score + semantic_score) / 2
                
                combined.append(result)
        
        # Apply intelligent filtering
        filtered_results = self.intelligent_filter.filter_results(combined)
        
        # Sort by enhanced relevance score
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Record analytics
        self.analytics.record_search(query, filtered_results)
        
        return filtered_results[:max_results]
    
    def format_results(self, results: List[SearchResult]) -> str:
        """Format search results for display with enhanced presentation"""
        if not results:
            return "No relevant results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            # Use content summarizer for better presentation
            summary = self.content_summarizer.summarize_result(result)
            formatted.append(f"{i}. {summary}")
            formatted.append(f"   🔗 {result.url}")
            formatted.append(f"   📅 {result.source} - {result.published_date or 'Unknown date'}")
            formatted.append(f"   ⭐ Relevance: {result.relevance_score:.2f}")
            if result.tags:
                formatted.append(f"   🏷️  Tags: {', '.join(result.tags[:3])}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def get_search_insights(self) -> Dict[str, Any]:
        """Get search analytics insights"""
        return self.analytics.get_search_insights()
    
    async def search_with_semantic_enhancement(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search with semantic query enhancement"""
        # Generate semantic variations of the query
        enhanced_queries = self.semantic_enhancer.enhance_query(query)
        
        all_results = []
        for enhanced_query in enhanced_queries[:2]:  # Limit to avoid too many requests
            results = await self.search_aggregated(enhanced_query, max_results // 2)
            all_results.extend(results)
        
        # Remove duplicates and re-rank
        unique_results = {}
        for result in all_results:
            if result.url not in unique_results:
                unique_results[result.url] = result
            else:
                # Merge relevance scores for duplicate results
                unique_results[result.url].relevance_score = max(
                    unique_results[result.url].relevance_score,
                    result.relevance_score
                )
        
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return final_results[:max_results]

# Configuration helper
def create_search_config() -> Dict[str, Any]:
    """Create search configuration from environment variables"""
    return {
        'medium_api_key': os.getenv("MEDIUM_API_KEY", ""),
        'dev_to_api_key': os.getenv("DEV_TO_API_KEY", ""),
        'hashnode_api_key': os.getenv("HASHNODE_API_KEY", ""),
        'google_cse_id': os.getenv("GOOGLE_CSE_ID", ""),
        'google_api_key': os.getenv("GOOGLE_API_KEY", ""),
        'bing_api_key': os.getenv("BING_API_KEY", ""),
        'arxiv_email': os.getenv("ARXIV_EMAIL", ""),
        'rate_limit_delay': 1.0,
        'max_results': 5
    } 