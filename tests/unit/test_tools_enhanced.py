"""
Unit tests for enhanced tools with rate limiting and error handling
"""

import pytest
import tempfile
import os
import shutil
import time
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime

from tools import GitRepositoryTools, RealResearchTools
from memory import ProjectBrain


class TestGitRepositoryToolsEnhanced:
    """Test enhanced Git repository tools"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = ProjectBrain(self.temp_dir)
        self.git_tools = GitRepositoryTools(self.memory)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_github_token_retrieval(self):
        """Test GitHub token retrieval from credential manager"""
        # Set token in credential manager
        self.memory.credential_manager.set_credential("GITHUB_TOKEN", "test_token_123")
        
        # Create new git tools instance
        new_git_tools = GitRepositoryTools(self.memory)
        
        # Token should be retrieved from credential manager
        assert new_git_tools.github_token == "test_token_123"
    
    def test_rate_limit_checking(self):
        """Test rate limit checking functionality"""
        # Test initial state
        assert self.git_tools._check_rate_limit('test_api') is True
        
        # Simulate hitting rate limit
        self.git_tools.rate_limits['test_api'] = {
            'count': self.git_tools.max_requests_per_hour,
            'reset_time': time.time() + 3600
        }
        
        assert self.git_tools._check_rate_limit('test_api') is False
    
    def test_rate_limit_reset(self):
        """Test rate limit reset after timeout"""
        # Set up expired rate limit
        self.git_tools.rate_limits['test_api'] = {
            'count': 100,
            'reset_time': time.time() - 100  # Expired
        }
        
        # Should reset and allow request
        assert self.git_tools._check_rate_limit('test_api') is True
        assert self.git_tools.rate_limits['test_api']['count'] == 0
    
    def test_repository_url_validation(self):
        """Test repository URL validation"""
        valid_urls = [
            "https://github.com/user/repo",
            "https://gitlab.com/user/repo",
            "https://bitbucket.org/user/repo",
            "file:///path/to/repo",
            "git@github.com:user/repo.git"
        ]
        
        invalid_urls = [
            "not-a-url",
            "http://invalid.com",
            "ftp://github.com/user/repo"
        ]
        
        for url in valid_urls:
            assert self.git_tools._validate_repo_url(url) is True
        
        for url in invalid_urls:
            assert self.git_tools._validate_repo_url(url) is False
    
    def test_url_sanitization(self):
        """Test URL sanitization for logging"""
        # Test URL with authentication
        auth_url = "https://token123@github.com/user/repo"
        sanitized = self.git_tools._sanitize_url(auth_url)
        assert sanitized == "https://github.com/user/repo"
        
        # Test URL without authentication
        normal_url = "https://github.com/user/repo"
        sanitized = self.git_tools._sanitize_url(normal_url)
        assert sanitized == normal_url
    
    def test_repository_size_calculation(self):
        """Test repository size calculation"""
        # Create test files
        test_file1 = os.path.join(self.temp_dir, "test1.txt")
        test_file2 = os.path.join(self.temp_dir, "test2.txt")
        
        with open(test_file1, 'w') as f:
            f.write("content1")
        with open(test_file2, 'w') as f:
            f.write("content2")
        
        size = self.git_tools._get_repo_size(self.temp_dir)
        assert size > 0
    
    def test_git_error_logging(self):
        """Test Git error logging"""
        # Log an error
        self.git_tools._log_git_error('clone', 'https://github.com/user/repo', 'Test error')
        
        # Check that error was logged
        git_operations = self.memory.memory.get('git_operations', [])
        assert len(git_operations) > 0
        
        last_operation = git_operations[-1]
        assert last_operation['operation'] == 'clone'
        assert last_operation['error'] == 'Test error'
        assert last_operation['success'] is False
    
    @patch('tools.git.Repo')
    def test_clone_repository_rate_limit(self, mock_repo):
        """Test repository cloning with rate limiting"""
        # Mock successful clone
        mock_repo.clone_from.return_value = Mock()
        
        # Test rate limit exceeded
        self.git_tools.rate_limits['github_clone'] = {
            'count': self.git_tools.max_requests_per_hour,
            'reset_time': time.time() + 3600
        }
        
        result = self.git_tools.clone_repository("https://github.com/user/repo")
        assert "Rate limit exceeded" in result
    
    @patch('tools.git.Repo')
    def test_clone_repository_invalid_url(self, mock_repo):
        """Test repository cloning with invalid URL"""
        result = self.git_tools.clone_repository("invalid-url")
        assert "Invalid repository URL format" in result
    
    @patch('tools.git.Repo')
    def test_clone_repository_authentication_error(self, mock_repo):
        """Test repository cloning with authentication error"""
        from git.exc import GitCommandError
        
        # Mock authentication error
        mock_repo.clone_from.side_effect = GitCommandError(
            'git', 'clone', 'Authentication failed'
        )
        
        result = self.git_tools.clone_repository("https://github.com/user/repo")
        assert "Authentication failed" in result


class TestRealResearchToolsEnhanced:
    """Test enhanced research tools"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = ProjectBrain(self.temp_dir)
        self.research_tools = RealResearchTools(self.memory)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_api_key_retrieval(self):
        """Test API key retrieval from credential manager"""
        # Set API keys in credential manager
        self.memory.credential_manager.set_credential("GOOGLE_API_KEY", "google_key_123")
        self.memory.credential_manager.set_credential("BING_API_KEY", "bing_key_456")
        
        # Create new research tools instance
        new_research_tools = RealResearchTools(self.memory)
        
        # Keys should be retrieved from credential manager
        assert new_research_tools.search_config['google_api_key'] == "google_key_123"
        assert new_research_tools.search_config['bing_api_key'] == "bing_key_456"
    
    def test_rate_limit_functionality(self):
        """Test rate limit functionality"""
        # Test initial state
        assert self.research_tools._check_rate_limit('test_api') is True
        
        # Simulate hitting rate limit
        self.research_tools.rate_limits['test_api'] = {
            'count': 100,
            'reset_time': time.time() + 3600
        }
        
        assert self.research_tools._check_rate_limit('test_api') is False
    
    def test_rate_limit_update(self):
        """Test rate limit update"""
        initial_time = time.time()
        self.research_tools._update_rate_limit('test_api')
        
        assert self.research_tools.rate_limits['test_api']['count'] == 1
        assert self.research_tools.last_request_time > initial_time
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        # Clear cache first to ensure clean state
        try:
            self.research_tools.cache.clear()
        except:
            pass
        
        # Test cache miss
        result = self.research_tools._get_cached_result('test_key')
        assert result is None
        
        # Test cache hit
        self.research_tools._cache_result('test_key', 'test_value')
        result = self.research_tools._get_cached_result('test_key')
        assert result == 'test_value'
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        # Set short timeout for testing
        self.research_tools.cache_timeout = 1
        
        # Cache a result
        self.research_tools._cache_result('expire_key', 'expire_value')
        
        # Should be available immediately
        result = self.research_tools._get_cached_result('expire_key')
        assert result == 'expire_value'
        
        # Wait for expiration
        time.sleep(2)
        
        # Should be expired
        result = self.research_tools._get_cached_result('expire_key')
        assert result is None
    
    @patch('tools.requests.get')
    def test_web_search_with_caching(self, mock_get):
        """Test web search with caching"""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            'AbstractText': 'Test abstract',
            'RelatedTopics': [{'Text': 'Related topic'}]
        }
        mock_get.return_value = mock_response
        
        # First search should cache result
        result1 = self.research_tools._fallback_duckduckgo_search("test query")
        assert "Test abstract" in result1
        
        # Second search should use cache
        result2 = self.research_tools._fallback_duckduckgo_search("test query")
        assert result1 == result2
    
    @patch('tools.requests.get')
    def test_web_search_rate_limit(self, mock_get):
        """Test web search with rate limiting"""
        # Set rate limit
        self.research_tools.rate_limits['web_search'] = {
            'count': 50,
            'reset_time': time.time() + 3600
        }
        
        result = self.research_tools.web_search_integration("test query")
        assert "Rate limit exceeded" in result
    
    @patch('tools.requests.get')
    def test_web_search_error_handling(self, mock_get):
        """Test web search error handling"""
        # Mock request failure
        mock_get.side_effect = Exception("Network error")
        
        result = self.research_tools._fallback_duckduckgo_search("test query")
        assert "Search failed" in result
    
    def test_minimum_request_interval(self):
        """Test minimum request interval enforcement"""
        # Make first request
        self.research_tools._update_rate_limit('test_api')
        first_time = self.research_tools.last_request_time
        
        # Make second request immediately
        self.research_tools._update_rate_limit('test_api')
        second_time = self.research_tools.last_request_time
        
        # Should have minimum interval
        assert second_time - first_time >= self.research_tools.min_request_interval


class TestToolsIntegration:
    """Test integration between tools and memory system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = ProjectBrain(self.temp_dir)
        self.git_tools = GitRepositoryTools(self.memory)
        self.research_tools = RealResearchTools(self.memory)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_credential_sharing(self):
        """Test that credentials are shared between tools"""
        # Set credential in memory
        self.memory.credential_manager.set_credential("GITHUB_TOKEN", "shared_token")
        
        # Refresh tokens to get latest from credential manager
        git_token = self.git_tools._refresh_github_token()
        research_token = self.research_tools._refresh_github_token()
        
        # Both tools should access the same credential
        assert git_token == "shared_token"
        assert research_token == "shared_token"
    
    def test_memory_logging_integration(self):
        """Test that tools log operations to memory"""
        # Perform operations
        self.git_tools._log_git_error('test', 'https://github.com/user/repo', 'test error')
        
        # Check memory
        git_operations = self.memory.memory.get('git_operations', [])
        assert len(git_operations) > 0
        
        # Check that operation was logged with security features
        operation = git_operations[-1]
        assert 'repo_url' in operation
        assert '@' not in operation['repo_url']  # Should be sanitized
    
    def test_rate_limit_isolation(self):
        """Test that rate limits are isolated between tools"""
        # Set rate limit for git tools
        self.git_tools.rate_limits['test_api'] = {'count': 100, 'reset_time': time.time() + 3600}
        
        # Research tools should not be affected
        assert self.research_tools._check_rate_limit('test_api') is True
    
    def test_cache_isolation(self):
        """Test that caches are isolated between tools"""
        # Cache data in research tools
        self.research_tools._cache_result('test_key', 'test_value')
        
        # Git tools should not have access to this cache
        # (Git tools don't use the same cache, but this tests isolation concept)
        assert hasattr(self.git_tools, 'cache') is False


if __name__ == "__main__":
    pytest.main([__file__]) 