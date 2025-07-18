"""
Test tools functionality and capabilities
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import (
    GitRepositoryTools, RealResearchTools, BuildTools, create_research_tools
)

class TestGitRepositoryTools:
    """Test Git repository tools functionality"""
    
    def test_git_tools_initialization(self, temp_brain):
        """Test Git tools initialization"""
        git_tools = GitRepositoryTools(temp_brain)
        
        assert git_tools is not None
        assert hasattr(git_tools, 'memory')
        assert hasattr(git_tools, 'github_token')
        assert hasattr(git_tools, 'temp_repos')
        assert isinstance(git_tools.temp_repos, dict)
    
    def test_clone_repository_local(self, git_tools, test_repo):
        """Test cloning local repository"""
        repo_url = f"file://{test_repo}"
        
        with patch('tools.git.Repo') as mock_git:
            mock_repo = MagicMock()
            mock_git.clone_from.return_value = mock_repo
            
            result = git_tools.clone_repository(repo_url)
            
            assert "Repository cloned to" in result
            assert len(git_tools.temp_repos) > 0
    
    def test_clone_repository_github(self, git_tools):
        """Test cloning GitHub repository"""
        repo_url = "https://github.com/test/repo"
        
        with patch('tools.git.Repo') as mock_git:
            mock_repo = MagicMock()
            mock_git.clone_from.return_value = mock_repo
            
            result = git_tools.clone_repository(repo_url)
            
            assert "Repository cloned to" in result
            assert len(git_tools.temp_repos) > 0
    
    def test_read_repository_file(self, git_tools, test_repo):
        """Test reading files from repository"""
        # First clone repository
        repo_url = f"file://{test_repo}"
        with patch('tools.git.Repo'):
            git_tools.clone_repository(repo_url)
        
        # Get repository name
        repo_name = list(git_tools.temp_repos.keys())[0]
        
        # Read a file
        result = git_tools.read_repository_file(repo_name, "README.md")
        
        assert "Test Repository" in result
        assert "File content for README.md" in result
    
    def test_list_repository_files(self, git_tools, test_repo):
        """Test listing repository files"""
        # First clone repository
        repo_url = f"file://{test_repo}"
        with patch('tools.git.Repo'):
            git_tools.clone_repository(repo_url)
        
        # Get repository name
        repo_name = list(git_tools.temp_repos.keys())[0]
        
        # List files
        result = git_tools.list_repository_files(repo_name)
        
        assert "README.md" in result
        assert "main.py" in result
        assert "requirements.txt" in result
        assert "Files in ." in result
    
    def test_create_repository_file(self, git_tools, test_repo):
        """Test creating files in repository"""
        # First clone repository
        repo_url = f"file://{test_repo}"
        with patch('tools.git.Repo'):
            git_tools.clone_repository(repo_url)
        
        # Get repository name
        repo_name = list(git_tools.temp_repos.keys())[0]
        
        # Create a new file
        content = "def new_function():\n    return 'Hello, World!'"
        result = git_tools.create_repository_file(
            repo_name, 
            "new_file.py", 
            content,
            "Add new file"
        )
        
        assert "created and committed" in result
    
    def test_modify_repository_file(self, git_tools, test_repo):
        """Test modifying files in repository"""
        # First clone repository
        repo_url = f"file://{test_repo}"
        with patch('tools.git.Repo'):
            git_tools.clone_repository(repo_url)
        
        # Get repository name
        repo_name = list(git_tools.temp_repos.keys())[0]
        
        # Modify an existing file
        new_content = "def modified_function():\n    return 'Modified!'"
        result = git_tools.modify_repository_file(
            repo_name,
            "main.py",
            new_content,
            "Update main function"
        )
        
        assert "modified and committed" in result
    
    def test_get_repository_status(self, git_tools, test_repo):
        """Test getting repository status"""
        # First clone repository
        repo_url = f"file://{test_repo}"
        with patch('tools.git.Repo'):
            git_tools.clone_repository(repo_url)
        
        # Get repository name
        repo_name = list(git_tools.temp_repos.keys())[0]
        
        # Get status
        result = git_tools.get_repository_status(repo_name)
        
        assert "Current Branch" in result
        assert "main" in result
    
    def test_cleanup_repository(self, git_tools, test_repo):
        """Test repository cleanup"""
        # First clone repository
        repo_url = f"file://{test_repo}"
        with patch('tools.git.Repo'):
            git_tools.clone_repository(repo_url)
        
        # Get repository name
        repo_name = list(git_tools.temp_repos.keys())[0]
        
        # Cleanup
        result = git_tools.cleanup_repository(repo_name)
        
        assert "cleaned up successfully" in result
        assert repo_name not in git_tools.temp_repos
    
    def test_list_cloned_repositories(self, git_tools):
        """Test listing cloned repositories"""
        # Initially should be empty
        result = git_tools.list_cloned_repositories()
        assert "No repositories currently cloned" in result
        
        # After cloning
        with patch('tools.git.Repo'):
            git_tools.clone_repository("https://github.com/test/repo")
        
        result = git_tools.list_cloned_repositories()
        assert "test_repo" in result or "test_repository" in result

class TestRealResearchTools:
    """Test real research tools functionality"""
    
    def test_research_tools_initialization(self, temp_brain):
        """Test research tools initialization"""
        research_tools = RealResearchTools(temp_brain)
        
        assert research_tools is not None
        assert hasattr(research_tools, 'memory')
        assert hasattr(research_tools, 'github_token')
    
    def test_web_search_integration(self, temp_brain):
        """Test web search integration"""
        research_tools = RealResearchTools(temp_brain)
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'results': [
                    {'title': 'Test Result', 'snippet': 'Test snippet'}
                ]
            }
            mock_get.return_value = mock_response
            
            result = research_tools.web_search_integration("test query")
            
            assert result is not None
            assert "Test Result" in result
    
    def test_x_search(self, temp_brain):
        """Test X (Twitter) search"""
        research_tools = RealResearchTools(temp_brain)
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'data': [
                    {'text': 'Test tweet', 'author_id': '123'}
                ]
            }
            mock_get.return_value = mock_response
            
            result = research_tools.x_search("test query")
            
            assert result is not None
            assert "Test tweet" in result
    
    def test_github_code_search(self, temp_brain):
        """Test GitHub code search"""
        research_tools = RealResearchTools(temp_brain)
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'items': [
                    {
                        'name': 'test.py',
                        'path': 'src/test.py',
                        'repository': {'full_name': 'test/repo'},
                        'html_url': 'https://github.com/test/repo/blob/main/src/test.py'
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            results = research_tools.github_code_search("python function")
            
            assert isinstance(results, list)
            assert len(results) > 0
            assert 'name' in results[0]
            assert 'path' in results[0]
    
    def test_analyze_github_repo(self, temp_brain):
        """Test GitHub repository analysis"""
        research_tools = RealResearchTools(temp_brain)
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'name': 'test-repo',
                'description': 'Test repository',
                'language': 'Python',
                'stargazers_count': 100,
                'forks_count': 50
            }
            mock_get.return_value = mock_response
            
            result = research_tools.analyze_github_repo("https://github.com/test/repo")
            
            assert result is not None
            assert "test-repo" in result
            assert "Python" in result
    
    def test_extract_github_content(self, temp_brain):
        """Test GitHub content extraction"""
        research_tools = RealResearchTools(temp_brain)
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'items': [
                    {
                        'title': 'Test Issue',
                        'body': 'Test issue body',
                        'number': 1,
                        'state': 'open'
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            results = research_tools.extract_github_content(
                "https://github.com/test/repo",
                content_types=["issues"]
            )
            
            assert isinstance(results, list)
            assert len(results) > 0
            assert 'title' in results[0]
            assert 'body' in results[0]

class TestBuildTools:
    """Test build tools functionality"""
    
    def test_build_tools_initialization(self, temp_brain):
        """Test build tools initialization"""
        build_tools = BuildTools(temp_brain)
        
        assert build_tools is not None
        assert hasattr(build_tools, 'memory')
    
    def test_analyze_python_deps(self, temp_brain):
        """Test Python dependency analysis"""
        build_tools = BuildTools(temp_brain)
        
        # Create a test requirements.txt
        test_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(test_dir, "requirements.txt"), 'w') as f:
                f.write("requests>=2.31.0\npytest>=7.4.0\nnumpy>=1.24.0")
            
            result = build_tools.analyze_python_deps(test_dir)
            
            assert result is not None
            assert 'dependencies' in result
            assert 'requests' in result['dependencies']
            assert 'pytest' in result['dependencies']
            assert 'numpy' in result['dependencies']
        finally:
            shutil.rmtree(test_dir)
    
    def test_run_linter(self, temp_brain):
        """Test linter execution"""
        build_tools = BuildTools(temp_brain)
        
        # Create a test Python file
        test_dir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(test_dir, "test.py")
            with open(test_file, 'w') as f:
                f.write("def test_function():\n    print('Hello, World!')")
            
            with patch('tools.subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "No issues found"
                mock_run.return_value = mock_result
                
                result = build_tools.run_linter([test_file])
                
                assert result is not None
                assert 'status' in result
                assert result['status'] == 'success'
        finally:
            shutil.rmtree(test_dir)

class TestResearchToolsIntegration:
    """Test research tools integration"""
    
    def test_create_research_tools(self, temp_brain):
        """Test research tools creation"""
        tools = create_research_tools(temp_brain)
        
        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check for expected tools
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            'web_search', 'x_search', 'github_search', 'analyze_github_repo',
            'extract_github_content', 'remember_decision', 'find_relevant_files',
            'analyze_dependencies', 'run_linter', 'git_commit'
        ]
        
        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"
    
    def test_web_search_tool(self, temp_brain):
        """Test web search tool"""
        tools = create_research_tools(temp_brain)
        
        web_search_tool = next(tool for tool in tools if tool.name == 'web_search')
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'results': [{'title': 'Test', 'snippet': 'Test result'}]
            }
            mock_get.return_value = mock_response
            
            result = web_search_tool.func("test query")
            
            assert result is not None
            assert "Test" in result
    
    def test_github_search_tool(self, temp_brain):
        """Test GitHub search tool"""
        tools = create_research_tools(temp_brain)
        
        github_search_tool = next(tool for tool in tools if tool.name == 'github_search')
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'items': [{'name': 'test.py', 'path': 'src/test.py'}]
            }
            mock_get.return_value = mock_response
            
            result = github_search_tool.func("python function")
            
            assert result is not None
            assert "test.py" in result
    
    def test_remember_decision_tool(self, temp_brain):
        """Test remember decision tool"""
        tools = create_research_tools(temp_brain)
        
        remember_tool = next(tool for tool in tools if tool.name == 'remember_decision')
        
        result = remember_tool.func(
            decision="Use FastAPI",
            reasoning="FastAPI is modern and fast",
            context="Web API project"
        )
        
        assert result is not None
        assert "decision remembered" in result.lower()
        
        # Verify decision was stored in brain
        assert len(temp_brain.memory['decisions']) > 0
        last_decision = temp_brain.memory['decisions'][-1]
        assert last_decision['decision'] == "Use FastAPI"
        assert last_decision['reasoning'] == "FastAPI is modern and fast"
    
    def test_find_relevant_files_tool(self, temp_brain):
        """Test find relevant files tool"""
        tools = create_research_tools(temp_brain)
        
        find_files_tool = next(tool for tool in tools if tool.name == 'find_relevant_files')
        
        result = find_files_tool.func("calculator function")
        
        assert result is not None
        assert "relevant files" in result.lower()

class TestToolsErrorHandling:
    """Test tools error handling"""
    
    def test_git_tools_error_handling(self, git_tools):
        """Test Git tools error handling"""
        # Test cloning non-existent repository
        result = git_tools.clone_repository("https://github.com/non-existent/repo")
        
        assert "Failed to clone" in result
    
    def test_research_tools_error_handling(self, temp_brain):
        """Test research tools error handling"""
        research_tools = RealResearchTools(temp_brain)
        
        # Test web search with network error
        with patch('tools.requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            result = research_tools.web_search_integration("test query")
            
            assert "error" in result.lower() or "failed" in result.lower()
    
    def test_build_tools_error_handling(self, temp_brain):
        """Test build tools error handling"""
        build_tools = BuildTools(temp_brain)
        
        # Test linter with non-existent file
        result = build_tools.run_linter(["non_existent_file.py"])
        
        assert result is not None
        assert 'status' in result
        assert result['status'] == 'error' or 'error' in result.get('message', '').lower()

class TestToolsPerformance:
    """Test tools performance"""
    
    def test_git_tools_performance(self, git_tools, test_repo):
        """Test Git tools performance"""
        import time
        
        repo_url = f"file://{test_repo}"
        
        with patch('tools.git.Repo'):
            start_time = time.time()
            result = git_tools.clone_repository(repo_url)
            end_time = time.time()
            
            # Should complete quickly with mocks
            assert end_time - start_time < 2.0
            assert "Repository cloned to" in result
    
    def test_research_tools_performance(self, temp_brain):
        """Test research tools performance"""
        import time
        
        research_tools = RealResearchTools(temp_brain)
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'results': []}
            mock_get.return_value = mock_response
            
            start_time = time.time()
            result = research_tools.web_search_integration("test query")
            end_time = time.time()
            
            # Should complete quickly with mocks
            assert end_time - start_time < 1.0
            assert result is not None
    
    def test_build_tools_performance(self, temp_brain):
        """Test build tools performance"""
        import time
        
        build_tools = BuildTools(temp_brain)
        
        # Create a test file
        test_dir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(test_dir, "test.py")
            with open(test_file, 'w') as f:
                f.write("def test():\n    pass")
            
            with patch('tools.subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "No issues"
                mock_run.return_value = mock_result
                
                start_time = time.time()
                result = build_tools.run_linter([test_file])
                end_time = time.time()
                
                # Should complete quickly with mocks
                assert end_time - start_time < 1.0
                assert result is not None
        finally:
            shutil.rmtree(test_dir)

class TestToolsIntegration:
    """Test tools integration with other components"""
    
    def test_tools_with_agents(self, temp_brain):
        """Test tools integration with agents"""
        from agents import create_planner_agent
        
        # Create agent with tools
        with patch('agents.get_default_model') as mock_model:
            mock_model.return_value = MagicMock()
            agent = create_planner_agent(temp_brain)
            
            # Verify agent has tools
            assert hasattr(agent, 'tools')
            assert len(agent.tools) > 0
            
            # Verify tools can be executed
            for tool in agent.tools:
                assert hasattr(tool, 'func')
    
    def test_tools_with_workflow(self, temp_brain):
        """Test tools integration with workflow"""
        from graph import run_workflow
        
        # Run workflow that uses tools
        with patch('graph.create_workflow_graph') as mock_graph:
            mock_graph_instance = MagicMock()
            mock_graph_instance.invoke.return_value = {
                'final_result': 'Workflow completed',
                'status': 'success'
            }
            mock_graph.return_value = mock_graph_instance
            
            result = run_workflow(
                task="Research REST API frameworks",
                brain=temp_brain,
                mode="research",
                repo_url=""
            )
            
            assert result is not None
            # Tools should be used during workflow execution
    
    def test_tools_with_memory(self, temp_brain):
        """Test tools integration with memory"""
        tools = create_research_tools(temp_brain)
        
        # Use a tool that interacts with memory
        remember_tool = next(tool for tool in tools if tool.name == 'remember_decision')
        
        remember_tool.func(
            decision="Test decision",
            reasoning="Test reasoning",
            context="Test context"
        )
        
        # Verify memory was updated
        assert len(temp_brain.memory['decisions']) > 0
        last_decision = temp_brain.memory['decisions'][-1]
        assert last_decision['decision'] == "Test decision"
        assert last_decision['reasoning'] == "Test reasoning"
        assert last_decision['context'] == "Test context"

class TestToolsQuality:
    """Test tools output quality"""
    
    def test_web_search_quality(self, temp_brain):
        """Test web search output quality"""
        research_tools = RealResearchTools(temp_brain)
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'results': [
                    {
                        'title': 'FastAPI Documentation',
                        'snippet': 'FastAPI is a modern, fast web framework for building APIs with Python 3.7+',
                        'link': 'https://fastapi.tiangolo.com/'
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            result = research_tools.web_search_integration("FastAPI framework")
            
            assert result is not None
            assert "FastAPI" in result
            assert "modern" in result
            assert "fast" in result
            assert "web framework" in result
    
    def test_github_search_quality(self, temp_brain):
        """Test GitHub search output quality"""
        research_tools = RealResearchTools(temp_brain)
        
        with patch('tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'items': [
                    {
                        'name': 'calculator.py',
                        'path': 'src/calculator.py',
                        'repository': {'full_name': 'test/calculator'},
                        'html_url': 'https://github.com/test/calculator/blob/main/src/calculator.py',
                        'content': 'def add(a, b):\n    return a + b'
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            results = research_tools.github_code_search("calculator function")
            
            assert isinstance(results, list)
            assert len(results) > 0
            result = results[0]
            assert 'name' in result
            assert 'path' in result
            assert 'repository' in result
            assert 'html_url' in result
            assert result['name'] == 'calculator.py'
    
    def test_linter_quality(self, temp_brain):
        """Test linter output quality"""
        build_tools = BuildTools(temp_brain)
        
        # Create a test file with issues
        test_dir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(test_dir, "test.py")
            with open(test_file, 'w') as f:
                f.write("def test_function():\n    x=1\n    return x")
            
            with patch('tools.subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 1
                mock_result.stdout = "E501: line too long"
                mock_result.stderr = "W291: trailing whitespace"
                mock_run.return_value = mock_result
                
                result = build_tools.run_linter([test_file])
                
                assert result is not None
                assert 'status' in result
                assert 'issues' in result or 'errors' in result
        finally:
            shutil.rmtree(test_dir) 