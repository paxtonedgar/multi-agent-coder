"""
LangChain tools for research and build operations
"""

import os
import sys
import json
import glob
import ast
import argparse
import asyncio
import subprocess
import tempfile
import shutil
import datetime
from typing import Dict, List, Any, Tuple, Optional
import requests
import numpy as np
import traceback
import tomllib  # Python 3.11+ for TOML parsing
import re
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool

from memory import ProjectBrain

# ==================== GIT REPOSITORY TOOLS ====================

class GitRepositoryTools:
    """Tools for Git repository operations with permission-based access"""
    
    def __init__(self, project_memory: ProjectBrain):
        self.memory = project_memory
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.temp_repos = {}  # Track cloned repositories
        
    def clone_repository(self, repo_url: str, branch: str = "main") -> str:
        """Clone a Git repository to a temporary directory"""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="git_repo_")
            
            # Handle file:// URLs for local repositories
            if repo_url.startswith("file://"):
                local_path = repo_url[7:]  # Remove file:// prefix
                repo = git.Repo.clone_from(local_path, temp_dir, branch=branch, depth=1)
            elif self.github_token and "github.com" in repo_url:
                # Use token for private repos
                auth_url = repo_url.replace("https://", f"https://{self.github_token}@")
                repo = git.Repo.clone_from(auth_url, temp_dir, branch=branch, depth=1)
            else:
                repo = git.Repo.clone_from(repo_url, temp_dir, branch=branch, depth=1)
            
            # Store reference
            repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            if repo_name.startswith('test_repo_'):
                # For test repositories, use a more descriptive name
                repo_name = 'test_repository'
            
            self.temp_repos[repo_name] = {
                'path': temp_dir,
                'repo': repo,
                'url': repo_url,
                'branch': branch,
                'cloned_at': datetime.now().isoformat()
            }
            
            # Log to memory
            if 'git_operations' not in self.memory.memory:
                self.memory.memory['git_operations'] = []
            self.memory.memory['git_operations'].append({
                'operation': 'clone',
                'repo_url': repo_url,
                'temp_path': temp_dir,
                'timestamp': datetime.now().isoformat()
            })
            self.memory._save()
            
            return f"Repository cloned to {temp_dir}"
            
        except Exception as e:
            return f"Failed to clone repository: {str(e)}"
    
    def read_repository_file(self, repo_name: str, file_path: str) -> str:
        """Read a file from a cloned repository"""
        if repo_name not in self.temp_repos:
            return f"Repository '{repo_name}' not found. Use clone_repository first."
        
        repo_info = self.temp_repos[repo_name]
        full_path = os.path.join(repo_info['path'], file_path)
        
        try:
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Log file read
                self.memory.memory['file_operations'].append({
                    'operation': 'read',
                    'repo': repo_name,
                    'file': file_path,
                    'size': len(content),
                    'timestamp': datetime.now().isoformat()
                })
                self.memory._save()
                
                return f"File content for {file_path}:\n\n{content}"
            else:
                return f"File {file_path} not found in repository {repo_name}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def list_repository_files(self, repo_name: str, directory: str = ".") -> str:
        """List files in a repository directory"""
        if repo_name not in self.temp_repos:
            return f"Repository '{repo_name}' not found. Use clone_repository first."
        
        repo_info = self.temp_repos[repo_name]
        full_path = os.path.join(repo_info['path'], directory)
        
        try:
            if not os.path.exists(full_path):
                return f"Directory {directory} not found in repository {repo_name}"
            
            files = []
            for root, dirs, filenames in os.walk(full_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for filename in filenames:
                    if not filename.startswith('.'):
                        rel_path = os.path.relpath(os.path.join(root, filename), repo_info['path'])
                        files.append(rel_path)
            
            return f"Files in {directory}:\n" + "\n".join(sorted(files))
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    def create_repository_file(self, repo_name: str, file_path: str, content: str, 
                             commit_message: str = "Add new file") -> str:
        """Create a new file in a cloned repository and commit it"""
        if repo_name not in self.temp_repos:
            return f"Repository '{repo_name}' not found. Use clone_repository first."
        
        repo_info = self.temp_repos[repo_name]
        full_path = os.path.join(repo_info['path'], file_path)
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Add and commit
            repo = repo_info['repo']
            repo.index.add([file_path])
            repo.index.commit(commit_message)
            
            # Log operation
            self.memory.memory['file_operations'].append({
                'operation': 'create',
                'repo': repo_name,
                'file': file_path,
                'commit_message': commit_message,
                'timestamp': datetime.now().isoformat()
            })
            self.memory._save()
            
            return f"File {file_path} created and committed with message: {commit_message}"
        except Exception as e:
            return f"Error creating file: {str(e)}"
    
    def modify_repository_file(self, repo_name: str, file_path: str, new_content: str,
                              commit_message: str = "Update file") -> str:
        """Modify an existing file in a cloned repository and commit it"""
        if repo_name not in self.temp_repos:
            return f"Repository '{repo_name}' not found. Use clone_repository first."
        
        repo_info = self.temp_repos[repo_name]
        full_path = os.path.join(repo_info['path'], file_path)
        
        try:
            if not os.path.exists(full_path):
                return f"File {file_path} not found in repository {repo_name}"
            
            # Write new content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Add and commit
            repo = repo_info['repo']
            repo.index.add([file_path])
            repo.index.commit(commit_message)
            
            # Log operation
            self.memory.memory['file_operations'].append({
                'operation': 'modify',
                'repo': repo_name,
                'file': file_path,
                'commit_message': commit_message,
                'timestamp': datetime.now().isoformat()
            })
            self.memory._save()
            
            return f"File {file_path} modified and committed with message: {commit_message}"
        except Exception as e:
            return f"Error modifying file: {str(e)}"
    
    def push_changes(self, repo_name: str, remote_name: str = "origin") -> str:
        """Push committed changes to remote repository"""
        if repo_name not in self.temp_repos:
            return f"Repository '{repo_name}' not found. Use clone_repository first."
        
        repo_info = self.temp_repos[repo_name]
        repo = repo_info['repo']
        
        try:
            # Check if we have a GitHub token for authentication
            if self.github_token and "github.com" in repo_info['url']:
                # Configure git with token
                repo.config_writer().set_value("user", "name", "Multi-Agent Coder").release()
                repo.config_writer().set_value("user", "email", "coder@example.com").release()
                
                # Push to remote
                origin = repo.remote(remote_name)
                origin.push()
                
                # Log operation
                self.memory.memory['git_operations'].append({
                    'operation': 'push',
                    'repo_url': repo_info['url'],
                    'timestamp': datetime.now().isoformat()
                })
                self.memory._save()
                
                return f"Changes pushed to {remote_name} successfully"
            else:
                return "Cannot push: No GitHub token configured or not a GitHub repository"
        except Exception as e:
            return f"Error pushing changes: {str(e)}"
    
    def create_branch(self, repo_name: str, branch_name: str) -> str:
        """Create a new branch in the repository"""
        if repo_name not in self.temp_repos:
            return f"Repository '{repo_name}' not found. Use clone_repository first."
        
        repo_info = self.temp_repos[repo_name]
        repo = repo_info['repo']
        
        try:
            # Create and checkout new branch
            new_branch = repo.create_head(branch_name)
            new_branch.checkout()
            
            # Log operation
            self.memory.memory['git_operations'].append({
                'operation': 'create_branch',
                'repo_url': repo_info['url'],
                'branch': branch_name,
                'timestamp': datetime.now().isoformat()
            })
            self.memory._save()
            
            return f"Created and switched to branch: {branch_name}"
        except Exception as e:
            return f"Error creating branch: {str(e)}"
    
    def get_repository_status(self, repo_name: str) -> str:
        """Get the current status of a repository"""
        if repo_name not in self.temp_repos:
            return f"Repository '{repo_name}' not found. Use clone_repository first."
        
        repo_info = self.temp_repos[repo_name]
        repo = repo_info['repo']
        
        try:
            # Get current branch
            current_branch = repo.active_branch.name
            
            # Get status
            status = repo.git.status()
            
            # Get recent commits
            commits = list(repo.iter_commits('HEAD', max_count=5))
            commit_info = "\n".join([f"- {c.hexsha[:8]}: {c.message}" for c in commits])
            
            return f"""Repository Status for {repo_name}:
            
Current Branch: {current_branch}
Repository Path: {repo_info['path']}
Cloned From: {repo_info['url']}

Git Status:
{status}

Recent Commits:
{commit_info}"""
        except Exception as e:
            return f"Error getting repository status: {str(e)}"
    
    def cleanup_repository(self, repo_name: str) -> str:
        """Clean up a cloned repository"""
        if repo_name not in self.temp_repos:
            return f"Repository '{repo_name}' not found."
        
        repo_info = self.temp_repos[repo_name]
        
        try:
            # Remove temporary directory
            shutil.rmtree(repo_info['path'])
            
            # Remove from tracking
            del self.temp_repos[repo_name]
            
            # Log cleanup
            self.memory.memory['git_operations'].append({
                'operation': 'cleanup',
                'repo_url': repo_info['url'],
                'timestamp': datetime.now().isoformat()
            })
            self.memory._save()
            
            return f"Repository {repo_name} cleaned up successfully"
        except Exception as e:
            return f"Error cleaning up repository: {str(e)}"
    
    def list_cloned_repositories(self) -> str:
        """List all currently cloned repositories"""
        if not self.temp_repos:
            return "No repositories currently cloned."
        
        repo_list = []
        for name, info in self.temp_repos.items():
            repo_list.append(f"- {name}: {info['path']} (from {info['url']})")
        
        return "Cloned repositories:\n" + "\n".join(repo_list)

# ==================== RESEARCH TOOLS ====================

class RealResearchTools:
    """Enhanced research tools with GitHub content extraction"""
    
    def __init__(self, project_memory: ProjectBrain):
        self.memory = project_memory
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.session = requests.Session()
        if self.github_token:
            self.session.headers.update({"Authorization": f"token {self.github_token}"})
    
    def web_search_integration(self, query: str) -> str:
        """Enhanced web search with multiple sources"""
        print(f"🔍 Searching: {query}")
        results = []
        
        # DuckDuckGo search
        try:
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1"},
                timeout=5
            )
            data = response.json()
            
            if data.get('AbstractText'):
                results.append(f"Summary: {data['AbstractText'][:500]}...")
            if data.get('RelatedTopics'):
                for topic in data['RelatedTopics'][:2]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append(f"Related: {topic['Text'][:200]}...")
            
        except Exception as e:
            results.append(f"DuckDuckGo failed: {e}")
        
        # Add suggestion for X search
        if any(term in query.lower() for term in ['2025', 'latest', 'recent', 'cloudflare', 'workers']):
            results.append(f"\n💡 For latest discussions, search X (Twitter): {query}")
        
        return "\n".join(results) if results else f"No results. Try manual search: {query}"
    
    def x_search(self, query: str) -> str:
        """Search X (Twitter) for latest discussions"""
        # For now, return search suggestion
        # In production, would use X API v2
        x_queries = [
            f'"{query}" lang:en -filter:replies min_faves:10',
            f'{query} (announcement OR released OR update)',
            f'{query} filter:links'
        ]
        
        return f"""X (Twitter) search suggestions for latest info:
        
1. General discussion: {x_queries[0]}
2. Announcements: {x_queries[1]}  
3. With links: {x_queries[2]}

Recent relevant topics might include:
- Cloudflare Workers Python support (Pyodide beta, stdlib only)
- Claude 4 capabilities for coding
- Browser automation detection methods
- Edge deployment patterns
"""
    
    def github_code_search(self, query: str) -> List[Dict[str, str]]:
        """Search GitHub for code and repos"""
        results = []
        
        headers = {"Authorization": f"token {self.github_token}"} if self.github_token else {}
        
        # Search repositories
        try:
            response = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "per_page": 3},
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                repos = response.json().get('items', [])
                for repo in repos:
                    results.append({
                        'type': 'repo',
                        'name': repo['full_name'],
                        'stars': repo['stargazers_count'],
                        'description': repo['description'],
                        'url': repo['html_url']
                    })
            elif response.status_code == 401:
                results.append({
                    'type': 'error',
                    'message': 'GitHub token invalid. Using public rate limit (10/min).'
                })
        except Exception as e:
            results.append({'type': 'error', 'message': f'GitHub search failed: {e}'})
        
        # Search code snippets
        if self.github_token and len(results) < 5:
            try:
                code_response = requests.get(
                    "https://api.github.com/search/code",
                    params={"q": f"{query} language:python", "per_page": 2},
                    headers=headers,
                    timeout=5
                )
                
                if code_response.status_code == 200:
                    items = code_response.json().get('items', [])
                    for item in items:
                        results.append({
                            'type': 'code',
                            'file': item['name'],
                            'repo': item['repository']['full_name'],
                            'path': item['path']
                        })
            except:
                pass
        
        return results
    
    def analyze_github_repo(self, repo_url: str) -> str:
        """Deep analysis of GitHub repository"""
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        owner = repo_url.rstrip('/').split('/')[-2]
        
        # Use temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone with --depth=1 for speed
                print(f"Cloning {repo_url}...")
                result = subprocess.run(
                    ['git', 'clone', '--depth=1', repo_url, temp_dir],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    return f"Clone failed: {result.stderr}"
                
                # Analyze structure
                analysis = {
                    'structure': self._analyze_repo_structure(temp_dir),
                    'patterns': self._extract_patterns(temp_dir),
                    'dependencies': self._analyze_dependencies(temp_dir),
                    'insights': []
                }
                
                # Generate insights
                deps_lower = ' '.join(analysis['dependencies']).lower()
                if 'playwright' in deps_lower:
                    analysis['insights'].append("Uses Playwright for browser automation")
                if 'selenium' in deps_lower:
                    analysis['insights'].append("Uses Selenium (consider Playwright for better detection avoidance)")
                if 'fastapi' in deps_lower or 'flask' in deps_lower:
                    analysis['insights'].append("Web API framework present")
                if 'pytest' in deps_lower:
                    analysis['insights'].append("Good test coverage with pytest")
                
                # Check for specific patterns
                if 'async def' in ' '.join(analysis['patterns']):
                    analysis['insights'].append("Modern async/await architecture")
                if analysis['structure'].get('services'):
                    analysis['insights'].append("Service-oriented architecture")
                
                # Format output
                output = f"""
Repository Analysis: {owner}/{repo_name}

STRUCTURE:
{self._format_structure(analysis['structure'])}

KEY PATTERNS:
{chr(10).join(f'- {p}' for p in analysis['patterns'][:5])}

DEPENDENCIES:
{chr(10).join(f'- {d}' for d in analysis['dependencies'][:10])}

INSIGHTS FOR YOUR PROJECT:
{chr(10).join(f'- {i}' for i in analysis['insights'])}

ADAPT TO YOUR PROJECT:
- Consider their project structure for organization
- Review their error handling patterns
- Adopt similar testing strategies
- Check their CI/CD setup for deployment ideas
"""
                
                # Log to brain
                self.memory.memory['github_examples'].append({
                    'repo': repo_url,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                })
                self.memory._save()
                
                return output
                
            except subprocess.TimeoutExpired:
                return "Repository clone timed out. Try a smaller repo or increase timeout."
            except Exception as e:
                return f"Analysis failed: {e}"
    
    def extract_github_content(self, repo_url: str, content_types: List[str] = None) -> List[Dict]:
        """Extract prompt-output pairs from GitHub repository content."""
        if content_types is None:
            content_types = ['issues', 'prs', 'notebooks', 'readme', 'examples']
        
        repo_parts = repo_url.rstrip('/').split('/')
        if len(repo_parts) < 2:
            return []
        
        owner, repo_name = repo_parts[-2], repo_parts[-1]
        examples = []
        rate_limited = False
        
        try:
            # Extract from issues
            if 'issues' in content_types and not rate_limited:
                try:
                    issue_examples = self._extract_issue_examples(owner, repo_name)
                    examples.extend(issue_examples)
                    print(f"Extracted {len(issue_examples)} examples from issues")
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        rate_limited = True
                        print("GitHub API rate limit reached, skipping remaining content types")
                    else:
                        print(f"Error extracting issues: {e}")
            
            # Extract from PRs
            if 'prs' in content_types and not rate_limited:
                try:
                    pr_examples = self._extract_pr_examples(owner, repo_name)
                    examples.extend(pr_examples)
                    print(f"Extracted {len(pr_examples)} examples from PRs")
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        rate_limited = True
                        print("GitHub API rate limit reached, skipping remaining content types")
                    else:
                        print(f"Error extracting PRs: {e}")
            
            # Extract from notebooks
            if 'notebooks' in content_types and not rate_limited:
                try:
                    notebook_examples = self._extract_notebook_examples(owner, repo_name)
                    examples.extend(notebook_examples)
                    print(f"Extracted {len(notebook_examples)} examples from notebooks")
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        rate_limited = True
                        print("GitHub API rate limit reached, skipping remaining content types")
                    else:
                        print(f"Error extracting notebooks: {e}")
            
            # Extract from README and examples
            if ('readme' in content_types or 'examples' in content_types) and not rate_limited:
                try:
                    readme_examples = self._extract_readme_examples(owner, repo_name)
                    examples.extend(readme_examples)
                    print(f"Extracted {len(readme_examples)} examples from README/examples")
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        rate_limited = True
                        print("GitHub API rate limit reached, skipping remaining content types")
                    else:
                        print(f"Error extracting README/examples: {e}")
            
            # Cache in memory
            if 'github_content_examples' not in self.memory.memory:
                self.memory.memory['github_content_examples'] = []
            self.memory.memory['github_content_examples'].extend(examples)
            self.memory._save()
            
            return examples
            
        except Exception as e:
            print(f"Error extracting content from {repo_url}: {e}")
            return []
    
    def _extract_issue_examples(self, owner: str, repo_name: str) -> List[Dict]:
        """Extract prompt-output pairs from GitHub issues."""
        examples = []
        
        try:
            # Get issues with labels that indicate examples/tutorials
            labels = ['example', 'tutorial', 'documentation', 'enhancement', 'feature']
            
            for label in labels:
                response = self.session.get(
                    f"https://api.github.com/repos/{owner}/{repo_name}/issues",
                    params={
                        'state': 'all',
                        'labels': label,
                        'per_page': 10,
                        'sort': 'updated',
                        'direction': 'desc'
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    issues = response.json()
                    
                    for issue in issues:
                        # Skip pull requests
                        if 'pull_request' in issue:
                            continue
                        
                        # Extract task from title and description
                        task = issue['title']
                        description = issue.get('body', '')
                        
                        # Look for code blocks in description
                        code_blocks = re.findall(r'```(?:python|py|js|javascript|json)?\n(.*?)\n```', 
                                               description, re.DOTALL)
                        
                        if code_blocks:
                            for code_block in code_blocks[:2]:  # Limit to 2 code blocks per issue
                                if len(code_block.strip()) > 50:  # Minimum code length
                                    examples.append({
                                        'task': f"Issue: {task}",
                                        'results': code_block.strip(),
                                        'source': f"GitHub Issue #{issue['number']}",
                                        'url': issue['html_url']
                                    })
                        
                        # If no code blocks, use the description as context
                        elif len(description) > 100:
                            examples.append({
                                'task': f"Issue: {task}",
                                'results': description[:500] + "..." if len(description) > 500 else description,
                                'source': f"GitHub Issue #{issue['number']}",
                                'url': issue['html_url']
                            })
                
                # Rate limiting
                if response.status_code == 403:
                    print("GitHub API rate limit reached")
                    break
                    
        except Exception as e:
            print(f"Error extracting issue examples: {e}")
        
        return examples[:20]  # Limit total examples
    
    def _extract_pr_examples(self, owner: str, repo_name: str) -> List[Dict]:
        """Extract prompt-output pairs from GitHub pull requests."""
        examples = []
        
        try:
            response = self.session.get(
                f"https://api.github.com/repos/{owner}/{repo_name}/pulls",
                params={
                    'state': 'all',
                    'per_page': 10,
                    'sort': 'updated',
                    'direction': 'desc'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                prs = response.json()
                
                for pr in prs:
                    # Get PR description
                    description = pr.get('body', '')
                    title = pr['title']
                    
                    # Look for code blocks in description
                    code_blocks = re.findall(r'```(?:python|py|js|javascript|json)?\n(.*?)\n```', 
                                           description, re.DOTALL)
                    
                    if code_blocks:
                        for code_block in code_blocks[:2]:
                            if len(code_block.strip()) > 50:
                                examples.append({
                                    'task': f"PR: {title}",
                                    'results': code_block.strip(),
                                    'source': f"GitHub PR #{pr['number']}",
                                    'url': pr['html_url']
                                })
                    
                    # Get file changes for code examples
                    try:
                        files_response = self.session.get(
                            f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr['number']}/files",
                            timeout=10
                        )
                        
                        if files_response.status_code == 200:
                            files = files_response.json()
                            
                            for file in files[:3]:  # Limit to 3 files per PR
                                if file['filename'].endswith('.py') and file.get('patch'):
                                    # Extract the patch content
                                    patch_content = file['patch']
                                    if patch_content and len(patch_content) > 100:
                                        examples.append({
                                            'task': f"PR: {title} - {file['filename']}",
                                            'results': patch_content,
                                            'source': f"GitHub PR #{pr['number']} file change",
                                            'url': pr['html_url']
                                        })
                    except Exception as e:
                        print(f"Error getting PR files: {e}")
                        pass  # Skip file changes if API fails
                
            elif response.status_code == 403:
                print("GitHub API rate limit reached")
                
        except Exception as e:
            print(f"Error extracting PR examples: {e}")
        
        return examples[:15]  # Limit total examples
    
    def _extract_notebook_examples(self, owner: str, repo_name: str) -> List[Dict]:
        """Extract prompt-output pairs from Jupyter notebooks."""
        examples = []
        
        try:
            # Search for notebook files with different extensions
            notebook_extensions = ['*.ipynb', '*.jupyter', '*.notebook']
            
            for ext in notebook_extensions:
                try:
                    response = self.session.get(
                        "https://api.github.com/search/code",
                        params={
                            'q': f'repo:{owner}/{repo_name} filename:{ext}',
                            'per_page': 5
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        notebook_files = response.json().get('items', [])
                        
                        for notebook_file in notebook_files:
                            try:
                                # Get notebook content
                                content_response = self.session.get(
                                    notebook_file['url'],
                                    timeout=10
                                )
                                
                                if content_response.status_code == 200:
                                    content = content_response.json()
                                    
                                    # Parse notebook cells
                                    if 'content' in content:
                                        notebook_content = content['content']
                                        if isinstance(notebook_content, str):
                                            # Handle base64 encoded content
                                            import base64
                                            try:
                                                notebook_content = base64.b64decode(notebook_content).decode('utf-8')
                                                notebook_data = json.loads(notebook_content)
                                            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                                                print(f"Error decoding notebook {notebook_file['name']}: {e}")
                                                continue
                                        else:
                                            notebook_data = notebook_content
                                        
                                        # Extract cells
                                        if 'cells' in notebook_data:
                                            for cell in notebook_data['cells']:
                                                if cell.get('cell_type') == 'code':
                                                    # Handle different source formats
                                                    source = cell.get('source', [])
                                                    if isinstance(source, list):
                                                        source = ''.join(source)
                                                    elif isinstance(source, str):
                                                        source = source
                                                    else:
                                                        continue
                                                    
                                                    outputs = cell.get('outputs', [])
                                                    
                                                    if source.strip() and len(source) > 50:
                                                        # Create example from code cell
                                                        task = f"Notebook cell: {notebook_file['name']}"
                                                        results = source.strip()
                                                        
                                                        # Add output if available
                                                        if outputs:
                                                            output_text = []
                                                            for output in outputs:
                                                                if 'text' in output:
                                                                    text = output['text']
                                                                    if isinstance(text, list):
                                                                        text = ''.join(text)
                                                                    output_text.append(text)
                                                                elif 'data' in output and 'text/plain' in output['data']:
                                                                    text = output['data']['text/plain']
                                                                    if isinstance(text, list):
                                                                        text = ''.join(text)
                                                                    output_text.append(text)
                                                            
                                                            if output_text:
                                                                results += f"\n\n# Output:\n" + '\n'.join(output_text)
                                                        
                                                        examples.append({
                                                            'task': task,
                                                            'results': results,
                                                            'source': f"Jupyter Notebook: {notebook_file['name']}",
                                                            'url': notebook_file['html_url']
                                                        })
                                                        
                                                        if len(examples) >= 5:  # Limit per notebook
                                                            break
                            
                            except Exception as e:
                                print(f"Error processing notebook {notebook_file['name']}: {e}")
                                continue
                    
                    elif response.status_code == 403:
                        print("GitHub API rate limit reached for notebook search")
                        break
                        
                except Exception as e:
                    print(f"Error searching for {ext} files: {e}")
                    continue
                
            # Also search for notebooks in common directories
            notebook_dirs = ['notebooks', 'examples', 'tutorials', 'docs']
            for dir_name in notebook_dirs:
                try:
                    response = self.session.get(
                        "https://api.github.com/search/code",
                        params={
                            'q': f'repo:{owner}/{repo_name} path:{dir_name} filename:*.ipynb',
                            'per_page': 3
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        additional_files = response.json().get('items', [])
                        # Process these files similarly to above
                        # (simplified to avoid code duplication)
                        
                except Exception as e:
                    print(f"Error searching in {dir_name}: {e}")
                    continue
                
        except Exception as e:
            print(f"Error extracting notebook examples: {e}")
        
        return examples[:20]  # Limit total examples
    
    def _extract_readme_examples(self, owner: str, repo_name: str) -> List[Dict]:
        """Extract prompt-output pairs from README and example files."""
        examples = []
        
        try:
            # Get README content
            readme_response = self.session.get(
                f"https://api.github.com/repos/{owner}/{repo_name}/readme",
                timeout=10
            )
            
            if readme_response.status_code == 200:
                readme_data = readme_response.json()
                
                # Decode content
                import base64
                readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                
                # Extract code blocks and their descriptions
                code_blocks = re.findall(r'```(?:python|py|js|javascript|json)?\n(.*?)\n```', 
                                       readme_content, re.DOTALL)
                
                # Find headers that might describe the code
                headers = re.findall(r'^#{1,3}\s+(.+?)$', readme_content, re.MULTILINE)
                
                # Pair headers with code blocks
                for i, code_block in enumerate(code_blocks[:10]):
                    if len(code_block.strip()) > 50:
                        task = f"README Example"
                        if i < len(headers):
                            task = f"README: {headers[i].strip()}"
                        
                        examples.append({
                            'task': task,
                            'results': code_block.strip(),
                            'source': f"README: {repo_name}",
                            'url': readme_data['html_url']
                        })
            
            # Search for example files
            example_response = self.session.get(
                "https://api.github.com/search/code",
                params={
                    'q': f'repo:{owner}/{repo_name} filename:example*.py OR filename:demo*.py OR filename:sample*.py',
                    'per_page': 5
                },
                timeout=10
            )
            
            if example_response.status_code == 200:
                example_files = example_response.json().get('items', [])
                
                for example_file in example_files:
                    try:
                        content_response = self.session.get(
                            example_file['url'],
                            timeout=10
                        )
                        
                        if content_response.status_code == 200:
                            content_data = content_response.json()
                            file_content = base64.b64decode(content_data['content']).decode('utf-8')
                            
                            if len(file_content) > 100:
                                examples.append({
                                    'task': f"Example: {example_file['name']}",
                                    'results': file_content,
                                    'source': f"Example file: {example_file['name']}",
                                    'url': example_file['html_url']
                                })
                    
                    except Exception as e:
                        print(f"Error processing example file {example_file['name']}: {e}")
                        continue
                
            elif example_response.status_code == 403:
                print("GitHub API rate limit reached")
                
        except Exception as e:
            print(f"Error extracting README examples: {e}")
        
        return examples[:15]  # Limit total examples
    
    def _analyze_repo_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure"""
        structure = {
            'total_files': 0,
            'python_files': 0,
            'test_files': 0,
            'has_ci': False,
            'has_docker': False,
            'main_dirs': []
        }
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and common ignore dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
            
            level = root.replace(repo_path, '').count(os.sep)
            if level == 1:  # Top-level directories
                structure['main_dirs'].extend(dirs)
            
            for file in files:
                structure['total_files'] += 1
                if file.endswith('.py'):
                    structure['python_files'] += 1
                    if 'test' in file.lower():
                        structure['test_files'] += 1
                elif file in ['.github/workflows', '.travis.yml', '.circleci/config.yml']:
                    structure['has_ci'] = True
                elif file == 'Dockerfile':
                    structure['has_docker'] = True
        
        return structure
    
    def _extract_patterns(self, repo_path: str) -> List[str]:
        """Extract coding patterns from repository"""
        patterns = set()
        
        # Sample up to 20 Python files
        py_files = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py') and 'test' not in file.lower():
                    py_files.append(os.path.join(root, file))
        
        for py_file in py_files[:20]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Pattern detection
                if 'async def' in content:
                    patterns.add("Async/await pattern")
                if '@retry' in content or 'tenacity' in content:
                    patterns.add("Retry pattern with decorators")
                if 'class.*Repository' in content:
                    patterns.add("Repository pattern")
                if 'class.*Service' in content:
                    patterns.add("Service layer pattern")
                if '@dataclass' in content or 'pydantic' in content:
                    patterns.add("Data validation with dataclasses/Pydantic")
                if 'logger' in content.lower():
                    patterns.add("Structured logging")
                if 'pytest' in content or 'unittest' in content:
                    patterns.add("Comprehensive testing")
                
            except:
                continue
        
        return list(patterns)
    
    def _analyze_dependencies(self, repo_path: str) -> List[str]:
        """Extract project dependencies"""
        deps = set()
        
        # Check various dependency files
        dep_files = {
            'requirements.txt': self._parse_requirements,
            'pyproject.toml': self._parse_pyproject,
            'setup.py': self._parse_setup_py,
            'Pipfile': self._parse_pipfile,
            'package.json': self._parse_package_json
        }
        
        for filename, parser in dep_files.items():
            filepath = os.path.join(repo_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    deps.update(parser(content))
                except:
                    continue
        
        return sorted(list(deps))
    
    def _parse_requirements(self, content: str) -> set:
        """Parse requirements.txt using regex for cleaner extraction"""
        deps = set()
        # Regex to match package names with version constraints
        pkg_pattern = r'^([a-zA-Z0-9_-]+)(?:[<>=!~].*)?$'
        
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                match = re.match(pkg_pattern, line)
                if match:
                    deps.add(match.group(1))
        return deps
    
    def _parse_pyproject(self, content: str) -> set:
        """Parse pyproject.toml using tomllib"""
        deps = set()
        try:
            data = tomllib.loads(content)
            # Extract dependencies from various possible locations
            if 'tool' in data and 'poetry' in data['tool']:
                poetry_deps = data['tool']['poetry'].get('dependencies', {})
                deps.update(poetry_deps.keys())
            if 'project' in data and 'dependencies' in data['project']:
                deps.update(data['project']['dependencies'])
        except Exception:
            # Fallback to basic parsing if tomllib fails
            in_deps = False
            for line in content.split('\n'):
                if '[tool.poetry.dependencies]' in line:
                    in_deps = True
                elif '[' in line:
                    in_deps = False
                elif in_deps and '=' in line:
                    pkg = line.split('=')[0].strip().strip('"')
                    if pkg and pkg != 'python':
                        deps.add(pkg)
        return deps
    
    def _parse_setup_py(self, content: str) -> set:
        """Parse setup.py using ast for safer extraction"""
        deps = set()
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == 'setup':
                    for keyword in node.keywords:
                        if keyword.arg == 'install_requires':
                            if isinstance(keyword.value, ast.List):
                                for item in keyword.value.elts:
                                    if isinstance(item, ast.Constant):
                                        pkg = item.value.split('>=')[0].split('==')[0].split('<=')[0]
                                        deps.add(pkg)
        except Exception:
            # Fallback to basic parsing
            if 'install_requires' in content:
                start = content.find('install_requires')
                if start > -1:
                    bracket_start = content.find('[', start)
                    bracket_end = content.find(']', bracket_start)
                    if bracket_start > -1 and bracket_end > -1:
                        requires_section = content[bracket_start+1:bracket_end]
                        for item in requires_section.split(','):
                            pkg = item.strip().strip('"').strip("'").split('>=')[0].split('==')[0]
                            if pkg:
                                deps.add(pkg)
        return deps
    
    def _parse_pipfile(self, content: str) -> set:
        """Parse Pipfile using regex for cleaner extraction"""
        deps = set()
        # Regex to match package definitions in Pipfile
        pkg_pattern = r'^([a-zA-Z0-9_-]+)\s*=\s*["\']?[^"\']*["\']?$'
        
        in_packages = False
        for line in content.split('\n'):
            if '[packages]' in line:
                in_packages = True
            elif '[' in line:
                in_packages = False
            elif in_packages:
                match = re.match(pkg_pattern, line.strip())
                if match:
                    deps.add(match.group(1))
        return deps
    
    def _parse_package_json(self, content: str) -> set:
        """Parse package.json using json module"""
        deps = set()
        try:
            data = json.loads(content)
            deps.update(data.get('dependencies', {}).keys())
            deps.update(data.get('devDependencies', {}).keys())
        except Exception:
            pass
        return deps
    
    def _format_structure(self, structure: Dict) -> str:
        """Format structure analysis"""
        return f"""- Total files: {structure['total_files']}
- Python files: {structure['python_files']}
- Test files: {structure['test_files']}
- Has CI/CD: {'Yes' if structure['has_ci'] else 'No'}
- Has Docker: {'Yes' if structure['has_docker'] else 'No'}
- Main directories: {', '.join(structure['main_dirs'][:5])}"""

# ==================== BUILD SYSTEM TOOLS ====================

class BuildTools:
    """Analyze and work with build systems"""
    
    def analyze_python_deps(self, project_path: str = ".") -> Dict[str, Any]:
        """Comprehensive dependency analysis"""
        results = {
            'manager': None,
            'dependencies': [],
            'dev_dependencies': [],
            'python_version': None,
            'total_size': 0
        }
        
        # Check for Poetry
        if os.path.exists(f"{project_path}/pyproject.toml"):
            results['manager'] = 'poetry'
            try:
                # Run poetry show
                output = subprocess.check_output(
                    ['poetry', 'show', '--tree'],
                    cwd=project_path,
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                results['dependencies'] = self._parse_poetry_tree(output)
                
                # Get size
                size_output = subprocess.check_output(
                    ['poetry', 'show', '--no-dev'],
                    cwd=project_path,
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                results['total_size'] = len(size_output.split('\n'))
            except:
                # Fallback to parsing pyproject.toml
                with open(f"{project_path}/pyproject.toml", 'r') as f:
                    content = f.read()
                    results['dependencies'] = list(self._parse_pyproject(content))
        
        # Check for pip
        elif os.path.exists(f"{project_path}/requirements.txt"):
            results['manager'] = 'pip'
            with open(f"{project_path}/requirements.txt", 'r') as f:
                content = f.read()
                results['dependencies'] = list(self._parse_requirements(content))
        
        # Check for pipenv
        elif os.path.exists(f"{project_path}/Pipfile"):
            results['manager'] = 'pipenv'
            with open(f"{project_path}/Pipfile", 'r') as f:
                content = f.read()
                results['dependencies'] = list(self._parse_pipfile(content))
        
        return results
    
    def _parse_poetry_tree(self, output: str) -> List[str]:
        """Parse poetry show --tree output using regex"""
        deps = []
        # Regex to extract package names from tree output
        pkg_pattern = r'^[├└│\s]*([a-zA-Z0-9_-]+)'
        
        for line in output.split('\n'):
            if line.strip():
                match = re.match(pkg_pattern, line)
                if match:
                    deps.append(match.group(1))
        return list(set(deps))  # Unique
    
    def run_linter(self, files: List[str] = None, auto_fix: bool = False) -> Dict[str, Any]:
        """Run ruff linter with optional auto-fix"""
        if files is None:
            files = glob.glob("**/*.py", recursive=True)
        
        results = {
            'total_issues': 0,
            'fixed_issues': 0,
            'files_checked': len(files),
            'issues_by_file': {}
        }
        
        # Check if ruff is installed
        try:
            subprocess.run(['ruff', '--version'], capture_output=True, check=True)
        except:
            results['error'] = "Ruff not installed. Install with: pip install ruff"
            return results
        
        # Run ruff
        cmd = ['ruff', 'check']
        if auto_fix:
            cmd.append('--fix')
        cmd.extend(files)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse output
            for line in result.stdout.split('\n'):
                if ':' in line and '.py' in line:
                    # Extract file and issue
                    parts = line.split(':')
                    if len(parts) >= 4:
                        filename = parts[0]
                        issue = ':'.join(parts[3:]).strip()
                        
                        if filename not in results['issues_by_file']:
                            results['issues_by_file'][filename] = []
                        results['issues_by_file'][filename].append(issue)
                        results['total_issues'] += 1
            
            if auto_fix:
                # Run again to count remaining issues
                result2 = subprocess.run(['ruff', 'check'] + files, capture_output=True, text=True)
                remaining = len([l for l in result2.stdout.split('\n') if ':' in l and '.py' in l])
                results['fixed_issues'] = results['total_issues'] - remaining
                results['total_issues'] = remaining
            
        except Exception as e:
            results['error'] = f"Linter error: {e}"
        
        return results

# ==================== LANGCHAIN TOOLS ====================

def create_research_tools(brain: ProjectBrain):
    """Create LangChain tools from research classes"""
    research_tools = RealResearchTools(brain)
    build_tools = BuildTools()
    git_tools = GitRepositoryTools(brain)
    
    @tool
    def web_search(query: str) -> str:
        """Search the web for current information"""
        return research_tools.web_search_integration(query)
    
    @tool
    def x_search(query: str) -> str:
        """Search X (Twitter) for latest discussions and trends"""
        return research_tools.x_search(query)
    
    @tool
    def github_search(query: str) -> str:
        """Search GitHub for code examples and repositories"""
        results = research_tools.github_code_search(query)
        return json.dumps(results, indent=2)
    
    @tool
    def analyze_github_repo(repo_url: str) -> str:
        """Deep analysis of a GitHub repository"""
        return research_tools.analyze_github_repo(repo_url)
    
    @tool
    def extract_github_content(repo_url: str, content_types: str = "issues,prs,notebooks,readme,examples") -> str:
        """Extract prompt-output pairs from GitHub repository content (issues, PRs, notebooks, README, examples)."""
        content_types_list = [ct.strip() for ct in content_types.split(',')]
        examples = research_tools.extract_github_content(repo_url, content_types_list)
        
        if examples:
            # Format results for display
            result = f"Extracted {len(examples)} examples from {repo_url}:\n\n"
            for i, example in enumerate(examples[:5], 1):  # Show first 5
                result += f"{i}. {example['task']}\n"
                result += f"   Source: {example['source']}\n"
                result += f"   URL: {example['url']}\n\n"
            
            if len(examples) > 5:
                result += f"... and {len(examples) - 5} more examples\n"
            
            return result
        else:
            return f"No examples found in {repo_url}"
    
    @tool
    def remember_decision(decision: str, reasoning: str, context: str = "") -> str:
        """Remember important architectural decisions"""
        context_dict = json.loads(context) if context else {}
        return brain.remember_decision(decision, reasoning, context_dict)
    
    @tool
    def find_relevant_files(query: str) -> str:
        """Find files in current project relevant to a query"""
        files = brain.find_relevant_files(query)
        return "\n".join([f"{f[0]} (similarity: {f[1]:.2f})" for f in files])
    
    @tool
    def analyze_dependencies(project_path: str = ".") -> str:
        """Analyze Python project dependencies"""
        results = build_tools.analyze_python_deps(project_path)
        return json.dumps(results, indent=2)
    
    @tool
    def run_linter(files: str = "", auto_fix: bool = False) -> str:
        """Run ruff linter on Python files"""
        file_list = files.split(',') if files else None
        results = build_tools.run_linter(file_list, auto_fix)
        return json.dumps(results, indent=2)
    
    @tool
    def git_commit(message: str, branch: str = "feature-branch") -> str:
        """Git operations"""
        try:
            # Check if git repo exists
            if not os.path.exists('.git'):
                os.system('git init')
            
            # Create/checkout branch
            os.system(f"git checkout -b {branch} 2>/dev/null || git checkout {branch}")
            
            # Add all changes
            os.system("git add .")
            
            # Commit
            result = os.system(f'git commit -m "{message}"')
            
            if result == 0:
                return f"✅ Committed to {branch}: {message}"
            else:
                return "ℹ️  No changes to commit or commit failed"
                
        except Exception as e:
            return f"❌ Git error: {str(e)}"
    
    @tool
    def clone_repository(repo_url: str, branch: str = "main") -> str:
        """Clone a Git repository to a temporary directory"""
        return git_tools.clone_repository(repo_url, branch)
    
    @tool
    def read_repository_file(repo_name: str, file_path: str) -> str:
        """Read a file from a cloned repository"""
        return git_tools.read_repository_file(repo_name, file_path)
    
    @tool
    def list_repository_files(repo_name: str, directory: str = ".") -> str:
        """List files in a repository directory"""
        return git_tools.list_repository_files(repo_name, directory)
    
    @tool
    def create_repository_file(repo_name: str, file_path: str, content: str, 
                             commit_message: str = "Add new file") -> str:
        """Create a new file in a cloned repository and commit it"""
        return git_tools.create_repository_file(repo_name, file_path, content, commit_message)
    
    @tool
    def modify_repository_file(repo_name: str, file_path: str, new_content: str,
                              commit_message: str = "Update file") -> str:
        """Modify an existing file in a cloned repository and commit it"""
        return git_tools.modify_repository_file(repo_name, file_path, new_content, commit_message)
    
    @tool
    def push_changes(repo_name: str, remote_name: str = "origin") -> str:
        """Push committed changes to remote repository"""
        return git_tools.push_changes(repo_name, remote_name)
    
    @tool
    def create_branch(repo_name: str, branch_name: str) -> str:
        """Create a new branch in the repository"""
        return git_tools.create_branch(repo_name, branch_name)
    
    @tool
    def get_repository_status(repo_name: str) -> str:
        """Get the current status of a repository"""
        return git_tools.get_repository_status(repo_name)
    
    @tool
    def cleanup_repository(repo_name: str) -> str:
        """Clean up a cloned repository"""
        return git_tools.cleanup_repository(repo_name)
    
    @tool
    def list_cloned_repositories() -> str:
        """List all currently cloned repositories"""
        return git_tools.list_cloned_repositories()
    
    return [
        web_search,
        x_search,
        github_search,
        analyze_github_repo,
        extract_github_content,
        remember_decision,
        find_relevant_files,
        analyze_dependencies,
        run_linter,
        git_commit,
        clone_repository,
        read_repository_file,
        list_repository_files,
        create_repository_file,
        modify_repository_file,
        push_changes,
        create_branch,
        get_repository_status,
        cleanup_repository,
        list_cloned_repositories
    ] 