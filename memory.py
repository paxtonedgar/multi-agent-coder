"""
Memory and persistence layer for LangGraph multi-agent system
"""

import os
import json
import glob
import ast
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import requests
import numpy as np

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated

# ==================== STATE DEFINITIONS ====================

class AgentState(TypedDict):
    """State passed between agents"""
    messages: Annotated[List[Dict], lambda old, new: old + new]
    current_step: Annotated[str, lambda old, new: new]
    task: Annotated[str, lambda old, new: new]
    repo_url: Annotated[str, lambda old, new: new]
    research_results: Annotated[List[Dict], lambda old, new: old + new]
    plan: Annotated[Dict, lambda old, new: new]
    code_files: Annotated[List[Dict], lambda old, new: old + new]
    review_feedback: Annotated[List[Dict], lambda old, new: old + new]
    deployment_status: Annotated[Dict, lambda old, new: new]
    brain_context: Annotated[Dict, lambda old, new: new]

# ==================== PROJECT BRAIN (PERSISTENCE) ====================

class ProjectBrain:
    """Persistent memory with real embeddings and graph understanding"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = project_path
        self.brain_file = f"{project_path}/.ai/brain.json"
        self.memory = self._load_or_init()
        
    def _load_or_init(self) -> Dict:
        """Load existing brain or initialize"""
        os.makedirs(os.path.dirname(self.brain_file), exist_ok=True)
        
        if os.path.exists(self.brain_file):
            try:
                with open(self.brain_file) as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'project_meta': {
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            },
            'codebase_map': {},
            'function_graph': {},  # Will be populated
            'decisions': [],
            'learnings': [],
            'github_examples': [],
            'deployment_history': [],
            'embeddings': {},
            'git_operations': [],
            'file_operations': []
        }
    
    def update_codebase_understanding(self):
        """Deep codebase analysis with AST and function graph"""
        print("Analyzing codebase...")
        file_count = 0
        
        for py_file in glob.glob(f"{self.project_path}/**/*.py", recursive=True):
            if any(skip in py_file for skip in ['__pycache__', '.venv', 'venv', '.ai']):
                continue
                
            file_count += 1
            rel_path = os.path.relpath(py_file, self.project_path)
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                # Extract comprehensive info
                functions = []
                classes = []
                imports = []
                calls = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append({
                            'name': node.name,
                            'args': [a.arg for a in node.args.args],
                            'lineno': node.lineno,
                            'docstring': ast.get_docstring(node)
                        })
                    elif isinstance(node, ast.ClassDef):
                        classes.append({
                            'name': node.name,
                            'bases': [b.id for b in node.bases if hasattr(b, 'id')],
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        })
                    elif isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module.split('.')[0])
                    elif isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                        calls.append(node.func.id)
                
                # Update codebase map
                self.memory['codebase_map'][rel_path] = {
                    'functions': functions,
                    'classes': classes,
                    'imports': imports,
                    'calls': calls,
                    'purpose': self._infer_purpose(rel_path, functions, classes, content),
                    'last_analyzed': datetime.now().isoformat(),
                    'size': len(content),
                    'lines': content.count('\n')
                }
                
                # Build function graph
                for func in functions:
                    func_key = f"{rel_path}::{func['name']}"
                    self.memory['function_graph'][func_key] = {
                        'calls': [c for c in calls if c in [f['name'] for f in functions]],
                        'called_by': []  # Will be populated in second pass
                    }
                
                # Create embedding
                self.memory['embeddings'][rel_path] = self._create_embedding(content)
                
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
        
        # Second pass: build reverse call graph
        for file_path, file_info in self.memory['codebase_map'].items():
            for func in file_info['functions']:
                func_key = f"{file_path}::{func['name']}"
                for call in file_info['calls']:
                    # Find where this call points to
                    for other_file, other_info in self.memory['codebase_map'].items():
                        for other_func in other_info['functions']:
                            if other_func['name'] == call:
                                called_key = f"{other_file}::{other_func['name']}"
                                if called_key in self.memory['function_graph']:
                                    self.memory['function_graph'][called_key]['called_by'].append(func_key)
        
        self._save()
        return f"Analyzed {file_count} Python files. Function graph has {len(self.memory['function_graph'])} nodes."
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding - use OpenAI API if available, else basic"""
        # Try to get OpenAI key from environment
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openai_key:
            # Real embedding via OpenAI
            try:
                response = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    json={"input": text[:8000], "model": "text-embedding-3-small"}
                )
                response.raise_for_status()
                return response.json()['data'][0]['embedding']
            except:
                pass
        
        # Fallback: TF-IDF style features
        words = text.lower().split()
        common_keywords = ['def', 'class', 'import', 'async', 'await', 'try', 'except',
                          'for', 'while', 'if', 'else', 'return', 'yield', 'with']
        features = [words.count(kw) / max(len(words), 1) for kw in common_keywords]
        return features
    
    def _infer_purpose(self, filepath: str, functions: List, classes: List, content: str) -> str:
        """Intelligently infer file purpose"""
        purpose_hints = []
        
        # Path-based
        path_lower = filepath.lower()
        if 'test' in path_lower:
            purpose_hints.append("Testing")
        if 'auth' in path_lower:
            purpose_hints.append("Authentication")
        if 'api' in path_lower:
            purpose_hints.append("API endpoints")
        if 'model' in path_lower:
            purpose_hints.append("Data models")
        if 'util' in path_lower or 'helper' in path_lower:
            purpose_hints.append("Utilities")
        
        # Content-based
        content_lower = content.lower()
        if any(term in content_lower for term in ['bet', 'odds', 'wager', 'gambling']):
            purpose_hints.append("Betting logic")
        if 'scrape' in content_lower or 'selenium' in content_lower or 'playwright' in content_lower:
            purpose_hints.append("Web scraping")
        if 'async def' in content:
            purpose_hints.append("Async operations")
        
        # Function name based
        func_names = [f['name'].lower() for f in functions]
        if any('login' in fn for fn in func_names):
            purpose_hints.append("Authentication")
        if any('scrape' in fn or 'extract' in fn for fn in func_names):
            purpose_hints.append("Data extraction")
        
        return " / ".join(purpose_hints) if purpose_hints else "General purpose"
    
    def find_relevant_files(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find files relevant to query using embeddings"""
        query_emb = self._create_embedding(query)
        
        similarities = []
        for file_path, file_emb in self.memory['embeddings'].items():
            try:
                sim = np.dot(query_emb, file_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(file_emb) + 1e-10)
            except:
                # Basic similarity without numpy
                sim = sum(q * f for q, f in zip(query_emb, file_emb)) / (len(query_emb) + 1)
            similarities.append((file_path, sim))
        
        # Return top files with scores
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def generate_refactoring_plan(self, description: str) -> Dict[str, Any]:
        """Generate multi-file refactoring plan"""
        relevant_files = self.find_relevant_files(description)
        
        plan = {
            'description': description,
            'affected_files': [f[0] for f in relevant_files[:10]],
            'steps': [],
            'estimated_changes': 0
        }
        
        # Analyze what needs to change
        if 'move' in description.lower() or 'extract' in description.lower():
            plan['steps'].append("1. Identify code to move")
            plan['steps'].append("2. Create new module/file")
            plan['steps'].append("3. Update imports in affected files")
            plan['steps'].append("4. Update tests")
            plan['estimated_changes'] = len(plan['affected_files']) * 3
        elif 'split' in description.lower():
            plan['steps'].append("1. Analyze monolithic file")
            plan['steps'].append("2. Extract components to new files")
            plan['steps'].append("3. Update references")
            plan['estimated_changes'] = len(plan['affected_files']) * 5
        else:
            plan['steps'].append("1. Analyze relevant files")
            plan['steps'].append("2. Identify changes")
            plan['steps'].append("3. Update code")
            plan['estimated_changes'] = len(plan['affected_files']) * 2
        
        return plan
    
    def remember_decision(self, decision: str, reasoning: str, context: Dict = None):
        """Remember architectural decisions"""
        self.memory['decisions'].append({
            'decision': decision,
            'reasoning': reasoning,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })
        self._save()
        return f"Remembered: {decision}"
    
    def _save(self):
        """Save brain to disk"""
        self.memory['project_meta']['last_updated'] = datetime.now().isoformat()
        with open(self.brain_file, 'w') as f:
            json.dump(self.memory, f, indent=2)

# ==================== LANGGRAPH CHECKPOINT ====================

class BrainCheckpoint(MemorySaver):
    """LangGraph checkpoint that integrates with ProjectBrain"""
    
    def __init__(self, brain: ProjectBrain):
        super().__init__()
        self.brain = brain
    
    def get(self, config: Dict) -> Optional[Dict]:
        """Get checkpoint data"""
        # Try to get from brain first
        checkpoint_id = config.get("configurable", {}).get("thread_id")
        if checkpoint_id:
            # Store in brain memory
            brain_key = f"checkpoint_{checkpoint_id}"
            if brain_key in self.brain.memory:
                return self.brain.memory[brain_key]
        
        # Fallback to parent
        return super().get(config)
    
    def put(self, config: Dict, value: Dict) -> None:
        """Store checkpoint data"""
        # Store in brain memory
        checkpoint_id = config.get("configurable", {}).get("thread_id")
        if checkpoint_id:
            brain_key = f"checkpoint_{checkpoint_id}"
            self.brain.memory[brain_key] = value
            self.brain._save()
        
        # Also store in parent
        super().put(config, value) 