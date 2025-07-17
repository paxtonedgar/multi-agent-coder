"""
Enhanced Memory and persistence layer for LangGraph multi-agent system
Features persistent reflections, tree structures, and graph-based memory
"""

import os
import json
import glob
import ast
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import requests
import numpy as np
import networkx as nx
from dataclasses import dataclass, asdict
from enum import Enum

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated

# ==================== ENHANCED STATE DEFINITIONS ====================

class AgentState(TypedDict):
    """Enhanced state passed between agents with reflection context"""
    messages: Annotated[List[Dict], lambda old, new: old + new]
    current_step: Annotated[str, lambda old, new: new]
    task: Annotated[str, lambda old, new: new]
    repo_url: Annotated[str, lambda old, new: new]
    research_results: Annotated[List[Dict], lambda old, new: old + new]
    plan: Annotated[Dict, lambda old, new: new]
    audit_feedback: Annotated[List[Dict], lambda old, new: old + new]
    code_files: Annotated[List[Dict], lambda old, new: old + new]
    review_feedback: Annotated[List[Dict], lambda old, new: old + new]
    deployment_status: Annotated[Dict, lambda old, new: new]
    brain_context: Annotated[Dict, lambda old, new: new]
    reflection_context: Annotated[Dict, lambda old, new: new]  # New: reflection context

# ==================== MEMORY NODE TYPES ====================

class NodeType(Enum):
    """Types of nodes in the memory tree"""
    CHAT = "chat"
    REFLECTION = "reflection"
    DECISION = "decision"
    LEARNING = "learning"
    CODE_CHANGE = "code_change"

@dataclass
class MemoryNode:
    """Represents a node in the memory tree"""
    id: str
    type: NodeType
    content: str
    embedding: List[float]
    timestamp: str
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    sources: List[str] = None
    refinements: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.sources is None:
            self.sources = []
        if self.refinements is None:
            self.refinements = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        # Ensure all values are JSON serializable
        for key, value in data.items():
            if isinstance(value, NodeType):
                data[key] = value.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create from dictionary"""
        data['type'] = NodeType(data['type'])
        return cls(**data)

# ==================== ENHANCED PROJECT BRAIN ====================

class ProjectBrain:
    """Enhanced persistent memory with tree structures and reflections"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = project_path
        self.brain_file = f"{project_path}/.ai/brain.json"
        self.memory = self._load_or_init()
        
        # Initialize graph structures
        self.conversation_tree = nx.DiGraph()
        self.reflection_graph = nx.DiGraph()
        
        # Load existing nodes into graphs
        self._load_nodes_to_graphs()
        
        # Initialize FAISS index for vector search (if available)
        self.faiss_index = None
        self._init_faiss()
    
    def _load_or_init(self) -> Dict:
        """Load existing brain or initialize with enhanced structure"""
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
                'last_updated': datetime.now().isoformat(),
                'version': '2.0'  # Enhanced memory system
            },
            'codebase_map': {},
            'function_graph': {},
            'decisions': [],
            'learnings': [],
            'github_examples': [],
            'deployment_history': [],
            'embeddings': {},
            'git_operations': [],
            'file_operations': [],
            'memory_nodes': [],  # New: tree-based memory nodes
            'reflections': [],   # Legacy: keep for backward compatibility
            'conversation_threads': {},  # New: thread-based organization
            'reflection_evolution': {}   # New: track reflection improvements
        }
    
    def _load_nodes_to_graphs(self):
        """Load memory nodes into NetworkX graphs"""
        # Load conversation tree
        for node_data in self.memory.get('memory_nodes', []):
            try:
                node = MemoryNode.from_dict(node_data)
                self.conversation_tree.add_node(
                    node.id, 
                    **node.to_dict()
                )
                if node.parent_id:
                    self.conversation_tree.add_edge(node.parent_id, node.id)
            except Exception as e:
                print(f"Error loading node {node_data.get('id', 'unknown')}: {e}")
        
        # Load reflection graph (subset of conversation tree)
        reflection_nodes = [
            n for n in self.conversation_tree.nodes 
            if self.conversation_tree.nodes[n]['type'] == NodeType.REFLECTION.value
        ]
        self.reflection_graph = self.conversation_tree.subgraph(reflection_nodes).copy()
    
    def _init_faiss(self):
        """Initialize FAISS index for vector search"""
        try:
            import faiss
            # Create index for 1536-dimensional embeddings (OpenAI text-embedding-3-small)
            self.faiss_index = faiss.IndexFlatIP(1536)
            print("✅ FAISS index initialized for vector search")
        except ImportError:
            print("⚠️  FAISS not available, using basic similarity search")
            self.faiss_index = None
    
    def add_node(self, node_type: NodeType, content: str, parent_id: str = None, 
                 sources: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Add a new node to the memory tree"""
        node_id = str(uuid.uuid4())
        embedding = self._create_embedding(content)
        
        node = MemoryNode(
            id=node_id,
            type=node_type,
            content=content,
            embedding=embedding,
            timestamp=datetime.now().isoformat(),
            parent_id=parent_id,
            sources=sources or [],
            metadata=metadata or {}
        )
        
        # Add to graph
        self.conversation_tree.add_node(node_id, **node.to_dict())
        if parent_id:
            self.conversation_tree.add_edge(parent_id, node_id)
        
        # Add to memory (ensure type is string)
        node_dict = node.to_dict()
        node_dict['type'] = node_dict['type'] if isinstance(node_dict['type'], str) else node_dict['type'].value
        self.memory['memory_nodes'].append(node_dict)
        
        # Update FAISS index if available
        if self.faiss_index is not None:
            try:
                import faiss
                embedding_array = np.array([embedding], dtype=np.float32)
                self.faiss_index.add(embedding_array)
            except:
                pass
        
        # Update reflection graph if this is a reflection
        if node_type == NodeType.REFLECTION:
            self.reflection_graph.add_node(node_id, **node.to_dict())
            if parent_id and parent_id in self.reflection_graph:
                self.reflection_graph.add_edge(parent_id, node_id)
        
        self._save()
        return node_id
    
    def add_reflection(self, content: str, parent_id: str = None, sources: List[str] = None) -> str:
        """Add a new reflection node"""
        return self.add_node(
            node_type=NodeType.REFLECTION,
            content=content,
            parent_id=parent_id,
            sources=sources
        )
    
    def refine_reflection(self, reflection_id: str, new_content: str, sources: List[str] = None) -> str:
        """Create a refined version of an existing reflection"""
        # Create child node linked to parent
        child_id = self.add_reflection(
            content=new_content,
            parent_id=reflection_id,
            sources=sources
        )
        
        # Update parent's refinements list
        if reflection_id in self.conversation_tree:
            self.conversation_tree.nodes[reflection_id]['refinements'].append(child_id)
            # Update in memory
            for node_data in self.memory['memory_nodes']:
                if node_data['id'] == reflection_id:
                    node_data['refinements'].append(child_id)
                    break
        
        self._save()
        return child_id
    
    def get_reflection_tree(self, query: str, depth: int = 3) -> Dict[str, Any]:
        """Get reflection tree context based on query"""
        query_embedding = self._create_embedding(query)
        
        # Find nearest reflection node
        nearest_node = self._find_nearest_node(query_embedding, NodeType.REFLECTION)
        if not nearest_node:
            return {"nodes": [], "summary": "No relevant reflections found"}
        
        # Traverse graph to get context
        context_nodes = self._traverse_graph(nearest_node, depth)
        
        # Create summary using DSPy-style approach
        summary = self._summarize_tree_branch(context_nodes, query)
        
        return {
            "nodes": [self.conversation_tree.nodes[n] for n in context_nodes],
            "summary": summary,
            "root_node": nearest_node
        }
    
    def _find_nearest_node(self, query_embedding: List[float], node_type: NodeType = None) -> Optional[str]:
        """Find nearest node using FAISS or basic similarity"""
        if self.faiss_index is not None:
            try:
                import faiss
                query_array = np.array([query_embedding], dtype=np.float32)
                scores, indices = self.faiss_index.search(query_array, 10)
                
                # Filter by node type if specified
                candidates = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.memory['memory_nodes']):
                        node_data = self.memory['memory_nodes'][idx]
                        if node_type is None or node_data['type'] == node_type.value:
                            candidates.append((node_data['id'], score))
                
                return candidates[0][0] if candidates else None
            except:
                pass
        
        # Fallback to basic similarity search
        best_score = -1
        best_node = None
        
        for node_data in self.memory['memory_nodes']:
            if node_type and node_data['type'] != node_type.value:
                continue
            
            try:
                sim = self._cosine_similarity(query_embedding, node_data['embedding'])
                if sim > best_score:
                    best_score = sim
                    best_node = node_data['id']
            except:
                continue
        
        return best_node
    
    def _traverse_graph(self, start_node: str, max_depth: int) -> List[str]:
        """Traverse graph to get context nodes"""
        visited = set()
        nodes_to_visit = [(start_node, 0)]
        context_nodes = []
        
        while nodes_to_visit:
            node, depth = nodes_to_visit.pop(0)
            
            if node in visited or depth > max_depth:
                continue
            
            visited.add(node)
            context_nodes.append(node)
            
            # Add ancestors (parents)
            for parent in self.conversation_tree.predecessors(node):
                if parent not in visited:
                    nodes_to_visit.append((parent, depth + 1))
            
            # Add descendants (children)
            for child in self.conversation_tree.successors(node):
                if child not in visited:
                    nodes_to_visit.append((child, depth + 1))
        
        return context_nodes
    
    def _summarize_tree_branch(self, node_ids: List[str], query: str) -> str:
        """Summarize a tree branch using DSPy-style approach"""
        if not node_ids:
            return "No context available"
        
        # Get node contents
        contents = []
        for node_id in node_ids:
            if node_id in self.conversation_tree:
                node_data = self.conversation_tree.nodes[node_id]
                contents.append(f"[{node_data['type']}] {node_data['content'][:200]}...")
        
        # Simple summarization (in production, use LLM)
        summary = f"Found {len(contents)} related nodes for query: '{query}'\n"
        summary += "\n".join(contents[:5])  # Show first 5 nodes
        
        return summary
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        except:
            # Fallback for non-numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm1 * norm2 + 1e-10)
    
    def retrieve_tree(self, query: str, max_depth: int = 5) -> Dict[str, Any]:
        """Retrieve relevant tree context for a query"""
        query_embedding = self._create_embedding(query)
        nearest_node = self._find_nearest_node(query_embedding)
        
        if not nearest_node:
            return {"nodes": [], "summary": "No relevant context found"}
        
        context_nodes = self._traverse_graph(nearest_node, max_depth)
        summary = self._summarize_tree_branch(context_nodes, query)
        
        return {
            "nodes": [self.conversation_tree.nodes[n] for n in context_nodes],
            "summary": summary,
            "root_node": nearest_node
        }
    
    def prune_old_branches(self, days_old: int = 30, similarity_threshold: float = 0.3):
        """Prune old, low-similarity branches"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        nodes_to_remove = []
        
        for node_id in self.conversation_tree.nodes():
            node_data = self.conversation_tree.nodes[node_id]
            node_date = datetime.fromisoformat(node_data['timestamp'])
            
            if node_date < cutoff_date:
                # Check if node has low similarity to recent nodes
                recent_nodes = [
                    n for n in self.conversation_tree.nodes()
                    if datetime.fromisoformat(self.conversation_tree.nodes[n]['timestamp']) >= cutoff_date
                ]
                
                if recent_nodes:
                    max_similarity = max(
                        self._cosine_similarity(
                            node_data['embedding'],
                            self.conversation_tree.nodes[recent]['embedding']
                        )
                        for recent in recent_nodes[:10]  # Check against 10 most recent
                    )
                    
                    if max_similarity < similarity_threshold:
                        nodes_to_remove.append(node_id)
        
        # Remove nodes
        for node_id in nodes_to_remove:
            self.conversation_tree.remove_node(node_id)
            # Remove from memory
            self.memory['memory_nodes'] = [
                n for n in self.memory['memory_nodes'] 
                if n['id'] != node_id
            ]
        
        if nodes_to_remove:
            print(f"Pruned {len(nodes_to_remove)} old nodes")
            self._save()
    
    def visualize_tree(self, thread_id: str = None, output_path: str = None) -> str:
        """Visualize the memory tree using Graphviz"""
        try:
            import graphviz
        except ImportError:
            return "Graphviz not available. Install with: pip install graphviz"
        
        # Create graph
        dot = graphviz.Digraph(comment='Memory Tree')
        dot.attr(rankdir='TB')
        
        # Add nodes
        for node_id in self.conversation_tree.nodes():
            node_data = self.conversation_tree.nodes[node_id]
            label = f"{node_data['type'][:3]}: {node_data['content'][:50]}..."
            
            # Color by type
            colors = {
                NodeType.CHAT.value: 'lightblue',
                NodeType.REFLECTION.value: 'lightgreen',
                NodeType.DECISION.value: 'lightyellow',
                NodeType.LEARNING.value: 'lightcoral',
                NodeType.CODE_CHANGE.value: 'lightgray'
            }
            color = colors.get(node_data['type'], 'white')
            
            dot.node(node_id, label, style='filled', fillcolor=color)
        
        # Add edges
        for edge in self.conversation_tree.edges():
            dot.edge(edge[0], edge[1])
        
        # Save or display
        if output_path:
            dot.render(output_path, format='png', cleanup=True)
            return f"Tree visualization saved to {output_path}.png"
        else:
            # Try to open on macOS
            try:
                dot.render('/tmp/memory_tree', format='png', cleanup=True)
                os.system(f"open /tmp/memory_tree.png")
                return "Tree visualization opened in Preview"
            except:
                return "Tree visualization created. Install graphviz to view."
    
    def get_conversation_context(self, thread_id: str) -> Dict[str, Any]:
        """Get conversation context for a specific thread"""
        thread_nodes = [
            n for n in self.conversation_tree.nodes()
            if self.conversation_tree.nodes[n].get('metadata', {}).get('thread_id') == thread_id
        ]
        
        if not thread_nodes:
            return {"nodes": [], "summary": "No conversation history found"}
        
        # Get the conversation tree for this thread
        thread_graph = self.conversation_tree.subgraph(thread_nodes)
        
        # Find root node (no parents)
        root_nodes = [n for n in thread_graph.nodes() if thread_graph.in_degree(n) == 0]
        
        if root_nodes:
            context_nodes = self._traverse_graph(root_nodes[0], 10)
            summary = self._summarize_tree_branch(context_nodes, f"Thread {thread_id}")
        else:
            summary = "Conversation thread found but no clear structure"
        
        return {
            "nodes": [self.conversation_tree.nodes[n] for n in thread_nodes],
            "summary": summary,
            "thread_id": thread_id
        }

    # ==================== LEGACY METHODS (BACKWARD COMPATIBILITY) ====================
    
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
                        'called_by': []
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
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openai_key:
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
        """Intelligently infer file purpose using efficient patterns"""
        purpose_hints = set()
        
        path_patterns = {
            r'test': "Testing",
            r'auth': "Authentication", 
            r'api': "API endpoints",
            r'model': "Data models",
            r'util|helper': "Utilities"
        }
        
        path_lower = filepath.lower()
        for pattern, purpose in path_patterns.items():
            if re.search(pattern, path_lower):
                purpose_hints.add(purpose)
        
        content_patterns = {
            r'\b(bet|odds|wager|gambling)\b': "Betting logic",
            r'\b(scrape|selenium|playwright)\b': "Web scraping",
            r'async def': "Async operations"
        }
        
        content_lower = content.lower()
        for pattern, purpose in content_patterns.items():
            if re.search(pattern, content_lower):
                purpose_hints.add(purpose)
        
        func_names = {f['name'].lower() for f in functions}
        func_patterns = {
            'login': "Authentication",
            'scrape|extract': "Data extraction"
        }
        
        for pattern, purpose in func_patterns.items():
            if any(re.search(pattern, fn) for fn in func_names):
                purpose_hints.add(purpose)
        
        return " / ".join(sorted(purpose_hints)) if purpose_hints else "General purpose"
    
    def find_relevant_files(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find files relevant to query using embeddings"""
        query_emb = self._create_embedding(query)
        
        similarities = []
        for file_path, file_emb in self.memory['embeddings'].items():
            try:
                sim = np.dot(query_emb, file_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(file_emb) + 1e-10)
            except:
                sim = sum(q * f for q, f in zip(query_emb, file_emb)) / (len(query_emb) + 1)
            similarities.append((file_path, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def remember_decision(self, decision: str, reasoning: str, context: Dict = None):
        """Remember architectural decisions"""
        # Add as a decision node
        self.add_node(
            node_type=NodeType.DECISION,
            content=f"Decision: {decision}\nReasoning: {reasoning}",
            metadata=context or {}
        )
        
        # Legacy support
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
        # Patch: ensure all memory_nodes have type as string
        for node in self.memory.get('memory_nodes', []):
            if isinstance(node.get('type'), NodeType):
                node['type'] = node['type'].value
        with open(self.brain_file, 'w') as f:
            json.dump(self.memory, f, indent=2)

# ==================== ENHANCED LANGGRAPH CHECKPOINT ====================

class BrainCheckpoint(MemorySaver):
    """Enhanced LangGraph checkpoint that integrates with ProjectBrain"""
    
    def __init__(self, brain: ProjectBrain):
        super().__init__()
        self.brain = brain
    
    def get(self, config: Dict) -> Optional[Dict]:
        """Get checkpoint data with tree context"""
        checkpoint_id = config.get("configurable", {}).get("thread_id")
        if checkpoint_id:
            # Get conversation context
            context = self.brain.get_conversation_context(checkpoint_id)
            
            # Store in brain memory
            brain_key = f"checkpoint_{checkpoint_id}"
            if brain_key in self.brain.memory:
                checkpoint_data = self.brain.memory[brain_key]
                # Add tree context
                checkpoint_data['tree_context'] = context
                return checkpoint_data
        
        return super().get(config)
    
    def put(self, config: Dict, value: Dict, metadata=None, new_versions=None) -> None:
        """Store checkpoint data with tree integration"""
        checkpoint_id = config.get("configurable", {}).get("thread_id")
        if checkpoint_id:
            brain_key = f"checkpoint_{checkpoint_id}"
            self.brain.memory[brain_key] = value
            self.brain._save()
        
        super().put(config, value, metadata, new_versions) 