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
import base64
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import requests
import numpy as np
import networkx as nx
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated

# ==================== SECURITY CONFIGURATION ====================

class SecurityConfig:
    """Security configuration for data encryption and privacy"""
    
    def __init__(self):
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.sensitive_fields = {
            'api_key', 'api_keys', 'token', 'tokens', 'password', 'passwords', 'credential', 'credentials', 
            'private_key', 'private_keys', 'secret', 'secrets', 'auth'
        }
        self.sanitization_patterns = [
            r'(password["\']?\s*[:=]\s*["\'])[^"\']*(["\'])',
            r'(token["\']?\s*[:=]\s*["\'])[^"\']*(["\'])',
            r'(api_key["\']?\s*[:=]\s*["\'])[^"\']*(["\'])',
            r'(secret["\']?\s*[:=]\s*["\'])[^"\']*(["\'])',
            r'(private_key["\']?\s*[:=]\s*["\'])[^"\']*(["\'])',
        ]
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get existing encryption key or create new one"""
        key_file = os.path.join(os.path.dirname(__file__), '.ai', '.encryption_key')
        os.makedirs(os.path.dirname(key_file), exist_ok=True)
        
        if os.path.exists(key_file):
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except:
                pass
        
        # Create new key
        key = Fernet.generate_key()
        with open(key_file, 'wb') as f:
            f.write(key)
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            print(f"Encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            print(f"Decryption failed: {e}")
            return encrypted_data
    
    def is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field contains sensitive data"""
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self.sensitive_fields)
    
    def sanitize_content(self, content: str) -> str:
        """Sanitize content by removing sensitive patterns"""
        sanitized = content
        for pattern in self.sanitization_patterns:
            sanitized = re.sub(pattern, r'\1[REDACTED]\2', sanitized, flags=re.IGNORECASE)
        return sanitized

# ==================== CREDENTIAL MANAGEMENT ====================

class CredentialManager:
    """Manages API keys and credentials securely"""
    
    def __init__(self, security_config: SecurityConfig):
        self.security = security_config
        self.credentials_file = os.path.join(os.path.dirname(__file__), '.ai', '.credentials')
        self.credentials = self._load_credentials()
    
    def _load_credentials(self) -> Dict[str, str]:
        """Load encrypted credentials from file"""
        if not os.path.exists(self.credentials_file):
            return {}
        
        try:
            with open(self.credentials_file, 'r') as f:
                encrypted_data = json.load(f)
            
            credentials = {}
            for key, encrypted_value in encrypted_data.items():
                credentials[key] = self.security.decrypt_data(encrypted_value)
            return credentials
        except Exception as e:
            print(f"Failed to load credentials: {e}")
            return {}
    
    def _save_credentials(self):
        """Save encrypted credentials to file"""
        try:
            encrypted_data = {}
            for key, value in self.credentials.items():
                encrypted_data[key] = self.security.encrypt_data(value)
            
            with open(self.credentials_file, 'w') as f:
                json.dump(encrypted_data, f)
        except Exception as e:
            print(f"Failed to save credentials: {e}")
    
    def get_credential(self, key: str) -> Optional[str]:
        """Get a credential by key"""
        return self.credentials.get(key)
    
    def set_credential(self, key: str, value: str):
        """Set a credential securely"""
        self.credentials[key] = value
        self._save_credentials()
    
    def remove_credential(self, key: str):
        """Remove a credential"""
        if key in self.credentials:
            del self.credentials[key]
            self._save_credentials()
    
    def list_credentials(self) -> List[str]:
        """List all credential keys (without values)"""
        return list(self.credentials.keys())
    
    def rotate_credential(self, key: str, new_value: str):
        """Rotate a credential to a new value"""
        if key in self.credentials:
            self.set_credential(key, new_value)
            return f"Credential '{key}' rotated successfully"
        else:
            return f"Credential '{key}' not found"

# ==================== CUSTOM JSON ENCODER ====================

class SecureEncoder(json.JSONEncoder):
    """Custom JSON encoder with security features"""
    
    def __init__(self, security_config: SecurityConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.security = security_config
    
    def default(self, obj):
        # Handle FAISS index
        if hasattr(obj, 'ntotal') and hasattr(obj, 'reconstruct'):
            try:
                # Convert FAISS index to serializable format
                vectors = []
                for i in range(min(obj.ntotal, 1000)):  # Limit to first 1000 vectors
                    try:
                        vector = obj.reconstruct(i).tolist()
                        vectors.append(vector)
                    except:
                        continue
                return {
                    'type': 'faiss_index',
                    'ntotal': obj.ntotal,
                    'vectors': vectors[:100]  # Keep only first 100 for memory
                }
            except:
                return {'type': 'faiss_index', 'error': 'Could not serialize'}
        
        # Handle NetworkX graphs
        if isinstance(obj, nx.Graph) or isinstance(obj, nx.DiGraph):
            try:
                return nx.to_dict_of_dicts(obj)
            except:
                return {'type': 'networkx_graph', 'nodes': list(obj.nodes()), 'edges': list(obj.edges())}
        
        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle numpy scalars
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        
        # Handle dict objects (should not happen but just in case)
        if isinstance(obj, dict):
            return obj
        
        return super().default(obj)
    
    def encode(self, obj):
        """Override encode to sanitize sensitive data"""
        if isinstance(obj, dict):
            obj = self._sanitize_dict(obj)
        return super().encode(obj)
    
    def _sanitize_dict(self, data: Dict) -> Dict:
        """Recursively sanitize dictionary for sensitive data"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_dict(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, str) and self.security.is_sensitive_field(key):
                sanitized[key] = self.security.encrypt_data(value)
            else:
                sanitized[key] = value
        return sanitized

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
        
        # Initialize security components
        self.security_config = SecurityConfig()
        self.credential_manager = CredentialManager(self.security_config)
        
        # Memory management settings (set before loading)
        self.max_memory_nodes = 10000  # Limit total memory nodes
        self.max_node_size = 1000000   # 1MB per node
        self.compression_enabled = True
        
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
                    memory_data = json.load(f)
                
                # Decompress memory if needed
                if self.compression_enabled:
                    self._decompress_memory()
                
                return memory_data
            except Exception as e:
                print(f"Failed to load brain: {e}")
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
        """Initialize FAISS index for vector search with comprehensive error handling"""
        try:
            import faiss
            # Create index for 1536-dimensional embeddings (OpenAI text-embedding-3-small)
            self.faiss_index = faiss.IndexFlatIP(1536)
            
            # Load existing embeddings into FAISS
            if self.memory.get('memory_nodes'):
                embeddings = []
                for node in self.memory['memory_nodes']:
                    if 'embedding' in node and node['embedding']:
                        embeddings.append(node['embedding'])
                
                if embeddings:
                    embedding_array = np.array(embeddings, dtype=np.float32)
                    self.faiss_index.add(embedding_array)
                    print(f"✅ FAISS index initialized with {len(embeddings)} existing embeddings")
                else:
                    print("✅ FAISS index initialized (no existing embeddings)")
            else:
                print("✅ FAISS index initialized for vector search")
                
        except ImportError:
            print("⚠️  FAISS not available, using basic similarity search")
            self.faiss_index = None
        except Exception as e:
            print(f"⚠️  FAISS initialization failed: {e}, using basic similarity search")
            self.faiss_index = None
    
    def add_node(self, node_type: NodeType, content: str, parent_id: str = None, 
                 sources: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Add a new node to the memory tree with scalability checks"""
        
        # Check content size limit
        if len(content) > self.max_node_size:
            content = content[:self.max_node_size] + "... [TRUNCATED]"
            print(f"⚠️  Node content truncated to {self.max_node_size} characters")
        
        # Check memory node limit before adding
        if len(self.memory['memory_nodes']) >= self.max_memory_nodes:
            nodes_to_remove = len(self.memory['memory_nodes']) - self.max_memory_nodes + 1
            self._prune_oldest_nodes(nodes_to_remove)
            print(f"⚠️  Memory limit reached, pruned {nodes_to_remove} oldest nodes")
        
        node_id = str(uuid.uuid4())
        embedding = self._create_embedding(content)
        
        # Encrypt sensitive data in metadata
        encrypted_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if self.security_config.is_sensitive_field(key) and isinstance(value, str):
                    encrypted_metadata[key] = self.security_config.encrypt_data(value)
                else:
                    encrypted_metadata[key] = value
        
        node = MemoryNode(
            id=node_id,
            type=node_type,
            content=content,
            embedding=embedding,
            timestamp=datetime.now().isoformat(),
            parent_id=parent_id,
            sources=sources or [],
            metadata=encrypted_metadata or {}
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
            except Exception as e:
                print(f"⚠️  Failed to add embedding to FAISS: {e}")
        
        # Update reflection graph if this is a reflection
        if node_type == NodeType.REFLECTION:
            self.reflection_graph.add_node(node_id, **node.to_dict())
            if parent_id and parent_id in self.reflection_graph:
                self.reflection_graph.add_edge(parent_id, node_id)
        
        self._save()
        return node_id
    
    def _prune_oldest_nodes(self, count: int = 100):
        """Remove oldest memory nodes to maintain size limits"""
        if len(self.memory['memory_nodes']) <= count:
            return
        
        # Sort by timestamp and remove oldest
        sorted_nodes = sorted(
            self.memory['memory_nodes'], 
            key=lambda x: x.get('timestamp', '1970-01-01')
        )
        
        nodes_to_remove = sorted_nodes[:count]
        nodes_to_keep = sorted_nodes[count:]
        
        # Remove from graphs
        for node in nodes_to_remove:
            node_id = node['id']
            if node_id in self.conversation_tree:
                self.conversation_tree.remove_node(node_id)
            if node_id in self.reflection_graph:
                self.reflection_graph.remove_node(node_id)
        
        # Update memory
        self.memory['memory_nodes'] = nodes_to_keep
        
        # Rebuild FAISS index if needed
        if self.faiss_index is not None:
            self._rebuild_faiss_index()
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from current memory nodes"""
        try:
            import faiss
            # Create new index
            self.faiss_index = faiss.IndexFlatIP(1536)
            
            # Add all current embeddings
            embeddings = []
            for node in self.memory['memory_nodes']:
                if 'embedding' in node and node['embedding']:
                    embeddings.append(node['embedding'])
            
            if embeddings:
                embedding_array = np.array(embeddings, dtype=np.float32)
                self.faiss_index.add(embedding_array)
                print(f"✅ FAISS index rebuilt with {len(embeddings)} embeddings")
        except Exception as e:
            print(f"⚠️  Failed to rebuild FAISS index: {e}")
            self.faiss_index = None
    
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
    
    def retrieve_tree(self, query: str, max_depth: int = 5, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """Retrieve relevant tree context for a query with pagination"""
        query_embedding = self._create_embedding(query)
        nearest_node = self._find_nearest_node(query_embedding)
        
        if not nearest_node:
            return {"nodes": [], "summary": "No relevant context found", "pagination": {"page": page, "total": 0}}
        
        context_nodes = self._traverse_graph(nearest_node, max_depth)
        
        # Apply pagination
        total_nodes = len(context_nodes)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_nodes = context_nodes[start_idx:end_idx]
        
        summary = self._summarize_tree_branch(paginated_nodes, query)
        
        return {
            "nodes": [self.conversation_tree.nodes[n] for n in paginated_nodes],
            "summary": summary,
            "root_node": nearest_node,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total_nodes,
                "total_pages": (total_nodes + page_size - 1) // page_size,
                "has_next": end_idx < total_nodes,
                "has_prev": page > 1
            }
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        total_nodes = len(self.memory['memory_nodes'])
        node_types = {}
        total_size = 0
        
        for node in self.memory['memory_nodes']:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            total_size += len(str(node.get('content', '')))
        
        return {
            "total_nodes": total_nodes,
            "node_types": node_types,
            "total_size_bytes": total_size,
            "max_nodes": self.max_memory_nodes,
            "max_node_size": self.max_node_size,
            "compression_enabled": self.compression_enabled,
            "faiss_available": self.faiss_index is not None,
            "graph_nodes": self.conversation_tree.number_of_nodes(),
            "graph_edges": self.conversation_tree.number_of_edges(),
            "reflection_nodes": self.reflection_graph.number_of_nodes(),
            "memory_usage_percent": (total_nodes / self.max_memory_nodes) * 100 if self.max_memory_nodes > 0 else 0
        }
    
    def search_memory(self, query: str, node_type: str = None, limit: int = 20, 
                     date_from: str = None, date_to: str = None) -> List[Dict]:
        """Search memory with advanced filtering"""
        query_embedding = self._create_embedding(query)
        results = []
        
        for node in self.memory['memory_nodes']:
            # Apply filters
            if node_type and node.get('type') != node_type:
                continue
            
            # Date filtering - check both timestamp and metadata date
            # A node passes if ANY of its dates fall within the range
            date_passed = True
            
            if date_from or date_to:
                date_passed = False
                dates_to_check = []
                
                # Check node timestamp
                if node.get('timestamp'):
                    try:
                        dates_to_check.append(datetime.fromisoformat(node.get('timestamp')))
                    except:
                        pass
                
                # Check metadata date
                if 'metadata' in node and 'date' in node['metadata']:
                    try:
                        dates_to_check.append(datetime.fromisoformat(node['metadata']['date']))
                    except:
                        pass
                
                # If no valid dates found, skip date filtering
                if not dates_to_check:
                    date_passed = True
                else:
                    # Check if any date falls within the range
                    for node_date in dates_to_check:
                        date_in_range = True
                        
                        if date_from:
                            try:
                                from_date = datetime.fromisoformat(date_from)
                                if node_date < from_date:
                                    date_in_range = False
                            except:
                                pass
                        
                        if date_to:
                            try:
                                to_date = datetime.fromisoformat(date_to)
                                if node_date > to_date:
                                    date_in_range = False
                            except:
                                pass
                        
                        if date_in_range:
                            date_passed = True
                            break
            
            if not date_passed:
                continue
            
            # Calculate similarity
            try:
                similarity = self._cosine_similarity(query_embedding, node.get('embedding', []))
                results.append({
                    'node': node,
                    'similarity': similarity
                })
            except:
                continue
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
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
        """Save brain to disk with custom encoder and compression"""
        self.memory['project_meta']['last_updated'] = datetime.now().isoformat()
        
        # Patch: ensure all memory_nodes have type as string
        for node in self.memory.get('memory_nodes', []):
            if isinstance(node.get('type'), NodeType):
                node['type'] = node['type'].value
        
        # Compress memory if enabled
        if self.compression_enabled:
            self._compress_memory()
        
        with open(self.brain_file, 'w') as f:
            json.dump(self.memory, f, indent=2, cls=lambda **kwargs: SecureEncoder(self.security_config, **kwargs))
    
    def _compress_memory(self):
        """Compress memory data to reduce size"""
        try:
            import gzip
            import base64
            
            # Compress large content fields
            for node in self.memory.get('memory_nodes', []):
                if 'content' in node and len(node['content']) > 1000:
                    compressed = gzip.compress(node['content'].encode('utf-8'))
                    node['content'] = f"COMPRESSED:{base64.b64encode(compressed).decode()}"
                    node['compressed'] = True
        except Exception as e:
            print(f"Memory compression failed: {e}")
    
    def _decompress_memory(self):
        """Decompress memory data when loading"""
        try:
            import gzip
            import base64
            
            for node in self.memory.get('memory_nodes', []):
                if node.get('compressed') and node['content'].startswith('COMPRESSED:'):
                    compressed_data = node['content'][11:]  # Remove 'COMPRESSED:' prefix
                    decompressed = gzip.decompress(base64.b64decode(compressed_data))
                    node['content'] = decompressed.decode('utf-8')
                    node['compressed'] = False
        except Exception as e:
            print(f"Memory decompression failed: {e}")
    
    def export_memory(self, format: str = 'json', include_encrypted: bool = False) -> str:
        """Export memory data in various formats"""
        if format == 'json':
            export_data = self.memory.copy()
            if not include_encrypted:
                # Remove encrypted fields
                export_data = self._remove_encrypted_fields(export_data)
            
            export_file = f"{self.project_path}/.ai/memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, cls=lambda **kwargs: SecureEncoder(self.security_config, **kwargs))
            return export_file
        
        elif format == 'csv':
            import csv
            export_file = f"{self.project_path}/.ai/memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            with open(export_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'type', 'content', 'timestamp', 'parent_id'])
                
                for node in self.memory.get('memory_nodes', []):
                    content = node.get('content', '')[:100]  # Truncate for CSV
                    writer.writerow([
                        node.get('id', ''),
                        node.get('type', ''),
                        content,
                        node.get('timestamp', ''),
                        node.get('parent_id', '')
                    ])
            return export_file
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _remove_encrypted_fields(self, data: Dict) -> Dict:
        """Remove encrypted fields from export data"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self._remove_encrypted_fields(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._remove_encrypted_fields(item) if isinstance(item, dict) else item 
                    for item in value
                ]
            elif isinstance(value, str) and self.security_config.is_sensitive_field(key):
                sanitized[key] = "[ENCRYPTED]"
            else:
                sanitized[key] = value
        return sanitized

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