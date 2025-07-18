"""
Unit tests for enhanced memory security features
"""

import pytest
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime

from memory import SecurityConfig, CredentialManager, ProjectBrain, NodeType


class TestSecurityConfig:
    """Test security configuration and encryption"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.security = SecurityConfig()
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_encryption_key_creation(self):
        """Test encryption key creation and storage"""
        # Test that encryption key is created
        assert self.security.encryption_key is not None
        assert len(self.security.encryption_key) > 0
        
        # Test that cipher suite is initialized
        assert self.security.cipher_suite is not None
    
    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption"""
        test_data = "sensitive information"
        
        # Encrypt data
        encrypted = self.security.encrypt_data(test_data)
        assert encrypted != test_data
        assert isinstance(encrypted, str)
        
        # Decrypt data
        decrypted = self.security.decrypt_data(encrypted)
        assert decrypted == test_data
    
    def test_sensitive_field_detection(self):
        """Test sensitive field detection"""
        sensitive_fields = [
            "api_key", "password", "token", "secret", 
            "private_key", "credentials", "auth"
        ]
        
        for field in sensitive_fields:
            assert self.security.is_sensitive_field(field) is True, f"Field '{field}' should be sensitive"
        
        # Test non-sensitive fields
        non_sensitive = ["name", "title", "content", "description"]
        for field in non_sensitive:
            assert self.security.is_sensitive_field(field) is False, f"Field '{field}' should not be sensitive"
    
    def test_content_sanitization(self):
        """Test content sanitization"""
        test_content = """
        {
            "api_key": "secret123",
            "password": "mypassword",
            "token": "abc123",
            "name": "test"
        }
        """
        
        sanitized = self.security.sanitize_content(test_content)
        
        # Check that sensitive data is redacted
        assert "secret123" not in sanitized
        assert "mypassword" not in sanitized
        assert "abc123" not in sanitized
        assert "test" in sanitized  # Non-sensitive data should remain
    
    def test_encryption_failure_handling(self):
        """Test handling of encryption failures"""
        # Test with invalid data
        with patch.object(self.security.cipher_suite, 'encrypt', side_effect=Exception("Encryption failed")):
            result = self.security.encrypt_data("test")
            assert result == "test"  # Should return original data
    
    def test_decryption_failure_handling(self):
        """Test handling of decryption failures"""
        # Test with invalid encrypted data
        result = self.security.decrypt_data("invalid_encrypted_data")
        assert result == "invalid_encrypted_data"  # Should return original data


class TestCredentialManager:
    """Test credential management system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.security = SecurityConfig()
        self.credential_manager = CredentialManager(self.security)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_credential_storage_and_retrieval(self):
        """Test storing and retrieving credentials"""
        # Set credentials
        self.credential_manager.set_credential("GITHUB_TOKEN", "test_token_123")
        self.credential_manager.set_credential("OPENAI_API_KEY", "test_key_456")
        
        # Retrieve credentials
        github_token = self.credential_manager.get_credential("GITHUB_TOKEN")
        openai_key = self.credential_manager.get_credential("OPENAI_API_KEY")
        
        assert github_token == "test_token_123"
        assert openai_key == "test_key_456"
    
    def test_credential_listing(self):
        """Test listing stored credentials"""
        # Clear existing credentials for clean test
        existing_keys = self.credential_manager.list_credentials()
        for key in existing_keys:
            self.credential_manager.remove_credential(key)
        
        # Add some credentials
        self.credential_manager.set_credential("KEY1", "value1")
        self.credential_manager.set_credential("KEY2", "value2")
        
        # List credentials
        keys = self.credential_manager.list_credentials()
        
        assert "KEY1" in keys
        assert "KEY2" in keys
        assert len(keys) == 2
    
    def test_credential_removal(self):
        """Test removing credentials"""
        # Add credential
        self.credential_manager.set_credential("TEST_KEY", "test_value")
        assert self.credential_manager.get_credential("TEST_KEY") == "test_value"
        
        # Remove credential
        self.credential_manager.remove_credential("TEST_KEY")
        assert self.credential_manager.get_credential("TEST_KEY") is None
    
    def test_credential_rotation(self):
        """Test credential rotation"""
        # Add initial credential
        self.credential_manager.set_credential("ROTATE_KEY", "old_value")
        
        # Rotate credential
        result = self.credential_manager.rotate_credential("ROTATE_KEY", "new_value")
        assert "rotated successfully" in result
        
        # Verify new value
        assert self.credential_manager.get_credential("ROTATE_KEY") == "new_value"
    
    def test_credential_rotation_nonexistent(self):
        """Test rotating non-existent credential"""
        result = self.credential_manager.rotate_credential("NONEXISTENT", "new_value")
        assert "not found" in result
    
    def test_credential_persistence(self):
        """Test that credentials persist across instances"""
        # Create first instance and add credential
        self.credential_manager.set_credential("PERSIST_KEY", "persist_value")
        
        # Create new instance and check credential
        new_manager = CredentialManager(self.security)
        assert new_manager.get_credential("PERSIST_KEY") == "persist_value"


class TestProjectBrainSecurity:
    """Test ProjectBrain security integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.brain = ProjectBrain(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_security_components_initialization(self):
        """Test that security components are properly initialized"""
        assert hasattr(self.brain, 'security_config')
        assert hasattr(self.brain, 'credential_manager')
        assert self.brain.security_config is not None
        assert self.brain.credential_manager is not None
    
    def test_secure_node_creation(self):
        """Test creating nodes with security features"""
        # Create a node with sensitive metadata
        node_id = self.brain.add_node(
            node_type=NodeType.REFLECTION,
            content="Test reflection",
            metadata={"api_key": "secret123", "normal_field": "normal_value"}
        )
        
        assert node_id is not None
        
        # Check that the node was added
        nodes = [n for n in self.brain.memory['memory_nodes'] if n['id'] == node_id]
        assert len(nodes) == 1
        
        # The sensitive data should be encrypted in memory
        node = nodes[0]
        assert node['metadata']['api_key'] != "secret123"  # Should be encrypted
        assert node['metadata']['normal_field'] == "normal_value"  # Should remain normal
    
    def test_memory_size_limits(self):
        """Test memory size limits and pruning"""
        # Set small limits for testing
        self.brain.max_memory_nodes = 5
        self.brain.max_node_size = 100
        
        # Add nodes up to the limit
        for i in range(7):
            self.brain.add_node(
                node_type=NodeType.REFLECTION,
                content=f"Test reflection {i}",
                metadata={"index": i}
            )
        
        # Should have pruned oldest nodes
        assert len(self.brain.memory['memory_nodes']) <= 5
    
    def test_memory_compression(self):
        """Test memory compression features"""
        # Enable compression
        self.brain.compression_enabled = True
        
        # Create a large content node
        large_content = "x" * 2000  # Larger than compression threshold
        node_id = self.brain.add_node(
            node_type=NodeType.REFLECTION,
            content=large_content
        )
        
        # Check that content was compressed
        nodes = [n for n in self.brain.memory['memory_nodes'] if n['id'] == node_id]
        assert len(nodes) == 1
        node = nodes[0]
        assert node.get('compressed', False) is True
        assert node['content'].startswith('COMPRESSED:')
    
    def test_memory_export_security(self):
        """Test memory export with security features"""
        # Add node with sensitive data
        self.brain.add_node(
            node_type=NodeType.REFLECTION,
            content="Test content",
            metadata={"api_key": "secret123"}
        )
        
        # Export without encrypted data
        export_file = self.brain.export_memory(format='json', include_encrypted=False)
        
        # Check export file
        assert os.path.exists(export_file)
        
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        # Sensitive data should be redacted
        for node in export_data.get('memory_nodes', []):
            if 'metadata' in node and 'api_key' in node['metadata']:
                assert node['metadata']['api_key'] == "[ENCRYPTED]"
    
    def test_memory_stats(self):
        """Test memory statistics"""
        # Add some nodes
        for i in range(3):
            self.brain.add_node(
                node_type=NodeType.REFLECTION,
                content=f"Test content {i}"
            )
        
        # Get stats
        stats = self.brain.get_memory_stats()
        
        assert stats['total_nodes'] == 3
        assert stats['node_types']['reflection'] == 3
        assert stats['faiss_available'] in [True, False]  # Depends on FAISS availability
        assert stats['memory_usage_percent'] > 0
    
    def test_search_memory_with_filters(self):
        """Test memory search with date and type filters"""
        # Add nodes with different types and dates
        self.brain.add_node(
            node_type=NodeType.REFLECTION,
            content="Old reflection",
            metadata={"date": "2023-01-01"}
        )
        
        self.brain.add_node(
            node_type=NodeType.DECISION,
            content="Recent decision",
            metadata={"date": "2024-01-01"}
        )
        
        # Search with filters
        results = self.brain.search_memory(
            query="reflection",
            node_type="reflection",
            date_from="2023-01-01",
            date_to="2023-12-31"
        )
        
        assert len(results) > 0
        for result in results:
            assert result['node']['type'] == 'reflection'


class TestMemoryPagination:
    """Test memory pagination features"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.brain = ProjectBrain(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_retrieve_tree_pagination(self):
        """Test tree retrieval with pagination"""
        # Add multiple nodes to create a tree
        root_id = self.brain.add_node(
            node_type=NodeType.REFLECTION,
            content="Root reflection"
        )
        
        for i in range(10):
            self.brain.add_node(
                node_type=NodeType.REFLECTION,
                content=f"Child reflection {i}",
                parent_id=root_id
            )
        
        # Test pagination
        result = self.brain.retrieve_tree(
            query="reflection",
            max_depth=5,
            page=1,
            page_size=3
        )
        
        assert 'pagination' in result
        assert result['pagination']['page'] == 1
        assert result['pagination']['page_size'] == 3
        assert result['pagination']['total'] > 0
        assert result['pagination']['has_next'] is True
        assert result['pagination']['has_prev'] is False
        
        # Test second page
        result2 = self.brain.retrieve_tree(
            query="reflection",
            max_depth=5,
            page=2,
            page_size=3
        )
        
        assert result2['pagination']['page'] == 2
        assert result2['pagination']['has_prev'] is True


if __name__ == "__main__":
    pytest.main([__file__]) 