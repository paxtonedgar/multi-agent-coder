"""
Tests for FastAPI monitoring endpoint
"""

import pytest
import asyncio
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the app from main
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from main import app, update_workflow_state, workflow_state

@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)

@pytest.fixture
def reset_workflow_state():
    """Reset workflow state before each test"""
    global workflow_state
    workflow_state.clear()
    workflow_state.update({
        "phase": "Idle",
        "progress": 0,
        "logs": [],
        "current_task": "",
        "start_time": None,
        "end_time": None,
        "success": False,
        "error": None
    })
    yield
    # Cleanup after test
    workflow_state.clear()
    workflow_state.update({
        "phase": "Idle",
        "progress": 0,
        "logs": [],
        "current_task": "",
        "start_time": None,
        "end_time": None,
        "success": False,
        "error": None
    })

def test_monitor_endpoint_initial_state(client, reset_workflow_state):
    """Test monitor endpoint returns correct initial state"""
    response = client.get("/monitor")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["phase"] == "Idle"
    assert data["progress"] == 0
    assert data["logs"] == []
    assert data["current_task"] == ""
    assert data["start_time"] is None
    assert data["success"] is False
    assert data["error"] is None
    assert "timestamp" in data

def test_monitor_endpoint_with_workflow_state(client, reset_workflow_state):
    """Test monitor endpoint with active workflow state"""
    # Update workflow state
    update_workflow_state(
        phase="Research",
        progress=25,
        current_task="Build a web app",
        start_time=datetime.now(),
        log="Starting research phase"
    )
    
    response = client.get("/monitor")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["phase"] == "Research"
    assert data["progress"] == 25
    assert len(data["logs"]) == 1
    assert "Starting research phase" in data["logs"][0]
    assert data["current_task"] == "Build a web app"
    assert data["start_time"] is not None
    assert "timestamp" in data

def test_monitor_endpoint_progress_calculation(client, reset_workflow_state):
    """Test progress calculation based on phase"""
    # Test different phases and their expected progress
    test_cases = [
        ("Research", 10),
        ("Planning", 25),
        ("Coding", 50),
        ("Audit", 70),
        ("Review", 85),
        ("Deploy", 95),
        ("Complete", 100)
    ]
    
    for phase, expected_progress in test_cases:
        update_workflow_state(
            phase=phase,
            current_task="Test task",
            start_time=datetime.now()
        )
        
        response = client.get("/monitor")
        data = response.json()
        
        assert data["phase"] == phase
        assert data["progress"] == expected_progress

def test_monitor_endpoint_log_rotation(client, reset_workflow_state):
    """Test that logs are rotated to prevent memory bloat"""
    # Add more than 50 logs
    for i in range(60):
        update_workflow_state(log=f"Log message {i}")
    
    response = client.get("/monitor")
    data = response.json()
    
    # Should only return last 10 logs in response
    assert len(data["logs"]) <= 10
    
    # But should have more logs in memory (up to 50)
    assert len(workflow_state["logs"]) <= 50

def test_monitor_endpoint_completed_workflow(client, reset_workflow_state):
    """Test monitor endpoint with completed workflow"""
    from datetime import timedelta
    
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=30)  # 30 seconds later
    
    update_workflow_state(
        phase="Complete",
        progress=100,
        current_task="Build a web app",
        start_time=start_time,
        end_time=end_time,
        success=True,
        log="Workflow completed successfully!"
    )
    
    response = client.get("/monitor")
    data = response.json()
    
    assert data["phase"] == "Complete"
    assert data["progress"] == 100
    assert data["success"] is True
    assert data["error"] is None
    assert data["start_time"] is not None
    assert data["end_time"] is not None

def test_monitor_endpoint_failed_workflow(client, reset_workflow_state):
    """Test monitor endpoint with failed workflow"""
    update_workflow_state(
        phase="Failed",
        progress=0,
        current_task="Build a web app",
        start_time=datetime.now(),
        end_time=datetime.now(),
        success=False,
        error="Something went wrong",
        log="Workflow failed: Something went wrong"
    )
    
    response = client.get("/monitor")
    data = response.json()
    
    assert data["phase"] == "Failed"
    assert data["progress"] == 0
    assert data["success"] is False
    assert data["error"] == "Something went wrong"

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_update_workflow_state_function(reset_workflow_state):
    """Test update_workflow_state function"""
    # Test basic update
    update_workflow_state(phase="Testing", progress=50)
    assert workflow_state["phase"] == "Testing"
    assert workflow_state["progress"] == 50
    
    # Test log addition
    update_workflow_state(log="Test log message")
    assert len(workflow_state["logs"]) == 1
    assert "Test log message" in workflow_state["logs"][0]
    
    # Test additional fields
    update_workflow_state(custom_field="custom_value")
    assert workflow_state["custom_field"] == "custom_value"

def test_monitor_endpoint_timestamp_format(client, reset_workflow_state):
    """Test that timestamps are in ISO format"""
    update_workflow_state(
        phase="Research",
        start_time=datetime.now()
    )
    
    response = client.get("/monitor")
    data = response.json()
    
    # Check timestamp format
    assert "timestamp" in data
    # Should be ISO format (contains 'T' and 'Z' or timezone info)
    assert 'T' in data["timestamp"] or 'Z' in data["timestamp"]
    
    if data["start_time"]:
        assert 'T' in data["start_time"] or 'Z' in data["start_time"]

@pytest.mark.asyncio
async def test_monitor_endpoint_async():
    """Test monitor endpoint as async function"""
    from main import get_monitor
    
    # Test async function directly
    result = await get_monitor()
    
    assert isinstance(result, dict)
    assert "phase" in result
    assert "progress" in result
    assert "logs" in result
    assert "timestamp" in result

@pytest.mark.asyncio
async def test_health_endpoint_async():
    """Test health endpoint as async function"""
    from main import health_check
    
    # Test async function directly
    result = await health_check()
    
    assert isinstance(result, dict)
    assert result["status"] == "healthy"
    assert "timestamp" in result 