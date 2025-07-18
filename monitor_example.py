#!/usr/bin/env python3
"""
Example script demonstrating the FastAPI monitoring endpoint
"""

import requests
import time
import json
from datetime import datetime

def monitor_workflow(monitor_url: str = "http://localhost:8000/monitor"):
    """Monitor workflow progress by polling the endpoint"""
    
    print("🔍 Monitoring workflow progress...")
    print(f"📊 Monitor URL: {monitor_url}")
    print("-" * 50)
    
    while True:
        try:
            response = requests.get(monitor_url)
            if response.status_code == 200:
                data = response.json()
                
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")
                
                # Display current status
                print(f"🔄 Phase: {data['phase']}")
                print(f"📈 Progress: {data['progress']}%")
                print(f"📝 Task: {data['current_task']}")
                print(f"⏰ Timestamp: {data['timestamp']}")
                
                if data['success'] is not None:
                    status = "✅ Success" if data['success'] else "❌ Failed"
                    print(f"🎯 Status: {status}")
                
                if data['error']:
                    print(f"🚨 Error: {data['error']}")
                
                # Display recent logs
                if data['logs']:
                    print("\n📋 Recent Logs:")
                    for log in data['logs'][-5:]:  # Show last 5 logs
                        print(f"  {log}")
                
                # Check if workflow is complete
                if data['phase'] in ['Complete', 'Failed']:
                    print("\n🏁 Workflow finished!")
                    break
                    
            else:
                print(f"❌ Failed to get status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to monitor server. Is it running?")
            print("💡 Start with: python main.py 'your task' --monitor")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break
        
        time.sleep(2)  # Poll every 2 seconds

def test_monitor_endpoint():
    """Test the monitor endpoint with sample data"""
    
    # This would be called from the main workflow
    from main import update_workflow_state
    
    print("🧪 Testing monitor endpoint...")
    
    # Simulate workflow phases
    phases = [
        ("Starting", 0, "Initializing workflow"),
        ("Research", 10, "Gathering information"),
        ("Planning", 25, "Creating implementation plan"),
        ("Coding", 50, "Writing code"),
        ("Audit", 70, "Reviewing code quality"),
        ("Review", 85, "Final review"),
        ("Complete", 100, "Workflow completed successfully!")
    ]
    
    for phase, progress, log in phases:
        update_workflow_state(
            phase=phase,
            progress=progress,
            current_task="Build a REST API with FastAPI",
            start_time=datetime.now(),
            log=log
        )
        
        print(f"✅ Updated state: {phase} ({progress}%)")
        time.sleep(1)  # Simulate work
    
    # Set completion
    update_workflow_state(
        success=True,
        end_time=datetime.now(),
        log="All done!"
    )
    
    print("✅ Test data populated!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode - populate sample data
        test_monitor_endpoint()
    else:
        # Monitor mode - watch the endpoint
        monitor_workflow() 