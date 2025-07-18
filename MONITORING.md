# FastAPI Monitoring Endpoint

The multi-agent coding system now includes a FastAPI monitoring endpoint that provides real-time status updates for UI integration.

## Features

- **Real-time Progress Tracking**: Monitor workflow phases and progress percentage
- **Live Logs**: View recent workflow logs with timestamps
- **Status Information**: Track success/failure states and error messages
- **Health Check**: Simple health endpoint for monitoring system status
- **Background Server**: Runs in a separate thread without blocking the main workflow

## Usage

### Starting with Monitoring

```bash
# Start workflow with monitoring server
python main.py "Build a REST API with FastAPI" --monitor

# Use custom port
python main.py "Create a web app" --monitor --monitor-port 8080
```

### API Endpoints

#### GET /monitor
Returns current workflow status:

```json
{
  "phase": "Research",
  "progress": 25,
  "logs": [
    "[14:30:15] Starting research phase",
    "[14:30:20] Found 3 relevant repositories"
  ],
  "current_task": "Build a REST API with FastAPI",
  "start_time": "2025-01-27T14:30:10.123456",
  "end_time": null,
  "success": null,
  "error": null,
  "timestamp": "2025-01-27T14:30:25.654321"
}
```

#### GET /health
Simple health check:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T14:30:25.654321"
}
```

### Workflow Phases

The system tracks these phases with estimated progress:

- **Starting** (0%): Initialization
- **Research** (10%): Information gathering
- **Planning** (25%): Implementation planning
- **Coding** (50%): Code generation
- **Audit** (70%): Code review and quality checks
- **Review** (85%): Final review
- **Deploy** (95%): Deployment preparation
- **Complete** (100%): Workflow finished successfully
- **Failed** (0%): Workflow failed

### Example Monitoring Script

Use the provided `monitor_example.py` script to watch workflow progress:

```bash
# Monitor a running workflow
python monitor_example.py

# Test with sample data
python monitor_example.py test
```

### Integration with UI

The monitoring endpoint is designed for easy integration with web UIs:

```javascript
// Poll the endpoint for updates
async function monitorWorkflow() {
  const response = await fetch('http://localhost:8000/monitor');
  const data = await response.json();
  
  // Update UI with progress
  updateProgressBar(data.progress);
  updatePhaseDisplay(data.phase);
  updateLogs(data.logs);
  
  // Continue polling if not complete
  if (!['Complete', 'Failed'].includes(data.phase)) {
    setTimeout(monitorWorkflow, 2000);
  }
}
```

### Configuration

The monitoring server can be configured via command line arguments:

- `--monitor`: Enable monitoring server
- `--monitor-port`: Specify port (default: 8000)
- `--monitor-host`: Specify host (default: 0.0.0.0)

### State Management

The monitoring system uses a global state object that tracks:

- Current workflow phase
- Progress percentage (0-100)
- Recent logs (last 50, returned as last 10)
- Task description
- Start/end timestamps
- Success/failure status
- Error messages

### Error Handling

The monitoring system gracefully handles:

- Connection errors (returns helpful messages)
- Missing state data (provides defaults)
- Invalid progress values (uses phase estimation)
- Server startup failures (logs errors)

### Performance

- **Lightweight**: Minimal overhead on main workflow
- **Efficient**: Only stores last 50 logs in memory
- **Fast**: Sub-second response times
- **Scalable**: Can handle multiple concurrent requests

### Security Considerations

- **No Authentication**: Currently open access (add auth for production)
- **Local Only**: Binds to localhost by default
- **Read-Only**: Only provides status, no modification endpoints
- **Limited Data**: Only exposes workflow status, not sensitive content

### Future Enhancements

- **WebSocket Support**: Real-time updates without polling
- **Authentication**: API key or session-based auth
- **Metrics**: Performance and usage statistics
- **Notifications**: Webhook support for status changes
- **History**: Persistent workflow history storage 