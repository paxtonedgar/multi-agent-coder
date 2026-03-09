# Cursor Agent Monitor Extension

A VS Code extension for monitoring multi-agent coding processes in Cursor IDE. This extension provides a real-time dashboard to track the status and progress of various AI agents working on coding tasks.

## Features

- **Real-time Agent Status**: Monitor multiple agents (Code Generator, Code Reviewer, Test Generator, Documentation) with live progress bars
- **Activity Logs**: View detailed logs of agent activities with timestamps
- **Visual Status Indicators**: Color-coded status indicators (active, idle, error)
- **Responsive UI**: Clean, modern interface that adapts to VS Code's theme
- **Keyboard Shortcut**: Quick access with `Ctrl+Shift+A` (or `Cmd+Shift+A` on Mac)

## Installation

### Development Setup

1. **Clone and install dependencies**:
   ```bash
   git clone <repository-url>
   cd cursor-multi-agent-coding-ui-extension-
   npm install
   ```

2. **Compile the extension**:
   ```bash
   npm run compile
   ```

3. **Package the extension** (optional):
   ```bash
   npm run package
   ```

### Running in Development

1. Open the project in VS Code
2. Press `F5` to launch a new Extension Development Host window
3. In the new window, use `Ctrl+Shift+A` (or `Cmd+Shift+A`) to open the Agent Monitor
4. Or use the Command Palette (`Ctrl+Shift+P`) and search for "Start Agent Monitor"

## Usage

### Opening the Monitor

- **Keyboard Shortcut**: `Ctrl+Shift+A` (Windows/Linux) or `Cmd+Shift+A` (Mac)
- **Command Palette**: Press `Ctrl+Shift+P` and search for "Start Agent Monitor"
- **Command**: The monitor opens in a new panel beside your current editor

### Understanding the Interface

1. **Status Grid**: Shows real-time status of each agent
   - Green dot: Active and working
   - Yellow dot: Idle, waiting for tasks
   - Red dot: Error state
   - Progress bars show completion percentage

2. **Activity Log**: Displays chronological log of agent activities
   - Timestamps for each entry
   - Agent-specific messages
   - Auto-scrolls to show latest entries

3. **Refresh Button**: Manually refresh agent status (currently simulates refresh)

## Development

### Project Structure

```
src/
├── extension.ts      # Main extension entry point
├── types.ts         # TypeScript type definitions
└── webview/         # Webview UI components (future)
```

### Key Components

- **Extension Activation**: Registers the `agentMonitor.start` command
- **Webview Panel**: Creates and manages the monitoring interface
- **Message Handling**: Processes communication between extension and webview
- **Error Handling**: Graceful error handling with user notifications

### Adding New Features

1. **New Agent Types**: Add to the `AgentStatus` interface in `types.ts`
2. **UI Components**: Extend the webview HTML/CSS in `getWebviewContent()`
3. **Real Data Integration**: Replace mock data with actual agent status polling

## Configuration

The extension currently uses mock data for demonstration. To integrate with real agent systems:

1. Implement agent status polling in the extension
2. Replace mock data in `getWebviewContent()` with dynamic data
3. Add configuration options for agent endpoints and polling intervals

## Troubleshooting

### Common Issues

1. **Extension not loading**: Ensure all dependencies are installed (`npm install`)
2. **TypeScript errors**: Run `npm run compile` to check for compilation issues
3. **Webview not displaying**: Check browser console in the webview for JavaScript errors

### Debug Mode

Enable debug logging by checking the Developer Tools in the Extension Development Host:
1. Open Developer Tools (`Ctrl+Shift+I` or `Cmd+Option+I`)
2. Check the Console tab for extension logs
3. Monitor the webview communication in the Network tab

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details
