import * as vscode from 'vscode';
import { WebviewMessage } from './types';

/**
 * Global reference to the agent monitor panel to prevent multiple instances
 */
let agentMonitorPanel: vscode.WebviewPanel | undefined = undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('Agent Monitor extension is now active!');

    // Get configuration
    const config = vscode.workspace.getConfiguration('agentMonitor');
    const apiUrl = config.get<string>('apiUrl', 'http://localhost:8000/monitor');
    const autoStart = config.get<boolean>('autoStart', true);
    const pollInterval = config.get<number>('pollInterval', 5000);

    // Register the command to start the agent monitor
    let disposable = vscode.commands.registerCommand('agentMonitor.start', () => {
        startAgentMonitor(context, apiUrl, pollInterval);
    });

    context.subscriptions.push(disposable);

    // Auto-start functionality
    if (autoStart) {
        // Check if main.py exists in workspace and auto-start
        if (vscode.workspace.workspaceFolders) {
            const mainPyPattern = new vscode.RelativePattern(vscode.workspace.workspaceFolders[0], '**/main.py');
            vscode.workspace.findFiles(mainPyPattern).then(files => {
                if (files.length > 0) {
                    console.log('main.py found, auto-starting agent monitor');
                    startAgentMonitor(context, apiUrl, pollInterval);
                }
            });
        }

        // Listen for main.py file saves
        const saveListener = vscode.workspace.onDidSaveTextDocument((document: vscode.TextDocument) => {
            if (document.fileName.endsWith('main.py')) {
                console.log('main.py saved, auto-starting agent monitor');
                startAgentMonitor(context, apiUrl, pollInterval);
            }
        });

        context.subscriptions.push(saveListener);
    }

    // Listen for configuration changes
    const configListener = vscode.workspace.onDidChangeConfiguration((event) => {
        if (event.affectsConfiguration('agentMonitor')) {
            console.log('Agent Monitor configuration changed');
            // Update the webview if it's open
            if (agentMonitorPanel) {
                const newConfig = vscode.workspace.getConfiguration('agentMonitor');
                const newApiUrl = newConfig.get<string>('apiUrl', 'http://localhost:8000/monitor');
                const newPollInterval = newConfig.get<number>('pollInterval', 5000);
                
                agentMonitorPanel.webview.html = getWebviewContent(
                    agentMonitorPanel.webview, 
                    context.extensionUri, 
                    newApiUrl, 
                    newPollInterval
                );
            }
        }
    });

    context.subscriptions.push(configListener);
}

/**
 * Starts the agent monitor webview panel
 * @param context - Extension context
 * @param apiUrl - API endpoint URL for agent data
 * @param pollInterval - Polling interval in milliseconds
 */
function startAgentMonitor(
    context: vscode.ExtensionContext, 
    apiUrl: string, 
    pollInterval: number
): void {
    try {
        // If panel already exists, show it instead of creating a new one
        if (agentMonitorPanel) {
            agentMonitorPanel.reveal();
            return;
        }

        agentMonitorPanel = vscode.window.createWebviewPanel(
            'agentMonitor',
            'Agent Monitor',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(context.extensionUri, 'webview')
                ]
            }
        );

        // Set the webview content with configuration
        agentMonitorPanel.webview.html = getWebviewContent(
            agentMonitorPanel.webview, 
            context.extensionUri, 
            apiUrl, 
            pollInterval
        );

        // Handle messages from the webview
        agentMonitorPanel.webview.onDidReceiveMessage(
            (message: WebviewMessage) => {
                switch (message.command) {
                    case 'refresh':
                        vscode.window.showInformationMessage('Refreshing agent status...');
                        // TODO: Implement actual agent status refresh
                        break;
                    case 'log':
                        console.log('Webview log:', message.text);
                        break;
                }
            },
            undefined,
            context.subscriptions
        );

        // Handle panel disposal
        agentMonitorPanel.onDidDispose(() => {
            console.log('Agent Monitor panel disposed');
            agentMonitorPanel = undefined;
        }, null, context.subscriptions);

    } catch (error) {
        vscode.window.showErrorMessage(`Failed to create Agent Monitor: ${error}`);
        console.error('Error creating Agent Monitor:', error);
    }
}

export function deactivate() {
    console.log('Agent Monitor extension deactivated');
}

function getWebviewContent(
    _webview: vscode.Webview, 
    _extensionUri: vscode.Uri, 
    apiUrl: string = 'http://localhost:8000/monitor', 
    pollInterval: number = 5000
): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Monitor</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        .status-container {
            background: var(--vscode-editor-inactiveSelectionBackground);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 20px;
        }
        .phase-display {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--vscode-editor-foreground);
        }
        .progress-container {
            margin-bottom: 16px;
        }
        .progress-bar {
            width: 100%;
            height: 12px;
            background: var(--vscode-progressBar-background);
            border-radius: 6px;
            overflow: hidden;
            margin: 8px 0;
        }
        .progress-fill {
            height: 100%;
            background: var(--vscode-progressBar-foreground);
            transition: width 0.3s ease;
            border-radius: 6px;
        }
        .progress-text {
            font-size: 14px;
            color: var(--vscode-descriptionForeground);
        }
        .logs-container {
            background: var(--vscode-editor-inactiveSelectionBackground);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 6px;
            padding: 16px;
            max-height: 400px;
            overflow-y: auto;
        }
        .logs-container h3 {
            margin: 0 0 12px 0;
            font-size: 14px;
            font-weight: 600;
        }
        .log-entry {
            margin-bottom: 8px;
            padding: 8px;
            background: var(--vscode-editor-background);
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 12px;
            border-left: 3px solid var(--vscode-progressBar-foreground);
        }
        .log-timestamp {
            color: var(--vscode-descriptionForeground);
            font-size: 11px;
            margin-bottom: 4px;
        }
        .button {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        .error-message {
            color: var(--vscode-errorForeground);
            background: var(--vscode-inputValidation-errorBackground);
            border: 1px solid var(--vscode-inputValidation-errorBorder);
            border-radius: 4px;
            padding: 12px;
            margin: 16px 0;
            font-size: 14px;
        }
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active { background: #4ade80; }
        .status-idle { background: #fbbf24; }
        .status-error { background: #f87171; }
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Agent Monitor</h1>
        <button class="button" onclick="manualRefresh()">Refresh</button>
    </div>

    <div class="status-container">
        <div class="phase-display" id="phase">Phase: Loading...</div>
        <div class="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar" style="width: 0%"></div>
            </div>
            <div class="progress-text" id="progressText">0%</div>
        </div>
    </div>

    <div class="logs-container">
        <h3>Activity Log</h3>
        <div id="logs">
            <div class="log-entry">
                <div class="log-timestamp">Initializing...</div>
                <div>Agent Monitor starting up</div>
            </div>
        </div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let isPolling = true;
        let pollInterval;

        async function updateMonitor() {
            try {
                const response = await fetch('${apiUrl}');
                
                if (!response.ok) {
                    throw new Error(\`HTTP \${response.status}: \${response.statusText}\`);
                }
                
                const data = await response.json();
                
                // Update phase display
                const phaseElement = document.getElementById('phase');
                phaseElement.textContent = \`Phase: \${data.phase || 'Unknown'}\`;
                
                // Update progress bar
                const progressBar = document.getElementById('progressBar');
                const progressText = document.getElementById('progressText');
                const progress = data.progress || 0;
                
                progressBar.style.width = \`\${progress}%\`;
                progressText.textContent = \`\${progress}%\`;
                
                // Update logs
                const logsList = document.getElementById('logs');
                logsList.innerHTML = '';
                
                if (data.logs && Array.isArray(data.logs)) {
                    data.logs.forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry';
                        logEntry.innerHTML = \`
                            <div class="log-timestamp">\${new Date().toLocaleTimeString()}</div>
                            <div>\${log}</div>
                        \`;
                        logsList.appendChild(logEntry);
                    });
                } else {
                    // Add default log entry if no logs provided
                    const logEntry = document.createElement('div');
                    logEntry.className = 'log-entry';
                    logEntry.innerHTML = \`
                        <div class="log-timestamp">\${new Date().toLocaleTimeString()}</div>
                        <div>No activity logs available</div>
                    \`;
                    logsList.appendChild(logEntry);
                }
                
                // Remove any error messages
                const errorElement = document.querySelector('.error-message');
                if (errorElement) {
                    errorElement.remove();
                }
                
            } catch (error) {
                console.error('Failed to fetch monitor data:', error);
                
                // Show error message
                const existingError = document.querySelector('.error-message');
                if (!existingError) {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'error-message';
                    errorDiv.innerHTML = \`
                        <strong>Error: Agent not running</strong><br>
                        Unable to connect to ${apiUrl}<br>
                        <small>Make sure the agent server is running and accessible.</small>
                    \`;
                    
                    const statusContainer = document.querySelector('.status-container');
                    statusContainer.parentNode.insertBefore(errorDiv, statusContainer.nextSibling);
                }
                
                // Update UI to show error state
                document.getElementById('phase').textContent = 'Phase: Error - Agent not running';
                document.getElementById('progressBar').style.width = '0%';
                document.getElementById('progressText').textContent = '0%';
            }
        }

        function manualRefresh() {
            vscode.postMessage({ command: 'refresh' });
            updateMonitor();
        }

        function startPolling() {
            if (pollInterval) {
                clearInterval(pollInterval);
            }
            
            // Initial update
            updateMonitor();
            
            // Set up polling with configured interval
            pollInterval = setInterval(() => {
                if (isPolling) {
                    updateMonitor();
                }
            }, ${pollInterval});
        }

        function stopPolling() {
            isPolling = false;
            if (pollInterval) {
                clearInterval(pollInterval);
            }
        }

        // Start polling when page loads
        startPolling();

        // Handle page visibility changes to pause/resume polling
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                stopPolling();
            } else {
                isPolling = true;
                startPolling();
            }
        });

        // Clean up on page unload
        window.addEventListener('beforeunload', () => {
            stopPolling();
        });
    </script>
</body>
</html>`;
} 