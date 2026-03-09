export interface WebviewMessage {
    command: 'refresh' | 'log';
    text?: string;
}

export interface AgentStatus {
    id: string;
    name: string;
    status: 'active' | 'idle' | 'error' | 'completed';
    progress: number;
    currentTask?: string;
    lastUpdate: Date;
}

export interface AgentLog {
    timestamp: Date;
    agentId: string;
    message: string;
    level: 'info' | 'warning' | 'error' | 'success';
} 