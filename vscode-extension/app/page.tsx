"use client"

import { useState, useEffect } from "react"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { MessageSquare, Sparkles, Users } from "lucide-react"

export default function MultiAgentExtension() {
  const [activeAgents, setActiveAgents] = useState<Record<string, boolean>>({
    planner: true,
    coder1: false,
    coder2: false,
    reviewer: false,
    integrator: false,
  })

  const [progress, setProgress] = useState(15)

  // Simulate agent activity
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveAgents((prev) => {
        const newState = { ...prev }
        // Randomly activate/deactivate agents
        Object.keys(newState).forEach((agent) => {
          newState[agent] = Math.random() > 0.6
        })
        return newState
      })

      // Progress simulation
      if (progress < 100) {
        const newProgress = Math.min(progress + Math.floor(Math.random() * 3), 100)
        setProgress(newProgress)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [progress])

  return (
    <div className="h-screen w-80 bg-zinc-950 text-white border-l border-zinc-800 flex flex-col">
      {/* Header */}
      <div className="p-3 border-b border-zinc-800">
        <div className="flex items-center justify-between">
          <h1 className="text-sm font-semibold">Multi-Agent System</h1>
          <Badge variant="outline" className="bg-green-900/30 text-green-400 text-xs">
            Active
          </Badge>
        </div>
        <p className="text-xs text-zinc-400 mt-1">E-commerce API Project</p>
      </div>

      {/* Progress */}
      <div className="p-3 border-b border-zinc-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-zinc-400">Progress</span>
          <span className="text-xs font-medium">{progress}%</span>
        </div>
        <Progress value={progress} className="h-1.5" />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <Tabs defaultValue="agents" className="h-full flex flex-col">
          <TabsList className="bg-zinc-900 border-zinc-800 mx-3 mt-3">
            <TabsTrigger value="agents" className="text-xs">
              <Users className="h-3 w-3 mr-1" />
              Agents
            </TabsTrigger>
            <TabsTrigger value="chat" className="text-xs">
              <MessageSquare className="h-3 w-3 mr-1" />
              Chat
            </TabsTrigger>
          </TabsList>

          <TabsContent value="agents" className="flex-1 overflow-y-auto px-3 pb-3">
            <div className="space-y-2 mt-2">
              <AgentCard
                name="Planner"
                role="Architecture"
                isActive={activeAgents.planner}
                color="blue"
                message="Analyzing requirements..."
              />

              <AgentCard
                name="Coder-1"
                role="Backend"
                isActive={activeAgents.coder1}
                color="green"
                message="Implementing auth system"
              />

              <AgentCard
                name="Coder-2"
                role="Frontend"
                isActive={activeAgents.coder2}
                color="purple"
                message="Building UI components"
              />

              <AgentCard
                name="Reviewer"
                role="Quality"
                isActive={activeAgents.reviewer}
                color="amber"
                message="Checking code style"
              />

              <AgentCard
                name="Integrator"
                role="Deploy"
                isActive={activeAgents.integrator}
                color="rose"
                message="Merging branches"
              />
            </div>
          </TabsContent>

          <TabsContent value="chat" className="flex-1 overflow-y-auto px-3 pb-3">
            <div className="space-y-2 mt-2">
              <ChatMessage
                agent="Planner"
                color="blue"
                message="I've created the system architecture. We need user auth, product catalog, and checkout."
                time="2m ago"
              />

              <ChatMessage
                agent="Coder-1"
                color="green"
                message="I'll handle the backend API. @Coder-2 can you work on the UI components?"
                time="1m ago"
              />

              <ChatMessage
                agent="Coder-2"
                color="purple"
                message="Starting with the product card component now."
                time="45s ago"
              />

              <ChatMessage
                agent="Reviewer"
                color="amber"
                message="@Coder-1 The auth schema looks good, but add indexes for performance."
                time="30s ago"
              />

              <ChatMessage
                agent="Coder-1"
                color="green"
                message="Good catch! Updated the schema with proper indexing."
                time="15s ago"
              />

              <ChatMessage
                agent="Integrator"
                color="rose"
                message="Ready to merge the auth module once tests pass."
                time="5s ago"
              />
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* Footer */}
      <div className="p-2 border-t border-zinc-800 text-center">
        <p className="text-xs text-zinc-500">Multi-Agent Coding</p>
      </div>
    </div>
  )
}

// Compact Agent Card Component
function AgentCard({
  name,
  role,
  isActive,
  color,
  message,
}: {
  name: string
  role: string
  isActive: boolean
  color: string
  message: string
}) {
  const colorClasses = {
    blue: "bg-blue-500/20 text-blue-400",
    green: "bg-green-500/20 text-green-400",
    purple: "bg-purple-500/20 text-purple-400",
    amber: "bg-amber-500/20 text-amber-400",
    rose: "bg-rose-500/20 text-rose-400",
  }

  const borderClasses = {
    blue: "border-blue-500/30",
    green: "border-green-500/30",
    purple: "border-purple-500/30",
    amber: "border-amber-500/30",
    rose: "border-rose-500/30",
  }

  return (
    <div
      className={`rounded-md border ${isActive ? borderClasses[color as keyof typeof borderClasses] : "border-zinc-800"} p-2 transition-all bg-zinc-900/50`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`p-1 rounded ${colorClasses[color as keyof typeof colorClasses]}`}>
            <Sparkles className="h-3 w-3" />
          </div>
          <div>
            <h3 className="text-xs font-medium">{name}</h3>
            <p className="text-xs text-zinc-500">{role}</p>
          </div>
        </div>
        <div className="flex items-center">
          {isActive ? (
            <div className="flex items-center gap-1">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
            </div>
          ) : (
            <span className="h-2 w-2 rounded-full bg-zinc-600"></span>
          )}
        </div>
      </div>

      {isActive && (
        <div className="mt-2">
          <p className="text-xs text-zinc-400 leading-relaxed">{message}</p>
        </div>
      )}
    </div>
  )
}

// Compact Chat Message Component
function ChatMessage({
  agent,
  color,
  message,
  time,
}: {
  agent: string
  color: string
  message: string
  time: string
}) {
  const colorClasses = {
    blue: "bg-blue-500/20 text-blue-400",
    green: "bg-green-500/20 text-green-400",
    purple: "bg-purple-500/20 text-purple-400",
    amber: "bg-amber-500/20 text-amber-400",
    rose: "bg-rose-500/20 text-rose-400",
  }

  return (
    <div className="bg-zinc-900/50 rounded-md p-2">
      <div className="flex items-center gap-2 mb-1">
        <div className={`p-1 rounded ${colorClasses[color as keyof typeof colorClasses]}`}>
          <Sparkles className="h-3 w-3" />
        </div>
        <span className={`text-xs font-medium text-${color}-400`}>{agent}</span>
        <span className="text-xs text-zinc-500 ml-auto">{time}</span>
      </div>
      <p className="text-xs text-zinc-300 leading-relaxed pl-6">{message}</p>
    </div>
  )
}
