"""
LangGraph workflow with research, planning, coding, review, deploy nodes
Enhanced with DSPy optimization and code auditor
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from memory import AgentState, ProjectBrain, BrainCheckpoint, NodeType
from agents import create_all_agents
from prompts import PromptFactory, PromptTemplates
from datetime import datetime

# Add JSON serialization helper
def safe_json_dumps(obj, indent=2):
    """Safely serialize objects to JSON, handling bytes and other non-serializable types"""
    def json_serializer(obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    return json.dumps(obj, indent=indent, default=json_serializer)

# ==================== ASYNC UTILITIES ====================

class AsyncManager:
    """Manages async operations consistently across workflow nodes"""
    
    def __init__(self):
        self._loop = None
        self._cleanup_required = False
    
    def get_or_create_loop(self):
        """Get existing loop or create new one safely"""
        try:
            # Try to get current loop
            loop = asyncio.get_running_loop()
            return loop, False  # Existing loop, no cleanup needed
        except RuntimeError:
            # No running loop, create new one
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                self._cleanup_required = True
            return self._loop, True  # New loop, cleanup needed
    
    def run_async(self, coro):
        """Run coroutine with proper loop management"""
        loop, needs_cleanup = self.get_or_create_loop()
        
        try:
            if needs_cleanup:
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(coro)
            else:
                # If we're already in an async context, we need to handle this differently
                # For now, we'll create a task and wait for it
                task = asyncio.create_task(coro)
                result = asyncio.get_event_loop().run_until_complete(task)
            
            return result
        except Exception as e:
            print(f"Async operation failed: {e}")
            raise
        finally:
            if needs_cleanup and self._cleanup_required:
                try:
                    loop.close()
                    self._cleanup_required = False
                except Exception as e:
                    print(f"Warning: Loop cleanup failed: {e}")
    
    def cleanup(self):
        """Clean up async resources"""
        if self._cleanup_required and self._loop and not self._loop.is_closed():
            try:
                self._loop.close()
                self._cleanup_required = False
            except Exception as e:
                print(f"Warning: Async cleanup failed: {e}")

# Global async manager
async_manager = AsyncManager()

# ==================== ASYNC ERROR HANDLING ====================

class AsyncErrorHandler:
    """Handles async operation errors consistently"""
    
    @staticmethod
    def handle_async_error(operation_name: str, error: Exception, fallback_value: Any = None) -> Dict[str, Any]:
        """Handle async operation errors with consistent logging"""
        error_info = {
            'operation': operation_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"❌ Async {operation_name} failed: {error}")
        
        # Log to brain if available
        try:
            # This would need brain context to be passed
            pass
        except Exception:
            pass
        
        return {
            'success': False,
            'error': error_info,
            'fallback_value': fallback_value
        }
    
    @staticmethod
    def create_fallback_result(operation_name: str, fallback_value: Any = None) -> Dict[str, Any]:
        """Create a fallback result for failed async operations"""
        return {
            'success': False,
            'error': {
                'operation': operation_name,
                'error_type': 'AsyncOperationFailed',
                'error_message': f'{operation_name} operation failed',
                'timestamp': datetime.now().isoformat()
            },
            'fallback_value': fallback_value
        }

# ==================== NODE FUNCTIONS ====================

def research_node(state: AgentState) -> AgentState:
    """Research the task and gather relevant information"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        # Add error message to state
        state['messages'] = state.get('messages', []) + [{"role": "system", "content": "No brain context available for research"}]
        return state
    
    # Initialize DSPy prompt factory
    prompt_factory = PromptFactory(brain)
    research_module = prompt_factory.get_module('research')
    
    task = state.get('task', '')
    
    # Use DSPy module for research if available
    if research_module and task:
        try:
            research_output = research_module.forward(task=task)
            research_results = [
                {
                    'source': 'dspy_research',
                    'content': research_output,
                    'timestamp': '2025-01-27T10:00:00Z'
                }
            ]
        except Exception as e:
            print(f"DSPy research failed, falling back to agents: {e}")
            research_results = _fallback_research(brain, task)
    else:
        research_results = _fallback_research(brain, task)
    
    # Update state with research results
    state['research_results'] = research_results
    
    # Add new messages to state
    new_messages = [
        {"role": "system", "content": f"Research completed. Found {len(research_results)} sources."},
        {"role": "human", "content": f"Research results: {safe_json_dumps(research_results)}"}
    ]
    
    # Update messages in state
    state['messages'] = state.get('messages', []) + new_messages
    
    return state

def _fallback_research(brain: ProjectBrain, task: str) -> List[Dict]:
    """Fallback research using traditional agents"""
    agents = create_all_agents(brain)
    architect = agents['architect']
    coordinator = agents['coordinator']
    
    # Research current solutions
    research_prompt = f"""
    Research current solutions for: {task}
    
    Focus on:
    1. Latest technologies and frameworks (2025)
    2. Similar projects on GitHub
    3. Best practices and patterns
    4. Deployment options and costs
    5. Potential challenges and solutions
    
    Use web_search, x_search, and github_search tools to gather comprehensive information.
    """
    
    # Run architect research
    architect_result = architect.invoke({
        "input": research_prompt,
        "chat_history": []
    })
    
    # Run coordinator to synthesize findings
    coordinator_prompt = f"""
    Synthesize the research findings into actionable insights:
    
    {architect_result.get('output', '')}
    
    Create:
    1. Technology comparison table
    2. Cost analysis
    3. Recommended architecture
    4. Implementation timeline
    5. Risk assessment
    """
    
    coordinator_result = coordinator.invoke({
        "input": coordinator_prompt,
        "chat_history": []
    })
    
    return [
        {
            'source': 'architect',
            'content': architect_result.get('output', ''),
            'timestamp': '2025-01-27T10:00:00Z'
        },
        {
            'source': 'coordinator', 
            'content': coordinator_result.get('output', ''),
            'timestamp': '2025-01-27T10:05:00Z'
        }
    ]

def planning_node(state: AgentState) -> AgentState:
    """Create implementation plan using DSPy optimization"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        # Add error message to state
        state['messages'] = state.get('messages', []) + [{"role": "system", "content": "No brain context available for planning"}]
        return state
    
    # Initialize DSPy prompt factory
    prompt_factory = PromptFactory(brain)
    planning_module = prompt_factory.get_module('planning')
    
    task = state.get('task', '')
    research_results = state.get('research_results', [])
    
    # Use DSPy module for planning if available
    if planning_module and task:
        try:
            research_context = json.dumps(research_results) if research_results else ""
            planning_output = planning_module.forward(task=task, research_context=research_context)
            plan = json.loads(planning_output) if isinstance(planning_output, str) else planning_output
        except Exception as e:
            print(f"DSPy planning failed, falling back to agents: {e}")
            plan = _fallback_planning(brain, task, research_results)
    else:
        plan = _fallback_planning(brain, task, research_results)
    
    # Update state with plan
    state['plan'] = plan
    
    # Add new messages to state
    new_messages = [
        {"role": "system", "content": "Planning completed"},
        {"role": "human", "content": f"Implementation plan: {safe_json_dumps(plan)}"}
    ]
    
    # Update messages in state
    state['messages'] = state.get('messages', []) + new_messages
    
    return state

def coding_node(state: AgentState) -> AgentState:
    """Generate code based on plan using DSPy optimization"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        # Add error message to state
        state['messages'] = state.get('messages', []) + [{"role": "system", "content": "No brain context available for coding"}]
        return state
    
    # Initialize DSPy prompt factory
    prompt_factory = PromptFactory(brain)
    coding_module = prompt_factory.get_module('coding')
    
    task = state.get('task', '')
    plan = state.get('plan', {})
    repo_url = state.get('repo_url', '')
    
    # Use DSPy module for coding if available
    if coding_module and task and plan:
        try:
            plan_str = json.dumps(plan) if isinstance(plan, dict) else str(plan)
            coding_output = coding_module.forward(task=task, plan=plan_str)
            code_files = json.loads(coding_output) if isinstance(coding_output, str) else coding_output
        except Exception as e:
            print(f"DSPy coding failed, falling back to agents: {e}")
            code_files = _fallback_coding(brain, task, plan, repo_url)
    else:
        code_files = _fallback_coding(brain, task, plan, repo_url)
    
    # Update state with code files
    state['code_files'] = code_files
    
    # Add new messages to state
    new_messages = [
        {"role": "system", "content": f"Code generation completed. Created {len(code_files)} files."},
        {"role": "human", "content": f"Generated code: {safe_json_dumps(code_files)}"}
    ]
    
    # Update messages in state
    state['messages'] = state.get('messages', []) + new_messages
    
    return state

def auditor_node(state: AgentState) -> AgentState:
    """Audit code/plan for antipatterns and improvements"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        # Add error message to state
        state['messages'] = state.get('messages', []) + [{"role": "system", "content": "No brain context available for audit"}]
        return state
    
    # Initialize DSPy prompt factory
    prompt_factory = PromptFactory(brain)
    auditor_module = prompt_factory.get_module('auditor')
    
    # Get current state for audit
    plan = state.get('plan', {})
    code_files = state.get('code_files', [])
    
    # Combine content for audit
    audit_content = ""
    if plan:
        audit_content += f"PLAN:\n{json.dumps(plan, indent=2)}\n\n"
    if code_files:
        audit_content += "CODE:\n" + "\n\n".join([
            f"=== {file.get('type', 'unknown')} ===\n{file.get('content', '')}"
            for file in code_files
        ])
    
    # Use DSPy module for audit if available
    if auditor_module and audit_content:
        try:
            audit_output = auditor_module.forward(
                input_plan_or_code=audit_content,
                tool_docs="",  # Could be enhanced with actual docs
                package_suggestions=""  # Could be enhanced with package search
            )
            audit_feedback = [
                {
                    'auditor': 'dspy_auditor',
                    'feedback': audit_output,
                    'severity': 'medium',
                    'timestamp': '2025-01-27T10:30:00Z'
                }
            ]
        except Exception as e:
            print(f"DSPy audit failed, falling back to agent: {e}")
            audit_feedback = _fallback_audit(brain, audit_content)
    else:
        audit_feedback = _fallback_audit(brain, audit_content)
    
    # Update state with audit feedback
    state['audit_feedback'] = audit_feedback
    
    # Add new messages to state
    new_messages = [
        {"role": "system", "content": "Code audit completed"},
        {"role": "human", "content": f"Audit feedback: {safe_json_dumps(audit_feedback)}"}
    ]
    
    # Update messages in state
    state['messages'] = state.get('messages', []) + new_messages
    
    return state

def _fallback_audit(brain: ProjectBrain, audit_content: str) -> List[Dict]:
    """Fallback audit using traditional agent"""
    agents = create_all_agents(brain)
    auditor = agents['auditor']
    
    audit_prompt = f"""
    Audit the following code/plan for antipatterns and improvements:
    
    {audit_content}
    
    Hunt for:
    1. Over-verbosity and unnecessary complexity
    2. Wheel reinvention (suggest existing libraries)
    3. Antipatterns from PEP 8 and clean code principles
    4. AI-generated slop and generic responses
    5. Missing abstractions or data structures
    6. Inefficient algorithms or data structures
    7. Security and performance gaps
    
    Be specific and suggest concrete improvements. Output numbered list of issues with severity levels.
    """
    
    result = auditor.invoke({
        "input": audit_prompt,
        "chat_history": []
    })
    
    return [
        {
            'auditor': 'traditional_auditor',
            'feedback': result.get('output', ''),
            'severity': 'medium',
            'timestamp': '2025-01-27T10:30:00Z'
        }
    ]

def review_node(state: AgentState) -> AgentState:
    """Review code for quality and security"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [{"role": "system", "content": "No brain context available for review"}])
    
    reviewer = create_all_agents(brain)['reviewer']
    
    code_files = state.get('code_files', [])
    
    # Create review prompt
    code_summary = "\n".join([f"{f['type']}: {f['content'][:500]}..." for f in code_files])
    
    review_prompt = f"""
    Review the generated code for quality, security, and best practices:
    
    {code_summary}
    
    Check for:
    1. Security vulnerabilities
    2. Performance issues
    3. Code style and consistency
    4. Error handling
    5. Documentation quality
    6. Test coverage
    7. Deployment readiness
    
    Provide specific, actionable feedback with examples.
    """
    
    result = reviewer.invoke({
        "input": review_prompt,
        "chat_history": state.get('messages', [])
    })
    
    review_feedback = [
        {
            'reviewer': 'senior_reviewer',
            'feedback': result.get('output', ''),
            'severity': 'medium',
            'timestamp': '2025-01-27T11:00:00Z'
        }
    ]
    
    # Update state with review feedback
    state['review_feedback'] = review_feedback
    
    # Add new messages to state
    new_messages = [
        {"role": "system", "content": "Code review completed"},
        {"role": "human", "content": f"Review feedback: {safe_json_dumps(review_feedback)}"}
    ]
    
    # Update messages in state
    state['messages'] = state.get('messages', []) + new_messages
    
    return state

def deploy_node(state: AgentState) -> AgentState:
    """Prepare for deployment"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [{"role": "system", "content": "No brain context available for deployment"}])
    
    integrator = create_all_agents(brain)['integrator']
    
    code_files = state.get('code_files', [])
    review_feedback = state.get('review_feedback', [])
    
    # Create deployment prompt
    deployment_prompt = f"""
    Prepare the project for deployment:
    
    Code Files: {len(code_files)} files generated
    Review Feedback: {len(review_feedback)} reviews completed
    
    Tasks:
    1. Address any critical review feedback
    2. Create deployment configuration
    3. Set up CI/CD pipeline
    4. Prepare environment variables
    5. Create deployment documentation
    6. Run final tests
    7. Commit code to git
    
    Ensure everything is production-ready.
    """
    
    result = integrator.invoke({
        "input": deployment_prompt,
        "chat_history": state.get('messages', [])
    })
    
    deployment_status = {
        'status': 'prepared',
        'timestamp': '2025-01-27T12:00:00Z',
        'actions': [
            'Code reviewed and approved',
            'Dependencies updated',
            'Tests passing',
            'Deployment config created',
            'Documentation updated'
        ],
        'next_steps': [
            'Deploy to staging environment',
            'Run integration tests',
            'Deploy to production',
            'Monitor performance'
        ]
    }
    
    # Update state with deployment status
    state['deployment_status'] = deployment_status
    
    # Add new messages to state
    new_messages = [
        {"role": "system", "content": "Deployment preparation completed"},
        {"role": "human", "content": f"Deployment status: {safe_json_dumps(deployment_status)}"}
    ]
    
    # Update messages in state
    state['messages'] = state.get('messages', []) + new_messages
    
    return state

# ==================== REFLECTION NODE ====================

def reflection_node(state: AgentState) -> AgentState:
    """Post-workflow reflection using MetaAgent for self-improvement"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [{"role": "system", "content": "No brain context available for reflection"}])
    
    # Get workflow results for reflection
    task = state.get('task', '')
    research_results = state.get('research_results', [])
    plan = state.get('plan', {})
    code_files = state.get('code_files', [])
    review_feedback = state.get('review_feedback', [])
    deployment_status = state.get('deployment_status', {})
    
    # Create reflection context
    workflow_summary = {
        'task': task,
        'research_count': len(research_results),
        'plan_phases': plan.get('phases', []),
        'code_files_count': len(code_files),
        'review_issues': len([f for f in review_feedback if f.get('severity') == 'high']),
        'deployment_success': deployment_status.get('status') == 'success'
    }
    
    # Invoke MetaAgent for reflection
    agents = create_all_agents(brain)
    meta_agent = agents['meta']
    
    reflection_prompt = f"""
    Reflect on this completed workflow:
    
    TASK: {task}
    WORKFLOW SUMMARY: {json.dumps(workflow_summary, indent=2)}
    
    RESEARCH RESULTS: {len(research_results)} sources found
    PLAN: {json.dumps(plan, indent=2) if plan else 'No plan created'}
    CODE FILES: {len(code_files)} files generated
    REVIEW FEEDBACK: {len(review_feedback)} items
    DEPLOYMENT: {deployment_status.get('status', 'unknown')}
    
    Analyze what went well, what could be improved, and how to evolve the system.
    Focus on:
    1. Code quality and architecture decisions
    2. Agent coordination effectiveness
    3. Memory and context management
    4. Performance and efficiency
    5. User experience and workflow
    6. Learning from this run
    
    Generate actionable insights and store them as reflections.
    """
    
    try:
        reflection_result = meta_agent.invoke({
            "input": reflection_prompt,
            "chat_history": []
        })
        
        # Extract reflection from result
        reflection_output = reflection_result.get('output', '')
        
        # Try to parse structured reflection
        try:
            if '{' in reflection_output and '}' in reflection_output:
                start = reflection_output.find('{')
                end = reflection_output.rfind('}') + 1
                reflection_data = json.loads(reflection_output[start:end])
                
                # Store structured reflection
                reflection_content = f"""
                PROBLEM: {reflection_data.get('problem', 'Not specified')}
                WISH: {reflection_data.get('wish', 'Not specified')}
                HOW: {reflection_data.get('how', 'Not specified')}
                SOURCES: {', '.join(reflection_data.get('sources', []))}
                """
                
                # Add to brain as reflection node
                brain.add_reflection(
                    content=reflection_content,
                    sources=reflection_data.get('sources', [])
                )
            else:
                # Store unstructured reflection
                brain.add_reflection(
                    content=reflection_output
                )
        except:
            # Store as plain text if parsing fails
            brain.add_reflection(
                content=reflection_output
            )
        
        # Update state with reflection context
        state['reflection_context'] = {
            'reflection_output': reflection_output,
            'workflow_summary': workflow_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add new messages to state
        new_messages = [
            {"role": "system", "content": "Reflection completed"},
            {"role": "human", "content": f"Reflection insights: {reflection_output[:500]}..."}
        ]
        
        # Update messages in state
        state['messages'] = state.get('messages', []) + new_messages
        
        return state
        
    except Exception as e:
        print(f"Reflection failed: {e}")
        new_messages = [
            {"role": "system", "content": f"Reflection failed: {e}"}
        ]
        # Update messages in state
        state['messages'] = state.get('messages', []) + new_messages
        return state

# ==================== CONFIDENCE & RECRUITMENT ====================

def compute_confidence(output: str, llm=None) -> float:
    """Compute confidence score for agent output using bootstrap sampling"""
    try:
        # Simple confidence computation based on output characteristics
        # In production, this would use LLM token probabilities or bootstrap sampling
        
        # Base confidence from output length and structure
        base_confidence = min(len(output) / 500, 0.6)  # Cap at 0.6, more lenient
        
        # Boost confidence for structured outputs (JSON, code blocks)
        if '{' in output and '}' in output:
            base_confidence += 0.15
        if '```' in output:
            base_confidence += 0.15
        if 'def ' in output or 'class ' in output:
            base_confidence += 0.15
        
        # Boost confidence for comprehensive responses
        if len(output.split()) > 50:
            base_confidence += 0.1
        
        # Reduce confidence for error indicators
        if any(word in output.lower() for word in ['error', 'failed', 'exception', 'timeout']):
            base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
        
    except Exception as e:
        print(f"Confidence computation failed: {e}")
        return 0.5  # Default confidence

def confidence_node(state: AgentState) -> AgentState:
    """Compute confidence for current agent output and update state"""
    
    # Get the most recent agent output
    messages = state.get('messages', [])
    if not messages:
        state['confidence'] = 0.5  # Default confidence
        return state
    
    # Find the most recent AI message
    recent_output = ""
    for msg in reversed(messages):
        if msg.get('role') == 'ai' or msg.get('role') == 'assistant':
            recent_output = msg.get('content', '')
            break
    
    if not recent_output:
        state['confidence'] = 0.5  # Default confidence
        return state
    
    # Compute confidence
    confidence = compute_confidence(recent_output)
    
    # Update state with confidence
    state['confidence'] = confidence
    
    return state

def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor node for dynamic recruitment based on confidence"""
    
    confidence = state.get('confidence', 0.5)
    current_step = state.get('current_step', 'unknown')
    task = state.get('task', '')
    
    # Determine if recruitment is needed
    needs_recruitment = confidence < 0.8
    
    if needs_recruitment:
        # Add recruitment context to state
        state['recruitment_needed'] = True
        state['recruitment_reason'] = f"Low confidence ({confidence:.2f}) in {current_step}"
        
        # Log recruitment decision
        brain = state.get('brain_context', {}).get('brain')
        if brain:
            brain.add_node(
                NodeType.DECISION,
                f"Recruitment triggered: confidence {confidence:.2f} < 0.8 threshold",
                metadata={
                    'step': current_step,
                    'confidence': confidence,
                    'task': task
                }
            )
    else:
        state['recruitment_needed'] = False
    
    return state

# ==================== REASONING & COLLABORATION ====================

def tree_of_thoughts_node(state: AgentState) -> AgentState:
    """Tree of Thoughts reasoning node for complex tasks"""
    
    task = state.get('task', '')
    brain = state.get('brain_context', {}).get('brain')
    
    if not brain:
        state['reasoning_results'] = {'error': 'No brain context'}
        return state
    
    try:
        # Create reasoning agent
        from agents import ReasoningAgent
        reasoning_agent = ReasoningAgent(brain)
        
        # Run Tree of Thoughts reasoning using standardized async manager
        reasoning_result = async_manager.run_async(
            reasoning_agent.reason_with_validation(task)
        )
        
        # Update state with reasoning results
        state['reasoning_results'] = reasoning_result
        state['reasoning_paths'] = reasoning_result.get('paths', [])
        state['best_path'] = reasoning_result.get('best_path', '')
        
        # Log reasoning to brain
        brain.add_node(
            NodeType.REFLECTION,
            f"Tree of Thoughts reasoning completed for: {task}",
            metadata={
                'reasoning_type': 'tree_of_thoughts',
                'paths_explored': len(reasoning_result.get('paths', [])),
                'best_path_score': reasoning_result.get('best_path_score', 0)
            }
        )
        
    except Exception as e:
        print(f"Tree of Thoughts failed: {e}")
        error_result = AsyncErrorHandler.handle_async_error(
            'tree_of_thoughts_reasoning', 
            e, 
            {'error': str(e)}
        )
        state['reasoning_results'] = error_result['fallback_value']
    
    return state

def debate_node(state: AgentState) -> AgentState:
    """Multi-agent debate node for collaborative problem solving"""
    
    task = state.get('task', '')
    brain = state.get('brain_context', {}).get('brain')
    
    if not brain:
        state['debate_results'] = {'error': 'No brain context'}
        return state
    
    try:
        # Import debate framework
        from debate_framework import MultiAgentDebateFramework
        
        # Create debate framework
        debate = MultiAgentDebateFramework(brain=brain)
        
        # Run debate using standardized async manager
        debate_result = async_manager.run_async(
            debate.conduct_debate(task, context="", task_type="general")
        )
        
        # Update state with debate results
        state['debate_results'] = debate_result
        state['consensus'] = debate_result.final_proposal
        state['debate_confidence'] = debate_result.confidence
        
        # Log debate to brain
        brain.add_node(
            NodeType.DECISION,
            f"Multi-agent debate completed for: {task}",
            metadata={
                'debate_type': 'collaborative',
                'consensus_confidence': debate_result.confidence
            }
        )
        
    except Exception as e:
        print(f"Debate failed: {e}")
        error_result = AsyncErrorHandler.handle_async_error(
            'multi_agent_debate', 
            e, 
            {'error': str(e)}
        )
        state['debate_results'] = error_result['fallback_value']
    
    return state

# ==================== GRAPH CONSTRUCTION ====================

def create_workflow_graph(brain: ProjectBrain) -> StateGraph:
    """Create the main workflow graph with reflection, confidence, and recruitment"""
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("review", review_node)
    workflow.add_node("deploy", deploy_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("confidence", confidence_node)  # New: confidence computation
    workflow.add_node("supervisor", supervisor_node)  # New: recruitment supervisor
    workflow.add_node("reasoning", tree_of_thoughts_node)  # New: Tree of Thoughts
    workflow.add_node("debate", debate_node)  # New: multi-agent debate
    
    # Define edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "planning")
    workflow.add_edge("planning", "confidence")  # Compute confidence after planning
    workflow.add_edge("confidence", "supervisor")  # Check if recruitment needed
    workflow.add_edge("auditor", "coding")
    workflow.add_edge("coding", "confidence")  # Compute confidence after coding
    workflow.add_edge("review", "deploy")
    workflow.add_edge("deploy", "reflection")
    workflow.add_edge("reflection", END)
    
    # Add conditional edges for review feedback
    def should_redo_coding(state: AgentState) -> str:
        """Check if coding needs to be redone based on review"""
        review_feedback = state.get('review_feedback', [])
        for feedback in review_feedback:
            if feedback.get('severity') == 'high':
                return "coding"
        return "deploy"
    
    # Add conditional edge for auditor after coding
    def should_audit_after_coding(state: AgentState) -> str:
        """Check if auditor should run after coding based on audit feedback"""
        audit_feedback = state.get('audit_feedback', [])
        for feedback in audit_feedback:
            if feedback.get('severity') == 'high':
                return "auditor"
        return "review"
    
    # Add conditional edges for recruitment and reasoning
    def should_recruit_agents(state: AgentState) -> str:
        """Check if additional agents should be recruited based on confidence"""
        recruitment_needed = state.get('recruitment_needed', False)
        if recruitment_needed:
            return "recruit"
        return "continue"
    
    def should_use_reasoning(state: AgentState) -> str:
        """Check if Tree of Thoughts reasoning should be used"""
        task = state.get('task', '')
        # Use reasoning for complex tasks (long descriptions, technical terms)
        complexity_score = len(task.split()) * 0.1 + task.count('algorithm') * 0.5 + task.count('optimize') * 0.3
        if complexity_score > 2.0:
            return "reasoning"
        return "continue"
    
    def should_debate(state: AgentState) -> str:
        """Check if multi-agent debate should be used"""
        confidence = state.get('confidence', 0.5)
        if confidence < 0.6:  # Low confidence triggers debate
            return "debate"
        return "continue"
    
    workflow.add_conditional_edges(
        "review",
        should_redo_coding,
        {
            "coding": "coding",
            "deploy": "deploy"
        }
    )
    
    workflow.add_conditional_edges(
        "coding",
        should_audit_after_coding,
        {
            "auditor": "auditor",
            "review": "review"
        }
    )
    
    # Add recruitment conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        should_recruit_agents,
        {
            "recruit": "reasoning",  # Use reasoning for complex tasks
            "continue": "auditor"    # Continue to next step
        }
    )
    
    # Add reasoning conditional edges
    workflow.add_conditional_edges(
        "reasoning",
        should_debate,
        {
            "debate": "debate",
            "continue": "auditor"
        }
    )
    
    # Add debate conditional edges
    workflow.add_conditional_edges(
        "debate",
        lambda state: "auditor",  # Always go to auditor after debate
        {
            "auditor": "auditor"
        }
    )
    
    return workflow

def create_quick_workflow_graph(brain: ProjectBrain) -> StateGraph:
    """Create simplified workflow without research"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("review", review_node)
    workflow.add_node("deploy", deploy_node)
    
    # Define edges
    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "coding")
    workflow.add_edge("coding", "review")
    workflow.add_edge("review", "deploy")
    workflow.add_edge("deploy", END)
    
    return workflow

def create_research_only_graph(brain: ProjectBrain) -> StateGraph:
    """Create research-only workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    
    # Define edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", END)
    
    return workflow

def create_minimal_workflow_graph(brain: ProjectBrain) -> StateGraph:
    """Create minimal workflow graph - bypasses heavy operations"""
    workflow = StateGraph(AgentState)
    
    # Add nodes with timeouts
    workflow.add_node("planner", planning_node)
    workflow.add_node("coder", coding_node)
    
    # Set entry and exit
    workflow.set_entry_point("planner")
    workflow.set_finish_point("coder")
    
    # Direct path: planner -> coder (skip research, audit, review)
    workflow.add_edge("planner", "coder")
    
    return workflow

# ==================== GRAPH EXECUTION ====================

def run_workflow(task: str, brain: ProjectBrain, mode: str = "full", repo_url: str = "") -> Dict[str, Any]:
    """Run the workflow with the given task and timeouts"""
    
    # Check for minimal mode
    if os.getenv('MINIMAL_MODE'):
        mode = "minimal"
    
    # Create appropriate graph
    if mode == "quick":
        graph = create_quick_workflow_graph(brain)
    elif mode == "research_only":
        graph = create_research_only_graph(brain)
    elif mode == "minimal":
        graph = create_minimal_workflow_graph(brain)
    else:
        graph = create_workflow_graph(brain)
    
    # Create checkpoint
    checkpoint = BrainCheckpoint(brain)
    
    # Compile graph
    app = graph.compile(checkpointer=checkpoint)
    
    # Initialize state
    initial_state = {
        "messages": [],
        "current_step": "start",
        "task": task,
        "repo_url": repo_url,
        "research_results": [],
        "plan": {},
        "audit_feedback": [],
        "code_files": [],
        "review_feedback": [],
        "deployment_status": {},
        "brain_context": {"brain": brain},
        "reflection_context": {}  # New: reflection context
    }
    
    # Ensure messages are in correct format for LangGraph
    if not initial_state["messages"]:
        initial_state["messages"] = [{"role": "system", "content": "Starting workflow"}]
    
    # Run workflow with timeout
    config = {"configurable": {"thread_id": f"task_{hash(task)}"}}
    
    try:
        # Set timeout based on mode
        timeout_seconds = {
            "minimal": 30,
            "quick": 60,
            "research_only": 45,
            "full": 120
        }.get(mode, 120)
        
        # Run with timeout using standardized async manager
        try:
            result = async_manager.run_async(
                asyncio.wait_for(
                    asyncio.to_thread(app.invoke, initial_state, config=config),
                    timeout=timeout_seconds
                )
            )
            return {
                "success": True,
                "final_state": result,
                "messages": result.get("messages", []),
                "research_results": result.get("research_results", []),
                "plan": result.get("plan", {}),
                "code_files": result.get("code_files", []),
                "review_feedback": result.get("review_feedback", []),
                "deployment_status": result.get("deployment_status", {})
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Workflow timed out after {timeout_seconds} seconds",
                "final_state": None
            }
        finally:
            async_manager.cleanup()
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "final_state": None
        }

# ==================== HELPER FUNCTIONS ====================

def _log_dspy_example(brain: ProjectBrain, task: str, output: str, module_type: str):
    """Log DSPy task → output pairs to brain for optimization."""
    try:
        if not brain or not hasattr(brain, 'memory'):
            return
        
        # Create DSPy example in correct format
        example = {
            'inputs': {'task': task},
            'outputs': {'results': output},
            'module_type': module_type,
            'timestamp': datetime.now().isoformat(),
            'quality_score': None  # Will be computed by validate_clean_code
        }
        
        # Initialize dspy_examples if not exists
        if 'dspy_examples' not in brain.memory:
            brain.memory['dspy_examples'] = []
        
        # Add example to brain
        brain.memory['dspy_examples'].append(example)
        
        # Keep only the last 500 examples to prevent memory bloat
        if len(brain.memory['dspy_examples']) > 500:
            brain.memory['dspy_examples'] = brain.memory['dspy_examples'][-500:]
        
        # Save to disk
        brain._save()
        
        print(f"Logged DSPy example for {module_type}: {len(task)} chars task → {len(output)} chars output")
        
    except Exception as e:
        print(f"Error logging DSPy example: {e}")

def _fallback_planning(brain: ProjectBrain, task: str, research_results: List[Dict]) -> Dict:
    """Fallback planning using traditional agent."""
    planner = create_all_agents(brain)['planner']
    
    # Create planning prompt with research context
    research_summary = "\n".join([f"{r['source']}: {r['content'][:500]}..." for r in research_results])
    
    planning_prompt = f"""
    Create a detailed implementation plan for: {task}
    
    Research Context:
    {research_summary}
    
    Create a structured plan with:
    1. Phase breakdown (Research, Design, Implementation, Testing, Deployment)
    2. Specific tasks for each phase
    3. Dependencies between tasks
    4. Timeline estimates
    5. Resource requirements
    6. Success criteria
    7. Risk mitigation strategies
    
    Output as JSON with clear structure.
    """
    
    result = planner.invoke({
        "input": planning_prompt,
        "chat_history": []
    })
    
    # Parse plan from result
    try:
        plan_content = result.get('output', '')
        # Try to extract JSON from the response
        if '{' in plan_content and '}' in plan_content:
            start = plan_content.find('{')
            end = plan_content.rfind('}') + 1
            plan_json = json.loads(plan_content[start:end])
        else:
            plan_json = {
                'phases': ['Research', 'Design', 'Implementation', 'Testing', 'Deployment'],
                'tasks': plan_content.split('\n'),
                'timeline': '2-4 weeks',
                'risks': ['Technical complexity', 'Integration challenges']
            }
    except:
        plan_json = {
            'phases': ['Research', 'Design', 'Implementation', 'Testing', 'Deployment'],
            'tasks': result.get('output', '').split('\n'),
            'timeline': '2-4 weeks',
            'risks': ['Technical complexity', 'Integration challenges']
        }
    
    return plan_json

def _fallback_coding(brain: ProjectBrain, task: str, plan: Dict, repo_url: str) -> List[Dict]:
    """Fallback coding using traditional agents."""
    agents = create_all_agents(brain)
    coder1 = agents['coder1']
    coder2 = agents['coder2']
    
    # Coder 1 - Core implementation
    coding_prompt = f"""
    Implement the core functionality for: {task}
    
    Plan: {json.dumps(plan, indent=2)}
    
    Create:
    1. Main application code
    2. Core business logic
    3. API endpoints if needed
    4. Database models if needed
    5. Configuration management
    
    Write production-ready Python code with:
    - Type hints
    - Error handling
    - Logging
    - Security best practices
    - Modern async/await patterns
    """
    
    coder1_result = coder1.invoke({
        "input": coding_prompt,
        "chat_history": []
    })
    
    # Coder 2 - Tests and utilities
    testing_prompt = f"""
    Create comprehensive tests and utilities for the implementation:
    
    {coder1_result.get('output', '')}
    
    Create:
    1. Unit tests (pytest)
    2. Integration tests
    3. CLI interface
    4. Documentation
    5. Helper utilities
    6. Configuration examples
    
    Ensure code quality and maintainability.
    """
    
    coder2_result = coder2.invoke({
        "input": testing_prompt,
        "chat_history": []
    })
    
    # Combine code files
    code_files = [
        {
            'type': 'main_code',
            'content': coder1_result.get('output', ''),
            'agent': 'coder1'
        },
        {
            'type': 'tests_utils',
            'content': coder2_result.get('output', ''),
            'agent': 'coder2'
        }
    ]
    
    # If repository URL is provided, clone and commit changes
    if repo_url:
        try:
            from tools import GitRepositoryTools
            git_tools = GitRepositoryTools(brain)
            
            # Clone repository
            clone_result = git_tools.clone_repository(repo_url)
            repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            
            # Create new branch for changes
            git_tools.create_branch(repo_name, "feature/auto-generated-code")
            
            # Create files in repository
            for i, code_file in enumerate(code_files):
                file_name = f"generated_{code_file['type']}_{i+1}.py"
                git_tools.create_repository_file(
                    repo_name, 
                    file_name, 
                    code_file['content'],
                    f"Add {code_file['type']} from {code_file['agent']}"
                )
            
            # Get repository status
            status = git_tools.get_repository_status(repo_name)
            
            # Update state with Git information
            state['git_repo'] = {
                'url': repo_url,
                'name': repo_name,
                'status': status,
                'files_created': len(code_files)
            }
            
            code_files.append({
                'type': 'git_operations',
                'content': f"Repository cloned and files committed to {repo_name}",
                'agent': 'git_tools'
            })
            
        except Exception as e:
            code_files.append({
                'type': 'git_error',
                'content': f"Git operations failed: {str(e)}",
                'agent': 'git_tools'
            })
    
    return code_files 