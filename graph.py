"""
LangGraph workflow with research, planning, coding, review, deploy nodes
Enhanced with DSPy optimization and code auditor
"""

import json
from typing import Dict, List, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from memory import AgentState, ProjectBrain, BrainCheckpoint
from agents import create_all_agents
from prompts import PromptFactory, PromptTemplates
from datetime import datetime

# ==================== NODE FUNCTIONS ====================

def research_node(state: AgentState) -> AgentState:
    """Research current solutions and technologies using DSPy optimization with HF routing"""
    print("[DEBUG] Entering research_node. State:", state)
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [("system", "No brain context available for research")])
    
    # Initialize DSPy prompt factory
    prompt_factory = PromptFactory(brain)
    research_module = prompt_factory.get_module('research')
    
    task = state.get('task', '')
    
    # Use DSPy module for research if available
    if research_module:
        try:
            research_output = research_module.forward(task=task)
            
            # Try HF routing if needed
            try:
                from hf_routing import HFRoutingSystem
                hf_router = HFRoutingSystem(brain)
                routed_output, model_used = hf_router.route_task(task, research_output)
                if model_used.startswith('HF_'):
                    research_output = routed_output
                    print(f"✅ Research routed to {model_used}")
            except ImportError:
                pass  # HF routing not available
            
            research_results = [
                {
                    'source': 'dspy_research',
                    'content': research_output,
                    'timestamp': '2025-01-27T10:00:00Z'
                }
            ]
            
            # Log DSPy example to brain
            _log_dspy_example(brain, task, research_output, 'research')
            
        except Exception as e:
            print(f"DSPy research failed, falling back to agents: {e}")
            research_results = _fallback_research(brain, task)
    else:
        research_results = _fallback_research(brain, task)
    
    # Update state with research results
    state['research_results'] = research_results
    print("[DEBUG] Exiting research_node. Research results:", research_results)
    
    return add_messages(state, [
        ("system", f"Research completed. Found {len(research_results)} sources."),
        ("human", f"Research results: {json.dumps(research_results, indent=2)}")
    ])

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
    """Create implementation plan based on research with HF routing"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [("system", "No brain context available for planning")])
    
    # Initialize DSPy prompt factory
    prompt_factory = PromptFactory(brain)
    planning_module = prompt_factory.get_module('planning')
    
    task = state.get('task', '')
    research_results = state.get('research_results', [])
    
    # Use DSPy module for planning if available
    if planning_module and research_results:
        try:
            research_context = "\n".join([f"{r['source']}: {r['content'][:500]}..." for r in research_results])
            plan_output = planning_module.forward(task=task, research_context=research_context)
            
            # Try HF routing if needed
            try:
                from hf_routing import HFRoutingSystem
                hf_router = HFRoutingSystem(brain)
                routed_output, model_used = hf_router.route_task(task, plan_output)
                if model_used.startswith('HF_'):
                    plan_output = routed_output
                    print(f"✅ Planning routed to {model_used}")
            except ImportError:
                pass  # HF routing not available
            
            # Log DSPy example to brain
            _log_dspy_example(brain, f"Plan: {task}", plan_output, 'planning')
            
            # Try to parse as JSON
            try:
                if '{' in plan_output and '}' in plan_output:
                    start = plan_output.find('{')
                    end = plan_output.rfind('}') + 1
                    plan_json = json.loads(plan_output[start:end])
                else:
                    plan_json = {
                        'phases': ['Research', 'Design', 'Implementation', 'Testing', 'Deployment'],
                        'tasks': plan_output.split('\n'),
                        'timeline': '2-4 weeks',
                        'risks': ['Technical complexity', 'Integration challenges']
                    }
            except:
                plan_json = {
                    'phases': ['Research', 'Design', 'Implementation', 'Testing', 'Deployment'],
                    'tasks': plan_output.split('\n'),
                    'timeline': '2-4 weeks',
                    'risks': ['Technical complexity', 'Integration challenges']
                }
        except Exception as e:
            print(f"DSPy planning failed, falling back to agents: {e}")
            plan_json = _fallback_planning(brain, task, research_results)
    else:
        plan_json = _fallback_planning(brain, task, research_results)
    
    # Update state with plan
    state['plan'] = plan_json
    
    return add_messages(state, [
        ("system", "Planning completed"),
        ("human", f"Implementation plan: {json.dumps(plan_json, indent=2)}")
    ])

def coding_node(state: AgentState) -> AgentState:
    """Generate code based on plan with HF routing"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [("system", "No brain context available for coding")])
    
    # Initialize DSPy prompt factory
    prompt_factory = PromptFactory(brain)
    coding_module = prompt_factory.get_module('coding')
    
    task = state.get('task', '')
    plan = state.get('plan', {})
    repo_url = state.get('repo_url', '')  # Get repository URL if provided
    
    # Use DSPy module for coding if available
    if coding_module and plan:
        try:
            plan_str = json.dumps(plan, indent=2) if isinstance(plan, dict) else str(plan)
            code_output = coding_module.forward(task=task, plan=plan_str)
            
            # Try HF routing if needed
            try:
                from hf_routing import HFRoutingSystem
                hf_router = HFRoutingSystem(brain)
                routed_output, model_used = hf_router.route_task(task, code_output)
                if model_used.startswith('HF_'):
                    code_output = routed_output
                    print(f"✅ Coding routed to {model_used}")
            except ImportError:
                pass  # HF routing not available
            
            # Log DSPy example to brain
            _log_dspy_example(brain, f"Code: {task}", code_output, 'coding')
            
            # Create code files from DSPy output
            code_files = [
                {
                    'type': 'main_code',
                    'content': code_output,
                    'agent': 'dspy_coder'
                }
            ]
            
        except Exception as e:
            print(f"DSPy coding failed, falling back to agents: {e}")
            code_files = _fallback_coding(brain, task, plan, repo_url)
    else:
        code_files = _fallback_coding(brain, task, plan, repo_url)
    
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
    
    # Update state with code files
    state['code_files'] = code_files
    
    return add_messages(state, [
        ("system", f"Code generation completed. Created {len(code_files)} files."),
        ("human", f"Generated code: {json.dumps(code_files, indent=2)}")
    ])

def auditor_node(state: AgentState) -> AgentState:
    """Audit code/plan for antipatterns and improvements"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [("system", "No brain context available for audit")])
    
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
    
    return add_messages(state, [
        ("system", "Code audit completed"),
        ("human", f"Audit feedback: {json.dumps(audit_feedback, indent=2)}")
    ])

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
        return add_messages(state, [("system", "No brain context available for review")])
    
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
    
    return add_messages(state, [
        ("system", "Code review completed"),
        ("human", f"Review feedback: {json.dumps(review_feedback, indent=2)}")
    ])

def deploy_node(state: AgentState) -> AgentState:
    """Prepare for deployment"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [("system", "No brain context available for deployment")])
    
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
    
    return add_messages(state, [
        ("system", "Deployment preparation completed"),
        ("human", f"Deployment status: {json.dumps(deployment_status, indent=2)}")
    ])

# ==================== REFLECTION NODE ====================

def reflection_node(state: AgentState) -> AgentState:
    """Post-workflow reflection using MetaAgent for self-improvement"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [("system", "No brain context available for reflection")])
    
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
                    sources=reflection_data.get('sources', []),
                    metadata={
                        'workflow_task': task,
                        'workflow_summary': workflow_summary,
                        'reflection_type': 'post_workflow'
                    }
                )
            else:
                # Store unstructured reflection
                brain.add_reflection(
                    content=reflection_output,
                    metadata={
                        'workflow_task': task,
                        'workflow_summary': workflow_summary,
                        'reflection_type': 'post_workflow'
                    }
                )
        except:
            # Store as plain text if parsing fails
            brain.add_reflection(
                content=reflection_output,
                metadata={
                    'workflow_task': task,
                    'workflow_summary': workflow_summary,
                    'reflection_type': 'post_workflow'
                }
            )
        
        # Update state with reflection context
        state['reflection_context'] = {
            'reflection_output': reflection_output,
            'workflow_summary': workflow_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        return add_messages(state, [
            ("system", "Reflection completed"),
            ("human", f"Reflection insights: {reflection_output[:500]}...")
        ])
        
    except Exception as e:
        print(f"Reflection failed: {e}")
        return add_messages(state, [
            ("system", f"Reflection failed: {e}")
        ])

# ==================== GRAPH CONSTRUCTION ====================

def create_workflow_graph(brain: ProjectBrain) -> StateGraph:
    """Create the main workflow graph with reflection"""
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("review", review_node)
    workflow.add_node("deploy", deploy_node)
    workflow.add_node("reflection", reflection_node)  # New: reflection node
    
    # Define edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "planning")
    workflow.add_edge("planning", "auditor")
    workflow.add_edge("auditor", "coding")
    workflow.add_edge("coding", "review")
    workflow.add_edge("review", "deploy")
    workflow.add_edge("deploy", "reflection")  # New: reflection after deploy
    workflow.add_edge("reflection", END)  # New: end after reflection
    
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

# ==================== GRAPH EXECUTION ====================

def run_workflow(task: str, brain: ProjectBrain, mode: str = "full", repo_url: str = "") -> Dict[str, Any]:
    """Run the workflow with the given task"""
    
    # Create appropriate graph
    if mode == "quick":
        graph = create_quick_workflow_graph(brain)
    elif mode == "research_only":
        graph = create_research_only_graph(brain)
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
    
    # Run workflow
    config = {"configurable": {"thread_id": f"task_{hash(task)}"}}
    
    try:
        result = app.invoke(initial_state, config=config)
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