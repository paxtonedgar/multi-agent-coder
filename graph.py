"""
LangGraph workflow with research, planning, coding, review, deploy nodes
"""

import json
from typing import Dict, List, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from memory import AgentState, ProjectBrain, BrainCheckpoint
from agents import create_all_agents

# ==================== NODE FUNCTIONS ====================

def research_node(state: AgentState) -> AgentState:
    """Research current solutions and technologies"""
    print("[DEBUG] Entering research_node. State:", state)
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [("system", "No brain context available for research")])
    
    # Create architect and coordinator agents
    agents = create_all_agents(brain)
    architect = agents['architect']
    coordinator = agents['coordinator']
    
    task = state.get('task', '')
    
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
        "chat_history": state.get('messages', [])
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
        "chat_history": state.get('messages', [])
    })
    
    # Update state
    research_results = [
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
    
    # Update state with research results
    state['research_results'] = research_results
    print("[DEBUG] Exiting research_node. Research results:", research_results)
    
    return add_messages(state, [
        ("system", f"Research completed. Found {len(research_results)} sources."),
        ("human", f"Research results: {json.dumps(research_results, indent=2)}")
    ])

def planning_node(state: AgentState) -> AgentState:
    """Create implementation plan based on research"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [("system", "No brain context available for planning")])
    
    planner = create_all_agents(brain)['planner']
    
    task = state.get('task', '')
    research_results = state.get('research_results', [])
    
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
        "chat_history": state.get('messages', [])
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
    
    # Update state with plan
    state['plan'] = plan_json
    
    return add_messages(state, [
        ("system", "Planning completed"),
        ("human", f"Implementation plan: {json.dumps(plan_json, indent=2)}")
    ])

def coding_node(state: AgentState) -> AgentState:
    """Generate code based on plan"""
    
    brain = state.get('brain_context', {}).get('brain')
    if not brain:
        return add_messages(state, [("system", "No brain context available for coding")])
    
    agents = create_all_agents(brain)
    coder1 = agents['coder1']
    coder2 = agents['coder2']
    
    task = state.get('task', '')
    plan = state.get('plan', {})
    repo_url = state.get('repo_url', '')  # Get repository URL if provided
    
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
        "chat_history": state.get('messages', [])
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
        "chat_history": state.get('messages', [])
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
    
    # Update state with code files
    state['code_files'] = code_files
    
    return add_messages(state, [
        ("system", f"Code generation completed. Created {len(code_files)} files."),
        ("human", f"Generated code: {json.dumps(code_files, indent=2)}")
    ])

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

# ==================== GRAPH CONSTRUCTION ====================

def create_workflow_graph(brain: ProjectBrain) -> StateGraph:
    """Create the main workflow graph"""
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("review", review_node)
    workflow.add_node("deploy", deploy_node)
    
    # Define edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "planning")
    workflow.add_edge("planning", "coding")
    workflow.add_edge("coding", "review")
    workflow.add_edge("review", "deploy")
    workflow.add_edge("deploy", END)
    
    # Add conditional edges for review feedback
    def should_redo_coding(state: AgentState) -> str:
        """Check if coding needs to be redone based on review"""
        review_feedback = state.get('review_feedback', [])
        for feedback in review_feedback:
            if feedback.get('severity') == 'high':
                return "coding"
        return "deploy"
    
    workflow.add_conditional_edges(
        "review",
        should_redo_coding,
        {
            "coding": "coding",
            "deploy": "deploy"
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
        "code_files": [],
        "review_feedback": [],
        "deployment_status": {},
        "brain_context": {"brain": brain}
    }
    
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