"""
Lethality Engine for High-Impact Effectiveness
Implements utility-based decision making, risk assessment, and autonomous action capabilities
"""

import os
import json
import asyncio
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

# Custom JSON encoder for Enum types
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

from memory import ProjectBrain, NodeType
from tools import GitRepositoryTools

# ==================== LETHALITY DATA STRUCTURES ====================

class ActionType(Enum):
    """Types of actions the system can take"""
    CODE_GENERATION = "code_generation"
    DEPLOYMENT = "deployment"
    GIT_OPERATION = "git_operation"
    SYSTEM_UPGRADE = "system_upgrade"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_OPERATION = "data_operation"
    NETWORK_OPERATION = "network_operation"

class RiskLevel(Enum):
    """Risk levels for actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ActionImpact:
    """Impact assessment for an action"""
    success_probability: float  # 0-1
    potential_benefit: float    # 0-1
    potential_harm: float       # 0-1
    reversibility: float        # 0-1 (1 = fully reversible)
    scope: str                  # "local", "project", "system", "global"
    affected_components: List[str]
    estimated_duration: int     # minutes

@dataclass
class LethalityDecision:
    """Decision made by the lethality engine"""
    action_type: ActionType
    action_description: str
    utility_score: float
    risk_level: RiskLevel
    impact_assessment: ActionImpact
    approval_required: bool
    human_veto_hooks: List[str]
    contingency_plan: str
    timestamp: str
    decision_id: str

# ==================== UTILITY-BASED DECISION MAKER ====================

class UtilityBasedAgent:
    """Agent that makes decisions based on utility maximization"""
    
    def __init__(self, brain: ProjectBrain, risk_tolerance: float = 0.7):
        self.brain = brain
        self.risk_tolerance = risk_tolerance
        self.decision_history = []
        self.impact_scores = {}
        
    async def evaluate_action(self, action_type: ActionType, action_description: str, 
                            context: Dict[str, Any]) -> LethalityDecision:
        """Evaluate an action using utility-based decision making"""
        
        # Assess impact
        impact = await self._assess_action_impact(action_type, action_description, context)
        
        # Calculate utility score
        utility_score = self._calculate_utility_score(impact)
        
        # Determine risk level
        risk_level = self._determine_risk_level(impact)
        
        # Check if approval is required
        approval_required = self._requires_approval(utility_score, risk_level)
        
        # Generate contingency plan
        contingency_plan = await self._generate_contingency_plan(action_type, impact)
        
        # Create decision
        decision = LethalityDecision(
            action_type=action_type,
            action_description=action_description,
            utility_score=utility_score,
            risk_level=risk_level,
            impact_assessment=impact,
            approval_required=approval_required,
            human_veto_hooks=self._generate_veto_hooks(impact),
            contingency_plan=contingency_plan,
            timestamp=datetime.now().isoformat(),
            decision_id=self._generate_decision_id()
        )
        
        # Log decision
        self._log_decision(decision)
        
        return decision
    
    async def _assess_action_impact(self, action_type: ActionType, description: str, 
                                  context: Dict[str, Any]) -> ActionImpact:
        """Assess the potential impact of an action"""
        
        # Use LLM to assess impact
        assessment_prompt = f"""
        Assess the impact of this action:
        Type: {action_type.value}
        Description: {description}
        Context: {json.dumps(context, indent=2)}
        
        Evaluate:
        1. Success probability (0-1)
        2. Potential benefit (0-1)
        3. Potential harm (0-1)
        4. Reversibility (0-1)
        5. Scope (local/project/system/global)
        6. Affected components
        7. Estimated duration (minutes)
        
        Return JSON: {{
            "success_probability": 0.8,
            "potential_benefit": 0.9,
            "potential_harm": 0.1,
            "reversibility": 0.9,
            "scope": "project",
            "affected_components": ["code", "tests"],
            "estimated_duration": 30
        }}
        """
        
        try:
            from agents import get_default_model
            llm = get_default_model()
            response = await llm.ainvoke(assessment_prompt)
            assessment_text = response.get('output', '') if isinstance(response, dict) else str(response)
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', assessment_text, re.DOTALL)
            if json_match:
                assessment_data = json.loads(json_match.group())
                return ActionImpact(**assessment_data)
            else:
                # Fallback assessment
                return self._fallback_impact_assessment(action_type)
                
        except Exception as e:
            return self._fallback_impact_assessment(action_type)
    
    def _fallback_impact_assessment(self, action_type: ActionType) -> ActionImpact:
        """Fallback impact assessment based on action type"""
        
        assessments = {
            ActionType.CODE_GENERATION: ActionImpact(
                success_probability=0.9,
                potential_benefit=0.8,
                potential_harm=0.1,
                reversibility=0.9,
                scope="project",
                affected_components=["source_code"],
                estimated_duration=15
            ),
            ActionType.DEPLOYMENT: ActionImpact(
                success_probability=0.7,
                potential_benefit=0.9,
                potential_harm=0.4,
                reversibility=0.6,
                scope="system",
                affected_components=["production", "users"],
                estimated_duration=60
            ),
            ActionType.GIT_OPERATION: ActionImpact(
                success_probability=0.95,
                potential_benefit=0.7,
                potential_harm=0.2,
                reversibility=0.8,
                scope="project",
                affected_components=["repository", "version_control"],
                estimated_duration=5
            ),
            ActionType.SYSTEM_UPGRADE: ActionImpact(
                success_probability=0.6,
                potential_benefit=0.9,
                potential_harm=0.6,
                reversibility=0.4,
                scope="system",
                affected_components=["system", "dependencies"],
                estimated_duration=120
            )
        }
        
        return assessments.get(action_type, ActionImpact(
            success_probability=0.5,
            potential_benefit=0.5,
            potential_harm=0.5,
            reversibility=0.5,
            scope="project",
            affected_components=["unknown"],
            estimated_duration=30
        ))
    
    def _calculate_utility_score(self, impact: ActionImpact) -> float:
        """Calculate utility score using expected value theory"""
        
        # Expected value = success_prob * benefit - (1-success_prob) * harm
        expected_benefit = impact.success_probability * impact.potential_benefit
        expected_harm = (1 - impact.success_probability) * impact.potential_harm
        
        # Adjust for reversibility
        reversibility_factor = impact.reversibility * 0.3  # Reduce risk if reversible
        
        # Base utility score
        utility = expected_benefit - expected_harm + reversibility_factor
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, utility))
    
    def _determine_risk_level(self, impact: ActionImpact) -> RiskLevel:
        """Determine risk level based on impact assessment"""
        
        risk_score = (1 - impact.success_probability) * impact.potential_harm * (1 - impact.reversibility)
        
        if risk_score < 0.1:
            return RiskLevel.LOW
        elif risk_score < 0.3:
            return RiskLevel.MEDIUM
        elif risk_score < 0.6:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _requires_approval(self, utility_score: float, risk_level: RiskLevel) -> bool:
        """Determine if human approval is required"""
        
        # Always require approval for critical risk
        if risk_level == RiskLevel.CRITICAL:
            return True
        
        # Require approval for high risk with low utility
        if risk_level == RiskLevel.HIGH and utility_score < 0.7:
            return True
        
        # Require approval for medium risk with very low utility
        if risk_level == RiskLevel.MEDIUM and utility_score < 0.4:
            return True
        
        return False
    
    def _generate_veto_hooks(self, impact: ActionImpact) -> List[str]:
        """Generate human veto hooks for monitoring"""
        
        hooks = []
        
        if impact.scope == "system":
            hooks.append("Monitor system logs for 5 minutes after action")
            hooks.append("Check for error alerts in monitoring dashboard")
        
        if impact.potential_harm > 0.5:
            hooks.append("Verify no data loss occurred")
            hooks.append("Check backup integrity")
        
        if impact.estimated_duration > 60:
            hooks.append("Monitor progress every 15 minutes")
            hooks.append("Have rollback plan ready")
        
        return hooks
    
    async def _generate_contingency_plan(self, action_type: ActionType, impact: ActionImpact) -> str:
        """Generate contingency plan for action failure"""
        
        contingency_prompt = f"""
        Generate a contingency plan for this action:
        Type: {action_type.value}
        Impact: {asdict(impact)}
        
        Create a step-by-step plan for:
        1. Immediate response if action fails
        2. Rollback procedures
        3. Recovery steps
        4. Communication plan
        5. Prevention measures for future
        
        Be specific and actionable.
        """
        
        try:
            from agents import get_default_model
            llm = get_default_model()
            response = await llm.ainvoke(contingency_prompt)
            return response.get('output', 'Standard rollback and recovery procedures') if isinstance(response, dict) else str(response)
        except Exception as e:
            return f"Standard rollback and recovery procedures. Error: {str(e)}"
    
    def _generate_decision_id(self) -> str:
        """Generate unique decision ID"""
        import uuid
        return f"decision_{uuid.uuid4().hex[:8]}"
    
    def _log_decision(self, decision: LethalityDecision):
        """Log decision to brain for analysis"""
        
        self.decision_history.append(decision)
        
        # Store in brain
        self.brain.add_node(
            node_type=NodeType.DECISION,
            content=f"Lethality decision: {json.dumps(asdict(decision), indent=2, cls=EnumEncoder)}",
            metadata={
                'decision_id': decision.decision_id,
                'action_type': decision.action_type.value,
                'utility_score': decision.utility_score,
                'risk_level': decision.risk_level.value
            }
        )

# ==================== AUTONOMOUS ACTION EXECUTOR ====================

class AutonomousActionExecutor:
    """Executor for autonomous actions with safety checks"""
    
    def __init__(self, brain: ProjectBrain, utility_agent: UtilityBasedAgent):
        self.brain = brain
        self.utility_agent = utility_agent
        self.git_tools = GitRepositoryTools(brain)
        self.execution_history = []
        
    async def execute_action(self, decision: LethalityDecision, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action based on lethality decision"""
        
        # Pre-execution safety checks
        safety_check = await self._perform_safety_checks(decision, context)
        if not safety_check['safe_to_proceed']:
            return {
                'success': False,
                'error': f"Safety check failed: {safety_check['reason']}",
                'decision_id': decision.decision_id
            }
        
        # Execute based on action type
        try:
            if decision.action_type == ActionType.CODE_GENERATION:
                result = await self._execute_code_generation(decision, context)
            elif decision.action_type == ActionType.DEPLOYMENT:
                result = await self._execute_deployment(decision, context)
            elif decision.action_type == ActionType.GIT_OPERATION:
                result = await self._execute_git_operation(decision, context)
            elif decision.action_type == ActionType.SYSTEM_UPGRADE:
                result = await self._execute_system_upgrade(decision, context)
            else:
                result = await self._execute_generic_action(decision, context)
            
            # Post-execution validation
            validation = await self._validate_execution_result(result, decision)
            
            # Log execution
            self._log_execution(decision, result, validation)
            
            return {
                'success': result['success'],
                'result': result,
                'validation': validation,
                'decision_id': decision.decision_id
            }
            
        except Exception as e:
            # Execute contingency plan
            contingency_result = await self._execute_contingency_plan(decision, str(e))
            return {
                'success': False,
                'error': str(e),
                'contingency_executed': contingency_result,
                'decision_id': decision.decision_id
            }
    
    async def _perform_safety_checks(self, decision: LethalityDecision, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform safety checks before execution"""
        
        checks = {
            'safe_to_proceed': True,
            'reason': None,
            'checks_performed': []
        }
        
        # Check if approval is required but not given
        if decision.approval_required:
            approval_given = context.get('human_approval', False)
            if not approval_given:
                checks['safe_to_proceed'] = False
                checks['reason'] = "Human approval required but not given"
                return checks
        
        # Check system resources
        resource_check = await self._check_system_resources(decision)
        checks['checks_performed'].append(('resource_check', resource_check))
        if not resource_check['sufficient']:
            checks['safe_to_proceed'] = False
            checks['reason'] = f"Insufficient resources: {resource_check['details']}"
            return checks
        
        # Check for conflicting operations
        conflict_check = await self._check_conflicting_operations(decision)
        checks['checks_performed'].append(('conflict_check', conflict_check))
        if conflict_check['has_conflicts']:
            checks['safe_to_proceed'] = False
            checks['reason'] = f"Conflicting operations: {conflict_check['conflicts']}"
            return checks
        
        return checks
    
    async def _check_system_resources(self, decision: LethalityDecision) -> Dict[str, Any]:
        """Check if system has sufficient resources"""
        
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check memory usage
            memory = psutil.virtual_memory()
            
            # Check disk space
            disk = psutil.disk_usage('/')
            
            # Determine if sufficient
            sufficient = (
                cpu_percent < 90 and
                memory.percent < 90 and
                disk.free > 1024 * 1024 * 1024  # 1GB free
            )
            
            return {
                'sufficient': sufficient,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_free_gb': disk.free / (1024**3)
                }
            }
            
        except ImportError:
            # psutil not available, assume sufficient
            return {'sufficient': True, 'details': 'psutil not available'}
    
    async def _check_conflicting_operations(self, decision: LethalityDecision) -> Dict[str, Any]:
        """Check for conflicting operations"""
        
        # Check recent executions
        recent_executions = [
            exec_record for exec_record in self.execution_history
            if (datetime.now() - datetime.fromisoformat(exec_record['timestamp'])).seconds < 300  # Last 5 minutes
        ]
        
        conflicts = []
        for execution in recent_executions:
            if execution['action_type'] == decision.action_type.value:
                conflicts.append(f"Recent {execution['action_type']} operation")
        
        return {
            'has_conflicts': len(conflicts) > 0,
            'conflicts': conflicts
        }
    
    async def _execute_code_generation(self, decision: LethalityDecision, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation action"""
        
        # This would integrate with the existing coding workflow
        from graph import run_workflow
        
        task = context.get('task', decision.action_description)
        
        result = run_workflow(
            task=task,
            brain=self.brain,
            mode="quick"
        )
        
        return {
            'success': result.get('success', False),
            'code_files': result.get('code_files', []),
            'execution_time': datetime.now().isoformat()
        }
    
    async def _execute_deployment(self, decision: LethalityDecision, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment action"""
        
        # Simulate deployment process
        deployment_config = context.get('deployment_config', {})
        
        # Create deployment script
        deployment_script = f"""
        #!/bin/bash
        echo "Starting deployment..."
        echo "Config: {json.dumps(deployment_config)}"
        
        # Simulate deployment steps
        sleep 5
        echo "Deployment completed successfully"
        """
        
        # Execute in temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(deployment_script)
            script_path = f.name
        
        try:
            result = subprocess.run(['bash', script_path], capture_output=True, text=True, timeout=60)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now().isoformat()
            }
        finally:
            os.unlink(script_path)
    
    async def _execute_git_operation(self, decision: LethalityDecision, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute git operation"""
        
        operation = context.get('git_operation', 'commit')
        repo_name = context.get('repo_name', 'default')
        
        if operation == 'commit':
            message = context.get('commit_message', 'Auto-commit from lethality engine')
            result = self.git_tools.create_repository_file(
                repo_name=repo_name,
                file_path=context.get('file_path', 'auto_generated.py'),
                content=context.get('content', '# Auto-generated content'),
                commit_message=message
            )
        elif operation == 'push':
            result = self.git_tools.push_changes(repo_name)
        else:
            result = f"Unknown git operation: {operation}"
        
        return {
            'success': 'successfully' in str(result).lower() or 'success' in str(result).lower(),
            'result': result,
            'execution_time': datetime.now().isoformat()
        }
    
    async def _execute_system_upgrade(self, decision: LethalityDecision, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system upgrade action"""
        
        # This would be a more sophisticated system upgrade process
        # For now, simulate package updates
        
        upgrade_script = """
        echo "Starting system upgrade..."
        pip install --upgrade pip
        pip list --outdated
        echo "Upgrade completed"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(upgrade_script)
            script_path = f.name
        
        try:
            result = subprocess.run(['bash', script_path], capture_output=True, text=True, timeout=120)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now().isoformat()
            }
        finally:
            os.unlink(script_path)
    
    async def _execute_generic_action(self, decision: LethalityDecision, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic action"""
        
        # Generic action execution
        action_script = context.get('action_script', f"echo 'Executing: {decision.action_description}'")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(action_script)
            script_path = f.name
        
        try:
            result = subprocess.run(['bash', script_path], capture_output=True, text=True, timeout=30)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now().isoformat()
            }
        finally:
            os.unlink(script_path)
    
    async def _validate_execution_result(self, result: Dict[str, Any], decision: LethalityDecision) -> Dict[str, Any]:
        """Validate execution result"""
        
        validation = {
            'valid': result.get('success', False),
            'issues': [],
            'recommendations': []
        }
        
        # Check for common issues
        if 'stderr' in result and result['stderr']:
            validation['issues'].append(f"Errors in execution: {result['stderr']}")
        
        if 'stdout' in result and 'error' in result['stdout'].lower():
            validation['issues'].append("Error detected in output")
        
        # Check execution time
        if 'execution_time' in result:
            try:
                exec_time = datetime.fromisoformat(result['execution_time'])
                expected_duration = decision.impact_assessment.estimated_duration
                if (datetime.now() - exec_time).seconds > expected_duration * 60:
                    validation['issues'].append("Execution took longer than expected")
            except:
                pass
        
        # Generate recommendations
        if validation['issues']:
            validation['recommendations'].append("Review execution logs for details")
            validation['recommendations'].append("Consider rolling back if critical")
        
        return validation
    
    async def _execute_contingency_plan(self, decision: LethalityDecision, error: str) -> Dict[str, Any]:
        """Execute contingency plan"""
        
        # Log the error and contingency execution
        self.brain.add_node(
            node_type=NodeType.LEARNING,
            content=f"Contingency executed for {decision.action_type.value}: {error}",
            metadata={'decision_id': decision.decision_id, 'error': error}
        )
        
        return {
            'contingency_executed': True,
            'error': error,
            'plan': decision.contingency_plan,
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_execution(self, decision: LethalityDecision, result: Dict[str, Any], validation: Dict[str, Any]):
        """Log execution to history and brain"""
        
        execution_record = {
            'decision_id': decision.decision_id,
            'action_type': decision.action_type.value,
            'success': result.get('success', False),
            'timestamp': datetime.now().isoformat(),
            'validation': validation
        }
        
        self.execution_history.append(execution_record)
        
        # Store in brain
        self.brain.add_node(
            node_type=NodeType.LEARNING,
            content=f"Action execution: {json.dumps(execution_record, indent=2)}",
            metadata={'decision_id': decision.decision_id, 'action_type': decision.action_type.value}
        )

# ==================== LETHALITY ENGINE MAIN CLASS ====================

class LethalityEngine:
    """Main lethality engine coordinating utility-based decisions and autonomous actions"""
    
    def __init__(self, brain: ProjectBrain, risk_tolerance: float = 0.7):
        self.brain = brain
        self.utility_agent = UtilityBasedAgent(brain, risk_tolerance)
        self.action_executor = AutonomousActionExecutor(brain, self.utility_agent)
        self.impact_scores = {}
        
    async def evaluate_and_execute(self, action_type: ActionType, action_description: str, 
                                 context: Dict[str, Any], auto_execute: bool = False) -> Dict[str, Any]:
        """Evaluate and optionally execute an action"""
        
        # Evaluate action
        decision = await self.utility_agent.evaluate_action(action_type, action_description, context)
        
        # Check if execution is allowed
        if auto_execute and not decision.approval_required:
            # Execute autonomously
            execution_result = await self.action_executor.execute_action(decision, context)
            
            return {
                'decision': asdict(decision),
                'execution': execution_result,
                'autonomous': True
            }
        else:
            # Return decision for human review
            return {
                'decision': asdict(decision),
                'execution': None,
                'autonomous': False,
                'requires_approval': decision.approval_required
            }
    
    async def execute_approved_action(self, decision_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a previously approved action"""
        
        # Find the decision
        decision = None
        for d in self.utility_agent.decision_history:
            if d.decision_id == decision_id:
                decision = d
                break
        
        if not decision:
            return {'success': False, 'error': 'Decision not found'}
        
        # Execute the action
        return await self.action_executor.execute_action(decision, context)
    
    def get_lethality_stats(self) -> Dict[str, Any]:
        """Get lethality engine statistics"""
        
        total_decisions = len(self.utility_agent.decision_history)
        autonomous_executions = len([d for d in self.utility_agent.decision_history if not d.approval_required])
        successful_executions = len([e for e in self.action_executor.execution_history if e['success']])
        
        avg_utility_score = np.mean([d.utility_score for d in self.utility_agent.decision_history]) if total_decisions > 0 else 0
        
        risk_distribution = {}
        for risk_level in RiskLevel:
            count = len([d for d in self.utility_agent.decision_history if d.risk_level == risk_level])
            risk_distribution[risk_level.value] = count
        
        return {
            'total_decisions': total_decisions,
            'autonomous_executions': autonomous_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / len(self.action_executor.execution_history) if self.action_executor.execution_history else 0,
            'average_utility_score': avg_utility_score,
            'risk_distribution': risk_distribution,
            'last_decision': self.utility_agent.decision_history[-1].timestamp if self.utility_agent.decision_history else None
        } 