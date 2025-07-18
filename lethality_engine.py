"""
Lethality Engine for High-Impact Effectiveness
Implements utility-based decision making and risk assessment
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from memory import ProjectBrain, NodeType
from tools import GitRepositoryTools

# ==================== BASIC DATA STRUCTURES ====================

class ActionType(Enum):
    """Types of actions the system can take"""
    CODE_GENERATION = "code_generation"
    DEPLOYMENT = "deployment"
    GIT_OPERATION = "git_operation"
    SYSTEM_UPGRADE = "system_upgrade"
    CONFIGURATION_CHANGE = "configuration_change"

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
        
        # Determine if approval is required
        approval_required = self._determine_approval_requirement(utility_score, risk_level)
        
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
    
    def _determine_approval_requirement(self, utility_score: float, risk_level: RiskLevel) -> bool:
        """Determine if human approval is required"""
        
        # Require approval for high risk actions
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return True
        
        # Require approval for low utility actions
        if utility_score < 0.3:
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
            content=f"Lethality decision: {json.dumps(asdict(decision), indent=2)}",
            metadata={
                'decision_id': decision.decision_id,
                'action_type': decision.action_type.value,
                'utility_score': decision.utility_score,
                'risk_level': decision.risk_level.value
            }
        )

# ==================== LETHALITY ENGINE MAIN CLASS ====================

class LethalityEngine:
    """Main lethality engine coordinating utility-based decisions"""
    
    def __init__(self, brain: ProjectBrain, risk_tolerance: float = 0.7):
        self.brain = brain
        self.utility_agent = UtilityBasedAgent(brain, risk_tolerance)
        self.impact_scores = {}
        
    async def evaluate_action(self, action_type: ActionType, action_description: str, 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an action using utility-based decision making"""
        
        # Evaluate action
        decision = await self.utility_agent.evaluate_action(action_type, action_description, context)
        
        return {
            'decision': asdict(decision),
            'requires_approval': decision.approval_required
        }
    
    def get_lethality_stats(self) -> Dict[str, Any]:
        """Get lethality engine statistics"""
        
        total_decisions = len(self.utility_agent.decision_history)
        autonomous_decisions = len([d for d in self.utility_agent.decision_history if not d.approval_required])
        
        avg_utility_score = np.mean([d.utility_score for d in self.utility_agent.decision_history]) if total_decisions > 0 else 0
        
        risk_distribution = {}
        for risk_level in RiskLevel:
            count = len([d for d in self.utility_agent.decision_history if d.risk_level == risk_level])
            risk_distribution[risk_level.value] = count
        
        return {
            'total_decisions': total_decisions,
            'autonomous_decisions': autonomous_decisions,
            'average_utility_score': avg_utility_score,
            'risk_distribution': risk_distribution,
            'last_decision': self.utility_agent.decision_history[-1].timestamp if self.utility_agent.decision_history else None
        } 