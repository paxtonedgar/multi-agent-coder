"""
Multi-Agent Debate Framework for Enhanced Self-Critique
Implements collaborative reasoning through structured agent debates
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np

from memory import ProjectBrain, NodeType
from agents import get_default_model

# ==================== DEBATE DATA STRUCTURES ====================

class DebateRole(Enum):
    """Roles for debate agents"""
    ADVOCATE = "advocate"
    CRITIC = "critic"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    EXPERT = "expert"
    VALIDATOR = "validator"

class DebateStyle(Enum):
    """Debate styles for different perspectives"""
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    NEUTRAL = "neutral"
    INTEGRATIVE = "integrative"
    TECHNICAL = "technical"
    PRAGMATIC = "pragmatic"

@dataclass
class DebateAgent:
    """Individual agent in the debate"""
    id: str
    role: DebateRole
    style: DebateStyle
    specialization: str
    confidence_threshold: float
    expertise_areas: List[str]

@dataclass
class DebateResponse:
    """Response from a debate agent"""
    agent_id: str
    role: DebateRole
    style: DebateStyle
    response: str
    confidence: float
    recommendations: List[str]
    concerns: List[str]
    evidence: List[str]
    timestamp: str

@dataclass
class DebateRound:
    """A single round of debate"""
    round_number: int
    responses: List[DebateResponse]
    synthesis: str
    consensus_score: float
    key_insights: List[str]
    remaining_concerns: List[str]

@dataclass
class DebateResult:
    """Final result of a debate"""
    final_proposal: str
    debate_rounds: List[DebateRound]
    consensus_score: float
    confidence: float
    key_decisions: List[str]
    implementation_plan: str
    risk_assessment: Dict[str, Any]
    success_metrics: List[str]

# ==================== DEBATE FRAMEWORK ====================

class MultiAgentDebateFramework:
    """Multi-agent debate system for enhanced reasoning validation"""
    
    def __init__(self, brain: ProjectBrain, max_rounds: int = 3):
        self.brain = brain
        self.max_rounds = max_rounds
        self.llm = get_default_model()
        self.debate_history = []
        
        # Initialize debate agents
        self.agents = self._initialize_debate_agents()
        
    def _initialize_debate_agents(self) -> List[DebateAgent]:
        """Initialize the debate agents with different roles and styles"""
        
        agents = [
            DebateAgent(
                id="advocate_1",
                role=DebateRole.ADVOCATE,
                style=DebateStyle.OPTIMISTIC,
                specialization="Solution advocacy",
                confidence_threshold=0.7,
                expertise_areas=["benefits", "opportunities", "positive_outcomes"]
            ),
            DebateAgent(
                id="critic_1", 
                role=DebateRole.CRITIC,
                style=DebateStyle.PESSIMISTIC,
                specialization="Risk assessment",
                confidence_threshold=0.8,
                expertise_areas=["risks", "challenges", "failure_modes"]
            ),
            DebateAgent(
                id="analyst_1",
                role=DebateRole.ANALYST,
                style=DebateStyle.NEUTRAL,
                specialization="Technical analysis",
                confidence_threshold=0.75,
                expertise_areas=["technical_feasibility", "performance", "scalability"]
            ),
            DebateAgent(
                id="synthesizer_1",
                role=DebateRole.SYNTHESIZER,
                style=DebateStyle.INTEGRATIVE,
                specialization="Solution integration",
                confidence_threshold=0.8,
                expertise_areas=["integration", "coherence", "optimization"]
            ),
            DebateAgent(
                id="expert_1",
                role=DebateRole.EXPERT,
                style=DebateStyle.TECHNICAL,
                specialization="Domain expertise",
                confidence_threshold=0.85,
                expertise_areas=["best_practices", "industry_standards", "advanced_patterns"]
            ),
            DebateAgent(
                id="validator_1",
                role=DebateRole.VALIDATOR,
                style=DebateStyle.PRAGMATIC,
                specialization="Implementation validation",
                confidence_threshold=0.9,
                expertise_areas=["testing", "validation", "quality_assurance"]
            )
        ]
        
        return agents
    
    async def conduct_debate(self, proposal: str, context: str, 
                           task_type: str = "general") -> DebateResult:
        """Conduct a multi-agent debate on a proposal"""
        
        print(f"🤖 Starting multi-agent debate with {len(self.agents)} agents")
        print(f"📋 Proposal: {proposal[:100]}...")
        
        # Initialize debate
        current_proposal = proposal
        debate_rounds = []
        all_responses = []
        
        # Conduct debate rounds
        for round_num in range(self.max_rounds):
            print(f"🔄 Debate Round {round_num + 1}/{self.max_rounds}")
            
            # Get responses from all agents
            round_responses = []
            for agent in self.agents:
                response = await self._get_agent_response(
                    agent, current_proposal, context, round_num, all_responses, task_type
                )
                round_responses.append(response)
                all_responses.append(response)
            
            # Synthesize round results
            synthesis_result = await self._synthesize_round(
                round_responses, current_proposal, round_num
            )
            
            # Create debate round
            debate_round = DebateRound(
                round_number=round_num + 1,
                responses=round_responses,
                synthesis=synthesis_result['synthesis'],
                consensus_score=synthesis_result['consensus_score'],
                key_insights=synthesis_result['key_insights'],
                remaining_concerns=synthesis_result['remaining_concerns']
            )
            
            debate_rounds.append(debate_round)
            
            # Update proposal for next round
            current_proposal = synthesis_result['refined_proposal']
            
            # Check for consensus
            if synthesis_result['consensus_score'] > 0.85:
                print(f"✅ Consensus reached in round {round_num + 1}")
                break
        
        # Generate final result
        final_result = await self._generate_final_result(
            debate_rounds, current_proposal, context, task_type
        )
        
        # Log debate to brain
        self._log_debate_to_brain(final_result, proposal, context)
        
        return final_result
    
    async def _get_agent_response(self, agent: DebateAgent, proposal: str, 
                                context: str, round_num: int, 
                                previous_responses: List[DebateResponse],
                                task_type: str) -> DebateResponse:
        """Get response from a specific debate agent"""
        
        # Build context from previous responses
        previous_context = self._build_previous_context(previous_responses)
        
        # Create role-specific prompt
        prompt = self._create_agent_prompt(agent, proposal, context, round_num, 
                                         previous_context, task_type)
        
        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.get('output', '') if isinstance(response, dict) else str(response)
            
            # Parse response for structured data
            parsed_response = self._parse_agent_response(response_text, agent)
            
            return DebateResponse(
                agent_id=agent.id,
                role=agent.role,
                style=agent.style,
                response=parsed_response['response'],
                confidence=parsed_response['confidence'],
                recommendations=parsed_response['recommendations'],
                concerns=parsed_response['concerns'],
                evidence=parsed_response['evidence'],
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Error getting response from {agent.id}: {e}")
            return self._create_fallback_response(agent)
    
    def _create_agent_prompt(self, agent: DebateAgent, proposal: str, context: str,
                           round_num: int, previous_context: str, task_type: str) -> str:
        """Create a role-specific prompt for the agent"""
        
        role_instructions = {
            DebateRole.ADVOCATE: "Focus on the benefits, opportunities, and positive outcomes. Be optimistic but realistic.",
            DebateRole.CRITIC: "Identify potential risks, challenges, and failure modes. Be thorough in risk assessment.",
            DebateRole.ANALYST: "Provide technical analysis focusing on feasibility, performance, and scalability.",
            DebateRole.SYNTHESIZER: "Integrate different perspectives and find optimal solutions.",
            DebateRole.EXPERT: "Apply domain expertise and best practices to the proposal.",
            DebateRole.VALIDATOR: "Validate the proposal for implementation readiness and quality."
        }
        
        prompt = f"""
        You are a {agent.role.value} with {agent.style.value} style, specializing in {agent.specialization}.
        
        TASK TYPE: {task_type}
        ROUND: {round_num + 1}
        
        ORIGINAL PROPOSAL: {proposal}
        CONTEXT: {context}
        
        PREVIOUS DEBATE CONTEXT: {previous_context}
        
        INSTRUCTIONS: {role_instructions[agent.role]}
        
        Provide your analysis in the following JSON format:
        {{
            "response": "Your detailed analysis and perspective",
            "confidence": 0.85,
            "recommendations": ["rec1", "rec2"],
            "concerns": ["concern1", "concern2"],
            "evidence": ["evidence1", "evidence2"]
        }}
        
        Be specific, actionable, and consider the previous responses in your analysis.
        """
        
        return prompt
    
    def _parse_agent_response(self, response_text: str, agent: DebateAgent) -> Dict[str, Any]:
        """Parse structured response from agent"""
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    'response': parsed.get('response', response_text),
                    'confidence': parsed.get('confidence', 0.7),
                    'recommendations': parsed.get('recommendations', []),
                    'concerns': parsed.get('concerns', []),
                    'evidence': parsed.get('evidence', [])
                }
        except Exception as e:
            print(f"Error parsing response from {agent.id}: {e}")
        
        # Fallback parsing
        return {
            'response': response_text,
            'confidence': 0.7,
            'recommendations': [],
            'concerns': [],
            'evidence': []
        }
    
    def _create_fallback_response(self, agent: DebateAgent) -> DebateResponse:
        """Create a fallback response if agent fails"""
        
        return DebateResponse(
            agent_id=agent.id,
            role=agent.role,
            style=agent.style,
            response=f"Unable to provide response for {agent.role.value}",
            confidence=0.3,
            recommendations=[],
            concerns=["Technical issue prevented response"],
            evidence=[],
            timestamp=datetime.now().isoformat()
        )
    
    def _build_previous_context(self, previous_responses: List[DebateResponse]) -> str:
        """Build context from previous responses"""
        
        if not previous_responses:
            return "No previous responses"
        
        context_parts = []
        for response in previous_responses[-6:]:  # Last 6 responses
            context_parts.append(f"{response.role.value}: {response.response[:200]}...")
        
        return "\n".join(context_parts)
    
    async def _synthesize_round(self, responses: List[DebateResponse], 
                              current_proposal: str, round_num: int) -> Dict[str, Any]:
        """Synthesize responses from a debate round"""
        
        synthesis_prompt = f"""
        Synthesize the following debate responses for round {round_num + 1}:
        
        CURRENT PROPOSAL: {current_proposal}
        
        RESPONSES:
        {chr(10).join([f"{r.role.value} ({r.style.value}): {r.response}" for r in responses])}
        
        Create a synthesis that:
        1. Identifies key insights and agreements
        2. Addresses major concerns and disagreements
        3. Refines the proposal based on feedback
        4. Calculates consensus score (0-1)
        5. Lists remaining concerns
        
        Return as JSON:
        {{
            "synthesis": "Detailed synthesis of the round",
            "refined_proposal": "Updated proposal based on feedback",
            "consensus_score": 0.75,
            "key_insights": ["insight1", "insight2"],
            "remaining_concerns": ["concern1", "concern2"]
        }}
        """
        
        try:
            response = await self.llm.ainvoke(synthesis_prompt)
            synthesis_text = response.get('output', '') if isinstance(response, dict) else str(response)
            
            # Parse synthesis
            import re
            json_match = re.search(r'\{.*\}', synthesis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._create_fallback_synthesis(responses, current_proposal)
                
        except Exception as e:
            print(f"Error in synthesis: {e}")
            return self._create_fallback_synthesis(responses, current_proposal)
    
    def _create_fallback_synthesis(self, responses: List[DebateResponse], 
                                 current_proposal: str) -> Dict[str, Any]:
        """Create fallback synthesis if LLM fails"""
        
        # Simple consensus calculation
        avg_confidence = np.mean([r.confidence for r in responses])
        
        return {
            'synthesis': f"Synthesis of {len(responses)} responses",
            'refined_proposal': current_proposal,
            'consensus_score': avg_confidence,
            'key_insights': ["Fallback synthesis generated"],
            'remaining_concerns': ["Synthesis generation failed"]
        }
    
    async def _generate_final_result(self, debate_rounds: List[DebateRound], 
                                   final_proposal: str, context: str, 
                                   task_type: str) -> DebateResult:
        """Generate final debate result"""
        
        # Calculate overall consensus
        consensus_scores = [round.consensus_score for round in debate_rounds]
        overall_consensus = np.mean(consensus_scores) if consensus_scores else 0.5
        
        # Generate implementation plan
        implementation_plan = await self._generate_implementation_plan(
            final_proposal, context, task_type
        )
        
        # Generate risk assessment
        risk_assessment = await self._generate_risk_assessment(
            final_proposal, debate_rounds
        )
        
        # Generate success metrics
        success_metrics = await self._generate_success_metrics(
            final_proposal, task_type
        )
        
        # Extract key decisions
        key_decisions = []
        for round_data in debate_rounds:
            key_decisions.extend(round_data.key_insights)
        
        return DebateResult(
            final_proposal=final_proposal,
            debate_rounds=debate_rounds,
            consensus_score=overall_consensus,
            confidence=overall_consensus,
            key_decisions=key_decisions[:10],  # Top 10 decisions
            implementation_plan=implementation_plan,
            risk_assessment=risk_assessment,
            success_metrics=success_metrics
        )
    
    async def _generate_implementation_plan(self, proposal: str, context: str, 
                                          task_type: str) -> str:
        """Generate implementation plan based on final proposal"""
        
        plan_prompt = f"""
        Generate a detailed implementation plan for:
        
        PROPOSAL: {proposal}
        CONTEXT: {context}
        TASK TYPE: {task_type}
        
        Include:
        1. Phase-by-phase breakdown
        2. Resource requirements
        3. Timeline estimates
        4. Success criteria
        5. Risk mitigation strategies
        """
        
        try:
            response = await self.llm.ainvoke(plan_prompt)
            return response.get('output', 'Implementation plan generation failed') if isinstance(response, dict) else str(response)
        except Exception as e:
            return f"Implementation plan generation failed: {str(e)}"
    
    async def _generate_risk_assessment(self, proposal: str, 
                                      debate_rounds: List[DebateRound]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        
        # Extract concerns from all rounds
        all_concerns = []
        for round_data in debate_rounds:
            all_concerns.extend(round_data.remaining_concerns)
            for response in round_data.responses:
                all_concerns.extend(response.concerns)
        
        # Remove duplicates
        unique_concerns = list(set(all_concerns))
        
        risk_assessment = {
            'high_risks': [c for c in unique_concerns if any(word in c.lower() 
                           for word in ['critical', 'severe', 'high', 'major'])],
            'medium_risks': [c for c in unique_concerns if any(word in c.lower() 
                            for word in ['moderate', 'medium', 'significant'])],
            'low_risks': [c for c in unique_concerns if any(word in c.lower() 
                          for word in ['minor', 'low', 'small'])],
            'mitigation_strategies': [],
            'overall_risk_level': 'medium' if len(unique_concerns) > 5 else 'low'
        }
        
        return risk_assessment
    
    async def _generate_success_metrics(self, proposal: str, task_type: str) -> List[str]:
        """Generate success metrics for the proposal"""
        
        metrics_prompt = f"""
        Generate specific, measurable success metrics for:
        
        PROPOSAL: {proposal}
        TASK TYPE: {task_type}
        
        Provide 5-10 quantifiable metrics that can be used to measure success.
        """
        
        try:
            response = await self.llm.ainvoke(metrics_prompt)
            metrics_text = response.get('output', '') if isinstance(response, dict) else str(response)
            
            # Extract metrics (simple parsing)
            lines = metrics_text.split('\n')
            metrics = [line.strip() for line in lines if line.strip() and 
                      any(word in line.lower() for word in ['metric', 'measure', 'kpi', 'success'])]
            
            return metrics[:10]  # Return top 10 metrics
            
        except Exception as e:
            return [f"Success metric generation failed: {str(e)}"]
    
    def _log_debate_to_brain(self, result: DebateResult, original_proposal: str, 
                           context: str):
        """Log debate results to brain for analysis"""
        
        debate_summary = {
            'original_proposal': original_proposal,
            'final_proposal': result.final_proposal,
            'consensus_score': result.consensus_score,
            'confidence': result.confidence,
            'rounds_count': len(result.debate_rounds),
            'key_decisions': result.key_decisions,
            'risk_assessment': result.risk_assessment,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in brain
        self.brain.add_node(
            node_type=NodeType.LEARNING,
            content=f"Multi-agent debate result: {json.dumps(debate_summary, indent=2)}",
            metadata={
                'debate_type': 'multi_agent',
                'consensus_score': result.consensus_score,
                'rounds_count': len(result.debate_rounds),
                'agents_count': len(self.agents)
            }
        )
        
        # Store in debate history
        self.debate_history.append(debate_summary)
    
    def get_debate_stats(self) -> Dict[str, Any]:
        """Get statistics about debate performance"""
        
        if not self.debate_history:
            return {'total_debates': 0}
        
        total_debates = len(self.debate_history)
        avg_consensus = np.mean([d['consensus_score'] for d in self.debate_history])
        avg_confidence = np.mean([d['confidence'] for d in self.debate_history])
        avg_rounds = np.mean([d['rounds_count'] for d in self.debate_history])
        
        return {
            'total_debates': total_debates,
            'average_consensus_score': avg_consensus,
            'average_confidence': avg_confidence,
            'average_rounds': avg_rounds,
            'agents_count': len(self.agents),
            'last_debate': self.debate_history[-1]['timestamp'] if self.debate_history else None
        }

# ==================== INTEGRATION WITH EXISTING SYSTEM ====================

class EnhancedReasoningWithDebate:
    """Enhanced reasoning system that integrates debate framework"""
    
    def __init__(self, brain: ProjectBrain):
        self.brain = brain
        self.debate_framework = MultiAgentDebateFramework(brain)
        self.llm = get_default_model()
    
    async def reason_with_debate(self, task: str, context: str = "", 
                               task_type: str = "general") -> Dict[str, Any]:
        """Execute reasoning with debate validation"""
        
        # Initial reasoning
        initial_reasoning = await self._generate_initial_reasoning(task, context)
        
        # Conduct debate on the reasoning
        debate_result = await self.debate_framework.conduct_debate(
            initial_reasoning, context, task_type
        )
        
        # Refine reasoning based on debate
        refined_reasoning = await self._refine_reasoning(
            initial_reasoning, debate_result
        )
        
        return {
            'initial_reasoning': initial_reasoning,
            'debate_result': debate_result,
            'refined_reasoning': refined_reasoning,
            'consensus_score': debate_result.consensus_score,
            'confidence': debate_result.confidence,
            'key_insights': debate_result.key_decisions,
            'risk_assessment': debate_result.risk_assessment
        }
    
    async def _generate_initial_reasoning(self, task: str, context: str) -> str:
        """Generate initial reasoning for the task"""
        
        reasoning_prompt = f"""
        Provide initial reasoning for this task:
        
        TASK: {task}
        CONTEXT: {context}
        
        Provide a comprehensive, step-by-step reasoning approach.
        """
        
        try:
            response = await self.llm.ainvoke(reasoning_prompt)
            return response.get('output', 'Initial reasoning generation failed') if isinstance(response, dict) else str(response)
        except Exception as e:
            return f"Initial reasoning failed: {str(e)}"
    
    async def _refine_reasoning(self, initial_reasoning: str, 
                              debate_result: DebateResult) -> str:
        """Refine reasoning based on debate results"""
        
        refinement_prompt = f"""
        Refine this reasoning based on the debate results:
        
        INITIAL REASONING: {initial_reasoning}
        
        DEBATE INSIGHTS: {chr(10).join(debate_result.key_decisions)}
        CONSENSUS SCORE: {debate_result.consensus_score}
        RISK ASSESSMENT: {json.dumps(debate_result.risk_assessment, indent=2)}
        
        Provide a refined, improved version that addresses the debate insights.
        """
        
        try:
            response = await self.llm.ainvoke(refinement_prompt)
            return response.get('output', initial_reasoning) if isinstance(response, dict) else str(response)
        except Exception as e:
            return f"Refinement failed: {str(e)}" 