"""
Creativity Engine for Novel Inventions and Creative Outputs
Implements evolutionary design, swarm agents, and novelty exploration
"""

import os
import json
import asyncio
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from memory import ProjectBrain, NodeType
from agents import get_default_model

# ==================== CREATIVITY DATA STRUCTURES ====================

class CreativityType(Enum):
    """Types of creativity approaches"""
    EVOLUTIONARY = "evolutionary"
    SWARM = "swarm"
    SYNESTHETIC = "synesthetic"
    DIVERGENT = "divergent"
    CONVERGENT = "convergent"
    COMBINATORIAL = "combinatorial"

@dataclass
class CreativeIdea:
    """Represents a creative idea"""
    id: str
    content: str
    creativity_type: CreativityType
    novelty_score: float  # 0-1
    feasibility_score: float  # 0-1
    originality_score: float  # 0-1
    tags: List[str]
    parent_ids: List[str] = None
    generation: int = 0
    timestamp: str = None
    fitness_score: float = 0.0  # Added for evolutionary algorithms
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class SwarmAgent:
    """Individual agent in the swarm"""
    id: str
    specialization: str
    creativity_style: str
    current_ideas: List[str]
    collaboration_history: List[str]
    performance_score: float

# ==================== EVOLUTIONARY DESIGN GENERATOR ====================

class EvolutionaryDesignGenerator:
    """Generates creative designs through evolutionary algorithms"""
    
    def __init__(self, brain: ProjectBrain, population_size: int = 10, generations: int = 5):
        self.brain = brain
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.fitness_history = []
        self.llm = get_default_model()
        
    async def evolve_design(self, task: str, constraints: Dict[str, Any] = None) -> List[CreativeIdea]:
        """Evolve creative designs for a given task"""
        
        # Initialize population
        await self._initialize_population(task, constraints)
        
        # Evolve through generations
        for generation in range(self.generations):
            print(f"🔄 Evolving generation {generation + 1}/{self.generations}")
            
            # Evaluate fitness
            await self._evaluate_fitness()
            
            # Select parents
            parents = self._select_parents()
            
            # Generate offspring through crossover and mutation
            offspring = await self._generate_offspring(parents, task)
            
            # Update population
            self._update_population(offspring)
            
            # Log generation stats
            self._log_generation_stats(generation)
        
        # Return best designs
        return self._get_best_designs()
    
    async def _initialize_population(self, task: str, constraints: Dict[str, Any] = None):
        """Initialize the initial population of designs"""
        
        population_prompt = f"""
        Generate {self.population_size} diverse, creative design ideas for: {task}
        
        Constraints: {json.dumps(constraints, indent=2) if constraints else 'None'}
        
        Each idea should be:
        1. Novel and innovative
        2. Technically feasible
        3. Different from traditional approaches
        4. Well-described with implementation details
        
        Return as JSON array:
        [
            {{
                "content": "Detailed design description",
                "novelty_score": 0.8,
                "feasibility_score": 0.7,
                "originality_score": 0.9,
                "tags": ["tag1", "tag2"]
            }}
        ]
        """
        
        try:
            response = await self.llm.ainvoke(population_prompt)
            ideas_text = response.get('output', '') if isinstance(response, dict) else str(response)
            
            # Extract JSON array
            import re
            json_match = re.search(r'\[.*\]', ideas_text, re.DOTALL)
            if json_match:
                ideas_data = json.loads(json_match.group())
                
                for i, idea_data in enumerate(ideas_data[:self.population_size]):
                    idea = CreativeIdea(
                        id=f"gen0_idea_{i}",
                        content=idea_data.get('content', ''),
                        creativity_type=CreativityType.EVOLUTIONARY,
                        novelty_score=idea_data.get('novelty_score', 0.5),
                        feasibility_score=idea_data.get('feasibility_score', 0.5),
                        originality_score=idea_data.get('originality_score', 0.5),
                        tags=idea_data.get('tags', []),
                        generation=0
                    )
                    self.population.append(idea)
            else:
                # Fallback: generate simple ideas
                await self._generate_fallback_population(task)
                
        except Exception as e:
            print(f"Error initializing population: {e}")
            await self._generate_fallback_population(task)
    
    async def _generate_fallback_population(self, task: str):
        """Generate fallback population if LLM fails"""
        
        base_ideas = [
            f"Traditional approach to {task}",
            f"Modern framework solution for {task}",
            f"Microservices architecture for {task}",
            f"Event-driven design for {task}",
            f"AI-powered solution for {task}",
            f"Blockchain-based approach to {task}",
            f"Edge computing solution for {task}",
            f"Serverless architecture for {task}",
            f"Reactive programming approach to {task}",
            f"Quantum-inspired algorithm for {task}"
        ]
        
        for i, idea_content in enumerate(base_ideas[:self.population_size]):
            idea = CreativeIdea(
                id=f"gen0_idea_{i}",
                content=idea_content,
                creativity_type=CreativityType.EVOLUTIONARY,
                novelty_score=random.uniform(0.3, 0.8),
                feasibility_score=random.uniform(0.5, 0.9),
                originality_score=random.uniform(0.4, 0.7),
                tags=[f"approach_{i}"],
                generation=0
            )
            self.population.append(idea)
    
    async def _evaluate_fitness(self):
        """Evaluate fitness of current population"""
        
        for idea in self.population:
            # Calculate composite fitness score
            fitness = (
                idea.novelty_score * 0.4 +
                idea.feasibility_score * 0.3 +
                idea.originality_score * 0.3
            )
            
            # Store fitness for selection
            idea.fitness_score = fitness
    
    def _select_parents(self) -> List[CreativeIdea]:
        """Select parents for crossover using tournament selection"""
        
        parents = []
        tournament_size = 3
        
        for _ in range(self.population_size // 2):
            # Tournament selection
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness_score)
            parents.append(winner)
        
        return parents
    
    async def _generate_offspring(self, parents: List[CreativeIdea], task: str) -> List[CreativeIdea]:
        """Generate offspring through crossover and mutation"""
        
        offspring = []
        current_generation = max(idea.generation for idea in self.population) + 1
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]
                
                # Crossover
                child_content = await self._crossover(parent1, parent2, task)
                
                # Mutation
                child_content = await self._mutate(child_content, task)
                
                # Create child idea
                child = CreativeIdea(
                    id=f"gen{current_generation}_idea_{len(offspring)}",
                    content=child_content,
                    creativity_type=CreativityType.EVOLUTIONARY,
                    novelty_score=self._inherit_trait(parent1.novelty_score, parent2.novelty_score),
                    feasibility_score=self._inherit_trait(parent1.feasibility_score, parent2.feasibility_score),
                    originality_score=self._inherit_trait(parent1.originality_score, parent2.originality_score),
                    tags=list(set(parent1.tags + parent2.tags)),
                    parent_ids=[parent1.id, parent2.id],
                    generation=current_generation
                )
                
                offspring.append(child)
        
        return offspring
    
    async def _crossover(self, parent1: CreativeIdea, parent2: CreativeIdea, task: str) -> str:
        """Perform crossover between two parent ideas"""
        
        crossover_prompt = f"""
        Combine these two design ideas for {task}:
        
        Parent 1: {parent1.content}
        Parent 2: {parent2.content}
        
        Create a new design that:
        1. Takes the best elements from both parents
        2. Introduces new innovative aspects
        3. Maintains feasibility
        4. Improves upon the original concepts
        
        Provide a detailed, creative design description.
        """
        
        try:
            response = await self.llm.ainvoke(crossover_prompt)
            return response.get('output', f"Combined design: {parent1.content[:100]} + {parent2.content[:100]}") if isinstance(response, dict) else str(response)
        except Exception as e:
            # Fallback crossover
            return f"Hybrid approach combining {parent1.content[:50]} with {parent2.content[:50]}"
    
    async def _mutate(self, content: str, task: str) -> str:
        """Apply mutation to introduce novelty"""
        
        mutation_prompt = f"""
        Apply creative mutation to this design for {task}:
        
        Original: {content}
        
        Introduce one of these mutations:
        1. Add AI/ML capabilities
        2. Incorporate blockchain technology
        3. Use edge computing
        4. Implement quantum-inspired algorithms
        5. Add real-time processing
        6. Include IoT integration
        7. Use microservices architecture
        8. Implement event-driven design
        
        Provide the mutated design with clear improvements.
        """
        
        try:
            response = await self.llm.ainvoke(mutation_prompt)
            return response.get('output', content) if isinstance(response, dict) else str(response)
        except Exception as e:
            # Fallback mutation
            mutations = [
                " with AI enhancement",
                " using blockchain",
                " with edge computing",
                " with quantum optimization",
                " with real-time capabilities"
            ]
            return content + random.choice(mutations)
    
    def _inherit_trait(self, trait1: float, trait2: float) -> float:
        """Inherit trait from parents with some variation"""
        
        # Weighted average with mutation
        inherited = (trait1 + trait2) / 2
        mutation = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, inherited + mutation))
    
    def _update_population(self, offspring: List[CreativeIdea]):
        """Update population with new offspring"""
        
        # Keep best 50% of current population
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        keep_count = self.population_size // 2
        self.population = self.population[:keep_count]
        
        # Add offspring
        self.population.extend(offspring[:self.population_size - keep_count])
    
    def _log_generation_stats(self, generation: int):
        """Log statistics for the generation"""
        
        avg_fitness = np.mean([idea.fitness_score for idea in self.population])
        max_fitness = max([idea.fitness_score for idea in self.population])
        
        stats = {
            'generation': generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'population_size': len(self.population)
        }
        
        self.fitness_history.append(stats)
        
        # Store in brain
        self.brain.add_node(
            node_type=NodeType.LEARNING,
            content=f"Evolutionary generation {generation} stats: {json.dumps(stats, indent=2)}",
            metadata={'evolutionary_generation': generation, 'avg_fitness': avg_fitness}
        )
    
    def _get_best_designs(self, top_k: int = 5) -> List[CreativeIdea]:
        """Get the best designs from the final population"""
        
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        return self.population[:top_k]

# ==================== SWARM INTELLIGENCE SYSTEM ====================

class SwarmIntelligenceSystem:
    """Swarm of creative agents for parallel ideation"""
    
    def __init__(self, brain: ProjectBrain, swarm_size: int = 8):
        self.brain = brain
        self.swarm_size = swarm_size
        self.agents = []
        self.ideas_pool = []
        self.collaboration_network = {}
        self.llm = get_default_model()
        
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        """Initialize the swarm with diverse agents"""
        
        specializations = [
            "AI/ML Specialist",
            "System Architect",
            "Frontend Innovator",
            "Backend Engineer",
            "DevOps Expert",
            "Security Specialist",
            "Performance Optimizer",
            "User Experience Designer"
        ]
        
        creativity_styles = [
            "Analytical",
            "Intuitive",
            "Experimental",
            "Systematic",
            "Disruptive",
            "Incremental",
            "Synthetic",
            "Divergent"
        ]
        
        for i in range(self.swarm_size):
            agent = SwarmAgent(
                id=f"agent_{i}",
                specialization=specializations[i % len(specializations)],
                creativity_style=creativity_styles[i % len(creativity_styles)],
                current_ideas=[],
                collaboration_history=[],
                performance_score=0.5
            )
            self.agents.append(agent)
    
    async def generate_swarm_ideas(self, task: str, rounds: int = 3) -> List[CreativeIdea]:
        """Generate ideas through swarm collaboration"""
        
        print(f"🐝 Starting swarm ideation with {self.swarm_size} agents")
        
        for round_num in range(rounds):
            print(f"🔄 Swarm round {round_num + 1}/{rounds}")
            
            # Individual idea generation
            await self._individual_ideation(task, round_num)
            
            # Collaborative refinement
            await self._collaborative_refinement(round_num)
            
            # Performance evaluation
            self._evaluate_agent_performance()
            
            # Log round stats
            self._log_swarm_round(round_num)
        
        # Synthesize final ideas
        return await self._synthesize_final_ideas(task)
    
    async def _individual_ideation(self, task: str, round_num: int):
        """Each agent generates individual ideas"""
        
        for agent in self.agents:
            idea_prompt = f"""
            You are a {agent.specialization} with {agent.creativity_style} creativity style.
            
            Generate a creative solution for: {task}
            
            Focus on your specialization and use your creativity style.
            Consider how your approach differs from traditional methods.
            
            Provide:
            1. Detailed technical approach
            2. Innovation aspects
            3. Implementation considerations
            4. Potential challenges and solutions
            """
            
            try:
                response = await self.llm.ainvoke(idea_prompt)
                idea_content = response.get('output', '') if isinstance(response, dict) else str(response)
                
                # Create idea
                idea = CreativeIdea(
                    id=f"swarm_agent_{agent.id}_round_{round_num}",
                    content=idea_content,
                    creativity_type=CreativityType.SWARM,
                    novelty_score=random.uniform(0.6, 0.9),
                    feasibility_score=random.uniform(0.5, 0.8),
                    originality_score=random.uniform(0.7, 0.9),
                    tags=[agent.specialization, agent.creativity_style],
                    generation=round_num
                )
                
                agent.current_ideas.append(idea.id)
                self.ideas_pool.append(idea)
                
            except Exception as e:
                print(f"Error in individual ideation for {agent.id}: {e}")
    
    async def _collaborative_refinement(self, round_num: int):
        """Agents collaborate to refine ideas"""
        
        # Pair agents for collaboration
        pairs = self._create_collaboration_pairs()
        
        for agent1, agent2 in pairs:
            # Get recent ideas from both agents
            agent1_ideas = [idea for idea in self.ideas_pool if idea.id in agent1.current_ideas[-2:]]
            agent2_ideas = [idea for idea in self.ideas_pool if idea.id in agent2.current_ideas[-2:]]
            
            if agent1_ideas and agent2_ideas:
                # Collaborative refinement
                refined_idea = await self._collaborate_on_ideas(agent1, agent2, agent1_ideas[0], agent2_ideas[0])
                
                if refined_idea:
                    self.ideas_pool.append(refined_idea)
                    agent1.current_ideas.append(refined_idea.id)
                    agent2.current_ideas.append(refined_idea.id)
                    
                    # Update collaboration history
                    agent1.collaboration_history.append(agent2.id)
                    agent2.collaboration_history.append(agent1.id)
    
    def _create_collaboration_pairs(self) -> List[Tuple[SwarmAgent, SwarmAgent]]:
        """Create pairs of agents for collaboration"""
        
        pairs = []
        agents_copy = self.agents.copy()
        random.shuffle(agents_copy)
        
        for i in range(0, len(agents_copy), 2):
            if i + 1 < len(agents_copy):
                pairs.append((agents_copy[i], agents_copy[i + 1]))
        
        return pairs
    
    async def _collaborate_on_ideas(self, agent1: SwarmAgent, agent2: SwarmAgent, 
                                  idea1: CreativeIdea, idea2: CreativeIdea) -> Optional[CreativeIdea]:
        """Two agents collaborate to refine their ideas"""
        
        collaboration_prompt = f"""
        Two specialists are collaborating to improve their ideas:
        
        {agent1.specialization} ({agent1.creativity_style} style):
        {idea1.content}
        
        {agent2.specialization} ({agent2.creativity_style} style):
        {idea2.content}
        
        Create a refined, collaborative solution that:
        1. Combines the best aspects of both approaches
        2. Leverages both specializations
        3. Introduces new collaborative innovations
        4. Addresses potential conflicts between approaches
        5. Provides a unified, superior solution
        
        Provide a detailed, integrated design.
        """
        
        try:
            response = await self.llm.ainvoke(collaboration_prompt)
            refined_content = response.get('output', '') if isinstance(response, dict) else str(response)
            
            if len(refined_content) > 100:  # Only accept substantial refinements
                return CreativeIdea(
                    id=f"collaboration_{agent1.id}_{agent2.id}_{datetime.now().timestamp()}",
                    content=refined_content,
                    creativity_type=CreativityType.SWARM,
                    novelty_score=(idea1.novelty_score + idea2.novelty_score) / 2 + 0.1,
                    feasibility_score=(idea1.feasibility_score + idea2.feasibility_score) / 2,
                    originality_score=(idea1.originality_score + idea2.originality_score) / 2 + 0.1,
                    tags=list(set(idea1.tags + idea2.tags + ['collaborative'])),
                    parent_ids=[idea1.id, idea2.id],
                    generation=max(idea1.generation, idea2.generation) + 1
                )
        
        except Exception as e:
            print(f"Error in collaboration: {e}")
        
        return None
    
    def _evaluate_agent_performance(self):
        """Evaluate performance of each agent"""
        
        for agent in self.agents:
            # Calculate performance based on idea quality and collaboration
            recent_ideas = [idea for idea in self.ideas_pool if idea.id in agent.current_ideas[-3:]]
            
            if recent_ideas:
                avg_quality = np.mean([
                    idea.novelty_score + idea.feasibility_score + idea.originality_score
                    for idea in recent_ideas
                ]) / 3
                
                collaboration_bonus = len(agent.collaboration_history) * 0.05
                
                agent.performance_score = min(1.0, avg_quality + collaboration_bonus)
    
    def _log_swarm_round(self, round_num: int):
        """Log statistics for the swarm round"""
        
        stats = {
            'round': round_num,
            'total_ideas': len(self.ideas_pool),
            'avg_performance': np.mean([agent.performance_score for agent in self.agents]),
            'collaborations': sum(len(agent.collaboration_history) for agent in self.agents)
        }
        
        # Store in brain
        self.brain.add_node(
            node_type=NodeType.LEARNING,
            content=f"Swarm round {round_num} stats: {json.dumps(stats, indent=2)}",
            metadata={'swarm_round': round_num, 'total_ideas': stats['total_ideas']}
        )
    
    async def _synthesize_final_ideas(self, task: str) -> List[CreativeIdea]:
        """Synthesize final ideas from the swarm"""
        
        # Get top ideas by quality
        self.ideas_pool.sort(key=lambda x: x.novelty_score + x.feasibility_score + x.originality_score, reverse=True)
        top_ideas = self.ideas_pool[:5]
        
        # Synthesize into final concepts
        synthesis_prompt = f"""
        Synthesize these top creative ideas for {task}:
        
        {chr(10).join([f"Idea {i+1}: {idea.content[:200]}..." for i, idea in enumerate(top_ideas)])}
        
        Create 3 final, innovative concepts that:
        1. Combine the best elements from multiple ideas
        2. Introduce novel synthesis approaches
        3. Are technically feasible and innovative
        4. Represent breakthrough thinking
        
        Provide detailed descriptions for each concept.
        """
        
        try:
            response = await self.llm.ainvoke(synthesis_prompt)
            synthesis_content = response.get('output', '') if isinstance(response, dict) else str(response)
            
            # Create final synthesized ideas
            final_ideas = []
            for i in range(3):
                idea = CreativeIdea(
                    id=f"synthesis_final_{i}",
                    content=f"Synthesized concept {i+1}: {synthesis_content}",
                    creativity_type=CreativityType.SWARM,
                    novelty_score=0.9,
                    feasibility_score=0.8,
                    originality_score=0.9,
                    tags=['synthesized', 'swarm_final'],
                    parent_ids=[idea.id for idea in top_ideas],
                    generation=len(self.agents)
                )
                final_ideas.append(idea)
            
            return final_ideas
            
        except Exception as e:
            print(f"Error in synthesis: {e}")
            return top_ideas[:3]

# ==================== NOVELTY EXPLORATION SYSTEM ====================

class NoveltyExplorationSystem:
    """System for exploring novel and unconventional approaches"""
    
    def __init__(self, brain: ProjectBrain):
        self.brain = brain
        self.llm = get_default_model()
        self.exploration_history = []
        
    async def explore_novel_approaches(self, task: str, exploration_depth: int = 3) -> List[CreativeIdea]:
        """Explore novel and unconventional approaches to a task"""
        
        print(f"🔍 Exploring novel approaches for: {task}")
        
        novel_ideas = []
        
        # Explore different novelty dimensions
        novelty_dimensions = [
            "Synesthetic approaches (cross-sensory design)",
            "Quantum-inspired algorithms",
            "Bio-inspired computing",
            "Emergent behavior systems",
            "Non-deterministic solutions",
            "Temporal paradox approaches",
            "Dimensional reduction techniques",
            "Chaos theory applications"
        ]
        
        for dimension in novelty_dimensions[:exploration_depth]:
            idea = await self._explore_dimension(task, dimension)
            if idea:
                novel_ideas.append(idea)
        
        # Synthesize novel insights
        synthesis = await self._synthesize_novel_insights(task, novel_ideas)
        if synthesis:
            novel_ideas.append(synthesis)
        
        return novel_ideas
    
    async def _explore_dimension(self, task: str, dimension: str) -> Optional[CreativeIdea]:
        """Explore a specific novelty dimension"""
        
        exploration_prompt = f"""
        Explore {dimension} for solving: {task}
        
        Consider:
        1. How can this unconventional approach be applied?
        2. What unique advantages does it offer?
        3. What are the implementation challenges?
        4. How does it differ from traditional approaches?
        5. What novel insights does it provide?
        
        Provide a detailed, innovative solution that leverages this dimension.
        Be creative and think outside conventional boundaries.
        """
        
        try:
            response = await self.llm.ainvoke(exploration_prompt)
            content = response.get('output', '') if isinstance(response, dict) else str(response)
            
            if len(content) > 100:
                return CreativeIdea(
                    id=f"novelty_{hashlib.md5(dimension.encode()).hexdigest()[:8]}",
                    content=content,
                    creativity_type=CreativityType.SYNESTHETIC,
                    novelty_score=0.95,
                    feasibility_score=0.6,  # Novel approaches may be less feasible
                    originality_score=0.9,
                    tags=['novelty', dimension.lower().replace(' ', '_')],
                    generation=0
                )
        
        except Exception as e:
            print(f"Error exploring dimension {dimension}: {e}")
        
        return None
    
    async def _synthesize_novel_insights(self, task: str, novel_ideas: List[CreativeIdea]) -> Optional[CreativeIdea]:
        """Synthesize insights from multiple novel approaches"""
        
        if not novel_ideas:
            return None
        
        synthesis_prompt = f"""
        Synthesize insights from these novel approaches for {task}:
        
        {chr(10).join([f"Approach {i+1}: {idea.content[:300]}..." for i, idea in enumerate(novel_ideas)])}
        
        Create a breakthrough synthesis that:
        1. Combines the most promising elements from each approach
        2. Introduces new meta-level insights
        3. Creates a unified novel framework
        4. Maintains feasibility while maximizing innovation
        
        Provide a comprehensive, innovative solution.
        """
        
        try:
            response = await self.llm.ainvoke(synthesis_prompt)
            content = response.get('output', '') if isinstance(response, dict) else str(response)
            
            if len(content) > 100:
                return CreativeIdea(
                    id=f"novelty_synthesis_{datetime.now().timestamp()}",
                    content=content,
                    creativity_type=CreativityType.SYNESTHETIC,
                    novelty_score=0.98,
                    feasibility_score=0.7,
                    originality_score=0.95,
                    tags=['novelty_synthesis', 'breakthrough'],
                    parent_ids=[idea.id for idea in novel_ideas],
                    generation=1
                )
        
        except Exception as e:
            print(f"Error in novelty synthesis: {e}")
        
        return None

# ==================== CREATIVITY ENGINE MAIN CLASS ====================

class CreativityEngine:
    """Main creativity engine coordinating all creative approaches"""
    
    def __init__(self, brain: ProjectBrain):
        self.brain = brain
        self.evolutionary_generator = EvolutionaryDesignGenerator(brain)
        self.swarm_system = SwarmIntelligenceSystem(brain)
        self.novelty_explorer = NoveltyExplorationSystem(brain)
        self.all_ideas = []
        
    async def generate_creative_solutions(self, task: str, 
                                        use_evolutionary: bool = True,
                                        use_swarm: bool = True,
                                        use_novelty: bool = True) -> Dict[str, List[CreativeIdea]]:
        """Generate creative solutions using multiple approaches"""
        
        results = {}
        
        # Evolutionary design
        if use_evolutionary:
            print("🧬 Starting evolutionary design...")
            evolutionary_ideas = await self.evolutionary_generator.evolve_design(task)
            results['evolutionary'] = evolutionary_ideas
            self.all_ideas.extend(evolutionary_ideas)
        
        # Swarm intelligence
        if use_swarm:
            print("🐝 Starting swarm intelligence...")
            swarm_ideas = await self.swarm_system.generate_swarm_ideas(task)
            results['swarm'] = swarm_ideas
            self.all_ideas.extend(swarm_ideas)
        
        # Novelty exploration
        if use_novelty:
            print("🔍 Starting novelty exploration...")
            novelty_ideas = await self.novelty_explorer.explore_novel_approaches(task)
            results['novelty'] = novelty_ideas
            self.all_ideas.extend(novelty_ideas)
        
        # Cross-synthesis
        if len(results) > 1:
            print("🔄 Performing cross-synthesis...")
            cross_synthesis = await self._cross_synthesize_approaches(task, results)
            results['cross_synthesis'] = cross_synthesis
            self.all_ideas.extend(cross_synthesis)
        
        # Store all ideas in brain
        self._store_ideas_in_brain()
        
        return results
    
    async def _cross_synthesize_approaches(self, task: str, 
                                         approach_results: Dict[str, List[CreativeIdea]]) -> List[CreativeIdea]:
        """Synthesize ideas across different creative approaches"""
        
        # Get best ideas from each approach
        best_ideas = []
        for approach, ideas in approach_results.items():
            if ideas:
                # Sort by quality and take top 2
                ideas.sort(key=lambda x: x.novelty_score + x.feasibility_score + x.originality_score, reverse=True)
                best_ideas.extend(ideas[:2])
        
        if len(best_ideas) < 2:
            return []
        
        synthesis_prompt = f"""
        Synthesize creative approaches for {task}:
        
        {chr(10).join([f"{idea.creativity_type.value} approach: {idea.content[:200]}..." for idea in best_ideas])}
        
        Create breakthrough solutions that:
        1. Combine the best elements from different creative approaches
        2. Create new meta-patterns and insights
        3. Maintain high novelty while improving feasibility
        4. Represent truly innovative thinking
        
        Provide 2-3 comprehensive, innovative solutions.
        """
        
        try:
            response = await self.llm.ainvoke(synthesis_prompt)
            content = response.get('output', '') if isinstance(response, dict) else str(response)
            
            # Create synthesis ideas
            synthesis_ideas = []
            for i in range(2):
                idea = CreativeIdea(
                    id=f"cross_synthesis_{i}_{datetime.now().timestamp()}",
                    content=f"Cross-synthesis solution {i+1}: {content}",
                    creativity_type=CreativityType.COMBINATORIAL,
                    novelty_score=0.95,
                    feasibility_score=0.8,
                    originality_score=0.95,
                    tags=['cross_synthesis', 'breakthrough'],
                    parent_ids=[idea.id for idea in best_ideas],
                    generation=2
                )
                synthesis_ideas.append(idea)
            
            return synthesis_ideas
            
        except Exception as e:
            print(f"Error in cross-synthesis: {e}")
            return []
    
    def _store_ideas_in_brain(self):
        """Store all generated ideas in the brain"""
        
        for idea in self.all_ideas:
            self.brain.add_node(
                node_type=NodeType.LEARNING,
                content=f"Creative idea: {idea.content[:500]}...",
                metadata={
                    'idea_id': idea.id,
                    'creativity_type': idea.creativity_type.value,
                    'novelty_score': idea.novelty_score,
                    'feasibility_score': idea.feasibility_score,
                    'originality_score': idea.originality_score,
                    'tags': idea.tags
                }
            )
    
    def get_creativity_stats(self) -> Dict[str, Any]:
        """Get creativity engine statistics"""
        
        total_ideas = len(self.all_ideas)
        
        if total_ideas == 0:
            return {'total_ideas': 0}
        
        avg_novelty = np.mean([idea.novelty_score for idea in self.all_ideas])
        avg_feasibility = np.mean([idea.feasibility_score for idea in self.all_ideas])
        avg_originality = np.mean([idea.originality_score for idea in self.all_ideas])
        
        creativity_distribution = {}
        for creativity_type in CreativityType:
            count = len([idea for idea in self.all_ideas if idea.creativity_type == creativity_type])
            creativity_distribution[creativity_type.value] = count
        
        return {
            'total_ideas': total_ideas,
            'avg_novelty_score': avg_novelty,
            'avg_feasibility_score': avg_feasibility,
            'avg_originality_score': avg_originality,
            'creativity_distribution': creativity_distribution,
            'top_ideas': [
                {
                    'id': idea.id,
                    'content': idea.content[:100] + "...",
                    'novelty_score': idea.novelty_score,
                    'feasibility_score': idea.feasibility_score,
                    'originality_score': idea.originality_score
                }
                for idea in sorted(self.all_ideas, 
                                 key=lambda x: x.novelty_score + x.feasibility_score + x.originality_score, 
                                 reverse=True)[:5]
            ]
        } 