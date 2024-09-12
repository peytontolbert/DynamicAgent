from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.procedural_knowledge import ProceduralKnowledgeSystem
from app.knowledge.episodic_knowledge import EpisodicKnowledgeSystem, Episode
from app.knowledge.conceptual_knowledge import ConceptualKnowledgeSystem
from app.knowledge.contextual_knowledge import ContextualKnowledgeSystem
from app.knowledge.meta_cognitive_knowledge import MetaCognitiveKnowledgeSystem
from app.knowledge.semantic_knowledge import SemanticKnowledgeSystem
from app.knowledge.embedding_manager import EmbeddingManager
from app.knowledge.community_manager import CommunityManager
from app.chat_with_ollama import ChatGPT
from typing import Dict, Any, List, Tuple
import re
from app.knowledge.spatial_knowledge import SpatialKnowledgeSystem
from app.knowledge.temporal_knowledge import TemporalKnowledgeSystem
from app.agent.agent_thoughts import AgentThoughts


class AgentKnowledgeInterface:
    def __init__(self, uri, user, password, base_path):
        self.knowledge_graph = KnowledgeGraph(uri, user, password)
        self.embedding_manager = EmbeddingManager()
        self.llm = ChatGPT()
        self.procedural_memory = ProceduralKnowledgeSystem(
            self.knowledge_graph, self.embedding_manager
        )
        self.episodic_memory = EpisodicKnowledgeSystem(
            self.knowledge_graph, self.embedding_manager
        )
        self.conceptual_knowledge = ConceptualKnowledgeSystem(self.knowledge_graph)
        self.community_manager = CommunityManager(
            self.knowledge_graph, self.embedding_manager
        )
        self.contextual_knowledge = ContextualKnowledgeSystem(
            self.knowledge_graph, self.community_manager
        )
        self.meta_cognitive = MetaCognitiveKnowledgeSystem(self.knowledge_graph)
        self.semantic_knowledge = SemanticKnowledgeSystem(
            self.knowledge_graph, self.embedding_manager
        )
        self.spatial_knowledge = SpatialKnowledgeSystem(self.knowledge_graph)
        self.temporal_knowledge = TemporalKnowledgeSystem(self.knowledge_graph)
        self.agent_thoughts = AgentThoughts()

    async def gather_knowledge(self, task: str) -> dict:
        context = await self.agent_knowledge_interface.contextual_knowledge.get_context(
            task
        )
        related_episodes = await self.episodic_memory.remember_related_episodes(task)
        recent_episodes = await self.episodic_memory.remember_recent_episodes(5)
        interpreted_task = await self.semantic_knowledge.retrieve_language_meaning(task)
        if not interpreted_task:
            interpreted_task = (
                await self.semantic_knowledge.enhance_language_understanding(task)
            )

        # Update procedural knowledge retrieval
        procedural_info = await self.procedural_memory.retrieve_relevant_tool_usage(
            task
        )
        tool_insights = await self.procedural_memory.get_tool_insights(task)

        related_concepts = await self.conceptual_knowledge.get_related_concepts(task)
        performance_data = await self.meta_cognitive.get_relevant_knowledge(task)
        generalized_knowledge = await self.meta_cognitive.get_generalized_knowledge(
            related_concepts
        )

        spatial_info = await self.spatial_knowledge.get_spatial_data(task)
        temporal_info = await self.temporal_knowledge.get_temporal_data(task)

        return {
            "context_info": context,
            "related_concepts": related_concepts,
            "generalized_knowledge": generalized_knowledge,
            "performance_data": performance_data,
            "interpreted_task": interpreted_task,
            "related_episodes": related_episodes,
            "recent_episodes": recent_episodes,
            "procedural_info": procedural_info,
            "tool_insights": tool_insights,
            "spatial_info": spatial_info,
            "temporal_info": temporal_info,
        }

    async def update_knowledge_step(
        self,
        task: str,
        result: str,
        action: str,
        thoughts: str,
        action_thoughts: str,
    ):
        await self.episodic_memory.log_task(task, result, thoughts, action_thoughts)
        await self.meta_cognitive.log_performance(
            task,
            {"result": result, "action": action, "action_thoughts": action_thoughts},
        )

        if action == "code_execute":
            insights, tool_usage = (
                await self.procedural_memory.enhance_procedural_knowledge(task, result)
            )

        episode = Episode(
            thoughts={"task": task, "thoughts": thoughts},
            action={"type": action, "details": action_thoughts},
            result=result,
            summary=await self._generate_episode_summary(
                task, thoughts, action, result
            ),
        )
        await self.episodic_memory.memorize_episode(episode)

        concepts = await self.meta_cognitive.extract_concepts(task)
        return concepts

    async def update_knowledge_complete(self, task: str):
        concepts = await self.meta_cognitive.extract_concepts(task)
        performance_data = await self.meta_cognitive.log_performance(task)
        generalized_knowledge = await self.meta_cognitive.generalize_knowledge(
            performance_data
        )
        await self.episodic_memory.organize_episodes()
        await self.episodic_memory.generate_hierarchical_summary()
        await self.conceptual_knowledge.update_concept_relations(concepts)
        await self.contextual_knowledge.update_context(task)
        await self.semantic_knowledge.update_language_understanding(task)
        await self.procedural_memory.enhance_tool_usage(task)

    async def _generate_episode_summary(
        self, task: str, thoughts: str, action: str, result: str
    ) -> str:
        prompt = f"""
        Summarize the following episode:
        Task: {task}
        Thoughts: {thoughts}
        Action: {action}
        Result: {result}
        
        Provide a concise summary that captures the key points of this episode.
        """
        summary = await self.llm.chat_with_ollama(
            "You are an episode summarizer.", prompt
        )
        return summary.strip()

    async def decide_action(
        self, task: str, knowledge, thoughts: str
    ) -> Tuple[str, str]:

        prompt = f"""
        Analyze the following task, thoughts, and knowledge to decide whether to use the 'respond' or 'code_execute' action:

        Task: {task}
        Thoughts: {thoughts}
        
        Knowledge Inputs:
        {knowledge}
        
        Consider the related episodes and community knowledge when making your decision. How do past experiences and collective knowledge inform the best action for this task?
        
        Provide your decision and additional thoughts in the following format:
        Decision: <respond or code_execute>
        Action Thoughts: <Your reasoning for this decision, including how past episodes and community knowledge influenced your choice>
        """
        response = await self.llm.chat_with_ollama(
            "You are a task analysis and decision-making expert.", prompt
        )
        decision, action_thoughts = self._extract_decision_and_thoughts(response)
        return decision.strip().lower(), action_thoughts.strip()

    def _format_episodic_context(self, episodes: List[Dict[str, Any]]) -> str:
        formatted_episodes = []
        for episode in episodes:
            formatted_episode = f"""
            Summary: {episode['summary']}
            Similarity: {episode['similarity']}
            Action: {episode['action']['type']}
            Result: {episode['result']}
            """
            formatted_episodes.append(formatted_episode.strip())
        return "\n\n".join(formatted_episodes)

    def _extract_decision_and_thoughts(self, response: str) -> Tuple[str, str]:
        decision_match = re.search(
            r"Decision:\s*(respond|code_execute)", response, re.IGNORECASE
        )
        thoughts_match = re.search(r"Action Thoughts:(.*)", response, re.DOTALL)

        decision = decision_match.group(1) if decision_match else ""
        thoughts = thoughts_match.group(1).strip() if thoughts_match else ""

        return decision, thoughts

    async def update_episode_relevance(
        self, episode_id: str, task: str, was_helpful: bool
    ):
        await self.episodic_memory.update_episode_relevance(
            episode_id, task, was_helpful
        )

    async def generate_response(
        self, task: str, thoughts: str, action_thoughts: str
    ) -> str:
        knowledge = await self.gather_knowledge(task, "")  # Empty context for now
        episodic_context = self._format_episodic_context(
            knowledge.get("related_episodes", [])
        )
        community_context = knowledge.get("community_knowledge", "")

        prompt = f"""
        Task: {task}
        Thoughts: {thoughts}
        Action Thoughts: {action_thoughts}
        Interpreted understanding: {knowledge['interpreted_task']}
        Related concepts: {knowledge['related_concepts']}
        
        Related Episodes:
        {episodic_context}
        
        Community Knowledge:
        {community_context}
        
        Consider the related episodes and community knowledge when formulating your response. How do past experiences and collective knowledge inform your approach to this task?
        
        Provide a response or question for clarification, taking into account the episodic and community knowledge.
        """
        response = await self.llm.chat_with_ollama(
            "You are a knowledgeable assistant with access to past experiences and community knowledge.",
            prompt,
        )
        return response.strip()
