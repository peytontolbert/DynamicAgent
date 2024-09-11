import json
import time
from typing import Dict, Any, List
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.embedding_manager import EmbeddingManager
from app.knowledge.community_manager import CommunityManager


class MetaCognitiveKnowledgeSystem:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.self_monitoring_data = {}
        self.knowledge_graph = knowledge_graph
        self.llm = ChatGPT()
        self.embedding_manager = EmbeddingManager()
        self.community_manager = CommunityManager(
            knowledge_graph, self.embedding_manager
        )

    async def initialize(self):
        await self.community_manager.initialize()

    async def log_performance(self, task_name, performance_data):
        self.self_monitoring_data[task_name] = performance_data
        await self.knowledge_graph.add_or_update_node(
            "PerformanceData",
            {"task_name": task_name, "performance_data": performance_data},
        )
        await self.community_manager.update_knowledge(
            {
                "label": "PerformanceData",
                "id": task_name,
                "content": json.dumps(performance_data),
            }
        )

    async def get_performance(self, task_name):
        return self.self_monitoring_data.get(task_name, {})

    async def export(self, path):
        with open(path, "w") as f:
            json.dump(self.self_monitoring_data, f)

    async def import_data(self, path):
        with open(path, "r") as f:
            self.self_monitoring_data = json.load(f)
            for task_name, performance_data in self.self_monitoring_data.items():
                await self.knowledge_graph.add_or_update_node(
                    "PerformanceData",
                    {"task_name": task_name, "performance_data": performance_data},
                )
                await self.community_manager.update_knowledge(
                    {
                        "label": "PerformanceData",
                        "id": task_name,
                        "content": json.dumps(performance_data),
                    }
                )

    async def enhance_performance(self, task_name):
        performance_data = await self.get_performance(task_name)
        prompt = f"""
        Analyze the performance of the following task and provide detailed feedback and improvements:
        Task: {task_name}
        Performance Data: {json.dumps(performance_data)}
        """
        performance_feedback = await self.llm.chat_with_ollama(
            "You are a performance analysis expert.", prompt
        )
        await self.log_performance(
            task_name, {"feedback": performance_feedback.strip()}
        )
        return performance_feedback.strip()

    async def generalize_knowledge(self, concept: str):
        prompt = f"""
        Analyze the following concept and provide a generalized understanding that can be applied across different scenarios:
        Concept: {concept}
        """
        generalized_knowledge = await self.llm.chat_with_ollama(
            "You are an expert in abstract concepts and generalization.", prompt
        )
        await self.knowledge_graph.add_or_update_node(
            "GeneralizedKnowledge",
            {
                "concept": concept,
                "generalized_knowledge": generalized_knowledge.strip(),
            },
        )
        await self.community_manager.update_knowledge(
            {
                "label": "GeneralizedKnowledge",
                "id": concept,
                "content": generalized_knowledge.strip(),
            }
        )
        return generalized_knowledge.strip()

    async def get_generalized_knowledge(self, concept: str):
        node = await self.knowledge_graph.get_node(
            "GeneralizedKnowledge", {"concept": concept}
        )
        return node.get("generalized_knowledge") if node else None

    async def extract_concepts(self, task: str):
        prompt = f"""
        Extract key concepts from the following task:
        Task: {task}
        """
        concepts = await self.llm.chat_with_ollama(
            "You are an expert in concept extraction.", prompt
        )
        return concepts.strip()

    async def generate_thoughts(self, task: str, knowledge: Dict[str, Any]) -> str:
        # Retrieve relevant knowledge from communities
        community_knowledge = await self.get_relevant_community_knowledge(task)

        prompt = f"""
        Generate thoughts for the following task and context:
        Task: {task}
        
        Available Knowledge:
        Procedural: {knowledge['procedural_info']}
        Conceptual: {knowledge['related_concepts']}
        Contextual: {knowledge['context_info']}
        Episodic: {knowledge['past_experiences']}
        Semantic: {knowledge['language_understanding']}
        Community Knowledge: {community_knowledge}
        
        Provide your thoughts in the following format:
        Thoughts: <Your analysis, reasoning, and approach>
        """
        thoughts = await self.llm.chat_with_ollama(
            "You are an expert in metacognition and task analysis.", prompt
        )

        # Store the generated thoughts
        thought_node = {
            "label": "Thoughts",
            "id": f"thought_{int(time.time())}",
            "task": task,
            "thoughts": thoughts.strip(),
            "timestamp": time.time(),
        }
        await self.knowledge_graph.add_or_update_node(**thought_node)
        await self.community_manager.update_knowledge(thought_node)

        return thoughts.strip()

    async def get_relevant_community_knowledge(self, query: str) -> str:
        relevant_communities = await self.community_manager.get_relevant_communities(
            query
        )
        community_knowledge = []
        for community_id in relevant_communities:
            summary = await self.community_manager.get_community_summary(community_id)
            community_knowledge.append(f"Community {community_id}: {summary}")
        return "\n".join(community_knowledge)

    async def query_knowledge(self, query: str) -> str:
        community_knowledge = await self.get_relevant_community_knowledge(query)
        relevant_thoughts = await self.retrieve_relevant_thoughts(query)

        prompt = f"""
        Answer the following query using the provided community knowledge and relevant thoughts:
        Query: {query}
        
        Community Knowledge:
        {community_knowledge}
        
        Relevant Thoughts:
        {' '.join(relevant_thoughts)}
        
        Provide a comprehensive answer that integrates information from both sources.
        """

        answer = await self.llm.chat_with_ollama(
            "You are an expert in knowledge integration and query answering.", prompt
        )
        return answer.strip()

    async def update_knowledge(self, new_knowledge: Dict[str, Any]):
        await self.community_manager.update_knowledge(new_knowledge)
        await self.reassign_communities()

    async def reassign_communities(self):
        await self.community_manager.detect_communities()
        await self.community_manager.summarize_communities()

    async def get_community_insights(self) -> List[Dict[str, Any]]:
        communities = await self.community_manager.get_all_communities()
        insights = []
        for community_id, nodes in communities.items():
            summary = await self.community_manager.get_community_summary(community_id)
            central_nodes = await self.community_manager.get_central_nodes(community_id)
            insights.append(
                {
                    "community_id": community_id,
                    "summary": summary,
                    "central_nodes": central_nodes,
                    "node_count": len(nodes),
                }
            )
        return insights

    async def cross_community_analysis(self, query: str) -> str:
        community_insights = await self.get_community_insights()

        prompt = f"""
        Perform a cross-community analysis based on the following query and community insights:
        Query: {query}
        
        Community Insights:
        {json.dumps(community_insights, indent=2)}
        
        Provide an analysis that:
        1. Identifies relationships between different communities
        2. Highlights potential knowledge gaps or areas for further exploration
        3. Suggests ways to leverage knowledge across communities to address the query
        """

        analysis = await self.llm.chat_with_ollama(
            "You are an expert in cross-community knowledge analysis.", prompt
        )
        return analysis.strip()

    async def generate_meta_knowledge(self) -> str:
        community_insights = await self.get_community_insights()
        recent_thoughts = await self.retrieve_recent_thoughts(n=10)

        prompt = f"""
        Generate meta-knowledge based on the following community insights and recent thoughts:
        
        Community Insights:
        {json.dumps(community_insights, indent=2)}
        
        Recent Thoughts:
        {json.dumps(recent_thoughts, indent=2)}
        
        Provide meta-knowledge that:
        1. Identifies patterns in knowledge organization and thought processes
        2. Suggests improvements in knowledge management and utilization
        3. Proposes strategies for more effective learning and knowledge application
        """

        meta_knowledge = await self.llm.chat_with_ollama(
            "You are an expert in meta-cognition and knowledge management.", prompt
        )
        return meta_knowledge.strip()

    async def retrieve_relevant_thoughts(self, task: str) -> List[str]:
        similar_thoughts = await self.knowledge_graph.get_similar_nodes(
            task, label="Thoughts", k=3
        )
        return [thought["thoughts"] for thought, _ in similar_thoughts]

    async def retrieve_recent_thoughts(self, n: int = 5) -> List[Dict[str, Any]]:
        current_time = time.time()
        recent_thoughts = await self.knowledge_graph.get_nodes(
            label="Thoughts",
            filter_func=lambda node: current_time - node.get("timestamp", 0)
            <= 24 * 60 * 60,  # Within last 24 hours
            sort_key=lambda node: node.get("timestamp", 0),
            reverse=True,
            limit=n,
        )
        return [
            {
                "task": thought.get("task"),
                "thoughts": thought.get("thoughts"),
                "timestamp": thought.get("timestamp"),
            }
            for thought in recent_thoughts
        ]

    async def get_community_summaries(self) -> Dict[int, str]:
        return self.community_manager.community_summaries

    async def add_or_update_knowledge(self, label: str, data: Dict[str, Any]):
        await self.knowledge_graph.add_or_update_node(label, data)
        await self.community_manager.update_knowledge(
            {
                "label": label,
                "id": data.get("id") or data.get("task_name"),
                "content": json.dumps(data),
            }
        )
        await self.reassign_communities()

    async def get_relevant_knowledge(self, query: str) -> Dict[str, Any]:
        community_knowledge = await self.get_relevant_community_knowledge(query)
        relevant_thoughts = await self.retrieve_relevant_thoughts(query)
        graph_knowledge = await self.knowledge_graph.get_relevant_knowledge(query)

        return {
            "community_knowledge": community_knowledge,
            "relevant_thoughts": relevant_thoughts,
            "graph_knowledge": graph_knowledge,
        }
