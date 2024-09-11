import json
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph
from typing import Dict, Any, List
import time

class MetaCognitiveKnowledgeSystem:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.self_monitoring_data = {}
        self.knowledge_graph = knowledge_graph
        self.llm = ChatGPT()

    async def log_performance(self, task_name, performance_data):
        self.self_monitoring_data[task_name] = performance_data
        await self.knowledge_graph.add_or_update_node("PerformanceData", {"task_name": task_name, "performance_data": performance_data})

    async def get_performance(self, task_name):
        return self.self_monitoring_data.get(task_name, {})

    async def export(self, path):
        with open(path, 'w') as f:
            json.dump(self.self_monitoring_data, f)

    async def import_data(self, path):
        with open(path, 'r') as f:
            self.self_monitoring_data = json.load(f)
            for task_name, performance_data in self.self_monitoring_data.items():
                await self.knowledge_graph.add_or_update_node("PerformanceData", {"task_name": task_name, "performance_data": performance_data})

    async def enhance_performance(self, task_name):
        prompt = f"""
        Analyze the performance of the following task and provide detailed feedback and improvements:
        Task: {task_name}
        """
        performance_feedback = await self.llm.chat_with_ollama("You are a performance analysis expert.", prompt)
        await self.log_performance(task_name, performance_feedback.strip())
        return performance_feedback.strip()

    async def generalize_knowledge(self, concept: str):
        prompt = f"""
        Analyze the following concept and provide a generalized understanding that can be applied across different scenarios:
        Concept: {concept}
        """
        generalized_knowledge = await self.llm.chat_with_ollama("You are an expert in abstract concepts and generalization.", prompt)
        await self.knowledge_graph.add_or_update_node("GeneralizedKnowledge", {"concept": concept, "generalized_knowledge": generalized_knowledge.strip()})
        return generalized_knowledge.strip()

    async def get_generalized_knowledge(self, concept: str):
        node = await self.knowledge_graph.get_node("GeneralizedKnowledge", {"concept": concept})
        return node.get("generalized_knowledge") if node else None

    async def extract_concepts(self, task: str):
        prompt = f"""
        Extract key concepts from the following task:
        Task: {task}
        """
        concepts = await self.llm.chat_with_ollama("You are an expert in concept extraction.", prompt)
        return concepts.strip()

    async def generate_thoughts(self, task: str, context: str, knowledge: Dict[str, Any]) -> str:
        prompt = f"""
        Generate thoughts for the following task and context:
        Task: {task}
        Context: {context}
        
        Available Knowledge:
        Procedural: {knowledge['procedural_info']}
        Conceptual: {knowledge['related_concepts']}
        Contextual: {knowledge['context_info']}
        Episodic: {knowledge['past_experiences']}
        Semantic: {knowledge['language_understanding']}
        
        Provide your thoughts in the following format:
        Thoughts: <Your analysis, reasoning, and approach>
        """
        thoughts = await self.llm.chat_with_ollama("You are an expert in metacognition and task analysis.", prompt)
        
        # Store the generated thoughts
        await self.knowledge_graph.add_or_update_node("Thoughts", {
            "task": task,
            "thoughts": thoughts.strip(),
            "timestamp": time.time()
        })
        
        return thoughts.strip()

    async def retrieve_relevant_thoughts(self, task: str) -> List[str]:
        similar_thoughts = await self.knowledge_graph.get_similar_nodes(task, label="Thoughts", k=3)
        return [thought['thoughts'] for thought, _ in similar_thoughts]

    async def retrieve_recent_thoughts(self, n: int = 5) -> List[Dict[str, Any]]:
        current_time = time.time()
        recent_thoughts = await self.knowledge_graph.get_nodes(
            label="Thoughts",
            filter_func=lambda node: current_time - node.get("timestamp", 0) <= 24 * 60 * 60,  # Within last 24 hours
            sort_key=lambda node: node.get("timestamp", 0),
            reverse=True,
            limit=n
        )
        return [
            {
                "task": thought.get("task"),
                "thoughts": thought.get("thoughts"),
                "timestamp": thought.get("timestamp")
            }
            for thought in recent_thoughts
        ]
