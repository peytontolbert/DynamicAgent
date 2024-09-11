import json
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph

class MetaCognitiveKnowledgeSystem:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.self_monitoring_data = {}
        self.knowledge_graph = knowledge_graph
        self.llm = ChatGPT()

    def log_performance(self, task_name, performance_data):
        self.self_monitoring_data[task_name] = performance_data
        self.knowledge_graph.add_or_update_node("PerformanceData", {"task_name": task_name, "performance_data": performance_data})

    def get_performance(self, task_name):
        return self.self_monitoring_data.get(task_name, {})

    def export(self, path):
        with open(path, 'w') as f:
            json.dump(self.self_monitoring_data, f)

    def import_data(self, path):
        with open(path, 'r') as f:
            self.self_monitoring_data = json.load(f)
            for task_name, performance_data in self.self_monitoring_data.items():
                self.knowledge_graph.add_or_update_node("PerformanceData", {"task_name": task_name, "performance_data": performance_data})

    async def enhance_performance(self, task_name):
        prompt = f"""
        Analyze the performance of the following task and provide detailed feedback and improvements:
        Task: {task_name}
        """
        performance_feedback = await self.llm.chat_with_ollama("You are a performance analysis expert.", prompt)
        self.log_performance(task_name, performance_feedback.strip())
        return performance_feedback.strip()

    async def generalize_knowledge(self, concept: str):
        prompt = f"""
        Analyze the following concept and provide a generalized understanding that can be applied across different scenarios:
        Concept: {concept}
        """
        generalized_knowledge = await self.llm.chat_with_ollama("You are an expert in abstract concepts and generalization.", prompt)
        self.knowledge_graph.add_or_update_node("GeneralizedKnowledge", {"concept": concept, "generalized_knowledge": generalized_knowledge.strip()})
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
