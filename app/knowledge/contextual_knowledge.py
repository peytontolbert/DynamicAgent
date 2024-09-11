import json
import time
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.embedding_manager import EmbeddingManager



"""
This system provides an understanding of the context in which knowledge should be applied. 
It’s essential for the AGI to recognize the relevance of certain information or actions based on the current situation.
 """
class ContextualKnowledgeSystem:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.llm = ChatGPT()
        self.embedding_manager = EmbeddingManager()

    async def get_context(self, task: str):
        # Try to retrieve existing context
        existing_context = await self.retrieve_context(task)
        
        if existing_context:
            # If context exists, update it
            return await self.update_context_over_time(task, existing_context)
        else:
            # If context doesn't exist, generate it
            return await self.enhance_context(task)

    async def retrieve_context(self, task: str):
        query = """
        MATCH (c:ContextData {task: $task})
        RETURN c
        ORDER BY c.timestamp DESC
        LIMIT 1
        """
        result = await self.knowledge_graph.execute_query(query, {"task": task})
        if result:
            return result[0]['c']['context']
        return None

    async def enhance_context(self, task: str):
        prompt = f"""
        Analyze the following task and provide detailed context information:
        Task: {task}
        Include:
        1. Relevant background information
        2. Potential challenges or considerations
        3. Related concepts or topics
        4. Any time-sensitive factors
        """
        context = await self.llm.chat_with_ollama("You are a context analysis expert.", prompt)
        enhanced_context = context.strip()
        await self.log_context(task, enhanced_context)
        return enhanced_context

    async def log_context(self, task: str, context: str):
        properties = {
            "task": task,
            "context": context,
            "timestamp": time.time()
        }
        embedding = self.embedding_manager.encode(context)
        properties["embedding"] = embedding.tolist()
        await self.knowledge_graph.add_or_update_node("ContextData", properties)

    async def get_related_contexts(self, task: str, k=3):
        current_context = await self.get_context(task)
        if not current_context:
            return []
        
        similar_contexts = await self.knowledge_graph.get_similar_nodes(current_context, label="ContextData", k=k)
        return [(context['task'], context['context']) for context, _ in similar_contexts]

    async def merge_contexts(self, task1: str, task2: str):
        context1 = await self.get_context(task1)
        context2 = await self.get_context(task2)
        
        if not context1 or not context2:
            return None

        prompt = f"""
        Merge the following two contexts, highlighting common elements and unique aspects:
        Context 1 (Task 1): {context1}
        Context 2 (Task 2): {context2}
        """
        merged_context = await self.llm.chat_with_ollama("You are a context merging expert.", prompt)
        return merged_context.strip()

    async def update_context_over_time(self, task: str, current_context: str):
        prompt = f"""
        Given the following context for a task, update it considering any potential changes over time:
        Task: {task}
        Current Context: {current_context}
        Consider:
        1. New developments or information
        2. Changes in relevance of certain aspects
        3. Emerging trends or shifts in the field
        """
        updated_context = await self.llm.chat_with_ollama("You are a context updating expert.", prompt)
        await self.log_context(task, updated_context.strip())
        return updated_context.strip()

    def get_context_embedding(self, context: str):
        return self.embedding_manager.encode(context)