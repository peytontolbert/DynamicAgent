import json
from typing import List, Dict, Any
from app.chat_with_ollama import ChatGPT
from collections import Counter
import re

class EntropyManager:
    def __init__(self, llm: ChatGPT):
        self.llm = llm

    async def consolidate_knowledge(self, recent_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Consolidate knowledge by summarizing recent nodes
        all_texts = " ".join(node.get("text", "") for node in recent_nodes)
        prompt = f"""
        You are an expert in knowledge consolidation. Summarize the following text and extract key insights:
        {all_texts}
        """
        summary = await self.llm.chat_with_ollama(prompt)
        consolidated_knowledge = {
            "summary": summary,
            "details": recent_nodes
        }
        return consolidated_knowledge

    async def compress_memories(self, old_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Compress memories by summarizing old memories
        all_texts = " ".join(memory.get("text", "") for memory in old_memories)
        prompt = f"""
        You are an expert in memory compression. Summarize the following text and highlight the most important points:
        {all_texts}
        """
        summary = await self.llm.chat_with_ollama(prompt)
        compressed_memory = {
            "summary": summary,
            "details": old_memories
        }
        return compressed_memory

    async def extract_concepts(self, content: str) -> List[str]:
        # Extract concepts using the LLM
        prompt = f"""
        You are an expert in concept extraction. Extract the key concepts from the following content:
        {content}
        """
        response = await self.llm.chat_with_ollama(prompt)
        concepts = response.split("\n")
        return [concept.strip() for concept in concepts if concept.strip()]

    async def manage_entropy(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Manage entropy by deciding the best action to take
        prompt = f"""
        Task: {task}
        Context: {json.dumps(context, indent=2)}
        
        Decide the best action to manage the entropy of the information and knowledge. Provide your decision as a JSON object with the following structure:
        {{
            "action": string,
            "confidence": float,
            "reasoning": string
        }}
        """
        decision = await self.llm.chat_with_ollama(prompt)
        return json.loads(decision)

    async def reflect_on_task(self, task: str, context: Dict[str, Any], result: str):
        # Reflect on the task to extract insights
        prompt = f"""
        Task: {task}
        Context: {json.dumps(context, indent=2)}
        Result: {result}
        
        Reflect on the task and extract key insights and learning points.
        """
        insights = await self.llm.chat_with_ollama(prompt)
        return insights

    async def consolidate_memory(self, recent_tasks: List[Dict[str, Any]]):
        # Consolidate memory by summarizing recent tasks
        all_texts = " ".join(task.get("text", "") for task in recent_tasks)
        prompt = f"""
        You are an expert in memory consolidation. Summarize the following recent tasks and extract key insights:
        {all_texts}
        """
        summary = await self.llm.chat_with_ollama(prompt)
        return summary

    async def generate_meta_learning_insights(self, recent_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Generate meta-learning insights from recent tasks
        all_texts = " ".join(task.get("text", "") for task in recent_tasks)
        prompt = f"""
        You are an expert in meta-learning. Analyze the following recent tasks and generate meta-learning insights:
        {all_texts}
        """
        insights = await self.llm.chat_with_ollama(prompt)
        return json.loads(insights)
