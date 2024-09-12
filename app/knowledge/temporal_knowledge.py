import json
from typing import Dict, Any, List
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.embedding_manager import EmbeddingManager
from app.chat_with_ollama import ChatGPT

"""
TemporalKnowledgeSystem for time-based reasoning

This module manages temporal knowledge, including storing and retrieving temporal data,
analyzing temporal relationships, and enhancing temporal reasoning.

Key features:
- Stores and retrieves temporal data
- Provides methods for temporal reasoning and analysis
- Integrates with the KnowledgeGraph for temporal data storage
"""


class TemporalKnowledgeSystem:
    def __init__(
        self, knowledge_graph: KnowledgeGraph, embedding_manager: EmbeddingManager
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_manager = embedding_manager
        self.llm = ChatGPT()

    async def add_temporal_data(self, temporal_data: Dict[str, Any]):
        """
        Add temporal data to the knowledge graph.
        """
        await self.knowledge_graph.add_or_update_node("TemporalData", temporal_data)

    async def get_temporal_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve temporal data related to a specific query.
        """
        return await self.knowledge_graph.get_nodes(label="TemporalData", query=query)

    async def analyze_temporal_relationships(
        self, temporal_data: Dict[str, Any]
    ) -> str:
        """
        Analyze temporal relationships within the provided temporal data.
        """
        prompt = f"""
        Analyze the following temporal data and provide insights on temporal relationships:
        Temporal Data: {json.dumps(temporal_data)}
        """
        analysis = await self.llm.chat_with_ollama(
            "You are a temporal analysis expert.", prompt
        )
        return analysis.strip()

    async def enhance_temporal_reasoning(self, temporal_data: Dict[str, Any]) -> str:
        """
        Enhance temporal reasoning based on the provided temporal data.
        """
        prompt = f"""
        Enhance temporal reasoning for the following temporal data:
        Temporal Data: {json.dumps(temporal_data)}
        """
        reasoning = await self.llm.chat_with_ollama(
            "You are an expert in temporal reasoning.", prompt
        )
        return reasoning.strip()
