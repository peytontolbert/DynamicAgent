import json
from typing import Dict, Any, List
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.embedding_manager import EmbeddingManager
from app.chat_with_ollama import ChatGPT

"""
SpatialKnowledgeSystem for spatial reasoning

This module manages spatial knowledge, including storing and retrieving spatial data,
analyzing spatial relationships, and enhancing spatial reasoning.

Key features:
- Stores and retrieves spatial data
- Provides methods for spatial reasoning and analysis
- Integrates with the KnowledgeGraph for spatial data storage
"""


class SpatialKnowledgeSystem:
    def __init__(
        self, knowledge_graph: KnowledgeGraph, embedding_manager: EmbeddingManager
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_manager = embedding_manager
        self.llm = ChatGPT()

    async def add_spatial_data(self, spatial_data: Dict[str, Any]):
        """
        Add spatial data to the knowledge graph.
        """
        await self.knowledge_graph.add_or_update_node("SpatialData", spatial_data)

    async def get_spatial_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve spatial data related to a specific query.
        """
        return await self.knowledge_graph.get_nodes(label="SpatialData", query=query)

    async def analyze_spatial_relationships(self, spatial_data: Dict[str, Any]) -> str:
        """
        Analyze spatial relationships within the provided spatial data.
        """
        prompt = f"""
        Analyze the following spatial data and provide insights on spatial relationships:
        Spatial Data: {json.dumps(spatial_data)}
        """
        analysis = await self.llm.chat_with_ollama(
            "You are a spatial analysis expert.", prompt
        )
        return analysis.strip()

    async def enhance_spatial_reasoning(self, spatial_data: Dict[str, Any]) -> str:
        """
        Enhance spatial reasoning based on the provided spatial data.
        """
        prompt = f"""
        Enhance spatial reasoning for the following spatial data:
        Spatial Data: {json.dumps(spatial_data)}
        """
        reasoning = await self.llm.chat_with_ollama(
            "You are an expert in spatial reasoning.", prompt
        )
        return reasoning.strip()
