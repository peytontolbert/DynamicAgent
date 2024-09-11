import json
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.knowledge.embedding_manager import EmbeddingManager
from app.knowledge.community_manager import CommunityManager
from typing import List, Dict, Any
import networkx as nx
import re


class ProceduralKnowledgeSystem:
    def __init__(
        self, knowledge_graph: KnowledgeGraph, embedding_manager: EmbeddingManager
    ):
        self.tool_usage = {}
        self.knowledge_graph = knowledge_graph
        self.embedding_manager = embedding_manager
        self.community_manager = CommunityManager(knowledge_graph, embedding_manager)
        self.llm = ChatGPT()
        self.tool_graph = nx.Graph()

    async def log_tool_usage(self, tool_name: str, usage: str):
        if tool_name not in self.tool_usage:
            self.tool_usage[tool_name] = []
        self.tool_usage[tool_name].append(usage)

        # Add to knowledge graph
        node_data = {"tool_name": tool_name, "usage": usage}
        await self.knowledge_graph.add_or_update_node("ToolUsage", node_data)

        # Update tool graph
        self.tool_graph.add_node(tool_name)
        for other_tool in self.tool_usage:
            if other_tool != tool_name:
                similarity = self.embedding_manager.cosine_similarity(
                    self.embedding_manager.encode(usage),
                    self.embedding_manager.encode(self.tool_usage[other_tool][-1]),
                )
                self.tool_graph.add_edge(tool_name, other_tool, weight=similarity)

        # Update community detection
        await self.community_manager.update_knowledge(node_data)

    async def retrieve_tool_usage(self, tool_name: str) -> List[str]:
        return self.tool_usage.get(tool_name, [])

    async def export(self, path: str):
        with open(path, "w") as f:
            json.dump(self.tool_usage, f)

    async def import_data(self, path: str):
        with open(path, "r") as f:
            self.tool_usage = json.load(f)
            for tool_name, usages in self.tool_usage.items():
                for usage in usages:
                    await self.knowledge_graph.add_or_update_node(
                        "ToolUsage", {"tool_name": tool_name, "usage": usage}
                    )
        await self.rebuild_tool_graph()

    async def enhance_tool_usage(self, tool_name: str) -> str:
        prompt = f"""
        Analyze the following tool usage and provide detailed insights and improvements:
        Tool: {tool_name}
        """
        usage_insights = await self.llm.chat_with_ollama(
            "You are a tool usage analysis expert.", prompt
        )
        await self.log_tool_usage(tool_name, usage_insights.strip())
        return usage_insights.strip()

    async def rebuild_tool_graph(self):
        self.tool_graph.clear()
        for tool_name, usages in self.tool_usage.items():
            self.tool_graph.add_node(tool_name)
            for usage in usages:
                embedding = self.embedding_manager.encode(usage)
                for other_tool, other_usages in self.tool_usage.items():
                    if other_tool != tool_name:
                        other_embedding = self.embedding_manager.encode(
                            other_usages[-1]
                        )
                        similarity = self.embedding_manager.cosine_similarity(
                            embedding, other_embedding
                        )
                        self.tool_graph.add_edge(
                            tool_name, other_tool, weight=similarity
                        )

    async def get_similar_tools(
        self, tool_name: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        if tool_name not in self.tool_graph:
            return []

        neighbors = sorted(
            self.tool_graph[tool_name].items(),
            key=lambda x: x[1]["weight"],
            reverse=True,
        )[:k]

        return [
            {"tool": neighbor, "similarity": data["weight"]}
            for neighbor, data in neighbors
        ]

    async def get_tool_communities(self) -> Dict[str, List[str]]:
        return self.community_manager.communities

    async def query_tool_knowledge(self, query: str) -> str:
        return await self.community_manager.query_communities(query)

    async def initialize(self):
        await self.rebuild_tool_graph()
        await self.community_manager.initialize()

    async def enhance_procedural_knowledge(
        self, task: str, result: str, context: str, thoughts: str
    ):
        prompt = f"""
        Analyze the following task, result, context, and thoughts to enhance procedural knowledge:
        Task: {task}
        Result: {result}
        Context: {context}
        Thoughts: {thoughts}

        Provide insights on:
        1. What tools or techniques were effective?
        2. What could be improved in the approach?
        3. Are there any patterns or strategies that could be applied to similar tasks?

        Format your response as:
        Insights: <Your analysis>
        Tool Usage: <Specific tool or technique recommendations>
        """

        # ... rest of the method ...

    def extract_insights_and_tool_usage(self, response: str) -> (str, Dict[str, str]):
        insights = ""
        tool_usage = {}

        insights_match = re.search(r"Insights:(.*?)Tool Usage:", response, re.DOTALL)
        if insights_match:
            insights = insights_match.group(1).strip()

        tool_usage_match = re.search(r"Tool Usage:(.*)", response, re.DOTALL)
        if tool_usage_match:
            tool_usage_text = tool_usage_match.group(1).strip()
            for line in tool_usage_text.split("\n"):
                if ":" in line:
                    tool, usage = line.split(":", 1)
                    tool_usage[tool.strip()] = usage.strip()

        return insights, tool_usage
