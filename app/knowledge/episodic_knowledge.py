import json
from app.chat_with_ollama import ChatGPT
from app.knowledge.knowledge_graph import KnowledgeGraph
from typing import Dict, Any, List, Optional
from app.knowledge.embedding_manager import EmbeddingManager
from app.knowledge.community_manager import CommunityManager
import time
import asyncio
from app.utils.logger import StructuredLogger
import uuid

logger = StructuredLogger("EpisodicKnowledge")


class Episode:
    def __init__(
        self,
        thoughts: Dict[str, Any],
        action: Dict[str, Any],
        result: Optional[str] = None,
        summary: Optional[str] = None,
    ):
        self.thoughts = thoughts
        self.action = action
        self.result = result
        self.summary = summary


class EpisodicKnowledgeSystem:
    def __init__(
        self, knowledge_graph: KnowledgeGraph, embedding_manager: EmbeddingManager
    ):
        self.task_history = []
        self.knowledge_graph = knowledge_graph
        self.llm = ChatGPT()
        self.embedding_manager = embedding_manager
        self.episode_cache = {}
        self.community_manager = CommunityManager(knowledge_graph, embedding_manager)

    async def log_task(self, task: str, result: str, context: str, thoughts: str):
        task_entry = {
            "id": str(uuid.uuid4()),
            "task": task,
            "result": result,
            "context": context,
            "thoughts": thoughts,
            "timestamp": time.time(),
        }
        await self.knowledge_graph.add_or_update_node("TaskHistory", task_entry)

    def retrieve_task_history(self):
        return self.task_history

    def export(self, path):
        with open(path, "w") as f:
            json.dump(self.task_history, f)

    def import_data(self, path):
        with open(path, "r") as f:
            self.task_history = json.load(f)
            for task in self.task_history:
                self.knowledge_graph.add_or_update_node("TaskHistory", task)

    async def analyze_task(self, task_description, context=None):
        prompt = f"""
        Analyze the following task and provide insights and potential improvements:
        Task: {task_description}
        Context: {context}
        """
        analysis = await self.llm.chat_with_ollama(
            "You are a task analysis expert.", prompt
        )
        self.log_task(task_description, analysis.strip(), context)
        return analysis.strip()

    async def update_task_result(self, task_description, new_result, context=None):
        for task in self.task_history:
            if task["task"] == task_description:
                task["result"] = new_result
                task["context"] = context
                break
        self.knowledge_graph.add_or_update_node(
            "TaskHistory",
            {"task": task_description, "result": new_result, "context": context},
        )

    async def get_task_analysis(self, task_description):
        task = next(
            (task for task in self.task_history if task["task"] == task_description),
            None,
        )
        if task:
            return task["result"]
        return None

    async def memorize_episode(self, episode: Episode):
        summary = await self._summarize(
            episode.thoughts, episode.action, episode.result
        )
        episode.summary = summary

        properties = {
            "thoughts": json.dumps(episode.thoughts),
            "action": json.dumps(episode.action),
            "result": episode.result,
            "summary": summary,
            "timestamp": time.time(),  # Add timestamp for sorting
        }

        embedding = self.embedding_manager.encode(summary)
        properties["embedding"] = embedding.tolist()

        node_id = await self.knowledge_graph.add_or_update_node("Episode", properties)
        self.episode_cache[node_id] = episode  # Cache the episode

        # After memorizing a new episode, update the communities
        new_knowledge = {"id": node_id, "label": "Episode", "content": episode.summary}
        await self.community_manager.update_knowledge(new_knowledge)

    async def _summarize(
        self, thoughts: Dict[str, Any], action: Dict[str, Any], result: str
    ) -> str:
        prompt = f"""
        [THOUGHTS]
        {thoughts}

        [ACTION]
        {action}

        [RESULT OF ACTION]
        {result}

        [INSTRUCTION]
        Using above [THOUGHTS], [ACTION], and [RESULT OF ACTION], please summarize the event.

        [SUMMARY]
        """
        summary = await self.llm.chat_with_ollama(
            "You are an event summarizer.", prompt
        )
        return summary.strip()

    async def remember_recent_episodes(self, n: int = 5) -> List[Episode]:
        query = """
        MATCH (e:Episode)
        RETURN e
        ORDER BY e.timestamp DESC
        LIMIT $n
        """
        result = await self.knowledge_graph.execute_query(query, {"n": n})
        return [self._node_to_episode(record["e"]) for record in result]

    async def remember_related_episodes(self, query: str, k: int = 5) -> List[Episode]:
        similar_nodes = await self.knowledge_graph.get_similar_nodes(
            query, label="Episode", k=k
        )
        return [self._node_to_episode(node) for node, _ in similar_nodes]

    async def find_related_episodes_and_tasks(self, query: str, k: int = 5):
        similar_episodes = await self.knowledge_graph.get_similar_nodes(
            query, label="Episode", k=k
        )
        similar_tasks = await self.knowledge_graph.get_similar_nodes(
            query, label="TaskHistory", k=k
        )

        episodes = [
            self._node_to_episode(node) for node, similarity in similar_episodes
        ]
        tasks = [node for node, similarity in similar_tasks]

        return {"related_episodes": episodes, "related_tasks": tasks}

    def _node_to_episode(self, node: Dict[str, Any]) -> Episode:
        node_id = node.get("id")
        if node_id in self.episode_cache:
            return self.episode_cache[node_id]

        episode = Episode(
            thoughts=json.loads(node["thoughts"]),
            action=json.loads(node["action"]),
            result=node["result"],
            summary=node["summary"],
        )
        self.episode_cache[node_id] = episode
        return episode

    async def update_episode(self, episode_id: str, new_data: Dict[str, Any]):
        await super().update_episode(episode_id, new_data)
        # After updating an episode, we need to update the communities
        await self.community_manager.update_knowledge(new_data)

    async def delete_episode(self, episode_id: str):
        await super().delete_episode(episode_id)
        # After deleting an episode, we need to update the communities
        await self.community_manager.initialize()

    async def clear_episode_cache(self):
        self.episode_cache.clear()

    def rank_tasks_by_success(self):
        ranked_tasks = sorted(
            self.task_history, key=lambda x: x.get("result", ""), reverse=True
        )
        return ranked_tasks

    def log_task_dependency(self, task1, task2):
        self.knowledge_graph.add_relationship("TaskHistory", task1, task2)

    async def organize_episodes(self):
        await self.community_manager.initialize()
        logger.info("Episodes organized into communities")

    async def generate_hierarchical_summary(self):
        community_summaries = self.community_manager.community_summaries
        overall_summary = await self._summarize_community(
            list(community_summaries.values())
        )
        await self.knowledge_graph.add_or_update_node(
            "OverallSummary", {"id": "overall_summary", "content": overall_summary}
        )

    async def _summarize_community(self, summaries):
        combined_summary = " ".join(summaries)
        prompt = f"Summarize the following community of episodes:\n\n{combined_summary}"
        summary = await self.llm.chat_with_ollama(
            "You are a community summarizer.", prompt
        )
        return summary.strip()

    async def query_focused_summary(self, query: str):
        return await self.community_manager.query_communities(query)

    async def map_reduce_summarization(self, query: str):
        return await self.community_manager.query_communities(query)
