from typing import Dict, Any, List
import json
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.entropy.entropy_manager import EntropyManager  # Import EntropyManager
import time


class ContextManager:
    def __init__(self, knowledge_graph: KnowledgeGraph, entropy_manager: EntropyManager):
        self.task_history: List[Dict[str, Any]] = []
        self.working_memory: Dict[str, Any] = {}
        self.knowledge_graph = knowledge_graph
        self.entropy_manager = entropy_manager  # Initialize EntropyManager

    async def add_task(self, task: str, action: str, result: str, score: float):
        task_entry = {
            "task": task,
            "action": action,
            "result": result,
            "score": score
        }
        self.task_history.append(task_entry)
        await self.knowledge_graph.add_task_result(task, result, score)
        
        # Update working memory with the latest task
        await self.update_working_memory("latest_task", task_entry)

    async def update_working_memory(self, key: str, value: Any):
        self.working_memory[key] = value
        await self.knowledge_graph.store_compressed_knowledge(json.dumps(self.working_memory))

    async def get_recent_context(self, num_tasks: int = 5) -> str:
        recent_tasks = self.task_history[-num_tasks:]
        context = "Recent tasks:\n"
        for task in recent_tasks:
            context += f"Task: {task['task']}\nAction: {task['action']}\nResult: {task['result']}\nScore: {task['score']}\n\n"
        
        relevant_knowledge = await self.knowledge_graph.get_relevant_knowledge(context)
        context += f"Relevant knowledge: {json.dumps(relevant_knowledge, indent=2)}\n\n"
        context += f"Working memory: {json.dumps(self.working_memory, indent=2)}"
        return context

    async def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.task_history[-limit:]

    async def get_performance_metrics(self) -> Dict[str, float]:
        return await self.knowledge_graph.get_system_performance()

    async def add_tool_usage(self, tool_name: str, subtask: Dict[str, Any], result: Dict[str, Any]):
        await self.knowledge_graph.store_tool_usage(tool_name, subtask, result)
        # Update working memory with the latest tool usage
        await self.update_working_memory("latest_tool_usage", {
            "tool_name": tool_name,
            "subtask": subtask,
            "result": result
        })

    async def get_recent_tool_usage(self, limit: int = 5) -> List[Dict[str, Any]]:
        return await self.knowledge_graph.get_tool_usage_history(limit=limit)

    async def get_working_memory(self, key: str) -> Any:
        return self.working_memory.get(key)

    async def store_spatial_memory(self, location: str, context: Dict[str, Any]):
        await self.knowledge_graph.add_or_update_node("SpatialMemory", {
            "location": location,
            "context": json.dumps(context),
            "timestamp": time.time()
        })

    async def retrieve_spatial_memory(self, location: str) -> Dict[str, Any]:
        result = await self.knowledge_graph.get_node("SpatialMemory", {"location": location})
        return json.loads(result['context']) if result else {}

    async def compress_long_term_memory(self):
        old_memories = await self.knowledge_graph.get_old_memories(threshold_days=30)
        compressed_memory = await self.entropy_manager.compress_memories(old_memories)  # Use EntropyManager for memory compression
        await self.knowledge_graph.store_compressed_memory(compressed_memory)

    async def update_spatial_memory(self, location: str, context: Dict[str, Any]):
        existing_context = await self.retrieve_spatial_memory(location)
        updated_context = {**existing_context, **context}
        await self.store_spatial_memory(location, updated_context)

    async def get_memory_summary(self) -> Dict[str, Any]:
        return {
            "working_memory": self.working_memory,
            "recent_tasks": self.task_history[-5:],
            "recent_tool_usage": await self.get_recent_tool_usage(),
        }

    async def create_task_context(self, task: str) -> Dict[str, Any]:
        return {"original_task": task, "steps": []}

    async def get_contextual_knowledge(self, task: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        recent_context = await self.get_recent_context()
        relevant_knowledge = await self.knowledge_graph.get_relevant_knowledge(task)
        relevant_episodes = await self.knowledge_graph.recall_relevant_episodes(task_context)
        
        return {
            "recent_context": recent_context,
            "relevant_knowledge": relevant_knowledge,
            "relevant_episodes": relevant_episodes
        }