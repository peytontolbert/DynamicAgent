from typing import Dict, Any, List
import json

class ContextManager:
    def __init__(self):
        self.task_history: List[Dict[str, Any]] = []
        self.working_memory: Dict[str, Any] = {}

    def add_task(self, task: str, action: str, result: str):
        self.task_history.append({
            "task": task,
            "action": action,
            "result": result
        })

    def update_working_memory(self, key: str, value: Any):
        self.working_memory[key] = value

    def get_recent_context(self, num_tasks: int = 5) -> str:
        recent_tasks = self.task_history[-num_tasks:]
        context = "Recent tasks:\n"
        for task in recent_tasks:
            context += f"Task: {task['task']}\nAction: {task['action']}\nResult: {task['result']}\n\n"
        context += f"Working memory: {json.dumps(self.working_memory, indent=2)}"
        return context

    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.task_history[-limit:]