from typing import Dict, Any
import json

class RewardModel:
    def __init__(self, llm):
        self.llm = llm

    async def evaluate_task(self, task: str, task_context: Dict[str, Any], result: str) -> float:
        prompt = f"""
        Task: {task}
        Task Context: {json.dumps(task_context, indent=2)}
        Result: {result}
        
        Evaluate the performance of the task on a scale from 0 to 1, where 0 is poor and 1 is excellent.
        Consider factors such as:
        1. Accuracy of the result
        2. Efficiency of the solution
        3. Completeness of the task
        
        Provide your evaluation as a single float value between 0 and 1.
        """
        evaluation = await self.llm.chat_with_ollama("You are an expert in task evaluation.", prompt)
        try:
            score = float(evaluation.strip())
            return max(0.0, min(1.0, score))  # Ensure the score is between 0 and 1
        except ValueError:
            return 0.5  # Return a default mid-range score if parsing fails