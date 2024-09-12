from typing import List, Dict, Any
from app.chat_with_ollama import ChatGPT


class AgentThoughts:
    def __init__(self):
        self.llm = ChatGPT()

    async def generate_thoughts_from_context_and_abstract(
        self, task: str, context_info: str, generalized_knowledge: str
    ) -> str:
        prompt = f"""
        Generate thoughts for the following complex task using the provided context and generalized knowledge:

        Task: {task}
        Context: {context_info}
        Generalized Knowledge: {generalized_knowledge}

        Consider how the context and generalized knowledge can be applied to approach this complex task.
        Provide a structured thought process as an autonomous agentthat breaks down the task and considers potential challenges and solutions.
        
        Format your response as:
        Thoughts: <Your thoughts>
        """
        thoughts = await self.llm.chat_with_ollama(
            "You are an expert in complex problem-solving and abstract thinking.",
            prompt,
        )
        return thoughts.strip()

    async def generate_thoughts_from_procedural_and_episodic(
        self, task: str, recent_episodes: List[Dict[str, Any]]
    ) -> str:
        formatted_episodes = self._format_episodic_context(recent_episodes)
        prompt = f"""
        Generate thoughts for the following simple task using recent episodic memories:

        Task: {task}
        Recent Episodes:
        {formatted_episodes}

        Consider how these recent experiences can inform your approach to this task.
        Provide a straightforward thought process as an autonomous agent that applies relevant past experiences to the current task.
        
        Format your response as:
        Thoughts: <Your thoughts>
        """
        thoughts = await self.llm.chat_with_ollama(
            "You are an expert in applying past experiences to current tasks.", prompt
        )
        return thoughts.strip()

    async def generate_thoughts_from_spatial(self, task: str, spatial_info: str) -> str:
        prompt = f"""
        Generate thoughts for the following task using the provided spatial information:

        Task: {task}
        Spatial Information: {spatial_info}

        Format your response as:
        Thoughts: <Your spacial thoughts>
        """
        thoughts = await self.llm.chat_with_ollama(
            "You are an expert in spatial reasoning and problem-solving.",
            prompt,
        )
        return thoughts.strip()

    async def generate_thoughts_from_temporal(
        self, task: str, temporal_info: str
    ) -> str:
        prompt = f"""
        Generate thoughts for the following task using the provided temporal information:

        Task: {task}
        Temporal Information: {temporal_info}

        Format your response as:
        Thoughts: <Your temporal thoughts>
        """
        thoughts = await self.llm.chat_with_ollama(
            "You are an expert in temporal reasoning and problem-solving.",
            prompt,
        )
        return thoughts.strip()

    def _format_episodic_context(self, episodes: List[Dict[str, Any]]) -> str:
        formatted_episodes = []
        for episode in episodes:
            formatted_episode = f"""
            Summary: {episode['summary']}
            Similarity: {episode['similarity']}
            Action: {episode['action']['type']}
            Result: {episode['result']}
            """
            formatted_episodes.append(formatted_episode.strip())
        return "\n\n".join(formatted_episodes)

    async def generate_thoughts_from_action_result(
        self, task: str, action_thoughts: str, result: str
    ) -> str:
        prompt = f"""
        Generate thoughts based on the task, action thoughts, and result:

        Task: {task}
        Action Thoughts: {action_thoughts}
        Result: {result}

        Analyze the outcome and provide insights on:
        1. Was the action successful in addressing the task?
        2. What can be learned from this result?
        3. Are there any adjustments needed for future similar tasks?
        4. What are the next steps or considerations based on this outcome?

        Format your response as:
        Thoughts: <Your thoughts>
        """
        thoughts = await self.llm.chat_with_ollama(
            "You are an expert in analyzing actions and their outcomes.",
            prompt,
        )
        return thoughts.strip()
