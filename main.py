import asyncio
import json
import uuid
from typing import Dict, Any
from app.chat_with_ollama import ChatGPT
from app.execution.code_execution_manager import CodeExecutionManager
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.virtual_env.virtual_environment import VirtualEnvironment
from app.learning.continuous_learner import ContinuousLearner
from app.logging.logging_manager import LoggingManager, PerformanceMonitor
from app.workspace.workspace_manager import WorkspaceManager
from app.project_management.project_manager import ProjectManager
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging_manager = LoggingManager()
performance_monitor = PerformanceMonitor(logging_manager)

# Provide the necessary arguments for KnowledgeGraph initialization
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

class DynamicAgent:
    def __init__(self):
        self.llm = ChatGPT()
        self.code_execution_manager = CodeExecutionManager(self.llm)
        self.knowledge_graph = KnowledgeGraph(uri, user, password)
        base_path = os.getenv("VIRTUAL_ENV_BASE_PATH", "./virtual_env")
        self.virtual_env = VirtualEnvironment(base_path)
        self.workspace_manager = WorkspaceManager(base_path)
        self.project_manager = ProjectManager(self.workspace_manager, self.knowledge_graph)
        self.env_id = None
        self.has_memory = False
        self.task_history = []
        self.continuous_learner = ContinuousLearner(self.knowledge_graph, self.llm)

    async def setup(self):
        if not os.path.exists(self.virtual_env.base_path):
            self.env_id = await self.virtual_env.create_environment(str(uuid.uuid4()))
        else:
            self.env_id = self.virtual_env.base_path
            logging_manager.log_info("Using existing virtual environment.")

        if not self.has_memory:
            await self.show_welcome_screen()
        else:
            logging_manager.log_info("Loading previous session...")

    async def show_welcome_screen(self):
        welcome_message = """
        Welcome to the Dynamic Agent!
        
        You can perform two types of actions:
        1. Respond: Get information or answers using natural language.
        2. Code Execute: Run Python or JavaScript code to perform tasks.
        
        Just type your task, and the agent will decide the best action to take.
        """
        logging_manager.log_info(welcome_message)

    async def process_task(self, task: str):
        action = await self.decide_action(task)
        if action == "respond":
            result = await self.respond(task)
        elif action == "code_execute":
            result = await self.code_execute(task)
        else:
            result = "Error: Unknown action."
        logging_manager.log_info(f"Result: {result}")
        self.task_history.append({"task": task, "result": result})
        return result

    async def decide_action(self, task: str) -> str:
        prompt = f"""
        Analyze the following task and decide whether to use the 'respond' or 'code_execute' action:
        Task: {task}
        
        Consider the following:
        1. If the task requires information retrieval or explanation, use 'respond'.
        2. If the task involves data manipulation, computation, system interaction, or web access, use 'code_execute'.
        
        Provide your decision as a single word: either 'respond' or 'code_execute'.
        """
        decision = await self.llm.chat_with_ollama("You are a task analysis expert.", prompt)
        return decision.strip().lower()

    async def respond(self, task: str) -> str:
        relevant_knowledge = await self.knowledge_graph.get_relevant_knowledge(task)
        prompt = f"""
        Task: {task}
        Relevant Knowledge: {json.dumps(relevant_knowledge)}
        
        Provide a concise and informative response to the task.
        """
        response = await self.llm.chat_with_ollama("You are a knowledgeable assistant.", prompt)
        return response

    async def code_execute(self, task: str) -> str:
        workspace_dir = self.virtual_env.base_path
        
        thoughts = await self.generate_thoughts(task, workspace_dir)
        code, language = await self.generate_code(task, workspace_dir, thoughts)
        
        if not code:
            return "Error: Failed to generate valid code."

        logging_manager.log_info(f"Generated code:\n{code}")
        
        formatted_code = self.format_code(code)
        logging_manager.log_info(f"Formatted code:\n{formatted_code}")

        # Execute the generated code
        performance_monitor.start_timer("code_execution")
        result = await self.code_execution_manager.execute_and_monitor(formatted_code, self.execution_callback, language)
        performance_monitor.stop_timer("code_execution")
        
        if result['status'] == 'success':
            await self.knowledge_graph.add_task_result(task, result['result'])
            await self.continuous_learner.learn({"content": task}, {"result": result['result']})
            return f"Thoughts: {thoughts}\n\nResult: {result['result']}\n\nTask completed successfully."
        else:
            error_analysis = await self.analyze_error(result['error'], formatted_code)
            return f"Thoughts: {thoughts}\n\nError: {result['error']}\n\nSuggested Fix: {error_analysis}"

    async def analyze_error(self, error: str, code: str) -> str:
        prompt = f"""
        The following error occurred while executing the code:
        {error}

        Code:
        {code}

        Analyze the error and suggest a fix.
        """
        analysis = await self.llm.chat_with_ollama("You are an expert Python programmer.", prompt)
        return analysis.strip()

    async def generate_thoughts(self, task: str, workspace_dir: str) -> str:
        thoughts_prompt = f"""
        Analyze the following task and provide your thoughts on how to approach it:
        Task: {task}
        Workspace directory: {workspace_dir}
        
        Provide your thoughts in the following format:
        Thoughts: <Your analysis and approach>
        """
        thoughts_response = await self.llm.chat_with_ollama("You are an expert Python programmer and task analyzer.", thoughts_prompt)
        return thoughts_response.split("Thoughts:")[1].strip() if "Thoughts:" in thoughts_response else ""

    async def generate_code(self, task: str, workspace_dir: str, thoughts: str) -> (str, str):
        code_prompt = f"""
        Generate code to accomplish the following task within the workspace directory {workspace_dir}:
        Task: {task}
        
        Thoughts: {thoughts}
        
        Provide your response in the following format:
        Language: <python|javascript>
        Code:
        ```<language>
        <Your generated code here>
        ```
        """
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                code_response = await self.llm.chat_with_ollama("You are an expert programmer.", code_prompt)
                language, code = self.extract_code_and_language(code_response)
                if code:
                    return code, language
            except Exception as e:
                logging_manager.log_error(f"Error generating code (attempt {attempt + 1}/{max_attempts}): {str(e)}")
        return "", ""

    def extract_code_and_language(self, response: str) -> (str, str):
        language_match = re.search(r'Language:\s*(python|javascript)', response, re.IGNORECASE)
        code_block = re.search(r'```(python|javascript)\n(.*?)```', response, re.DOTALL)
        if language_match and code_block:
            language = language_match.group(1).strip().lower()
            code = code_block.group(2).strip()
            return language, code
        else:
            raise ValueError("No valid code block or language found in the response.")

    def format_code(self, code: str) -> str:
        return f"""
{code}
"""

    async def execution_callback(self, status: Dict[str, Any]):
        logging_manager.log_info(f"Execution status: {status['status']}")

    async def run(self):
        await self.setup()
        while True:
            task = input("Enter your task (or 'exit' to quit): ")
            if task.lower() == 'exit':
                break
            result = await self.process_task(task)
            logging_manager.log_info(f"Task result: {result}")
        await self.cleanup()

    async def cleanup(self):
        if self.env_id and self.env_id != self.virtual_env.base_path:
            await self.virtual_env.destroy_environment(self.env_id)

async def main():
    agent = DynamicAgent()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())