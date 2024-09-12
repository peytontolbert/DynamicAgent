import os
import uuid
from typing import Dict, Any, List, Tuple
import re
from app.chat_with_ollama import ChatGPT
from app.logging.logging_manager import LoggingManager
from app.execution.code_execution_manager import CodeExecutionManager
from app.agent.agent_knowledge_interface import AgentKnowledgeInterface
from app.virtual_env.virtual_environment import VirtualEnvironment
import time
import json

"""
Dynamic Agent is a dynamic LLM agent equipped with two primary actions: respond and code_execute. 
This agent operates within a sophisticated knowledge system that includes:
Systematic, 
Episodic, 
Periodic, 
Conceptual, 
Contextual, 
Meta-cognitive layers, 
all integrated with a graph database and retrieval-augmented generation (RAG) technology. 
The agent is capable of generating and executing Python, JavaScript, and Bash code dynamically, 
enabling it to perform a wide range of digital actions.
"""


class DynamicAgent:
    def __init__(self, uri, user, password, base_path):
        self.llm = ChatGPT()
        self.agent_knowledge_interface = AgentKnowledgeInterface(uri, user, password, base_path)
        self.logging_manager = LoggingManager()
        self.code_execution_manager = CodeExecutionManager(self.llm)
        self.virtual_env = VirtualEnvironment(base_path)
        self.env_id = None
        self.has_memory = False
        base_path = os.getenv("VIRTUAL_ENV_BASE_PATH", "./virtual_env")
        self.logging_manager.log_info("DynamicAgent initialized")

    async def setup(self):
        self.logging_manager.log_info("Setting up DynamicAgent")
        if not os.path.exists(self.virtual_env.base_path):
            self.env_id = await self.virtual_env.create_environment(str(uuid.uuid4()))
        else:
            self.env_id = self.virtual_env.base_path
            self.logging_manager.log_info("Using existing virtual environment.")

        if not self.has_memory:
            await self.show_welcome_screen()
        else:
            self.logging_manager.log_info("Loading previous session...")

        self.logging_manager.log_info("DynamicAgent setup complete")

    async def show_welcome_screen(self):
        welcome_message = """
        Welcome to the Dynamic Agent!
        
        You can perform two types of actions:
        1. Respond: Get information or answers using natural language.
        2. Code Execute: Run Python/JavaScript/Bash code to perform tasks.
        
        Just type your task, and the agent will decide the best action to take.
        """
        self.logging_manager.log_info(welcome_message)

    async def process_task(self, task: str):
        start_time = time.time()
        self.logging_manager.log_info(f"Processing task: {task}")
        context = await self.agent_knowledge_interface.contextual_knowledge.get_context(task)
        self.logging_manager.log_info(f"Retrieved context for task")
        
        self.logging_manager.log_info("Gathering knowledge")
        knowledge = await self.agent_knowledge_interface.gather_knowledge(task, context)
        
        self.logging_manager.log_info("Determining task complexity")
        is_complex = await self.determine_task_complexity(task, knowledge)
        self.logging_manager.log_info(f"Task complexity: {'Complex' if is_complex else 'Simple'}")
        
        while True:
            self.logging_manager.log_info("Generating thoughts")
            if is_complex:
                thoughts = await self.agent_knowledge_interface.generate_thoughts_from_context_and_abstract(task, knowledge['context_info'], knowledge['generalized_knowledge'])
            else:
                thoughts = await self.agent_knowledge_interface.generate_thoughts_from_procedural_and_episodic(task, knowledge['recent_episodes'])
            
            self.logging_manager.log_info("Deciding action")
            action, action_thoughts = await self.agent_knowledge_interface.decide_action(task, knowledge, thoughts)

            self.logging_manager.log_info(f"Chosen action: {action}")
            if action == "respond":
                result = await self.respond(task, thoughts, action_thoughts)
            elif action == "code_execute":
                result = await self.code_execute(task, thoughts, action_thoughts)
            else:
                result = "Error: Unknown action."
            self.logging_manager.log_info("Updating knowledge")
            await self.agent_knowledge_interface.update_knowledge_step(task, result, action, context, thoughts, action_thoughts)

            if self._is_task_complete(result):
                self.logging_manager.log_info("Task completed")
                break

        end_time = time.time()
        self.logging_manager.log_info(f"Task processing completed in {end_time - start_time:.2f} seconds")
        self.logging_manager.log_info("Updating knowledge for completed task")
        await self.agent_knowledge_interface.update_knowledge_complete(task, result, action, context, thoughts)
        return result

    async def determine_task_complexity(self, task: str, knowledge: Dict[str, Any]) -> bool:
        prompt = f"""
        Analyze the following task and knowledge to determine if the task is complex or simple:

        Task: {task}

        Knowledge:
        {json.dumps(knowledge, indent=2)}

        Consider the following factors:
        1. The number of steps or sub-tasks required to complete the task
        2. The depth of knowledge or expertise required
        3. The potential for unexpected complications or edge cases
        4. The interdependence of different components or systems
        5. The level of abstraction or conceptual understanding needed

        Provide your decision in the following format:
        Decision: <complex or simple>
        Reasoning: <Your explanation for this decision>
        """
        response = await self.llm.chat_with_ollama("You are a task complexity analysis expert.", prompt)
        decision, reasoning = self._extract_complexity_decision(response)
        return decision.strip().lower() == "complex"

    def _extract_complexity_decision(self, response: str) -> Tuple[str, str]:
        decision_match = re.search(r"Decision:\s*(complex|simple)", response, re.IGNORECASE)
        reasoning_match = re.search(r"Reasoning:(.*)", response, re.DOTALL)

        decision = decision_match.group(1) if decision_match else ""
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return decision, reasoning

    async def respond(self, task: str, thoughts: str, action_thoughts: str) -> str:
        self.logging_manager.log_info("Generating response")
        response = await self.agent_knowledge_interface.generate_response(task, thoughts, action_thoughts)
        self.logging_manager.log_info("Response generated, waiting for user input")
        user_input = input(f"Please provide additional input for the task: {task} {response}")
        task_with_input = f"{task} {user_input}"

        is_complete = input("Is the task complete? (yes/no): ").strip().lower()
        self.logging_manager.log_info(f"User indicated task completion: {is_complete}")
        if is_complete == "yes":
            return f"{task_with_input} Task completed successfully."
        else:
            return task_with_input

    async def code_execute(self, task: str, thoughts: str, action_thoughts: str) -> str:
        self.logging_manager.log_info("Executing code")
        workspace_dir = self.virtual_env.base_path
        code, language = await self.generate_code(task, thoughts, action_thoughts)

        if not code:
            return "Error: Failed to generate valid code."

        self.logging_manager.log_info(f"Generated code:\n{code}")
        formatted_code = self.format_code(code)
        self.logging_manager.log_info(f"Formatted code:\n{formatted_code}")

        result = await self.code_execution_manager.execute_and_monitor(
            formatted_code, self.execution_callback, language, cwd=workspace_dir
        )
        if result["status"] == "success":
            return f"Thoughts: {thoughts}\n\nResult: {result['result']}\n\nTask completed successfully."
        else:
            error_analysis = await self.analyze_error(result["error"], formatted_code)
            return f"Thoughts: {action_thoughts}\n\nError: {result['error']}\n\nSuggested Fix: {error_analysis}"

    async def generate_code(
        self, task: str, thoughts: str, action_thoughts: str
    ) -> (str, str):
        self.logging_manager.log_info("Generating code")
        code_prompt = f"""
        Generate code to accomplish the following task within the current directory:
        Task: {task}
        
        Thoughts: {thoughts}

        Action Thoughts: {action_thoughts}
        
        Provide your response in the following format:
        Language: <python|javascript|bash>
        Code:
        ```<language>
        <Your generated code here>
        ```
        """

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                code_response = await self.llm.chat_with_ollama(
                    "You are an expert programmer.", code_prompt
                )
                language, code = self.extract_code_and_language(code_response)
                if code:
                    return code, language
            except Exception as e:
                self.logging_manager.log_error(
                    f"Error generating code (attempt {attempt + 1}/{max_attempts}): {str(e)}"
                )
        return "", ""

    def extract_code_and_language(self, response: str) -> (str, str):
        language_match = re.search(
            r"Language:\s*(python|javascript|bash)", response, re.IGNORECASE
        )
        code_block = re.search(
            r"```(python|javascript|bash)\n(.*?)```", response, re.DOTALL
        )
        if language_match and code_block:
            language = language_match.group(1).strip().lower()
            code = code_block.group(2).strip()
            return language, code
        else:
            raise ValueError("No valid code block or language found in the response.")

    def format_code(self, code: str) -> str:
        return f"\n{code}\n"

    async def execution_callback(self, status: Dict[str, Any]):
        self.logging_manager.log_info(f"Execution status: {status['status']}")

    async def analyze_error(self, error: str, code: str) -> str:
        prompt = f"""
        The following error occurred while executing the code:
        {error}

        Code:
        {code}

        Analyze the error and suggest a fix.
        """
        analysis = await self.llm.chat_with_ollama(
            "You are an expert Python programmer.", prompt
        )
        return analysis.strip()

    def _is_task_complete(self, result: str) -> bool:
        return "Task completed successfully." in result

    async def run(self):
        await self.setup()
        while True:
            task = input("Enter your task (or 'exit' to quit): ")
            if task.lower() == "exit":
                self.logging_manager.log_info("Exiting DynamicAgent")
                break
            self.logging_manager.log_info(f"Received task: {task}")
            result = await self.process_task(task)
            self.logging_manager.log_info(f"Task result: {result}")
        await self.cleanup()

    async def cleanup(self):
        if self.env_id and self.env_id != self.virtual_env.base_path:
            await self.virtual_env.destroy_environment(self.env_id)
