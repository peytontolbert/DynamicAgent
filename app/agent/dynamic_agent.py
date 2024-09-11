import os
import uuid
import json
import re
from typing import Dict, Any, List, Tuple

from app.chat_with_ollama import ChatGPT
from app.logging.logging_manager import LoggingManager
from app.execution.code_execution_manager import CodeExecutionManager
from app.agent.agent_knowledge_interface import AgentKnowledgeInterface
from app.virtual_env.virtual_environment import VirtualEnvironment

"""
Dynamic Agent is a dynamic LLM agent equipped with two primary actions: respond and code_execute. 
This agent operates within a sophisticated knowledge system that includes Systematic, Episodic, Periodic, Conceptual, Contextual, and Meta-cognitive layers, all integrated with a graph database and retrieval-augmented generation (RAG) technology. The agent is capable of generating and executing Python, JavaScript, and Bash code dynamically, enabling it to perform a wide range of digital actions.
"""


class DynamicAgent:
    def __init__(self, uri, user, password, base_path):
        self.llm = ChatGPT()
        self.agent_knowledge_interface = AgentKnowledgeInterface(
            uri, user, password, base_path
        )
        self.logging_manager = LoggingManager()  # Initialize logging manager first
        self.code_execution_manager = CodeExecutionManager(self.llm)
        self.virtual_env = VirtualEnvironment(base_path)
        self.env_id = None
        self.has_memory = False
        self.task_history = []
        base_path = os.getenv("VIRTUAL_ENV_BASE_PATH", "./virtual_env")

    async def setup(self):
        if not os.path.exists(self.virtual_env.base_path):
            self.env_id = await self.virtual_env.create_environment(str(uuid.uuid4()))
        else:
            self.env_id = self.virtual_env.base_path
            self.logging_manager.log_info("Using existing virtual environment.")

        if not self.has_memory:
            await self.show_welcome_screen()
        else:
            self.logging_manager.log_info("Loading previous session...")

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
        context = await self.agent_knowledge_interface.contextual_knowledge.get_context(
            task
        )
        await self.agent_knowledge_interface.episodic_memory.log_task(
            task, "Processing", context
        )
        while True:
            knowledge = await self.agent_knowledge_interface.gather_knowledge(
                task, context
            )
            thoughts = (
                await self.agent_knowledge_interface.meta_cognitive.generate_thoughts(
                    task, knowledge
                )
            )

            action, action_thoughts = await self.decide_action(task, thoughts)

            if action == "respond":
                result = await self.respond(task, action_thoughts)
            elif action == "code_execute":
                result = await self.code_execute(task, action_thoughts)
            else:
                result = "Error: Unknown action."

            await self.agent_knowledge_interface.update_knowledge_step(
                task, result, action, context, thoughts
            )

            if action == "respond" and result.endswith("Task completed successfully."):
                break

        await self.agent_knowledge_interface.update_knowledge_complete(
            task, result, action, context, thoughts
        )
        return result

    async def decide_action(self, task: str, thoughts: str) -> Tuple[str, str]:
        knowledge = await self.agent_knowledge_interface.gather_knowledge(task)

        prompt = f"""
        Analyze the following task, thoughts, and knowledge to decide whether to use the 'respond' or 'code_execute' action:

        Task: {task}
        Thoughts: {thoughts}
        
        Knowledge Inputs:
        {json.dumps(knowledge, indent=2)}
        
        Provide your decision and additional thoughts in the following format:
        Decision: <respond or code_execute>
        Action Thoughts: <Your reasoning for this decision>
        """
        response = await self.llm.chat_with_ollama(
            "You are a task analysis and decision-making expert.", prompt
        )
        decision, action_thoughts = self.extract_decision_and_thoughts(response)
        return decision.strip().lower(), action_thoughts.strip()

    def extract_decision_and_thoughts(self, response: str) -> Tuple[str, str]:
        decision_match = re.search(
            r"Decision:\s*(respond|code_execute)", response, re.IGNORECASE
        )
        thoughts_match = re.search(r"Action Thoughts:(.*)", response, re.DOTALL)

        decision = decision_match.group(1) if decision_match else ""
        thoughts = thoughts_match.group(1).strip() if thoughts_match else ""

        return decision, thoughts

    async def respond(self, task: str, thoughts: str) -> str:
        while True:
            knowledge = await self.agent_knowledge_interface.gather_knowledge(task)

            prompt = f"""
            Task: {task}
            Thoughts: {thoughts}
            Interpreted understanding: {knowledge['interpreted_task']}
            Related concepts: {knowledge['related_concepts']}
            
            Provide a question or clarification to the task.
            """
            response = await self.llm.chat_with_ollama(
                "You are a knowledgeable assistant.", prompt
            )
            user_input = input(
                f"Please provide additional input for the task: {task} {response}"
            )
            task_with_input = f"{task} {user_input}"

            # Ask user if the task is complete
            is_complete = input("Is the task complete? (yes/no): ").strip().lower()
            if is_complete == "yes":
                return f"{task_with_input} Task completed successfully."
            else:
                task = task_with_input

    async def code_execute(self, task: str, thoughts: str) -> str:
        workspace_dir = self.virtual_env.base_path

        thoughts = await self.generate_code_thoughts(task, thoughts)
        context = await self.agent_knowledge_interface.contextual_knowledge.get_related_contexts(
            task
        )
        code, language = await self.generate_code(task, thoughts, context)

        if not code:
            return "Error: Failed to generate valid code."

        self.logging_manager.log_info(f"Generated code:\n{code}")

        formatted_code = self.format_code(code)
        self.logging_manager.log_info(f"Formatted code:\n{formatted_code}")

        # Execute the generated code
        result = await self.code_execution_manager.execute_and_monitor(
            formatted_code, self.execution_callback, language, cwd=workspace_dir
        )
        if result["status"] == "success":
            await self.knowledge_graph.add_task_result(task, result["result"])
            await self.procedural_memory.log_tool_usage(language, formatted_code)
            return f"Thoughts: {thoughts}\n\nResult: {result['result']}\n\nTask completed successfully."
        else:
            error_analysis = await self.analyze_error(result["error"], formatted_code)
            return f"Thoughts: {thoughts}\n\nError: {result['error']}\n\nSuggested Fix: {error_analysis}"

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

    async def generate_code_thoughts(
        self, task: str, tool_usage: List[str], context: str, thoughts: str
    ) -> str:
        thoughts_prompt = f"""
        Analyze the following task, thoughts, and context to provide additional thoughts on how to approach it:
        Task: {task}
        Thoughts: {thoughts}
        Context: {context}
        
        Previous tool usage:
        {tool_usage}
        
        Provide your additional thoughts in the following format:
        Additional Thoughts: <Your analysis and approach>
        """
        thoughts_response = await self.llm.chat_with_ollama(
            "You are an expert Python programmer and task analyzer.", thoughts_prompt
        )
        return (
            thoughts_response.split("Additional Thoughts:")[1].strip()
            if "Additional Thoughts:" in thoughts_response
            else ""
        )

    async def generate_code(
        self, task: str, thoughts: str, tool_usage: List[str], context: str
    ) -> (str, str):
        code_prompt = f"""
        Generate code to accomplish the following task within the current directory:
        Task: {task}
        
        Thoughts: {thoughts}
        Context: {context}
        
        Previous tool usage:
        {tool_usage}
        
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
        return f"""
        {code}
        """

    async def execution_callback(self, status: Dict[str, Any]):
        self.logging_manager.log_info(f"Execution status: {status['status']}")

    async def run(self):
        await self.setup()
        while True:
            task = input("Enter your task (or 'exit' to quit): ")
            if task.lower() == "exit":
                break
            result = await self.process_task(task)
            self.logging_manager.log_info(f"Task result: {result}")
        await self.cleanup()

    async def cleanup(self):
        if self.env_id and self.env_id != self.virtual_env.base_path:
            await self.virtual_env.destroy_environment(self.env_id)
