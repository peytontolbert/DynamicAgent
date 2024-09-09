import json
import uuid
from typing import Dict, Any, List
import os
from app.chat_with_ollama import ChatGPT
from app.execution.code_execution_manager import CodeExecutionManager
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.virtual_env.virtual_environment import VirtualEnvironment
from app.learning.continuous_learner import ContinuousLearner
from app.logging.logging_manager import LoggingManager
from app.utils.code_utils import format_code
from app.agent.context_manager import ContextManager
from app.learning.reward_model import RewardModel

class DynamicAgent:
    def __init__(self, uri, user, password, base_path):
        self.llm = ChatGPT()
        self.code_execution_manager = CodeExecutionManager(self.llm)
        self.knowledge_graph = KnowledgeGraph(uri, user, password)
        self.virtual_env = VirtualEnvironment(base_path)
        self.env_id = None
        self.has_memory = False
        self.task_history = []
        self.continuous_learner = ContinuousLearner(self.knowledge_graph, self.llm)
        self.logging_manager = LoggingManager()
        self.context_manager = ContextManager()
        self.reward_model = RewardModel(self.llm)

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
        2. Code Execute: Run Python or JavaScript code to perform tasks.
        
        Just type your task, and the agent will decide the best action to take.
        """
        self.logging_manager.log_info(welcome_message)

    async def process_task(self, task: str):
        task_context = {"original_task": task, "steps": []}
        while True:
            decision = await self.decide_action(task, task_context)
            action = decision["action"]
            confidence = decision["confidence"]
            reasoning = decision["reasoning"]
            
            self.logging_manager.log_info(f"Decided action: {action} (confidence: {confidence})")
            self.logging_manager.log_info(f"Reasoning: {reasoning}")
            
            if action == "respond":
                result, is_complete = await self.respond(task, task_context)
                print(result)
                task_context["steps"].append({
                    "action": "respond",
                    "result": result
                })
                if is_complete:
                    break
                task = input("User: ")
            elif action == "code_execute":
                result = await self.code_execute(task, task_context)
                task_context["steps"].append({
                    "action": "code_execute",
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "result": result
                })
                self.logging_manager.log_info(f"Step result: {result}")
                print(result)
            else:
                result = f"Error: Unknown action '{action}'. The agent can only 'respond' or 'code_execute'."
                print(result)

        # Evaluate the task using the reward model
        score = await self.reward_model.evaluate_task(task, task_context, result)
        self.logging_manager.log_info(f"Task evaluation score: {score}")

        # Update the knowledge graph with the task result and score
        await self.knowledge_graph.add_task_result(task, result, score)

        return "Task completed."

    async def respond(self, task: str, task_context: Dict[str, Any]) -> str:
        relevant_knowledge = await self.knowledge_graph.get_relevant_knowledge(task)
        
        prompt = f"""
        Task: {task}
        
        Relevant Knowledge:
        {json.dumps(relevant_knowledge, indent=2)}
        
        Task Context:
        {json.dumps(task_context, indent=2)}
        
        Based on the given information, formulate a question or request for the user to gather more information or clarify the task.
        
        Response format:
        {{
            "question": "<question or request for the user>",
            "reasoning": "<brief explanation of why this information is needed>"
        }}
        """
        
        response = await self.llm.chat_with_ollama("You are an expert in task analysis and information gathering.", prompt)
        
        try:
            decision = json.loads(response)
            user_input = input(decision["question"] + " ")
            return f"User input: {user_input}\nReasoning: {decision['reasoning']}", False
        except json.JSONDecodeError:
            return "Error: Failed to parse response. Please try again.", False

    def get_workspace_info(self) -> Dict[str, Any]:
        workspace_dir = self.virtual_env.base_path
        files_and_folders = []
        
        try:
            for item in os.listdir(workspace_dir):
                item_path = os.path.join(workspace_dir, item)
                item_type = "folder" if os.path.isdir(item_path) else "file"
                files_and_folders.append({"name": item, "type": item_type})
        except Exception as e:
            self.logging_manager.log_error(f"Error listing workspace contents: {str(e)}")
        
        return {
            "workspace_path": workspace_dir,
            "contents": files_and_folders
        }

    async def code_execute(self, task: str, task_context: Dict[str, Any]) -> str:
        workspace_dir = self.virtual_env.base_path
        
        # Generate thoughts and code to execute
        thoughts = await self.generate_thoughts(task)
        code, language = await self.code_execution_manager.generate_code(task, workspace_dir, thoughts)
        
        if not code:
            return "Error: Failed to generate valid code."

        self.logging_manager.log_info(f"Generated code:\n{code}")
        
        formatted_code = format_code(code)
        self.logging_manager.log_info(f"Formatted code:\n{formatted_code}")

        try:
            result = await self.code_execution_manager.execute_and_monitor(formatted_code, self.execution_callback, language, cwd=workspace_dir)
            if result['status'] == 'success':
                # Evaluate the task using the reward model
                score = await self.reward_model.evaluate_task(task, task_context, result['result'])
                self.logging_manager.log_info(f"Task evaluation score: {score}")

                # Update the knowledge graph with the task result and score
                task_result = await self.knowledge_graph.add_task_result(task, result['result'], score)
                await self.knowledge_graph.add_relationships_to_concepts(task_result['id'], task)
                await self.continuous_learner.learn({"content": task}, {"result": result['result']})
                return f"Thoughts: {thoughts}\n\nResult: {result['result']}\n\nTask completed successfully."
            else:
                error_analysis = await self.handle_error(result['error'], formatted_code)
                return f"Thoughts: {thoughts}\n\nError: {result['error']}\n\nSuggested Fix: {error_analysis}"
        except Exception as e:
            error_analysis = await self.handle_error(str(e), formatted_code)
            return f"Unexpected error: {str(e)}\n\nSuggested Fix: {error_analysis}"

    async def generate_thoughts(self, task: str) -> str:
        thoughts_prompt = f"""
        Analyze the following task and provide your thoughts on how to approach it:
        Task: {task}
        
        Provide your thoughts in the following format:
        Thoughts: <Your analysis and approach>
        """
        thoughts_response = await self.llm.chat_with_ollama("You are an expert Python|Javascript|Bash programmer and task analyzer.", thoughts_prompt)
        return thoughts_response.split("Thoughts:")[1].strip() if "Thoughts:" in thoughts_response else ""


    async def execution_callback(self, status: Dict[str, Any]):
        self.logging_manager.log_info(f"Execution status: {status['status']}")

    async def handle_error(self, error: str, code: str = None):
        error_prompt = """
        An error occurred during task execution:
        {error}

        {f'Code:\n{code}' if code else ''}

        Analyze the error and suggest a fix or next steps to resolve the issue.
        """
        analysis = await self.llm.chat_with_ollama("You are an expert programmer and error analyst.", error_prompt)
        self.context_manager.update_working_memory("last_error", {"error": error, "analysis": analysis})
        return analysis

    async def run(self):
        await self.setup()
        while True:
            task = input("Enter your task (or 'exit' to quit): ")
            if task.lower() == 'exit':
                break
            result = await self.process_task(task)
            self.logging_manager.log_info(f"Task result: {result}")
            print(result)  # Display the result to the user
        await self.cleanup()

    async def cleanup(self):
        if self.env_id and self.env_id != self.virtual_env.base_path:
            await self.virtual_env.destroy_environment(self.env_id)

    async def export_agent_knowledge(self, file_path: str):
        """Export the agent's knowledge to a file."""
        await self.knowledge_graph.export_knowledge(file_path)

    async def import_agent_knowledge(self, file_path: str):
        """Import knowledge from a file into the agent."""
        await self.knowledge_graph.import_knowledge(file_path)

    async def export_knowledge_framework(self, file_path: str):
        """Export the agent's knowledge framework to a file."""
        await self.knowledge_graph.export_advanced_knowledge_framework(file_path)

    async def import_knowledge_framework(self, file_path: str):
        """Import a knowledge framework from a file into the agent."""
        await self.knowledge_graph.import_advanced_knowledge_framework(file_path)

    async def bootstrap_agent(self, framework_file: str, knowledge_file: str):
        """Bootstrap the agent with a knowledge framework and initial knowledge."""
        await self.knowledge_graph.bootstrap_agent(framework_file, knowledge_file)
        self.logging_manager.log_info("Agent bootstrapped with advanced knowledge framework and initial knowledge")