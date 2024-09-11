import asyncio
import os
import uuid
import json
import re
from typing import Dict, Any, List, Tuple
from app.logging.logging_manager import LoggingManager
from app.chat_with_ollama import ChatGPT
from app.execution.code_execution_manager import CodeExecutionManager
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.virtual_env.virtual_environment import VirtualEnvironment
from app.knowledge.procedural_knowledge import ProceduralKnowledgeSystem
from app.knowledge.episodic_knowledge import EpisodicKnowledgeSystem
from app.knowledge.conceptual_knowledge import ConceptualKnowledgeSystem
from app.knowledge.contextual_knowledge import ContextualKnowledgeSystem
from app.knowledge.meta_cognitive_knowledge import MetaCognitiveKnowledgeSystem
from app.knowledge.semantic_knowledge import SemanticKnowledgeSystem
from app.knowledge.embedding_manager import EmbeddingManager

"""
Dynamic Agent is a dynamic LLM agent equipped with two primary actions: respond and code_execute. 
This agent operates within a sophisticated knowledge system that includes Systematic, Episodic, Periodic, Conceptual, Contextual, and Meta-cognitive layers, all integrated with a graph database and retrieval-augmented generation (RAG) technology. The agent is capable of generating and executing Python, JavaScript, and Bash code dynamically, enabling it to perform a wide range of digital actions.
"""


class DynamicAgent:
    def __init__(self, uri, user, password, base_path):
        self.llm = ChatGPT()
        self.logging_manager = LoggingManager()  # Initialize logging manager first
        self.code_execution_manager = CodeExecutionManager(self.llm)
        self.knowledge_graph = KnowledgeGraph(uri, user, password)
        self.embedding_manager = EmbeddingManager()
        self.procedural_memory = ProceduralKnowledgeSystem(self.knowledge_graph, self.embedding_manager)
        self.episodic_memory = EpisodicKnowledgeSystem(self.knowledge_graph, self.embedding_manager)
        self.conceptual_knowledge = ConceptualKnowledgeSystem(self.knowledge_graph)
        self.contextual_knowledge = ContextualKnowledgeSystem(self.knowledge_graph)
        self.meta_cognitive = MetaCognitiveKnowledgeSystem(self.knowledge_graph)
        self.semantic_knowledge = SemanticKnowledgeSystem(self.knowledge_graph, self.embedding_manager)
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
        context = await self.contextual_knowledge.get_context(task)
        await self.episodic_memory.log_task(task, "Processing", context)
        
        while True:
            knowledge = await self.gather_knowledge(task)
            thoughts = await self.meta_cognitive.generate_thoughts(task, context, knowledge)
            
            action, action_thoughts = await self.decide_action(task, thoughts)
            
            if action == "respond":
                result = await self.respond(task, thoughts)
            elif action == "code_execute":
                result = await self.code_execute(task, thoughts)
            else:
                result = "Error: Unknown action."

            await self.update_knowledge_step(task, result, action, context, thoughts)

            if action == "respond" and result.endswith("Task completed successfully."):
                break

        await self.update_knowledge_complete(task, result, action, context, thoughts)
        return result

    async def update_knowledge_step(self, task: str, result: str, action: str, context: str, thoughts: str):
        await self.episodic_memory.log_task(task, result, context)
        await self.meta_cognitive.log_performance(task, {"result": result, "action": action})
        
        if action == "code_execute":
            insights, tool_usage = await self.procedural_memory.enhance_procedural_knowledge(task, result, context)
            self.logging_manager.log_info(f"Procedural Knowledge Insights: {insights}")
            self.logging_manager.log_info(f"Tool Usage Recommendations: {tool_usage}")
        
        # Extract concepts for this step
        concepts = await self.meta_cognitive.extract_concepts(task)
        
        self.logging_manager.log_info(f"Step Extracted Concepts: {concepts}")
        self.task_history.append({
            "task": task,
            "result": result,
            "concepts": concepts,
            "context": context,
            "action": action,
            "thoughts": thoughts,
            "procedural_insights": insights if action == "code_execute" else None,
            "tool_usage": tool_usage if action == "code_execute" else None
        })

    async def update_knowledge_complete(self, task: str, result: str, action: str, context: str, thoughts: str):
        # Extract concepts and generalize knowledge for the complete task
        concepts = await self.meta_cognitive.extract_concepts(task)
        generalized_knowledge = await self.meta_cognitive.generalize_knowledge(concepts)
        
        self.logging_manager.log_info(f"Task Completed - Extracted Concepts: {concepts}")
        self.logging_manager.log_info(f"Task Completed - Generalized Knowledge: {generalized_knowledge}")
        
        # Update the final task history entry with the generalized knowledge
        if self.task_history:
            self.task_history[-1]["generalized_knowledge"] = generalized_knowledge

        # Additional updates for task completion
        await self.conceptual_knowledge.update_concept_relations(concepts)
        await self.contextual_knowledge.update_context(task, context, result)
        await self.semantic_knowledge.update_language_understanding(task, result)
        
        if action == "code_execute":
            await self.procedural_memory.enhance_tool_usage(task)

    async def decide_action(self, task: str, thoughts: str) -> Tuple[str, str]:
        knowledge = await self.gather_knowledge(task)
        
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
        response = await self.llm.chat_with_ollama("You are a task analysis and decision-making expert.", prompt)
        decision, action_thoughts = self.extract_decision_and_thoughts(response)
        return decision.strip().lower(), action_thoughts.strip()

    def extract_decision_and_thoughts(self, response: str) -> Tuple[str, str]:
        decision_match = re.search(r'Decision:\s*(respond|code_execute)', response, re.IGNORECASE)
        thoughts_match = re.search(r'Action Thoughts:(.*)', response, re.DOTALL)
        
        decision = decision_match.group(1) if decision_match else ""
        thoughts = thoughts_match.group(1).strip() if thoughts_match else ""
        
        return decision, thoughts

    async def respond(self, task: str, thoughts: str) -> str:
        while True:
            knowledge = await self.gather_knowledge(task)
            
            prompt = f"""
            Task: {task}
            Thoughts: {thoughts}
            Interpreted understanding: {knowledge['interpreted_task']}
            Related concepts: {knowledge['related_concepts']}
            
            Provide a question or clarification to the task.
            """
            response = await self.llm.chat_with_ollama("You are a knowledgeable assistant.", prompt)
            user_input = input(f"Please provide additional input for the task: {task} {response}")
            task_with_input = f"{task} {user_input}"
            
            # Ask user if the task is complete
            is_complete = input("Is the task complete? (yes/no): ").strip().lower()
            if is_complete == 'yes':
                return f"{task_with_input} Task completed successfully."
            else:
                task = task_with_input

    async def gather_knowledge(self, task: str) -> dict:
        procedural_info = await self.procedural_memory.retrieve_tool_usage(task)
        
        # Retrieve related concepts from conceptual knowledge
        related_concepts = self.conceptual_knowledge.get_related_concepts(task)
        
        # Use contextual knowledge to understand the environment
        context_info = await self.contextual_knowledge.get_context(task)
        
        # Reflect on past performance using meta-cognitive knowledge
        performance_data = self.meta_cognitive.get_performance(task)
        
        # Ensure the task is understood using semantic knowledge
        interpreted_task = await self.semantic_knowledge.retrieve_language_meaning(task)
        if not interpreted_task:
            interpreted_task = await self.semantic_knowledge.enhance_language_understanding(task)
        
        # Retrieve generalized knowledge for the task
        concepts = await self.meta_cognitive.extract_concepts(task)
        generalized_knowledge = await self.meta_cognitive.get_generalized_knowledge(concepts)
        
        return {
            "procedural_info": procedural_info,
            "related_concepts": related_concepts,
            "context_info": context_info,
            "performance_data": performance_data,
            "interpreted_task": interpreted_task,
            "generalized_knowledge": generalized_knowledge
        }

    async def code_execute(self, task: str, thoughts: str) -> str:
        workspace_dir = self.virtual_env.base_path
        
        # Use procedural knowledge to guide code generation
        tool_usage = await self.procedural_memory.retrieve_tool_usage(task)
        context = await self.contextual_knowledge.get_context(task)
        
        thoughts = await self.generate_thoughts(task, workspace_dir, tool_usage, context, thoughts)
        code, language = await self.generate_code(task, workspace_dir, thoughts, tool_usage, context)
        
        if not code:
            return "Error: Failed to generate valid code."

        self.logging_manager.log_info(f"Generated code:\n{code}")
        
        formatted_code = self.format_code(code)
        self.logging_manager.log_info(f"Formatted code:\n{formatted_code}")

        # Execute the generated code
        result = await self.code_execution_manager.execute_and_monitor(formatted_code, self.execution_callback, language, cwd=workspace_dir)
        if result['status'] == 'success':
            await self.knowledge_graph.add_task_result(task, result['result'])
            await self.procedural_memory.log_tool_usage(language, formatted_code)
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

    async def generate_thoughts(self, task: str, workspace_dir: str, tool_usage: List[str], context: str, thoughts: str) -> str:
        thoughts_prompt = f"""
        Analyze the following task, thoughts, and context to provide additional thoughts on how to approach it:
        Task: {task}
        Thoughts: {thoughts}
        Workspace directory: {workspace_dir}
        Context: {context}
        
        Previous tool usage:
        {tool_usage}
        
        Provide your additional thoughts in the following format:
        Additional Thoughts: <Your analysis and approach>
        """
        thoughts_response = await self.llm.chat_with_ollama("You are an expert Python programmer and task analyzer.", thoughts_prompt)
        return thoughts_response.split("Additional Thoughts:")[1].strip() if "Additional Thoughts:" in thoughts_response else ""

    async def generate_code(self, task: str, workspace_dir: str, thoughts: str, tool_usage: List[str], context: str) -> (str, str):
        code_prompt = f"""
        Generate code to accomplish the following task within the workspace directory {workspace_dir}:
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
                code_response = await self.llm.chat_with_ollama("You are an expert programmer.", code_prompt)
                language, code = self.extract_code_and_language(code_response)
                if code:
                    return code, language
            except Exception as e:
                self.logging_manager.log_error(f"Error generating code (attempt {attempt + 1}/{max_attempts}): {str(e)}")
        return "", ""

    def extract_code_and_language(self, response: str) -> (str, str):
        language_match = re.search(r'Language:\s*(python|javascript|bash)', response, re.IGNORECASE)
        code_block = re.search(r'```(python|javascript|bash)\n(.*?)```', response, re.DOTALL)
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
            if task.lower() == 'exit':
                break
            result = await self.process_task(task)
            self.logging_manager.log_info(f"Task result: {result}")
        await self.cleanup()

    async def cleanup(self):
        if self.env_id and self.env_id != self.virtual_env.base_path:
            await self.virtual_env.destroy_environment(self.env_id)


            
