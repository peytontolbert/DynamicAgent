from typing import Dict, Any, List
from app.logging.logging_manager import logging_manager
import json
from app.utils.logger import logger

class Skill:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("This method should be implemented by subclasses")

class RespondSkill(Skill):
    def __init__(self, name: str, description: str, llm):
        super().__init__(name, description)
        self.llm = llm

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task = params.get("task", "")
        relevant_knowledge = params.get("relevant_knowledge", {})
        workspace_info = params.get("workspace_info", {})
        
        try:
            prompt = f"""
            Task: {task}
            Relevant Knowledge: {json.dumps(relevant_knowledge, indent=2)}
            Workspace Information:
            Path: {workspace_info.get('workspace_path')}
            Contents:
            {json.dumps(workspace_info.get('contents'), indent=2)}
            
            Based on the task and available information, generate a question to ask the user for more information or to clarify the next step.
            
            Respond in the following JSON format:
            {{
                "needs_more_info": true,
                "question": "Your question to ask the user"
            }}
            """
            llm_response = await self.llm.chat_with_ollama("You are a knowledgeable assistant with expertise in various fields.", prompt)
            
            try:
                response_data = json.loads(llm_response)
                if not response_data.get("needs_more_info", False) or "question" not in response_data:
                    raise ValueError("Invalid response format")
                return response_data
            except (json.JSONDecodeError, ValueError):
                logging_manager.log_error(f"Failed to parse LLM response JSON: {llm_response}")
                return {"needs_more_info": True, "question": "Could you please provide more information or clarify your request?"}
        except Exception as e:
            logging_manager.log_error(f"Error in RespondSkill: {str(e)}")
            return {"needs_more_info": True, "question": "An error occurred. Could you please rephrase your request?"}

class CodeExecuteSkill(Skill):
    def __init__(self, name: str, description: str, llm, code_execution_manager):
        super().__init__(name, description)
        self.llm = llm
        self.code_execution_manager = code_execution_manager

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task = params.get("task", "")
        relevant_knowledge = params.get("relevant_knowledge", {})
        try:
            code = await self.generate_code(task, relevant_knowledge)
            result = await self.execute_code(code)
            return {"status": "success", "result": result, "code": code}
        except Exception as e:
            return {"status": "error", "error": str(e), "code": code if 'code' in locals() else None}

    async def generate_code(self, task: str, relevant_knowledge: Dict[str, Any]) -> str:
        prompt = f"""
        Generate code to accomplish the following task:
        Task: {task}
        
        Relevant Knowledge: {json.dumps(relevant_knowledge, indent=2)}
        
        Consider the following when generating the code:
        1. Use efficient algorithms and data structures
        2. Handle potential errors and edge cases
        3. Include comments to explain complex parts
        4. Use relevant libraries or modules if necessary
        
        Provide only the code without any explanations.
        """
        code = await self.llm.chat_with_ollama("You are an expert programmer.", prompt)
        return code.strip()

    async def execute_code(self, code: str) -> str:
        # Use the existing CodeExecutionManager to execute the code
        result = await self.code_execution_manager.execute_and_monitor(code, self.execution_callback)
        return result['result'] if result['status'] == 'success' else f"Error: {result['error']}"

    async def execution_callback(self, status: Dict[str, Any]):
        logger.info(f"Code execution status: {status['status']}")