import asyncio
from typing import Dict, Any, Callable
from app.utils.code_utils import extract_code_and_language

class CodeExecutionManager:
    def __init__(self, llm):
        self.llm = llm

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
                language, code = extract_code_and_language(code_response)
                if code:
                    return code, language
            except Exception as e:
                self.logging_manager.log_error(f"Error generating code (attempt {attempt + 1}/{max_attempts}): {str(e)}")
        return "", ""
    
    async def execute_and_monitor(self, code: str, callback: Callable[[Dict[str, Any]], None], language: str = "python") -> Dict[str, Any]:
        if language == "python":
            execution_task = asyncio.create_task(self.execute_python(code))
        elif language == "javascript":
            execution_task = asyncio.create_task(self.execute_javascript(code))
        else:
            raise ValueError("Unsupported language")

        while not execution_task.done():
            await asyncio.sleep(0.1)  # Check status every 100ms
            await callback({"status": "running"})

        try:
            result = execution_task.result()
        except Exception as e:
            result = {"status": "error", "error": str(e)}
        await callback(result)
        return result

    async def execute_python(self, code: str) -> Dict[str, Any]:
        try:
            exec_globals = {}
            exec(code, exec_globals)
            return {"status": "success", "result": str(exec_globals.get('result', 'No result variable found.'))}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def execute_javascript(self, code: str) -> Dict[str, Any]:
        try:
            process = await asyncio.create_subprocess_exec(
                'node', '-e', code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return {"status": "success", "result": stdout.decode()}
            else:
                return {"status": "error", "error": stderr.decode()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def analyze_error(self, error: str) -> str:
        prompt = f"Analyze the following error and suggest a fix:\n\n{error}"
        return await self.llm.generate(prompt)

    async def adapt_code(self, original_code: str, error_analysis: str) -> str:
        prompt = f"""
        Original code:
        {original_code}

        Error analysis:
        {error_analysis}

        Please modify the code to fix the error. Provide only the modified code without explanations.
        """
        return await self.llm.generate(prompt)