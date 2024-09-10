import asyncio
from typing import Dict, Any, Callable
from app.utils.code_utils import extract_code_and_language
import os
from app.logging.logging_manager import LoggingManager
import subprocess

class CodeExecutionManager:
    def __init__(self, llm, logging_manager):
        self.llm = llm
        self.logging_manager = logging_manager  # Ensure this line is present

    async def generate_code(self, task: str, workspace_dir: str, thoughts: str) -> (str, str):
        code_prompt = f"""
        Generate code to accomplish the following task within current working directory:
        Task: {task}
        
        Thoughts: {thoughts}
        
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
                language, code = extract_code_and_language(code_response)
                if code:
                    return code, language
            except Exception as e:
                self.logging_manager.log_error(f"Error generating code (attempt {attempt + 1}/{max_attempts}): {str(e)}")
        return "", ""
    
    async def execute_and_monitor(self, code: str, callback: Callable[[Dict[str, Any]], None], language: str = "python", cwd: str = None) -> Dict[str, Any]:
        if language == "python":
            execution_task = asyncio.create_task(self.execute_python(code, cwd=cwd))
        elif language == "javascript":
            execution_task = asyncio.create_task(self.execute_javascript(code, cwd=cwd))
        elif language == "bash":
            execution_task = asyncio.create_task(self.execute_bash(code, cwd))
        else:
            raise ValueError("Unsupported language")

        while not execution_task.done():
            await asyncio.sleep(1)  # Check status every 100ms
            await callback({"status": "running"})

        try:
            result = execution_task.result()
        except Exception as e:
            result = {"status": "error", "error": str(e)}
        await callback(result)
        return result
    
    async def execute_bash(self, code: str, cwd: str = 'virtual_env') -> Dict[str, Any]:
        try:
            # Create a temporary bash script file
            temp_file = os.path.join(cwd, "temp_script.sh")
            with open(temp_file, "w") as f:
                f.write(code)

            # Run the Bash script in the specified working directory (cwd)
            process = await asyncio.create_subprocess_exec(
                'bash', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd  # Execute in the virtual environment or given directory
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {"status": "success", "result": stdout.decode()}
            else:
                return {"status": "error", "error": stderr.decode()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def execute_python(self, code: str, cwd: str = 'virtual_env') -> Dict[str, Any]:
        try:
            # Ensure we're using an absolute path
            cwd = os.path.abspath(cwd)
            
            # Create a temporary Python file to execute the code
            temp_file = os.path.join(cwd, "temp_script.py")
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            with open(temp_file, "w") as f:
                f.write(code)
            
            # Run the Python script in the specified working directory
            process = await asyncio.create_subprocess_exec(
                'python', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            stdout, stderr = await process.communicate()
            
            # Clean up the temporary file
            os.remove(temp_file)
            
            if process.returncode == 0:
                return {"status": "success", "result": stdout.decode()}
            else:
                return {"status": "error", "error": stderr.decode()}
        except Exception as e:
            return {"status": "error", "error": f"Execution error: {str(e)}"}

    async def execute_javascript(self, code: str, cwd: str = 'virtual_env') -> Dict[str, Any]:
        try:
            process = await asyncio.create_subprocess_exec(
                'node', '-e', code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd  # Use the provided cwd for the virtual environment
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return {"status": "success", "result": stdout.decode()}
            else:
                return {"status": "error", "error": stderr.decode()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def analyze_error(self, error: str) -> str:
        prompt = f"""
        Analyze the following error and suggest a fix:

        Error: {error}

        Provide a concise explanation of the error and a specific, actionable fix.
        Do not include any code snippets in your response unless absolutely necessary.
        """
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