import asyncio
from typing import Dict, Any, Callable
from app.chat_with_ollama import ChatGPT
import os

class CodeExecutionManager:
    def __init__(self, llm: ChatGPT):
        self.llm = llm

    async def execute_and_monitor(self, code: str, callback: Callable[[Dict[str, Any]], None], language: str = "python") -> Dict[str, Any]:
        if language == "python":
            execution_task = asyncio.create_task(self.execute({"code": code}))
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

    async def _analyze_error(self, error: str) -> str:
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

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code = params.get("code", "")
        venv_path = os.path.join(os.getcwd(), "virtual_env")
        python_executable = os.path.join(venv_path, "bin", "python")

        try:
            process = await asyncio.create_subprocess_exec(
                python_executable,
                "-c",
                code,
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