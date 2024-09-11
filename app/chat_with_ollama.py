import aiohttp
import json
import re
from typing import Dict, Any
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import jsonschema
from jsonschema import validate
import requests
import time
from app.utils.logger import logger  # Use the new structured logger

class ChatGPT:
    def __init__(self, base_url: str="http://localhost:11434"):
        self.base_url = base_url

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def chat_with_ollama(self, system_prompt: str, user_prompt: str) -> str:
        max_tokens = 20000  # Adjust based on your model's actual limit
        chunked_user_prompt = await self.chunk_and_summarize(user_prompt, max_tokens)
        logger.info(f"Sending request to Ollama with system prompt: {system_prompt} and user_prompt: {user_prompt}", {"component": "ChatGPT", "method": "chat_with_ollama"})
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": "hermes3",
                        "prompt": f"{system_prompt}\n\nUser: {chunked_user_prompt}\nAssistant:",
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'response' in data:
                            logger.debug(f"Received response from Ollama: {data['response']}", {"component": "ChatGPT", "method": "chat_with_ollama"})
                            return data['response']
                        else:
                            logger.error(f"Unexpected response structure: {data}", {"component": "ChatGPT", "method": "chat_with_ollama"})
                            raise ValueError("Unexpected response structure from Ollama API")
                    else:
                        error_msg = f"Error from Ollama API: {response.status} - {await response.text()}"
                        logger.error(error_msg, {"component": "ChatGPT", "method": "chat_with_ollama"})
                        raise Exception(error_msg)
            except aiohttp.ClientError as e:
                logger.error(f"Network error in Ollama API call: {str(e)}", {"component": "ChatGPT", "method": "chat_with_ollama"})
                raise

    async def chunk_and_summarize(self, text: str, max_tokens: int = 10000, overlap_ratio: float = 0.3) -> str:
        """
        Chunk the input text and summarize it when it exceeds the maximum token limit.
        
        Args:
            text (str): The input text to be chunked and summarized.
            max_tokens (int): The maximum number of tokens allowed before chunking.
            overlap_ratio (float): The ratio of overlap between chunks.
        
        Returns:
            str: The summarized text.
        """
        # Estimate token count (rough estimation, adjust as needed)
        estimated_tokens = len(text.split())
        
        if estimated_tokens <= max_tokens:
            return text
        
        chunk_size = max_tokens
        overlap_size = int(chunk_size * overlap_ratio)
        chunks = []
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap_size
        
        summarized_chunks = []
        for chunk in chunks:
            summary_prompt = f"Summarize the following text, preserving key information:\n\n{chunk}"
            summary = await self.chat_with_ollama("You are a skilled text summarizer.", summary_prompt)
            summarized_chunks.append(summary)
        
        final_summary = "\n\n".join(summarized_chunks)
        return final_summary

    async def generate(self, prompt: str) -> str:
        logger.info(f"Generating response for prompt: {prompt}", {"component": "ChatGPT", "method": "generate"})
        return await self.chat_with_ollama("You are a helpful AI assistant.", prompt)

    async def robust_chat_with_ollama(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        logger.info(f"Starting robust chat with Ollama for system prompt: {system_prompt} and user_prompt: {user_prompt}", {"component": "ChatGPT", "method": "robust_chat_with_ollama"})
        response = await self.chat_with_ollama(system_prompt, user_prompt)
        return await self._ensure_json_response(system_prompt, user_prompt, response)

    async def _extract_json(self, response: str) -> Dict[str, Any]:
        try:
            # Find JSON-like content using regex
            json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in the response")
            
            json_response = json.loads(json_match.group(0))
            return json_response
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to extract JSON from response: {e}", {"component": "ChatGPT", "method": "_extract_json"})
            raise

    async def _ensure_json_response(self, system_prompt: str, user_prompt: str, response: str) -> Dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "tool": {"type": "string"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["description", "tool", "dependencies"]
                    }
                }
            },
            "required": ["plan"]
        }

        for attempt in range(3):
            try:
                json_response = await self._extract_json(response)
                validate(instance=json_response, schema=schema)
                logger.debug(f"Validated JSON response: {json_response}", {"component": "ChatGPT", "method": "_ensure_json_response"})
                return json_response
            except (json.JSONDecodeError, jsonschema.exceptions.ValidationError, ValueError) as e:
                logger.error(f"Failed to parse or validate JSON response from Ollama: {e}", {"component": "ChatGPT", "method": "_ensure_json_response"})
                logger.error(f"Response content: {response}", {"component": "ChatGPT", "method": "_ensure_json_response"})
                feedback_prompt = f"""
                The previous response was not in valid JSON format or did not match the required schema. Please correct it.
                Original prompt: {user_prompt}
                Previous response: {response}
                Provide a valid JSON response with the following structure:
                {{
                    "plan": [
                        {{
                            "description": "Step description",
                            "tool": "Tool to use",
                            "dependencies": ["List of dependencies"]
                        }}
                    ]
                }}
                """
                response = await self.chat_with_ollama(system_prompt, feedback_prompt)
        raise ValueError("Failed to get a valid JSON response after 3 attempts")

    async def chat_with_ollama_with_fallback(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            response = await self.robust_chat_with_ollama(system_prompt, user_prompt)
            if "error" in response:
                raise ValueError("Invalid JSON response")
            return response
        except Exception as e:
            logger.error(f"Error in chat_with_ollama_with_fallback: {str(e)}", {"component": "ChatGPT", "method": "chat_with_ollama_with_fallback"})
            return {"error": "Fallback response due to error"}

    def chat_with_ollama_nojson(self, system_prompt: str, prompt: str, retries: int=5, delay: int=5):
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.1",
            "prompt": f"{system_prompt}\n{prompt}",
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        for i in range(retries):
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                response = response.json()
                return response['response']
            except requests.exceptions.RequestException as e:
                if i < retries - 1:  # i is zero indexed
                    time.sleep(delay)  # wait before trying again
                else:
                    logger.error(f"Failed after {retries} attempts: {str(e)}", {"component": "ChatGPT", "method": "chat_with_ollama_nojson"})
                    raise e  # re-raise the last exception if all retries fail

    async def generate_code(self, prompt: str) -> str:
        logger.info(f"Generating code for prompt: {prompt}", {"component": "ChatGPT", "method": "generate_code"})
        return await self.chat_with_ollama("You are a code generation expert. Return only the JSON object.", prompt)

# Example usage
if __name__ == "__main__":
    async def main():
        chatgpt = ChatGPT()
        response = await chatgpt.chat_with_ollama_with_fallback("You are a helpful AI assistant.", "What is the weather today?")
        print(response)

    asyncio.run(main())