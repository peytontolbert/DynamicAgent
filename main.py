import asyncio
import uvicorn
from app.agent.dynamic_agent import DynamicAgent
from app.server import app
import os
from dotenv import load_dotenv

async def initialize_agent():
    load_dotenv()
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USER')
    password = os.getenv('NEO4J_PASSWORD')
    base_path = os.getenv('VIRTUAL_ENV_BASE_PATH', './virtual_env')

    agent = DynamicAgent(uri, user, password, base_path)
    await agent.setup()
    return agent

async def main():
    agent = await initialize_agent()
    app.state.agent = agent  # Store the agent in the FastAPI app state

    # Run the FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
