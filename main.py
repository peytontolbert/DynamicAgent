import asyncio
import os
from dotenv import load_dotenv
from app.logging.logging_manager import LoggingManager, PerformanceMonitor
from app.agent.dynamic_agent import DynamicAgent

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging_manager = LoggingManager()
performance_monitor = PerformanceMonitor(logging_manager)

# Provide the necessary arguments for KnowledgeGraph initialization
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")
base_path = os.getenv("VIRTUAL_ENV_BASE_PATH", "virtual_env")

async def main():
    agent = DynamicAgent(uri, user, password, base_path)
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())