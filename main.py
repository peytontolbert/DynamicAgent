import asyncio
from app.agent.dynamic_agent import DynamicAgent


async def main():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    base_path = "./virtual_env"

    agent = DynamicAgent(uri, user, password, base_path)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
