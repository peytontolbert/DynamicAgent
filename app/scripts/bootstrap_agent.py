import asyncio
import argparse
import json
import os
from app.agent.dynamic_agent import DynamicAgent
from app.knowledge.knowledge_graph import KnowledgeGraph
from app.logging.logging_manager import LoggingManager

logger = LoggingManager().get_logger("BootstrapAgent")

async def bootstrap_agent(uri, user, password, base_path, framework_file, knowledge_file, export_path=None):
    agent = DynamicAgent(uri, user, password, base_path)
    await agent.setup()

    if framework_file:
        await agent.import_knowledge_framework(framework_file)
        logger.info(f"Imported knowledge framework from {framework_file}")

    if knowledge_file:
        await agent.import_agent_knowledge(knowledge_file)
        logger.info(f"Imported initial knowledge from {knowledge_file}")

    if export_path:
        await agent.export_knowledge_framework(export_path)
        logger.info(f"Exported current knowledge framework to {export_path}")

    return agent

async def export_agent_knowledge(agent, export_path):
    await agent.export_agent_knowledge(export_path)
    logger.info(f"Exported agent knowledge to {export_path}")

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

async def main():
    parser = argparse.ArgumentParser(description="Bootstrap a Dynamic Agent with knowledge frameworks and initial knowledge.")
    parser.add_argument("--config", help="Path to the configuration file", required=True)
    parser.add_argument("--framework", help="Path to the knowledge framework file")
    parser.add_argument("--knowledge", help="Path to the initial knowledge file")
    parser.add_argument("--export-framework", help="Path to export the current knowledge framework")
    parser.add_argument("--export-knowledge", help="Path to export the agent's knowledge after bootstrapping")
    args = parser.parse_args()

    config = load_config(args.config)

    agent = await bootstrap_agent(
        config['neo4j_uri'],
        config['neo4j_user'],
        config['neo4j_password'],
        config['base_path'],
        args.framework,
        args.knowledge,
        args.export_framework
    )

    if args.export_knowledge:
        await export_agent_knowledge(agent, args.export_knowledge)

    logger.info("Agent bootstrapping completed.")

if __name__ == "__main__":
    asyncio.run(main())