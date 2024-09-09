from neo4j import GraphDatabase
from app.utils.logger import StructuredLogger
from dotenv import load_dotenv
import os

load_dotenv()

logger = StructuredLogger("BaseKnowledgeGraph")

class BaseKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Initialized BaseKnowledgeGraph")

    async def connect(self):
        try:
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            raise

    async def close(self):
        if self.driver:
            await self.driver.close()
            logger.info("Closed connection to Neo4j database")