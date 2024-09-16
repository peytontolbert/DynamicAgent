from neo4j import GraphDatabase
from typing import Dict, Any, List
from app.utils.logger import StructuredLogger
from dotenv import load_dotenv
import asyncio
import json
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential
from neo4j.exceptions import AuthError
import numpy as np
from app.knowledge.embedding_manager import EmbeddingManager
load_dotenv()

"""
This module represents the central knowledge storage system using a Neo4j graph database.
It stores nodes and relationships and integrates with the EmbeddingManager for semantic search.

Key Features:
- **Connection Management**: Establishes and verifies connections to the Neo4j database.
- **Asynchronous Operations**: Supports asynchronous execution of database operations.
- **Node Management**: Adds, updates, and retrieves nodes with properties.
- **Relationship Management**: Creates relationships between nodes with properties.
- **Retry Mechanism**: Retries failed operations with exponential backoff.
- **Semantic Search**: Integrates with EmbeddingManager to perform semantic searches using embeddings.
- **Logging**: Provides structured logging for all operations.
"""

logger = StructuredLogger("KnowledgeGraph")


class KnowledgeGraph:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.embedding_manager = EmbeddingManager()
            logger.info("Successfully connected to Neo4j database")
            self.is_async = asyncio.iscoroutinefunction(self.driver.session)
        except AuthError:
            logger.error(
                "Failed to authenticate with Neo4j. Please check your credentials."
            )
            self.driver = None
            self.is_async = False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            self.driver = None
            self.is_async = False

    async def connect(self):
        try:
            if self.is_async:
                await self.driver.verify_connectivity()
            else:
                self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            raise

    async def close(self):
        if self.driver:
            await self.driver.close()
            logger.info("Closed connection to Neo4j database")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def execute_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        if not self.driver:
            logger.error("No active connection to Neo4j. Unable to execute query.")
            return []

        try:
            if self.is_async:
                async with self.driver.session() as session:
                    result = await session.run(query, parameters)
                    return await result.data()
            else:
                with self.driver.session() as session:
                    result = session.run(query, parameters)
                    return result.data()
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise

    async def add_or_update_node(self, label: str, properties: Dict[str, Any]):
        node_id = properties.get("id") or str(uuid.uuid4())
        properties["id"] = node_id

        # Serialize any non-primitive types
        for key, value in properties.items():
            if not isinstance(value, (str, int, float, bool, list)) or (
                isinstance(value, list)
                and not all(isinstance(item, (str, int, float, bool)) for item in value)
            ):
                properties[key] = json.dumps(value)

        # First, try to find an existing node with the same 'name' property
        find_query = f"""
        MATCH (n:{label} {{name: $name}})
        RETURN n
        """
        result = await self.execute_query(find_query, {"name": properties.get("name")})

        if result:
            # If a node with the same name exists, update it
            update_query = f"""
            MATCH (n:{label} {{name: $name}})
            SET n += $properties
            RETURN n
            """
            await self.execute_query(
                update_query, {"name": properties.get("name"), "properties": properties}
            )
            logger.info(f"Updated existing node with name: {properties.get('name')}")
        else:
            # If no node with the same name exists, create a new one
            create_query = f"""
            CREATE (n:{label} $properties)
            RETURN n
            """
            await self.execute_query(create_query, {"properties": properties})
            logger.info(f"Created new node with ID: {node_id}")

    async def add_relationship(
        self,
        start_node: Dict[str, Any],
        end_node: Dict[str, Any],
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ):
        start_node_id = start_node.get("id")
        end_node_id = end_node.get("id")

        if not start_node_id or not end_node_id:
            raise ValueError("Both start_node and end_node must have an 'id' property")

        relationship_id = str(uuid.uuid4())
        properties = properties or {}
        properties["id"] = relationship_id

        query = f"""
        MATCH (a {{id: $start_node_id}})
        MATCH (b {{id: $end_node_id}})
        CREATE (a)-[r:{relationship_type} $properties]->(b)
        RETURN r
        """
        await self.execute_query(
            query,
            {
                "start_node_id": start_node_id,
                "end_node_id": end_node_id,
                "properties": properties,
            },
        )
        logger.info(
            f"Created relationship {relationship_type} between {start_node_id} and {end_node_id}"
        )

    async def get_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        query = f"""
        MATCH (n:{label} {{name: $name}})
        RETURN n
        """
        result = await self.execute_query(query, {"name": properties.get("name")})
        return result[0]["n"] if result else None

    async def get_all_nodes(self, label: str) -> List[Dict[str, Any]]:
        query = f"""
        MATCH (n:{label})
        RETURN n
        """
        result = await self.execute_query(query)
        return [record["n"] for record in result]

    async def get_similar_nodes(
        self, query: str, label: str = None, k: int = 5, use_faiss: bool = False
    ):
        query_embedding = self.embedding_manager.encode(query)

        # Fetch all nodes with embeddings
        cypher_query = f"""
        MATCH (n{':' + label if label else ''})
        WHERE EXISTS(n.embedding)
        RETURN n
        """
        result = await self.execute_query(cypher_query)

        # Extract embeddings and node data
        nodes = [record["n"] for record in result]
        embeddings = [np.array(node["embedding"]) for node in nodes]

        if use_faiss:
            self.embedding_manager.build_faiss_index(embeddings)
            distances, indices = self.embedding_manager.faiss_search(query_embedding, k)
            return [(nodes[i], 1 - d) for i, d in zip(indices[0], distances[0])]
        else:
            # Find most similar nodes
            similar_indices = self.embedding_manager.find_most_similar(
                query_embedding, embeddings, k
            )

            # Return similar nodes with their similarity scores
            return [(nodes[i], score) for i, score in similar_indices]
