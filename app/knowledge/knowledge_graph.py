from neo4j import GraphDatabase
from typing import Dict, Any, List
from app.utils.logger import StructuredLogger
from dotenv import load_dotenv
import asyncio
import numpy as np
import time
import json
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from app.knowledge.embedding_manager import EmbeddingManager

load_dotenv()

logger = StructuredLogger("KnowledgeGraph")

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))  # Correctly initialize the driver
        self.is_async = asyncio.iscoroutinefunction(self.driver.session)
        self.embeddings = {}
        self.temporal_data = {}
        self.nodes = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_manager = EmbeddingManager()
        logger.info(f"Initialized KnowledgeGraph with {'async' if self.is_async else 'sync'} driver")

    def __getitem__(self, key):
        return self.nodes.get(key)

    def __setitem__(self, key, value):
        self.nodes[key] = value

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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
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

    async def add_or_update_node(self, label: str, properties: Dict[str, Any], embedding: np.ndarray = None):
        node_id = properties.get('id') or str(uuid.uuid4())
        properties['id'] = node_id

        # Serialize any non-primitive types
        for key, value in properties.items():
            if not isinstance(value, (str, int, float, bool, list)) or (isinstance(value, list) and not all(isinstance(item, (str, int, float, bool)) for item in value)):
                properties[key] = json.dumps(value)

        if embedding is None and 'summary' in properties:
            embedding = self.embedding_manager.encode(properties['summary'])
        
        if embedding is not None:
            properties['embedding'] = embedding.tolist()  # Store embedding as a list

        # First, try to find an existing node with the same 'name' property
        find_query = f"""
        MATCH (n:{label} {{name: $name}})
        RETURN n
        """
        result = await self.execute_query(find_query, {"name": properties.get('name')})

        if result:
            # If a node with the same name exists, update it
            update_query = f"""
            MATCH (n:{label} {{name: $name}})
            SET n += $properties
            RETURN n
            """
            await self.execute_query(update_query, {"name": properties.get('name'), "properties": properties})
            logger.info(f"Updated existing node with name: {properties.get('name')}")
        else:
            # If no node with the same name exists, create a new one
            create_query = f"""
            CREATE (n:{label} $properties)
            RETURN n
            """
            await self.execute_query(create_query, {"properties": properties})
            logger.info(f"Created new node with ID: {node_id}")

        self.nodes[node_id] = properties
        logger.info(f"Added or updated node with ID: {node_id}")

    async def add_relationship(self, start_node: Dict[str, Any], end_node: Dict[str, Any], relationship_type: str, properties: Dict[str, Any] = None):
        start_node_id = start_node.get('id')
        end_node_id = end_node.get('id')

        if not start_node_id or not end_node_id:
            raise ValueError("Both start_node and end_node must have an 'id' property")

        relationship_id = str(uuid.uuid4())
        properties = properties or {}
        properties['id'] = relationship_id

        query = f"""
        MATCH (a {{id: $start_node_id}})
        MATCH (b {{id: $end_node_id}})
        CREATE (a)-[r:{relationship_type} $properties]->(b)
        RETURN r
        """
        await self.execute_query(query, {
            "start_node_id": start_node_id,
            "end_node_id": end_node_id,
            "properties": properties
        })
        logger.info(f"Created relationship {relationship_type} between {start_node_id} and {end_node_id}")

    async def get_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        query = f"""
        MATCH (n:{label} {{name: $name}})
        RETURN n
        """
        result = await self.execute_query(query, {"name": properties.get('name')})
        return result[0]['n'] if result else None

    async def get_all_nodes(self, label: str) -> List[Dict[str, Any]]:
        query = f"""
        MATCH (n:{label})
        RETURN n
        """
        result = await self.execute_query(query)
        return [record['n'] for record in result]

    async def add_task_result(self, task: str, result: str):
        task_node = {
            "id": str(uuid.uuid4()),
            "content": task,
            "result": result,
            "timestamp": time.time()
        }
        await self.add_or_update_node("TaskResult", task_node)
        logger.info(f"Added task result for task: {task[:100]}...")

    async def add_improvement_suggestion(self, improvement: str):
        improvement_id = str(uuid.uuid4())
        improvement_node = {
            "id": improvement_id,
            "content": improvement[:1000],
            "timestamp": time.time()
        }
        await self.add_or_update_node("Improvement", improvement_node)
        logger.info(f"Added improvement suggestion: {improvement[:100]}...")

    async def get_system_performance(self) -> Dict[str, Any]:
        try:
            query = """
            MATCH (p:Performance)
            WHERE p.timestamp > $timestamp
            RETURN p.metric AS metric, AVG(p.value) AS avg_value
            """
            timestamp = time.time() - 86400  # Get performance data from the last 24 hours
            result = await self.execute_query(query, {"timestamp": timestamp})
            
            performance_data = {}
            for record in result:
                performance_data[record["metric"]] = record["avg_value"]
            
            return performance_data
        except Exception as e:
            logger.error(f"Error getting system performance: {str(e)}", exc_info=True)
            return {}

    async def store_performance_metric(self, metric: str, value: float):
        try:
            query = """
            CREATE (p:Performance {metric: $metric, value: $value, timestamp: $timestamp})
            """
            await self.execute_query(query, {"metric": metric, "value": value, "timestamp": time.time()})
            logger.info(f"Stored performance metric: {metric} = {value}")
        except Exception as e:
            logger.error(f"Error storing performance metric: {str(e)}", exc_info=True)

    async def get_all_knowledge(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (n)
        RETURN n
        """
        result = await self.execute_query(query)
        return [record['n'] for record in result]

    async def store_tool_usage(self, tool_name: str, subtask: Dict[str, Any], result: Dict[str, Any]):
        tool_usage = {
            "id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "subtask": json.dumps(subtask),
            "result": json.dumps(result),
            "timestamp": time.time()
        }
        await self.add_or_update_node("ToolUsage", tool_usage)

    async def get_tool_usage_history(self, tool_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = """
        MATCH (t:ToolUsage {tool_name: $tool_name})
        RETURN t
        ORDER BY t.timestamp DESC
        LIMIT $limit
        """
        result = await self.execute_query(query, {"tool_name": tool_name, "limit": limit})
        return [record['t'] for record in result]

    async def store_tool(self, tool_name: str, source_code: str):
        tool_node = {
            "id": str(uuid.uuid4()),
            "name": tool_name,
            "source_code": source_code,
            "created_at": time.time()
        }
        await self.add_or_update_node("Tool", tool_node)
        logger.info(f"Stored tool in knowledge graph: {tool_name}")

    async def get_tool(self, tool_name: str) -> Dict[str, Any]:
        query = """
        MATCH (t:Tool {name: $tool_name})
        RETURN t
        """
        result = await self.execute_query(query, {"tool_name": tool_name})
        return result[0]['t'] if result else None

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (t:Tool)
        RETURN t
        """
        result = await self.execute_query(query)
        return [record['t'] for record in result]

    async def get_relevant_knowledge(self, content: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (n)
        WHERE n.content CONTAINS $content
        RETURN n
        """
        result = await self.execute_query(query, {"content": content})
        return [record['n'] for record in result]

    async def store_compressed_knowledge(self, compressed_knowledge: str):
        compressed_node = {
            "id": str(uuid.uuid4()),
            "content": compressed_knowledge,
            "timestamp": time.time()
        }
        await self.add_or_update_node("CompressedKnowledge", compressed_node)
        logger.info(f"Stored compressed knowledge: {compressed_knowledge[:100]}...")

    async def get_similar_nodes(self, query: str, label: str = None, k: int = 5, use_faiss: bool = False):
        query_embedding = self.embedding_manager.encode(query)
        
        # Fetch all nodes with embeddings
        cypher_query = f"""
        MATCH (n{':' + label if label else ''})
        WHERE EXISTS(n.embedding)
        RETURN n
        """
        result = await self.execute_query(cypher_query)
        
        # Extract embeddings and node data
        nodes = [record['n'] for record in result]
        embeddings = [np.array(node['embedding']) for node in nodes]
        
        if use_faiss:
            self.embedding_manager.build_faiss_index(embeddings)
            distances, indices = self.embedding_manager.faiss_search(query_embedding, k)
            return [(nodes[i], 1 - d) for i, d in zip(indices[0], distances[0])]
        else:
            # Find most similar nodes
            similar_indices = self.embedding_manager.find_most_similar(query_embedding, embeddings, k)
            
            # Return similar nodes with their similarity scores
            return [(nodes[i], score) for i, score in similar_indices]