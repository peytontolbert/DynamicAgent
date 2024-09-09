from app.utils.logger import StructuredLogger
import json
import uuid
import time

logger = StructuredLogger("NodeManager")

class NodeManager:
    def __init__(self, query_executor):
        self.query_executor = query_executor

    async def add_or_update_node(self, label: str, properties: dict):
        node_id = properties.get('id') or str(uuid.uuid4())
        properties['id'] = node_id

        for key, value in properties.items():
            if not isinstance(value, (str, int, float, bool, list)) or (isinstance(value, list) and not all(isinstance(item, (str, int, float, bool)) for item in value)):
                properties[key] = json.dumps(value)

        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $properties
        RETURN n
        """
        result = await self.query_executor.execute_query(query, {"id": node_id, "properties": properties})
        logger.info(f"Added or updated node with ID: {node_id}")
        return result[0]['n'] if result else None

    async def get_node(self, label: str, node_id: str):
        query = f"""
        MATCH (n:{label} {{id: $id}})
        RETURN n
        """
        result = await self.query_executor.execute_query(query, {"id": node_id})
        return result[0]['n'] if result else None

    async def get_all_nodes(self, label: str):
        query = f"""
        MATCH (n:{label})
        RETURN n
        """
        result = await self.query_executor.execute_query(query)
        return [record['n'] for record in result]