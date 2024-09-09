from app.utils.logger import StructuredLogger
import uuid

logger = StructuredLogger("RelationshipManager")

class RelationshipManager:
    def __init__(self, query_executor):
        self.query_executor = query_executor

    async def add_relationship(self, start_node_id: str, end_node_id: str, relationship_type: str, properties: dict = None):
        relationship_id = str(uuid.uuid4())
        properties = properties or {}
        properties['id'] = relationship_id

        query = f"""
        MATCH (a {{id: $start_node_id}})
        MATCH (b {{id: $end_node_id}})
        CREATE (a)-[r:{relationship_type} $properties]->(b)
        RETURN r
        """
        await self.query_executor.execute_query(query, {
            "start_node_id": start_node_id,
            "end_node_id": end_node_id,
            "properties": properties
        })
        logger.info(f"Created relationship {relationship_type} between {start_node_id} and {end_node_id}")

    async def get_relationships(self, node_id: str, relationship_type: str = None):
        query = f"""
        MATCH (n {{id: $node_id}})-[r{f':{relationship_type}' if relationship_type else ''}]->(m)
        RETURN type(r) AS type, r AS relationship, m AS target_node
        """
        result = await self.query_executor.execute_query(query, {"node_id": node_id})
        return result