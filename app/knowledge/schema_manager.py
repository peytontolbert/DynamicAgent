from app.utils.logger import StructuredLogger

logger = StructuredLogger("SchemaManager")

class SchemaManager:
    def __init__(self, query_executor):
        self.query_executor = query_executor

    async def create_node_constraint(self, label: str):
        query = f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.id IS UNIQUE"
        await self.query_executor.execute_query(query)
        logger.info(f"Created constraint for {label}")

    async def create_property_index(self, label: str, property: str):
        query = f"CREATE INDEX ON :{label}({property})"
        await self.query_executor.execute_query(query)
        logger.info(f"Created index on {label}.{property}")

    async def get_schema(self):
        query = """
        CALL apoc.meta.schema()
        YIELD value
        RETURN value
        """
        result = await self.query_executor.execute_query(query)
        return result[0]['value'] if result else {}