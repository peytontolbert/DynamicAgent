from neo4j import GraphDatabase
from app.utils.logger import StructuredLogger
from tenacity import retry, stop_after_attempt, wait_exponential

logger = StructuredLogger("QueryExecutor")

class QueryExecutor:
    def __init__(self, driver):
        self.driver = driver

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def execute_query(self, query: str, parameters: dict = None):
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return result.data()
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise