from app.knowledge.base_knowledge_graph import BaseKnowledgeGraph
from app.knowledge.node_manager import NodeManager
from app.knowledge.relationship_manager import RelationshipManager
from app.knowledge.query_executor import QueryExecutor
from app.knowledge.schema_manager import SchemaManager
from app.knowledge.knowledge_import_export import KnowledgeImportExport
import time
import json
import uuid
class KnowledgeGraph(BaseKnowledgeGraph):
    def __init__(self, uri, user, password):
        super().__init__(uri, user, password)
        self.query_executor = QueryExecutor(self.driver)
        self.node_manager = NodeManager(self.query_executor)
        self.relationship_manager = RelationshipManager(self.query_executor)
        self.schema_manager = SchemaManager(self.query_executor)
        self.knowledge_import_export = KnowledgeImportExport(self.node_manager, self.schema_manager)

    async def add_or_update_node(self, label: str, properties: dict):
        return await self.node_manager.add_or_update_node(label, properties)

    async def add_relationship(self, start_node_id: str, end_node_id: str, relationship_type: str, properties: dict = None):
        await self.relationship_manager.add_relationship(start_node_id, end_node_id, relationship_type, properties)

    async def get_node(self, label: str, node_id: str):
        return await self.node_manager.get_node(label, node_id)

    async def get_all_nodes(self, label: str):
        return await self.node_manager.get_all_nodes(label)

    async def add_task_result(self, task: str, result: str, score: float):
        task_node = {
            "id": str(uuid.uuid4()),
            "content": task,
            "result": result,
            "score": score,
            "timestamp": time.time()
        }
        task_result = await self.add_or_update_node("TaskResult", task_node)
        await self.add_relationships_to_concepts(task_result['id'], task)
        return task_result

    async def add_improvement_suggestion(self, improvement: str):
        improvement_node = {
            "id": str(uuid.uuid4()),
            "content": improvement[:1000],
            "timestamp": time.time()
        }
        await self.add_or_update_node("Improvement", improvement_node)

    async def store_tool_usage(self, tool_name: str, subtask: dict, result: dict):
        tool_usage = {
            "id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "subtask": json.dumps(subtask),
            "result": json.dumps(result),
            "timestamp": time.time()
        }
        await self.add_or_update_node("ToolUsage", tool_usage)

    async def store_tool(self, tool_name: str, source_code: str):
        tool_node = {
            "id": str(uuid.uuid4()),
            "name": tool_name,
            "source_code": source_code,
            "created_at": time.time()
        }
        await self.add_or_update_node("Tool", tool_node)

    async def store_compressed_knowledge(self, compressed_knowledge: str):
        compressed_node = {
            "id": str(uuid.uuid4()),
            "content": compressed_knowledge,
            "timestamp": time.time()
        }
        await self.add_or_update_node("CompressedKnowledge", compressed_node)

    async def add_relationships_to_concepts(self, task_id: str, task_content: str):
        concepts = await self.extract_concepts(task_content)
        for concept in concepts:
            concept_node = await self.add_or_update_node("Concept", {"name": concept})
            await self.add_relationship(task_id, concept_node['id'], "RELATES_TO")

    async def extract_concepts(self, content: str):
        # This is a placeholder implementation. You might want to use NLP techniques here.
        return [word.strip() for word in content.split() if len(word.strip()) > 3]

    async def get_relevant_knowledge(self, content: str):
        query = """
        MATCH (n)
        WHERE n.content CONTAINS $content OR n.name CONTAINS $content
        RETURN n
        LIMIT 5
        """
        result = await self.query_executor.execute_query(query, {"content": content})
        return [record['n'] for record in result]

    async def get_system_performance(self):
        query = """
        MATCH (p:Performance)
        WHERE p.timestamp > $timestamp
        RETURN p.metric AS metric, AVG(p.value) AS avg_value
        """
        timestamp = time.time() - 86400  # Get performance data from the last 24 hours
        result = await self.query_executor.execute_query(query, {"timestamp": timestamp})
        return {record["metric"]: record["avg_value"] for record in result}

    async def store_performance_metric(self, metric: str, value: float):
        performance_node = {
            "id": str(uuid.uuid4()),
            "metric": metric,
            "value": value,
            "timestamp": time.time()
        }
        await self.add_or_update_node("Performance", performance_node)

    async def get_tool_usage_history(self, tool_name: str, limit: int = 10):
        query = """
        MATCH (t:ToolUsage {tool_name: $tool_name})
        RETURN t
        ORDER BY t.timestamp DESC
        LIMIT $limit
        """
        result = await self.query_executor.execute_query(query, {"tool_name": tool_name, "limit": limit})
        return [record['t'] for record in result]

    async def export_knowledge(self, file_path: str):
        await self.knowledge_import_export.export_knowledge(file_path)

    async def import_knowledge(self, file_path: str):
        await self.knowledge_import_export.import_knowledge(file_path)

    async def export_knowledge_framework(self, file_path: str):
        await self.knowledge_import_export.export_knowledge_framework(file_path)

    async def import_knowledge_framework(self, file_path: str):
        await self.knowledge_import_export.import_knowledge_framework(file_path)

    async def bootstrap_agent(self, framework_file: str, knowledge_file: str):
        await self.import_knowledge_framework(framework_file)
        await self.import_knowledge(knowledge_file)

    # Add other methods as needed, delegating to the appropriate manager classes