from app.knowledge.base_knowledge_graph import BaseKnowledgeGraph
from app.knowledge.node_manager import NodeManager
from app.knowledge.relationship_manager import RelationshipManager
from app.knowledge.query_executor import QueryExecutor
from app.knowledge.schema_manager import SchemaManager
from app.knowledge.knowledge_import_export import KnowledgeImportExport
from app.entropy.entropy_manager import EntropyManager  # Import EntropyManager
import time
import json
from typing import Dict, Any, List
import uuid

class KnowledgeGraph(BaseKnowledgeGraph):
    def __init__(self, uri, user, password, llm):
        super().__init__(uri, user, password)
        self.query_executor = QueryExecutor(self.driver)
        self.node_manager = NodeManager(self.query_executor)
        self.relationship_manager = RelationshipManager(self.query_executor)
        self.schema_manager = SchemaManager(self.query_executor)
        self.knowledge_import_export = KnowledgeImportExport(self.node_manager, self.schema_manager)
        self.entropy_manager = EntropyManager(llm)  # Initialize EntropyManager

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
        concepts = await self.entropy_manager.extract_concepts(task_content)  # Use EntropyManager for concept extraction
        for concept in concepts:
            concept_node = await self.add_or_update_node("Concept", {"name": concept})
            await self.add_relationship(task_id, concept_node['id'], "RELATES_TO")

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

    async def store_episode(self, task_context: Dict[str, Any]):
        episode_node = {
            "id": str(uuid.uuid4()),
            "task": task_context.get("original_task"),
            "context": json.dumps(task_context),
            "timestamp": time.time()
        }
        await self.add_or_update_node("Episode", episode_node)
        await self.add_relationships_to_concepts(episode_node['id'], task_context.get("original_task"))

    async def recall_relevant_episodes(self, current_context: Dict[str, Any], limit: int = 5):
        query = """
        MATCH (e:Episode)
        WHERE e.task CONTAINS $task
        RETURN e
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        try:
            result = await self.query_executor.execute_query(query, {
                "task": current_context.get("original_task"),
                "limit": limit
            })
            return [record['e'] for record in result]
        except Exception as e:
            self.logging_manager.log_warning(f"Failed to recall relevant episodes: {str(e)}")
            return []

    async def consolidate_memory(self, recent_tasks: List[Dict[str, Any]]):
        consolidated_knowledge = await self.entropy_manager.consolidate_knowledge(recent_tasks)
        await self.add_or_update_node("ConsolidatedKnowledge", {
            "content": json.dumps(consolidated_knowledge),
            "timestamp": time.time()
        })
        
        # Extract concepts from consolidated knowledge and create relationships
        for task in recent_tasks:
            await self.add_relationships_to_concepts(consolidated_knowledge['id'], task.get("task"))

    async def store_meta_learning_weights(self, weights: Dict[str, float]):
        weights_node = {
            "id": str(uuid.uuid4()),
            "weights": json.dumps(weights),
            "timestamp": time.time()
        }
        await self.add_or_update_node("MetaLearningWeights", weights_node)

    async def get_recent_improvement_suggestions(self, limit: int = 50) -> List[str]:
        query = """
        MATCH (i:Improvement)
        RETURN i.content AS suggestion
        ORDER BY i.timestamp DESC
        LIMIT $limit
        """
        result = await self.query_executor.execute_query(query, {"limit": limit})
        return [record['suggestion'] for record in result]

    async def store_improvement_analysis(self, analysis: Dict[str, Any]):
        analysis_node = {
            "id": str(uuid.uuid4()),
            "common_themes": json.dumps(analysis.get("common_themes", [])),
            "priority_improvements": json.dumps(analysis.get("priority_improvements", [])),
            "long_term_strategies": json.dumps(analysis.get("long_term_strategies", [])),
            "timestamp": time.time()
        }
        await self.add_or_update_node("ImprovementAnalysis", analysis_node)

    async def get_latest_meta_learning_insights(self) -> Dict[str, Any]:
        query = """
        MATCH (w:MetaLearningWeights)
        WITH w ORDER BY w.timestamp DESC LIMIT 1
        MATCH (a:ImprovementAnalysis)
        WITH w, a ORDER BY a.timestamp DESC LIMIT 1
        RETURN w.weights AS weights, a.common_themes AS common_themes,
               a.priority_improvements AS priority_improvements,
               a.long_term_strategies AS long_term_strategies
        """
        result = await self.query_executor.execute_query(query)
        if result:
            return {
                "weights": json.loads(result[0]['weights']),
                "common_themes": json.loads(result[0]['common_themes']),
                "priority_improvements": json.loads(result[0]['priority_improvements']),
                "long_term_strategies": json.loads(result[0]['long_term_strategies'])
            }
        return {}

    async def get_old_memories(self, threshold_days: int) -> List[Dict[str, Any]]:
        threshold_timestamp = time.time() - (threshold_days * 24 * 60 * 60)
        query = """
        MATCH (n:Memory)
        WHERE n.timestamp < $threshold
        RETURN n
        """
        result = await self.query_executor.execute_query(query, {"threshold": threshold_timestamp})
        return [record['n'] for record in result]

    async def store_compressed_memory(self, compressed_memory: Dict[str, Any]):
        compressed_data = await self.entropy_manager.compress_memories(compressed_memory)
        await self.add_or_update_node("CompressedMemory", {
            "content": json.dumps(compressed_data),
            "timestamp": time.time()
        })

    async def consolidate_knowledge(self):
        recent_nodes = await self.get_recent_nodes(limit=1000)
        consolidated_knowledge = await self.entropy_manager.consolidate_knowledge(recent_nodes)
        await self.add_or_update_node("ConsolidatedKnowledge", {
            "content": json.dumps(consolidated_knowledge),
            "timestamp": time.time()
        })

    async def store_meta_learning_insights(self, insights: List[Dict[str, Any]]):
        for insight in insights:
            await self.add_or_update_node("MetaLearningInsight", {
                "content": json.dumps(insight),
                "timestamp": time.time()
            })

    async def export_knowledge_subset(self, node_types: List[str], file_path: str):
        await self.knowledge_import_export.export_knowledge_subset(node_types, file_path)

    async def import_knowledge_subset(self, file_path: str, merge_strategy: str = 'update'):
        await self.knowledge_import_export.import_knowledge_subset(file_path, merge_strategy)

    async def compare_knowledge_graphs(self, other_graph_file: str) -> Dict[str, Any]:
        return await self.knowledge_import_export.compare_knowledge_graphs(other_graph_file)

    async def merge_knowledge_graphs(self, other_graph_file: str, merge_strategy: str = 'update'):
        await self.knowledge_import_export.merge_knowledge_graphs(other_graph_file, merge_strategy)

    # Add other methods as needed, delegating to the appropriate manager classes