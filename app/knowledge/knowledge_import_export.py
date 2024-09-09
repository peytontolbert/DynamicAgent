from app.utils.logger import StructuredLogger
import json
import os
from typing import List, Dict, Any
import time
logger = StructuredLogger("KnowledgeImportExport")

class KnowledgeImportExport:
    def __init__(self, node_manager, schema_manager):
        self.node_manager = node_manager
        self.schema_manager = schema_manager

    async def export_knowledge(self, file_path: str):
        all_knowledge = await self.node_manager.get_all_nodes(None)  # Get all nodes
        
        serializable_knowledge = []
        for item in all_knowledge:
            serializable_item = {}
            for key, value in item.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    serializable_item[key] = value
                else:
                    serializable_item[key] = str(value)
            serializable_knowledge.append(serializable_item)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_knowledge, f, indent=2)
        logger.info(f"Exported knowledge to {file_path}")

    async def import_knowledge(self, file_path: str):
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        with open(file_path, 'r') as f:
            imported_knowledge = json.load(f)

        for item in imported_knowledge:
            label = item.pop('label', 'Concept')
            await self.node_manager.add_or_update_node(label, item)

        logger.info(f"Imported knowledge from {file_path}")

    async def export_knowledge_framework(self, file_path: str):
        schema = await self.schema_manager.get_schema()
        with open(file_path, 'w') as f:
            json.dump(schema, f, indent=2)
        logger.info(f"Exported knowledge framework to {file_path}")

    async def import_knowledge_framework(self, file_path: str):
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        with open(file_path, 'r') as f:
            schema = json.load(f)

        for node_type, properties in schema.items():
            if node_type != 'relationship':
                await self.schema_manager.create_node_constraint(node_type)
                for prop in properties['properties']:
                    await self.schema_manager.create_property_index(node_type, prop)

        logger.info(f"Imported and applied knowledge framework from {file_path}")

    async def export_knowledge_subset(self, node_types: List[str], file_path: str):
        all_knowledge = []
        for node_type in node_types:
            nodes = await self.node_manager.get_all_nodes(node_type)
            all_knowledge.extend(nodes)
        
        # Include relationships
        relationships = await self.relationship_manager.get_relationships_for_nodes(all_knowledge)
        
        export_data = {
            "version": "1.0",
            "nodes": self._serialize_knowledge(all_knowledge),
            "relationships": relationships,
            "metadata": {
                "export_date": time.time(),
                "node_types": node_types
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        logger.info(f"Exported knowledge subset to {file_path}")

    async def import_knowledge_subset(self, file_path: str, merge_strategy: str = 'update'):
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        with open(file_path, 'r') as f:
            imported_data = json.load(f)

        # Version check
        if imported_data.get("version") != "1.0":
            logger.warning(f"Importing data with version {imported_data.get('version')}. Current version is 1.0.")

        for item in imported_data["nodes"]:
            label = item.pop('label', 'Concept')
            await self._import_node(label, item, merge_strategy)

        # Import relationships
        for rel in imported_data["relationships"]:
            await self.relationship_manager.add_relationship(
                rel['start_node'], rel['end_node'], rel['type'], rel.get('properties', {})
            )

        logger.info(f"Imported knowledge subset from {file_path}")

    async def _import_node(self, label: str, item: Dict[str, Any], merge_strategy: str):
        if merge_strategy == 'update':
            await self.node_manager.add_or_update_node(label, item)
        elif merge_strategy == 'skip_existing':
            existing_node = await self.node_manager.get_node(label, item['id'])
            if not existing_node:
                await self.node_manager.add_or_update_node(label, item)
        elif merge_strategy == 'merge':
            existing_node = await self.node_manager.get_node(label, item['id'])
            if existing_node:
                merged_item = self._merge_nodes(existing_node, item)
                await self.node_manager.add_or_update_node(label, merged_item)
            else:
                await self.node_manager.add_or_update_node(label, item)

    def _merge_nodes(self, existing_node: Dict[str, Any], new_node: Dict[str, Any]) -> Dict[str, Any]:
        # Implement your merging logic here
        # This is a simple example; you might need more complex logic
        merged = existing_node.copy()
        for key, value in new_node.items():
            if key not in merged or merged[key] != value:
                merged[key] = value
        return merged

    async def compare_knowledge_graphs(self, other_graph_file: str) -> Dict[str, Any]:
        with open(other_graph_file, 'r') as f:
            other_graph = json.load(f)

        current_graph = await self.node_manager.get_all_nodes(None)
        
        comparison = {
            "only_in_current": [],
            "only_in_other": [],
            "different": [],
            "identical": []
        }

        # Implement comparison logic here
        # ...

        return comparison

    async def merge_knowledge_graphs(self, other_graph_file: str, merge_strategy: str = 'update'):
        comparison = await self.compare_knowledge_graphs(other_graph_file)

        if merge_strategy == 'update':
            for node in comparison['only_in_other'] + comparison['different']:
                await self.node_manager.add_or_update_node(node['label'], node)
        elif merge_strategy == 'keep_current':
            for node in comparison['only_in_other']:
                await self.node_manager.add_or_update_node(node['label'], node)

        logger.info(f"Merged knowledge graph from {other_graph_file}")

    def _serialize_knowledge(self, knowledge_list):
        serializable_knowledge = []
        for item in knowledge_list:
            serializable_item = {}
            for key, value in item.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    serializable_item[key] = value
                else:
                    serializable_item[key] = str(value)
            serializable_knowledge.append(serializable_item)
        return serializable_knowledge

    async def export_knowledge_framework(self, file_path: str):
        schema = await self.schema_manager.get_schema()
        with open(file_path, 'w') as f:
            json.dump(schema, f, indent=2)
        logger.info(f"Exported knowledge framework to {file_path}")

    async def import_knowledge_framework(self, file_path: str):
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        with open(file_path, 'r') as f:
            schema = json.load(f)

        for node_type, properties in schema.items():
            if node_type != 'relationship':
                await self.schema_manager.create_node_constraint(node_type)
                for prop in properties['properties']:
                    await self.schema_manager.create_property_index(node_type, prop)

        logger.info(f"Imported and applied knowledge framework from {file_path}")