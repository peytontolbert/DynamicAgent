from app.utils.logger import StructuredLogger
import json
import os

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