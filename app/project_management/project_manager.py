import os
import git
from typing import List, Dict, Any
from app.workspace.workspace_manager import WorkspaceManager
from app.knowledge.knowledge_graph import KnowledgeGraph

class ProjectManager:
    def __init__(self, workspace_manager: WorkspaceManager, knowledge_graph: KnowledgeGraph):
        self.workspace_manager = workspace_manager
        self.knowledge_graph = knowledge_graph

    async def create_project(self, project_name: str) -> str:
        # Create a new directory for the project
        project_path = os.path.join(self.workspace_manager.virtual_env.base_path, project_name)
        self.workspace_manager.virtual_env.create_directory(project_path)

        # Initialize version control (e.g., Git) here if needed

        # Add project to knowledge graph
        self.knowledge_graph.add_node("Project", {"name": project_name, "path": project_path})

        return f"Project '{project_name}' created successfully"

    async def get_project_status(self, project_name: str) -> dict:
        # Implement logic to get project status
        # This is a placeholder implementation
        return {"name": project_name, "status": "Active"}

    async def commit_changes(self, project_name: str, commit_message: str) -> str:
        # Implement version control commit logic
        # This is a placeholder implementation
        return f"Changes committed for project '{project_name}': {commit_message}"

    async def create_branch(self, project_name: str, branch_name: str) -> str:
        # Implement branch creation logic
        # This is a placeholder implementation
        return f"Branch '{branch_name}' created for project '{project_name}'"

    async def switch_branch(self, project_name: str, branch_name: str) -> str:
        # Implement branch switching logic
        # This is a placeholder implementation
        return f"Switched to branch '{branch_name}' for project '{project_name}'"

    async def generate_documentation(self, project_name: str) -> str:
        # Implement documentation generation logic
        # This is a placeholder implementation
        return f"Documentation generated for project '{project_name}'"

class DocumentationGenerator:
    def generate_documentation(self, project_name: str, project_path: str) -> str:
        documentation = f"# {project_name} Documentation\n\n"
        
        # Generate project structure
        documentation += "## Project Structure\n\n"
        documentation += self._generate_project_structure(project_path)
        
        # Generate module documentation
        documentation += "\n## Modules\n\n"
        documentation += self._generate_module_documentation(project_path)
        
        return documentation

    def _generate_project_structure(self, project_path: str) -> str:
        structure = "```\n"
        for root, dirs, files in os.walk(project_path):
            level = root.replace(project_path, '').count(os.sep)
            indent = ' ' * 4 * level
            structure += f"{indent}{os.path.basename(root)}/\n"
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                structure += f"{sub_indent}{file}\n"
        structure += "```\n"
        return structure

    def _generate_module_documentation(self, project_path: str) -> str:
        documentation = ""
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    module_path = os.path.join(root, file)
                    with open(module_path, 'r') as f:
                        content = f.read()
                    documentation += f"### {file}\n\n"
                    documentation += f"```python\n{content}\n```\n\n"
        return documentation