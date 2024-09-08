import os
import shutil
from typing import List
import logging
import uuid

logger = logging.getLogger(__name__)

class WorkspaceManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.workspace_path = os.path.join(self.base_path, "workspace")
        if not os.path.exists(self.workspace_path):
            os.makedirs(self.workspace_path)
        logger.info(f"WorkspaceManager initialized with workspace path: {self.workspace_path}")

    def create_task_workspace(self) -> str:
        task_id = str(uuid.uuid4())
        task_workspace = os.path.join(self.workspace_path, task_id)
        os.makedirs(task_workspace)
        logger.info(f"Created task workspace: {task_workspace}")
        return task_workspace

    def get_workspace_path(self) -> str:
        return self.workspace_path

    def create_file(self, task_workspace: str, filename: str, content: str) -> str:
        file_path = os.path.join(task_workspace, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        logger.info(f"Created file: {file_path}")
        return file_path

    def read_file(self, task_workspace: str, filename: str) -> str:
        file_path = os.path.join(task_workspace, filename)
        with open(file_path, 'r') as f:
            content = f.read()
        logger.info(f"Read file: {file_path}")
        return content

    def delete_file(self, task_workspace: str, filename: str) -> None:
        file_path = os.path.join(task_workspace, filename)
        try:
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        except PermissionError:
            logger.warning(f"Unable to delete file: {file_path}. It may be in use.")

    def list_files(self, task_workspace: str) -> List[str]:
        return os.listdir(task_workspace)

    def clear_task_workspace(self, task_workspace: str) -> None:
        for filename in os.listdir(task_workspace):
            file_path = os.path.join(task_workspace, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
                logger.info(f"Removed {file_path}")
            except PermissionError:
                logger.warning(f"Unable to remove {file_path}. It may be in use or have restricted permissions.")

    def copy_to_workspace(self, task_workspace: str, source_path: str) -> str:
        destination_path = os.path.join(task_workspace, os.path.basename(source_path))
        try:
            if os.path.isfile(source_path):
                shutil.copy2(source_path, destination_path)
            elif os.path.isdir(source_path):
                shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            logger.info(f"Copied {source_path} to {destination_path}")
            return destination_path
        except PermissionError:
            logger.warning(f"Unable to copy {source_path} to workspace. It may have restricted permissions.")
            return ""