import logging
import json
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('agi_system.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _log(self, level: int, message: str, context: Dict[str, Any] = None):
        log_entry = {
            "message": message,
            "context": context or {}
        }
        self.logger.log(level, json.dumps(log_entry))

    def debug(self, message: str, context: Dict[str, Any] = None):
        self._log(logging.DEBUG, message, context)

    def info(self, message: str, context: Dict[str, Any] = None):
        self._log(logging.INFO, message, context)

    def warning(self, message: str, context: Dict[str, Any] = None):
        self._log(logging.WARNING, message, context)

    def error(self, message: str, context: Dict[str, Any] = None):
        self._log(logging.ERROR, message, context)

logger = StructuredLogger("AGI_System")