import logging
from logging.handlers import RotatingFileHandler
import time
from typing import Dict, Any

class LoggingManager:
    def __init__(self, log_file: str = "app.log", max_size: int = 1024 * 1024, backup_count: int = 5):
        self.logger = logging.getLogger("LLMAgentSystem")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        file_handler = RotatingFileHandler(log_file, maxBytes=max_size, backupCount=backup_count)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_info(self, message: str):
        self.logger.info(message)

    def log_warning(self, message: str):
        self.logger.warning(message)

    def log_error(self, message: str):
        self.logger.error(message)

    def log_debug(self, message: str):
        self.logger.debug(message)

class PerformanceMonitor:
    def __init__(self, logging_manager: LoggingManager):
        self.logging_manager = logging_manager
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def start_timer(self, operation: str):
        self.start_times[operation] = time.time()

    def stop_timer(self, operation: str):
        if operation in self.start_times:
            elapsed_time = time.time() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = {"count": 0, "total_time": 0, "avg_time": 0}
            
            self.metrics[operation]["count"] += 1
            self.metrics[operation]["total_time"] += elapsed_time
            self.metrics[operation]["avg_time"] = self.metrics[operation]["total_time"] / self.metrics[operation]["count"]

            self.logging_manager.log_debug(f"Operation '{operation}' took {elapsed_time:.4f} seconds")
            del self.start_times[operation]

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        return self.metrics

    def log_metrics(self):
        for operation, data in self.metrics.items():
            self.logging_manager.log_info(f"Performance metrics for '{operation}': {data}")