import time
import logging
from queue import PriorityQueue
from typing import Any, Tuple, Optional
from threading import Lock

class QueueManager:
    def __init__(self):
        self.task_queue = PriorityQueue()
        self.queue_lock = Lock()
        self.logger = logging.getLogger(__name__)
        
    def add_task(self, priority: int, task_id: str, payload: Any):
        """Add task to priority queue"""
        with self.queue_lock:
            entry_time = time.time()
            self.task_queue.put((priority, entry_time, task_id, payload))
            self.logger.info(f"Added task {task_id} with priority {priority}")
            
    def get_next_task(self) -> Optional[Tuple[int, float, str, Any]]:
        """Get next task from queue with timeout"""
        try:
            return self.task_queue.get(timeout=1.0)
        except Exception:
            return None
            
    def get_queue_status(self) -> list:
        """Get current queue status"""
        with self.queue_lock:
            # Create a list of tasks without removing them
            tasks = []
            while not self.task_queue.empty():
                task = self.task_queue.get()
                tasks.append(task)
                self.task_queue.put(task)
            return tasks
