import threading
import queue
import time

class HPCScheduler:
    def __init__(self):
        self.job_queue = queue.PriorityQueue()
        self.lock = threading.Lock()

    def schedule_job(self, priority: int, job: callable, *args, **kwargs):
        # The lower the number, the higher the priority.
        self.job_queue.put((priority, (job, args, kwargs)))
        print(f"Job scheduled with priority {priority}")

    def run(self):
        # Executes jobs in a priority order, supporting dynamic reallocation if needed.
        while not self.job_queue.empty():
            priority, (job, args, kwargs) = self.job_queue.get()
            print(f"Executing job with priority {priority}")
            job(*args, **kwargs)
            time.sleep(0.1)  # simulate processing delay

# ...existing code or test usage...
if __name__ == "__main__":
    scheduler = HPCScheduler()
    
    def example_job(message):
        print(f"Job executed: {message}")
        
    scheduler.schedule_job(1, example_job, "High priority task")
    scheduler.schedule_job(5, example_job, "Low priority task")
    scheduler.run()
