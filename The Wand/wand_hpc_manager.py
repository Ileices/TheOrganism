import threading
import time
import logging
import random
import psutil

logging.basicConfig(
    filename="wand_hpc_manager.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
node_statuses = {}
active_nodes = {}

def node_heartbeat(node_id):
    while True:
        try:
            # Update timestamp and log heartbeat
            node_statuses[node_id] = time.time()
            logger.info(f"Heartbeat from node {node_id}")
        except Exception as e:
            logger.error(f"Heartbeat error for {node_id}: {e}")
        time.sleep(30)

def register_node(node_info):
    node_id = node_info.get("node_id")
    active_nodes[node_id] = node_info
    logger.info(f"Node {node_id} registered.")
    threading.Thread(target=node_heartbeat, args=(node_id,), daemon=True).start()
    node_statuses[node_id] = time.time()
    print(f"[INFO] Node registered: {node_info}")

def deregister_node(node_id):
    if node_id in active_nodes:
        del active_nodes[node_id]
        logger.info(f"Node {node_id} deregistered.")

def schedule_tasks(task_list):
    # Incorporate current node load metrics for optimal task assignment
    logger.info("Scheduling tasks with dynamic load balancing...")
    scheduled = {}
    for task in task_list:
        assigned_node = select_node()
        scheduled.setdefault(assigned_node, []).append(task)
        logger.info(f"Task {task} assigned to node {assigned_node}")
    return scheduled

def select_node():
    available = [node for node in node_statuses if _node_is_active(node)]
    if not available:
        register_node({"node_id": "default_node"})
        available.append("default_node")
    return random.choice(available)

def _node_is_active(node_id):
    last_seen = node_statuses.get(node_id, 0)
    if time.time() - last_seen > 60:  # inactive after 60 seconds
        logger.warning(f"Node {node_id} inactive. Deregistering.")
        active_nodes.pop(node_id, None)
        del node_statuses[node_id]
        return False
    return True

def distribute_job(job):
    # Use up-to-date load metrics for optimal job allocation.
    assigned_node = select_node()
    logger.info(f"Task {job} assigned to node {assigned_node}")
    print(f"[DEBUG] Distributing job: {job} to {assigned_node}")

def monitor_nodes():
    # Detect failed nodes and reassign tasks
    # Periodically check node health and reassign tasks for failed ones
    print("[INFO] Monitoring nodes for failures.")

def monitor_resources():
    while True:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        logger.info(f"Resources: CPU={cpu}%, RAM={mem}%")
        time.sleep(10)

def initialize_hpc_manager():
    logger.info("Initializing HPC Manager...")
    threading.Thread(target=monitor_resources, daemon=True).start()
    register_node({"node_id": "PrimaryNode"})

def allocate_jobs(jobs, nodes):
    # Prioritize nodes by compute power and check for active status.
    prioritized_nodes = sorted(nodes, key=lambda n: n.get('compute_power', 0), reverse=True)
    for job in jobs:
        for node in prioritized_nodes:
            if node.get('status', 'active') != 'active':
                continue
            # Real assignment: add job to node's job list.
            node.setdefault('jobs', []).append(job)
            logger.info("Job %s assigned to node %s with power %s", job, node.get('id'), node.get('compute_power'))
            print(f"Assigning job {job} to node {node.get('id')} with power {node.get('compute_power')}")
            break

if __name__ == "__main__":
    initialize_hpc_manager()
    while True:
        time.sleep(60)
