import time, threading, json
from typing import Dict, Any, List

class NodeRegistryService:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        threading.Thread(target=periodic_persistence, args=(self,), daemon=True).start()

    def register_node(self, node_id: str, info: Dict[str, Any]):
        with self.lock:
            self.nodes[node_id] = {"info": info, "last_seen": time.time()}
            self._persist_metadata()

    def update_node(self, node_id: str, info: Dict[str, Any]):
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id]["info"].update(info)
                self.nodes[node_id]["last_seen"] = time.time()
                self._persist_metadata()

    def deregister_node(self, node_id: str):
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self._persist_metadata()

    def get_active_nodes(self, timeout: int = 60) -> List[str]:
        current_time = time.time()
        with self.lock:
            return [node_id for node_id, data in self.nodes.items() if current_time - data["last_seen"] <= timeout]

    def get_all_nodes(self) -> List[str]:
        with self.lock:
            return list(self.nodes.keys())

    def _persist_metadata(self):
        # Persist metadata to a file for continuity.
        with open("node_metadata.json", "w") as f:
            json.dump(self.nodes, f)

def heartbeat_check(registry: NodeRegistryService):
    # Send heartbeat signals; remove nodes that do not respond.
    for node in registry.get_all_nodes():
        # Assume node_heartbeat() returns False if node fails.
        if not node_heartbeat(node):
            registry.deregister_node(node)

def node_heartbeat(node):
    # Simulate heartbeat check.
    # In production, ping the node or check a heartbeat timestamp.
    return True

def periodic_persistence(registry, interval=60):
    import time
    while True:
        with registry.lock:
            registry._persist_metadata()
            logger.info("Periodic metadata persistence complete.")
        time.sleep(interval)

def repair_node_allocation(node_id):
    """
    Attempt to reassign a node that has failed.
    """
    print(f"[REPAIR] Reassigning node allocation for {node_id}.")
    # ...repair logic...
    return True

def track_compute_clusters(clusters):
    """
    Log summary of compute clusters.
    """
    for cluster in clusters:
        print(f"[CLUSTER] {cluster} status: Active")
    return clusters

def analyze_node_metadata(node_metadata):
    """
    Analyze node metadata to suggest load balancing actions.
    """
    analysis = {"average_load": sum(node_metadata.values()) / len(node_metadata)}
    print(f"[ANALYSIS] Computed average node load: {analysis['average_load']}")
    return analysis

def get_node_load_indicator(node_metadata):
    """
    Calculate and return a load indicator for a node.
    """
    load = node_metadata.get("cpu_usage", 0) + node_metadata.get("memory_usage", 0)
    status = "overloaded" if load > 150 else "normal"
    print(f"[LOAD INDICATOR] Node load: {load} ({status}).")
    return {"load": load, "status": status}

def multi_cluster_tracking(clusters):
    """
    Enhance tracking by logging the status of each compute cluster.
    """
    for cluster in clusters:
        print(f"[CLUSTER TRACKING] Cluster {cluster.get('id')} - Nodes: {len(cluster.get('nodes', []))}")
    # ...further aggregation logic...

def recommend_load_balance_action(cluster_metadata):
    """
    Analyze cluster metadata and recommend load balancing adjustments.
    """
    avg_load = sum(cluster_metadata.values()) / len(cluster_metadata) if cluster_metadata else 0
    if avg_load > 100:
        recommendation = "Redirect new jobs to additional nodes."
    elif avg_load < 50:
        recommendation = "Consolidate nodes to save energy."
    else:
        recommendation = "Maintain current distribution."
    print(f"[LOAD BALANCE] Recommendation: {recommendation}")
    return recommendation

def real_time_node_health_dashboard(registry):
    """
    Aggregate node health metrics into a JSON summary for dashboard display.
    """
    dashboard = {}
    with registry.lock:
        for node_id, data in registry.nodes.items():
            health = {
                "last_seen": data.get("last_seen"),
                "cpu_usage": data.get("info", {}).get("cpu", 50),
                "memory_usage": data.get("info", {}).get("memory", 50)
            }
            dashboard[node_id] = health
    import json
    dashboard_json = json.dumps(dashboard, indent=2)
    print("[NODE HEALTH DASHBOARD]\n", dashboard_json)
    return dashboard_json

if __name__ == "__main__":
    registry = NodeRegistryService()
    registry.register_node("Node_A", {"ip": "192.168.1.2"})
    print("Active nodes:", registry.get_active_nodes())
