import numpy as np
from typing import List, Tuple
from .data_structures import Node, TimingPath, EndpointBasedGraph


class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate_node(self, node_id: str, depth: int = 0) -> Node:
        slack = self.rng.uniform(-1000, 1000)
        arrival_time = self.rng.uniform(0, 10000)
        required_time = arrival_time + slack
        
        return Node(
            node_id=node_id,
            arrival_time=arrival_time,
            required_time=required_time,
            slack=slack,
            capacitance=self.rng.uniform(0.1, 10.0),
            transition=self.rng.uniform(10, 500),
            fanout=self.rng.randint(1, 20),
            depth=depth
        )
    
    def generate_timing_path(self, 
                            path_id: str, 
                            num_nodes: int, 
                            endpoint: str,
                            base_slack: float = None) -> TimingPath:
        if base_slack is None:
            base_slack = self.rng.uniform(-1000, 100)
        
        nodes = []
        for i in range(num_nodes):
            node_id = f"{path_id}_node_{i}"
            node = self.generate_node(node_id, depth=i)
            
            node.slack = base_slack + self.rng.uniform(-50, 50)
            
            nodes.append(node)
        
        delay = self.rng.uniform(1000, 15000)
        
        return TimingPath(
            path_id=path_id,
            nodes=nodes,
            slack=base_slack,
            delay=delay,
            endpoint=endpoint,
            startpoint=f"start_{path_id}"
        )
    
    def generate_endpoint_graph(self, 
                               endpoint_id: str, 
                               num_paths: int = 100,
                               min_path_length: int = 10,
                               max_path_length: int = 50,
                               critical_ratio: float = 0.2) -> EndpointBasedGraph:
        avg_length = (min_path_length + max_path_length) // 2
        num_shared_nodes = int(avg_length * 0.6)
        
        shared_node_pool = []
        for i in range(num_shared_nodes):
            node = self.generate_node(f"{endpoint_id}_shared_{i}", depth=i)
            shared_node_pool.append(node)
        
        paths = []
        num_critical = int(num_paths * critical_ratio)
        
        for i in range(num_paths):
            is_critical = i < num_critical
            path_length = self.rng.randint(min_path_length, max_path_length)
            base_slack = self.rng.uniform(-500, -50) if is_critical else self.rng.uniform(-50, 500)
            
            nodes = []
            num_shared_in_path = int(path_length * self.rng.uniform(0.4, 0.8))
            num_unique = path_length - num_shared_in_path
            
            for j in range(path_length):
                if j < num_unique // 2:
                    node = self.generate_node(f"{endpoint_id}_path{i}_start_{j}", depth=j)
                elif j >= path_length - num_unique // 2:
                    node = self.generate_node(f"{endpoint_id}_path{i}_end_{j}", depth=j)
                else:
                    shared_idx = self.rng.randint(0, len(shared_node_pool))
                    node = shared_node_pool[shared_idx]
                
                node.slack = base_slack + self.rng.uniform(-50, 50)
                nodes.append(node)
            
            path_type = "critical" if is_critical else "normal"
            path = TimingPath(
                path_id=f"{endpoint_id}_{path_type}_{i}",
                nodes=nodes,
                slack=base_slack,
                delay=self.rng.uniform(1000, 15000),
                endpoint=endpoint_id,
                startpoint=f"start_{endpoint_id}_{i}"
            )
            paths.append(path)
        
        return EndpointBasedGraph(endpoint_id, paths)
    
    def generate_design(self, 
                       design_name: str,
                       num_endpoints: int = 10,
                       paths_per_endpoint: Tuple[int, int] = (50, 200)) -> List[EndpointBasedGraph]:
        endpoint_graphs = []
        
        for i in range(num_endpoints):
            endpoint_id = f"{design_name}_endpoint_{i}"
            num_paths = self.rng.randint(paths_per_endpoint[0], paths_per_endpoint[1])
            
            endpoint_graph = self.generate_endpoint_graph(
                endpoint_id,
                num_paths=num_paths,
                min_path_length=15,
                max_path_length=60
            )
            endpoint_graphs.append(endpoint_graph)
        
        return endpoint_graphs
    
    def generate_training_data(self, 
                              num_samples: int = 100) -> List[Tuple[EndpointBasedGraph, List[Node], int]]:
        training_data = []
        
        for i in range(num_samples):
            endpoint_graph = self.generate_endpoint_graph(
                f"train_ep_{i}",
                num_paths=self.rng.randint(50, 150)
            )
            
            candidates = endpoint_graph.get_middle_region_nodes()[:30]
            
            if not candidates:
                continue
            
            critical_paths = endpoint_graph.get_top_k_critical_paths(10)
            
            best_anchor_idx = 0
            best_score = -float('inf')
            
            for idx, node in enumerate(candidates):
                paths_through = endpoint_graph.get_paths_through_node(node)
                
                coverage = len(paths_through) / max(endpoint_graph.num_paths, 1)
                
                critical_coverage = sum(1 for p in paths_through if p in critical_paths)
                critical_ratio = critical_coverage / max(len(critical_paths), 1)
                
                avg_slack = np.mean([p.slack for p in paths_through]) if paths_through else 0
                
                score = coverage * 0.5 + critical_ratio * 0.4 + (-avg_slack / 1000) * 0.1
                
                if score > best_score:
                    best_score = score
                    best_anchor_idx = idx
            
            training_data.append((endpoint_graph, candidates, best_anchor_idx))
        
        return training_data
    
    def generate_benchmark_designs(self, num_designs: int = 5) -> List[Tuple[str, List[EndpointBasedGraph]]]:
        designs = []
        design_configs = [
            ("small_design", 5, (30, 80)),
            ("medium_design", 10, (50, 150)),
            ("large_design", 15, (100, 250)),
            ("xlarge_design", 20, (150, 300)),
            ("xxlarge_design", 25, (200, 400))
        ]
        
        for i in range(min(num_designs, len(design_configs))):
            name, num_endpoints, paths_range = design_configs[i]
            endpoint_graphs = []
            
            for j in range(num_endpoints):
                endpoint_id = f"{name}_ep_{j}"
                num_paths = self.rng.randint(paths_range[0], paths_range[1])
                endpoint_graph = self.generate_endpoint_graph(endpoint_id, num_paths=num_paths)
                endpoint_graphs.append(endpoint_graph)
            
            designs.append((name, endpoint_graphs))
        
        return designs
