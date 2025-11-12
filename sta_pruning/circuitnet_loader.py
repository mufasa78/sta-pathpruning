
import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from .data_structures import Node, TimingPath, EndpointBasedGraph


class CircuitNetLoader:
    """
    Loader for CircuitNet dataset timing and graph data.
    Supports both CircuitNet-N14 and CircuitNet-N28 formats.
    Also includes synthetic data generation that mimics real CircuitNet structure.
    """
    
    def __init__(self, data_path: str, use_synthetic: bool = False):
        self.data_path = Path(data_path)
        self.graph_cache = {}
        self.use_synthetic = use_synthetic
        
    def generate_synthetic_circuitnet_design(
        self,
        design_name: str,
        num_endpoints: int = 100,
        paths_per_endpoint_range: Tuple[int, int] = (50, 200),
        num_nodes_range: Tuple[int, int] = (15, 50)
    ) -> List[EndpointBasedGraph]:
        """
        Generate synthetic data that mimics CircuitNet design structure.
        This creates realistic circuit timing data without downloading huge files.
        
        Args:
            design_name: Name of the synthetic design (e.g., "ariane", "mempool")
            num_endpoints: Number of endpoints in the design
            paths_per_endpoint_range: Min/max paths per endpoint
            num_nodes_range: Min/max nodes per path
        """
        endpoint_graphs = []
        
        # Realistic design parameters based on CircuitNet
        design_params = {
            'ariane': {'scale': 1.5, 'complexity': 'high'},
            'mempool': {'scale': 2.0, 'complexity': 'very_high'},
            'nvdla': {'scale': 1.8, 'complexity': 'high'},
            'black_parrot': {'scale': 1.3, 'complexity': 'medium'},
            'default': {'scale': 1.0, 'complexity': 'medium'}
        }
        
        params = design_params.get(design_name.lower(), design_params['default'])
        scale = params['scale']
        
        for ep_idx in range(num_endpoints):
            endpoint_id = f"{design_name}_endpoint_{ep_idx:04d}"
            
            num_paths = int(np.random.randint(*paths_per_endpoint_range) * scale)
            paths = []
            
            # Generate realistic critical slack distribution
            base_slack = np.random.uniform(-500, 100)  # ps
            
            for path_idx in range(num_paths):
                num_nodes = np.random.randint(*num_nodes_range)
                nodes = []
                
                # Generate path with realistic timing characteristics
                total_delay = 0
                path_slack = base_slack + np.random.normal(0, 50)
                
                for node_idx in range(num_nodes):
                    node_id = f"{design_name}_n{ep_idx}_{path_idx}_{node_idx}"
                    
                    # Realistic delay distribution (gate + wire)
                    gate_delay = np.random.gamma(2, 20)  # ps
                    wire_delay = np.random.gamma(1.5, 15)  # ps
                    node_delay = gate_delay + wire_delay
                    
                    total_delay += node_delay
                    arrival_time = total_delay
                    required_time = arrival_time + path_slack
                    
                    # Realistic physical parameters
                    capacitance = np.random.gamma(2, 0.5) * 1e-15  # fF
                    transition = np.random.gamma(3, 30)  # ps
                    fanout = int(np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05]))
                    
                    node = Node(
                        node_id=node_id,
                        arrival_time=arrival_time,
                        required_time=required_time,
                        slack=path_slack,
                        capacitance=capacitance,
                        transition=transition,
                        fanout=fanout,
                        depth=node_idx
                    )
                    nodes.append(node)
                
                if nodes:
                    path = TimingPath(
                        path_id=f"{endpoint_id}_path_{path_idx:04d}",
                        nodes=nodes,
                        slack=path_slack,
                        delay=total_delay,
                        endpoint=endpoint_id,
                        startpoint=f"{design_name}_input_{path_idx % 20}"
                    )
                    paths.append(path)
            
            if paths:
                endpoint_graph = EndpointBasedGraph(endpoint_id, paths)
                endpoint_graphs.append(endpoint_graph)
        
        return endpoint_graphs
    
    def list_available_designs(self) -> List[str]:
        """List all available designs (synthetic or real)."""
        if self.use_synthetic:
            # Return synthetic design names that mimic real CircuitNet designs
            return [
                'ariane133',
                'ariane136', 
                'mempool_tile',
                'nvdla',
                'black_parrot',
                'cva6',
                'swerv',
                'rocket'
            ]
        
        # Check for real data
        designs = set()
        
        graph_dir = self.data_path / "graph"
        if graph_dir.exists():
            for file in graph_dir.glob("*.pkl"):
                designs.add(file.stem)
        
        timing_dir = self.data_path / "timing"
        if timing_dir.exists():
            for file in timing_dir.glob("*.json"):
                designs.add(file.stem)
        
        raw_dir = self.data_path / "raw"
        if raw_dir.exists():
            for design_dir in raw_dir.iterdir():
                if design_dir.is_dir():
                    designs.add(design_dir.name)
        
        if not designs:
            # Fallback to synthetic
            return self.list_available_designs()
        
        return sorted(list(designs))
    
    def load_design(self, design_name: str) -> List[EndpointBasedGraph]:
        """
        Load a design (synthetic or real).
        """
        if self.use_synthetic:
            # Generate synthetic data
            return self.generate_synthetic_circuitnet_design(
                design_name,
                num_endpoints=50,  # Smaller for faster generation
                paths_per_endpoint_range=(80, 150)
            )
        
        # Try to load real data
        graph_features = self.load_graph_features(design_name)
        timing_features = self.load_timing_features(design_name)
        
        if not graph_features or not timing_features:
            print(f"Real data not found for {design_name}, using synthetic data")
            return self.generate_synthetic_circuitnet_design(
                design_name,
                num_endpoints=50,
                paths_per_endpoint_range=(80, 150)
            )
        
        return self.convert_to_endpoint_graph(design_name, graph_features, timing_features)
    
    def load_batch_designs(
        self, 
        design_names: List[str] = None, 
        max_designs: int = None
    ) -> List[Tuple[str, List[EndpointBasedGraph]]]:
        """
        Load multiple designs (synthetic or real).
        """
        if design_names is None:
            design_names = self.list_available_designs()
        
        if max_designs:
            design_names = design_names[:max_designs]
        
        batch_data = []
        
        for design_name in design_names:
            print(f"Loading design: {design_name}")
            endpoint_graphs = self.load_design(design_name)
            
            if endpoint_graphs:
                batch_data.append((design_name, endpoint_graphs))
                print(f"  Loaded {len(endpoint_graphs)} endpoints")
        
        return batch_data
    
    # Keep original methods for real data loading
    def load_graph_features(self, design_name: str) -> Optional[Dict]:
        """Load pre-built graph features from CircuitNet format."""
        possible_paths = [
            self.data_path / "graph" / f"{design_name}.pkl",
            self.data_path / "raw" / design_name / "graph.pkl",
            self.data_path / f"{design_name}.pkl"
        ]
        
        for graph_path in possible_paths:
            if graph_path.exists():
                try:
                    with open(graph_path, 'rb') as f:
                        graph_data = pickle.load(f)
                    return graph_data
                except Exception as e:
                    print(f"Error loading graph from {graph_path}: {e}")
                    continue
        
        return None
    
    def load_timing_features(self, design_name: str) -> Optional[Dict]:
        """Load timing features from CircuitNet dataset."""
        timing_path = self.data_path / "timing" / f"{design_name}.json"
        
        if not timing_path.exists():
            timing_path = self.data_path / "features" / f"{design_name}_timing.json"
        
        if not timing_path.exists():
            return None
            
        try:
            with open(timing_path, 'r') as f:
                timing_data = json.load(f)
            return timing_data
        except Exception as e:
            print(f"Error loading timing data for {design_name}: {e}")
            return None
    
    def convert_to_endpoint_graph(
        self,
        design_name: str,
        graph_features: Dict,
        timing_features: Dict
    ) -> List[EndpointBasedGraph]:
        """Convert CircuitNet graph and timing data to EndpointBasedGraph format."""
        endpoint_graphs = []
        
        node_features = graph_features.get('node_features', {})
        timing_paths = timing_features.get('paths', [])
        
        endpoints = {}
        
        for path_data in timing_paths:
            endpoint_id = path_data.get('endpoint', f"{design_name}_ep_unknown")
            
            if endpoint_id not in endpoints:
                endpoints[endpoint_id] = []
            
            nodes = []
            node_sequence = path_data.get('nodes', [])
            
            for idx, node_id in enumerate(node_sequence):
                node_attrs = node_features.get(node_id, {})
                
                arrival_time = node_attrs.get('arrival_time', path_data.get('delay', 0) * idx / len(node_sequence))
                required_time = node_attrs.get('required_time', arrival_time + path_data.get('slack', 0))
                slack = required_time - arrival_time
                
                node = Node(
                    node_id=str(node_id),
                    arrival_time=float(arrival_time),
                    required_time=float(required_time),
                    slack=float(slack),
                    capacitance=float(node_attrs.get('capacitance', 1.0)),
                    transition=float(node_attrs.get('transition', 100.0)),
                    fanout=int(node_attrs.get('fanout', 1)),
                    depth=idx
                )
                nodes.append(node)
            
            if nodes:
                path = TimingPath(
                    path_id=path_data.get('path_id', f"{endpoint_id}_path_{len(endpoints[endpoint_id])}"),
                    nodes=nodes,
                    slack=float(path_data.get('slack', nodes[-1].slack)),
                    delay=float(path_data.get('delay', sum(n.arrival_time for n in nodes))),
                    endpoint=endpoint_id,
                    startpoint=path_data.get('startpoint', str(node_sequence[0]) if node_sequence else 'unknown')
                )
                endpoints[endpoint_id].append(path)
        
        for endpoint_id, paths in endpoints.items():
            if paths:
                endpoint_graph = EndpointBasedGraph(endpoint_id, paths)
                endpoint_graphs.append(endpoint_graph)
        
        return endpoint_graphs
