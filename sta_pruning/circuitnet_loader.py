
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
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.graph_cache = {}
        
    def load_graph_features(self, design_name: str) -> Optional[Dict]:
        """Load pre-built graph features from CircuitNet format."""
        # Try multiple possible paths
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
    
    def parse_def_file(self, def_path: str) -> Dict:
        """Parse DEF file to extract circuit topology information."""
        nets = {}
        pins = {}
        components = {}
        
        try:
            with open(def_path, 'r') as f:
                current_section = None
                
                for line in f:
                    line = line.strip()
                    
                    if line.startswith('NETS'):
                        current_section = 'NETS'
                    elif line.startswith('COMPONENTS'):
                        current_section = 'COMPONENTS'
                    elif line.startswith('PINS'):
                        current_section = 'PINS'
                    elif line.startswith('END'):
                        current_section = None
                    
                    if current_section == 'NETS' and line.startswith('-'):
                        net_name = line.split()[1]
                        nets[net_name] = []
                    elif current_section == 'COMPONENTS' and line.startswith('-'):
                        parts = line.split()
                        comp_name = parts[1]
                        components[comp_name] = {'type': parts[2] if len(parts) > 2 else 'unknown'}
                    elif current_section == 'PINS' and line.startswith('-'):
                        parts = line.split()
                        pin_name = parts[1]
                        pins[pin_name] = {'direction': parts[3] if len(parts) > 3 else 'unknown'}
        
        except Exception as e:
            print(f"Error parsing DEF file: {e}")
        
        return {
            'nets': nets,
            'components': components,
            'pins': pins
        }
    
    def convert_to_endpoint_graph(self, 
                                  design_name: str,
                                  graph_features: Dict,
                                  timing_features: Dict) -> List[EndpointBasedGraph]:
        """
        Convert CircuitNet graph and timing data to EndpointBasedGraph format.
        """
        endpoint_graphs = []
        
        node_features = graph_features.get('node_features', {})
        edge_list = graph_features.get('edge_list', [])
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
    
    def load_design(self, design_name: str) -> List[EndpointBasedGraph]:
        """
        Load a complete design from CircuitNet and convert to endpoint graphs.
        """
        graph_features = self.load_graph_features(design_name)
        timing_features = self.load_timing_features(design_name)
        
        if not graph_features:
            print(f"Warning: No graph features found for {design_name}")
            return []
        
        if not timing_features:
            print(f"Warning: No timing features found for {design_name}")
            return []
        
        return self.convert_to_endpoint_graph(design_name, graph_features, timing_features)
    
    def list_available_designs(self) -> List[str]:
        """List all available designs in the CircuitNet dataset."""
        designs = set()
        
        # Check graph directory
        graph_dir = self.data_path / "graph"
        if graph_dir.exists():
            for file in graph_dir.glob("*.pkl"):
                designs.add(file.stem)
        
        # Check timing directory
        timing_dir = self.data_path / "timing"
        if timing_dir.exists():
            for file in timing_dir.glob("*.json"):
                designs.add(file.stem)
        
        # Check raw data directory (CircuitNet-N14 format)
        raw_dir = self.data_path / "raw"
        if raw_dir.exists():
            for design_dir in raw_dir.iterdir():
                if design_dir.is_dir():
                    designs.add(design_dir.name)
        
        return sorted(list(designs))
    
    def load_batch_designs(self, design_names: List[str] = None, max_designs: int = None) -> List[Tuple[str, List[EndpointBasedGraph]]]:
        """
        Load multiple designs from CircuitNet.
        If design_names is None, loads all available designs.
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
            else:
                print(f"  Warning: No data loaded for {design_name}")
        
        return batch_data
