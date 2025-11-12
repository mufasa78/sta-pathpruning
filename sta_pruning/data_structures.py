import numpy as np
from typing import List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class Node:
    node_id: str
    arrival_time: float = 0.0
    required_time: float = 0.0
    slack: float = 0.0
    capacitance: float = 0.0
    transition: float = 0.0
    fanout: int = 0
    depth: int = 0
    
    @property
    def timing_criticality(self) -> float:
        return abs(self.slack)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False


@dataclass
class TimingPath:
    path_id: str
    nodes: List[Node]
    slack: float
    delay: float
    endpoint: str
    startpoint: str = ""
    
    @property
    def length(self) -> int:
        return len(self.nodes)
    
    @property
    def worst_slack(self) -> float:
        return self.slack
    
    def get_node_at_position(self, position: float) -> Optional[Node]:
        if not 0 <= position <= 1:
            return None
        idx = int(position * (len(self.nodes) - 1))
        return self.nodes[idx] if idx < len(self.nodes) else None
    
    def contains_node(self, node: Node) -> bool:
        return node in self.nodes
    
    def get_middle_region_nodes(self, start: float = 0.3, end: float = 0.7) -> List[Node]:
        if len(self.nodes) < 3:
            return self.nodes[:]
        
        start_idx = int(start * (len(self.nodes) - 1))
        end_idx = int(end * (len(self.nodes) - 1))
        
        return self.nodes[start_idx:end_idx + 1]


class EndpointBasedGraph:
    def __init__(self, endpoint_id: str, paths: List[TimingPath]):
        self.endpoint = endpoint_id
        self.paths = paths
        self._node_cache: Optional[Set[Node]] = None
    
    @property
    def num_paths(self) -> int:
        return len(self.paths)
    
    @property
    def worst_slack(self) -> float:
        if not self.paths:
            return 0.0
        return min(path.slack for path in self.paths)
    
    @property
    def all_nodes(self) -> Set[Node]:
        if self._node_cache is None:
            self._node_cache = set()
            for path in self.paths:
                self._node_cache.update(path.nodes)
        return self._node_cache
    
    def get_middle_region_nodes(self, start: float = 0.3, end: float = 0.7) -> List[Node]:
        middle_nodes = set()
        for path in self.paths:
            middle_nodes.update(path.get_middle_region_nodes(start, end))
        return list(middle_nodes)
    
    def get_paths_through_node(self, node: Node) -> List[TimingPath]:
        return [path for path in self.paths if path.contains_node(node)]
    
    def get_top_k_critical_paths(self, k: int = 10) -> List[TimingPath]:
        sorted_paths = sorted(self.paths, key=lambda p: p.slack)
        return sorted_paths[:k]
