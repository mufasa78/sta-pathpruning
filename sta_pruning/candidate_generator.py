import numpy as np
from typing import List
from .data_structures import Node, EndpointBasedGraph


class CandidateGenerator:
    def __init__(self, max_candidates: int = 30, middle_start: float = 0.3, middle_end: float = 0.7):
        self.max_candidates = max_candidates
        self.middle_start = middle_start
        self.middle_end = middle_end
    
    def generate(self, endpoint_graph: EndpointBasedGraph) -> List[Node]:
        middle_nodes = endpoint_graph.get_middle_region_nodes(self.middle_start, self.middle_end)
        
        if not middle_nodes:
            return []
        
        if len(middle_nodes) <= self.max_candidates:
            return middle_nodes
        
        scored_nodes = []
        for node in middle_nodes:
            paths_through = endpoint_graph.get_paths_through_node(node)
            
            coverage = len(paths_through) / max(endpoint_graph.num_paths, 1)
            
            avg_slack = np.mean([p.slack for p in paths_through]) if paths_through else 0
            criticality = -avg_slack
            
            score = coverage * 0.6 + (criticality / max(abs(criticality), 1e-6)) * 0.4
            
            scored_nodes.append((node, score))
        
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, score in scored_nodes[:self.max_candidates]]
