import numpy as np
from typing import List
from .data_structures import Node, TimingPath, EndpointBasedGraph


class FeatureExtractor:
    def __init__(self):
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        path_features = [
            'avg_path_slack',
            'min_path_slack',
            'max_path_slack',
            'std_path_slack',
            'avg_path_delay',
            'num_critical_paths',
            'critical_path_ratio'
        ]
        
        node_features = [
            'node_slack',
            'node_arrival_time',
            'node_required_time',
            'node_capacitance',
            'node_transition',
            'node_fanout',
            'node_depth',
            'paths_through_node',
            'avg_slack_paths_through',
            'min_slack_paths_through',
            'position_in_critical_path',
            'criticality_score',
            'upstream_criticality',
            'downstream_criticality'
        ]
        
        return path_features + node_features
    
    def extract_path_features(self, paths: List[TimingPath]) -> np.ndarray:
        if not paths:
            return np.zeros(7)
        
        slacks = np.array([p.slack for p in paths])
        delays = np.array([p.delay for p in paths])
        
        critical_threshold = np.percentile(slacks, 10) if len(slacks) > 0 else 0
        num_critical = np.sum(slacks <= critical_threshold)
        
        features = [
            np.mean(slacks),
            np.min(slacks),
            np.max(slacks),
            np.std(slacks) if len(slacks) > 1 else 0.0,
            np.mean(delays),
            num_critical,
            num_critical / len(paths) if len(paths) > 0 else 0.0
        ]
        
        return np.array(features, dtype=np.float32)
    
    def extract_node_features(self, node: Node, endpoint_graph: EndpointBasedGraph) -> np.ndarray:
        paths_through = endpoint_graph.get_paths_through_node(node)
        
        if not paths_through:
            return np.zeros(14)
        
        slacks = [p.slack for p in paths_through]
        
        position_in_critical = 0.5
        if paths_through:
            critical_path = min(paths_through, key=lambda p: p.slack)
            if node in critical_path.nodes:
                idx = critical_path.nodes.index(node)
                position_in_critical = idx / max(len(critical_path.nodes) - 1, 1)
        
        criticality_score = len(paths_through) / max(endpoint_graph.num_paths, 1)
        
        upstream_crit = np.mean([p.slack for p in paths_through[:len(paths_through)//2]]) if len(paths_through) > 1 else node.slack
        downstream_crit = np.mean([p.slack for p in paths_through[len(paths_through)//2:]]) if len(paths_through) > 1 else node.slack
        
        features = [
            node.slack,
            node.arrival_time,
            node.required_time,
            node.capacitance,
            node.transition,
            float(node.fanout),
            float(node.depth),
            len(paths_through),
            np.mean(slacks),
            np.min(slacks),
            position_in_critical,
            criticality_score,
            upstream_crit,
            downstream_crit
        ]
        
        return np.array(features, dtype=np.float32)
    
    def extract(self, node: Node, endpoint_graph: EndpointBasedGraph) -> np.ndarray:
        path_features = self.extract_path_features(endpoint_graph.get_paths_through_node(node))
        node_features = self.extract_node_features(node, endpoint_graph)
        
        return np.concatenate([path_features, node_features])
    
    def extract_batch(self, nodes: List[Node], endpoint_graph: EndpointBasedGraph) -> np.ndarray:
        if not nodes:
            return np.array([]).reshape(0, 21)
        
        features = [self.extract(node, endpoint_graph) for node in nodes]
        return np.array(features, dtype=np.float32)
