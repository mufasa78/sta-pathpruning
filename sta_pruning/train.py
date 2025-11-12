import numpy as np
from typing import List, Tuple
from .data_structures import EndpointBasedGraph, Node
from .anchor_predictor import AnchorPredictor
from .candidate_generator import CandidateGenerator


class ModelTrainer:
    def __init__(self, model_type: str = 'rf', n_estimators: int = 1000):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.predictor = None
    
    def prepare_training_data(self, 
                             endpoint_graphs: List[EndpointBasedGraph]) -> List[Tuple]:
        candidate_gen = CandidateGenerator(max_candidates=30)
        training_data = []
        
        for endpoint_graph in endpoint_graphs:
            candidates = candidate_gen.generate(endpoint_graph)
            
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
    
    def train(self, training_data: List[Tuple]) -> AnchorPredictor:
        self.predictor = AnchorPredictor(
            model_type=self.model_type,
            n_estimators=self.n_estimators
        )
        
        self.predictor.train(training_data)
        
        return self.predictor
    
    def train_from_endpoint_graphs(self, 
                                  endpoint_graphs: List[EndpointBasedGraph]) -> AnchorPredictor:
        training_data = self.prepare_training_data(endpoint_graphs)
        return self.train(training_data)
    
    def get_predictor(self) -> AnchorPredictor:
        return self.predictor
