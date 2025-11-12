import time
from typing import List, Optional, Tuple
from .data_structures import TimingPath, EndpointBasedGraph, Node
from .candidate_generator import CandidateGenerator
from .anchor_predictor import AnchorPredictor


class Pipeline:
    def __init__(self, 
                 max_candidates: int = 30,
                 top_k_paths: int = 10,
                 model_type: str = 'rf',
                 n_estimators: int = 1000):
        self.candidate_gen = CandidateGenerator(max_candidates=max_candidates)
        self.anchor_pred = AnchorPredictor(model_type=model_type, n_estimators=n_estimators)
        self.top_k_paths = top_k_paths
        self._trained = False
    
    def train(self, training_data: List[tuple]) -> 'Pipeline':
        self.anchor_pred.train(training_data)
        self._trained = True
        return self
    
    def process_endpoint(self, endpoint_graph: EndpointBasedGraph) -> Tuple[List[TimingPath], dict]:
        start_time = time.time()
        
        candidates = self.candidate_gen.generate(endpoint_graph)
        candidate_gen_time = time.time() - start_time
        
        if not candidates:
            return endpoint_graph.get_top_k_critical_paths(self.top_k_paths), {
                'num_candidates': 0,
                'anchor': None,
                'candidate_gen_time': candidate_gen_time,
                'anchor_pred_time': 0,
                'pruning_time': 0,
                'total_time': time.time() - start_time
            }
        
        anchor_start = time.time()
        anchor = self.anchor_pred.predict(candidates, endpoint_graph)
        anchor_pred_time = time.time() - anchor_start
        
        pruning_start = time.time()
        if anchor:
            pruned_paths = [p for p in endpoint_graph.paths if p.contains_node(anchor)]
        else:
            pruned_paths = endpoint_graph.paths
        
        top_paths = sorted(pruned_paths, key=lambda p: p.slack)[:self.top_k_paths]
        pruning_time = time.time() - pruning_start
        
        total_time = time.time() - start_time
        
        stats = {
            'num_candidates': len(candidates),
            'anchor': anchor,
            'num_pruned_paths': len(pruned_paths),
            'candidate_gen_time': candidate_gen_time,
            'anchor_pred_time': anchor_pred_time,
            'pruning_time': pruning_time,
            'total_time': total_time
        }
        
        return top_paths, stats
    
    def process_baseline(self, endpoint_graph: EndpointBasedGraph) -> Tuple[List[TimingPath], dict]:
        start_time = time.time()
        
        pba_cost_per_path = 0.0001
        simulated_pba_time = len(endpoint_graph.paths) * pba_cost_per_path
        time.sleep(simulated_pba_time)
        
        top_paths = endpoint_graph.get_top_k_critical_paths(self.top_k_paths)
        
        total_time = time.time() - start_time
        
        stats = {
            'method': 'baseline',
            'total_time': total_time,
            'simulated_pba_time': simulated_pba_time,
            'num_paths_analyzed': len(endpoint_graph.paths)
        }
        
        return top_paths, stats
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_feature_importance(self) -> Optional[dict]:
        return self.anchor_pred.get_feature_importance()
