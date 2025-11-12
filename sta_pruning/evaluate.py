import numpy as np
import pandas as pd
from typing import List, Tuple
from .data_structures import TimingPath, EndpointBasedGraph
from .pipeline import Pipeline


class Evaluator:
    def __init__(self):
        self.results = []
    
    def compute_mse(self, baseline_paths: List[TimingPath], proposed_paths: List[TimingPath]) -> float:
        if not baseline_paths or not proposed_paths:
            return 0.0
        
        baseline_slacks = np.array([p.slack for p in baseline_paths])
        proposed_slacks = np.array([p.slack for p in proposed_paths])
        
        min_len = min(len(baseline_slacks), len(proposed_slacks))
        
        return np.mean((baseline_slacks[:min_len] - proposed_slacks[:min_len]) ** 2)
    
    def compute_mae(self, baseline_paths: List[TimingPath], proposed_paths: List[TimingPath]) -> float:
        if not baseline_paths or not proposed_paths:
            return 0.0
        
        baseline_slacks = np.array([p.slack for p in baseline_paths])
        proposed_slacks = np.array([p.slack for p in proposed_paths])
        
        min_len = min(len(baseline_slacks), len(proposed_slacks))
        
        return np.mean(np.abs(baseline_slacks[:min_len] - proposed_slacks[:min_len]))
    
    def compute_path_overlap(self, baseline_paths: List[TimingPath], proposed_paths: List[TimingPath]) -> float:
        if not baseline_paths or not proposed_paths:
            return 0.0
        
        baseline_ids = set(p.path_id for p in baseline_paths)
        proposed_ids = set(p.path_id for p in proposed_paths)
        
        intersection = len(baseline_ids & proposed_ids)
        union = len(baseline_ids | proposed_ids)
        
        return intersection / max(union, 1)
    
    def benchmark_endpoint(self, 
                          endpoint_graph: EndpointBasedGraph, 
                          pipeline: Pipeline,
                          design_name: str = "unknown") -> dict:
        baseline_paths, baseline_stats = pipeline.process_baseline(endpoint_graph)
        
        proposed_paths, proposed_stats = pipeline.process_endpoint(endpoint_graph)
        
        mse = self.compute_mse(baseline_paths, proposed_paths)
        mae = self.compute_mae(baseline_paths, proposed_paths)
        path_overlap = self.compute_path_overlap(baseline_paths, proposed_paths)
        
        speedup = baseline_stats['total_time'] / max(proposed_stats['total_time'], 1e-6)
        
        result = {
            'design': design_name,
            'endpoint': endpoint_graph.endpoint,
            'num_paths': endpoint_graph.num_paths,
            'worst_slack': endpoint_graph.worst_slack,
            'mse': mse,
            'mae': mae,
            'path_overlap': path_overlap,
            'speedup': speedup,
            'baseline_time': baseline_stats['total_time'],
            'proposed_time': proposed_stats['total_time'],
            'num_candidates': proposed_stats.get('num_candidates', 0),
            'num_pruned_paths': proposed_stats.get('num_pruned_paths', 0)
        }
        
        self.results.append(result)
        return result
    
    def benchmark_multiple_designs(self, 
                                   designs: List[Tuple[str, List[EndpointBasedGraph]]], 
                                   pipeline: Pipeline) -> pd.DataFrame:
        all_results = []
        
        for design_name, endpoint_graphs in designs:
            for endpoint_graph in endpoint_graphs:
                result = self.benchmark_endpoint(endpoint_graph, pipeline, design_name)
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        return df
    
    def get_summary_statistics(self) -> dict:
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            'avg_mse': df['mse'].mean(),
            'avg_mae': df['mae'].mean(),
            'avg_speedup': df['speedup'].mean(),
            'avg_path_overlap': df['path_overlap'].mean(),
            'total_endpoints': len(df),
            'avg_baseline_time': df['baseline_time'].mean(),
            'avg_proposed_time': df['proposed_time'].mean(),
            'min_speedup': df['speedup'].min(),
            'max_speedup': df['speedup'].max()
        }
        
        return summary
    
    def clear_results(self):
        self.results = []
