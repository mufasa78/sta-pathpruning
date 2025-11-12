import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
from .data_structures import EndpointBasedGraph, TimingPath
from .pipeline import Pipeline


class BatchProcessor:
    def __init__(self, pipeline: Pipeline, n_workers: int = 4):
        self.pipeline = pipeline
        self.n_workers = n_workers
    
    def process_single_endpoint(self, 
                                endpoint_graph: EndpointBasedGraph) -> Tuple[str, List[TimingPath], Dict]:
        top_paths, stats = self.pipeline.process_endpoint(endpoint_graph)
        return endpoint_graph.endpoint, top_paths, stats
    
    def process_batch_sequential(self, 
                                 endpoint_graphs: List[EndpointBasedGraph]) -> List[Tuple]:
        results = []
        start_time = time.time()
        
        for idx, endpoint_graph in enumerate(endpoint_graphs):
            endpoint_id, top_paths, stats = self.process_single_endpoint(endpoint_graph)
            results.append((endpoint_id, top_paths, stats))
            
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = (len(endpoint_graphs) - idx - 1) * avg_time
                print(f"Processed {idx + 1}/{len(endpoint_graphs)} endpoints. "
                      f"Est. remaining: {remaining:.1f}s")
        
        return results
    
    def process_batch_parallel(self, 
                               endpoint_graphs: List[EndpointBasedGraph],
                               use_processes: bool = False) -> List[Tuple]:
        results = []
        start_time = time.time()
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.n_workers) as executor:
            future_to_endpoint = {
                executor.submit(self.process_single_endpoint, ep_graph): ep_graph.endpoint
                for ep_graph in endpoint_graphs
            }
            
            completed = 0
            total = len(endpoint_graphs)
            
            for future in as_completed(future_to_endpoint):
                endpoint_id = future_to_endpoint[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 100 == 0:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed
                        remaining = (total - completed) * avg_time
                        print(f"Processed {completed}/{total} endpoints. "
                              f"Est. remaining: {remaining:.1f}s")
                except Exception as exc:
                    print(f"Endpoint {endpoint_id} generated an exception: {exc}")
        
        return results
    
    def process_batch_chunked(self,
                             endpoint_graphs: List[EndpointBasedGraph],
                             chunk_size: int = 100) -> List[Tuple]:
        all_results = []
        total_endpoints = len(endpoint_graphs)
        
        for chunk_start in range(0, total_endpoints, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_endpoints)
            chunk = endpoint_graphs[chunk_start:chunk_end]
            
            print(f"Processing chunk {chunk_start}-{chunk_end}/{total_endpoints}")
            
            chunk_results = self.process_batch_sequential(chunk)
            all_results.extend(chunk_results)
        
        return all_results
    
    def benchmark_batch(self,
                       endpoint_graphs: List[EndpointBasedGraph],
                       baseline_pipeline: Pipeline = None) -> Dict[str, Any]:
        start_time = time.time()
        
        proposed_results = self.process_batch_sequential(endpoint_graphs)
        proposed_time = time.time() - start_time
        
        baseline_results = []
        baseline_time = 0
        
        if baseline_pipeline:
            start_time = time.time()
            for endpoint_graph in endpoint_graphs:
                top_paths, stats = baseline_pipeline.process_baseline(endpoint_graph)
                baseline_results.append((endpoint_graph.endpoint, top_paths, stats))
            baseline_time = time.time() - start_time
        
        total_endpoints = len(endpoint_graphs)
        avg_proposed_time = proposed_time / max(total_endpoints, 1)
        avg_baseline_time = baseline_time / max(total_endpoints, 1) if baseline_time > 0 else 0
        
        speedup = baseline_time / proposed_time if proposed_time > 0 else 0
        
        return {
            'total_endpoints': total_endpoints,
            'proposed_total_time': proposed_time,
            'baseline_total_time': baseline_time,
            'avg_proposed_time': avg_proposed_time,
            'avg_baseline_time': avg_baseline_time,
            'speedup': speedup,
            'throughput_proposed': total_endpoints / proposed_time if proposed_time > 0 else 0,
            'throughput_baseline': total_endpoints / baseline_time if baseline_time > 0 else 0
        }
    
    def get_batch_statistics(self, batch_results: List[Tuple]) -> Dict[str, Any]:
        if not batch_results:
            return {}
        
        total_time = sum(stats['total_time'] for _, _, stats in batch_results)
        num_candidates_list = [stats.get('num_candidates', 0) for _, _, stats in batch_results]
        num_pruned_list = [stats.get('num_pruned_paths', 0) for _, _, stats in batch_results]
        
        return {
            'total_endpoints': len(batch_results),
            'total_time': total_time,
            'avg_time_per_endpoint': total_time / len(batch_results),
            'avg_candidates': np.mean(num_candidates_list),
            'avg_pruned_paths': np.mean(num_pruned_list),
            'min_time': min(stats['total_time'] for _, _, stats in batch_results),
            'max_time': max(stats['total_time'] for _, _, stats in batch_results),
            'std_time': np.std([stats['total_time'] for _, _, stats in batch_results])
        }
