# API Documentation

Complete API reference for the STA Path Pruning System.

## Core Data Structures

### Node
```python
from sta_pruning import Node

node = Node(
    node_id="n1",
    arrival_time=1.5,
    required_time=2.0,
    slack=0.5,
    capacitance=0.01,
    transition=0.05,
    fanout=3,
    depth=5
)
```

**Attributes:**
- `node_id` (str): Unique identifier
- `arrival_time` (float): Signal arrival time
- `required_time` (float): Required arrival time
- `slack` (float): Timing slack (required - arrival)
- `capacitance` (float): Node capacitance
- `transition` (float): Transition time
- `fanout` (int): Number of fanout connections
- `depth` (int): Depth in timing path

### TimingPath
```python
from sta_pruning import TimingPath, Node

path = TimingPath(
    path_id="path_1",
    nodes=[node1, node2, node3],
    startpoint="FF1",
    endpoint="FF2"
)
```

**Attributes:**
- `path_id` (str): Unique path identifier
- `nodes` (List[Node]): Ordered list of nodes
- `startpoint` (str): Starting flip-flop/port
- `endpoint` (str): Ending flip-flop/port

**Properties:**
- `slack` (float): Minimum slack along path
- `delay` (float): Total path delay
- `length` (int): Number of nodes in path

### EndpointBasedGraph
```python
from sta_pruning import EndpointBasedGraph, TimingPath

graph = EndpointBasedGraph(
    endpoint_id="endpoint_1",
    paths=[path1, path2, path3]
)
```

**Attributes:**
- `endpoint` (str): Endpoint identifier
- `paths` (List[TimingPath]): All paths to this endpoint

**Properties:**
- `num_paths` (int): Total number of paths
- `worst_slack` (float): Minimum slack across all paths
- `all_nodes` (Set[str]): All unique node IDs

**Methods:**
- `get_middle_region_nodes(start=0.3, end=0.7)`: Extract nodes from middle region

## Pipeline

### Pipeline
Main orchestration class for the complete algorithm.

```python
from sta_pruning import Pipeline

# Create pipeline
pipeline = Pipeline(
    max_candidates=30,
    top_k=10,
    model_type='rf',
    n_estimators=500
)

# Train on data
training_data = [(endpoint_graph, optimal_anchor), ...]
pipeline.train(training_data)

# Process endpoint
top_paths, stats = pipeline.process_endpoint(endpoint_graph)
```

**Parameters:**
- `max_candidates` (int): Maximum candidate nodes (default: 30)
- `top_k` (int): Number of top paths to return (default: 10)
- `model_type` (str): 'rf' or 'xgb' (default: 'rf')
- `n_estimators` (int): Number of trees/estimators (default: 1000)

**Methods:**
- `train(training_data)`: Train the ML model
- `process_endpoint(graph)`: Run ML-accelerated analysis
- `process_baseline(graph)`: Run exhaustive baseline
- `get_feature_importance()`: Get feature importance dict

**Returns (process_endpoint):**
- `top_paths` (List[TimingPath]): Top K critical paths
- `stats` (dict): Performance statistics
  - `total_time`: Total processing time
  - `candidate_gen_time`: Stage 1 time
  - `anchor_pred_time`: Stage 2 time
  - `pruning_time`: Stage 3 time
  - `num_candidates`: Number of candidates
  - `num_pruned_paths`: Paths after pruning
  - `anchor`: Selected anchor node

## Feature Extraction

### FeatureExtractor
Extracts 21-dimensional feature vectors.

```python
from sta_pruning import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract(node, endpoint_graph)
```

**Feature Dimensions:**

**Path Features (7):**
1. Average path slack
2. Minimum path slack
3. Maximum path slack
4. Standard deviation of slack
5. Average path delay
6. Number of critical paths
7. Critical path ratio

**Node Features (14):**
1. Node slack
2. Arrival time
3. Required time
4. Capacitance
5. Transition time
6. Fanout count
7. Depth in path
8. Paths through node
9. Average slack of paths through
10. Minimum slack of paths through
11. Position in critical path
12. Criticality score
13. Upstream criticality
14. Downstream criticality

## Candidate Generation

### CandidateGenerator
Stage 1: Generate anchor candidates.

```python
from sta_pruning import CandidateGenerator

generator = CandidateGenerator(
    max_candidates=30,
    middle_start=0.3,
    middle_end=0.7
)

candidates = generator.generate(endpoint_graph)
```

**Parameters:**
- `max_candidates` (int): Maximum candidates to return
- `middle_start` (float): Start of middle region (0.0-1.0)
- `middle_end` (float): End of middle region (0.0-1.0)

**Returns:**
- List of (node, score) tuples sorted by score

## Anchor Prediction

### AnchorPredictor
Stage 2: ML-based anchor selection.

```python
from sta_pruning import AnchorPredictor

predictor = AnchorPredictor(
    model_type='rf',
    n_estimators=500
)

predictor.train(training_data)
anchor = predictor.predict(candidates, endpoint_graph)
```

**Parameters:**
- `model_type` (str): 'rf' (Random Forest) or 'xgb' (XGBoost)
- `n_estimators` (int): Number of trees/estimators

**Methods:**
- `train(training_data)`: Train classifier
- `predict(candidates, graph)`: Predict best anchor
- `get_feature_importance()`: Get feature importance

## Data Generation

### SyntheticDataGenerator
Generate synthetic circuit data for testing.

```python
from sta_pruning import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)

# Single endpoint
endpoint = generator.generate_endpoint_graph(
    "ep1",
    num_paths=150,
    min_path_length=15,
    max_path_length=50
)

# Full design
design = generator.generate_design(
    "design1",
    num_endpoints=10,
    paths_per_endpoint=(50, 200)
)

# Training data
training_data = generator.generate_training_data(100)

# Benchmark suite
designs = generator.generate_benchmark_designs(5)
```

## Evaluation

### Evaluator
Benchmark and evaluate performance.

```python
from sta_pruning import Evaluator

evaluator = Evaluator()

# Single endpoint
result = evaluator.benchmark_endpoint(
    endpoint_graph,
    pipeline,
    "design_name"
)

# Multiple designs
results_df = evaluator.benchmark_multiple_designs(
    designs,
    pipeline
)

# Summary statistics
summary = evaluator.get_summary_statistics()
```

**Metrics:**
- `mse`: Mean squared error
- `mae`: Mean absolute error
- `speedup`: Runtime speedup factor
- `path_overlap`: Fraction of overlapping paths
- `num_paths`: Total paths analyzed
- `num_candidates`: Candidates evaluated

## Visualization

### Visualizer
Create interactive visualizations.

```python
from sta_pruning import Visualizer

viz = Visualizer()

# Timing path graph
fig = viz.create_timing_path_graph(
    endpoint_graph,
    paths,
    anchor_node=anchor,
    max_paths=10
)

# Slack distribution
fig = viz.create_slack_distribution(
    endpoint_graph,
    baseline_paths=baseline,
    proposed_paths=proposed
)

# Anchor analysis
fig = viz.create_anchor_analysis(
    endpoint_graph,
    candidates,
    selected_anchor
)
```

## Model Tuning

### ModelTuner
Hyperparameter optimization.

```python
from sta_pruning import ModelTuner

tuner = ModelTuner(model_type='rf')

# Cross-validation
cv_results = tuner.cross_validate(training_data, cv=5)

# Grid search
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [10, 20, None]
}
grid_results = tuner.grid_search_cv(
    training_data,
    param_grid=param_grid,
    cv=3
)

# Random search
random_results = tuner.random_search_cv(
    training_data,
    n_iter=20,
    cv=3
)
```

## Batch Processing

### BatchProcessor
Large-scale parallel processing.

```python
from sta_pruning import BatchProcessor

processor = BatchProcessor(pipeline, n_workers=4)

# Sequential processing
results = processor.process_batch_sequential(endpoint_graphs)

# Parallel processing (threads)
results = processor.process_batch_parallel(
    endpoint_graphs,
    use_processes=False
)

# Parallel processing (processes)
results = processor.process_batch_parallel(
    endpoint_graphs,
    use_processes=True
)

# Statistics
stats = processor.get_batch_statistics(results)
```

## Report Generation

### ReportGenerator
Generate benchmark reports.

```python
from sta_pruning import ReportGenerator

report_gen = ReportGenerator()

# Markdown report
markdown = report_gen.generate_markdown_report(
    results_df,
    summary_stats
)

# Save to file
with open('report.md', 'w') as f:
    f.write(markdown)

# CSV export
results_df.to_csv('results.csv', index=False)
```

## CircuitNet Integration

### CircuitNetLoader
Load CircuitNet dataset.

```python
from sta_pruning import CircuitNetLoader

# Synthetic mode (no download needed)
loader = CircuitNetLoader(
    "./circuitnet_data",
    use_synthetic=True
)

# Real data mode
loader = CircuitNetLoader(
    "./circuitnet_data",
    use_synthetic=False
)

# List available designs
designs = loader.list_available_designs()

# Load single design
endpoint_graphs = loader.load_design("design_name")

# Load multiple designs
batch_data = loader.load_batch_designs(
    ["design1", "design2", "design3"]
)
```

## Model Training

### ModelTrainer
Prepare training data and train models.

```python
from sta_pruning import ModelTrainer

trainer = ModelTrainer(
    model_type='rf',
    n_estimators=1000
)

# Prepare training data from endpoint graphs
training_data = trainer.prepare_training_data(
    endpoint_graphs
)

# Train model
model = trainer.train(training_data)
```

## Complete Example

```python
from sta_pruning import (
    SyntheticDataGenerator,
    Pipeline,
    Evaluator
)

# 1. Generate test data
data_gen = SyntheticDataGenerator(seed=42)
endpoint_graph = data_gen.generate_endpoint_graph(
    "test_endpoint",
    num_paths=200
)

# 2. Create and train pipeline
pipeline = Pipeline(model_type='rf', n_estimators=500)
training_data = data_gen.generate_training_data(50)
pipeline.train(training_data)

# 3. Analyze endpoint
top_paths, stats = pipeline.process_endpoint(endpoint_graph)

# 4. Evaluate performance
evaluator = Evaluator()
result = evaluator.benchmark_endpoint(
    endpoint_graph,
    pipeline,
    "test_design"
)

# 5. Print results
print(f"MSE: {result['mse']:.2e}")
print(f"MAE: {result['mae']:.2e}")
print(f"Speedup: {result['speedup']:.2f}×")
print(f"Path Overlap: {result['path_overlap']*100:.1f}%")
print(f"Processing Time: {stats['total_time']*1000:.2f} ms")
print(f"Anchor Node: {stats['anchor'].node_id}")
```

## Error Handling

All functions raise standard Python exceptions:
- `ValueError`: Invalid parameters or data
- `TypeError`: Incorrect types
- `RuntimeError`: Processing errors

Example:
```python
try:
    pipeline.train(training_data)
except ValueError as e:
    print(f"Training failed: {e}")
```

## Performance Tips

1. **Model Selection**: Random Forest is faster, XGBoost slightly more accurate
2. **Candidate Count**: 30 candidates balances accuracy and speed
3. **Training Size**: 50-100 samples sufficient for good performance
4. **Batch Processing**: Use parallel mode for >100 endpoints
5. **Caching**: Reuse trained pipelines across multiple analyses
