# Quick Start Guide

Get up and running with STA Path Pruning in 5 minutes!

## Installation (2 minutes)

### Step 1: Install UV
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Setup Project
```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -e .

# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

## Run the Dashboard (30 seconds)

```bash
streamlit run app.py
```

Visit `http://localhost:8502` in your browser!

## Your First Analysis (2 minutes)

### Option 1: Use the Dashboard
1. Go to the **Interactive Demo** tab
2. Adjust sliders for circuit parameters
3. Click **Generate & Analyze Circuit**
4. View results and visualizations!

### Option 2: Use Python API

```python
from sta_pruning import SyntheticDataGenerator, Pipeline

# Generate test circuit
data_gen = SyntheticDataGenerator(seed=42)
endpoint = data_gen.generate_endpoint_graph("test", num_paths=150)

# Create and train pipeline
pipeline = Pipeline(model_type='rf')
training_data = data_gen.generate_training_data(50)
pipeline.train(training_data)

# Analyze!
top_paths, stats = pipeline.process_endpoint(endpoint)

print(f"Found {len(top_paths)} critical paths")
print(f"Processing time: {stats['total_time']*1000:.2f} ms")
print(f"Anchor node: {stats['anchor'].node_id}")
```

## Common Tasks

### Run Benchmarks
```python
from sta_pruning import Evaluator

evaluator = Evaluator()
result = evaluator.benchmark_endpoint(endpoint, pipeline, "design1")

print(f"Speedup: {result['speedup']:.2f}×")
print(f"Accuracy (MAE): {result['mae']:.2e}")
```

### Tune Hyperparameters
```python
from sta_pruning import ModelTuner

tuner = ModelTuner(model_type='rf')
cv_results = tuner.cross_validate(training_data, cv=5)

print(f"Mean F1 Score: {cv_results['mean_score']:.4f}")
```

### Batch Processing
```python
from sta_pruning import BatchProcessor

processor = BatchProcessor(pipeline, n_workers=4)
results = processor.process_batch_parallel(endpoint_graphs)

stats = processor.get_batch_statistics(results)
print(f"Throughput: {stats['total_endpoints']/stats['total_time']:.2f} eps/s")
```

### Generate Reports
```python
from sta_pruning import ReportGenerator

report_gen = ReportGenerator()
markdown = report_gen.generate_markdown_report(results_df, summary)

with open('report.md', 'w') as f:
    f.write(markdown)
```

## Dashboard Features

### 9 Interactive Tabs:

1. **Overview** - System architecture and metrics
2. **Interactive Demo** - Real-time circuit analysis
3. **Advanced Visualizations** - Graphs and distributions
4. **Benchmark Results** - Multi-design evaluation
5. **Model Tuning** - Hyperparameter optimization
6. **Batch Processing** - Large-scale analysis
7. **Model Analysis** - Feature importance
8. **CircuitNet Dataset** - Real circuit data
9. **Documentation** - Complete reference

## Troubleshooting

### Port Already in Use?
```bash
streamlit run app.py --server.port 8503
```

### Import Errors?
```bash
# Make sure virtual environment is activated
.venv\Scripts\activate

# Reinstall if needed
uv pip install -e .
```

### Slow Performance?
- Use Random Forest instead of XGBoost
- Reduce number of candidates (try 20 instead of 30)
- Use parallel processing for batch jobs

## Next Steps

- Read the full [README.md](README.md) for detailed information
- Check [API.md](API.md) for complete API reference
- Explore the dashboard tabs for advanced features
- Try different model configurations
- Run benchmarks on your own data

## Key Metrics to Watch

- **MAE < 1.6e-04**: Sub-picosecond accuracy ✓
- **Speedup 1.3-1.7×**: 30% faster than baseline ✓
- **Path Overlap >90%**: High coverage ✓

## Getting Help

- Check the **Documentation** tab in the dashboard
- Review examples in [API.md](API.md)
- Read troubleshooting section in [README.md](README.md)

Happy analyzing! ⚡
