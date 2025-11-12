# Documentation Index

Complete documentation for the STA Path Pruning System.

## Getting Started

### New Users
1. **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
   - Installation steps
   - First analysis
   - Common tasks
   - Dashboard overview

2. **[README.md](README.md)** - Complete project overview
   - System architecture
   - Algorithm pipeline
   - Performance metrics
   - Usage examples

### Developers
3. **[API.md](API.md)** - Complete API reference
   - All classes and methods
   - Parameter descriptions
   - Return values
   - Code examples

4. **[CHANGELOG.md](CHANGELOG.md)** - Version history
   - Feature additions
   - Technical details
   - Release notes

## Project Structure

```
sta-pathpruning/
├── README.md              # Main documentation
├── QUICKSTART.md          # Quick start guide
├── API.md                 # API reference
├── CHANGELOG.md           # Version history
├── DOCUMENTATION.md       # This file
├── pyproject.toml         # Project configuration
├── app.py                 # Streamlit dashboard
├── main.py                # CLI entry point
│
├── sta_pruning/           # Core package
│   ├── __init__.py
│   ├── data_structures.py      # Node, TimingPath, EndpointBasedGraph
│   ├── feature_extractor.py    # 21-dim feature extraction
│   ├── candidate_generator.py  # Stage 1: Candidate selection
│   ├── anchor_predictor.py     # Stage 2: ML prediction
│   ├── pipeline.py             # End-to-end orchestration
│   ├── evaluate.py             # Benchmarking
│   ├── data_generator.py       # Synthetic data
│   ├── train.py                # Model training
│   ├── visualizer.py           # Visualizations
│   ├── model_tuner.py          # Hyperparameter tuning
│   ├── batch_processor.py      # Parallel processing
│   ├── report_generator.py     # Report generation
│   └── circuitnet_loader.py    # Dataset integration
│
├── circuitnet_data/       # Dataset directory
│   ├── README.md          # Dataset documentation
│   ├── raw/               # Original data
│   ├── graph/             # Processed graphs
│   ├── timing/            # Timing features
│   └── features/          # Extracted features
│
├── .streamlit/            # Streamlit configuration
│   └── config.toml
│
└── .venv/                 # Virtual environment

```

## Documentation by Topic

### Installation & Setup
- [QUICKSTART.md](QUICKSTART.md) - Quick installation
- [README.md](README.md#installation) - Detailed setup
- [README.md](README.md#troubleshooting) - Common issues

### Usage
- [QUICKSTART.md](QUICKSTART.md#your-first-analysis) - First steps
- [README.md](README.md#usage) - Dashboard and API usage
- [API.md](API.md#complete-example) - Complete examples

### Algorithm
- [README.md](README.md#algorithm-pipeline) - Pipeline overview
- [README.md](README.md#architecture) - System architecture
- [API.md](API.md#feature-extraction) - Feature details

### API Reference
- [API.md](API.md#core-data-structures) - Data structures
- [API.md](API.md#pipeline) - Pipeline API
- [API.md](API.md#evaluation) - Evaluation API
- [API.md](API.md#visualization) - Visualization API

### Advanced Features
- [README.md](README.md#advanced-features) - Feature overview
- [API.md](API.md#model-tuning) - Hyperparameter tuning
- [API.md](API.md#batch-processing) - Parallel processing
- [API.md](API.md#circuitnet-integration) - Dataset integration

### Performance
- [README.md](README.md#performance-metrics) - Metrics overview
- [README.md](README.md#performance-optimization-tips) - Optimization
- [API.md](API.md#performance-tips) - API-level tips

### Dashboard
- [README.md](README.md#dashboard-features) - Feature list
- [QUICKSTART.md](QUICKSTART.md#dashboard-features) - Quick overview
- Tab-by-tab guide in the dashboard itself

## Quick Reference

### Installation
```bash
uv venv
uv pip install -e .
.venv\Scripts\activate
streamlit run app.py
```

### Basic Usage
```python
from sta_pruning import SyntheticDataGenerator, Pipeline

data_gen = SyntheticDataGenerator()
endpoint = data_gen.generate_endpoint_graph("test", num_paths=150)

pipeline = Pipeline(model_type='rf')
training_data = data_gen.generate_training_data(50)
pipeline.train(training_data)

top_paths, stats = pipeline.process_endpoint(endpoint)
```

### Key Metrics
- **Accuracy:** MAE < 1.6e-04 (sub-picosecond)
- **Speed:** 1.3-1.7× speedup (30% faster)
- **Coverage:** >90% path overlap

## Documentation Standards

### Code Examples
All code examples in the documentation:
- Are tested and working
- Include necessary imports
- Show expected output
- Follow Python best practices

### API Documentation
Each API function includes:
- Purpose description
- Parameter types and defaults
- Return value description
- Usage example
- Error conditions

### Versioning
- Version numbers follow semantic versioning (MAJOR.MINOR.PATCH)
- Changes documented in [CHANGELOG.md](CHANGELOG.md)
- Breaking changes clearly marked

## Contributing to Documentation

When adding features:
1. Update [API.md](API.md) with new APIs
2. Add examples to [README.md](README.md)
3. Update [CHANGELOG.md](CHANGELOG.md)
4. Add quick reference to [QUICKSTART.md](QUICKSTART.md) if applicable
5. Update this index if adding new documentation files

## External Resources

### Research Background
- Static Timing Analysis (STA) fundamentals
- Path-Based Analysis (PBA) techniques
- Machine learning for EDA

### Related Tools
- Synopsys PrimeTime (commercial STA tool)
- Cadence Tempus (commercial STA tool)
- OpenSTA (open-source STA tool)

### Datasets
- CircuitNet: Large-scale circuit dataset
- Hugging Face: circuitnet/CircuitNet-N14

## Support

### Getting Help
1. Check [QUICKSTART.md](QUICKSTART.md) for common tasks
2. Review [README.md](README.md#troubleshooting) for issues
3. Consult [API.md](API.md) for detailed API info
4. Check the **Documentation** tab in the dashboard

### Reporting Issues
When reporting issues, include:
- Python version
- Operating system
- Error messages
- Minimal reproducible example
- Expected vs actual behavior

## Version Information

- **Current Version:** 1.0.0
- **Python Required:** 3.11+
- **Last Updated:** November 12, 2025

## License

This implementation is for research and educational purposes.

---

**Navigation:**
- [← Back to README](README.md)
- [Quick Start →](QUICKSTART.md)
- [API Reference →](API.md)
