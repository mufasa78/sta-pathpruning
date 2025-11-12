# Contributing Guide

Thank you for your interest in contributing to the STA Path Pruning System!

## Development Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd sta-pathpruning

# Install UV if not already installed
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment
uv venv

# Install in development mode
uv pip install -e .

# Activate environment
.venv\Scripts\activate
```

### 2. Verify Installation
```bash
python -c "from sta_pruning import *; print('All imports successful!')"
streamlit run app.py
```

## Project Structure

```
sta_pruning/
├── data_structures.py      # Core data types
├── feature_extractor.py    # Feature engineering
├── candidate_generator.py  # Stage 1 algorithm
├── anchor_predictor.py     # Stage 2 ML model
├── pipeline.py            # Main orchestration
├── evaluate.py            # Benchmarking
├── data_generator.py      # Synthetic data
├── train.py              # Model training
├── visualizer.py         # Plotly visualizations
├── model_tuner.py        # Hyperparameter tuning
├── batch_processor.py    # Parallel processing
├── report_generator.py   # Report generation
└── circuitnet_loader.py  # Dataset integration
```

## Coding Standards

### Python Style
- Follow PEP 8 style guide
- Use type hints for function parameters and returns
- Maximum line length: 100 characters
- Use descriptive variable names

### Example:
```python
def extract_features(node: Node, graph: EndpointBasedGraph) -> np.ndarray:
    """
    Extract 21-dimensional feature vector for a node.
    
    Args:
        node: The timing node to extract features from
        graph: The endpoint-based graph containing the node
        
    Returns:
        Feature vector as numpy array of shape (21,)
    """
    features = []
    # Implementation...
    return np.array(features)
```

### Documentation
- All public functions must have docstrings
- Include parameter types and descriptions
- Provide usage examples for complex functions
- Update API.md when adding new public APIs

### Testing
- Write tests for new features
- Ensure existing tests pass
- Test with both Random Forest and XGBoost models
- Verify performance metrics meet targets

## Adding New Features

### 1. Core Algorithm Changes

If modifying the algorithm pipeline:

1. Update relevant module in `sta_pruning/`
2. Add tests to verify correctness
3. Run benchmarks to ensure performance targets are met
4. Update documentation in `API.md`
5. Add example to `README.md` if applicable

### 2. Dashboard Features

If adding dashboard functionality:

1. Add new section to appropriate tab in `app.py`
2. Test interactivity and responsiveness
3. Ensure visualizations are clear and informative
4. Update `README.md` dashboard features section
5. Add screenshots if helpful

### 3. New Modules

If adding a new module:

1. Create module in `sta_pruning/`
2. Add to `__all__` in `sta_pruning/__init__.py`
3. Write comprehensive docstrings
4. Add full API documentation to `API.md`
5. Include usage examples
6. Update `DOCUMENTATION.md` structure

## Testing Guidelines

### Running Tests
```bash
pytest
```

### Writing Tests
```python
def test_feature_extraction():
    """Test that feature extraction produces correct dimensions."""
    from sta_pruning import FeatureExtractor, SyntheticDataGenerator
    
    data_gen = SyntheticDataGenerator(seed=42)
    graph = data_gen.generate_endpoint_graph("test", num_paths=50)
    
    extractor = FeatureExtractor()
    node = list(graph.all_nodes)[0]
    features = extractor.extract(node, graph)
    
    assert features.shape == (21,), "Feature vector should be 21-dimensional"
    assert not np.any(np.isnan(features)), "Features should not contain NaN"
```

### Performance Tests
Ensure changes maintain performance targets:
- MAE < 1.6e-04
- Speedup 1.3-1.7×
- Path overlap >90%

## Documentation Updates

### When to Update Documentation

Update documentation when:
- Adding new features or APIs
- Changing existing behavior
- Fixing bugs that affect usage
- Improving performance
- Adding examples

### Documentation Files

- **README.md**: Overview, installation, basic usage
- **QUICKSTART.md**: Quick start guide for new users
- **API.md**: Complete API reference
- **CHANGELOG.md**: Version history and changes
- **DOCUMENTATION.md**: Documentation index

### Documentation Style

- Use clear, concise language
- Include code examples
- Show expected output
- Explain parameters and return values
- Add troubleshooting tips

## Commit Guidelines

### Commit Message Format
```
<type>: <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```
feat: Add parallel batch processing support

Implement multi-threaded batch processing for large-scale
endpoint analysis. Includes both thread and process-based
parallelization options.

Closes #123
```

```
fix: Correct feature extraction for boundary nodes

Fixed issue where nodes at path boundaries had incorrect
depth calculations, affecting feature vector accuracy.
```

## Pull Request Process

### Before Submitting

1. ✓ Code follows style guidelines
2. ✓ All tests pass
3. ✓ Documentation updated
4. ✓ Performance benchmarks run
5. ✓ Commit messages are clear

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe testing performed

## Performance Impact
- MAE: X.XXe-XX
- Speedup: X.XX×
- Path Overlap: XX.X%

## Documentation
- [ ] README.md updated
- [ ] API.md updated
- [ ] CHANGELOG.md updated
- [ ] Examples added

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Performance verified
```

## Performance Optimization

### Profiling
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
pipeline.process_endpoint(endpoint_graph)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Optimization Tips
- Use NumPy vectorization
- Cache expensive computations
- Minimize graph traversals
- Use efficient data structures
- Profile before optimizing

## Release Process

### Version Numbering
Follow semantic versioning: MAJOR.MINOR.PATCH

- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Run performance benchmarks
5. Update documentation
6. Create git tag
7. Build and test package

## Getting Help

### Resources
- [README.md](README.md) - Project overview
- [API.md](API.md) - API reference
- [DOCUMENTATION.md](DOCUMENTATION.md) - Documentation index

### Questions
- Check existing documentation first
- Review code examples
- Look at similar implementations in the codebase

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for their contributions
- Project documentation
- Release notes

Thank you for contributing to the STA Path Pruning System! 🎉
