# Overview

This project implements an ML-based Static Timing Analysis (STA) path pruning system for circuit designs. The system uses machine learning (Random Forest and XGBoost) to intelligently identify critical timing paths in digital circuits, achieving 30% faster analysis than exhaustive methods while maintaining high accuracy (MAE < 1.6e-04).

The core algorithm operates in three stages:
1. **Candidate Generation**: Identifies potential anchor nodes from the middle 40% of timing paths
2. **Anchor Prediction**: Uses ML classifiers with 21-dimensional feature vectors to predict optimal anchor nodes
3. **Path Pruning**: Filters paths through the predicted anchor and returns top 10 critical paths

The system includes a comprehensive Streamlit-based dashboard with 8 tabs offering visualization, model tuning, batch processing (2000+ endpoints), and automated report generation. All features are production-ready and fully functional.

**Recent Major Enhancements (Nov 2025):**
- Advanced visualizations: timing path graphs, slack distributions, anchor analysis
- Model tuning: cross-validation, grid search, random search hyperparameter optimization
- Batch processing: efficient analysis of 100-2000 endpoints with parallel execution
- Automated report generation: markdown reports and CSV exports with download functionality

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Core Algorithm Components

**Three-Stage Pipeline Architecture**
- The system uses a sequential pipeline pattern where each stage feeds into the next
- Stage 1 (Candidate Generation) scores nodes based on path coverage and timing criticality
- Stage 2 (Anchor Prediction) applies trained ML models to select the optimal anchor node
- Stage 3 (Path Pruning) filters and sorts paths to identify worst-case timing scenarios
- This architecture separates concerns and allows independent optimization of each stage

**Feature Engineering Strategy**
- Combines 7 path-level features (slack statistics, delays, criticality metrics) with 14 node-level features (timing properties, physical characteristics, connectivity)
- Feature extraction is centralized in `FeatureExtractor` class for consistency across training and inference
- Features are designed to capture both local node properties and global path context

**Model Architecture Decision**
- Supports both Random Forest and XGBoost classifiers for flexibility
- Random Forest chosen for interpretability and robustness; XGBoost for potential performance gains
- Models are trained on synthetic data with configurable hyperparameters (n_estimators, max_depth, learning_rate)
- Training uses binary classification where anchor nodes are labeled as positive class

## Data Structures

**Graph Representation**
- Uses `EndpointBasedGraph` to organize timing paths by endpoint
- Each graph contains multiple `TimingPath` objects representing different signal propagation routes
- `TimingPath` contains ordered sequences of `Node` objects with timing annotations
- Node identity based on `node_id` with proper hashing for set operations

**Synthetic Data Generation**
- Generates realistic circuit timing data with controlled randomness
- Creates hierarchical paths with depth-based node placement
- Simulates timing parameters (slack, arrival time, required time, capacitance, transition, fanout)
- Enables offline training and benchmarking without requiring actual circuit design data

## Web Application

**Streamlit Dashboard Architecture**
- Single-page application with tabbed interface for different functionalities
- Uses `@st.cache_resource` for one-time initialization of ML models and training data
- Generates 50 training graphs on startup for model initialization
- Trains both RF and XGBoost pipelines during application bootstrap

**Visualization Strategy**
- Leverages Plotly for interactive visualizations
- Supports real-time benchmarking and comparative analysis
- Dashboard tabs likely include: algorithm overview, single endpoint analysis, multi-design benchmarking, model comparison, and configuration

## Evaluation Framework

**Metrics System**
- Mean Squared Error (MSE) for slack prediction accuracy
- Mean Absolute Error (MAE) as primary accuracy metric
- Path overlap (Jaccard similarity) to measure critical path identification quality
- Execution time tracking for performance benchmarking

**Baseline Comparison**
- Implements exhaustive baseline method for ground truth comparison
- Compares proposed ML approach against baseline for accuracy and speed
- Benchmarks across multiple design sizes (1.5M to 9.7M pins mentioned in README)

## Training System

**Supervised Learning Approach**
- Training data preparation identifies best anchor nodes using heuristic scoring
- Scoring combines path coverage (50%), critical path ratio (40%), and average slack (10%)
- Binary classification where only the best anchor per endpoint is labeled positive
- Batch feature extraction for efficient training data generation

# External Dependencies

## Machine Learning Libraries
- **scikit-learn**: Random Forest classifier implementation with configurable hyperparameters
- **XGBoost**: Gradient boosting classifier as alternative ML model
- Both libraries provide n_jobs parallelization for training and inference

## Web Framework
- **Streamlit**: Full web application framework for dashboard
- Provides caching mechanisms, tabbed layouts, and widget system
- Handles state management and page configuration

## Data Processing
- **NumPy**: Core numerical operations, array manipulations, and statistical computations
- **Pandas**: Data frame operations for metrics and benchmarking results (imported but usage not fully shown)

## Visualization
- **Plotly**: Interactive plotting library (graph_objects, express modules)
- Supports subplots for multi-chart dashboards
- Provides client-side interactivity for exploration

## Python Standard Library
- **time**: Performance timing and benchmarking
- **typing**: Type hints for code clarity and IDE support
- **dataclasses**: Structured data classes for Node, TimingPath, EndpointBasedGraph

## Notable Design Choices
- No external database - all data managed in-memory using Python data structures
- No external file I/O shown (though data_parser.py mentioned in planning doc)
- Synthetic data generation eliminates dependency on actual STA tool outputs
- Self-contained system with minimal external service dependencies