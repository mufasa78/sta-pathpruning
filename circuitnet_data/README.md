
# CircuitNet Data Directory

## Directory Structure

- `raw/` - Original LEF/DEF files from CircuitNet dataset
- `graph/` - Processed graph pickle files (.pkl)
- `timing/` - Timing feature JSON files
- `features/` - Extracted feature vectors

## Setup Instructions

### 1. Download CircuitNet Dataset

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Download CircuitNet-N14
huggingface-cli download circuitnet/CircuitNet-N14 --repo-type dataset --local-dir ./circuitnet_data/raw
```

### 2. Clone CircuitNet Processing Scripts

```bash
git clone https://github.com/circuitnet/CircuitNet.git /tmp/circuitnet
```

### 3. Process Data (if needed)

If you need to build graphs from scratch:

```bash
# Copy build_graph_demo scripts
cp -r /tmp/circuitnet/build_graph_demo ./circuitnet_tools/

# Run graph building
python circuitnet_tools/build_graph.py --data_path ./circuitnet_data/raw --save_path ./circuitnet_data/graph
```

## Pre-processed Data

The CircuitNet-N14 dataset from Hugging Face already includes:
- Pre-built graph features
- Timing annotations
- Net delay labels
- IR drop data

You can use these directly without running the build scripts.

## Usage in Application

The `CircuitNetLoader` class in `sta_pruning/circuitnet_loader.py` will automatically:
1. Load graph features from `graph/` directory
2. Load timing features from `timing/` directory
3. Convert to `EndpointBasedGraph` format for your pipeline
