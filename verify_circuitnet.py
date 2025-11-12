
#!/usr/bin/env python3
"""
Verify CircuitNet dataset installation and structure.
This script helps diagnose data loading issues.
"""

import os
from pathlib import Path
from sta_pruning import CircuitNetLoader

def verify_dataset(data_path: str = "./circuitnet_data"):
    """Verify CircuitNet dataset structure and contents."""
    
    print("=" * 60)
    print("CircuitNet Dataset Verification")
    print("=" * 60)
    
    data_dir = Path(data_path)
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"\n❌ Dataset directory not found: {data_path}")
        print("\nTo download CircuitNet-N14:")
        print("  pip install huggingface-hub")
        print("  huggingface-cli download circuitnet/CircuitNet-N14 --repo-type dataset --local-dir ./circuitnet_data/raw")
        return
    
    print(f"\n✓ Dataset directory found: {data_path}")
    
    # Check subdirectories
    subdirs = ['raw', 'graph', 'timing', 'features']
    for subdir in subdirs:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            num_files = len(list(subdir_path.glob("*")))
            print(f"✓ {subdir}/ exists ({num_files} items)")
        else:
            print(f"  {subdir}/ not found (optional)")
    
    # Try to load designs
    print("\n" + "=" * 60)
    print("Loading Designs")
    print("=" * 60)
    
    loader = CircuitNetLoader(data_path)
    designs = loader.list_available_designs()
    
    if not designs:
        print("\n❌ No designs found!")
        print("\nPossible issues:")
        print("  1. Dataset not downloaded")
        print("  2. Incorrect directory structure")
        print("  3. Files in unexpected format")
        return
    
    print(f"\n✓ Found {len(designs)} designs:")
    for i, design in enumerate(designs[:10], 1):
        print(f"  {i}. {design}")
    
    if len(designs) > 10:
        print(f"  ... and {len(designs) - 10} more")
    
    # Try to load one design
    print("\n" + "=" * 60)
    print("Testing Design Load")
    print("=" * 60)
    
    test_design = designs[0]
    print(f"\nAttempting to load: {test_design}")
    
    endpoint_graphs = loader.load_design(test_design)
    
    if endpoint_graphs:
        print(f"✓ Successfully loaded {len(endpoint_graphs)} endpoints")
        
        total_paths = sum(ep.num_paths for ep in endpoint_graphs)
        print(f"✓ Total paths: {total_paths}")
        
        if endpoint_graphs:
            sample_ep = endpoint_graphs[0]
            print(f"\nSample endpoint: {sample_ep.endpoint}")
            print(f"  Paths: {sample_ep.num_paths}")
            print(f"  Worst slack: {sample_ep.worst_slack:.2e}")
    else:
        print("❌ Failed to load design")
        print("\nThis may mean:")
        print("  - Graph features are missing")
        print("  - Timing features are missing")
        print("  - Data format is incompatible")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    verify_dataset()
