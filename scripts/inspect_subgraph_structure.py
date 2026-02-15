#!/usr/bin/env python3
"""
Quick script to inspect the structure of subgraph pickle files.
"""

import pickle
from pathlib import Path

def inspect_subgraph_pkl(pkl_path, max_samples=3):
    """Load and inspect structure of subgraph pickle."""
    print(f"Inspecting: {pkl_path}")
    print("=" * 80)
    
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"Type: {type(df)}")
    print(f"Length: {len(df)}")
    
    if hasattr(df, 'columns'):
        print(f"Columns: {df.columns.tolist()}")
        print()
        
        # Show a few examples
        for idx in range(min(max_samples, len(df))):
            print(f"\n--- Sample {idx} ---")
            row = df.iloc[idx]
            for col in df.columns:
                val = row[col]
                if col == "subgraph" or col == "walked":
                    print(f"{col} (type: {type(val)}):")
                    if isinstance(val, list) and len(val) > 0:
                        print(f"  First few items: {val[:3]}")
                        print(f"  Total length: {len(val)}")
                    elif isinstance(val, dict):
                        print(f"  Keys: {val.keys()}")
                        for k, v in val.items():
                            if isinstance(v, list):
                                print(f"    {k}: {len(v)} items")
                                if v:
                                    print(f"      Sample: {v[0]}")
                            else:
                                print(f"    {k}: {v}")
                    else:
                        print(f"  Value: {val}")
                else:
                    print(f"{col}: {val}")
    else:
        # Not a DataFrame, show raw structure
        print(f"First item type: {type(df[0])}")
        print(f"First item: {df[0]}")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    base_path = Path("/home/bs_thesis/shift (Copy)/Fact-or-Fiction/data/subgraphs")
    
    for split in ["train", "val", "test"]:
        pkl_file = base_path / f"subgraphs_direct_filled_{split}.pkl"
        if pkl_file.exists():
            inspect_subgraph_pkl(pkl_file, max_samples=2)
        else:
            print(f"File not found: {pkl_file}")
