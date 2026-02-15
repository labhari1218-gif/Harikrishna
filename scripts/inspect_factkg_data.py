#!/usr/bin/env python3
"""
Inspect FactKG claim data to understand claim structure.
"""

import pickle
from pathlib import Path
import pandas as pd

def inspect_factkg_data(pkl_path, max_samples=3):
    """Load and inspect FactKG claim data."""
    print(f"Inspecting: {pkl_path}")
    print("=" * 80)
    
    data_dict = pd.read_pickle(pkl_path)
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Sentence"}, inplace=True)
    
    print(f"Type: {type(df)}")
    print(f"Length: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print()
    
    for idx in range(min(max_samples, len(df))):
        print(f"\n--- Sample {idx} ---")
        row = df.iloc[idx]
        for col in df.columns:
            val = row[col]
            if col == "Label":
                print(f"{col}: {val} (type: {type(val)})")
            else:
                print(f"{col}: {val}")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    base_path = Path("/home/bs_thesis/shift (Copy)/Fact-or-Fiction/data/factkg")
    
    for split, filename in [("train", "factkg_train.pickle"), 
                            ("val", "factkg_dev.pickle"), 
                            ("test", "factkg_test.pickle")]:
        pkl_file = base_path / filename
        if pkl_file.exists():
            inspect_factkg_data(pkl_file, max_samples=2)
        else:
            print(f"File not found: {pkl_file}")

