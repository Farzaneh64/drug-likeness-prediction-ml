"""
utils.py

Utility functions for file handling and common operations
Helper functions for:
- Setting up console+file logging (pipeline.log)
- Randomizing SMILES strings (preserving stereochemistry)
- Creating directories
"""

import os
import pandas as pd

def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file into a DataFrame, raising if not found"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str):
    """Save a DataFrame to CSV, creating directories as needed"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
  
