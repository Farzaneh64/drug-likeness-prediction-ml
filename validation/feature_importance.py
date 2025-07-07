#!/usr/bin/env python3
"""
Feature Importance: Plot top N descriptors from trained RF model
"""
import argparse
import os
import sys

# Ensure project root is on Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import defaults

import joblib
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Feature Importance: Plot top N descriptors from trained RF model"
    )
    parser.add_argument(
        "--model", "-m",
        default=defaults['MODEL_FILE'],
        help="Path to the trained RandomForest .joblib file"
    )
    parser.add_argument(
        "--descriptors", "-d",
        default=defaults['DESCRIPTOR_FILE'],
        help="Path to the descriptor CSV used for training"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(
            os.path.dirname(defaults['PREDICTION_OUTPUT']),
            'feature_importance.png'
        ),
        help="Path to save the feature importance plot (PNG)"
    )
    parser.add_argument(
        "--top_n", "-n",
        type=int,
        default=20,
        help="Number of top features to plot"
    )
    args = parser.parse_args()

    # Load trained model
    clf = joblib.load(args.model)

    # Determine feature names
    try:
        feature_names = list(clf.feature_names_in_)
    except AttributeError:
        df_desc = pd.read_csv(args.descriptors)
        feature_names = [
            c for c in df_desc.columns
            if c not in ('ID', 'SMILES', 'Target', 'Label')
        ]

    # Get importances and prepare DataFrame
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(args.top_n)

    # Plot horizontal bar chart
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.barh(fi_df['feature'][::-1], fi_df['importance'][::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {args.top_n} Descriptor Importances")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Feature importance plot saved to {args.output}")


if __name__ == "__main__":
    main()
