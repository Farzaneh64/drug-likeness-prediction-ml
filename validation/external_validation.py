#!/usr/bin/env python3
"""
external_validation.py

Run predictions and metrics on an external dataset of SMILES.
"""
import argparse
import os
import sys
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from config import defaults
from predict import predict


def main():
    parser = argparse.ArgumentParser(
        description="External validation: predict & compute metrics on an external SMILES dataset"
    )
    parser.add_argument(
    "--external", "-e",
    default=os.path.join(defaults['DATA_DIR'], 'external_validation_set.csv'),
    help="Path to external CSV with columns 'SMILES' and 'Label' (plus optional ID/Target)"
    )
    parser.add_argument(
        "--predictions_out", "-p",
        default=os.path.join(
            os.path.dirname(defaults['PREDICTION_OUTPUT']),
            'external_predictions.csv'
        ),
        help="Path to save external predictions CSV"
    )
    parser.add_argument(
        "--metrics_out", "-m",
        default=os.path.join(
            os.path.dirname(defaults['TEST_METRICS_OUTPUT']),
            'external_metrics.csv'
        ),
        help="Path to save external test-set metrics CSV"
    )
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.predictions_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    print(f"🔗 Running external validation on dataset: {args.external}")
    # This will featurize, predict, and compute metrics
    predict(
        args.external,
        defaults['MODEL_FILE'],
        args.predictions_out,
        args.metrics_out
    )
    print(f"✅ External validation complete.\n  Predictions: {args.predictions_out}\n  Metrics:     {args.metrics_out}")


if __name__ == '__main__':
    main()
