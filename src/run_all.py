"""
run_all.py

End-to-end pipeline for Drug-Likeness Prediction ML:
1. Compute descriptors
2. Train Random Forest model
3. Generate example predictions
"""

import os
import argparse

# Ensure required directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

from config import defaults
from descriptors import compute_descriptors
from model import train_model
from predict import predict

def main():
    # Paths from config
    ds_input         = defaults['RAW_DATASET']
    ds_output        = defaults['DESCRIPTOR_FILE']
    model_file       = defaults['MODEL_FILE']
    train_metrics    = defaults['TRAIN_METRICS_OUTPUT']
    pred_input       = ds_input
    pred_output      = defaults['PREDICTION_OUTPUT']
    test_metrics     = defaults['TEST_METRICS_OUTPUT']

    # 1. Compute molecular descriptors
    print("🔬 Computing descriptors...")
    compute_descriptors(ds_input, ds_output)

    # 2. Train the model
    print("🤖 Training model...")
    train_model(ds_output, model_file, train_metrics)

    # 3. Generate predictions as a sample check
    print("📊 Generating predictions...")
    predict(pred_input, model_file, pred_output, test_metrics)

    print("✅ Pipeline completed successfully!")


if __name__ == '__main__':
    main()
    
