"""
config.py

Configuration constants for the Drug-Likeness Prediction ML pipeline
"""
import os

# Base directory is the parent of this file (i.e., project root)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Data, model, and output directories under project root
DATA_DIR      = os.path.join(BASE_DIR, 'data')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR   = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)  
    
defaults = {
    # File paths
    'DATA_DIR':           DATA_DIR,
    'RAW_DATASET':        os.path.join(DATA_DIR, 'drug_likeness_dataset.csv'),
    'DESCRIPTOR_FILE':    os.path.join(DATA_DIR, 'drug_likeness_descriptors.csv'),
    'MODEL_FILE':         os.path.join(MODELS_DIR, 'rf_drug_likeness_model.joblib'),
    'PREDICTION_OUTPUT':  os.path.join(OUTPUTS_DIR, 'predictions.csv'),
    'TRAIN_METRICS_OUTPUT': os.path.join(OUTPUTS_DIR, 'train_metrics.json'),  # training-set metrics
    'TEST_METRICS_OUTPUT':  os.path.join(OUTPUTS_DIR, 'test_metrics.csv'),    # test-set metrics
    
    # Descriptor-cleanup parameters
    'MAX_DESCRIPTOR_ABS': 1e6,    # drop any descriptor with |value| > this

    # Train/test split settings
    'TEST_SIZE':          0.2,     # fraction of data reserved for test set
    'RANDOM_STATE':       42,      # seed for reproducibility

    # Random Forest hyperparameters
    'train_params': {
        'n_estimators':   100,
        'random_state':   42,
        'class_weight':   'balanced',
    },
}