# 🔬💊 Drug-Likeness Prediction ML

**ML-powered pipeline for predicting drug-likeness of small molecules from 2D descriptors.**

---

## 🔍 Overview

This repository provides an end-to-end workflow to go from a ready-to-use, labeled SMILES dataset to a trained Random Forest model that classifies molecules as drug-like or not. It also includes utilities for feature computation, model training, prediction, and a simple Streamlit demo.

---

## 📂 Data

All dataset files live in the `data/` folder. We include:

- **`cleaned_drugs_chembl.csv`**  
  – ~12 000 approved drug molecules from ChEMBL, cleaned to remove entries with missing or invalid SMILES.  
- **`cleaned_decoys_dude.csv`**  
  – 15 000 drug-like decoy molecules sampled from eight DUD-E targets (1 875 per target), cleaned for invalid SMILES.  
- **`drug_likeness_dataset.csv`**  
  – *Ready-to-use*, labeled dataset combining the two above:  
  > - **Label = 1** for drugs  
  > - **Label = 0** for decoys  

> **Note:** We ship with the labeled dataset already prepared. The raw source files (`cleaned_drugs_chembl.csv` and `cleaned_decoys_dude.csv`) are provided for transparency; no merging code is required to reproduce the training data.

---

## ⚙️ Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/Farzaneh64/drug-likeness-prediction-ml.git
   cd drug-likeness-prediction-ml
   
2. Create & activate a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
3. Install dependencies
   ```bash
   pip install -r requirements.txt

## 🚀 Usage
  All commands assume you are in the project root and your virtualenv is activated.
1. Compute Molecular Descriptors
   ```bash
   python src/descriptors.py \
   --input data/drug_likeness_dataset.csv \
   --output data/drug_likeness_descriptors.csv
  This reads your labeled SMILES file, computes ~200 RDKit 2D descriptors per molecule, and writes a CSV with ID, Target, Label plus descriptor columns.

2. Train the Model
   ```bash
   python src/model.py \
   --input data/drug_likeness_descriptors.csv \
   --model_output models/rf_drug_likeness_model.joblib
  Trains a Random Forest classifier (with class-weight balancing), prints performance metrics (precision, recall, ROC-AUC), and saves the .joblib model.

3. Make Predictions
   ```bash
   python src/predict.py \
   --model models/rf_drug_likeness_model.joblib \
   --input data/new_smiles.csv \
   --output predictions.csv
  Loads the trained model, featurizes new SMILES, and outputs predicted labels and probabilities.

4. Launch the Streamlit Demo 
   ```bash
   streamlit run app/streamlit_app.py
  A simple web interface to paste a SMILES string and view drug-likeness predictions interactively.


## 🚀 End-to-End Pipeline
For a one-command run of the full workflow—from raw data to final outputs—use:
    ```bash
    python src/run_all.py 

This script will automatically:

Load and preprocess all raw CSVs in data/.

Compute RDKit descriptors.

Train the Random Forest model with class-weight balancing.

Evaluate on both the test set and your external validation set.

Emit metrics, plots, and prediction files into the outputs/ folder.


## 📁 Project Structure
   
    ```bash
    drug-likeness-prediction-ml/
    │
    ├── data/                                   # Raw & processed CSV datasets
    │   ├── cleaned_drugs_chembl.csv            # ChEMBL-derived drug structures, cleaned
    │   ├── cleaned_decoys_dude.csv             # DUDE-E decoy structures, cleaned
    │   ├── drug_likeness_dataset.csv           # Combined dataset of actives and decoys
    │   └── external_validation_set.csv         # Hold-out set for external validation
    │
    ├── src/                                    # Source Python modules and scripts
    │   ├── __init__.py                         # Package initializer
    |   ├── config.py                           # Configuration (paths, parameters)
    │   ├── descriptors.py                      # Functions to compute RDKit molecular descriptors
    │   ├── train_model.py                      # Model training and evaluation routines
    │   ├── predict.py                          # Script for making predictions on new SMILES
    │   ├── run_all.py                          # End-to-end pipeline: data prep, training, evaluation
    │   └── utils.py                            # Helper functions (I/O, preprocessing)
    |
    ├── models/                                 # Trained model artifacts
    │   └── rf_drug_likeness_model.joblib       # Uploaded on HUGGING fACE
    │
    ├── outputs/                                # Generated outputs (figures, metrics, predictions)
    │   ├── calibration.png                     # Calibration plot for model probabilities
    |   ├── external_metrics.csv                # Metrics from external validation
    │   ├── external_predictions.csv            # Predicted labels on external validation set
    │   ├── predictions.csv                     # Predicted labels on test set
    │   ├── test_metrics.csv                    # Performance metrics on test set
    │   └── train_metrics.json                  # Training performance statistics
    │
    ├── validation/                             # Scripts for calibration and validation analyses
    |   ├── calibrate.py                        # Generate and evaluate calibration curves
    │   ├── external_validation.py              # Run and assess external validation experiments
    │   └── feature_importance.py               # Plot and report feature importance
    │
    ├── app/                                    # Streamlit app for interactive demo
    │   └── streamlit_app.py
    │
    ├── requirements.txt                        # Python dependencies
    ├── .gitignore                              # Excluded files/folders
    └── README.md                               # Project overview & instructions
    
## 📜 License

This project and its bundled datasets are released under CC BY-SA 4.0.
See LICENSE for details.


## 📚 Citations

DUD-E: Mysinger MM, Carchia M, Irwin JJ, Shoichet BK.
Directory of Useful Decoys, Enhanced (DUD-E): Better Ligands and Decoys for Better Benchmarking.
Journal of Medicinal Chemistry, 2012, 55(14):6582–6594.

ChEMBL: Mendez D, Gaulton A, Bento AP, Chambers J, De Veij M, Félix E, et al.
ChEMBL: towards direct deposition of bioassay data.
Nucleic Acids Research, 2019, 47(D1):D930–D940.


## 🤝 Acknowledgments

Thanks to the RDKit community for their robust cheminformatics toolkit, and to the maintainers of scikit-learn and Streamlit for powering this pipeline.










   
