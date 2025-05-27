# Drug-Likeness Prediction ML

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
   git clone https://github.com/<your-username>/drug-likeness-prediction-ml.git
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

4. Launch the Streamlit Demo (Optional)
   ```bash
   streamlit run app/streamlit_app.py
A simple web interface to paste a SMILES string and view drug-likeness predictions interactively.


📁 Project Structure

drug-likeness-prediction-ml/
│
├── data/                     # Raw & processed CSV datasets
│   ├── cleaned_drugs_chembl.csv
│   ├── cleaned_decoys_dude.csv
│   └── drug_likeness_dataset.csv
│
├── notebooks/                # Jupyter notebooks for EDA & prototyping
│   └── 01_model_training.ipynb
│
├── src/                      # Python scripts
│   ├── descriptors.py        # Compute RDKit descriptors
│   ├── model.py              # Train & evaluate the ML model
│   └── predict.py            # Predict on new SMILES
│
├── app/                      # Streamlit app for interactive demo
│   └── streamlit_app.py
│
├── models/                   # Saved models (e.g. .joblib files)
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Excluded files/folders
└── README.md                 # Project overview & instructions

📜 License
This project and its bundled datasets are released under CC BY-SA 4.0.
See LICENSE for details.

📚 Citations
DUD-E: Mysinger MM, Carchia M, Irwin JJ, Shoichet BK.
Directory of Useful Decoys, Enhanced (DUD-E): Better Ligands and Decoys for Better Benchmarking.
Journal of Medicinal Chemistry, 2012, 55(14):6582–6594.

ChEMBL: Mendez D, Gaulton A, Bento AP, Chambers J, De Veij M, Félix E, et al.
ChEMBL: towards direct deposition of bioassay data.
Nucleic Acids Research, 2019, 47(D1):D930–D940.

🤝 Acknowledgments
Thanks to the RDKit community for their robust cheminformatics toolkit, and to the maintainers of scikit-learn and Streamlit for powering this pipeline.










   
