### src/predict.py
```python
import argparse
import os
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import read_csv, save_csv
from config import defaults

def featurize(smiles, calculator):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descs = calculator.CalcDescriptors(mol)
    if any(np.isnan(descs)):
        return None
    return list(descs)

def predict(input_csv, model_path, output_csv, metrics_output=None):
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if metrics_output:
        os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
        
    # Load trained model
    clf = joblib.load(model_path)

    # Get the exact feature names used in training
    try:
        feat_names = list(clf.feature_names_in_)
    except AttributeError:
        # Fallback: read descriptor file columns
        df_desc = read_csv(defaults['DESCRIPTOR_FILE'])
        feat_names = [c for c in df_desc.columns if c not in ('ID', 'Target', 'Label', 'SMILES')]

    # Prepare full descriptor calculator
    full_desc_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(full_desc_names)

    # Read input CSV (should include SMILES, ID, Target, Label)
    df = read_csv(input_csv)
    results = []

    # Featurize & predict each molecule
    for _, row in df.iterrows():
        feats = featurize(row['SMILES'], calculator)
        if feats is None:
            continue

        # Map to dict then filter/order by feat_names
        feat_dict = dict(zip(full_desc_names, feats))
        try:
            x = [feat_dict[name] for name in feat_names]
        except KeyError:
            # skip if some trained feature is missing
            continue

        prob = clf.predict_proba([x])[0, 1]
        pred_label = int(prob >= 0.5)
        results.append({
            'SMILES': row['SMILES'],
            'ID': row.get('ID', ''),
            'Target': row.get('Target', ''),
            'Label': row.get('Label', np.nan),
            'Prediction': pred_label,
            'Probability': prob
        })

    # Create DataFrame of predictions
    out_df = pd.DataFrame(results)

    # Save predictions
    save_csv(out_df, output_csv)
    print(f"Predictions saved to {output_csv}, total: {len(out_df)}")

    # Compute and save metrics if requested
    if metrics_output and not out_df.empty:
        y_true = out_df['Label']
        y_pred = out_df['Prediction']
        y_proba = out_df['Probability']

        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics_dict = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
        metrics_df = pd.DataFrame([metrics_dict])
        save_csv(metrics_df, metrics_output)
        print(f"Metrics saved to {metrics_output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict drug-likeness for a test set of SMILES and compute metrics'
    )
    parser.add_argument(
        '--input', '-i',
        default=defaults['RAW_DATASET'],
        help='Input CSV with SMILES, ID, Target, Label'
    )
    parser.add_argument(
        '--model', '-m',
        default=defaults['MODEL_FILE'],
        help='Path to trained model (.joblib)'
    )
    parser.add_argument(
        '--output', '-o',
        default=defaults['PREDICTION_OUTPUT'],
        help='Output CSV path for predictions'
    )
    parser.add_argument(
        '--metrics', '-e',
        default=defaults['TEST_METRICS_OUTPUT'],
        help='Output CSV path for test-set metrics'
    )
    args = parser.parse_args()
    predict(args.input, args.model, args.output, args.metrics)

```
