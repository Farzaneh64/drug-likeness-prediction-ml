### src/predict.py
```python
import argparse
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np


def featurize(smiles, calculator):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descs = calculator.CalcDescriptors(mol)
    if any(np.isnan(descs)):
        return None
    return list(descs)


def predict(input_csv, model_path, output_csv):
    # Load model
    clf = joblib.load(model_path)

    # Prepare descriptor calculator names
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    # Read input SMILES file (expects column 'SMILES' and optional ID)
    df = pd.read_csv(input_csv)
    results = []

    # Featurize and predict
    for _, row in df.iterrows():
        feats = featurize(row['SMILES'], calculator)
        if feats is None:
            continue
        prob = clf.predict_proba([feats])[0,1]
        label = int(prob >= 0.5)
        out = {
            'SMILES': row['SMILES'],
            'ID': row.get('ID', ''),
            'Prediction': label,
            'Probability': prob
        }
        results.append(out)

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}, total: {len(out_df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict drug-likeness for new SMILES')
    parser.add_argument('--input', '-i', required=True, help='Input CSV with SMILES and optional ID')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model (.joblib)')
    parser.add_argument('--output', '-o', required=True, help='Output CSV for predictions')
    args = parser.parse_args()
    predict(args.input, args.model, args.output)
