### src/descriptors.py
```python
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np

def compute_descriptors(input_csv, output_csv):
    # Read labeled SMILES dataset
    df = pd.read_csv(input_csv)

    # Prepare descriptor calculator
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    # Function to featurize a single SMILES
    def featurize(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            descs = calculator.CalcDescriptors(mol)
            if any(np.isnan(descs)):
                return None
            return list(descs)
        except:
            return None

    # Lists to collect results
    rows = []
    feat_data = []

    # Iterate and compute
    for _, row in df.iterrows():
        features = featurize(row['SMILES'])
        if features:
            rows.append(row)
            feat_data.append(features)

    # Build final DataFrame
    meta_df = pd.DataFrame(rows).reset_index(drop=True)
    desc_df = pd.DataFrame(feat_data, columns=descriptor_names)
    final_df = pd.concat([meta_df[['ID','Target','Label']], desc_df], axis=1)

    # Save
    final_df.to_csv(output_csv, index=False)
    print(f"Descriptors saved to {output_csv}, shape: {final_df.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute RDKit descriptors from SMILES CSV')
    parser.add_argument('--input', '-i', required=True, help='Input CSV with SMILES,ID,Target,Label')
    parser.add_argument('--output', '-o', required=True, help='Output CSV with descriptors')
    args = parser.parse_args()
    compute_descriptors(args.input, args.output)
```

---
