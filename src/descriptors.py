### src/descriptors.py
```python
import argparse
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from joblib import Parallel, delayed
from utils import read_csv, save_csv
from config import defaults

def process_record(record, descriptor_names):
    """
    Compute descriptors for a single record dict.
    Returns (record, descriptor_list) or None if featurization fails.
    """
    try:
        mol = Chem.MolFromSmiles(record['SMILES'])
        if mol is None:
            return None
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        descs = calc.CalcDescriptors(mol)
        if any(np.isnan(descs)):
            return None
        return record, list(descs)
    except Exception:
        return None

def compute_descriptors(input_csv, output_csv):
    # 1. Read labeled SMILES dataset
    df = read_csv(input_csv)

    # 2. Prepare descriptor names
    descriptor_names = [desc[0] for desc in Descriptors._descList]

    # 3. Parallel featurization
    records = df.to_dict('records')
    results = Parallel(
        n_jobs=-1,
        backend='multiprocessing',
        verbose=5)
    (delayed(process_record)(rec, descriptor_names) for rec in records)

    # 4. Collect successful features
    rows, feat_data = [], []
    for item in results:
        if item:
            rec, feats = item
            rows.append(rec)
            feat_data.append(feats)

    # 5. Build DataFrame with metadata & descriptors
    meta_df = pd.DataFrame(rows).reset_index(drop=True)
    desc_df = pd.DataFrame(feat_data, columns=descriptor_names)
    final_df = pd.concat([meta_df[['ID','SMILES','Target','Label']], desc_df], axis=1)

    # 6. Clean infinities: replace ±inf with NaN, drop any column containing NaN
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_cols = [c for c in final_df.columns if c not in ('ID','SMILES','Target','Label')]
    good_feats = [c for c in feat_cols if final_df[c].notna().all()]
    if len(good_feats) < len(feat_cols):
        print(f"Dropped {len(feat_cols)-len(good_feats)} unreliable descriptor columns.")
    final_df = final_df[['ID','SMILES','Target','Label'] + good_feats]

    # 7. DROP OVERFLOWING FEATURES
    THRESH = defaults['MAX_DESCRIPTOR_ABS']
    max_abs = final_df[good_feats].abs().max()
    overflow_feats = max_abs[max_abs > THRESH].index.tolist()
    if overflow_feats:
        print(f"Dropping {len(overflow_feats)} descriptors exceeding ±{THRESH}: {overflow_feats}")
        final_df.drop(columns=overflow_feats, inplace=True)

    # 8. Save cleaned descriptor file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    save_csv(final_df, output_csv)
    print(f"Descriptors saved to {output_csv}, shape: {final_df.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute and clean RDKit descriptors from SMILES CSV'
    )
    parser.add_argument(
        '--input', '-i',
        default=defaults['RAW_DATASET'],
        help='Input CSV with SMILES,ID,Target,Label'
    )
    parser.add_argument(
        '--output', '-o',
        default=defaults['DESCRIPTOR_FILE'],
        help='Output CSV with cleaned descriptors'
    )
    args = parser.parse_args()
    compute_descriptors(args.input, args.output)

```

---
