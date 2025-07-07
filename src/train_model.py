### src/train_model.py
```python

import argparse
import os
import joblib
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from utils import read_csv
from config import defaults

train_params = defaults['train_params']

def train_model(input_csv, model_output, metrics_output=None):
    # 1. Load descriptor dataset
    df = read_csv(input_csv)

    # 2. Show counts per target (and per label, if you like)
    print(">>> Samples per target:")
    print(df['Target'].value_counts(), "\n")
    print(">>> Samples per (Target, Label):")
    print(df.groupby(['Target','Label']).size(), "\n")

    # 3. Prepare splitting parameters
    test_frac   = defaults.get('TEST_SIZE', 0.20)
    rnd_state   = defaults.get('RANDOM_STATE', train_params['random_state'])

    # 4. Split each target into train/test
    train_parts = []
    test_parts  = []
    for tgt, group in df.groupby('Target'):
        # if group is too small, skip or take single split
        if len(group) < 5:
            # for very small groups, just put all in train
            train_parts.append(group)
            continue
        # sample test set
        test_grp  = group.sample(frac=test_frac, random_state=rnd_state)
        train_grp = group.drop(test_grp.index)
        train_parts.append(train_grp)
        test_parts.append(test_grp)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df  = pd.concat(test_parts).reset_index(drop=True)

    # 5. Separate features and labels
    X_train = train_df.drop(columns=['ID','SMILES','Target','Label'])
    y_train = train_df['Label']
    X_test  = test_df.drop(columns=['ID','SMILES','Target','Label'])
    y_test  = test_df['Label']

    print(f"▶️ After split: {len(train_df)} train samples, {len(test_df)} test samples")

    # 6. Train the Random Forest
    clf = RandomForestClassifier(
        n_estimators  = train_params['n_estimators'],
        random_state  = train_params['random_state'],
        class_weight  = train_params['class_weight']
    )
    clf.fit(X_train, y_train)

    # 7. Evaluate
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    print("🔍 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"📈 ROC AUC: {roc_auc:.4f}")

    # 8. Save model & metrics
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(clf, model_output)
    print(f"✅ Model saved to {model_output}")

    if metrics_output:
        os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
        metrics = {
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        with open(metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved to {metrics_output}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train drug-likeness classifier')
    parser.add_argument(
        '--input', '-i',
        default=defaults['DESCRIPTOR_FILE'],
        help='Input descriptor CSV'
    )
    parser.add_argument(
        '--model_output', '-m',
        default=defaults['MODEL_FILE'],
        help='Path to save trained model (.joblib)'
    )
    parser.add_argument(
        '--metrics_output', '-e',
        default=defaults['TRAIN_METRICS_OUTPUT'],
        help='Path to save metrics JSON'
    )
    args = parser.parse_args()
    train_model(args.input, args.model_output, args.metrics_output)
```

---
