### src/train_model.py
```python
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def train_model(input_csv, model_output):
    # Load descriptor dataset
    df = pd.read_csv(input_csv)

    # Split features and labels
    X = df.drop(columns=['ID','Target','Label'])
    y = df['Label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Save model
    joblib.dump(clf, model_output)
    print(f"Model saved to {model_output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train drug-likeness classifier')
    parser.add_argument('--input', '-i', required=True, help='Input descriptor CSV')
    parser.add_argument('--model_output', '-m', required=True, help='Path to save trained model (.joblib)')
    args = parser.parse_args()
    train_model(args.input, args.model_output)
```

---
