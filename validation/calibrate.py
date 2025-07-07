# validation/calibrate.py
"""
Calibration check: Brier score + reliability diagram
Reads the default predictions CSV from outputs/ and computes calibration.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import os, sys
# Ensure project root is on Python path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import defaults

def main():
    parser = argparse.ArgumentParser(
        description="Calibration check: Brier score + reliability diagram"
    )
    parser.add_argument(
        "--predictions", "-p",
        default=defaults['PREDICTION_OUTPUT'],
        help="Path to the CSV of model predictions (must include 'Label' and 'Probability')"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(os.path.dirname(defaults['PREDICTION_OUTPUT']), 'calibration.png'),
        help="Path to save the reliability diagram (PNG)"
    )
    parser.add_argument(
        "--n_bins", "-b",
        type=int,
        default=10,
        help="Number of bins for calibration curve"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1) Load predictions
    df = pd.read_csv(args.predictions)
    if 'Label' not in df or 'Probability' not in df:
        raise ValueError("CSV must contain 'Label' and 'Probability' columns")
    y_true = df['Label']
    y_proba = df['Probability']

    # 2) Brier score
    brier = brier_score_loss(y_true, y_proba)
    print(f"Brier score: {brier:.4f}")

    # 3) Calibration curve
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_proba,
        n_bins=args.n_bins,
        strategy='uniform'
    )

    # 4) Plot
    plt.figure(figsize=(6,6))
    plt.plot(mean_pred, frac_pos, "o-", label="Model")
    plt.plot([0,1], [0,1], "--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)

    # 5) Save figure
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Reliability diagram saved to {args.output}")

if __name__ == "__main__":
    main()

