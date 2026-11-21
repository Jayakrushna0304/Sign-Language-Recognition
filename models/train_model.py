"""
models/train_model.py
=====================
Sign Language Recognition — Model Training Script
Capstone Project: Real-Time Sign Language Recognition Using Machine Learning

Trains and compares Random Forest, SVM, and KNN classifiers on the
prepared landmark dataset, then saves the best model and evaluation artifacts.

Run:
    .\.venv\Scripts\python.exe models/train_model.py
"""

# ── Standard / third-party imports ──────────────────────────────────────────
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")          # Suppress sklearn convergence warnings

# ── Path configuration ───────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV     = os.path.join(BASE_DIR, "dataset", "train_data.csv")
TEST_CSV      = os.path.join(BASE_DIR, "dataset", "test_data.csv")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
SAVED_DIR     = os.path.join(BASE_DIR, "saved_model")

MODEL_PATH    = os.path.join(SAVED_DIR, "sign_language_model.pkl")
ENCODER_PATH  = os.path.join(SAVED_DIR, "label_encoder.pkl")
CM_PATH       = os.path.join(RESULTS_DIR, "confusion_matrix.png")
REPORT_PATH   = os.path.join(RESULTS_DIR, "classification_report.txt")
COMPARE_PATH  = os.path.join(RESULTS_DIR, "model_comparison.png")

# ── Helpers ──────────────────────────────────────────────────────────────────

def info(msg: str) -> None:
    """Print a formatted [INFO] progress message."""
    print(f"[INFO] {msg}")


def ensure_dirs() -> None:
    """Create output directories if they do not already exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SAVED_DIR,   exist_ok=True)


# ── 1. Load data ─────────────────────────────────────────────────────────────

def load_data():
    """
    Load train and test CSVs.
    Returns X_train, X_test (DataFrames) and y_train, y_test (Series).
    """
    info(f"Loading training data from: {TRAIN_CSV}")
    if not os.path.exists(TRAIN_CSV):
        print(f"[ERROR] File not found: {TRAIN_CSV}")
        print("        Run  preprocessing/prepare_dataset.py  first.")
        sys.exit(1)

    info(f"Loading test data from: {TEST_CSV}")
    if not os.path.exists(TEST_CSV):
        print(f"[ERROR] File not found: {TEST_CSV}")
        print("        Run  preprocessing/prepare_dataset.py  first.")
        sys.exit(1)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    # ── Validate 'label' column ──────────────────────────────────────────────
    for name, df in [("train_data.csv", train_df), ("test_data.csv", test_df)]:
        if "label" not in df.columns:
            print(f"[ERROR] 'label' column missing in {name}.")
            sys.exit(1)

    # ── Separate features and target ─────────────────────────────────────────
    # Feature columns: everything except 'label'
    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["label"]

    info(f"Dataset shape  — Train: {X_train.shape}  |  Test: {X_test.shape}")
    info(f"Number of features : {len(feature_cols)}")
    info(f"Train samples : {len(X_train)}  |  Test samples : {len(X_test)}")

    # ── Class verification ───────────────────────────────────────────────────
    train_classes = set(y_train.unique())
    test_classes  = set(y_test.unique())
    all_classes   = sorted(train_classes | test_classes)
    info(f"Total unique classes found : {len(all_classes)}")

    missing_in_test  = train_classes - test_classes
    missing_in_train = test_classes  - train_classes
    if missing_in_test:
        print(f"[WARNING] Classes in train but NOT in test  : {missing_in_test}")
    if missing_in_train:
        print(f"[WARNING] Classes in test  but NOT in train : {missing_in_train}")

    return X_train, X_test, y_train, y_test, feature_cols


# ── 2. Encode labels ─────────────────────────────────────────────────────────

def encode_labels(y_train, y_test):
    """
    Fit a LabelEncoder on the combined label set so both splits share
    the same integer mapping regardless of class imbalance.
    Returns encoded arrays and the fitted encoder.
    """
    info("Encoding class labels with LabelEncoder …")
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test]))    # Fit on all known labels
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)
    info(f"Classes (alphabetical): {list(le.classes_)}")
    return y_train_enc, y_test_enc, le


# ── 3. Define models ─────────────────────────────────────────────────────────

def build_models() -> dict:
    """Return a dictionary of model name → unfitted estimator."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,          # Use all CPU cores
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),
    }


# ── 4. Train and evaluate ────────────────────────────────────────────────────

def train_and_evaluate(models, X_train, X_test, y_train_enc, y_test_enc):
    """
    Train each model, predict on the test set, and compute four metrics.
    Returns a dict of {model_name: {metric: value, 'fitted_model': estimator}}.
    """
    results = {}

    for name, model in models.items():
        info(f"Training  [{name}] …")
        model.fit(X_train, y_train_enc)

        info(f"Evaluating [{name}] …")
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test_enc, y_pred)
        prec = precision_score(y_test_enc, y_pred, average="macro", zero_division=0)
        rec  = recall_score   (y_test_enc, y_pred, average="macro", zero_division=0)
        f1   = f1_score       (y_test_enc, y_pred, average="macro", zero_division=0)

        results[name] = {
            "Accuracy"   : round(acc  * 100, 2),
            "Precision"  : round(prec * 100, 2),
            "Recall"     : round(rec  * 100, 2),
            "Macro F1"   : round(f1   * 100, 2),
            "fitted_model": model,
            "y_pred"     : y_pred,
        }

        print(
            f"         Accuracy={acc*100:.2f}%  Precision={prec*100:.2f}%  "
            f"Recall={rec*100:.2f}%  F1={f1*100:.2f}%"
        )

    return results


# ── 5. Print comparison table ─────────────────────────────────────────────────

def print_comparison_table(results: dict) -> None:
    """Display a neatly formatted comparison table in the console."""
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON TABLE")
    print("=" * 65)
    header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Macro F1':>10}"
    print(header)
    print("-" * 65)
    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['Accuracy']:>9.2f}% "
            f"{metrics['Precision']:>9.2f}% "
            f"{metrics['Recall']:>9.2f}% "
            f"{metrics['Macro F1']:>9.2f}%"
        )
    print("=" * 65 + "\n")


# ── 6. Select best model ──────────────────────────────────────────────────────

def select_best_model(results: dict):
    """Select and return the model name + estimator with the highest Macro F1."""
    best_name = max(results, key=lambda n: results[n]["Macro F1"])
    best_model = results[best_name]["fitted_model"]
    info(
        f"Best model → [{best_name}]  "
        f"Macro F1 = {results[best_name]['Macro F1']:.2f}%"
    )
    return best_name, best_model


# ── 7. Save artifacts ─────────────────────────────────────────────────────────

def save_confusion_matrix(best_name, results, le):
    """Generate and save the confusion matrix heatmap for the best model."""
    info(f"Saving confusion matrix → {CM_PATH}")

    y_pred = results[best_name]["y_pred"]
    # We need y_test_enc — pass it in via results dict (stored below)
    y_true = results[best_name]["y_true"]
    cm     = confusion_matrix(y_true, y_pred)
    labels = le.classes_

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    fig.savefig(CM_PATH, dpi=150)
    plt.close(fig)
    info(f"  ✓ Confusion matrix saved.")


def save_classification_report(best_name, results, le):
    """Write the full sklearn classification report to a text file."""
    info(f"Saving classification report → {REPORT_PATH}")

    y_true = results[best_name]["y_true"]
    y_pred = results[best_name]["y_pred"]
    labels = le.classes_

    # Decode integer predictions back to class names for readability
    report = classification_report(
        le.inverse_transform(y_true),
        le.inverse_transform(y_pred),
        target_names=labels,
        zero_division=0,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(f"Best Model: {best_name}\n")
        fh.write("=" * 60 + "\n")
        fh.write(report)

    info(f"  ✓ Classification report saved.")


def save_model_comparison_graph(results: dict) -> None:
    """
    Bar chart comparing all three models across four metrics.
    Groups bars by metric with one bar per model.
    """
    info(f"Saving model comparison graph → {COMPARE_PATH}")

    model_names = list(results.keys())
    metrics     = ["Accuracy", "Precision", "Recall", "Macro F1"]
    colors      = ["#4C72B0", "#DD8452", "#55A868"]  # Blue, Orange, Green

    x       = np.arange(len(metrics))
    n       = len(model_names)
    width   = 0.22                                    # Bar width
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (name, color) in enumerate(zip(model_names, colors)):
        values = [results[name][m] for m in metrics]
        bars   = ax.bar(
            x + offsets[idx],
            values,
            width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        # Annotate bars with percentage values
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("Metric",           fontsize=12)
    ax.set_ylabel("Score (%)",        fontsize=12)
    ax.set_title(
        "Model Comparison — Random Forest vs SVM vs KNN",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(COMPARE_PATH, dpi=150)
    plt.close(fig)
    info(f"  ✓ Model comparison graph saved.")


def save_model_and_encoder(best_model, le):
    """Persist the best fitted model and LabelEncoder to disk."""
    info(f"Saving best model       → {MODEL_PATH}")
    joblib.dump(best_model, MODEL_PATH)

    info(f"Saving label encoder    → {ENCODER_PATH}")
    joblib.dump(le, ENCODER_PATH)

    info("  ✓ Model and encoder saved.")


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  SIGN LANGUAGE RECOGNITION — MODEL TRAINING")
    print("=" * 65 + "\n")

    ensure_dirs()

    # ── Load data ────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    # ── Encode labels ────────────────────────────────────────────────────────
    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    # ── Build, train, evaluate ───────────────────────────────────────────────
    models  = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train_enc, y_test_enc)

    # Attach y_true to each result entry (needed by artifact helpers)
    for name in results:
        results[name]["y_true"] = y_test_enc

    # ── Summary table ────────────────────────────────────────────────────────
    print_comparison_table(results)

    # ── Select best model ────────────────────────────────────────────────────
    best_name, best_model = select_best_model(results)

    # ── Save artifacts ────────────────────────────────────────────────────────
    save_confusion_matrix       (best_name, results, le)
    save_classification_report  (best_name, results, le)
    save_model_comparison_graph (results)
    save_model_and_encoder      (best_model, le)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best Model   : {best_name}")
    print(f"  Macro F1     : {results[best_name]['Macro F1']:.2f}%")
    print(f"  Accuracy     : {results[best_name]['Accuracy']:.2f}%")
    print(f"\n  Saved files:")
    print(f"    {MODEL_PATH}")
    print(f"    {ENCODER_PATH}")
    print(f"    {CM_PATH}")
    print(f"    {REPORT_PATH}")
    print(f"    {COMPARE_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
