"""
Dataset Preparation & Sequence-Aware Train/Test Splitting Script.
Part of the 'Real-Time Sign Language Recognition' Capstone Project.

Functionality:
- Loads cleaned hand landmark coordinates from dataset/cleaned_landmarks_data.csv.
- Extracts 126 coordinate features (h0_x0 to h1_z20) and target 'label'.
- Excludes metadata columns (label, sequence_id, frame_id, timestamp, num_hands) from features X.
- Performs an 80/20 sequence-aware stratified split based on unique sequence_id values.
- Guarantees zero sequence leakage between train and test sets.
- Verifies that all 20 sign vocabulary classes exist in both train and test partitions.
- Saves prepared partitions to dataset/train_data.csv and dataset/test_data.csv.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# File paths
DATASET_DIR = PROJECT_ROOT / "dataset"
INPUT_CSV = DATASET_DIR / "cleaned_landmarks_data.csv"
TRAIN_CSV = DATASET_DIR / "train_data.csv"
TEST_CSV = DATASET_DIR / "test_data.csv"

# Configuration
TEST_SIZE = 0.20
RANDOM_STATE = 42
EXPECTED_CLASS_COUNT = 20

# Metadata columns to exclude from feature matrix X
METADATA_COLUMNS = ["label", "sequence_id", "frame_id", "timestamp", "num_hands"]

# Expected 126 hand landmark coordinate columns (2 hands x 21 landmarks x 3 coords)
EXPECTED_FEATURE_COLUMNS = []
for h in range(2):
    for pt in range(21):
        EXPECTED_FEATURE_COLUMNS.extend([f"h{h}_x{pt}", f"h{h}_y{pt}", f"h{h}_z{pt}"])


def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Loads dataset from CSV and validates the presence of required target,
    sequence tracking, and 126 coordinate feature columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Cleaned dataset not found at: {csv_path}\n"
            "Please ensure 'dataset/cleaned_landmarks_data.csv' exists before running this script."
        )

    print(f"[INFO] Loading cleaned dataset from: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)

    # 1. Validate Target Column
    if "label" not in df.columns:
        raise ValueError("[ERROR] Missing required target column: 'label'")

    # 2. Validate Sequence ID Column
    if "sequence_id" not in df.columns:
        raise ValueError(
            "[ERROR] Missing required grouping column: 'sequence_id'. "
            "A sequence-aware split requires unique sequence identifiers."
        )

    # 3. Validate 126 Hand Landmark Feature Columns
    missing_features = [col for col in EXPECTED_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"[ERROR] Missing {len(missing_features)} required landmark feature columns!\n"
            f"Sample missing columns: {missing_features[:5]}"
        )

    # 4. Check for Empty Dataset
    if len(df) == 0:
        raise ValueError("[ERROR] Dataset is empty (0 rows).")

    # 5. Check Class Count
    unique_classes = df["label"].dropna().unique()
    if len(unique_classes) < EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"[ERROR] Expected {EXPECTED_CLASS_COUNT} classes, but found only {len(unique_classes)}: {list(unique_classes)}"
        )

    return df


def sequence_aware_stratified_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> tuple[pd.Index, pd.Index]:
    """
    Performs a stratified split at the sequence level so that all frames
    belonging to a given sequence_id stay strictly together in either Train or Test.
    """
    # Verify that each sequence_id belongs to exactly one label
    seq_label_counts = df.groupby("sequence_id")["label"].nunique()
    if (seq_label_counts > 1).any():
        conflicted_seqs = seq_label_counts[seq_label_counts > 1].index.tolist()
        raise ValueError(
            f"[ERROR] {len(conflicted_seqs)} sequence IDs map to multiple classes! "
            f"Conflicted sequences: {conflicted_seqs[:5]}"
        )

    # Extract unique (sequence_id, label) pairs
    seq_df = df[["sequence_id", "label"]].drop_duplicates(subset=["sequence_id"])

    # Convert sequence IDs and labels into compatible 1D NumPy arrays
    unique_seqs = np.array(seq_df["sequence_id"].to_list(), dtype=str)
    unique_labels = np.array(seq_df["label"].to_list(), dtype=str)

    # Perform stratified split across unique sequences
    try:
        from sklearn.model_selection import train_test_split
        train_seqs, test_seqs = train_test_split(
            unique_seqs,
            test_size=test_size,
            random_state=random_state,
            stratify=unique_labels,
        )
    except ImportError:
        # Fallback stratified split if scikit-learn is not available
        rng = np.random.RandomState(random_state)
        train_seq_list, test_seq_list = [], []
        for cls in np.unique(unique_labels):
            cls_seqs = unique_seqs[unique_labels == cls]
            rng.shuffle(cls_seqs)
            n_test = max(1, int(round(len(cls_seqs) * test_size)))
            test_seq_list.extend(cls_seqs[:n_test])
            train_seq_list.extend(cls_seqs[n_test:])
        train_seqs = np.array(train_seq_list, dtype=str)
        test_seqs = np.array(test_seq_list, dtype=str)

    train_seq_set = set(train_seqs)
    test_seq_set = set(test_seqs)

    # Map sequence assignments back to frame indices in the main dataframe
    train_mask = df["sequence_id"].isin(train_seq_set)
    test_mask = df["sequence_id"].isin(test_seq_set)

    train_indices = df.index[train_mask]
    test_indices = df.index[test_mask]

    return train_indices, test_indices, train_seq_set, test_seq_set


def prepare_dataset(
    input_path: Path = INPUT_CSV,
    train_out_path: Path = TRAIN_CSV,
    test_out_path: Path = TEST_CSV,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main orchestration routine for data validation, sequence-aware splitting,
    verification, and saving.
    """
    print("=" * 68)
    print("      DATASET PREPARATION & SEQUENCE-AWARE STRATIFIED SPLIT      ")
    print("=" * 68)

    # 1. Load and validate cleaned CSV
    df = load_and_validate_dataset(input_path)

    # 2. Identify feature columns (strictly landmark coordinates h0_x0..h1_z20)
    feature_cols = [c for c in df.columns if c in EXPECTED_FEATURE_COLUMNS]

    # 3. Exclude metadata columns from model features X
    metadata_cols_found = [c for c in df.columns if c in METADATA_COLUMNS]
    excluded_from_x = [c for c in metadata_cols_found if c != "label"]

    total_samples = len(df)
    total_sequences = df["sequence_id"].nunique()
    unique_classes = sorted(df["label"].unique())

    print(f"[INFO] Loaded Samples (Rows)   : {total_samples}")
    print(f"[INFO] Total Unique Sequences  : {total_sequences}")
    print(f"[INFO] Total Vocabulary Classes: {len(unique_classes)}")
    print(f"[INFO] Extracted Feature Count : {len(feature_cols)} coordinate columns")
    print(f"[INFO] Excluded Metadata from X: {excluded_from_x}")
    print("-" * 68)

    # 4. Perform sequence-aware stratified split (80/20)
    print(f"[INFO] Splitting dataset into {int((1 - test_size) * 100)}% Train / {int(test_size * 100)}% Test...")
    print(f"[INFO] Reproducibility Random State: {random_state}")

    train_indices, test_indices, train_seqs, test_seqs = sequence_aware_stratified_split(
        df, test_size=test_size, random_state=random_state
    )

    # Partition data
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    # 5. Requirement 10: Verify ZERO sequence_id overlap between train and test
    overlap = train_seqs.intersection(test_seqs)
    print("\n" + "=" * 68)
    print("                 SEQUENCE OVERLAP VERIFICATION                  ")
    print("=" * 68)
    print(f"Train Unique Sequences : {len(train_seqs)}")
    print(f"Test Unique Sequences  : {len(test_seqs)}")
    print(f"Overlapping Sequences  : {len(overlap)}")

    if len(overlap) > 0:
        raise ValueError(
            f"[FATAL] Sequence leakage detected! {len(overlap)} sequence IDs exist in both sets: {list(overlap)[:5]}"
        )
    print(">> [PASSED] Zero sequence overlap confirmed. No data leakage between sets.")

    # 6. Requirement 8: Check that all 20 classes are present in both sets
    train_classes = set(train_df["label"].unique())
    test_classes = set(test_df["label"].unique())
    expected_set = set(unique_classes)

    missing_train = expected_set - train_classes
    missing_test = expected_set - test_classes

    print("\n" + "=" * 68)
    print("              CLASS REPRESENTATION VERIFICATION                 ")
    print("=" * 68)
    print(f"Classes present in Train Set: {len(train_classes)} / {len(expected_set)}")
    print(f"Classes present in Test Set : {len(test_classes)} / {len(expected_set)}")

    if missing_train:
        raise ValueError(f"[FATAL] Missing classes in Train set: {missing_train}")
    if missing_test:
        raise ValueError(f"[FATAL] Missing classes in Test set: {missing_test}")

    print(">> [PASSED] All 20 classes are verified present in both Train and Test sets.")

    # 7. Requirement 9: Print dataset shape, feature count, sample counts, and class distribution
    print("\n" + "=" * 68)
    print("                    DATASET SUMMARY METRICS                     ")
    print("=" * 68)
    print(f"Original Input Dataset Shape : {df.shape}")
    print(f"Model Feature Columns (X)    : {len(feature_cols)} (h0_x0 through h1_z20)")
    print(f"Target Variable (y)          : 'label'")
    print(f"Train Sample Count (Frames)  : {len(train_df)} ({len(train_df) / total_samples * 100:.1f}%)")
    print(f"Test Sample Count (Frames)   : {len(test_df)} ({len(test_df) / total_samples * 100:.1f}%)")
    print("-" * 68)

    print("\n" + "=" * 68)
    print("                 PER-CLASS DISTRIBUTION BREAKDOWN               ")
    print("=" * 68)
    print(f"{'#':<3} | {'Class Label':<14} | {'Train Frames':<13} | {'Test Frames':<12} | {'Total':<8} | {'% Test':<7}")
    print("-" * 68)

    train_counts = train_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()

    for idx, cls in enumerate(unique_classes, 1):
        tr_cnt = int(train_counts.get(cls, 0))
        te_cnt = int(test_counts.get(cls, 0))
        tot_cnt = tr_cnt + te_cnt
        pct_test = (te_cnt / tot_cnt * 100) if tot_cnt > 0 else 0
        print(f"{idx:<3} | {cls:<14} | {tr_cnt:<13} | {te_cnt:<12} | {tot_cnt:<8} | {pct_test:>5.1f}%")

    print("-" * 68)
    print(
        f"{'':<3} | {'TOTAL':<14} | {len(train_df):<13} | {len(test_df):<12} | "
        f"{total_samples:<8} | {(len(test_df) / total_samples * 100):>5.1f}%"
    )
    print("=" * 68)

    # 8. Requirement 11: Save prepared datasets with 'label' + landmark feature columns
    # Excludes raw recording metadata (sequence_id, frame_id, timestamp, num_hands) from the model training files
    save_cols = ["label"] + feature_cols
    train_export = train_df[save_cols]
    test_export = test_df[save_cols]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_export.to_csv(train_out_path, index=False)
    test_export.to_csv(test_out_path, index=False)

    print("\n" + "=" * 68)
    print("                       FILES SAVED                              ")
    print("=" * 68)
    print(f"[SUCCESS] Train dataset saved to : {train_out_path.resolve()}")
    print(f"          Shape: {train_export.shape} (1 label + 126 landmark features)")
    print(f"[SUCCESS] Test dataset saved to  : {test_out_path.resolve()}")
    print(f"          Shape: {test_export.shape} (1 label + 126 landmark features)")
    print("=" * 68)

    return train_export, test_export


if __name__ == "__main__":
    prepare_dataset()
