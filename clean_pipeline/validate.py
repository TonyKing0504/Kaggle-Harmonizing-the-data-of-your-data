"""Validation and scoring utilities."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from config import (SUBMISSION_COLUMNS, SAMPLE_SUB, DEFAULT_FILL,
                    META_COLUMNS, ARTIFACTS_DIR)


def validate_submission(submission_path: str) -> Tuple[bool, list]:
    """Validate submission.csv against SampleSubmission.csv structure."""
    errors = []
    ss = pd.read_csv(SAMPLE_SUB)
    sub = pd.read_csv(submission_path)

    # Column check
    ss_cols = list(ss.columns)
    sub_cols = list(sub.columns)
    if ss_cols != sub_cols:
        missing = set(ss_cols) - set(sub_cols)
        extra = set(sub_cols) - set(ss_cols)
        if missing:
            errors.append(f"Missing columns: {missing}")
        if extra:
            errors.append(f"Extra columns: {extra}")
        if ss_cols != sub_cols and not missing and not extra:
            errors.append("Column ORDER mismatch")

    # Row count
    if len(sub) != len(ss):
        errors.append(f"Row count mismatch: expected {len(ss)}, got {len(sub)}")

    # ID, PXD, Raw Data File, Usage must match exactly
    for col in ['ID', 'PXD', 'Raw Data File']:
        if col in sub.columns and col in ss.columns:
            if not (sub[col].astype(str).values == ss[col].astype(str).values).all():
                errors.append(f"{col} values don't match SampleSubmission")

    # No null/NaN check
    null_count = sub.isnull().sum().sum()
    if null_count > 0:
        null_cols = sub.columns[sub.isnull().any()].tolist()
        errors.append(f"Found {null_count} null values in columns: {null_cols}")

    # No empty string check
    for col in sub.columns:
        empty = (sub[col].astype(str) == '').sum()
        if empty > 0:
            errors.append(f"Column {col} has {empty} empty strings")

    is_valid = len(errors) == 0
    return is_valid, errors


def compute_fill_stats(submission_path: str) -> pd.DataFrame:
    """Compute fill statistics per PXD and column."""
    sub = pd.read_csv(submission_path)
    stats = []

    for pxd in sub['PXD'].unique():
        pxd_df = sub[sub['PXD'] == pxd]
        n_rows = len(pxd_df)

        for col in META_COLUMNS:
            if col not in pxd_df.columns:
                continue
            vals = pxd_df[col].astype(str)
            n_na = (vals == DEFAULT_FILL).sum()
            n_real = n_rows - n_na
            n_unique = vals[vals != DEFAULT_FILL].nunique()

            stats.append({
                'pxd': pxd,
                'column': col,
                'n_rows': n_rows,
                'n_real': n_real,
                'n_na': n_na,
                'pct_real': round(n_real / n_rows * 100, 1),
                'n_unique_real': n_unique,
            })

    df = pd.DataFrame(stats)
    df.to_csv(ARTIFACTS_DIR / "fill_stats.csv", index=False)
    return df


def compute_coverage_summary(submission_path: str) -> pd.DataFrame:
    """Compute per-column coverage across all PXDs."""
    sub = pd.read_csv(submission_path)
    summary = []

    for col in META_COLUMNS:
        if col not in sub.columns:
            continue
        vals = sub[col].astype(str)
        n_total = len(vals)
        n_real = (vals != DEFAULT_FILL).sum()
        n_unique = vals[vals != DEFAULT_FILL].nunique()

        summary.append({
            'column': col,
            'n_total': n_total,
            'n_real': int(n_real),
            'pct_real': round(n_real / n_total * 100, 1),
            'n_unique': int(n_unique),
        })

    df = pd.DataFrame(summary)
    df.to_csv(ARTIFACTS_DIR / "coverage_summary.csv", index=False)
    return df


def run_offline_scoring(submission_path: str) -> Tuple[float, pd.DataFrame]:
    """
    Run offline grouped CV scoring using training data.
    Train on N-1 PXDs, score on held-out PXD, repeat.
    """
    # Add scorer to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from Scoring import score as scorer_fn

    from io_utils import load_train_sdrfs
    from config import SUBMISSION_COLUMNS

    sdrfs = load_train_sdrfs()

    # Build consolidated training solution
    all_rows = []
    for pxd, df in sdrfs.items():
        # Map training columns to submission columns
        mapped = _map_training_to_submission(pxd, df)
        all_rows.extend(mapped)

    if not all_rows:
        return 0.0, pd.DataFrame()

    solution_df = pd.DataFrame(all_rows)
    # Ensure all submission columns exist
    for col in SUBMISSION_COLUMNS:
        if col not in solution_df.columns:
            solution_df[col] = DEFAULT_FILL

    solution_df = solution_df[SUBMISSION_COLUMNS]

    # Generate submission for training PXDs using our pipeline
    # (This is expensive - skip for now, use test submission validation only)
    print("Offline scoring requires generating predictions for training PXDs.")
    print("Use validate_submission() for structural checks instead.")
    return 0.0, pd.DataFrame()


def _map_training_to_submission(pxd: str, df: pd.DataFrame) -> list:
    """Map a training SDRF to submission column format."""
    # Column name mapping from training -> submission
    col_map = {
        'Organism': 'Characteristics[Organism]',
        'OrganismPart': 'Characteristics[OrganismPart]',
        'CellType': 'Characteristics[CellType]',
        'CellLine': 'Characteristics[CellLine]',
        'Disease': 'Characteristics[Disease]',
        'MaterialType': 'Characteristics[MaterialType]',
        'Sex': 'Characteristics[Sex]',
        'Age': 'Characteristics[Age]',
        'DevelopmentalStage': 'Characteristics[DevelopmentalStage]',
        'AncestryCategory': 'Characteristics[AncestryCategory]',
        'BiologicalReplicate': 'Characteristics[BiologicalReplicate]',
        'Label': 'Characteristics[Label]',
        'Instrument': 'Comment[Instrument]',
        'FragmentationMethod': 'Comment[FragmentationMethod]',
        'CleavageAgent': 'Characteristics[CleavageAgent]',
        'FractionIdentifier': 'Comment[FractionIdentifier]',
        'PrecursorMassTolerance': 'Comment[PrecursorMassTolerance]',
        'FragmentMassTolerance': 'Comment[FragmentMassTolerance]',
        'CollisionEnergy': 'Comment[CollisionEnergy]',
        'MS2MassAnalyzer': 'Comment[MS2MassAnalyzer]',
        'FractionationMethod': 'Comment[FractionationMethod]',
        'Specimen': 'Characteristics[Specimen]',
        'Treatment': 'Characteristics[Treatment]',
        'Modification': 'Characteristics[Modification]',
        'Modification.1': 'Characteristics[Modification].1',
        'Modification.2': 'Characteristics[Modification].2',
        'Modification.3': 'Characteristics[Modification].3',
        'Modification.4': 'Characteristics[Modification].4',
        'Modification.5': 'Characteristics[Modification].5',
        'Modification.6': 'Characteristics[Modification].6',
    }

    rows = []
    for idx, row in df.iterrows():
        r = {
            'PXD': pxd,
            'Raw Data File': row.get('comment[data file]', row.get('AssayName', '')),
        }
        for train_col, sub_col in col_map.items():
            if train_col in df.columns:
                val = row[train_col]
                r[sub_col] = str(val) if pd.notna(val) else DEFAULT_FILL
        rows.append(r)

    return rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = str(Path(__file__).parent / "output" / "submission.csv")

    valid, errors = validate_submission(path)
    if valid:
        print("VALID: Submission passes all structural checks.")
    else:
        print("INVALID:")
        for e in errors:
            print(f"  - {e}")

    print("\nCoverage summary:")
    cov = compute_coverage_summary(path)
    print(cov.to_string(index=False))
