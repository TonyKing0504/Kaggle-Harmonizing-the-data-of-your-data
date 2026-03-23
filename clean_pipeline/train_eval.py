#!/usr/bin/env python3
"""Evaluate pipeline on training data using the official scorer."""
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (SUBMISSION_COLUMNS, META_COLUMNS, DEFAULT_FILL,
                    ARTIFACTS_DIR, TRAIN_SDRF_DIR)
from io_utils import load_train_sdrfs, load_train_pub_texts
from family_classifier import classify_experiment, get_primary_family
from text_extractors import extract_all
from filename_parser import parse_filename_group
from ontology import normalize_value
from row_reconstructor import reconstruct_rows, infer_number_counts
from retrieval import build_training_index, retrieve_similar, get_template_values
from fill_policy import apply_fill_policy_to_rows, HIGH_VALUE_COLUMNS, GLOBAL_DEFAULTS
from Scoring import score as scorer_fn


def build_solution_df(sdrfs: dict) -> pd.DataFrame:
    """Build solution DataFrame from training SDRFs."""
    from validate import _map_training_to_submission

    all_rows = []
    for pxd, df in sdrfs.items():
        rows = _map_training_to_submission(pxd, df)
        all_rows.extend(rows)

    sol = pd.DataFrame(all_rows)
    sol['ID'] = range(1, len(sol) + 1)
    sol['Usage'] = 'Not Applicable'

    for col in SUBMISSION_COLUMNS:
        if col not in sol.columns:
            sol[col] = DEFAULT_FILL

    sol = sol[SUBMISSION_COLUMNS].fillna(DEFAULT_FILL)
    return sol


def run_pipeline_on_pxd(pxd, pub_text, filenames, n_rows, training_index):
    """Run extraction pipeline on a single PXD."""
    family_scores = classify_experiment(pxd, pub_text, filenames, n_rows)
    primary_family = get_primary_family(family_scores)
    evidence = extract_all(pub_text)

    # Build skeleton
    skeleton_rows = []
    for i, fn in enumerate(filenames):
        skeleton_rows.append({
            'ID': i + 1,  # placeholder
            'PXD': pxd,
            'Raw Data File': fn,
            'Usage': 'Not Applicable',
        })

    similar = retrieve_similar(pxd, pub_text, filenames, n_rows, training_index, top_k=5)
    results = reconstruct_rows(pxd, skeleton_rows, filenames, evidence,
                               family_scores, pub_text)

    counts = infer_number_counts(pub_text, n_rows, filenames)
    for col, val in counts.items():
        for r in results:
            if col not in r or r[col] in ('Text Span', 'Not Applicable', None, ''):
                r[col] = val

    # Retrieval for safe columns
    RETRIEVAL_SAFE = {
        'Characteristics[CleavageAgent]', 'Comment[Instrument]',
        'Comment[FragmentationMethod]', 'Comment[Separation]',
        'Comment[AcquisitionMethod]', 'Characteristics[Label]',
        'Characteristics[Modification]', 'Characteristics[Modification].1',
        'Characteristics[Modification].2',
        'Comment[NumberOfMissedCleavages]',
        'Characteristics[AlkylationReagent]', 'Characteristics[ReductionReagent]',
    }
    for col in RETRIEVAL_SAFE:
        ret_cands = get_template_values(similar, col)
        if ret_cands:
            for r in results:
                current = r.get(col)
                if not current or current in ('Text Span', 'Not Applicable', None, ''):
                    best = ret_cands[0]
                    if best[1] > 5.0:
                        r[col] = normalize_value(col, best[0])

    results = apply_fill_policy_to_rows(results, primary_family, similar)
    return results


def main():
    print("Loading training data...")
    sdrfs = load_train_sdrfs()
    texts = load_train_pub_texts()

    # Build solution
    print("Building solution...")
    solution_df = build_solution_df(sdrfs)

    # Build training index (using all training PXDs)
    print("Building training index...")
    training_index = build_training_index()

    # Run pipeline on a sample of training PXDs
    # Use leave-one-out: for each PXD, remove it from training index first
    sample_pxds = sorted(sdrfs.keys())[:20]  # evaluate on first 20
    print(f"Evaluating on {len(sample_pxds)} training PXDs...")

    all_pred_rows = []
    for pxd in sample_pxds:
        if pxd not in texts:
            continue

        pub_text = texts[pxd]
        sdrf = sdrfs[pxd]

        # Get filenames from training SDRF
        if 'comment[data file]' in sdrf.columns:
            filenames = sdrf['comment[data file]'].tolist()
        elif 'AssayName' in sdrf.columns:
            filenames = sdrf['AssayName'].tolist()
        else:
            filenames = [f"file_{i}" for i in range(len(sdrf))]

        # Remove this PXD from training index for leave-one-out
        temp_index = {k: v for k, v in training_index.items() if k != pxd}

        n_rows = len(sdrf)
        results = run_pipeline_on_pxd(pxd, pub_text, filenames, n_rows, temp_index)
        all_pred_rows.extend(results)

    # Build prediction DataFrame
    pred_df = pd.DataFrame(all_pred_rows)
    pred_df['ID'] = range(1, len(pred_df) + 1)
    for col in SUBMISSION_COLUMNS:
        if col not in pred_df.columns:
            pred_df[col] = DEFAULT_FILL
    pred_df = pred_df[SUBMISSION_COLUMNS].fillna(DEFAULT_FILL)

    # Filter solution to only evaluated PXDs
    eval_pxds = set(pred_df['PXD'].unique())
    sol_filtered = solution_df[solution_df['PXD'].isin(eval_pxds)].copy()
    sol_filtered['ID'] = range(1, len(sol_filtered) + 1)

    print(f"\nScoring {len(eval_pxds)} PXDs...")
    print(f"Solution rows: {len(sol_filtered)}, Prediction rows: {len(pred_df)}")

    try:
        eval_df, final_score = scorer_fn(sol_filtered, pred_df, 'ID')
        print(f"\n{'='*50}")
        print(f"OFFLINE F1 SCORE: {final_score:.4f}")
        print(f"{'='*50}")

        # Per-PXD breakdown
        pxd_scores = eval_df.groupby('pxd')['f1'].mean().sort_values()
        print("\nPer-PXD F1 scores:")
        for pxd, f1 in pxd_scores.items():
            print(f"  {pxd}: {f1:.4f}")

        # Per-field breakdown
        field_scores = eval_df.groupby('AnnotationType')['f1'].mean().sort_values()
        print("\nPer-field F1 scores (bottom 20):")
        for field, f1 in field_scores.head(20).items():
            print(f"  {field}: {f1:.4f}")

        print("\nPer-field F1 scores (top 20):")
        for field, f1 in field_scores.tail(20).items():
            print(f"  {field}: {f1:.4f}")

        # Save
        eval_df.to_csv(ARTIFACTS_DIR / "train_eval_detailed.csv", index=False)
        print(f"\nDetailed metrics saved to {ARTIFACTS_DIR / 'train_eval_detailed.csv'}")

    except Exception as e:
        print(f"Scoring error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
