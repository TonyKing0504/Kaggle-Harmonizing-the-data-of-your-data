#!/usr/bin/env python3
"""
Autoresearch-style experiment runner for the Kaggle pipeline.
Runs the pipeline on training data, measures F1, returns the metric.

Usage: python3 experiment.py
Output: prints key metrics for grep extraction
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_experiment():
    """Run full pipeline evaluation on training data and return F1 score."""
    start = time.time()

    from config import SUBMISSION_COLUMNS, META_COLUMNS, DEFAULT_FILL, ARTIFACTS_DIR
    from io_utils import load_train_sdrfs, load_train_pub_texts
    from family_classifier import classify_experiment, get_primary_family
    from text_extractors import extract_all
    from filename_parser import parse_filename_group
    from ontology import normalize_value
    from row_reconstructor import reconstruct_rows, infer_number_counts
    from retrieval import build_training_index, retrieve_similar, get_template_values
    from fill_policy import apply_fill_policy_to_rows, HIGH_VALUE_COLUMNS
    from validate import _map_training_to_submission
    from Scoring import score as scorer_fn
    import pandas as pd

    sdrfs = load_train_sdrfs()
    texts = load_train_pub_texts()

    # Build solution
    all_sol_rows = []
    for pxd, df in sdrfs.items():
        rows = _map_training_to_submission(pxd, df)
        all_sol_rows.extend(rows)
    sol = pd.DataFrame(all_sol_rows)
    sol['ID'] = range(1, len(sol) + 1)
    sol['Usage'] = 'Not Applicable'
    for col in SUBMISSION_COLUMNS:
        if col not in sol.columns:
            sol[col] = DEFAULT_FILL
    sol = sol[SUBMISSION_COLUMNS].fillna(DEFAULT_FILL)

    # Build training index
    training_index = build_training_index()

    # Evaluate on 30 training PXDs (more than before for better signal)
    sample_pxds = sorted(sdrfs.keys())[:30]
    all_pred_rows = []

    for pxd in sample_pxds:
        if pxd not in texts:
            continue
        pub_text = texts[pxd]
        sdrf = sdrfs[pxd]

        if 'comment[data file]' in sdrf.columns:
            filenames = sdrf['comment[data file]'].tolist()
        elif 'AssayName' in sdrf.columns:
            filenames = sdrf['AssayName'].tolist()
        else:
            filenames = [f"file_{i}" for i in range(len(sdrf))]

        temp_index = {k: v for k, v in training_index.items() if k != pxd}
        n_rows = len(sdrf)

        # Run pipeline
        family_scores = classify_experiment(pxd, pub_text, filenames, n_rows)
        primary_family = get_primary_family(family_scores)
        evidence = extract_all(pub_text)

        skeleton_rows = []
        for i, fn in enumerate(filenames):
            skeleton_rows.append({
                'ID': i + 1,
                'PXD': pxd,
                'Raw Data File': fn,
                'Usage': 'Not Applicable',
            })

        similar = retrieve_similar(pxd, pub_text, filenames, n_rows, temp_index, top_k=5)
        results = reconstruct_rows(pxd, skeleton_rows, filenames, evidence,
                                   family_scores, pub_text)

        counts = infer_number_counts(pub_text, n_rows, filenames)
        for col, val in counts.items():
            for r in results:
                if col not in r or r[col] in ('Text Span', 'Not Applicable', None, ''):
                    r[col] = val

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
        all_pred_rows.extend(results)

    # Build prediction DataFrame
    pred_df = pd.DataFrame(all_pred_rows)
    pred_df['ID'] = range(1, len(pred_df) + 1)
    for col in SUBMISSION_COLUMNS:
        if col not in pred_df.columns:
            pred_df[col] = DEFAULT_FILL
    pred_df = pred_df[SUBMISSION_COLUMNS].fillna(DEFAULT_FILL)

    # Filter solution
    eval_pxds = set(pred_df['PXD'].unique())
    sol_filtered = sol[sol['PXD'].isin(eval_pxds)].copy()
    sol_filtered['ID'] = range(1, len(sol_filtered) + 1)

    # Score
    eval_df, final_score = scorer_fn(sol_filtered, pred_df, 'ID')

    elapsed = time.time() - start

    # Per-field breakdown
    field_scores = eval_df.groupby('AnnotationType')['f1'].mean()
    pxd_scores = eval_df.groupby('pxd')['f1'].mean()

    # Print in grep-friendly format
    print("---")
    print(f"val_f1:           {final_score:.6f}")
    print(f"eval_seconds:     {elapsed:.1f}")
    print(f"num_pxds:         {len(eval_pxds)}")
    print(f"num_rows:         {len(pred_df)}")
    print(f"num_fields_eval:  {len(field_scores)}")
    print(f"worst_pxd:        {pxd_scores.idxmin()} ({pxd_scores.min():.4f})")
    print(f"best_pxd:         {pxd_scores.idxmax()} ({pxd_scores.max():.4f})")

    # Top 5 worst fields
    worst_fields = field_scores.nsmallest(5)
    for field, f1 in worst_fields.items():
        print(f"weak_field:       {field} = {f1:.4f}")

    # Save detailed metrics
    eval_df.to_csv(ARTIFACTS_DIR / "experiment_eval.csv", index=False)

    return final_score


if __name__ == "__main__":
    try:
        score = run_experiment()
    except Exception as e:
        print("---")
        print(f"val_f1:           0.000000")
        print(f"status:           crash")
        print(f"error:            {e}")
        traceback.print_exc()
        sys.exit(1)
