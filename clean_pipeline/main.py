#!/usr/bin/env python3
"""
Clean pipeline for Kaggle "Harmonizing the Data of Your Data" competition.
Builds SDRF-format submission from publication text and filename analysis.

Usage:
    cd clean_pipeline
    python main.py
"""
import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Setup
sys.path.insert(0, str(Path(__file__).parent))
from config import (OUTPUT_DIR, ARTIFACTS_DIR, CACHE_DIR,
                    SUBMISSION_COLUMNS, META_COLUMNS, DEFAULT_FILL, SAMPLE_SUB)
from io_utils import (load_sample_submission, load_test_pub_texts,
                      load_train_pub_texts, load_train_sdrfs,
                      get_test_pxds, get_skeleton)
from audit import build_canonical_inventory
from family_classifier import classify_experiment, get_primary_family, is_multiplexed
from text_extractors import extract_all
from filename_parser import parse_filename, parse_filename_group
from ontology import normalize_value
from row_reconstructor import reconstruct_rows, infer_number_counts
from retrieval import build_training_index, retrieve_similar, get_template_values
from fill_policy import (apply_fill_policy, apply_fill_policy_to_rows,
                         GLOBAL_DEFAULTS, HIGH_VALUE_COLUMNS)
from validate import validate_submission, compute_fill_stats, compute_coverage_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info("CLEAN PIPELINE — Harmonizing the Data of Your Data")
    log.info("=" * 60)

    # Stage 1: Audit
    log.info("STAGE 1: Building canonical inventory from training data...")
    col_freq, val_freq, rows_per_pxd = build_canonical_inventory()
    log.info(f"  Training: {len(rows_per_pxd)} PXDs, {len(col_freq)} unique columns")

    # Stage 2: Build training index for retrieval
    log.info("STAGE 2: Building training index for retrieval...")
    training_index = build_training_index()
    log.info(f"  Indexed {len(training_index)} training PXDs")

    # Save family classifications
    family_log = []
    for pxd, entry in training_index.items():
        family_log.append({
            'pxd': pxd,
            'primary_family': entry['primary_family'],
            'n_rows': entry['n_rows'],
            **{f'score_{k}': round(v, 2) for k, v in entry['family_scores'].items()}
        })
    pd.DataFrame(family_log).to_csv(ARTIFACTS_DIR / "training_families.csv", index=False)

    # Stage 3: Load test data
    log.info("STAGE 3: Loading test data...")
    test_texts = load_test_pub_texts()
    ss = load_sample_submission()
    test_pxds = get_test_pxds()
    log.info(f"  Test PXDs: {len(test_pxds)}, Total rows: {len(ss)}")

    # Stage 4-8: Process each test PXD
    all_results = []
    pxd_diagnostics = []

    for pxd in test_pxds:
        log.info(f"\n{'='*40}")
        log.info(f"Processing {pxd}...")
        log.info(f"{'='*40}")

        # Get skeleton rows
        skel = ss[ss['PXD'] == pxd].copy()
        n_rows = len(skel)
        filenames = skel['Raw Data File'].tolist()
        pub_text = test_texts.get(pxd, {})

        log.info(f"  Rows: {n_rows}, Files: {len(set(filenames))}")

        # Stage 4: Family classification
        family_scores = classify_experiment(pxd, pub_text, filenames, n_rows)
        primary_family = get_primary_family(family_scores)
        log.info(f"  Family: {primary_family} (scores: {family_scores})")

        # Stage 5: Text extraction
        evidence = extract_all(pub_text)
        n_evidence = sum(len(v) for v in evidence.values())
        log.info(f"  Evidence items: {n_evidence}")

        for col, items in evidence.items():
            if items:
                log.info(f"    {col}: {[(v, round(c, 2)) for v, c, _ in items[:3]]}")

        # Stage 6: Filename analysis
        fn_analysis = parse_filename_group(filenames)
        log.info(f"  Filename signals: fractions={fn_analysis['has_fractions']}, "
                 f"channels={fn_analysis['has_channels']}, "
                 f"replicates={fn_analysis['has_replicates']}, "
                 f"treatment={fn_analysis['has_treatment']}, "
                 f"temperature={fn_analysis['has_temperature']}")

        # Stage 7: Retrieval
        similar = retrieve_similar(pxd, pub_text, filenames, n_rows, training_index, top_k=5)
        log.info(f"  Similar training PXDs: {[(p, round(s, 2)) for p, s, _ in similar[:3]]}")

        # Build skeleton rows as list of dicts
        skeleton_rows = []
        for _, row in skel.iterrows():
            skeleton_rows.append({
                'ID': row['ID'],
                'PXD': row['PXD'],
                'Raw Data File': row['Raw Data File'],
                'Usage': row['Usage'],
            })

        # Stage 6: Row reconstruction
        results = reconstruct_rows(pxd, skeleton_rows, filenames, evidence,
                                   family_scores, pub_text)

        # Apply number counts
        counts = infer_number_counts(pub_text, n_rows, filenames)
        for col, val in counts.items():
            for r in results:
                if col not in r or r[col] in ('Text Span', 'Not Applicable', None, ''):
                    r[col] = val

        # Stage 7: Apply retrieval-backed candidates for unfilled columns
        # Only use retrieval for safe, high-signal columns where similar PXDs
        # reliably share the same values
        RETRIEVAL_SAFE_COLUMNS = {
            'Characteristics[CleavageAgent]', 'Comment[Instrument]',
            'Comment[FragmentationMethod]', 'Comment[Separation]',
            'Comment[AcquisitionMethod]', 'Characteristics[Label]',
            'Characteristics[Modification]', 'Characteristics[Modification].1',
            'Characteristics[Modification].2',
            'Comment[NumberOfMissedCleavages]',
            'Characteristics[AlkylationReagent]', 'Characteristics[ReductionReagent]',
        }
        for col in RETRIEVAL_SAFE_COLUMNS:
            if col not in HIGH_VALUE_COLUMNS:
                continue
            ret_cands = get_template_values(similar, col)
            if ret_cands:
                for r in results:
                    current = r.get(col)
                    if not current or current in ('Text Span', 'Not Applicable', None, ''):
                        best = ret_cands[0]
                        if best[1] > 5.0:  # strong retrieval signal
                            r[col] = normalize_value(col, best[0])

        # Stage 8: Fill policy
        results = apply_fill_policy_to_rows(results, primary_family, similar)

        # Track diagnostics
        n_real = 0
        n_total = 0
        for r in results:
            for col in META_COLUMNS:
                n_total += 1
                val = r.get(col, DEFAULT_FILL)
                if val != DEFAULT_FILL:
                    n_real += 1

        pxd_diagnostics.append({
            'pxd': pxd,
            'n_rows': n_rows,
            'primary_family': primary_family,
            'n_real_values': n_real,
            'n_total_cells': n_total,
            'pct_real': round(n_real / max(n_total, 1) * 100, 1),
            'n_evidence': n_evidence,
            'top_similar': similar[0][0] if similar else '',
        })

        all_results.extend(results)
        log.info(f"  Real values: {n_real}/{n_total} ({round(n_real/max(n_total,1)*100,1)}%)")

    # Stage 9: Build final submission
    log.info("\n" + "=" * 60)
    log.info("STAGE 9: Building final submission...")

    submission_df = pd.DataFrame(all_results)

    # Ensure all submission columns exist
    for col in SUBMISSION_COLUMNS:
        if col not in submission_df.columns:
            submission_df[col] = DEFAULT_FILL

    # Ensure all submission columns exist
    for col in SUBMISSION_COLUMNS:
        if col not in submission_df.columns:
            submission_df[col] = DEFAULT_FILL

    # Reorder columns to match exactly
    submission_df = submission_df[SUBMISSION_COLUMNS]

    # Sort by ID to match SampleSubmission row order
    submission_df['ID'] = submission_df['ID'].astype(int)
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)

    # Final null/NaN sweep
    submission_df = submission_df.fillna(DEFAULT_FILL)
    for col in submission_df.columns:
        submission_df[col] = submission_df[col].astype(str)
        submission_df[col] = submission_df[col].replace('', DEFAULT_FILL)
        submission_df[col] = submission_df[col].replace('nan', DEFAULT_FILL)
        submission_df[col] = submission_df[col].replace('None', DEFAULT_FILL)

    # Verify row order matches SampleSubmission
    ss_reload = load_sample_submission()
    assert len(submission_df) == len(ss_reload), f"Row count: {len(submission_df)} vs {len(ss_reload)}"
    assert (submission_df['ID'].astype(str).values == ss_reload['ID'].astype(str).values).all(), "ID mismatch"
    assert (submission_df['PXD'].values == ss_reload['PXD'].values).all(), "PXD mismatch"
    assert (submission_df['Raw Data File'].values == ss_reload['Raw Data File'].values).all(), "Raw Data File mismatch"

    # Save
    submission_path = OUTPUT_DIR / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    log.info(f"Submission saved to {submission_path}")
    log.info(f"Shape: {submission_df.shape}")

    # Stage 10: Validation
    log.info("\nSTAGE 10: Validation...")
    valid, errors = validate_submission(str(submission_path))
    if valid:
        log.info("  VALID: Submission passes all structural checks.")
    else:
        log.error("  INVALID:")
        for e in errors:
            log.error(f"    - {e}")

    # Coverage summary
    cov = compute_coverage_summary(str(submission_path))
    log.info("\nCoverage summary (top 20 by real value %):")
    cov_sorted = cov.sort_values('pct_real', ascending=False)
    for _, row in cov_sorted.head(20).iterrows():
        log.info(f"  {row['column']}: {row['pct_real']}% real ({row['n_unique']} unique)")

    log.info(f"\nOverall: {cov['n_real'].sum()} real / {cov['n_total'].sum()} total "
             f"({round(cov['n_real'].sum()/cov['n_total'].sum()*100, 1)}%)")

    # Fill stats
    fill_stats = compute_fill_stats(str(submission_path))

    # Save diagnostics
    pd.DataFrame(pxd_diagnostics).to_csv(ARTIFACTS_DIR / "pxd_diagnostics.csv", index=False)

    # Test family classifications
    test_families = []
    for pxd in test_pxds:
        pub = test_texts.get(pxd, {})
        fns = ss[ss['PXD'] == pxd]['Raw Data File'].tolist()
        nr = len(ss[ss['PXD'] == pxd])
        fscores = classify_experiment(pxd, pub, fns, nr)
        test_families.append({
            'pxd': pxd,
            'primary_family': get_primary_family(fscores),
            'n_rows': nr,
            **{f'score_{k}': round(v, 2) for k, v in fscores.items()}
        })
    pd.DataFrame(test_families).to_csv(ARTIFACTS_DIR / "test_families.csv", index=False)

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"Submission: {submission_path}")
    log.info(f"Artifacts: {ARTIFACTS_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
