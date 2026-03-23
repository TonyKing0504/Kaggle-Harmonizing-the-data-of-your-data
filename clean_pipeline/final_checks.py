#!/usr/bin/env python3
"""Final structural checks and submission validator."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import SAMPLE_SUB, SUBMISSION_COLUMNS, DEFAULT_FILL, OUTPUT_DIR


def final_check(submission_path: str = None):
    """Run comprehensive final checks on submission."""
    if submission_path is None:
        submission_path = str(OUTPUT_DIR / "submission.csv")

    print(f"Checking: {submission_path}")
    ss = pd.read_csv(SAMPLE_SUB)
    sub = pd.read_csv(submission_path)

    checks = []

    # 1. Column match
    ok = list(ss.columns) == list(sub.columns)
    checks.append(('Column names and order match', ok))
    if not ok:
        print(f"  Expected: {list(ss.columns)[:5]}...")
        print(f"  Got:      {list(sub.columns)[:5]}...")

    # 2. Row count
    ok = len(ss) == len(sub)
    checks.append(('Row count match', ok))
    if not ok:
        print(f"  Expected: {len(ss)}, Got: {len(sub)}")

    # 3. ID exact match
    ok = (sub['ID'].astype(str).values == ss['ID'].astype(str).values).all()
    checks.append(('ID values match', ok))

    # 4. PXD exact match
    ok = (sub['PXD'].values == ss['PXD'].values).all()
    checks.append(('PXD values match', ok))

    # 5. Raw Data File match
    ok = (sub['Raw Data File'].values == ss['Raw Data File'].values).all()
    checks.append(('Raw Data File values match', ok))

    # 6. No null/NaN
    null_count = sub.isnull().sum().sum()
    ok = null_count == 0
    checks.append(('No null/NaN values', ok))
    if not ok:
        print(f"  Null count: {null_count}")

    # 7. No empty strings
    empty_count = sum((sub[col].astype(str) == '').sum() for col in sub.columns)
    ok = empty_count == 0
    checks.append(('No empty strings', ok))

    # 8. No 'nan' strings
    nan_str_count = sum((sub[col].astype(str) == 'nan').sum() for col in sub.columns)
    ok = nan_str_count == 0
    checks.append(('No "nan" strings', ok))

    # 9. No 'None' strings
    none_str_count = sum((sub[col].astype(str) == 'None').sum() for col in sub.columns)
    ok = none_str_count == 0
    checks.append(('No "None" strings', ok))

    # 10. No "Text Span" values
    text_span_count = sum((sub[col].astype(str) == 'Text Span').sum() for col in sub.columns)
    ok = text_span_count == 0
    checks.append(('No "Text Span" values', ok))
    if not ok:
        print(f"  Text Span count: {text_span_count}")

    # Print results
    print("\n" + "=" * 50)
    print("FINAL VALIDATION RESULTS")
    print("=" * 50)
    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    # Coverage stats
    print("\n" + "-" * 50)
    print("COVERAGE STATISTICS")
    print("-" * 50)
    total_cells = 0
    real_cells = 0
    na_cells = 0
    for col in sub.columns:
        if col in ('ID', 'PXD', 'Raw Data File', 'Usage'):
            continue
        vals = sub[col].astype(str)
        n = len(vals)
        n_na = (vals == DEFAULT_FILL).sum()
        total_cells += n
        real_cells += n - n_na
        na_cells += n_na

    print(f"  Total metadata cells: {total_cells}")
    print(f"  Real values: {real_cells} ({round(real_cells/total_cells*100, 1)}%)")
    print(f"  'Not Applicable': {na_cells} ({round(na_cells/total_cells*100, 1)}%)")

    # Per-PXD coverage
    print("\n  Per-PXD coverage:")
    for pxd in sub['PXD'].unique():
        pxd_df = sub[sub['PXD'] == pxd]
        n_rows = len(pxd_df)
        n_real = 0
        n_total = 0
        for col in sub.columns:
            if col in ('ID', 'PXD', 'Raw Data File', 'Usage'):
                continue
            n_total += n_rows
            n_real += (pxd_df[col].astype(str) != DEFAULT_FILL).sum()
        print(f"    {pxd}: {n_rows} rows, {round(n_real/max(n_total,1)*100,1)}% real")

    print("\n" + "=" * 50)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 50)

    return all_pass


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    final_check(path)
