"""Audit training data to build canonical value inventories."""
import json
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from config import ARTIFACTS_DIR
from io_utils import load_train_sdrfs, load_train_pub_texts


def build_canonical_inventory():
    """Build frequency tables of canonical values per column from training SDRFs."""
    sdrfs = load_train_sdrfs()

    col_freq = Counter()       # column_name -> count of PXDs having it
    val_freq = defaultdict(Counter)  # column_name -> {value: count}
    rows_per_pxd = {}
    col_non_na = defaultdict(int)  # column -> count of non-NA values total

    for pxd, df in sdrfs.items():
        rows_per_pxd[pxd] = len(df)
        for c in df.columns:
            col_freq[c] += 1
            for v in df[c].dropna().unique():
                sv = str(v).strip()
                if sv and sv.lower() not in ('nan', ''):
                    val_freq[c][sv] += 1
                    col_non_na[c] += 1

    # Save column frequency
    col_freq_df = pd.DataFrame([
        {'column': k, 'pxd_count': v, 'pct': round(v / len(sdrfs) * 100, 1)}
        for k, v in col_freq.most_common()
    ])
    col_freq_df.to_csv(ARTIFACTS_DIR / "column_frequency.csv", index=False)

    # Save canonical values per column
    for col, ctr in val_freq.items():
        safe_name = col.replace('[', '_').replace(']', '_').replace('/', '_').replace('.', '_')
        rows = [{'value': v, 'count': c} for v, c in ctr.most_common()]
        pd.DataFrame(rows).to_csv(
            ARTIFACTS_DIR / f"canonical_{safe_name}.csv", index=False
        )

    # Save rows per PXD
    pd.DataFrame([
        {'pxd': k, 'rows': v} for k, v in sorted(rows_per_pxd.items())
    ]).to_csv(ARTIFACTS_DIR / "rows_per_pxd_training.csv", index=False)

    print(f"Audit complete: {len(sdrfs)} training PXDs, {len(col_freq)} unique columns")
    return col_freq, val_freq, rows_per_pxd


def get_canonical_values(col_name: str, val_freq: dict = None) -> list:
    """Get ordered list of canonical values for a column from training data."""
    if val_freq is None:
        _, val_freq, _ = build_canonical_inventory()

    # Try exact match first
    if col_name in val_freq:
        return [v for v, _ in val_freq[col_name].most_common()]

    # Try without prefix
    bare = col_name.split('[')[-1].rstrip(']') if '[' in col_name else col_name
    for k, v in val_freq.items():
        if bare.lower() in k.lower():
            return [val for val, _ in v.most_common()]
    return []


if __name__ == "__main__":
    build_canonical_inventory()
