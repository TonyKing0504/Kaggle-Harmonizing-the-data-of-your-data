"""I/O utilities for loading competition data."""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import (DATA_DIR, TRAIN_SDRF_DIR, TRAIN_TEXT_DIR, TEST_TEXT_DIR,
                    SAMPLE_SUB, SUBMISSION_COLUMNS)


def load_sample_submission() -> pd.DataFrame:
    """Load SampleSubmission.csv as the row skeleton."""
    return pd.read_csv(SAMPLE_SUB)


def load_test_pub_texts() -> Dict[str, Dict[str, Any]]:
    """Load all test publication JSON files into {PXD: {section: text}}."""
    agg = TEST_TEXT_DIR / "PubText.json"
    if agg.exists():
        with open(agg) as f:
            return json.load(f)
    # Fallback: individual files
    result = {}
    for p in sorted(TEST_TEXT_DIR.glob("PXD*_PubText.json")):
        pxd = p.stem.replace("_PubText", "")
        with open(p) as f:
            result[pxd] = json.load(f)
    return result


def load_train_pub_texts() -> Dict[str, Dict[str, Any]]:
    """Load all training publication JSON files."""
    agg = TRAIN_TEXT_DIR / "PubText.json"
    if agg.exists():
        with open(agg) as f:
            return json.load(f)
    result = {}
    for p in sorted(TRAIN_TEXT_DIR.glob("PXD*_PubText.json")):
        pxd = p.stem.replace("_PubText", "")
        with open(p) as f:
            result[pxd] = json.load(f)
    return result


def load_train_sdrfs() -> Dict[str, pd.DataFrame]:
    """Load all training SDRF TSV files as {PXD: DataFrame}."""
    result = {}
    for p in sorted(TRAIN_SDRF_DIR.glob("PXD*_cleaned.sdrf.tsv")):
        pxd = p.stem.replace("_cleaned.sdrf", "")
        df = pd.read_csv(p, sep='\t')
        result[pxd] = df
    return result


def get_test_pxds() -> List[str]:
    """Return sorted list of test PXD IDs."""
    ss = load_sample_submission()
    return sorted(ss['PXD'].unique().tolist())


def get_skeleton(pxd: str) -> pd.DataFrame:
    """Return the SampleSubmission rows for a single PXD."""
    ss = load_sample_submission()
    return ss[ss['PXD'] == pxd].copy().reset_index(drop=True)


def build_submission_template() -> pd.DataFrame:
    """Build empty submission from SampleSubmission skeleton with proper structure."""
    ss = load_sample_submission()
    # Keep ID, PXD, Raw Data File, Usage as-is
    result = ss[['ID', 'PXD', 'Raw Data File', 'Usage']].copy()
    return result
