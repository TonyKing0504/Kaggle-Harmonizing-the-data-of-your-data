"""Training PXD retrieval: find similar training experiments to guide extraction."""
import re
from typing import Dict, List, Tuple
from collections import Counter
from io_utils import load_train_sdrfs, load_train_pub_texts
from family_classifier import classify_experiment, get_primary_family
from filename_parser import parse_filename_group


def build_training_index() -> Dict[str, Dict]:
    """Build an index of training PXD characteristics for retrieval."""
    sdrfs = load_train_sdrfs()
    texts = load_train_pub_texts()

    index = {}
    for pxd, df in sdrfs.items():
        pub = texts.get(pxd, {})
        filenames = df['comment[data file]'].dropna().tolist() if 'comment[data file]' in df.columns else []

        # Classify family
        family_scores = classify_experiment(pxd, pub, filenames, len(df))

        # Extract key features
        organisms = df['Organism'].dropna().unique().tolist() if 'Organism' in df.columns else []
        instruments = df['Instrument'].dropna().unique().tolist() if 'Instrument' in df.columns else []
        labels = df['Label'].dropna().unique().tolist() if 'Label' in df.columns else []

        fn_analysis = parse_filename_group(filenames) if filenames else {}

        index[pxd] = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': list(df.columns),
            'family_scores': family_scores,
            'primary_family': get_primary_family(family_scores),
            'organisms': organisms,
            'instruments': instruments,
            'labels': labels,
            'fn_analysis': fn_analysis,
            'has_fractions': fn_analysis.get('has_fractions', False),
            'has_channels': fn_analysis.get('has_channels', False),
            'sdrf': df,
        }

    return index


def retrieve_similar(test_pxd: str, test_pub: Dict, test_filenames: List[str],
                     test_n_rows: int, training_index: Dict,
                     top_k: int = 5) -> List[Tuple[str, float, Dict]]:
    """
    Find most similar training PXDs.

    Returns: [(pxd, score, index_entry), ...] sorted by score descending
    """
    test_family_scores = classify_experiment(test_pxd, test_pub, test_filenames, test_n_rows)
    test_primary = get_primary_family(test_family_scores)
    test_fn_analysis = parse_filename_group(test_filenames)
    test_methods = (test_pub.get('METHODS', '') or '').lower()

    scores = []
    for train_pxd, entry in training_index.items():
        sim = 0.0

        # Family similarity (highest weight)
        if entry['primary_family'] == test_primary:
            sim += 3.0
        # Partial family overlap
        for fam, s in test_family_scores.items():
            if fam in entry['family_scores']:
                sim += min(s, entry['family_scores'][fam]) * 1.0

        # Row count similarity
        ratio = min(test_n_rows, entry['n_rows']) / max(test_n_rows, entry['n_rows'], 1)
        sim += ratio * 2.0

        # Fractionation match
        if test_fn_analysis.get('has_fractions') == entry.get('has_fractions'):
            sim += 1.0
        if test_fn_analysis.get('has_channels') == entry.get('has_channels'):
            sim += 1.0

        # Organism overlap (from text)
        test_organisms = set()
        for org in entry['organisms']:
            org_lower = org.lower()
            if any(x in test_methods for x in [org_lower, org_lower.split()[0]]):
                test_organisms.add(org_lower)
        if test_organisms:
            sim += 1.5

        # Column count similarity
        col_ratio = min(len(entry['columns']), 30) / 30
        sim += col_ratio * 0.5

        scores.append((train_pxd, sim, entry))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def get_template_values(similar_pxds: List[Tuple[str, float, Dict]],
                        column: str) -> List[Tuple[str, float]]:
    """
    Get candidate values for a column from similar training PXDs.

    Returns: [(value, weight), ...] sorted by weight
    """
    value_weights = Counter()

    for pxd, sim_score, entry in similar_pxds:
        sdrf = entry['sdrf']
        # Check for the column in the training SDRF
        matching_cols = [c for c in sdrf.columns if _column_matches(c, column)]
        for mc in matching_cols:
            for v in sdrf[mc].dropna().unique():
                sv = str(v).strip()
                if sv and sv.lower() not in ('nan', 'not available', 'not applicable', ''):
                    value_weights[sv] += sim_score

    result = [(v, w) for v, w in value_weights.most_common(10)]
    return result


def _column_matches(train_col: str, submission_col: str) -> bool:
    """Check if a training column name matches a submission column name."""
    # Extract bare name from submission format
    # e.g. 'Characteristics[Organism]' -> 'Organism'
    # e.g. 'Comment[Instrument]' -> 'Instrument'
    bare_sub = submission_col
    m = re.match(r'(?:Characteristics|Comment|FactorValue)\[(\w+)\]', submission_col)
    if m:
        bare_sub = m.group(1)

    train_lower = train_col.lower()
    bare_lower = bare_sub.lower()

    return (train_lower == bare_lower or
            bare_lower in train_lower or
            train_lower.replace(' ', '') == bare_lower.replace(' ', ''))
