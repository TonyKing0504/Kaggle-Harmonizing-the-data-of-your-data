"""Classify each PXD into experiment family types."""
import re
from typing import Dict, List, Set
from filename_parser import parse_filename_group


# Experiment family types
FAMILIES = [
    'label_free_simple',       # Simple one-row-per-file label-free
    'label_free_fractionated', # Label-free with fractionation
    'tmt_multiplexed',         # TMT labeled experiments
    'itraq_multiplexed',       # iTRAQ labeled experiments
    'silac',                   # SILAC labeling
    'ap_ms',                   # Affinity purification MS
    'dia',                     # DIA acquisition
    'dda',                     # DDA acquisition
    'clinical_cohort',         # Patient/clinical specimens
    'cell_line',               # Cell line experiments
    'temperature_series',      # CETSA/TPP temperature profiling
    'time_series',             # Time course experiments
    'dose_response',           # Dose response
    'treatment_control',       # Treatment vs control
    'tissue',                  # Tissue proteomics
]


def classify_experiment(pxd: str, pub_text: Dict, filenames: List[str],
                        n_rows: int) -> Dict[str, float]:
    """Classify a PXD into experiment families with confidence scores."""
    scores = {}
    methods = (pub_text.get('METHODS', '') or '').lower()
    abstract = (pub_text.get('ABSTRACT', '') or '').lower()
    title = (pub_text.get('TITLE', '') or '').lower()
    full_text = methods + ' ' + abstract + ' ' + title
    results = (pub_text.get('RESULTS', '') or '').lower()
    full_text += ' ' + results

    fn_analysis = parse_filename_group(filenames)

    # Label-free detection
    lf_signals = _count_patterns(full_text, [
        r'label[\s-]*free', r'lfq', r'shotgun', r'data[\s-]*dependent'
    ])
    if lf_signals > 0 or (not fn_analysis['has_channels'] and 'tmt' not in full_text
                          and 'itraq' not in full_text and 'silac' not in full_text):
        if fn_analysis['has_fractions']:
            scores['label_free_fractionated'] = 0.8
        else:
            scores['label_free_simple'] = 0.8

    # TMT detection
    tmt_signals = _count_patterns(full_text, [
        r'\btmt\b', r'tandem mass tag', r'tmt[\s-]*pro', r'tmt\d+plex',
        r'tmt[\s-]*16', r'tmt[\s-]*11', r'tmt[\s-]*10', r'tmt[\s-]*6'
    ])
    if tmt_signals > 0 or fn_analysis['has_channels']:
        scores['tmt_multiplexed'] = min(1.0, 0.5 + tmt_signals * 0.15)

    # iTRAQ detection
    itraq_signals = _count_patterns(full_text, [r'\bitraq\b', r'isobaric tag'])
    if itraq_signals > 0:
        scores['itraq_multiplexed'] = min(1.0, 0.5 + itraq_signals * 0.2)

    # SILAC detection
    silac_signals = _count_patterns(full_text, [
        r'\bsilac\b', r'stable isotope labeling', r'heavy[\s/]*light',
        r'heavy[\s/]*medium'
    ])
    if silac_signals > 0:
        scores['silac'] = min(1.0, 0.5 + silac_signals * 0.2)

    # DIA detection
    dia_signals = _count_patterns(full_text, [
        r'\bdia\b', r'data[\s-]*independent', r'swath', r'dia[\s-]*nn'
    ])
    if dia_signals > 0 or fn_analysis['has_dia']:
        scores['dia'] = min(1.0, 0.5 + dia_signals * 0.15)

    # AP-MS / bait detection
    apms_signals = _count_patterns(full_text, [
        r'affinity purif', r'immunoprecipitat', r'\bap[\s-]*ms\b',
        r'pull[\s-]*down', r'co[\s-]*ip\b', r'bait'
    ])
    if apms_signals > 0 or fn_analysis['has_bait']:
        scores['ap_ms'] = min(1.0, 0.4 + apms_signals * 0.15)

    # Clinical cohort detection
    clinical_signals = _count_patterns(full_text, [
        r'patient', r'cohort', r'clinical', r'biopsy', r'tumor',
        r'tissue[\s]*sample', r'serum', r'plasma', r'csf',
        r'urine', r'blood', r'biofluid'
    ])
    if clinical_signals >= 2:
        scores['clinical_cohort'] = min(1.0, 0.3 + clinical_signals * 0.1)

    # Cell line detection
    cell_signals = _count_patterns(full_text, [
        r'hela\b', r'hek[\s-]*293', r'a549\b', r'mcf[\s-]*7',
        r'jurkat', r'u2os', r'cell[\s]*line', r'cell[\s]*culture',
        r'cultured cells'
    ])
    if cell_signals > 0:
        scores['cell_line'] = min(1.0, 0.4 + cell_signals * 0.15)

    # Temperature series (CETSA/TPP)
    temp_signals = _count_patterns(full_text, [
        r'cetsa', r'thermal proteome profiling', r'\btpp\b',
        r'thermal stability', r'melting curve', r'temperature[\s]*gradient'
    ])
    if temp_signals > 0 or fn_analysis['has_temperature']:
        scores['temperature_series'] = min(1.0, 0.5 + temp_signals * 0.2)

    # Time series
    time_signals = _count_patterns(full_text, [
        r'time[\s]*course', r'time[\s]*point', r'kinetic',
        r'temporal', r'\d+\s*h\b.*\d+\s*h\b'
    ])
    if time_signals > 0 or fn_analysis['has_timepoint']:
        scores['time_series'] = min(1.0, 0.4 + time_signals * 0.15)

    # Treatment/control
    treat_signals = _count_patterns(full_text, [
        r'treated', r'untreated', r'control.*treated',
        r'vehicle', r'dmso.*control', r'drug[\s]*treatment',
        r'stimulat', r'inhibit'
    ])
    if treat_signals > 0 or fn_analysis['has_treatment']:
        scores['treatment_control'] = min(1.0, 0.3 + treat_signals * 0.1)

    # Tissue
    tissue_signals = _count_patterns(full_text, [
        r'tissue', r'organ', r'brain', r'liver', r'kidney', r'heart',
        r'lung', r'muscle', r'skin'
    ])
    if tissue_signals >= 2:
        scores['tissue'] = min(1.0, 0.3 + tissue_signals * 0.1)

    # Dose response
    dose_signals = _count_patterns(full_text, [
        r'dose[\s-]*response', r'concentration[\s-]*dependent',
        r'ic50', r'ec50', r'dose[\s-]*dependent'
    ])
    if dose_signals > 0:
        scores['dose_response'] = min(1.0, 0.5 + dose_signals * 0.2)

    # Normalize: at least one family
    if not scores:
        scores['label_free_simple'] = 0.5

    return scores


def get_primary_family(scores: Dict[str, float]) -> str:
    """Return the highest-scoring family."""
    if not scores:
        return 'label_free_simple'
    return max(scores, key=scores.get)


def _count_patterns(text: str, patterns: List[str]) -> int:
    """Count how many distinct patterns match in text."""
    count = 0
    for pat in patterns:
        if re.search(pat, text):
            count += 1
    return count


def is_multiplexed(scores: Dict[str, float]) -> bool:
    """Check if experiment uses multiplexing (TMT/iTRAQ/SILAC)."""
    return any(scores.get(f, 0) > 0.3 for f in
               ['tmt_multiplexed', 'itraq_multiplexed', 'silac'])


def is_fractionated(scores: Dict[str, float]) -> bool:
    """Check if experiment is fractionated."""
    return scores.get('label_free_fractionated', 0) > 0.3
