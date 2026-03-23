"""Row-level reconstruction: assign values to individual rows based on filename and evidence."""
import re
from typing import Dict, List, Optional, Tuple
from filename_parser import parse_filename, parse_filename_group
from family_classifier import is_multiplexed, is_fractionated, get_primary_family
from ontology import normalize_value


def reconstruct_rows(pxd: str, skeleton_rows: List[Dict], filenames: List[str],
                     evidence: Dict[str, List[Tuple[str, float, str]]],
                     family_scores: Dict[str, float],
                     pub_text: Dict[str, str]) -> List[Dict]:
    """
    Reconstruct row-level values for a PXD.

    Args:
        pxd: PXD identifier
        skeleton_rows: list of dicts with {ID, PXD, Raw Data File, Usage}
        filenames: raw data filenames in skeleton order
        evidence: extracted evidence per column
        family_scores: experiment family classification
        pub_text: publication text sections

    Returns:
        list of dicts with all column values filled
    """
    primary_family = get_primary_family(family_scores)
    n_rows = len(skeleton_rows)
    fn_group = parse_filename_group(filenames)
    parsed_fns = [parse_filename(fn) for fn in filenames]

    # Initialize result with skeleton
    results = []
    for row in skeleton_rows:
        r = dict(row)
        results.append(r)

    # Step 1: Apply paper-level evidence (same for all rows)
    _apply_paper_level(results, evidence)

    # Step 2: Apply row-level reconstruction based on family
    _apply_row_level(results, parsed_fns, filenames, evidence, family_scores, pub_text)

    # Step 3: Reconstruct fraction identifiers
    _reconstruct_fractions(results, parsed_fns, filenames, family_scores)

    # Step 4: Reconstruct biological replicates
    _reconstruct_replicates(results, parsed_fns, filenames, family_scores)

    # Step 5: Handle label / channel assignments
    _reconstruct_labels(results, parsed_fns, filenames, evidence, family_scores, pub_text)

    # Step 6: Reconstruct treatment/bait/temperature assignments
    _reconstruct_factor_values(results, parsed_fns, filenames, evidence, family_scores, pub_text)

    return results


def _apply_paper_level(results: List[Dict], evidence: Dict):
    """Apply paper-level extracted evidence to all rows."""
    from config import PAPER_LEVEL_COLUMNS

    for col in PAPER_LEVEL_COLUMNS:
        if col in evidence and evidence[col]:
            # Take highest-confidence candidate
            candidates = sorted(evidence[col], key=lambda x: -x[1])
            best_val = candidates[0][0]
            normalized = normalize_value(col, best_val)
            for r in results:
                if col not in r or r[col] in ('Text Span', 'Not Applicable', None, ''):
                    r[col] = normalized

    # Handle modification slots
    # Only fill slots 0 and 1 (Carbamidomethyl + Oxidation)
    # Slots 2-6 have <40% accuracy and hurt the score
    mod_candidates = evidence.get('Characteristics[Modification]', [])
    if mod_candidates:
        mod_candidates = sorted(mod_candidates, key=lambda x: -x[1])
        mod_cols = [
            'Characteristics[Modification]', 'Characteristics[Modification].1',
        ]
        for i, (val, conf, src) in enumerate(mod_candidates):
            if i < len(mod_cols):
                for r in results:
                    if mod_cols[i] not in r or r[mod_cols[i]] in ('Text Span', 'Not Applicable', None, ''):
                        r[mod_cols[i]] = normalize_value(mod_cols[i], val)


def _apply_row_level(results: List[Dict], parsed_fns: List[Dict],
                     filenames: List[str], evidence: Dict,
                     family_scores: Dict, pub_text: Dict):
    """Apply row-level values based on filename signals."""
    methods = (pub_text.get('METHODS', '') or '').lower()

    for i, (r, pfn) in enumerate(zip(results, parsed_fns)):
        fn = filenames[i]
        fn_lower = fn.lower()

        # Treatment from filename
        treat = pfn.get('treatment_signal')
        if treat:
            if treat == 'control':
                if 'dmso' in fn_lower:
                    r.setdefault('Characteristics[Treatment]', 'DMSO')
                elif 'mock' in fn_lower:
                    r.setdefault('Characteristics[Treatment]', 'mock')
                else:
                    r.setdefault('Characteristics[Treatment]', 'control')
            elif treat == 'treated':
                r.setdefault('Characteristics[Treatment]', 'treated')

        # Temperature from filename
        temp = pfn.get('temperature')
        if temp:
            r.setdefault('Characteristics[Temperature]', temp)

        # Bait from filename
        bait = pfn.get('bait')
        if bait:
            r.setdefault('Characteristics[Bait]', bait)
            r.setdefault('FactorValue[Bait]', bait)


def _reconstruct_fractions(results: List[Dict], parsed_fns: List[Dict],
                           filenames: List[str], family_scores: Dict):
    """Reconstruct fraction identifiers from filenames."""
    fractions = [pfn.get('fraction') for pfn in parsed_fns]
    has_fractions = any(f is not None for f in fractions)

    if has_fractions:
        for i, r in enumerate(results):
            frac = fractions[i]
            if frac:
                r['Comment[FractionIdentifier]'] = frac
                r['FactorValue[FractionIdentifier]'] = frac
    # Don't assign fraction IDs from trailing numbers unless we have strong
    # evidence of fractionation (to avoid confusing replicates with fractions)


def _reconstruct_replicates(results: List[Dict], parsed_fns: List[Dict],
                            filenames: List[str], family_scores: Dict):
    """Reconstruct biological replicate assignments.

    NOTE: BiologicalReplicate extraction has 13% accuracy on training data.
    Disabled for now as wrong values hurt more than help (scorer-neutral with Not Applicable).
    """
    # Disabled: 13% accuracy causes harm=11.3 (highest of any field)
    # bio_reps = [pfn.get('biological_replicate', pfn.get('replicate'))
    #             for pfn in parsed_fns]
    # has_bioreps = any(b is not None for b in bio_reps)
    # if has_bioreps:
    #     for i, r in enumerate(results):
    #         rep = bio_reps[i]
    #         if rep:
    #             r['Characteristics[BiologicalReplicate]'] = rep
    pass


def _reconstruct_labels(results: List[Dict], parsed_fns: List[Dict],
                        filenames: List[str], evidence: Dict,
                        family_scores: Dict, pub_text: Dict):
    """Reconstruct label/channel assignments."""
    if is_multiplexed(family_scores):
        # TMT/iTRAQ: channels may be encoded in filenames or in row structure
        channels = [pfn.get('channel') for pfn in parsed_fns]
        has_channels = any(c is not None for c in channels)

        if has_channels:
            for i, r in enumerate(results):
                ch = channels[i]
                if ch:
                    r['Characteristics[Label]'] = ch
        else:
            # If multiplexed but no channel in filename, check if rows_per_file > 1
            # indicating channel expansion in the skeleton
            label_evidence = evidence.get('Characteristics[Label]', [])
            if label_evidence:
                # Use the label type but don't assign specific channels without evidence
                best = sorted(label_evidence, key=lambda x: -x[1])[0]
                label_type = best[0]
                # Determine TMT channel count and assign
                _assign_tmt_channels(results, filenames, label_type, pub_text)
    else:
        # Label-free: all rows get label free sample
        label_evidence = evidence.get('Characteristics[Label]', [])
        if label_evidence:
            best = sorted(label_evidence, key=lambda x: -x[1])[0]
            for r in results:
                if 'Characteristics[Label]' not in r or r['Characteristics[Label]'] in ('Text Span', 'Not Applicable'):
                    r['Characteristics[Label]'] = normalize_value('Characteristics[Label]', best[0])
        else:
            for r in results:
                if 'Characteristics[Label]' not in r or r['Characteristics[Label]'] in ('Text Span', 'Not Applicable'):
                    r['Characteristics[Label]'] = 'AC=MS:1002038;NT=label free sample'


def _assign_tmt_channels(results: List[Dict], filenames: List[str],
                         label_type: str, pub_text: Dict):
    """Assign TMT channels to rows when multiplexed."""
    methods = (pub_text.get('METHODS', '') or '').lower()

    # Determine TMT plex
    tmt_channels_16 = [
        'TMT126', 'TMT127N', 'TMT127C', 'TMT128N', 'TMT128C',
        'TMT129N', 'TMT129C', 'TMT130N', 'TMT130C', 'TMT131N',
        'TMT131C', 'TMT132N', 'TMT132C', 'TMT133N', 'TMT133C', 'TMT134N',
    ]
    tmt_channels_18 = tmt_channels_16 + ['TMT134C', 'TMT135N']
    tmt_channels_11 = tmt_channels_16[:11]
    tmt_channels_10 = tmt_channels_16[:10]
    tmt_channels_6 = ['TMT126', 'TMT127', 'TMT128', 'TMT129', 'TMT130', 'TMT131']

    # Detect plex
    if '18' in label_type or '18plex' in methods:
        channels = tmt_channels_18
    elif '16' in label_type or '16plex' in methods or 'tmtpro' in methods:
        channels = tmt_channels_16
    elif '11' in label_type or '11plex' in methods:
        channels = tmt_channels_11
    elif '10' in label_type or '10plex' in methods:
        channels = tmt_channels_10
    elif '6' in label_type or '6plex' in methods:
        channels = tmt_channels_6
    else:
        channels = tmt_channels_16  # default to 16plex for TMTpro

    # Group rows by file
    file_groups = {}
    for i, fn in enumerate(filenames):
        if fn not in file_groups:
            file_groups[fn] = []
        file_groups[fn].append(i)

    # Assign channels within each file group
    for fn, indices in file_groups.items():
        n = len(indices)
        if n <= len(channels):
            for j, idx in enumerate(indices):
                results[idx]['Characteristics[Label]'] = channels[j]
        else:
            # More rows than channels - assign cyclically
            for j, idx in enumerate(indices):
                results[idx]['Characteristics[Label]'] = channels[j % len(channels)]


def _reconstruct_factor_values(results: List[Dict], parsed_fns: List[Dict],
                               filenames: List[str], evidence: Dict,
                               family_scores: Dict, pub_text: Dict):
    """Reconstruct FactorValue columns based on filename patterns and text."""
    methods = (pub_text.get('METHODS', '') or '').lower()

    for i, (r, pfn, fn) in enumerate(zip(results, parsed_fns, filenames)):
        fn_lower = fn.lower()

        # Temperature factor value
        temp = pfn.get('temperature')
        if temp:
            r.setdefault('FactorValue[Temperature]', temp)

        # Treatment factor
        treat = pfn.get('treatment_signal')
        if treat:
            r.setdefault('FactorValue[Treatment]', r.get('Characteristics[Treatment]', treat))

        # Compound from filename
        # Look for known drug names or compound patterns
        compound_patterns = [
            (r'DMSO', 'DMSO'),
            (r'(?:CDV|cidofovir)', 'cidofovir'),
        ]
        for pat, compound in compound_patterns:
            if re.search(pat, fn, re.IGNORECASE):
                r.setdefault('Characteristics[Compound]', compound)
                r.setdefault('FactorValue[Compound]', compound)
                break

        # Genetic modification from filename
        gm_patterns = [
            (r'[_-]KO[_-]', 'knockout'),
            (r'[_-]WT[_-]', 'wild type'),
            (r'delta[_-](\w+)', None),  # generic deletion
            (r'[_-]OE[_-]', 'overexpression'),
        ]
        for pat, label in gm_patterns:
            m = re.search(pat, fn, re.IGNORECASE)
            if m:
                val = label or m.group(0).strip('_-')
                r.setdefault('Characteristics[GeneticModification]', val)
                r.setdefault('FactorValue[GeneticModification]', val)
                break


def infer_number_counts(pub_text: Dict[str, str], n_rows: int,
                        filenames: List[str]) -> Dict[str, str]:
    """Infer NumberOfBiologicalReplicates, NumberOfSamples, NumberOfTechnicalReplicates."""
    methods = (pub_text.get('METHODS', '') or '')
    result = {}

    # Number of biological replicates
    m = re.search(r'(\d+)\s*(?:biological\s*)?replicates?', methods, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 20:
            result['Characteristics[NumberOfBiologicalReplicates]'] = str(n)

    # Number of technical replicates
    m = re.search(r'(\d+)\s*technical\s*replicates?', methods, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 10:
            result['Characteristics[NumberOfTechnicalReplicates]'] = str(n)

    # Number of samples
    unique_files = len(set(filenames))
    result.setdefault('Characteristics[NumberOfSamples]', str(unique_files))

    return result
