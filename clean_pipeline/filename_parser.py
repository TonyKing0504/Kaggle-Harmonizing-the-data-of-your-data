"""Parse raw data filenames for experimental metadata signals."""
import re
from typing import Dict, Optional, List


def parse_filename(filename: str) -> Dict[str, Optional[str]]:
    """Extract metadata signals from a raw data filename."""
    fn = filename.replace('.raw', '').replace('.mzML', '').replace('.wiff', '')
    fn_lower = fn.lower()
    result = {}

    # Biological replicate
    bio_rep = _extract_replicate(fn, 'bio')
    if bio_rep:
        result['biological_replicate'] = bio_rep

    # Technical replicate
    tech_rep = _extract_replicate(fn, 'tech')
    if tech_rep:
        result['technical_replicate'] = tech_rep

    # Generic replicate (BR/rep patterns)
    rep = _extract_generic_replicate(fn)
    if rep and 'biological_replicate' not in result:
        result['replicate'] = rep

    # Fraction
    frac = _extract_fraction(fn)
    if frac:
        result['fraction'] = frac

    # TMT/iTRAQ channel
    channel = _extract_channel(fn)
    if channel:
        result['channel'] = channel

    # Treatment / Control
    treatment = _extract_treatment(fn_lower)
    if treatment:
        result['treatment_signal'] = treatment

    # Bait (AP-MS)
    bait = _extract_bait(fn)
    if bait:
        result['bait'] = bait

    # Temperature
    temp = _extract_temperature(fn)
    if temp:
        result['temperature'] = temp

    # Time point
    time = _extract_timepoint(fn_lower)
    if time:
        result['timepoint'] = time

    # DIA/DDA
    acq = _extract_acquisition(fn_lower)
    if acq:
        result['acquisition'] = acq

    # Concentration/dose
    dose = _extract_dose(fn_lower)
    if dose:
        result['dose'] = dose

    return result


def _extract_replicate(fn: str, rep_type: str) -> Optional[str]:
    """Extract bio or tech replicate number."""
    patterns = {
        'bio': [r'BR(\d+)', r'bio[_-]?rep[_-]?(\d+)', r'biorep(\d+)'],
        'tech': [r'TR(\d+)', r'tech[_-]?rep[_-]?(\d+)', r'techrep(\d+)'],
    }
    for pat in patterns.get(rep_type, []):
        m = re.search(pat, fn, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _extract_generic_replicate(fn: str) -> Optional[str]:
    """Extract generic replicate number from rep/BR patterns."""
    patterns = [
        r'_rep(\d+)', r'_r(\d+)(?:[_.]|$)', r'BR(\d+)',
        r'[-_](\d+)$',  # trailing number often is replicate
    ]
    for pat in patterns:
        m = re.search(pat, fn, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _extract_fraction(fn: str) -> Optional[str]:
    """Extract fraction identifier."""
    patterns = [
        r'[Ff]rac(?:tion)?[_-]?(\d+)',
        r'[Ff](\d+)(?:[_.]|$)',
        r'_(\d+)(?:of\d+)',
        r'[-_]f(\d+)[-_.]',
    ]
    for pat in patterns:
        m = re.search(pat, fn)
        if m:
            return m.group(1)
    return None


def _extract_channel(fn: str) -> Optional[str]:
    """Extract TMT/iTRAQ channel."""
    # TMT specific channel names (TMT126, TMT127N, etc.) - NOT plex sizes
    m = re.search(r'(TMT1[23]\d[CN]?)\b', fn, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Skip plex identifiers like TMT16, TMT11, TMT10, TMT6
    # These indicate the plex size, not a specific channel
    m = re.search(r'TMT(\d+)', fn, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if num in (6, 10, 11, 16, 18):
            return None  # plex size, not channel

    # iTRAQ
    m = re.search(r'(iTRAQ\d+)', fn, re.IGNORECASE)
    if m:
        return m.group(1)

    # 126-134 channel numbers in specific position
    m = re.search(r'[_-](1[23]\d[CN])(?:[_.-]|$)', fn)
    if m:
        return 'TMT' + m.group(1)
    return None


def _extract_treatment(fn: str) -> Optional[str]:
    """Extract treatment/control signal."""
    if any(x in fn for x in ['ctrl', 'control', 'mock', 'dmso', 'vehicle', 'untreated']):
        return 'control'
    if any(x in fn for x in ['treated', 'drug', 'stim', 'activated']):
        return 'treated'
    return None


def _extract_bait(fn: str) -> Optional[str]:
    """Extract bait protein signal from AP-MS filenames."""
    m = re.search(r'(?:bait|pulldown|IP)[_-]?(\w+)', fn, re.IGNORECASE)
    if m:
        return m.group(1)
    # IgG control
    if re.search(r'\bIgG\b', fn, re.IGNORECASE):
        return 'IgG'
    return None


def _extract_temperature(fn: str) -> Optional[str]:
    """Extract temperature from filename."""
    m = re.search(r'(\d+)[_-]?[Cc](?:[_.-]|$)', fn)
    if m:
        return m.group(1) + 'C'
    m = re.search(r'(\d+)C(?:_|\b)', fn)
    if m:
        return m.group(1) + 'C'
    return None


def _extract_timepoint(fn: str) -> Optional[str]:
    """Extract timepoint from filename."""
    patterns = [
        r'(\d+)\s*(?:hr|hrs|hour|hours|h)(?:[_.-]|$)',
        r'(\d+)\s*(?:min|mins|minutes|m)(?:[_.-]|$)',
        r'(\d+)\s*(?:sec|secs|seconds|s)(?:[_.-]|$)',
        r'(\d+)sec',
        r'(\d+)min',
    ]
    for pat in patterns:
        m = re.search(pat, fn, re.IGNORECASE)
        if m:
            return m.group(0).strip('_.-')
    return None


def _extract_acquisition(fn: str) -> Optional[str]:
    """Extract DIA/DDA acquisition mode."""
    if 'dia' in fn:
        return 'DIA'
    if 'dda' in fn:
        return 'DDA'
    return None


def _extract_dose(fn: str) -> Optional[str]:
    """Extract dose/concentration from filename."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:um|nm|mm|ug|ng|mg|pm)', fn, re.IGNORECASE)
    if m:
        return m.group(0)
    return None


def parse_filename_group(filenames: List[str]) -> Dict[str, any]:
    """Analyze a group of filenames to infer experimental structure."""
    parsed = [parse_filename(fn) for fn in filenames]
    result = {
        'n_files': len(filenames),
        'has_fractions': any('fraction' in p for p in parsed),
        'has_channels': any('channel' in p for p in parsed),
        'has_replicates': any('replicate' in p or 'biological_replicate' in p for p in parsed),
        'has_treatment': any('treatment_signal' in p for p in parsed),
        'has_temperature': any('temperature' in p for p in parsed),
        'has_bait': any('bait' in p for p in parsed),
        'has_timepoint': any('timepoint' in p for p in parsed),
        'has_dia': any(p.get('acquisition') == 'DIA' for p in parsed),
        'unique_fractions': len(set(p.get('fraction', '') for p in parsed if 'fraction' in p)),
        'unique_channels': len(set(p.get('channel', '') for p in parsed if 'channel' in p)),
        'unique_replicates': len(set(
            p.get('replicate', p.get('biological_replicate', ''))
            for p in parsed if 'replicate' in p or 'biological_replicate' in p
        )),
        'parsed': parsed,
    }
    return result
