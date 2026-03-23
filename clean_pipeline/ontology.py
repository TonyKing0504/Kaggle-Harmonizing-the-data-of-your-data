"""Ontology-driven normalization: map extracted values to canonical SDRF strings."""
import re
from typing import Optional, Dict, List, Tuple


# Canonical instrument mappings (NT=...;AC=... format)
INSTRUMENT_CANONICAL = {
    'q exactive hf-x': 'NT=Q Exactive HF-X;AC=MS:1002877',
    'q exactive hf': 'NT=Q Exactive HF;AC=MS:1002523',
    'q exactive plus': 'NT=Q Exactive Plus;AC=MS:1002634',
    'q exactive': 'NT=Q Exactive;AC=MS:1001911',
    'orbitrap fusion lumos': 'NT=Orbitrap Fusion Lumos;AC=MS:1002732',
    'orbitrap fusion': 'NT=Orbitrap Fusion;AC=MS:1002416',
    'orbitrap eclipse': 'NT=Orbitrap Eclipse;AC=MS:1003029',
    'exploris 480': 'NT=Orbitrap Exploris 480;AC=MS:1003028',
    'exploris 240': 'NT=Orbitrap Exploris 240;AC=MS:1003096',
    'orbitrap astral': 'NT=Orbitrap Astral;AC=MS:1003378',
    'ltq orbitrap velos': 'AC=MS:1001742;NT=LTQ Orbitrap Velos',
    'orbitrap velos': 'AC=MS:1001742;NT=LTQ Orbitrap Velos',
    'ltq orbitrap elite': 'NT=LTQ Orbitrap Elite;AC=MS:1001910',
    'orbitrap elite': 'NT=LTQ Orbitrap Elite;AC=MS:1001910',
    'ltq orbitrap xl': 'NT=LTQ Orbitrap XL;AC=MS:1000556',
    'ltq orbitrap': 'NT=LTQ Orbitrap;AC=MS:1000449',
    'orbitrap ascend': 'NT=Orbitrap Ascend;AC=MS:1003356',
    'timstof pro 2': 'NT=timsTOF Pro 2;AC=MS:1003230',
    'timstof pro': 'NT=timsTOF Pro;AC=MS:1003005',
    'timstof ht': 'NT=timsTOF HT;AC=MS:1003303',
    'timstof scp': 'NT=timsTOF SCP;AC=MS:1003229',
    'timstof': 'NT=timsTOF;AC=MS:1003005',
    'triple tof 5600': 'NT=Triple TOF 5600;AC=MS:1000932',
    'triple tof 6600': 'NT=Triple TOF 6600;AC=MS:1002533',
    'zenotof 7600': 'NT=SCIEX ZenoTOF 7600;AC=MS:1003387',
    'lumos': 'NT=Orbitrap Fusion Lumos;AC=MS:1002732',
    'astral': 'NT=Orbitrap Astral;AC=MS:1003378',
    'eclipse': 'NT=Orbitrap Eclipse;AC=MS:1003029',
}

# Canonical organism names
ORGANISM_CANONICAL = {
    'human': 'Homo sapiens',
    'homo sapiens': 'Homo sapiens',
    'mouse': 'Mus musculus',
    'mus musculus': 'Mus musculus',
    'rat': 'Rattus norvegicus',
    'rattus norvegicus': 'Rattus norvegicus',
    'yeast': 'Saccharomyces cerevisiae',
    'saccharomyces cerevisiae': 'Saccharomyces cerevisiae',
    'e. coli': 'Escherichia coli',
    'escherichia coli': 'Escherichia coli',
    'drosophila': 'Drosophila melanogaster',
    'drosophila melanogaster': 'Drosophila melanogaster',
    'c. elegans': 'Caenorhabditis elegans',
    'zebrafish': 'Danio rerio',
    'danio rerio': 'Danio rerio',
    'pig': 'Sus scrofa',
    'sus scrofa': 'Sus scrofa',
    'bovine': 'Bos taurus',
    'bos taurus': 'Bos taurus',
    'arabidopsis': 'Arabidopsis thaliana',
    'rice': 'Oryza sativa',
    'oryza sativa': 'Oryza sativa',
    'plasmodium falciparum': 'plasmodium falciparum',
    'haloferax volcanii': 'Haloferax volcanii DS2',
}

# Canonical cleavage agent strings (most common training format)
CLEAVAGE_CANONICAL = {
    'trypsin': 'AC=MS:1001251;NT=Trypsin',
    'trypsin/p': 'AC=MS:1001313;NT=Trypsin/P',
    'lys-c': 'AC=MS:1001309;NT=Lys-C',
    'lysc': 'AC=MS:1001309;NT=Lys-C',
    'asp-n': 'NT=Asp-N;AC=MS:1001303',
    'glu-c': 'NT=Glu-C;AC=MS:1001917',
    'chymotrypsin': 'NT=chymotrypsin;AC=MS:1001306',
    'arg-c': 'NT=Arg-C;AC=MS:1001303',
    'v8-de': 'AC=MS:1001314;NT=V8-DE',
}

# Canonical fragmentation strings (most common training format)
FRAGMENTATION_CANONICAL = {
    'hcd': 'AC=MS:1000422;NT=HCD',
    'cid': 'AC=MS:1000133;NT=CID',
    'etd': 'AC=MS:1000598;NT=ETD',
    'ethcd': 'NT=EThcD;AC=MS:1002631',
    'uvpd': 'NT=UVPD;AC=MS:1003246',
}

# Also update fill_policy defaults


# Canonical label strings
LABEL_CANONICAL = {
    'label free': 'AC=MS:1002038;NT=label free sample',
    'label free sample': 'AC=MS:1002038;NT=label free sample',
    'tmt126': 'TMT126',
    'tmt127n': 'TMT127N',
    'tmt127c': 'TMT127C',
    'tmt128n': 'TMT128N',
    'tmt128c': 'TMT128C',
    'tmt129n': 'TMT129N',
    'tmt129c': 'TMT129C',
    'tmt130n': 'TMT130N',
    'tmt130c': 'TMT130C',
    'tmt131': 'TMT131',
    'tmt131n': 'TMT131N',
    'tmt131c': 'TMT131C',
    'tmt132n': 'TMT132N',
    'tmt132c': 'TMT132C',
    'tmt133n': 'TMT133N',
    'tmt133c': 'TMT133C',
    'tmt134n': 'TMT134N',
    'tmt134c': 'TMT134C',
    'tmt135n': 'TMT135N',
}


def normalize_instrument(raw: str) -> str:
    """Normalize instrument string to canonical format."""
    if 'NT=' in raw and 'AC=' in raw:
        return raw  # Already canonical
    key = raw.lower().strip()
    # Try exact match
    if key in INSTRUMENT_CANONICAL:
        return INSTRUMENT_CANONICAL[key]
    # Try substring match
    for k, v in INSTRUMENT_CANONICAL.items():
        if k in key:
            return v
    return raw


def normalize_organism(raw: str) -> str:
    """Normalize organism to canonical name."""
    key = raw.lower().strip()
    if key in ORGANISM_CANONICAL:
        return ORGANISM_CANONICAL[key]
    # fuzzy match
    for k, v in ORGANISM_CANONICAL.items():
        if k in key or key in k:
            return v
    return raw


def normalize_cleavage_agent(raw: str) -> str:
    """Normalize cleavage agent."""
    if 'NT=' in raw and 'AC=' in raw:
        return raw
    key = raw.lower().strip()
    if key in CLEAVAGE_CANONICAL:
        return CLEAVAGE_CANONICAL[key]
    for k, v in CLEAVAGE_CANONICAL.items():
        if k in key:
            return v
    return raw


def normalize_fragmentation(raw: str) -> str:
    """Normalize fragmentation method."""
    if 'NT=' in raw and 'AC=' in raw:
        return raw
    key = raw.lower().strip()
    if key in FRAGMENTATION_CANONICAL:
        return FRAGMENTATION_CANONICAL[key]
    for k, v in FRAGMENTATION_CANONICAL.items():
        if k in key:
            return v
    return raw


def normalize_label(raw: str) -> str:
    """Normalize label."""
    key = raw.lower().strip()
    if key in LABEL_CANONICAL:
        return LABEL_CANONICAL[key]
    for k, v in LABEL_CANONICAL.items():
        if k in key:
            return v
    return raw


def normalize_modification(raw: str) -> str:
    """Normalize modification string."""
    if 'NT=' in raw:
        return raw  # Already in NT/AC format
    return raw


def normalize_value(column: str, raw_value: str) -> str:
    """Normalize a value based on column type."""
    if not raw_value or raw_value.strip() == '':
        return 'Not Applicable'

    if 'Instrument' in column:
        return normalize_instrument(raw_value)
    elif 'Organism' in column and 'Part' not in column:
        return normalize_organism(raw_value)
    elif 'CleavageAgent' in column:
        return normalize_cleavage_agent(raw_value)
    elif 'FragmentationMethod' in column:
        return normalize_fragmentation(raw_value)
    elif 'Label' in column:
        return normalize_label(raw_value)
    elif 'Modification' in column:
        return normalize_modification(raw_value)

    return raw_value
