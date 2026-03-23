"""Extract metadata from publication text sections."""
import re
from typing import Dict, List, Optional, Tuple


def extract_organism(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract organism from publication text. Returns [(value, confidence, source)]."""
    candidates = []
    methods = text.get('METHODS', '') or ''
    abstract = text.get('ABSTRACT', '') or ''
    title = text.get('TITLE', '') or ''
    full = methods + ' ' + abstract + ' ' + title

    # Common organisms with canonical names
    organism_patterns = {
        'Homo sapiens': [r'\bhuman\b', r'homo\s*sapiens', r'\bhuman\s+(?:cell|tissue|blood|serum|plasma|brain|liver|kidney|sample)',
                         r'patient', r'\bhek\b', r'\bhela\b', r'a549', r'mcf-?7', r'jurkat', r'u2os',
                         r'k562\b', r'lncap', r'pc-?3', r'sh-sy5y', r'thp-?1'],
        'Mus musculus': [r'\bmouse\b', r'\bmurine\b', r'mus\s*musculus', r'\bmice\b',
                         r'c57bl', r'balb/c'],
        'Rattus norvegicus': [r'\brat\b', r'rattus\s*norvegicus', r'\brats\b'],
        'Saccharomyces cerevisiae': [r'saccharomyces\s*cerevisiae', r'\byeast\b',
                                      r's\.\s*cerevisiae'],
        'Escherichia coli': [r'escherichia\s*coli', r'e\.\s*coli', r'\be\.coli\b'],
        'Drosophila melanogaster': [r'drosophila', r'\bfly\b', r'fruit fly'],
        'Arabidopsis thaliana': [r'arabidopsis'],
        'Caenorhabditis elegans': [r'c\.\s*elegans', r'caenorhabditis'],
        'Danio rerio': [r'zebrafish', r'danio\s*rerio'],
        'Sus scrofa': [r'\bpig\b', r'\bporcine\b', r'sus\s*scrofa'],
        'Bos taurus': [r'\bbovine\b', r'\bcattle\b', r'bos\s*taurus', r'\bcow\b'],
        'Plasmodium falciparum': [r'plasmodium\s*falciparum', r'p\.\s*falciparum', r'malaria'],
        'Oryza sativa': [r'oryza\s*sativa', r'\brice\b'],
    }

    for org, patterns in organism_patterns.items():
        for pat in patterns:
            if re.search(pat, full, re.IGNORECASE):
                # Higher confidence for methods section
                conf = 0.9 if re.search(pat, methods, re.IGNORECASE) else 0.7
                candidates.append((org, conf, 'text'))
                break

    # Deduplicate keeping highest confidence
    seen = {}
    for val, conf, src in candidates:
        if val not in seen or conf > seen[val][1]:
            seen[val] = (val, conf, src)

    # Prioritize organism mentioned in title
    title_orgs = []
    for org, patterns in organism_patterns.items():
        for pat in patterns:
            if re.search(pat, title, re.IGNORECASE):
                if org in seen:
                    title_orgs.append(seen[org])
                break

    if title_orgs:
        return title_orgs[:1]

    # Otherwise return the highest-confidence single organism
    results = sorted(seen.values(), key=lambda x: -x[1])
    return results[:1] if results else []


def extract_instrument(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract MS instrument from text."""
    methods = text.get('METHODS', '') or ''
    full = methods + ' ' + (text.get('ABSTRACT', '') or '')
    candidates = []

    # Instrument patterns -> (canonical NT, AC)
    instruments = {
        ('Q Exactive HF-X', 'MS:1002877'): [r'q\s*exactive\s*hf[\s-]*x'],
        ('Q Exactive HF', 'MS:1002523'): [r'q\s*exactive\s*hf\b(?![\s-]*x)'],
        ('Q Exactive Plus', 'MS:1002634'): [r'q\s*exactive\s*plus'],
        ('Q Exactive', 'MS:1001911'): [r'q\s*exactive\b(?!\s*(?:hf|plus))'],
        ('Orbitrap Fusion Lumos', 'MS:1002732'): [r'(?:orbitrap\s*)?fusion\s*lumos', r'lumos'],
        ('Orbitrap Fusion', 'MS:1002416'): [r'orbitrap\s*fusion\b(?!\s*lumos)'],
        ('Orbitrap Eclipse', 'MS:1003029'): [r'orbitrap\s*eclipse'],
        ('Orbitrap Exploris 480', 'MS:1003028'): [r'exploris\s*480'],
        ('Orbitrap Exploris 240', 'MS:1003096'): [r'exploris\s*240'],
        ('Orbitrap Astral', 'MS:1003378'): [r'orbitrap\s*astral', r'\bastral\b'],
        ('LTQ Orbitrap Velos', 'MS:1001742'): [r'ltq\s*orbitrap\s*velos', r'orbitrap\s*velos'],
        ('LTQ Orbitrap Elite', 'MS:1001910'): [r'ltq\s*orbitrap\s*elite', r'orbitrap\s*elite'],
        ('LTQ Orbitrap XL', 'MS:1000556'): [r'ltq\s*orbitrap\s*xl', r'orbitrap\s*xl'],
        ('LTQ Orbitrap', 'MS:1000449'): [r'ltq\s*orbitrap\b(?!\s*(?:velos|elite|xl))'],
        ('Orbitrap Ascend', 'MS:1003356'): [r'orbitrap\s*ascend'],
        ('timsTOF Pro', 'MS:1003005'): [r'timstof\s*pro\b(?!\s*2)'],
        ('timsTOF Pro 2', 'MS:1003230'): [r'timstof\s*pro\s*2'],
        ('timsTOF HT', 'MS:1003303'): [r'timstof\s*ht'],
        ('timsTOF SCP', 'MS:1003229'): [r'timstof\s*scp'],
        ('timsTOF', 'MS:1003005'): [r'timstof\b(?!\s*(?:pro|ht|scp))'],
        ('Triple TOF 5600', 'MS:1000932'): [r'triple\s*tof\s*5600', r'5600\+?'],
        ('Triple TOF 6600', 'MS:1002533'): [r'triple\s*tof\s*6600', r'6600'],
        ('SCIEX ZenoTOF 7600', 'MS:1003387'): [r'zenotof\s*7600'],
        ('impact II', 'MS:1002667'): [r'impact\s*ii', r'maxi?\s*impact'],
        ('TSQ Quantiva', 'MS:1002672'): [r'tsq\s*quantiva'],
        ('TSQ Altis', 'MS:1003207'): [r'tsq\s*altis'],
        ('Orbitrap ID-X', 'MS:1003112'): [r'orbitrap\s*id[\s-]*x'],
    }

    for (nt, ac), patterns in instruments.items():
        for pat in patterns:
            if re.search(pat, full, re.IGNORECASE):
                canonical = f"NT={nt};AC={ac}"
                conf = 0.9 if re.search(pat, methods, re.IGNORECASE) else 0.7
                candidates.append((canonical, conf, 'text'))
                break

    return candidates


def extract_cleavage_agent(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract proteolytic enzyme from text."""
    methods = text.get('METHODS', '') or ''
    full = methods + ' ' + (text.get('ABSTRACT', '') or '')
    candidates = []

    enzymes = {
        'AC=MS:1001251;NT=Trypsin': [r'\btrypsin\b(?!/p)', r'tryptic\s*digest'],
        'AC=MS:1001313;NT=Trypsin/P': [r'trypsin/p\b'],
        'AC=MS:1001309;NT=Lys-C': [r'lys[\s-]*c\b', r'lysyl\s*endopeptidase'],
        'NT=Asp-N;AC=MS:1001303': [r'asp[\s-]*n\b'],
        'NT=Glu-C;AC=MS:1001917': [r'glu[\s-]*c\b', r'v8\s*protease'],
        'NT=chymotrypsin;AC=MS:1001306': [r'chymotrypsin'],
        'NT=Arg-C;AC=MS:1001303': [r'arg[\s-]*c\b'],
    }

    for canonical, patterns in enzymes.items():
        for pat in patterns:
            if re.search(pat, full, re.IGNORECASE):
                conf = 0.9 if re.search(pat, methods, re.IGNORECASE) else 0.7
                candidates.append((canonical, conf, 'text'))
                break

    return candidates


def extract_modifications(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract PTM modifications from text."""
    methods = text.get('METHODS', '') or ''
    full = methods + ' ' + (text.get('ABSTRACT', '') or '')
    candidates = []

    mods = {
        'NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed': [
            r'carbamidomethyl', r'iodoacetamide', r'iam\b', r'alkylat.*cysteine'
        ],
        'NT=Oxidation;MT=Variable;TA=M;AC=UNIMOD:35': [
            r'oxidation.*methionine', r'methionine\s*oxidation',
            r'variable.*oxidation', r'oxidation\s*\(m\)'
        ],
        'NT=Acetyl;AC=UNIMOD:1;PP=Protein N-term;MT=Variable': [
            r'n[\s-]*terminal\s*acetylation', r'acetyl.*protein\s*n[\s-]*term',
            r'acetylation.*n[\s-]*term'
        ],
        'NT=Phospho;AC=UNIMOD:21;TA=S,T,Y;MT=Variable': [
            r'phospho(?:rylation|proteom)', r'phospho\s*\(sty\)',
            r'phospho[\s-]*enrichment'
        ],
        'NT=Deamidated;AC=UNIMOD:7;TA=N,Q;MT=Variable': [
            r'deamidation', r'deamidated'
        ],
        'NT=TMT6plex;AC=UNIMOD:737;TA=K,N-term;MT=Fixed': [
            r'tmt[\s-]*6[\s-]*plex'
        ],
        'NT=TMTpro;AC=UNIMOD:2016;TA=K,N-term;MT=Fixed': [
            r'tmtpro', r'tmt[\s-]*pro', r'tmt[\s-]*16[\s-]*plex', r'tmt[\s-]*18[\s-]*plex'
        ],
    }

    for canonical, patterns in mods.items():
        for pat in patterns:
            if re.search(pat, full, re.IGNORECASE):
                conf = 0.85 if re.search(pat, methods, re.IGNORECASE) else 0.6
                candidates.append((canonical, conf, 'text'))
                break

    return candidates


def extract_fragmentation(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract fragmentation method."""
    methods = text.get('METHODS', '') or ''
    full = methods + ' ' + (text.get('ABSTRACT', '') or '')
    candidates = []

    frags = {
        'AC=MS:1000422;NT=HCD': [r'\bhcd\b', r'higher[\s-]*energy\s*c.*dissociation'],
        'AC=MS:1000133;NT=CID': [r'\bcid\b', r'collision[\s-]*induced\s*dissociation'],
        'AC=MS:1000598;NT=ETD': [r'\betd\b', r'electron\s*transfer\s*dissociation'],
        'NT=EThcD;AC=MS:1002631': [r'\bethcd\b', r'electron[\s-]*transfer.*hcd'],
    }

    for canonical, patterns in frags.items():
        for pat in patterns:
            if re.search(pat, full, re.IGNORECASE):
                conf = 0.85 if re.search(pat, methods, re.IGNORECASE) else 0.6
                candidates.append((canonical, conf, 'text'))
                break

    return candidates


def extract_cell_line(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract cell line from text."""
    methods = text.get('METHODS', '') or ''
    full = methods + ' ' + (text.get('ABSTRACT', '') or '') + ' ' + (text.get('RESULTS', '') or '')
    candidates = []

    cell_lines = [
        'HeLa', 'HEK293', 'HEK293T', 'HEK-293', 'A549', 'MCF-7', 'MCF7',
        'Jurkat', 'U2OS', 'K562', 'HCT116', 'HCT-116', 'MDA-MB-231',
        'SH-SY5Y', 'THP-1', 'THP1', 'LNCaP', 'PC-3', 'PC3', 'RAW264.7',
        'RAW 264.7', 'NIH-3T3', 'NIH3T3', 'CHO', 'Vero', 'COS-7', 'BHK-21',
        'HepG2', 'Caco-2', 'Caco2', 'PANC-1', 'SW480', 'SW620', 'HT-29',
        'DLD-1', 'LoVo', 'RKO', 'SK-BR-3', 'BT-474', 'T47D', 'ZR-75-1',
        'MOLT-4', 'BMDM', 'iPSC', 'hPSC', 'ESC', 'HFF',
    ]

    for cl in cell_lines:
        pattern = r'\b' + re.escape(cl) + r'\b'
        # Check primarily in methods section for cell lines
        if re.search(pattern, methods, re.IGNORECASE):
            candidates.append((cl, 0.9, 'text'))
        elif re.search(pattern, full, re.IGNORECASE):
            candidates.append((cl, 0.7, 'text'))

    # Return only the best cell line match - avoid returning all mentions
    if len(candidates) > 1:
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:1]

    return candidates


def extract_disease(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract disease context from text - requires clinical/specimen context."""
    methods = text.get('METHODS', '') or ''
    abstract = text.get('ABSTRACT', '') or ''
    title = text.get('TITLE', '') or ''
    full = methods + ' ' + abstract + ' ' + title

    candidates = []

    # Cancer-specific disease patterns (match training SDRF values)
    cancer_diseases = {
        'Colon adenocarcinoma': [r'colon\s*(?:adeno)?carcinom', r'colorectal\s*(?:adeno)?carcinom',
                                  r'colon\s*cancer', r'colorectal\s*cancer', r'rectal\s*cancer'],
        'breast adenocarcinoma': [r'breast\s*(?:adeno)?carcinom', r'breast\s*cancer'],
        'adenocarcinoma': [r'\badenocarcinom'],
        'lung cancer': [r'lung\s*cancer', r'nsclc', r'lung\s*adenocarcinoma'],
        'prostate cancer': [r'prostate\s*cancer'],
        'ovarian cancer': [r'ovarian\s*cancer'],
        'pancreatic cancer': [r'pancreatic\s*cancer'],
        'malignant melanoma': [r'\bmelanoma\b'],
        'glioblastoma': [r'glioblastoma', r'\bgbm\b'],
        'leukemia': [r'leukemia'],
        'lymphoma': [r'lymphoma'],
        'hepatocellular carcinoma': [r'hepatocellular\s*carcinoma', r'\bhcc\b'],
    }

    # Non-cancer diseases
    other_diseases = {
        "Alzheimer's disease": [r"alzheimer"],
        "Parkinson's disease": [r"parkinson"],
        'diabetes': [r'diabet'],
        'COVID-19': [r'covid[\s-]*19', r'sars[\s-]*cov[\s-]*2'],
        'malaria': [r'\bmalaria\b'],
        'osteoarthritis': [r'osteoarthritis'],
    }

    # Status values
    status_values = {
        'uninfected': [r'\buninfect'],
        'normal': [r'\bnormal\b.*\bcontrol\b', r'\bhealthy\b.*\bcontrol\b',
                   r'control\s*(?:group|subject|sample)'],
        'not available': [],
    }

    # Check cancer diseases first (higher priority in title)
    for disease, patterns in cancer_diseases.items():
        for pat in patterns:
            if re.search(pat, title, re.IGNORECASE):
                candidates.append((disease, 0.9, 'text'))
                break
            elif re.search(pat, abstract, re.IGNORECASE):
                candidates.append((disease, 0.8, 'text'))
                break
            elif re.search(pat, full, re.IGNORECASE):
                candidates.append((disease, 0.5, 'text'))
                break

    # Other diseases
    for disease, patterns in other_diseases.items():
        for pat in patterns:
            if re.search(pat, full, re.IGNORECASE):
                candidates.append((disease, 0.7, 'text'))
                break

    # Status values
    for status, patterns in status_values.items():
        for pat in patterns:
            if re.search(pat, full, re.IGNORECASE):
                candidates.append((status, 0.5, 'text'))
                break

    # Default if nothing found
    if not candidates:
        candidates.append(('not available', 0.4, 'text'))

    # Return only the best disease
    candidates.sort(key=lambda x: -x[1])
    return candidates[:1]


def extract_organism_part(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract organism part / tissue."""
    methods = text.get('METHODS', '') or ''
    abstract = text.get('ABSTRACT', '') or ''
    full = methods + ' ' + abstract

    parts = {
        'brain': [r'\bbrain\b'],
        'liver': [r'\bliver\b', r'\bhepatic\b'],
        'kidney': [r'\bkidney\b', r'\brenal\b'],
        'heart': [r'\bheart\b', r'\bcardiac\b'],
        'lung': [r'\blung\b', r'\bpulmonary\b'],
        'blood': [r'\bblood\b', r'\bplasma\b', r'\bserum\b'],
        'colon': [r'\bcolon\b'],
        'breast': [r'\bbreast\b', r'\bmammary\b'],
        'skin': [r'\bskin\b', r'\bdermal\b'],
        'muscle': [r'\bmuscle\b', r'\bskeletal\s*muscle\b'],
        'bone marrow': [r'bone\s*marrow'],
        'spleen': [r'\bspleen\b'],
        'pancreas': [r'\bpancreas\b', r'\bpancreatic\b'],
        'prostate': [r'\bprostate\b'],
        'ovary': [r'\bovary\b', r'\bovarian\b'],
        'uterus': [r'\buterus\b', r'\buterine\b'],
        'thyroid': [r'\bthyroid\b'],
        'lymph node': [r'lymph\s*node'],
        'cerebrospinal fluid': [r'cerebrospinal\s*fluid', r'\bcsf\b'],
        'saliva': [r'\bsaliva\b'],
        'urine': [r'\burine\b'],
        'milk': [r'\bmilk\b'],
        'cervix': [r'\bcervix\b', r'\bcervical\b'],
        'retina': [r'\bretina\b'],
        'testis': [r'\btestis\b', r'\btesticul'],
        'adipose tissue': [r'adipose'],
        'synovial fluid': [r'synovial'],
        'whole organism': [r'whole\s*(?:organism|cell|body)'],
    }

    candidates = []
    for part, patterns in parts.items():
        for pat in patterns:
            if re.search(pat, full, re.IGNORECASE):
                conf = 0.75 if re.search(pat, methods, re.IGNORECASE) else 0.55
                candidates.append((part, conf, 'text'))
                break

    # Return only the best organism part - avoid picking up tangential mentions
    if len(candidates) > 1:
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:1]

    return candidates


def extract_label(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract labeling strategy."""
    methods = text.get('METHODS', '') or ''
    abstract = text.get('ABSTRACT', '') or ''
    full = methods + ' ' + abstract

    candidates = []

    # TMT variants
    if re.search(r'tmt[\s-]*pro|tmt\s*16[\s-]*plex|tmt\s*18[\s-]*plex', full, re.IGNORECASE):
        candidates.append(('TMTpro', 0.9, 'text'))
    elif re.search(r'tmt[\s-]*11[\s-]*plex', full, re.IGNORECASE):
        candidates.append(('TMT11plex', 0.9, 'text'))
    elif re.search(r'tmt[\s-]*10[\s-]*plex', full, re.IGNORECASE):
        candidates.append(('TMT10plex', 0.9, 'text'))
    elif re.search(r'tmt[\s-]*6[\s-]*plex', full, re.IGNORECASE):
        candidates.append(('TMT6plex', 0.9, 'text'))
    elif re.search(r'\btmt\b', full, re.IGNORECASE):
        candidates.append(('TMT', 0.7, 'text'))

    if re.search(r'\bitraq\b', full, re.IGNORECASE):
        candidates.append(('iTRAQ', 0.8, 'text'))

    if re.search(r'\bsilac\b', full, re.IGNORECASE):
        candidates.append(('SILAC', 0.8, 'text'))

    if re.search(r'label[\s-]*free', full, re.IGNORECASE):
        candidates.append(('AC=MS:1002038;NT=label free sample', 0.8, 'text'))

    return candidates


def extract_material_type(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract material type (cell, tissue, biofluid, etc.)."""
    methods = text.get('METHODS', '') or ''
    abstract = text.get('ABSTRACT', '') or ''
    full = methods + ' ' + abstract

    candidates = []

    # Check for specific material types with precise patterns
    if re.search(r'cell\s*line|cultured\s*cell', full, re.IGNORECASE):
        candidates.append(('cell line', 0.85, 'text'))
    if re.search(r'\blysate\b', full, re.IGNORECASE):
        candidates.append(('lysate', 0.8, 'text'))
    if re.search(r'\bsynthetic\b.*\bpeptide\b|\bpeptide\s*standard', full, re.IGNORECASE):
        candidates.append(('synthetic', 0.8, 'text'))
    if re.search(r'\bbacter', full, re.IGNORECASE) and not re.search(r'\bhuman\b|\bhela\b', full, re.IGNORECASE):
        candidates.append(('bacterial strain', 0.7, 'text'))

    # Tissue vs organism part: "organism part" is common in training for
    # studies using tissue/organ samples
    if re.search(r'tumor\s*tissue|tissue\s*sample|biopsy|resect|surgical', full, re.IGNORECASE):
        candidates.append(('tissue', 0.75, 'text'))
    elif re.search(r'\btissue\b', full, re.IGNORECASE):
        candidates.append(('organism part', 0.7, 'text'))
    if re.search(r'plasma|serum|blood|csf|urine|saliva|biofluid|milk', full, re.IGNORECASE):
        candidates.append(('organism part', 0.7, 'text'))

    # If cell line was detected, "cell" is material type
    if not candidates:
        if re.search(r'cell\s*culture', full, re.IGNORECASE):
            candidates.append(('cell', 0.6, 'text'))

    # Default for studies with identifiable biological context
    if not candidates and re.search(r'proteom|protein|mass\s*spectro', full, re.IGNORECASE):
        # For human tissue studies, "organism part" is common
        if re.search(r'\btissue\b|\borgan\b|\bbrain\b|\bliver\b|\bkidney\b|\bheart\b|\blung\b|\bcolon\b', full, re.IGNORECASE):
            candidates.append(('organism part', 0.5, 'text'))

    # Return highest confidence
    candidates.sort(key=lambda x: -x[1])
    return candidates[:1] if candidates else []


def extract_enrichment(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract enrichment method."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    enrichments = {
        'NT=Phosphopeptide enrichment;AC=PRIDE:0000260': [
            r'(?:phospho|TiO2|IMAC|Fe-NTA)[\s-]*enrichment',
            r'(?:TiO2|titanium dioxide)\b',
        ],
        'NT=Glycopeptide enrichment;AC=PRIDE:0000261': [
            r'glyco[\s-]*(?:peptide|protein)[\s-]*enrichment',
            r'lectin\s*enrichment',
        ],
        'NT=Ubiquitin enrichment;AC=PRIDE:0000590': [
            r'(?:ubiquitin|diGly|K-GG)[\s-]*enrichment',
        ],
        'NT=Acetylation enrichment;AC=PRIDE:0000591': [
            r'acetyl(?:ation)?[\s-]*enrichment',
            r'anti[\s-]*acetyl',
        ],
    }

    for canonical, patterns in enrichments.items():
        for pat in patterns:
            if re.search(pat, methods, re.IGNORECASE):
                candidates.append((canonical, 0.85, 'text'))
                break

    return candidates


def extract_separation(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract separation/chromatography method."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    seps = {
        'NT=reverse phase;AC=sep:00185': [r'reverse[\s-]*phase', r'\brp[\s-]*lc\b', r'c18\b', r'rplc'],
        'NT=high-pH reversed-phase;AC=sep:00185': [r'high[\s-]*ph\b.*reverse', r'basic[\s-]*ph.*reverse',
                                                     r'high[\s-]*ph\s*fractionat'],
        'NT=SCX;AC=sep:00184': [r'\bscx\b', r'strong\s*cation\s*exchange'],
        'NT=SAX;AC=sep:00185': [r'\bsax\b', r'strong\s*anion\s*exchange'],
        'NT=SEC;AC=sep:00185': [r'\bsec\b', r'size[\s-]*exclusion'],
        'NT=HILIC;AC=sep:00185': [r'\bhilic\b', r'hydrophilic\s*interaction'],
    }

    for canonical, patterns in seps.items():
        for pat in patterns:
            if re.search(pat, methods, re.IGNORECASE):
                candidates.append((canonical, 0.8, 'text'))
                break

    return candidates


def extract_acquisition_method(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract acquisition method (DDA/DIA)."""
    methods = text.get('METHODS', '') or ''
    full = methods + ' ' + (text.get('ABSTRACT', '') or '')
    candidates = []

    if re.search(r'data[\s-]*independent|dia\b|swath|dia[\s-]*nn', full, re.IGNORECASE):
        candidates.append(('data-independent acquisition', 0.85, 'text'))
    if re.search(r'data[\s-]*dependent|dda\b|top[\s-]*\d+|topn', full, re.IGNORECASE):
        candidates.append(('data-dependent acquisition', 0.85, 'text'))

    return candidates


def extract_alkylation_reagent(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract alkylation reagent."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    if re.search(r'iodoacetamide|IAA\b', methods, re.IGNORECASE):
        candidates.append(('NT=Iodoacetamide;AC=CHEBI:53024', 0.85, 'text'))
    elif re.search(r'chloroacetamide|CAA\b', methods, re.IGNORECASE):
        candidates.append(('NT=Chloroacetamide;AC=CHEBI:27869', 0.85, 'text'))

    return candidates


def extract_reduction_reagent(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract reduction reagent."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    if re.search(r'dithiothreitol|DTT\b', methods, re.IGNORECASE):
        candidates.append(('NT=Dithiothreitol;AC=CHEBI:18320', 0.85, 'text'))
    elif re.search(r'TCEP\b|tris\(2-carboxyethyl\)phosphine', methods, re.IGNORECASE):
        candidates.append(('NT=TCEP;AC=CHEBI:78510', 0.85, 'text'))

    return candidates


def extract_fractionation_method(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract fractionation method."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    fracs = {
        'NT=high-pH reversed-phase chromatography;AC=sep:00185': [
            r'high[\s-]*ph\b.*fractionat', r'basic[\s-]*ph.*fractionat',
            r'high[\s-]*ph.*reverse'
        ],
        'NT=SDS-PAGE;AC=sep:00173': [r'sds[\s-]*page\b.*fractionat', r'gel[\s-]*based\s*fractionat'],
        'NT=SCX chromatography;AC=sep:00184': [r'scx\b.*fractionat'],
        'NT=SAX chromatography;AC=sep:00185': [r'sax\b.*fractionat'],
        'NT=Off-gel fractionation;AC=sep:00185': [r'off[\s-]*gel'],
    }

    for canonical, patterns in fracs.items():
        for pat in patterns:
            if re.search(pat, methods, re.IGNORECASE):
                candidates.append((canonical, 0.8, 'text'))
                break

    return candidates


def extract_ms2_analyzer(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract MS2 mass analyzer."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    analyzers = {
        'NT=orbitrap;AC=MS:1000484': [r'orbitrap\b.*ms2', r'ms2.*orbitrap', r'orbitrap\b.*fragment'],
        'NT=ion trap;AC=MS:1000264': [r'ion\s*trap\b.*ms2', r'ms2.*ion\s*trap'],
        'NT=TOF;AC=MS:1000084': [r'tof\b.*ms2', r'ms2.*tof'],
    }

    for canonical, patterns in analyzers.items():
        for pat in patterns:
            if re.search(pat, methods, re.IGNORECASE):
                candidates.append((canonical, 0.7, 'text'))
                break

    return candidates


def _normalize_tolerance_unit(unit: str) -> str:
    """Normalize tolerance unit to training-canonical format."""
    u = unit.lower()
    if u in ('da', 'dalton', 'daltons'):
        return 'Da'
    if u == 'ppm':
        return 'ppm'
    if u == 'amu':
        return 'Da'
    if u == 'mmu':
        return 'mmu'
    return u


def extract_tolerances(text: Dict[str, str]) -> Dict[str, List[Tuple[str, float, str]]]:
    """Extract precursor and fragment mass tolerances."""
    methods = text.get('METHODS', '') or ''
    result = {'precursor': [], 'fragment': []}

    # Precursor tolerance
    m = re.search(r'(?:precursor|ms1|parent).*?(\d+(?:\.\d+)?)\s*(ppm|da|dalton|mmu)', methods, re.IGNORECASE)
    if m:
        val = m.group(1) + ' ' + _normalize_tolerance_unit(m.group(2))
        result['precursor'].append((val, 0.8, 'text'))

    # Fragment tolerance
    m = re.search(r'(?:fragment|ms2|product).*?(\d+(?:\.\d+)?)\s*(ppm|da|dalton|amu|mmu)', methods, re.IGNORECASE)
    if m:
        val = m.group(1) + ' ' + _normalize_tolerance_unit(m.group(2))
        result['fragment'].append((val, 0.8, 'text'))

    return result


def extract_missed_cleavages(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract number of missed cleavages."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    m = re.search(r'(\d)\s*missed\s*cleavage', methods, re.IGNORECASE)
    if m:
        candidates.append((m.group(1), 0.85, 'text'))

    return candidates


def extract_gradient_time(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract LC gradient time."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    patterns = [
        r'(\d+)[\s-]*min(?:ute)?[\s]*gradient',
        r'gradient[\s]*(?:of\s*)?(\d+)[\s-]*min',
        r'(\d+)[\s-]*min(?:ute)?[\s]*(?:lc|liquid\s*chromatography)',
        r'linear\s*gradient.*?(\d+)\s*min',
    ]
    for pat in patterns:
        m = re.search(pat, methods, re.IGNORECASE)
        if m:
            candidates.append((m.group(1) + ' min', 0.75, 'text'))
            break

    return candidates


def extract_collision_energy(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract collision energy."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    patterns = [
        r'(?:normalized\s*)?collision\s*energy\s*(?:of\s*)?(\d+)[\s]*%?',
        r'(?:NCE|nce|HCD)\s*(?:of\s*)?(\d+)[\s]*%?',
    ]
    for pat in patterns:
        m = re.search(pat, methods, re.IGNORECASE)
        if m:
            candidates.append((m.group(1), 0.75, 'text'))
            break

    return candidates


def extract_number_of_fractions(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract number of fractions."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    patterns = [
        r'(\d+)\s*fractions?\b',
        r'fractionat.*?(\d+)\s*fractions?',
    ]
    for pat in patterns:
        m = re.search(pat, methods, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            if 2 <= n <= 200:
                candidates.append((str(n), 0.7, 'text'))
                break

    return candidates


def extract_specimen(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract specimen type."""
    methods = text.get('METHODS', '') or ''
    abstract = text.get('ABSTRACT', '') or ''
    full = methods + ' ' + abstract

    candidates = []
    specimens = {
        'biopsy': [r'\bbiopsy\b', r'\bbiopsies\b'],
        'autopsy': [r'\bautopsy\b'],
        'surgical resection': [r'surgical\s*resect'],
        'blood draw': [r'blood\s*draw', r'venipuncture'],
        'not applicable': [],
    }

    for spec, patterns in specimens.items():
        for pat in patterns:
            if re.search(pat, full, re.IGNORECASE):
                candidates.append((spec, 0.7, 'text'))
                break

    return candidates


def extract_sex(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract sex information."""
    methods = text.get('METHODS', '') or ''
    abstract = text.get('ABSTRACT', '') or ''
    full = methods + ' ' + abstract

    candidates = []
    if re.search(r'\bmale\b.*\bfemale\b|\bfemale\b.*\bmale\b', full, re.IGNORECASE):
        candidates.append(('male', 0.5, 'text'))
        candidates.append(('female', 0.5, 'text'))
    elif re.search(r'\bmale\b', full, re.IGNORECASE):
        candidates.append(('male', 0.6, 'text'))
    elif re.search(r'\bfemale\b', full, re.IGNORECASE):
        candidates.append(('female', 0.6, 'text'))

    return candidates


def extract_strain(text: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """Extract strain information for model organisms."""
    methods = text.get('METHODS', '') or ''
    candidates = []

    strains = {
        'C57BL/6': [r'c57bl/6', r'c57bl6'],
        'BALB/c': [r'balb/c', r'balbc'],
        'BY4741': [r'by4741'],
        'W303': [r'\bw303\b'],
        'BL21': [r'\bbl21\b'],
        'K-12': [r'\bk[\s-]*12\b'],
        'Col-0': [r'\bcol[\s-]*0\b'],
        'Wistar': [r'\bwistar\b'],
        'Sprague-Dawley': [r'sprague[\s-]*dawley'],
    }

    for strain, patterns in strains.items():
        for pat in patterns:
            if re.search(pat, methods, re.IGNORECASE):
                candidates.append((strain, 0.8, 'text'))
                break

    return candidates


def extract_all(pub_text: Dict[str, str]) -> Dict[str, List[Tuple[str, float, str]]]:
    """Run all extractors and return combined evidence dict."""
    tolerances = extract_tolerances(pub_text)
    return {
        'Characteristics[Organism]': extract_organism(pub_text),
        'Comment[Instrument]': extract_instrument(pub_text),
        'Characteristics[CleavageAgent]': extract_cleavage_agent(pub_text),
        'Characteristics[Modification]': extract_modifications(pub_text),
        'Comment[FragmentationMethod]': extract_fragmentation(pub_text),
        'Characteristics[CellLine]': extract_cell_line(pub_text),
        'Characteristics[Disease]': extract_disease(pub_text),
        'Characteristics[OrganismPart]': extract_organism_part(pub_text),
        'Characteristics[Label]': extract_label(pub_text),
        'Characteristics[MaterialType]': extract_material_type(pub_text),
        'Comment[EnrichmentMethod]': extract_enrichment(pub_text),
        'Comment[Separation]': extract_separation(pub_text),
        'Comment[AcquisitionMethod]': extract_acquisition_method(pub_text),
        'Characteristics[AlkylationReagent]': extract_alkylation_reagent(pub_text),
        'Characteristics[ReductionReagent]': extract_reduction_reagent(pub_text),
        'Comment[FractionationMethod]': extract_fractionation_method(pub_text),
        'Comment[MS2MassAnalyzer]': extract_ms2_analyzer(pub_text),
        'Comment[PrecursorMassTolerance]': tolerances['precursor'],
        'Comment[FragmentMassTolerance]': tolerances['fragment'],
        'Comment[NumberOfMissedCleavages]': extract_missed_cleavages(pub_text),
        'Comment[GradientTime]': extract_gradient_time(pub_text),
        'Comment[CollisionEnergy]': extract_collision_energy(pub_text),
        'Comment[NumberOfFractions]': extract_number_of_fractions(pub_text),
        'Characteristics[Specimen]': extract_specimen(pub_text),
        'Characteristics[Sex]': extract_sex(pub_text),
        'Characteristics[Strain]': extract_strain(pub_text),
    }
