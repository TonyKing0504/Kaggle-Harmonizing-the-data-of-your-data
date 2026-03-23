import pandas as pd, os, re, json, glob

def get_methods_text(pub):
    sections = pub.get("SECTIONS", {})
    methods = ""
    for key, val in sections.items():
        if any(m in key.lower() for m in ["method", "material", "experiment", "procedure", "protocol"]):
            methods += " " + val
    return methods

def get_full_text(pub):
    text = pub.get("ABSTRACT", "")
    for key, val in pub.get("SECTIONS", {}).items():
        text += " " + val
    return text

def extract_label(pub):
    methods = get_methods_text(pub).lower()
    text = get_full_text(pub).lower()
    primary = methods if methods.strip() else text
    if "tmt" in primary or "tandem mass tag" in primary:
        if "tmt10" in primary or "tmt-10" in primary or "10plex" in primary:
            return "TMT10plex", []
        if "tmt11" in primary or "11plex" in primary:
            return "TMT11plex", []
        if "tmt16" in primary or "tmtpro" in primary or "16plex" in primary:
            return "TMT16plex", []
        if "tmt6" in primary or "6plex" in primary:
            return "TMT6plex", []
        return "TMT6plex", []
    if "itraq" in primary:
        if "8plex" in primary or "8-plex" in primary:
            return "iTRAQ8plex", []
        return "iTRAQ4plex", []
    if "silac" in primary:
        return "SILAC", []
    return "label free", []

def extract_modifications(pub):
    methods = get_methods_text(pub).lower()
    text = get_full_text(pub).lower()
    mods = []
    if "carbamidomethyl" in methods or "iodoacetamide" in methods or " iaa " in methods:
        mods.append("Carbamidomethyl")
    elif "propionamide" in methods or "acrylamide" in methods:
        mods.append("propionamide")
    elif "carbamyl" in methods or "urea" in methods:
        mods.append("Carbamyl")
    if "oxidation" in methods:
        mods.append("Oxidation")
    if "acetyl" in methods and ("n-term" in methods or "protein" in methods):
        mods.append("Acetyl")
    if "phospho" in text or "phosphorylation" in text:
        mods.append("Phospho")
    if "deamid" in methods:
        mods.append("Deamidated")
    if "ubiquitin" in text or "diglycine" in text or "gly-gly" in text:
        mods.append("GlyGly")
    if re.search(r"gln\s*-?>?\s*pyro-?glu", methods):
        mods.append("Gln->pyro-Glu")
    if "ammonia-loss" in methods or "ammonia loss" in methods:
        mods.append("Ammonia-loss")
    if "pyro-glu" in methods and "Gln->pyro-Glu" not in mods:
        mods.append("Glu->pyro-Glu")
    label_type, _ = extract_label(pub)
    if "TMT" in label_type:
        mods.append(label_type)
    elif "iTRAQ" in label_type:
        mods.append(label_type)
    if "Oxidation" not in mods:
        mods.append("Oxidation")
    return mods

vm = pd.read_csv('validation_metrics.csv')
mod_f0 = vm[(vm['AnnotationType'].str.contains('Modification')) & (vm['f1'] == 0)]
f0_pxds = sorted(mod_f0['pxd'].unique())

pub_dir = 'data/TrainingPubText'
pubs = {}
for f in glob.glob(os.path.join(pub_dir, '*.json')):
    with open(f) as fp:
        pub = json.load(fp)
    pxd = pub.get('PXD', os.path.basename(f).replace('.json', ''))
    pubs[pxd] = pub

for pxd in f0_pxds:
    if pxd not in pubs:
        continue
    pub = pubs[pxd]
    mods = extract_modifications(pub)

    sdrf_file = f'data/TrainingSDRFs/{pxd}_cleaned.sdrf.tsv'
    if not os.path.exists(sdrf_file):
        print(f'{pxd}: pred={mods[:3]} | NO_SDRF')
        continue
    sdrf = pd.read_csv(sdrf_file, sep='\t')
    mod_cols = sorted(
        [c for c in sdrf.columns if c.startswith('Modification')],
        key=lambda x: (x.count('.'), x)
    )

    gold = {}
    for c in mod_cols[:3]:
        vals = sdrf[c].dropna().unique()
        real_vals = set()
        for v in vals:
            sv = str(v).strip()
            if sv.lower() not in ('not applicable', 'not available', 'nan'):
                m2 = re.search(r'NT=([^;]+)', sv)
                real_vals.add(m2.group(1) if m2 else sv)
        if real_vals:
            gold[c] = sorted(real_vals)

    f0_slots = mod_f0[mod_f0['pxd'] == pxd]['AnnotationType'].tolist()
    slots = ['s1' if '.1' not in s else 's2' for s in f0_slots]
    g1 = gold.get("Modification", ["-"])
    g2 = gold.get("Modification.1", ["-"])
    print(f'{pxd}: pred={mods[:3]} | g1={g1} g2={g2} | F1=0:{slots}')
