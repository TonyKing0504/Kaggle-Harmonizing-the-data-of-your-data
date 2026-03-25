#!/usr/bin/env python3
"""
Kaggle Competition: Harmonizing the Data of Your Data
Rule-based + text-mining pipeline for SDRF metadata extraction from proteomics publications.

Strategy: Precision-first. Only fill values when we have high confidence from text.
The scorer only evaluates common (PXD, column) pairs with non-trivial values.
Wrong values hurt more than missing values.

Usage:
    python pipeline.py                          # Generate submission.csv
    python pipeline.py --validate               # Run local validation on training data
    python pipeline.py --validate --fold 0      # Run single fold validation
"""

import os
import re
import json
import argparse
import difflib
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_SUB = DATA_DIR / "SampleSubmission.csv"
TRAIN_SDRF_DIR = DATA_DIR / "TrainingSDRFs"
TRAIN_PUB_DIR = DATA_DIR / "TrainingPubText"
TEST_PUB_DIR = DATA_DIR / "TestPubText"

# ─── Column Mapping: native SDRF → competition submission columns ─────────────
NATIVE_TO_SUBMISSION = {
    "comment[data file]": "Raw Data File",
    "comment[data file].1": "Raw Data File",
    "Organism": "Characteristics[Organism]",
    "Organism.1": "Characteristics[Organism]",
    "OrganismPart": "Characteristics[OrganismPart]",
    "OrganismPart.1": "Characteristics[OrganismPart]",
    "Disease": "Characteristics[Disease]",
    "Disease.1": "Characteristics[Disease]",
    "CellType": "Characteristics[CellType]",
    "CellType.1": "Characteristics[CellType]",
    "CellLine": "Characteristics[CellLine]",
    "CellLine.1": "Characteristics[CellLine]",
    "Label": "Characteristics[Label]",
    "Label.1": "Characteristics[Label]",
    "Label.2": "Characteristics[Label]",
    "Age": "Characteristics[Age]",
    "Age.1": "Characteristics[Age]",
    "Sex": "Characteristics[Sex]",
    "Sex.1": "Characteristics[Sex]",
    "AncestryCategory": "Characteristics[AncestryCategory]",
    "DevelopmentalStage": "Characteristics[DevelopmentalStage]",
    "MaterialType": "Characteristics[MaterialType]",
    "BiologicalReplicate": "Characteristics[BiologicalReplicate]",
    "BiologicalReplicate.1": "Characteristics[BiologicalReplicate]",
    # TechnicalReplicate in training SDRFs is a sequential per-run ID (1,2,3...),
    # NOT a count; Characteristics[NumberOfTechnicalReplicates] is a count field.
    # Mapping them would corrupt the gold values used in validation — omitted.
    "Modification": "Characteristics[Modification]",
    "Modification.1": "Characteristics[Modification].1",
    "Modification.2": "Characteristics[Modification].2",
    "Modification.3": "Characteristics[Modification].3",
    "Modification.4": "Characteristics[Modification].4",
    "Modification.5": "Characteristics[Modification].5",
    "Modification.6": "Characteristics[Modification].6",
    "Instrument": "Comment[Instrument]",
    "Instrument.1": "Comment[Instrument]",
    "CleavageAgent": "Characteristics[CleavageAgent]",
    "CleavageAgent.1": "Characteristics[CleavageAgent]",
    "FragmentationMethod": "Comment[FragmentationMethod]",
    "FragmentationMethod.1": "Comment[FragmentationMethod]",
    "PrecursorMassTolerance": "Comment[PrecursorMassTolerance]",
    "FragmentMassTolerance": "Comment[FragmentMassTolerance]",
    "FragmentMassTolerance.1": "Comment[FragmentMassTolerance]",
    "FractionIdentifier": "Comment[FractionIdentifier]",
    "CollisionEnergy": "Comment[CollisionEnergy]",
    "CollisionEnergy.1": "Comment[CollisionEnergy]",
    "FractionationMethod": "Comment[FractionationMethod]",
    "EnrichmentMethod": "Comment[EnrichmentMethod]",
    "EnrichmentMethod.1": "Comment[EnrichmentMethod]",
    "MS2MassAnalyzer": "Comment[MS2MassAnalyzer]",
    "MS2MassAnalyzer.1": "Comment[MS2MassAnalyzer]",
    "NumberOfMissedCleavages": "Comment[NumberOfMissedCleavages]",
    "Separation": "Comment[Separation]",
    "Compound": "Characteristics[Compound]",
    "Compound.1": "Characteristics[Compound]",
    "Depletion": "Characteristics[Depletion]",
    "GrowthRate": "Characteristics[GrowthRate]",
    "PooledSample": "Characteristics[PooledSample]",
    "ReductionReagent": "Characteristics[ReductionReagent]",
    "AlkylationReagent": "Characteristics[AlkylationReagent]",
    "SamplingTime": "Characteristics[SamplingTime]",
    "Specimen": "Characteristics[Specimen]",
    "SpikedCompound": "Characteristics[SpikedCompound]",
    "Staining": "Characteristics[Staining]",
    "Strain": "Characteristics[Strain]",
    "SyntheticPeptide": "Characteristics[SyntheticPeptide]",
    "Temperature": "Characteristics[Temperature]",
    "Temperature.1": "Characteristics[Temperature]",
    "Time": "Characteristics[Time]",
    "Time.1": "Characteristics[Time]",
    "Treatment": "Characteristics[Treatment]",
    "Treatment.1": "Characteristics[Treatment]",
    "Treatment.2": "Characteristics[Treatment]",
    "Treatment.3": "Characteristics[Treatment]",
    "BMI": "Characteristics[BMI]",
    "Bait": "Characteristics[Bait]",
    "CellPart": "Characteristics[CellPart]",
    "TumorSize": "Characteristics[TumorSize]",
    # AcquisitionMethod: training column name
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[Proteomics Data Acquisition Method]": "Comment[AcquisitionMethod]",
}

FACTOR_VALUE_MAP = {
    # phenotype is a proxy for disease condition in most training SDRFs
    "factor value[phenotype]": "FactorValue[Disease]",
    "factor value[phenotype].1": "FactorValue[Disease]",
    "factor value[ disease response]": "FactorValue[Disease]",
    "factor value[chemical entity]": "FactorValue[Compound]",
    "factor value[overproduction]": "FactorValue[GeneticModification]",
    "factor value[overproduction].1": "FactorValue[GeneticModification]",
    # "individual" is a patient/donor ID, NOT a disease → removed
    # "subtype" is ambiguous and NOT reliably disease → removed
    "factor value[induced by]": "FactorValue[Treatment]",
}


def extract_nt(value: str) -> str:
    """Extract NT= display term from ontology-annotated values."""
    if pd.isna(value):
        return ""
    v = str(value).strip()
    if "NT=" in v:
        parts = [r for r in v.split(";") if "NT=" in r]
        if parts:
            return parts[0].replace("NT=", "").strip()
    return v


# ─── Known value dictionaries from training ─────────────────────────────────

INSTRUMENTS = [
    "Q Exactive HF-X", "Q Exactive HF", "Q Exactive Plus", "Q Exactive",
    "LTQ Orbitrap Velos", "LTQ Orbitrap Elite", "LTQ Orbitrap XL ETD",
    "LTQ Orbitrap XL", "LTQ Orbitrap",
    "Orbitrap Fusion Lumos", "Orbitrap Fusion", "Orbitrap Exploris 480",
    "Orbitrap Exploris 240", "Orbitrap Eclipse", "Orbitrap Astral", "Orbitrap Ascend",
    "timsTOF Pro 2", "timsTOF Pro", "timsTOF HT", "timsTOF SCP", "timsTOF Ultra",
    "Triple TOF 6600", "Triple TOF 5600", "TripleTOF 6600", "TripleTOF 5600",
    "Zeno TOF 7600",
    "SYNAPT G2-Si", "TSQ Altis", "TSQ Quantiva",
    "maXis II", "Impact II",
    "Exploris 480", "Exploris 240",
    "Synapt XS", "SYNAPT XS", "Synapt G2-S",
    "Orbitrap Ascend",
]

LABELS_TMT16 = [
    "TMT126", "TMT127N", "TMT127C", "TMT128N", "TMT128C",
    "TMT129N", "TMT129C", "TMT130N", "TMT130C", "TMT131N", "TMT131C",
    "TMT132N", "TMT132C", "TMT133N", "TMT133C", "TMT134N",
]
LABELS_TMT11 = LABELS_TMT16[:11]
LABELS_TMT10 = LABELS_TMT16[:10]
LABELS_TMT6 = ["TMT126", "TMT127", "TMT128", "TMT129", "TMT130", "TMT131"]

# Gold-standard surface forms from training for key columns
# Built by inspecting actual gold values; used for normalisation.
ACQUISITION_METHOD_VOCAB = {
    "Data-Dependent Acquisition": [
        "data-dependent", "data dependent", r"\bdda\b", "data-dependent acquisition",
    ],
    "Data-Independent Acquisition": [
        "data-independent", "data independent", r"\bdia\b", "data-independent acquisition",
        "swath", "sonar", "dia-nn", "diann",
    ],
}

SEPARATION_VOCAB = "Reversed-phase chromatography"
SEPARATION_HPLC = "High-performance liquid chromatography"

FRACTIONATION_VOCAB = {
    "no fractionation": [
        r"\bno\s+fractionation\b", r"\bwithout\s+fractionation\b",
        r"\bnon-?fractionated\b",
    ],
    # BN-PAGE BEFORE SDS-PAGE (more specific; "in-gel" can occur in BN-PAGE too)
    "BN-PAGE": [
        r"\bblu?e[\s-]?native\b", r"\bbn[\s-]?page\b", r"\bcomplexome\b",
        r"\bcsbn[\s-]?ms\b", r"\bnative\s+gel\b", r"\bnative\s+page\b",
    ],
    # Use full name to match training gold ('High-pH reversed-phase chromatography')
    "High-pH reversed-phase chromatography": [
        r"\bhigh[\s-]pH\s+(?:reversed[\s-]phase|rp)", r"\bhigh[\s-]pH\s+rplc\b",
        r"\bhphrp\b", r"\bhigh[\s-]pH\s+rp\b",
        r"\boffline\s+(?:reversed[\s-]phase|rp)\s+fractionat",
        r"\bbasic\s+(?:rp|reversed[\s-]phase)\s+(?:lc|fractionat|hplc)",
        r"\boff-?line\s+high[\s-]pH\b",
    ],
    # Use full name to match training gold format
    "Sodium dodecyl sulfate polyacrylamide gel electrophoresis": [
        # Unambiguous gel fractionation indicators (not just sample prep in-gel digest)
        r"\bgel\s+(?:band|slice|frac)", r"\bgel-based\s+fractionation",
        r"\bsds[\s-]?page\s+(?:fraction|gel\s+band|separ)",
        r"\bexcised\s+(?:gel|band)", r"\bsliced?\s+(?:gel|band)",
        r"\bgel\s+slice", r"\bmultiple\s+(?:gel\s+)?band",
    ],
    "Strong cation-exchange chromatography (SCX)": [
        r"\bscx\b", r"\bstrong\s+cation[\s-]exchange\b",
    ],
    "Strong anion-exchange chromatography (SAX)": [
        r"\bsax\b", r"\bstrong\s+anion[\s-]exchange\b",
    ],
}

ENRICHMENT_VOCAB = {
    "no enrichment": [
        r"\bno\s+enrichment\b", r"\bwithout\s+enrichment\b",
        r"\bnon-?enriched\b",
    ],
    "enrichment of phosphorylated Protein": [
        r"\bphospho[\w-]+\s+enrichment\b", r"\btio2\b",
        r"\btitanium\s+dioxide\b", r"\bimac\b",
        r"\bphosphopeptide\s+enrichment\b", r"\bfe[\s-]?imac\b",
    ],
    "Extraction purification": [
        r"\bimmunoprecipitation\b", r"\bco[\s-]?ip\b", r"\bpulldown\b",
        r"\bpull[\s-]?down\b", r"\bap[\s-]?ms\b",
        r"\baffinity\s+purification\b",
    ],
}


# ─── Text extraction helpers ─────────────────────────────────────────────────

def load_pub_json(json_path: str) -> Dict[str, Any]:
    with open(json_path) as f:
        return json.load(f)


def get_full_text(pub: Dict) -> str:
    sections = ["TITLE", "ABSTRACT", "INTRO", "RESULTS", "DISCUSS", "FIG", "METHODS"]
    return "\n\n".join(pub.get(s, "") for s in sections if pub.get(s, ""))


def get_methods_text(pub: Dict) -> str:
    m = pub.get("METHODS", "")
    if len(m) > 100:
        return m
    return get_full_text(pub)


def extract_organism(pub: Dict) -> str:
    """Extract organism. High confidence - nearly always identifiable."""
    text = get_full_text(pub).lower()
    title = pub.get("TITLE", "").lower()
    abstract = pub.get("ABSTRACT", "").lower()

    organism_patterns = {
        "Homo sapiens": ["homo sapiens", "human", "patient", "hek293", "hela", "a549", "mcf7",
                         "jurkat", "hepg2", "u2os", "k562", "clinical", "huvec"],
        "Mus musculus": ["mus musculus", "mouse", "mice", "murine", "c57bl"],
        "Rattus norvegicus": ["rattus norvegicus", "rat ", "rats "],
        "Saccharomyces cerevisiae": ["saccharomyces cerevisiae", "yeast", "s. cerevisiae", "budding yeast"],
        "Escherichia coli": ["escherichia coli", "e. coli", "e.coli"],
        "Arabidopsis thaliana": ["arabidopsis thaliana", "arabidopsis", "a. thaliana"],
        "Drosophila melanogaster": ["drosophila melanogaster", "drosophila", "fruit fly", "d. melanogaster"],
        "Caenorhabditis elegans": ["caenorhabditis elegans", "c. elegans"],
        "Sus scrofa": ["sus scrofa", "pig ", "pigs ", "porcine", "swine"],
        "Bos taurus": ["bos taurus", "bovine", "cattle", "cow milk", "whey protein",
                       "milk serum", "milk protein", "raw milk", "skim milk"],
        "Danio rerio": ["danio rerio", "zebrafish"],
        "Plasmodium falciparum": ["plasmodium falciparum", "p. falciparum"],
        "Oryza sativa": ["oryza sativa", "rice "],
        "Gallus gallus": ["gallus gallus", "chicken"],
        "Xenopus laevis": ["xenopus"],
        "Plasmodium berghei": ["plasmodium berghei"],
        "Toxoplasma gondii": ["toxoplasma gondii"],
        "Trypanosoma brucei": ["trypanosoma brucei"],
    }

    methods = get_methods_text(pub).lower()

    scores = {}
    for org, patterns in organism_patterns.items():
        score = 0
        for p in patterns:
            if p in title:
                score += 10
            if p in abstract:
                score += 5
            if p in methods:
                score += 3
            if p in text:
                score += 1
        if score > 0:
            scores[org] = score

    if scores:
        return max(scores, key=scores.get)
    return "Homo sapiens"


def extract_organism_part(pub: Dict) -> str:
    """Extract organism part using gold-standard value forms from training."""
    abstract = pub.get("ABSTRACT", "").lower()
    title = pub.get("TITLE", "").lower()
    methods = get_methods_text(pub).lower()

    tissue_patterns = {
        "brain": [r"\bbrain\b", r"\bcerebr", r"\bcortex\b", r"\bhippocampus\b"],
        "liver": [r"\bliver\b", r"\bhepat"],
        "heart": [r"\bheart\b", r"\bcardiac\b", r"\bmyocardi"],
        "kidney": [r"\bkidney\b", r"\brenal\b"],
        "lung": [r"\blung\b", r"\bpulmonary\b"],
        "colon": [r"\bcolon\b"],
        "colorectal": [r"\bcolorectal\b"],
        "breast": [r"\bbreast\b", r"\bmammary\b"],
        "prostate": [r"\bprostate\b"],
        "pancreas": [r"\bpancrea"],
        "ovary": [r"\bovary\b", r"\bovarian\b"],
        "blood plasma": [r"\bblood\s+plasma\b", r"\bplasma\s+(?:sample|protein|proteom)"],
        "spleen": [r"\bspleen\b"],
        "skin": [r"\bskin\b", r"\bmelanocyte"],
        "stomach": [r"\bstomach\b", r"\bgastric\b"],
        "retina": [r"\bretina\b"],
        "lymph node": [r"\blymph\s+node"],
        "cerebrospinal fluid": [r"\bcerebrospinal\s+fluid\b", r"(?<![mg]-)\bcsf\b"],
        "cervix": [r"\bcervix\b", r"\bcervical\b"],
        "rectum": [r"\brectum\b", r"\brectal\b"],
        "tonsil": [r"\btonsil"],
        "Leaf": [r"\bleaf\b", r"\bleaves\b"],
        "bone marrow": [r"\bbone\s+marrow\b"],
        "muscle": [r"\bmuscle\b"],
        "culture supernatant": [r"\bculture\s+supernatant"],
        "B cells": [r"\bB[\s-]cell", r"\bB\s+lymphocyte", r"\bbursa"],
        "NK cells": [r"\bNK[\s-]cell", r"\bnatural\s+killer\s+cell"],
        "T cells": [r"\bT[\s-]cell\b", r"\bT\s+lymphocyte"],
        "human erythrocytes": [r"\berythrocyte"],
        "urine": [r"\burine\b"],
        "blood serum": [r"\bblood\s+serum\b", r"\bserum\s+(?:sample|protein|proteom|biomark)"],
        "saliva": [r"\bsaliva\b"],
        "testis": [r"\btestis\b", r"\btestes\b"],
        "thymus": [r"\bthymus\b"],
        "adrenal gland": [r"\badrenal\s+gland"],
        "esophagus": [r"\besophag"],
        "placenta": [r"\bplacenta\b"],
        "intestine": [r"\bintestine\b", r"\bintestinal\b"],
        "bladder": [r"\bbladder\b"],
        "thyroid": [r"\bthyroid\b"],
        "synovial": [r"\bsynovial\b", r"\bsynovium\b"],
        # milk/whey proteomics must appear in title or abstract as subject (not methods buffer)
        "milk": [r"\bmilk\s+(?:protein|proteom|sample|serum|whey)", r"\bwhey\s+(?:protein|proteom)", r"\bbovine\s+milk\b"],
    }

    results = pub.get("RESULTS", "").lower()
    intro = pub.get("INTRO", "").lower()

    scores = {}
    for tissue, patterns in tissue_patterns.items():
        score = 0
        for p in patterns:
            if re.search(p, title):
                score += 10
            if re.search(p, abstract):
                score += 5
            if re.search(p, results):
                score += 2
            if re.search(p, intro):
                score += 1
        if score > 0:
            scores[tissue] = score

    methods_only_terms = {
        # Only cerebrospinal fluid is safe for methods-only detection
        "cerebrospinal fluid": [r"\bcerebrospinal\s+fluid\b", r"(?<![mg]-)\bcsf\b"],
        # Removed human erythrocytes - gold often has "not available" for these studies
    }
    for tissue, patterns in methods_only_terms.items():
        if tissue not in scores:
            for p in patterns:
                if re.search(p, methods):
                    scores[tissue] = 3
                    break

    # Do NOT infer organism part from cell lines - gold standard typically has
    # "not available" for cell line studies, and guessing the organ creates false positives.
    # If the study primarily uses cell lines (mentioned in title/abstract), suppress organism part.
    cell_line_terms = [
        r"\bhek[\s-]?293\w*\b", r"\bhela\b", r"\ba549\b", r"\bmcf[\s-]?7\b",
        r"\bu2os\b", r"\bjurkat\b", r"\bk562\b", r"\bthp[\s-]?1\b", r"\bhuvec\b",
        r"\braw\s*264\b", r"\bc2c12\b", r"\bnih[\s-]?3t3\b", r"\bvero\b",
        r"\bmrc[\s-]?5\b", r"\bhepg2\b", r"\bipsc\b", r"\bcaco[\s-]?2\b",
        r"\bsh[\s-]?sy5y\b", r"\bhct[\s-]?116\b",
    ]
    title_abstract = title + " " + abstract
    has_cell_line_in_abstract = any(re.search(p, title_abstract, re.IGNORECASE)
                                    for p in cell_line_terms)
    if has_cell_line_in_abstract:
        # Cell line study — don't return organism part (gold typically has "not available")
        return ""

    # Use raw data file names as strong evidence (e.g. 'mousebrain-ko-*.raw' → brain)
    filename_organ = {
        "brain": r"brain|cerebr|cortex|hippocamp",
        "liver": r"liver|hepat",
        "heart": r"heart|cardiac",
        "lung": r"lung|pulmon",
        "kidney": r"kidney|renal",
        "blood plasma": r"plasma",
        "muscle": r"muscle",
    }
    raw_files = pub.get("Raw Data Files", [])
    if isinstance(raw_files, (list, tuple)):
        fname_text = " ".join(str(f).lower() for f in raw_files)
    else:
        fname_text = str(raw_files).lower()
    if fname_text.strip():
        for organ, pat in filename_organ.items():
            if re.search(pat, fname_text):
                scores[organ] = scores.get(organ, 0) + 8  # strong filename signal

    if scores:
        best = max(scores, key=lambda k: (scores[k], len(k)))
        # Require TITLE mention (score >= 10) for strongest confidence,
        # or abstract+other (score >= 8) for good confidence.
        # Previous threshold of 5 allowed too many false positives.
        if scores[best] >= 8:
            return best
    return ""


def extract_disease(pub: Dict) -> str:
    """Extract disease using training gold-standard value forms.
    Use SPECIFIC names that match gold forms to hit 80% string similarity threshold."""
    title = pub.get("TITLE", "").lower()
    abstract = pub.get("ABSTRACT", "").lower()

    # Use gold-standard names. More specific patterns first.
    disease_patterns = {
        # Specific cancer subtypes (must come before generic)
        "Rectum adenocarcinoma": [r"\brectum?\s+adenocarcinoma"],
        "rectal mucinous adenocarcinoma": [r"\brectal\s+mucinous"],
        "colorectal cancer": [r"\bcolorectal\s+(?:cancer|carcinoma|tumor)"],
        "Colon adenocarcinoma": [r"\bcolon\b.*?\b(?:tumor|cancer|carcinoma|adenocarcinoma)"],
        "squamous cell lung cancer": [r"\bsquamous\s+cell\s+lung"],
        "Lung adenocarcinoma": [r"\blung\s+adenocarcinoma"],
        "lung cancer": [r"\blung\s+cancer\b", r"\bnon[\s-]small\s+cell\s+lung"],
        "Metastatic melanoma": [r"\bmetastatic\s+melanoma\b"],
        "malignant melanoma": [r"\bmalignant\s+melanoma"],
        "amelanotic melanoma": [r"\bamelanotic\s+melanoma"],
        "melanoma": [r"\bmelanoma\b"],
        "Triple-negative breast cancer": [r"\btriple[\s-]negative\s+breast"],
        "breast cancer": [r"\bbreast\s+cancer\b"],
        "breast adenocarcinoma": [r"\bbreast\s+adenocarcinoma"],
        "Glioblastoma": [r"\bglioblastoma\b"],
        "high grade serous ovarian cancer": [r"\bhigh[\s-]grade\s+serous\s+ovarian"],
        "ovarian cancer": [r"\bovarian\s+cancer"],
        "Prostate carcinoma": [r"\bprostate\s+(?:carcinoma|cancer)"],
        "hepatocellular carcinoma": [r"\bhepatocellular\s+carcinoma"],
        "Hepatoblastoma": [r"\bhepatoblastoma"],
        "Osteosarcoma": [r"\bosteosarcoma"],
        "cholangiocellular carcinoma": [r"\bcholangiocarcinoma", r"\bcholangiocellular"],
        "squamous cell carcinoma": [r"\bsquamous\s+cell\s+carcinoma"],
        "pancreatic ductal adenocarcinoma": [r"\bpancreatic\s+ductal\s+adenocarcinoma", r"\bpdac\b"],
        "stomach cancer": [r"\b(?:gastric|stomach)\s+cancer"],
        "Papillary Renal Cell Carcinomas": [r"\bpapillary\s+renal"],
        "renal cell carcinoma": [r"\brenal\s+cell\s+carcinoma"],
        "bladder cancer": [r"\bbladder\s+cancer"],
        "thyroid cancer": [r"\bthyroid\s+cancer"],
        "adenocarcinoma": [r"\badenocarcinoma\b"],
        "Alzheimer's disease": [r"\balzheimer"],
        "Prion disease": [r"\bprion\s+disease\b", r"\bprion\s+protein\s+amyloid",
                          r"\bgerstmann", r"\bgss\b", r"\bfatal\s+familial\s+insomnia"],
        "obesity": [r"\bobesity\b"],
        "diabetes mellitus": [r"\bdiabetes\s+mellitus\b"],
        "diabetes": [r"\bdiabetes\b"],
        "Parkinson's disease": [r"\bparkinson"],
        "leukemia": [r"\bleukemia\b", r"\bleukaemia\b"],
        "lymphoma": [r"\blymphoma\b"],
        "osteoarthritis": [r"\bosteoarthritis\b"],
        "mitochondrial disease": [r"\bmitochondrial\s+disease\b", r"\bpolg[\s-]related\b"],
        "SARS-CoV-2 infection": [r"\bsars[\s-]cov[\s-]?2\b", r"\bcovid[\s-]?19\b"],
        "HCMV infection": [r"\bhcmv\b", r"\bhuman\s+cytomegalovirus\b"],
    }

    scores = {}
    for disease, patterns in disease_patterns.items():
        score = 0
        for p in patterns:
            if re.search(p, title):
                score += 10
            if re.search(p, abstract):
                score += 5
        if score > 0:
            scores[disease] = score

    if scores:
        # Prefer specific (longer name) over generic at equal score
        best = max(scores, key=lambda k: (scores[k], len(k)))
        if scores[best] >= 5:  # Require title or abstract mention
            return best
    return ""


def extract_cell_line(pub: Dict) -> str:
    """Extract cell line. Only return if explicitly mentioned."""
    methods = get_methods_text(pub)
    abstract = pub.get("ABSTRACT", "")

    cell_lines = [
        ("HEK293T", [r"\bHEK\s*293\s*T\b", r"\bHEK-?293T\b"]),
        ("HEK-293 cell", [r"\bHEK\s*293\b", r"\bHEK-?293\b"]),
        ("HeLa cells", [r"\bHeLa\b"]),
        ("A549", [r"\bA549\b"]),
        ("MCF7", [r"\bMCF-?7\b"]),
        ("U2OS", [r"\bU2OS\b"]),
        ("Jurkat", [r"\bJurkat\b"]),
        ("K562", [r"\bK562\b"]),
        ("THP-1", [r"\bTHP-?1\b"]),
        ("Vero E6", [r"\bVero\s+E6\b"]),
        ("A375", [r"\bA375\b"]),
        ("OVCAR3", [r"\bOVCAR-?3\b"]),
        ("HUVEC", [r"\bHUVEC\b"]),
        ("PC3", [r"\bPC-?3\b"]),
        ("LNCaP", [r"\bLNCaP\b"]),
        ("SH-SY5Y", [r"\bSH-?SY5Y\b"]),
        ("HCT116", [r"\bHCT\s*116\b"]),
        ("SW480", [r"\bSW480\b"]),
        ("SW620", [r"\bSW620\b"]),
        ("RAW264.7", [r"\bRAW\s*264\.?7\b"]),
        ("C2C12", [r"\bC2C12\b"]),
        ("NIH3T3", [r"\bNIH\s*-?3T3\b"]),
        ("iPSC", [r"\biPSC\b"]),
        ("MRC5", [r"\bMRC-?5\b"]),
        ("DU145", [r"\bDU-?145\b"]),
        ("H460", [r"\bH460\b", r"\bNCI-H460\b"]),
        ("WM266-4", [r"\bWM266[\s-]?4\b"]),
        ("WM115", [r"\bWM115\b"]),
        ("Mel526", [r"\bMel526\b"]),
        ("SKOV3", [r"\bSKOV[\s-]?3\b"]),
        ("Caco-2", [r"\bCaco[\s-]?2\b"]),
        ("HepG2", [r"\bHepG2\b"]),
    ]

    title = pub.get("TITLE", "")

    title = pub.get("TITLE", "")

    # If the title/abstract mentions tissue/organ keywords, cell lines in methods are likely
    # ancillary (expression/validation) and NOT the primary proteomics sample.
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    title_has_tissue = bool(re.search(
        r"\bbrain\b|\bliver\b|\bheart\b|\bkidney\b|\btissue\b|\btumor\b|"
        r"\bpostmortem\b|\bpost[\s-]?mortem\b|\bpatient\b|\bclinical\b|"
        r"\bserum\b|\bplasma\b|\bmilk\b|\bwhey\b|\bbiopsy\b",
        title_lower
    )) or bool(re.search(
        r"\bisolated\s+from\s+(?:the\s+)?(?:brain|liver|heart|kidney|tissue|tumor)|"
        r"\bbrain\s+(?:tissue|sample|extract|lysate)|"
        r"\bpostmortem\b|\bpost[\s-]?mortem\b|"
        r"\btissue\s+(?:sample|lysate|homogenat|extract|proteom)",
        abstract_lower
    ))

    # Require cell line mention in abstract (primary subject) OR title
    # Methods-only mentions are often ancillary
    search_text_primary = abstract + " " + title
    search_text_full = methods + " " + abstract

    for name, patterns in cell_lines:
        for p in patterns:
            # If title has tissue signals, only match cell line if it's in the abstract
            if title_has_tissue:
                if re.search(p, search_text_primary, re.IGNORECASE):
                    return name
            else:
                if re.search(p, search_text_full, re.IGNORECASE):
                    return name
    return ""


def extract_cell_type(pub: Dict) -> str:
    """Extract cell type. Conservative - only when clearly stated."""
    abstract = pub.get("ABSTRACT", "").lower()
    title = pub.get("TITLE", "").lower()
    methods = get_methods_text(pub).lower()

    cell_type_patterns = {
        "macrophage": [r"\bmacrophage"],
        "fibroblast": [r"\bfibroblast"],
        "epithelial": [r"\bepithelial\s+cell", r"\bepithelial\b"],
        "endothelial": [r"\bendothelial\s+cell", r"\bendothelial\b"],
        "neuron": [r"\bneuron"],
        "astrocyte": [r"\bastrocyte"],
        "heart cell": [r"\bcardiomyocyte", r"\bheart\s+cell"],
        "cardiac": [r"\bcardiac\b"],
        "hepatocyte": [r"\bhepatocyte"],
        "schizont": [r"\bschizont"],
        "prokaryotic cell": [r"\bprokaryot", r"\bbacterial\s+cell"],
    }

    scores = {}
    for ct, patterns in cell_type_patterns.items():
        score = 0
        for p in patterns:
            if re.search(p, title):
                score += 10
            if re.search(p, abstract):
                score += 5
        if score > 0:
            scores[ct] = score

    # Don't infer cell type from cell lines - gold often has "not available"
    # Only return cell type if found in title/abstract text

    if scores:
        best = max(scores, key=scores.get)
        if scores[best] >= 5:  # Require title or abstract mention
            return best
    return ""


def extract_instrument(pub: Dict) -> str:
    """Extract MS instrument. Usually findable in methods."""
    methods = get_methods_text(pub)
    text = get_full_text(pub)
    title = pub.get("TITLE", "")
    search_text = methods + " " + text

    for inst in INSTRUMENTS:
        if inst.lower() in title.lower():
            return inst

    title_lower = title.lower()
    title_instrument_patterns = [
        ("Orbitrap Astral", ["astral"]),
        ("Orbitrap Fusion Lumos", ["lumos"]),
        ("Orbitrap Eclipse", ["eclipse"]),
        ("timsTOF Pro", ["timstof"]),
    ]
    for inst, pats in title_instrument_patterns:
        for p in pats:
            if p in title_lower:
                return inst

    # Check filenames too
    raw_files = pub.get("Raw Data Files", [])
    for fn in raw_files[:5]:
        fn_lower = fn.lower()
        if "lumos" in fn_lower:
            return "Orbitrap Fusion Lumos"
        if "eclipse" in fn_lower:
            return "Orbitrap Eclipse"
        if "astral" in fn_lower:
            return "Orbitrap Astral"
        if "exploris" in fn_lower:
            return "Orbitrap Exploris 480"

    for inst in INSTRUMENTS:
        if inst.lower() in search_text.lower():
            return inst

    lower = search_text.lower()
    fallbacks = [
        ("Orbitrap Fusion Lumos", [r"fusion\s*lumos", r"\blumos\b"]),
        ("Orbitrap Fusion", [r"orbitrap\s+fusion"]),
        ("Orbitrap Exploris 480", [r"exploris\s*480"]),
        ("Orbitrap Exploris 240", [r"exploris\s*240"]),
        ("Orbitrap Eclipse", [r"eclipse"]),
        ("Orbitrap Astral", [r"\bastral\b"]),
        ("Q Exactive HF-X", [r"hf[\s-]?x\b"]),
        ("Q Exactive HF", [r"q[\s-]*exactive\s+hf", r"qe-?hf"]),
        ("Q Exactive Plus", [r"q[\s-]*exactive\s+plus"]),
        ("Q Exactive", [r"q[\s-]*exactive"]),
        ("Zeno TOF 7600", [r"zeno\s*tof\s*7600"]),
        ("LTQ Orbitrap Velos", [r"orbitrap\s+velos", r"velos\s+(?:pro|mass|spectromet)"]),
        ("LTQ Orbitrap Elite", [r"orbitrap\s+elite"]),
        ("LTQ Orbitrap XL", [r"orbitrap\s+xl"]),
        ("LTQ Orbitrap", [r"ltq\s+orbitrap", r"\bltq\b"]),
        ("timsTOF Pro", [r"timstof\s*pro"]),
        ("Triple TOF 5600", [r"triple\s*tof\s*5600", r"tripletof\s*5600"]),
        ("Triple TOF 6600", [r"triple\s*tof\s*6600", r"tripletof\s*6600"]),
        ("Synapt XS", [r"synapt\s*xs"]),
        ("SYNAPT G2-Si", [r"synapt\s*g2"]),
        ("Orbitrap Astral", [r"orbitrap\s*astral"]),
        ("Orbitrap Ascend", [r"orbitrap\s*ascend"]),
        ("Orbitrap Eclipse", [r"orbitrap\s*eclipse"]),
    ]

    for inst, pats in fallbacks:
        for p in pats:
            if re.search(p, lower):
                return inst
    return ""


def extract_cleavage_agent(pub: Dict) -> str:
    """Extract protease. Almost always trypsin."""
    methods = get_methods_text(pub).lower()

    # Pepsin check first — HDX-MS studies typically use pepsin, not trypsin
    # Check if this is an HDX study before defaulting to trypsin
    is_hdx = bool(re.search(r"\bhdx\b|\bhydrogen[\s-]?deuterium\s+exchange\b", methods))
    if is_hdx and "pepsin" in methods:
        return "Pepsin"

    if "trypsin/p" in methods:
        return "Trypsin/P"
    has_trypsin = "trypsin" in methods or "tryptic" in methods or "trypsinization" in methods
    has_lysc = "lys-c" in methods or "lysc" in methods or "lysyl endopeptidase" in methods
    if has_trypsin:
        return "Trypsin"
    if has_lysc:
        return "Lys-C"
    if "pepsin" in methods:
        return "Pepsin"
    if "asp-n" in methods:
        return "Asp-N"
    if "chymotrypsin" in methods:
        return "Chymotrypsin"
    if "glu-c" in methods or "gluc" in methods or "v8" in methods:
        return "Glutamyl endopeptidase"
    # Default to Trypsin only if methods suggest standard bottom-up proteomics
    if re.search(r"\bdigest(?:ed|ion)\b|\bproteo(?:lysis|lytic)\b|\bbottom[\s-]?up\b", methods):
        return "Trypsin"
    return ""


def extract_fragmentation(pub: Dict) -> str:
    """Extract fragmentation method from text."""
    methods = get_methods_text(pub).lower()
    text = get_full_text(pub).lower()

    if "ethcd" in methods:
        return "EThcD"
    if re.search(r"\bhcd\b", methods) or "higher-energy" in methods or "higher energy collisional" in methods:
        return "HCD"
    if re.search(r"\betd\b", methods) and "electron transfer" in methods:
        return "ETD"
    if re.search(r"\bcid\b", methods) or "collision-induced dissociation" in methods or "collision induced dissociation" in methods:
        if re.search(r"\bhcd\b", methods):
            return "HCD"
        return "CID"
    if re.search(r"\bhcd\b", text) or "higher-energy" in text:
        return "HCD"
    if re.search(r"\bcid\b", text) or "collision-induced dissociation" in text:
        return "CID"
    return ""


def extract_label(pub: Dict) -> Tuple[str, List[str]]:
    """Extract labeling strategy."""
    text = get_full_text(pub).lower()
    methods = get_methods_text(pub).lower()
    abstract = pub.get("ABSTRACT", "").lower()
    primary = methods + " " + abstract
    combined = methods + " " + text

    if "tmt" in primary or "tandem mass tag" in primary:
        # If paper also mentions "label-free"/"label free" in abstract, it describes
        # multiple datasets. Only use TMT if raw filenames suggest TMT data.
        mentions_label_free = bool(re.search(r"\blabel[\s-]?free\b", abstract))
        raw_files = pub.get("Raw Data Files", [])
        raw_text = " ".join(str(f).lower() for f in raw_files) if isinstance(raw_files, (list, tuple)) else str(raw_files).lower()
        # TMT filenames typically contain "tmt", channel names, or multiplexed patterns
        raw_suggests_tmt = bool(re.search(r"tmt|plex|channel|fraction.*rep|_\d{3}[nc]", raw_text))
        if mentions_label_free and not raw_suggests_tmt:
            return "label free sample", ["label free sample"]

        if "tmtpro" in combined or "tmt16" in combined or "16plex" in combined or "16-plex" in combined:
            return "TMT16plex", LABELS_TMT16
        if "tmt11" in combined or "11plex" in combined or "11-plex" in combined:
            return "TMT11plex", LABELS_TMT11
        if "tmt10" in combined or "10plex" in combined or "10-plex" in combined:
            return "TMT10plex", LABELS_TMT10
        if "tmt6" in combined or "6plex" in combined or "6-plex" in combined:
            return "TMT6plex", LABELS_TMT6
        if "134n" in combined or "134" in combined:
            return "TMT16plex", LABELS_TMT16
        if "131c" in combined or "131n" in combined:
            return "TMT11plex", LABELS_TMT11
        return "TMT10plex", LABELS_TMT10

    if "itraq" in primary:
        if re.search(r"8[\s-]?plex|itraq[\s-]?8", combined) or "113" in combined:
            return "iTRAQ8plex", ["iTRAQ8plex-113", "iTRAQ8plex-114", "iTRAQ8plex-115",
                                   "iTRAQ8plex-116", "iTRAQ8plex-117", "iTRAQ8plex-118",
                                   "iTRAQ8plex-119", "iTRAQ8plex-121"]
        return "iTRAQ4plex", ["iTRAQ4plex-114", "iTRAQ4plex-115",
                               "iTRAQ4plex-116", "iTRAQ4plex-117"]

    if "silac" in primary:
        return "SILAC", ["SILAC heavy", "SILAC medium", "SILAC light"]

    return "label free sample", ["label free sample"]


def extract_modifications(pub: Dict) -> List[str]:
    """Extract PTMs. Gold ordering: Carbamidomethyl slot 0, Oxidation slot 1."""
    methods = get_methods_text(pub).lower()
    text = get_full_text(pub).lower()

    mods = []

    # Slot 0: Alkylation modification (fixed mod) - gold has Carbamidomethyl in slot 0 for 64/100 PXDs
    if "carbamidomethyl" in methods or "iodoacetamide" in methods or " iaa " in methods:
        mods.append("Carbamidomethyl")
    elif "propionamide" in methods or "acrylamide" in methods:
        mods.append("propionamide")
    elif "carbamyl" in methods:
        mods.append("Carbamyl")

    # Slot 1: Oxidation (variable mod) - gold has Oxidation in slot 1 for 833 values
    if "oxidation" in methods:
        mods.append("Oxidation")
    if "acetyl" in methods and ("n-term" in methods or "protein" in methods):
        mods.append("Acetyl")
    # Only add Phospho when the study IS specifically phosphoproteomics
    # (not just any mention of "phosphorylation" in introduction/discussion)
    title_abs = (pub.get("TITLE", "") + " " + pub.get("ABSTRACT", "")).lower()
    is_phospho_study = (
        re.search(r"\bphospho(?:proteo|peptide)", title_abs) or
        re.search(r"\bphospho(?:proteo|peptide)", methods) or
        re.search(r"\bphospho[\w-]*\s+enrichment\b", methods) or
        (re.search(r"\btio2\b|\btitanium\s+dioxide\b|\bimac\b|\bfe[\s-]?imac\b|\bmoac\b", methods)
         and ("phospho" in methods or "phosphorylation" in methods))
    )
    if is_phospho_study:
        mods.append("Phospho")
    if "deamid" in methods:
        mods.append("Deamidated")
    if "ubiquitin" in text or "diglycine" in text or "gly-gly" in text:
        mods.append("GlyGly")
    if re.search(r"gln\s*-?>?\s*pyro-?glu", methods):
        mods.append("Gln->pyro-Glu")

    label_type, _ = extract_label(pub)
    if "TMT" in label_type:
        mods.append(label_type)
    elif "iTRAQ" in label_type:
        mods.append(label_type)

    # Ensure Oxidation is present (almost universally in gold slot 1)
    if "Oxidation" not in mods:
        mods.append("Oxidation")

    return mods


def extract_mass_tolerance(pub: Dict) -> Tuple[str, str]:
    """Extract precursor and fragment mass tolerances."""
    methods = get_methods_text(pub)

    precursor = ""
    fragment = ""

    prec_pats = [
        r"precursor\s*(?:mass\s*)?(?:ion\s*)?tolerance\s*(?:of\s*|was\s*(?:set\s*(?:to\s*)?)?|=\s*)?(\d+\.?\d*)\s*(ppm|Da|da|mmu)",
        r"precursor\s*(?:ion\s*)?(?:mass\s*)?(?:tolerance|accuracy|error)\s*(?:of\s*|was\s*(?:set\s*(?:to\s*|at\s*)?)?|=\s*)?(\d+\.?\d*)\s*(ppm|Da|da|mmu)",
        r"(\d+\.?\d*)\s*(ppm|Da)\s*(?:for\s*)?(?:precursor|parent|MS1|ms1|MS\s+mass)",
        r"(?:parent|precursor|MS1|ms1)\s*(?:ion\s*)?(?:mass\s*)?(?:tolerance|accuracy|error)\s*(?:of\s*|was\s*(?:set\s*(?:to\s*|at\s*)?)?|=\s*)?(\d+\.?\d*)\s*(ppm|Da|mmu)",
        r"mass\s+tolerance\s+(?:of\s+)?(\d+\.?\d*)\s*(ppm|Da)[\s,]*(?:for\s+)?precursor",
        r"[Pp]recursor\s+and\s+fragment\s+mass\s+tolerances?\s+(?:were\s+)?set\s+to\s+(\d+\.?\d*)\s*(ppm|Da|mmu)",
    ]
    for pat in prec_pats:
        m = re.search(pat, methods, re.IGNORECASE)
        if m:
            precursor = f"{m.group(1)} {m.group(2)}"
            break

    frag_pats = [
        r"fragment\s*(?:ion\s*)?(?:mass\s*)?tolerance\s*(?:of\s*|was\s*(?:set\s*(?:to\s*|at\s*)?)?|=\s*)?(\d+\.?\d*)\s*(ppm|Da|da|mmu)",
        r"fragment\s*(?:ion\s*)?(?:mass\s*)?(?:tolerance|accuracy|error)\s*(?:of\s*|was\s*(?:set\s*(?:to\s*|at\s*)?)?|=\s*)?(\d+\.?\d*)\s*(ppm|Da|da|mmu)",
        r"(\d+\.?\d*)\s*(Da|ppm|mmu)\s*(?:for\s*)?(?:fragment|MS2|ms2|MS/MS|product)",
        r"(?:fragment|MS2|ms2|MS/MS|product)\s*(?:ion\s*)?(?:mass\s*)?(?:tolerance|accuracy|error)\s*(?:of\s*|was\s*(?:set\s*(?:to\s*|at\s*)?)?|=\s*)?(\d+\.?\d*)\s*(ppm|Da|mmu)",
        r"fragment\s*(?:ion\s*)?(?:mass\s*)?tolerance\s*(?:was\s*)?(?:set\s*)?[^.]{0,30}\((\d+\.?\d*)\s*(Da|ppm|mmu)\)",
        r"(\d+\.?\d*)\s*(Da|ppm|mmu)\s+for\s+MS/MS",
        r"[Pp]recursor\s+and\s+fragment\s+mass\s+tolerances?\s+(?:were\s+)?set\s+to\s+(\d+\.?\d*)\s*(ppm|Da|mmu)",
        r"fragment\s+mass\s+error\s+(?:set\s+at|of)\s+(\d+\.?\d*)\s*(ppm|Da|mmu)",
    ]
    for pat in frag_pats:
        m = re.search(pat, methods, re.IGNORECASE)
        if m:
            fragment = f"{m.group(1)} {m.group(2)}"
            break

    if not precursor or not fragment:
        combo = re.search(
            r"(?:peptide\s+and\s+fragment|precursor\s+and\s+fragment)\s+mass\s+tolerance\s+(?:was\s+)?set\s+to\s+(\d+\.?\d*)\s*(ppm|Da)\s+and\s+(\d+\.?\d*)\s*(Da|ppm)",
            methods, re.IGNORECASE
        )
        if combo:
            if not precursor:
                precursor = f"{combo.group(1)} {combo.group(2)}"
            if not fragment:
                fragment = f"{combo.group(3)} {combo.group(4)}"

    if not precursor:
        precursor = ""
    if not fragment:
        fragment = ""

    return precursor, fragment


def extract_collision_energy(pub: Dict) -> str:
    """Extract collision energy."""
    methods = get_methods_text(pub)

    patterns = [
        r"(\d+)\s*%?\s*(?:normalized\s*)?(?:NCE|nce|collision\s*energy)",
        r"(?:NCE|nce|normalized\s*collision\s*energy)\s*(?:of\s*|=\s*)?(\d+)\s*%?",
        r"(?:collision\s*energy)\s*(?:of\s*|was\s*(?:set\s*to\s*)?)?(\d+)\s*%?\s*(?:NCE)?",
    ]

    for pat in patterns:
        m = re.search(pat, methods, re.IGNORECASE)
        if m:
            nce = m.group(1)
            if 15 <= int(nce) <= 50:
                return f"{nce} NCE"
    return ""


def extract_missed_cleavages(pub: Dict) -> str:
    """Extract number of missed cleavages. Handles digit and word-form numbers."""
    methods = get_methods_text(pub)
    _word_to_digit = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    _num_pat = r"(\d+|zero|one|two|three|four|five)"
    patterns = [
        rf"{_num_pat}\s+missed\s+cleavage",
        rf"missed\s+cleavage[s]?\s*(?:of\s*|up\s*to\s*|set\s*to\s*)?{_num_pat}",
        rf"maximum\s*(?:of\s*)?{_num_pat}\s+missed",
        rf"up\s*to\s*{_num_pat}\s+missed",
        rf"allow(?:ing|ed)?\s+{_num_pat}\s+missed",
    ]
    for pat in patterns:
        m = re.search(pat, methods, re.IGNORECASE)
        if m:
            raw = m.group(1).lower()
            val = _word_to_digit.get(raw)
            if val is None:
                try:
                    val = int(raw)
                except ValueError:
                    continue
            if 0 <= val <= 5:
                return str(val)
    return ""


def extract_ms2_analyzer(pub: Dict) -> str:
    """Extract MS2 mass analyzer based on instrument type."""
    instrument = extract_instrument(pub).lower()
    frag = extract_fragmentation(pub)

    if "ltq" in instrument:
        if frag == "CID":
            return "ion trap"
        elif frag == "HCD":
            return "Orbitrap"
        return "ion trap"

    if "orbitrap" in instrument or "exploris" in instrument or "eclipse" in instrument:
        return "Orbitrap"
    if "q exactive" in instrument:
        return "Orbitrap"
    if "tof" in instrument or "astral" in instrument:
        return "TOF"
    if "synapt" in instrument or "impact" in instrument or "maxis" in instrument:
        return "TOF"
    if "tsq" in instrument:
        return "Quadrupole"
    return ""


def extract_material_type(pub: Dict) -> str:
    """Extract material type from paper context.
    Be conservative - wrong material type is worse than missing."""
    title = pub.get("TITLE", "").lower()
    abstract = pub.get("ABSTRACT", "").lower()
    methods = get_methods_text(pub).lower()
    text = title + " " + abstract + " " + methods

    # Bacterial/microbial strains
    if re.search(r"\bbacterial\s+strains?\b", text) or re.search(r"\bmicrobial\s+strains?\b", text):
        return "bacterial strain"

    # Synthetic peptides (e.g., peptide library studies)
    if re.search(r"\bsynthetic\s+peptide\b", title + " " + abstract):
        return "synthetic"

    # FFPE tissue (specific subtype, high priority)
    if re.search(r"\bffpe\b", text) or re.search(r"\bformalin[- ]fixed\b", text):
        return "Formalin-fixed paraffin-embedded tissue"

    # Tissue/organ/body fluid signals in TITLE or ABSTRACT take priority over cell line in methods.
    # Papers often mention cell lines in methods (for expression/validation) while
    # the actual proteomics data comes from tissue samples.
    title_tissue_signals = [
        r"\btissue\b", r"\bbrain\b", r"\bliver\b", r"\bheart\b", r"\bkidney\b",
        r"\blung\b", r"\btumor\b", r"\bbiopsy\b", r"\bbiopsies\b",
        r"\bpostmortem\b", r"\bpost[\s-]?mortem\b",
        r"\bpatient\b", r"\bclinical\b", r"\bcohort\b",
        r"\bserum\b", r"\bplasma\b", r"\burine\b", r"\bcerebrospinal\b",
        r"\bmilk\b", r"\bwhey\b",
    ]
    # Check abstract too — "isolated from the brain" clearly indicates tissue sample
    abstract_tissue_signals = [
        r"\bisolated\s+from\s+(?:the\s+)?(?:brain|liver|heart|kidney|tissue|tumor)",
        r"\bbrain\s+(?:tissue|sample|extract|lysate|homogenat)",
        r"\bpostmortem\b", r"\bpost[\s-]?mortem\b",
        r"\bpatient\s+(?:tissue|sample|biopsy|serum|plasma)",
        r"\btissue\s+(?:sample|lysate|homogenat|extract|proteom)",
    ]
    title_has_tissue = (
        any(re.search(p, title) for p in title_tissue_signals) or
        any(re.search(p, abstract) for p in abstract_tissue_signals)
    )

    # Body fluids → organism part (check early, before cell line detection)
    fluid_patterns = [
        r"\bplasma\s+(?:sample|proteom|fraction)", r"\bserum\s+(?:sample|proteom|fraction)",
        r"\burine\b", r"(?<![mg]-)\bcsf\b",
        r"\bcerebrospinal\s+fluid\b", r"\bsaliva\b",
        r"\bbody\s+fluid\b", r"\bsecretom",
        r"\bwhey\s+(?:protein|proteom)", r"\bbovine\s+milk\b",
        r"\bmilk\s+(?:protein|proteom|sample|serum)",
        r"\bextracellular\s+vesicle",
    ]
    for p in fluid_patterns:
        if re.search(p, title + " " + abstract):
            return "organism part"

    # Tissue in title/abstract — prioritize over cell line mentions in methods
    if re.search(r"\bbiopsy\b|\bbiopsies\b", title + " " + abstract):
        return "tissue"
    if re.search(r"\btissue\s+(?:sample|section|specimen|proteom)", title + " " + abstract):
        return "tissue"
    if re.search(r"\btumor\s+tissue\b", title + " " + abstract):
        return "tissue"
    if re.search(r"\bpostmortem\b|\bpost[\s-]?mortem\b", title + " " + abstract):
        return "tissue"

    # Primary cells trump cell line detection (BMDMs, primary macrophages, etc.)
    primary_cell_patterns = [
        r"\bprimary\s+(?:cells?|culture|macrophage|fibroblast|neuron|hepatocyte)",
        r"\bbone\s+marrow[- ]derived\b",
        r"\bbmdm\b",
        r"\bprimary\s+(?:bone\s+marrow|peritoneal|alveolar)\b",
    ]
    for p in primary_cell_patterns:
        if re.search(p, text, re.IGNORECASE):
            return "cell"

    # Cell line detection — only if title doesn't suggest tissue
    if not title_has_tissue:
        # HeLa → 'cell line'
        if re.search(r"\bhela\b", text, re.IGNORECASE):
            return "cell line"

        # Specific non-HeLa cell lines → 'cell'
        non_hela_cell_lines = [
            r"\ba375\b", r"\bmda[\s-]?mb[\s-]?435\w*\b", r"\bhct[\s-]?116\b",
            r"\bhek[\s-]?293\w*\b", r"\ba549\b", r"\bvero\b", r"\bmrc[\s-]?5\b",
            r"\bdu[\s-]?145\b", r"\bpc[\s-]?3\b", r"\bsw[46]\d{2}\b", r"\blncap\b",
            r"\bmcf[\s-]?7\b", r"\bsh[\s-]?sy5y\b", r"\bjurkat\b", r"\bk562\b",
            r"\bthp[\s-]?1\b", r"\bu2os\b", r"\braw\s*264\b", r"\bc2c12\b",
            r"\bnih[\s-]?3t3\b", r"\bcos[\s-]?7\b", r"\bsf9\b", r"\b3t3\b",
            r"\bovcar\b", r"\bcalu[\s-]?3\b", r"\bhuh[\s-]?7\b", r"\bmel526\b",
            r"\bwm266\b", r"\bwm115\b", r"\bh460\b",
        ]
        for p in non_hela_cell_lines:
            if re.search(p, text, re.IGNORECASE):
                return "cell"

        # Generic cell line indicators → 'cell line'
        if re.search(r"\bcell\s+line\b", text) or re.search(r"\btransfect(?:ed|ion)\b", text):
            return "cell line"

    # Lysate
    if re.search(r"\blysate\b", title + " " + abstract):
        return "lysate"

    # Cell culture
    if re.search(r"\bcell\s+(?:culture|pellet|extract|suspension)\b", title + " " + abstract):
        return "cell"

    if re.search(r"\bsingle[- ]cell\b", title + " " + abstract):
        return "cell"

    # Don't guess if not confident
    return ""


def extract_acquisition_method(pub: Dict, raw_files: List[str] = None) -> str:
    """Extract proteomics data acquisition method (DDA or DIA).

    Check filenames first (most reliable), then text.
    Default to Data-Dependent Acquisition (most common).
    """
    # Check filenames first
    if raw_files:
        for fn in raw_files:
            fn_up = fn.upper()
            # DIA clearly in filename
            if re.search(r'[_\-.]DIA[_\-.]', fn_up) or fn_up.endswith('_DIA.RAW') or fn_up.endswith('.DIA.RAW'):
                return "Data-Independent Acquisition"
            # SWATH is DIA
            if "SWATH" in fn_up:
                return "Data-Independent Acquisition"

    methods = get_methods_text(pub).lower()
    abstract = pub.get("ABSTRACT", "").lower()
    title = pub.get("TITLE", "").lower()
    text = methods + " " + abstract + " " + title

    # DIA-specific keywords
    dia_patterns = [
        r"\bdata[\s-]?independent\s+acquisition\b",
        r"\bdia[\s-]?nn\b",
        r"\bdia\b.*\bacquisition\b",
        r"\bdia\b[\s,.]",                    # standalone "DIA" followed by space/punctuation
        r"\bswath[\s-]?ms\b",
        r"\bswath\b",
        r"\bwindowed\s+(?:data[\s-]?independent|acquisition)\b",
        r"\bsonar\b.*\bmass\s+spec",
        r"\bms[eE]\b",                       # Waters MSE (DIA mode)
        r"\bhdms[eE]\b",                     # Waters HDMSe
        r"\ball[\s-]?ion\s+fragmentation\b", # AIF is DIA
        r"\bdia-?pasef\b",                   # diaPASEF
    ]
    for pat in dia_patterns:
        if re.search(pat, text):
            return "Data-Independent Acquisition"

    # DDA-specific keywords
    dda_patterns = [
        r"\bdata[\s-]?dependent\s+acquisition\b",
        r"\bdata[\s-]?dependent\b",
        r"\bdda\b",
        r"\bshotgun\s+proteomics\b",
        r"\bbottom[\s-]?up\s+proteomics\b",
    ]
    for pat in dda_patterns:
        if re.search(pat, text):
            return "Data-Dependent Acquisition"

    # Don't guess — wrong acquisition method hurts more than missing
    return ""


def extract_separation(pub: Dict) -> str:
    """Extract separation method. Almost always reversed-phase LC for LC-MS/MS."""
    methods = get_methods_text(pub).lower()
    abstract = pub.get("ABSTRACT", "").lower()
    text = methods + " " + abstract

    # SAX is specifically strong anion exchange
    if re.search(r'\bsax\b|\bstrong\s+anion[\s-]?exchange\b', text):
        return "SAX"

    # Explicit reversed-phase indicators (check BEFORE generic HPLC)
    if re.search(r'\bnano[\s-]?lc\b|\bnanoflow\b|\brp[\s-]?lc\b|\breversed[\s-]?phase\b|\bc18\b|\brp\s+column\b', text):
        return "Reversed-phase chromatography"

    # Generic LC-MS/MS (almost always RP in proteomics)
    if re.search(r'\blc[\s-]?ms(?:/ms)?\b|\bliquid\s+chroma', text):
        return "Reversed-phase chromatography"

    # HPLC mentioned explicitly without more specific RP context
    if re.search(r'\bhplc\b|\bhigh[\s-]performance\s+liquid\s+chrom', text):
        return "High-performance liquid chromatography"

    # If instrument found → LC-MS/MS assumed
    if extract_instrument(pub):
        return "Reversed-phase chromatography"

    return ""


def extract_fractionation_method(pub: Dict) -> str:
    """Extract sample fractionation method.
    Only fill when positively detected; do NOT default to 'no fractionation'
    because many papers DO have fractionation and a wrong fill hurts score."""
    methods = get_methods_text(pub).lower()
    abstract = pub.get("ABSTRACT", "").lower()
    text = methods + " " + abstract

    for method, patterns in FRACTIONATION_VOCAB.items():
        for pat in patterns:
            if re.search(pat, text):
                return method

    # Detect generic fractionation with partial classification
    if re.search(r'\bfractionat', text):
        if re.search(r'\breversed[\s-]?phase\b|\brplc\b|\bhigh[\s-]?ph\b', text):
            return "High-pH reversed-phase chromatography"
        # Don't guess blindly if we see "fractionation" but can't classify it
        return ""

    # Only fill "no fractionation" when paper explicitly says so
    # (covered by the FRACTIONATION_VOCAB 'no fractionation' patterns above)
    return ""


def extract_enrichment_method(pub: Dict) -> str:
    """Extract enrichment method.
    Only fill when positively detected; do NOT default to 'no enrichment'
    because an incorrect fill hurts when gold has a specific enrichment."""
    methods = get_methods_text(pub).lower()
    abstract = pub.get("ABSTRACT", "").lower()
    text = methods + " " + abstract

    for method, patterns in ENRICHMENT_VOCAB.items():
        for pat in patterns:
            if re.search(pat, text):
                return method

    return ""


def extract_sex(pub: Dict) -> str:
    """Extract sex - only if clearly stated."""
    methods = get_methods_text(pub).lower()
    abstract = pub.get("ABSTRACT", "").lower()
    title = pub.get("TITLE", "").lower()

    has_female = bool(re.search(r"\bfemale\b", title) or re.search(r"\bfemale\b", abstract))
    has_male = bool(re.search(r"\bmale\b", title) or re.search(r"\bmale\b", abstract))

    if has_female and has_male:
        return ""
    if has_female and re.search(r"\bfemale\b", title):
        return "female"
    if has_male and re.search(r"\bmale\b", title):
        return "male"

    cell_line_sex = {
        "hela": "female", "mcf-7": "female", "mcf7": "female",
        "ovcar": "female",
    }
    text = title + " " + abstract + " " + methods
    for cl, sex in cell_line_sex.items():
        if cl in text:
            return sex

    return ""


def extract_developmental_stage(pub: Dict) -> str:
    """Extract developmental stage. Only returns 'adult' when explicitly stated."""
    abstract = pub.get("ABSTRACT", "").lower()
    title = pub.get("TITLE", "").lower()
    methods = get_methods_text(pub).lower()
    text = title + " " + abstract + " " + methods

    # Non-adult developmental stages (only when explicit)
    if re.search(r"\bfetal\b|\bfetus\b", title + " " + abstract):
        return "fetus"
    if re.search(r"\bembryo(?:nic)?\b", title + " " + abstract):
        return "embryo"
    if re.search(r"\bneonatal\b|\bnewborn\b", title + " " + abstract):
        return "neonatal"

    # Only say "adult" when explicitly stated with a subject noun
    if re.search(r"\badult\s+(?:patient|subject|volunteer|donor|human|mouse|mice|rat|individual)s?\b", text):
        return "adult"

    # Do NOT infer "adult" from clinical studies - too many false positives
    return ""


def extract_alkylation_reagent(pub: Dict) -> str:
    """Extract alkylation reagent. IAA (iodoacetamide) or CAA (chloroacetamide) are most common."""
    methods = get_methods_text(pub).lower()
    text = get_full_text(pub).lower()
    combined = methods + " " + text
    if "iodoacetamide" in combined or re.search(r'\biaa\b', combined):
        return "IAA"
    if "chloroacetamide" in combined or re.search(r'\bcaa\b', combined):
        return "CAA"
    return ""


def extract_reduction_reagent(pub: Dict) -> str:
    """Extract reduction reagent. DTT or TCEP are most common."""
    methods = get_methods_text(pub).lower()
    text = get_full_text(pub).lower()
    combined = methods + " " + text
    # TCEP check before DTT since TCEP text has no overlap risk
    if re.search(r'\btcep\b', combined):
        return "TCEP"
    if "dithiothreitol" in combined or re.search(r'\bdtt\b', combined):
        return "DTT"
    return ""


def extract_ionization_type(pub: Dict) -> str:
    """Extract ionization type. Nano-ESI is standard for LC-MS/MS proteomics."""
    methods = get_methods_text(pub).lower()
    text = get_full_text(pub).lower()
    combined = methods + " " + text
    if re.search(r'\bnano[\s-]?esi\b|\bnano[\s-]?electrospray\b', combined):
        return "nano-electrospray ionization"
    if re.search(r'\bnano[\s-]?lc\b|\bnanoflow\b|\bnano[\s-]?uplc\b', combined):
        return "nano-electrospray ionization"
    # All LC-MS/MS = electrospray ionization
    instrument = extract_instrument(pub)
    if instrument or re.search(r'\blc[\s-]?ms\b', combined):
        return "electrospray ionization"
    return ""


# ─── Training data loading ────────────────────────────────────────────────────

def load_training_data() -> Dict[str, pd.DataFrame]:
    """Load all training SDRFs mapped to submission schema."""
    train_data = {}
    for f in sorted(TRAIN_SDRF_DIR.glob("*.tsv")):
        pxd = f.stem.split("_")[0]
        df = pd.read_csv(f, sep="\t")

        mapped = {"PXD": pxd}
        for col in df.columns:
            sub_col = NATIVE_TO_SUBMISSION.get(col)
            if sub_col:
                vals = df[col].apply(extract_nt)
                if sub_col in mapped:
                    existing = mapped[sub_col]
                    if isinstance(existing, pd.Series):
                        if vals.notna().sum() > existing.notna().sum():
                            mapped[sub_col] = vals
                    else:
                        mapped[sub_col] = vals
                else:
                    mapped[sub_col] = vals

            fv_col = FACTOR_VALUE_MAP.get(col)
            if fv_col and fv_col not in mapped:
                mapped[fv_col] = df[col].apply(extract_nt)

        train_data[pxd] = pd.DataFrame(mapped)
    return train_data


def build_training_value_banks(train_data: Dict[str, pd.DataFrame]) -> Dict[str, Counter]:
    """Build per-column value frequency banks from training gold SDRFs."""
    banks: Dict[str, Counter] = defaultdict(Counter)
    skip_vals = {"not applicable", "not available", "nan", "", "text span"}
    for pxd, df in train_data.items():
        for col in df.columns:
            if col in ("PXD", "ID", "Raw Data File"):
                continue
            for v in df[col].dropna().astype(str):
                if v.strip().lower() not in skip_vals:
                    banks[col][v.strip()] += 1
    return banks


# ─── Nearest training PXD retrieval ──────────────────────────────────────────

def tokenize(text: str) -> set:
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def tokenize_filenames(raw_files: List[str]) -> set:
    """Extract meaningful tokens from raw file names."""
    tokens = set()
    for fn in raw_files:
        stem = Path(fn).stem.lower()
        # Split on common delimiters and extract meaningful tokens
        parts = re.split(r'[-_.]', stem)
        for p in parts:
            if len(p) >= 3 and not p.isdigit():
                tokens.add(p)
    return tokens


def find_nearest_training_pxds(
    test_pub: Dict, train_pubs: Dict[str, Dict],
    train_data: Dict[str, pd.DataFrame],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Composite retrieval: text similarity + filename token overlap + rows/file ratio."""
    test_text_tokens = tokenize(get_full_text(test_pub))
    test_method_tokens = tokenize(test_pub.get("METHODS", ""))
    test_files = test_pub.get("Raw Data Files", [])
    test_file_tokens = tokenize_filenames(test_files)
    test_rows_per_file = 1  # default, will be overridden per-PXD

    scores = []
    for pxd, pub in train_pubs.items():
        train_full_tokens = tokenize(get_full_text(pub))
        train_method_tokens = tokenize(pub.get("METHODS", ""))
        train_files = pub.get("Raw Data Files", [])
        train_file_tokens = tokenize_filenames(train_files)

        # Text Jaccard (methods weighted more heavily)
        def jaccard(a, b):
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        text_sim = 0.6 * jaccard(test_method_tokens, train_method_tokens) + \
                   0.4 * jaccard(test_text_tokens, train_full_tokens)

        # Filename token overlap
        file_sim = jaccard(test_file_tokens, train_file_tokens) if test_file_tokens and train_file_tokens else 0.0

        # Rows-per-file ratio similarity (to detect multiplexed experiments)
        if pxd in train_data and "Raw Data File" in train_data[pxd].columns:
            tr_df = train_data[pxd]
            tr_files = tr_df["Raw Data File"].nunique()
            if tr_files > 0:
                tr_rpf = len(tr_df) / tr_files
            else:
                tr_rpf = 1
        else:
            tr_rpf = 1
        te_rpf = len(test_files) if test_files else 1
        # rows_per_file might be given separately - use file count for now
        rpf_sim = 1.0 / (1.0 + abs(tr_rpf - 1))  # baseline, will be per-PXD

        composite = 0.7 * text_sim + 0.3 * file_sim
        scores.append((pxd, composite))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


# ─── Filename-level per-row parsers ──────────────────────────────────────────

def parse_filename_fragmentation(fname: str) -> str:
    """Detect fragmentation method encoded in filename."""
    stem = Path(fname).stem
    stem_up = stem.upper()

    # Order matters: check more specific first
    if "ETHCD" in stem_up or re.search(r'HCD[\s_-]?ET|ET[\s_-]?HCD', stem_up):
        return "EThcD"
    if re.search(r'[_\-.]ETHCD[_\-.]|[_\-.]ETHCD$', stem_up):
        return "EThcD"
    # HCD_etHCD combined → use EThcD as broader
    if re.search(r'HCD_ETHCD|ETHCD', stem_up):
        return "EThcD"
    if re.search(r'[_\-.]HCD[_\-.]|[_\-.]HCD$', stem_up):
        return "HCD"
    if re.search(r'[_\-.]CID[_\-.]|[_\-.]CID$', stem_up):
        return "CID"
    if re.search(r'[_\-.]ETD[_\-.]|[_\-.]ETD$', stem_up):
        return "ETD"
    return ""


def parse_filename_acquisition(fname: str) -> str:
    """Detect acquisition method encoded in filename."""
    stem = Path(fname).stem.upper()
    # Check for DIA explicitly
    if re.search(r'[_\-.]DIA[_\-.]|[_\-.]DIA$|^DIA[_\-.]', stem):
        return "Data-Independent Acquisition"
    if "SWATH" in stem:
        return "Data-Independent Acquisition"
    if re.search(r'[_\-.]DDA[_\-.]|[_\-.]DDA$|^DDA[_\-.]', stem):
        return "Data-Dependent Acquisition"
    return ""


def parse_filename_bioreplicate(fname: str) -> str:
    """Extract biological replicate number from filename."""
    stem = Path(fname).stem

    # Explicit BR1/BR2/BR3 pattern (with delimiter boundary)
    m = re.search(r'(?:^|[_\-\.])BR(\d+)(?:[_\-\.]|$)', stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # rb1, rb2, rb3 style (e.g. mousebrain-KO-...-rb1_u.raw)
    m = re.search(r'[_\-]rb(\d+)(?:[_\-\.]|$)', stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # rep1, rep2, Rep#1, Rep_1 patterns
    m = re.search(r'[_\-\s]rep[#_\-]?(\d+)(?:[_\-\.]|$)', stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # Rep at end: _rep1, _rep2
    m = re.search(r'rep(\d+)$', stem, re.IGNORECASE)
    if m:
        return m.group(1)

    return ""


def parse_filename_treatment(fname: str, pub: Dict) -> str:
    """Extract treatment condition from filename.
    Note: underscores are word chars in Python regex, so we use explicit delimiters."""
    stem = Path(fname).stem.lower()

    # Infection studies: infected vs uninfected/mock/distal
    if re.search(r'(?:^|[_\-\.])infected?(?:[_\-\.]|$)', stem):
        return "infected"
    if re.search(r'(?:^|[_\-\.])(?:mock|uninfected?|distal)(?:[_\-\.]|$)', stem):
        return "uninfected"

    # Drug/compound treatment: DMSO vs inhibitor/compound
    # Use delimiter-based matching since _ is a word char
    if re.search(r'(?:^|[_\-\.])dmso(?:[_\-\.]|$)', stem):
        return "DMSO"
    if re.search(r'(?:^|[_\-\.])inhib(?:[_\-\.]|$)|(?:^|[_\-\.])inhb(?:[_\-\.]|$)|inhibitor', stem):
        return "inhibitor"
    if re.search(r'(?:^|[_\-\.])treated(?:[_\-\.]|$)', stem):
        return "treated"
    if re.search(r'(?:^|[_\-\.])(?:untreated|vehicle|ctrl|control)(?:[_\-\.]|$)', stem):
        return "untreated"

    # Heat treatment: temperature from filename like "0C", "65C", "100C", "90C10B"
    # Allow any non-letter after C (handles TPP suffixes like "90C10B")
    m = re.search(r'[_\-](\d{1,3})[cC](?:[^a-zA-Z]|$)', stem)
    if m:
        temp = int(m.group(1))
        if 0 <= temp <= 120:
            return f"heat treatment {temp}°C"
    # Handle "from_raw_milk" = unheated or "_0_" temperature
    if "raw_milk" in stem or re.search(r'[_\-]0[_\-]', stem):
        return "unheated"

    # Genetic: WT vs KO
    if re.search(r'(?:^|[_\-\.])ko(?:[_\-\.]|$)|knockout', stem):
        return "KO"
    if re.search(r'(?:^|[_\-\.])wt(?:[_\-\.]|$)|wildtype|wild[_\-]type', stem):
        return "WT"

    return ""


# ─── Main extraction pipeline ────────────────────────────────────────────────

def extract_pxd_metadata(
    pxd: str,
    pub: Dict,
    raw_files: List[str],
    sample_rows: pd.DataFrame,
    train_data: Dict[str, pd.DataFrame],
    train_pubs: Dict[str, Dict],
) -> pd.DataFrame:
    """Extract metadata for a single PXD. Precision-first approach."""

    n_rows = len(sample_rows)
    n_files = sample_rows["Raw Data File"].nunique()
    rows_per_file = n_rows // n_files if n_files > 0 else 1

    # ── Document-level extraction ───────────────────────────────────────────
    organism = extract_organism(pub)
    organism_part = extract_organism_part(pub)
    disease = extract_disease(pub)
    cell_line = extract_cell_line(pub)
    cell_type = extract_cell_type(pub)
    instrument = extract_instrument(pub)
    cleavage_agent = extract_cleavage_agent(pub)
    fragmentation = extract_fragmentation(pub)
    label_type, label_list = extract_label(pub)
    modifications = extract_modifications(pub)
    precursor_tol, fragment_tol = extract_mass_tolerance(pub)
    collision_energy = extract_collision_energy(pub)
    sex = extract_sex(pub)
    ms2_analyzer = extract_ms2_analyzer(pub)
    material_type = extract_material_type(pub)
    acquisition_method = extract_acquisition_method(pub, raw_files)
    separation = extract_separation(pub)
    fractionation_method = extract_fractionation_method(pub)
    enrichment_method = extract_enrichment_method(pub)
    developmental_stage = extract_developmental_stage(pub)
    missed_cleavages = extract_missed_cleavages(pub)
    alkylation_reagent = extract_alkylation_reagent(pub)
    reduction_reagent = extract_reduction_reagent(pub)
    ionization_type = extract_ionization_type(pub)

    # ── Nearest training PXDs (retrieval) ────────────────────────────────
    nearest = find_nearest_training_pxds(pub, train_pubs, train_data, top_k=3)

    result = sample_rows.copy()

    # ── Fill confident document-level fields ────────────────────────────
    confident_fills = {}
    if organism:
        confident_fills["Characteristics[Organism]"] = organism
    if organism_part:
        confident_fills["Characteristics[OrganismPart]"] = organism_part
    if disease:
        confident_fills["Characteristics[Disease]"] = disease
    if cell_line:
        confident_fills["Characteristics[CellLine]"] = cell_line
    if cell_type:
        confident_fills["Characteristics[CellType]"] = cell_type
    if instrument:
        confident_fills["Comment[Instrument]"] = instrument
    if cleavage_agent:
        confident_fills["Characteristics[CleavageAgent]"] = cleavage_agent
    if fragmentation:
        confident_fills["Comment[FragmentationMethod]"] = fragmentation
    if precursor_tol:
        confident_fills["Comment[PrecursorMassTolerance]"] = precursor_tol
    if fragment_tol:
        confident_fills["Comment[FragmentMassTolerance]"] = fragment_tol
    if collision_energy:
        confident_fills["Comment[CollisionEnergy]"] = collision_energy
    # Only fill sex when very confident (title mention)
    if sex and (re.search(r"\b(?:fe)?male\b", pub.get("TITLE", "").lower())):
        confident_fills["Characteristics[Sex]"] = sex
    if ms2_analyzer:
        confident_fills["Comment[MS2MassAnalyzer]"] = ms2_analyzer
    if material_type:
        confident_fills["Characteristics[MaterialType]"] = material_type
    if acquisition_method:
        confident_fills["Comment[AcquisitionMethod]"] = acquisition_method
    if separation:
        confident_fills["Comment[Separation]"] = separation
    if fractionation_method:
        confident_fills["Comment[FractionationMethod]"] = fractionation_method
    if enrichment_method:
        confident_fills["Comment[EnrichmentMethod]"] = enrichment_method
    if missed_cleavages:
        confident_fills["Comment[NumberOfMissedCleavages]"] = missed_cleavages
    if alkylation_reagent:
        confident_fills["Characteristics[AlkylationReagent]"] = alkylation_reagent
    if reduction_reagent:
        confident_fills["Characteristics[ReductionReagent]"] = reduction_reagent
    if ionization_type:
        confident_fills["Comment[IonizationType]"] = ionization_type

    for col, val in confident_fills.items():
        if col in result.columns:
            result[col] = val

    # ── Fill modifications (up to 2 slots) ────────────────────────────────
    mod_cols = sorted(
        [c for c in result.columns if c.startswith("Characteristics[Modification]")],
        key=lambda x: (x.count("."), x)
    )
    for i, mod in enumerate(modifications[:2]):
        if i < len(mod_cols):
            result[mod_cols[i]] = mod

    # ── Row-level handling: multiplexed or structured files ───────────────
    text_lower = get_full_text(pub).lower()
    is_apms = any(t in text_lower for t in [
        "affinity purification", "ap-ms", "pull-down", "pulldown",
        "immunoprecipitation", "flag ip", "flag-ip", "co-ip",
    ])
    _bait_norm = {
        "nsp31": "nsp3.1", "nsp32": "nsp3.2", "nsp33": "nsp3.3",
        "nsp11": "nsp1.1", "nsp12": "nsp1.2",
    }

    # Pre-scan all filenames to detect explicit baits (used to decide GFP default)
    def _extract_bait_from_stem(stem_lower: str) -> Optional[str]:
        if "nsp33" in stem_lower:
            return "nsp3.3"
        if "nsp31" in stem_lower:
            return "nsp3.1"
        if "nsp32" in stem_lower:
            return "nsp3.2"
        if "egfp" in stem_lower:
            return "EGFP"
        if re.search(r"(?<![a-z])gfp(?![a-z])", stem_lower):
            return "GFP"
        m = re.search(r"\b(orf\d+[a-z]?|nsp\d+)\b", stem_lower)
        if m:
            return _bait_norm.get(m.group(1), m.group(1))
        return None

    # Determine whether this PXD has explicit bait-named files:
    # only apply GFP default when the majority of files have recognizable baits
    unique_files = sample_rows["Raw Data File"].unique()
    n_files_with_explicit_bait = sum(
        1 for f in unique_files if _extract_bait_from_stem(Path(f).stem.lower()) is not None
    )
    use_gfp_default = (
        is_apms
        and n_files_with_explicit_bait >= max(2, len(unique_files) // 4)
        and re.search(r'\bgfp\b', text_lower)
    )

    # Track which columns have been set per-row
    has_per_row_frag = False
    has_per_row_acq = False
    has_per_row_treatment = False

    for raw_file in unique_files:
        mask = result["Raw Data File"] == raw_file
        n_rows_this = mask.sum()
        fname = Path(raw_file).stem
        fname_lower = fname.lower()

        # ── Fragmentation from filename (per-file, overrides doc-level) ──
        fname_frag = parse_filename_fragmentation(raw_file)
        if fname_frag and "Comment[FragmentationMethod]" in result.columns:
            result.loc[mask, "Comment[FragmentationMethod]"] = fname_frag
            has_per_row_frag = True

        # ── Acquisition method from filename ────────────────────────────
        fname_acq = parse_filename_acquisition(raw_file)
        if fname_acq and "Comment[AcquisitionMethod]" in result.columns:
            result.loc[mask, "Comment[AcquisitionMethod]"] = fname_acq
            has_per_row_acq = True

        # ── Biological replicate from filename ───────────────────────────
        br = parse_filename_bioreplicate(raw_file)
        if br and "Characteristics[BiologicalReplicate]" in result.columns:
            result.loc[mask, "Characteristics[BiologicalReplicate]"] = br

        # ── Treatment from filename ───────────────────────────────────────
        trt = parse_filename_treatment(raw_file, pub)
        if trt:
            if "Characteristics[Treatment]" in result.columns:
                result.loc[mask, "Characteristics[Treatment]"] = trt
                has_per_row_treatment = True
            if "FactorValue[Treatment]" in result.columns:
                result.loc[mask, "FactorValue[Treatment]"] = trt

        # ── Temperature from filename (CETSA/TPP) ───────────────────────
        temp_matches = re.findall(r"[-_](\d+)[Cc]", fname)
        file_temp = None
        if temp_matches:
            temp = int(temp_matches[-1])
            if 0 <= temp <= 120:
                file_temp = str(temp)

        # ── Bait from filename (AP-MS) ───────────────────────────────────
        file_bait = None
        if is_apms:
            file_bait = _extract_bait_from_stem(fname_lower)
            if file_bait is None and use_gfp_default:
                # AP-MS experiment where most files have recognized baits but this
                # file doesn't → it's likely the GFP negative control run
                file_bait = "GFP"

        # ── Assign per-row values within this file ────────────────────────
        if rows_per_file > 1 and label_type != "label free sample":
            # Multiplexed: cycle through label list
            for i in range(n_rows_this):
                row_idx = result.index[mask][i]
                if i < len(label_list):
                    result.at[row_idx, "Characteristics[Label]"] = label_list[i]
                if file_temp is not None:
                    result.at[row_idx, "Characteristics[Temperature]"] = file_temp
                    if "FactorValue[Temperature]" in result.columns:
                        result.at[row_idx, "FactorValue[Temperature]"] = file_temp
                if file_bait is not None:
                    result.at[row_idx, "Characteristics[Bait]"] = file_bait
                    if "FactorValue[Bait]" in result.columns:
                        result.at[row_idx, "FactorValue[Bait]"] = file_bait
        else:
            # Single-row per file
            if label_type == "label free sample" or rows_per_file <= 1:
                result.loc[mask, "Characteristics[Label]"] = "label free sample"
            else:
                result.loc[mask, "Characteristics[Label]"] = label_list[0] if label_list else "label free sample"

            if file_temp is not None:
                result.loc[mask, "Characteristics[Temperature]"] = file_temp
                if "FactorValue[Temperature]" in result.columns:
                    result.loc[mask, "FactorValue[Temperature]"] = file_temp

            if file_bait is not None:
                result.loc[mask, "Characteristics[Bait]"] = file_bait
                if "FactorValue[Bait]" in result.columns:
                    result.loc[mask, "FactorValue[Bait]"] = file_bait

            # (bait already handled above via _extract_bait_from_stem)

    # ── Fraction identifiers from filenames ───────────────────────────────
    for idx, row in result.iterrows():
        fname = str(row["Raw Data File"])
        fname_lower = fname.lower()

        frac_id = None
        frac_patterns = [
            r"[_\-\s]FR[-_]?(\d+)",
            r"[_\-\s]F(\d+)(?:\.|_|$)",
            r"[_\-]frac(?:tion)?[-_]?(\d+)",
            r"(?:^|[_\-])frac(?:tion)?[-_]?(\d+)",
            r"Fr(\d+)(?:\.|_|$)",
            r"fraction[-_]?(\d+)",
            # Band/gel slice patterns (e.g. B34 for SDS-PAGE fractions)
            r"[_\-]B(\d+)(?:\.|_|$)",
        ]
        for pat in frac_patterns:
            m = re.search(pat, fname, re.IGNORECASE)
            if m:
                frac_id = str(int(m.group(1)))
                break

        # If fractionation detected and file ends with _N or -N, use as fraction
        if not frac_id and fractionation_method and fractionation_method != "no fractionation":
            m = re.search(r'[_\-](\d{1,3})(?:\.\w+)?$', fname)
            if m:
                val = int(m.group(1))
                if 1 <= val <= 200:  # reasonable fraction range
                    frac_id = str(val)

        if frac_id:
            result.at[idx, "Comment[FractionIdentifier]"] = frac_id

    # ── Conservative defaults ─────────────────────────────────────────────
    # Only fill defaults that are SAFE - wrong values create F1=0 evaluated pairs
    defaults = {
        "Characteristics[BiologicalReplicate]": "1",
        "Characteristics[NumberOfTechnicalReplicates]": "1",
        "Comment[FractionIdentifier]": "1",
        # DevelopmentalStage: only fill when explicitly extracted, else leave empty
    }
    if developmental_stage:
        defaults["Characteristics[DevelopmentalStage]"] = developmental_stage

    for col, val in defaults.items():
        if col in result.columns:
            current = result[col].replace("Text Span", "").replace("", np.nan).dropna()
            if len(current) == 0 and val:
                result[col] = val

    # ── Disease / OrganismPart fills (only when extractor found a real value) ──
    for col, extracted_val in [
        ("Characteristics[Disease]", disease),
        ("Characteristics[OrganismPart]", organism_part),
    ]:
        if col in result.columns and extracted_val and extracted_val != "not available":
            current = result[col].replace("Text Span", "").replace("", np.nan).dropna()
            if len(current) == 0:
                result[col] = extracted_val

    # ── Nearest-neighbour fill for empty technical columns ────────────────
    if nearest:
        best_pxd, best_score = nearest[0]
        if best_score > 0.35 and best_pxd in train_data:
            ref_df = train_data[best_pxd]
            safe_cols = [
                "Characteristics[Organism]",
                "Comment[Instrument]", "Characteristics[CleavageAgent]",
                "Comment[FragmentationMethod]",
                "Comment[PrecursorMassTolerance]", "Comment[FragmentMassTolerance]",
                "Comment[CollisionEnergy]",
            ]
            for col in safe_cols:
                if col not in result.columns or col not in ref_df.columns:
                    continue
                current_vals = result[col].replace("Text Span", "").replace("", np.nan).dropna().unique()
                if len(current_vals) == 0:
                    ref_vals = ref_df[col].dropna().astype(str).unique()
                    ref_vals = [v for v in ref_vals if v.lower() not in ("not applicable", "not available", "nan", "")]
                    if ref_vals:
                        result[col] = ref_vals[0]

    # ── FactorValue[Disease]: DO NOT propagate from Characteristics[Disease].
    # Gold FactorValue[Disease] values are study-specific experimental groups
    # (e.g., "malignant"/"benign", "control"/"treatment", "early braak stage")
    # that can't be predicted from text. Wrong predictions create F1=0 pairs.

    # FactorValue[FractionIdentifier] ← Comment[FractionIdentifier]
    if "FactorValue[FractionIdentifier]" in result.columns and "Comment[FractionIdentifier]" in result.columns:
        frac_vals = result["Comment[FractionIdentifier]"].replace("Text Span", "").replace("", np.nan)
        fv_frac = result["FactorValue[FractionIdentifier]"].replace("Text Span", "").replace("", np.nan)
        empty_frac = fv_frac.isna()
        if empty_frac.any():
            result.loc[empty_frac, "FactorValue[FractionIdentifier]"] = frac_vals[empty_frac]

    # Replace "Text Span" placeholder
    result = result.replace("Text Span", "")

    return result


# ─── Validation ───────────────────────────────────────────────────────────────

def _build_val_split(
    pxd: str,
    train_data: Dict,
    train_pubs: Dict,
    train_subset: Dict,
    train_pubs_subset: Dict,
    ss_cols: List[str],
):
    """Return (sol_df, pred_df) for a single PXD, or (None, None) on failure."""
    if pxd not in train_pubs or pxd not in train_data:
        return None, None
    pub = train_pubs[pxd]
    gold_df = train_data[pxd]
    if "Raw Data File" not in gold_df.columns:
        return None, None

    n_rows = len(gold_df)
    template_rows = []
    for i in range(n_rows):
        row = {"ID": i + 1, "PXD": pxd}
        row["Raw Data File"] = gold_df.iloc[i].get("Raw Data File", "")
        for col in ss_cols:
            if col not in row:
                row[col] = "Text Span"
        template_rows.append(row)
    template_df = pd.DataFrame(template_rows)[ss_cols]
    raw_files = gold_df["Raw Data File"].dropna().unique().tolist()

    pred_df = extract_pxd_metadata(
        pxd, pub, raw_files, template_df, train_subset, train_pubs_subset
    )

    pred_df = pred_df.replace("Text Span", np.nan)
    pred_df = pred_df.replace("", np.nan)
    pred_df = pred_df.replace(r'^\s*$', np.nan, regex=True)
    pred_df = pred_df.fillna("Not Applicable")

    # Build solution (gold standard)
    sol_rows = []
    for i in range(n_rows):
        row = {"ID": i + 1, "PXD": pxd}
        for col in ss_cols:
            if col in gold_df.columns:
                val = gold_df.iloc[i].get(col, "")
                row[col] = val if pd.notna(val) and str(val).strip() != "" else "Not Applicable"
            elif col not in row:
                row[col] = "Not Applicable"
        sol_rows.append(row)
    sol_df = pd.DataFrame(sol_rows)[ss_cols]
    return sol_df, pred_df


def run_local_validation(fold: Optional[int] = None) -> float:
    """Run robust 5-fold grouped cross-validation over all training PXDs."""
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))
    from Scoring import score as compute_score

    print("Loading training data...")
    train_data = load_training_data()
    train_pubs = {}
    for f in sorted(TRAIN_PUB_DIR.glob("*_PubText.json")):
        pxd = f.stem.replace("_PubText", "")
        if pxd == "PubText":
            continue
        train_pubs[pxd] = load_pub_json(str(f))

    all_pxds = sorted(set(train_data.keys()) & set(train_pubs.keys()))
    n_folds = 5
    ss = pd.read_csv(SAMPLE_SUB, nrows=0)
    ss_cols = list(ss.columns)

    fold_assignments = {p: i % n_folds for i, p in enumerate(all_pxds)}

    folds_to_run = [fold] if fold is not None else list(range(n_folds))
    fold_scores: List[float] = []
    all_eval_rows: List[pd.DataFrame] = []

    for fold_i in folds_to_run:
        val_pxds = [p for p in all_pxds if fold_assignments[p] == fold_i]
        train_pxds = [p for p in all_pxds if fold_assignments[p] != fold_i]

        train_subset = {p: train_data[p] for p in train_pxds}
        train_pubs_subset = {p: train_pubs[p] for p in train_pxds}

        fold_sols, fold_preds = [], []
        for pxd in val_pxds:
            sol_df, pred_df = _build_val_split(
                pxd, train_data, train_pubs,
                train_subset, train_pubs_subset, ss_cols
            )
            if sol_df is not None:
                fold_sols.append(sol_df)
                fold_preds.append(pred_df)

        if not fold_sols:
            print(f"Fold {fold_i}: no data")
            continue

        solution = pd.concat(fold_sols, ignore_index=True)
        prediction = pd.concat(fold_preds, ignore_index=True)
        for col in ss_cols:
            if col not in solution.columns:
                solution[col] = "Not Applicable"
            if col not in prediction.columns:
                prediction[col] = "Not Applicable"
        solution = solution[ss_cols]
        prediction = prediction[ss_cols]
        solution["ID"] = range(1, len(solution) + 1)
        prediction["ID"] = range(1, len(prediction) + 1)

        eval_df, fold_score = compute_score(solution, prediction, "ID")
        fold_scores.append(fold_score)
        all_eval_rows.append(eval_df)
        print(f"Fold {fold_i} ({len(val_pxds)} PXDs): {fold_score:.4f}  val_pxds={val_pxds[:3]}...")

    if not fold_scores:
        print("No validation data available")
        return 0.0

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    print(f"\n{'='*60}")
    print(f"5-Fold CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Per-fold: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"{'='*60}")

    if all_eval_rows:
        full_eval = pd.concat(all_eval_rows, ignore_index=True)

        # ── Per-field F1 ──────────────────────────────────────────────────
        col_scores = full_eval.groupby("AnnotationType")["f1"].mean().sort_values(ascending=False)
        col_counts = full_eval.groupby("AnnotationType")["f1"].count()
        print(f"\nPer-field F1 (mean across folds, n=evaluated pairs):")
        for col, f1 in col_scores.items():
            print(f"  {col}: {f1:.4f}  (n={col_counts[col]})")

        # ── Worst PXDs ────────────────────────────────────────────────────
        pxd_scores = full_eval.groupby("pxd")["f1"].mean().sort_values()
        print(f"\nWorst PXDs:")
        for p, f1 in pxd_scores.head(10).items():
            print(f"  {p}: {f1:.4f}")
        print(f"\nBest PXDs:")
        for p, f1 in pxd_scores.tail(5).items():
            print(f"  {p}: {f1:.4f}")

        # ── NA masking coverage ───────────────────────────────────────────
        # Columns in submission schema that were NEVER evaluated (likely 100% NA in gold)
        all_sub_cols = [c for c in pd.read_csv(SAMPLE_SUB, nrows=0).columns
                        if c not in ("ID", "PXD", "Raw Data File", "Usage")]
        evaluated_cols = set(full_eval["AnnotationType"].unique())
        unevaluated = sorted(set(all_sub_cols) - evaluated_cols)
        print(f"\nSubmission columns never evaluated (likely all-NA in gold or mapping gap):")
        print(f"  [{len(unevaluated)}] {unevaluated[:20]}")

        full_eval.to_csv(BASE_DIR / "validation_metrics.csv", index=False)
        print(f"\nDetailed metrics saved to validation_metrics.csv")

        # ── Save field-level summary CSV ──────────────────────────────────
        field_summary = pd.DataFrame({
            "field": col_scores.index,
            "mean_f1": col_scores.values,
            "n_evaluated_pairs": col_counts[col_scores.index].values,
        })
        field_summary.to_csv(BASE_DIR / "field_f1_summary.csv", index=False)
        print(f"Field F1 summary saved to field_f1_summary.csv")

        # ── Save fold summary ─────────────────────────────────────────────
        summary_rows = []
        for fi, (fs, ev) in enumerate(zip(fold_scores, all_eval_rows)):
            summary_rows.append({"fold": fi, "score": fs, "n_pxds": len(ev["pxd"].unique())})
        summary_df = pd.DataFrame(summary_rows)
        summary_df.loc[len(summary_df)] = {"fold": "mean", "score": mean_score, "n_pxds": ""}
        summary_df.loc[len(summary_df)] = {"fold": "std", "score": std_score, "n_pxds": ""}
        summary_df.to_csv(BASE_DIR / "validation_summary.csv", index=False)
        print(f"Summary saved to validation_summary.csv")

    return mean_score


# ─── Generate submission ──────────────────────────────────────────────────────

def generate_submission():
    """Generate the final submission.csv."""
    print("Loading training data...")
    train_data = load_training_data()

    train_pubs = {}
    for f in sorted(TRAIN_PUB_DIR.glob("*_PubText.json")):
        pxd = f.stem.replace("_PubText", "")
        if pxd == "PubText":
            continue
        train_pubs[pxd] = load_pub_json(str(f))

    sample_sub = pd.read_csv(SAMPLE_SUB)
    ss_cols = list(sample_sub.columns)

    pxd_order = []
    for pxd in sample_sub["PXD"]:
        if pxd not in pxd_order:
            pxd_order.append(pxd)

    all_results = []

    for pxd in pxd_order:
        print(f"Processing {pxd}...")

        json_path = TEST_PUB_DIR / f"{pxd}_PubText.json"
        if not json_path.exists():
            print(f"  WARNING: No JSON for {pxd}")
            continue

        pub = load_pub_json(str(json_path))
        pxd_rows = sample_sub[sample_sub["PXD"] == pxd].copy()
        raw_files = pub.get("Raw Data Files", [])

        result = extract_pxd_metadata(
            pxd, pub, raw_files, pxd_rows,
            train_data, train_pubs
        )

        all_results.append(result)

    submission = pd.concat(all_results, ignore_index=True)

    for col in ss_cols:
        if col not in submission.columns:
            submission[col] = ""
    submission = submission[ss_cols]

    assert list(submission["PXD"]) == list(sample_sub["PXD"]), "PXD alignment mismatch after concat!"
    submission["ID"] = sample_sub["ID"].values
    submission["Raw Data File"] = sample_sub["Raw Data File"].values

    # ── Clean up and fill policy ────────────────────────────────────────────
    # Per-row columns: never broadcast mode; fill remaining NaN with Not Applicable
    PER_ROW_COLS = {
        "Characteristics[Label]",
        "Characteristics[Bait]",
        "FactorValue[Bait]",
        "Characteristics[Temperature]",
        "FactorValue[Temperature]",
        "Comment[FractionIdentifier]",
        "FactorValue[FractionIdentifier]",
        "Characteristics[BiologicalReplicate]",
        "Characteristics[Treatment]",
        "FactorValue[Treatment]",
        "Comment[FragmentationMethod]",  # can vary per file (e.g. CID vs HCD)
        "Comment[AcquisitionMethod]",    # can vary per file (e.g. DDA vs DIA)
    }

    submission = submission.replace("Text Span", np.nan)
    submission = submission.replace("", np.nan)
    submission = submission.replace(r'^\s*$', np.nan, regex=True)

    for pxd in submission["PXD"].unique():
        mask = submission["PXD"] == pxd
        for col in ss_cols[3:]:  # skip ID, PXD, Raw Data File
            col_vals = submission.loc[mask, col].dropna().unique()
            real_vals = [v for v in col_vals if str(v).strip() not in ("", "Not Applicable")]
            if len(real_vals) == 0:
                # No real values → Not Applicable (scorer will skip)
                submission.loc[mask, col] = "Not Applicable"
            elif col in PER_ROW_COLS:
                # Per-row column: fill NaN with Not Applicable, never broadcast
                submission.loc[mask, col] = submission.loc[mask, col].fillna("Not Applicable")
            else:
                # Document-level column: broadcast mode to remaining NaN cells
                real_series = submission.loc[mask, col].dropna()
                real_series = real_series[real_series.astype(str).str.strip() != ""]
                fill_val = real_series.mode().iloc[0] if len(real_series) > 0 else real_vals[0]
                submission.loc[mask, col] = submission.loc[mask, col].fillna(fill_val)

    submission = submission.fillna("Not Applicable")

    if "Usage" in submission.columns:
        submission["Usage"] = "Raw Data File"

    assert len(submission) == len(sample_sub), f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
    assert list(submission.columns) == ss_cols, "Column mismatch"
    assert list(submission["ID"]) == list(sample_sub["ID"]), "ID mismatch"
    assert list(submission["PXD"]) == list(sample_sub["PXD"]), "PXD mismatch"
    assert list(submission["Raw Data File"]) == list(sample_sub["Raw Data File"]), "Raw Data File mismatch"

    output_path = BASE_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Shape: {submission.shape}")

    # ── Post-generation audit ──────────────────────────────────────────────
    print("\n=== Post-generation audit ===")
    audit_rows = []
    for pxd in sorted(submission["PXD"].unique()):
        rows = submission[submission["PXD"] == pxd]
        na_pct = (rows == "Not Applicable").sum().sum() / rows.size * 100
        na_val_pct = rows.isin(["not available", "Not Applicable"]).sum().sum() / rows.size * 100
        real_cols = [c for c in ss_cols[3:] if not all(rows[c].isin(["Not Applicable", "not available"]))]
        # Check for suspicious always-same per-row columns
        suspicious = [c for c in real_cols if c in PER_ROW_COLS and rows[c].nunique() == 1]
        print(f"  {pxd}: {len(rows)} rows | {na_pct:.0f}% N/A | {len(real_cols)} real cols | "
              f"suspicious_const: {suspicious}")
        audit_rows.append({
            "PXD": pxd,
            "n_rows": len(rows),
            "pct_not_applicable": round(na_pct, 1),
            "pct_filler": round(na_val_pct, 1),
            "n_real_cols": len(real_cols),
            "real_cols_sample": str(real_cols[:8]),
            "suspicious_constant_per_row_cols": str(suspicious),
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(BASE_DIR / "submission_audit.csv", index=False)
    print(f"\nSubmission audit saved to submission_audit.csv")

    # ── Zero-null sanity check ─────────────────────────────────────────────
    null_count = pd.read_csv(BASE_DIR / "submission.csv").isnull().sum().sum()
    print(f"Post-read null count: {null_count} (must be 0)")
    assert null_count == 0, f"CRITICAL: {null_count} nulls found in submission.csv!"

    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDRF extraction pipeline")
    parser.add_argument("--validate", action="store_true", help="Run local validation")
    parser.add_argument("--fold", type=int, default=None, help="Validation fold (0-4)")
    args = parser.parse_args()

    if args.validate:
        run_local_validation(args.fold)
    else:
        generate_submission()
