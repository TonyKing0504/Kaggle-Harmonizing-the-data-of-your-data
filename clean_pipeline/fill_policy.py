"""Fill policy: smart defaults and fallback logic for unfilled columns."""
from typing import Dict, List, Tuple, Optional
from config import DEFAULT_FILL, META_COLUMNS


# Columns where "Not Applicable" is genuinely the right answer in most cases
SAFE_NA_COLUMNS = {
    'Characteristics[Age]', 'Characteristics[AnatomicSiteTumor]',
    'Characteristics[AncestryCategory]', 'Characteristics[BMI]',
    'Characteristics[Bait]',
    'Characteristics[CellPart]',
    'Characteristics[Compound]', 'Characteristics[ConcentrationOfCompound]',
    'Characteristics[Depletion]', 'Characteristics[DevelopmentalStage]',
    'Characteristics[DiseaseTreatment]',
    'Characteristics[GeneticModification]', 'Characteristics[Genotype]',
    'Characteristics[GrowthRate]',
    'Characteristics[OriginSiteDisease]',
    'Characteristics[PooledSample]',
    'Characteristics[SamplingTime]',
    'Characteristics[SpikedCompound]', 'Characteristics[Staining]',
    'Characteristics[Strain]', 'Characteristics[SyntheticPeptide]',
    'Characteristics[Temperature]', 'Characteristics[Time]',
    'Characteristics[TumorCellularity]', 'Characteristics[TumorGrade]',
    'Characteristics[TumorSite]', 'Characteristics[TumorSize]',
    'Characteristics[TumorStage]',
    'Comment[CollisionEnergy]',
    'Comment[FlowRateChromatogram]',
    'Comment[GradientTime]',
    'Comment[IonizationType]',
    'FactorValue[Bait]', 'FactorValue[CellPart]',
    'FactorValue[Compound]', 'FactorValue[ConcentrationOfCompound].1',
    'FactorValue[Disease]', 'FactorValue[FractionIdentifier]',
    'FactorValue[GeneticModification]', 'FactorValue[Temperature]',
    'FactorValue[Treatment]',
}

# Columns where we should try harder to fill with real values
HIGH_VALUE_COLUMNS = {
    'Characteristics[Organism]',
    'Characteristics[CellLine]',
    'Characteristics[CellType]',
    'Characteristics[CleavageAgent]',
    'Characteristics[Disease]',
    'Characteristics[Label]',
    'Characteristics[MaterialType]',
    'Characteristics[Modification]',
    'Characteristics[Modification].1',
    'Characteristics[OrganismPart]',
    'Characteristics[Sex]',
    'Characteristics[Specimen]',
    'Comment[AcquisitionMethod]',
    'Comment[EnrichmentMethod]',
    'Comment[FractionIdentifier]',
    'Comment[FractionationMethod]',
    'Comment[FragmentMassTolerance]',
    'Comment[FragmentationMethod]',
    'Comment[Instrument]',
    'Comment[MS2MassAnalyzer]',
    'Comment[NumberOfFractions]',
    'Comment[NumberOfMissedCleavages]',
    'Comment[PrecursorMassTolerance]',
    'Comment[Separation]',
    'Characteristics[BiologicalReplicate]',
    'Characteristics[NumberOfBiologicalReplicates]',
    'Characteristics[NumberOfSamples]',
    'Characteristics[NumberOfTechnicalReplicates]',
    'Characteristics[AlkylationReagent]',
    'Characteristics[ReductionReagent]',
    'Characteristics[Treatment]',
}

# Family-specific defaults
FAMILY_DEFAULTS = {
    'label_free_simple': {
        'Characteristics[Label]': 'AC=MS:1002038;NT=label free sample',
    },
    'label_free_fractionated': {
        'Characteristics[Label]': 'AC=MS:1002038;NT=label free sample',
    },
    'tmt_multiplexed': {},
    'clinical_cohort': {
        'Characteristics[MaterialType]': 'tissue',
    },
    'cell_line': {
        'Characteristics[MaterialType]': 'cell',
    },
    'ap_ms': {
        'Characteristics[MaterialType]': 'cell',
    },
}

# Global safe defaults for columns that are almost always the same
GLOBAL_DEFAULTS = {
    'Characteristics[CleavageAgent]': 'AC=MS:1001251;NT=Trypsin',
    'Comment[NumberOfMissedCleavages]': '2',
    'Characteristics[NumberOfTechnicalReplicates]': '1',
}


def apply_fill_policy(row: Dict, column: str,
                      family: str,
                      retrieval_candidates: List[Tuple[str, float]] = None) -> str:
    """
    Determine the best fill value for a column.

    Priority:
    1. Already filled with real evidence -> keep
    2. Retrieval-backed candidate -> use if high weight
    3. Family-specific default -> use
    4. Global default -> use
    5. "Not Applicable"
    """
    current = row.get(column, None)

    # Already filled with real value
    if current and current not in ('Text Span', 'Not Applicable', None, '', 'nan'):
        return current

    # Retrieval candidates
    if retrieval_candidates:
        for val, weight in retrieval_candidates:
            if weight > 2.0:  # reasonable confidence from similar PXDs
                return val

    # Family-specific default
    family_defs = FAMILY_DEFAULTS.get(family, {})
    if column in family_defs:
        return family_defs[column]

    # Global defaults
    if column in GLOBAL_DEFAULTS:
        return GLOBAL_DEFAULTS[column]

    # Final fallback
    return DEFAULT_FILL


def apply_fill_policy_to_rows(rows: List[Dict], family: str,
                              retrieval_results=None) -> List[Dict]:
    """Apply fill policy to all rows, tracking fill stats."""
    from config import META_COLUMNS
    fill_stats = {'real': 0, 'retrieval': 0, 'family_default': 0,
                  'global_default': 0, 'na': 0}

    for r in rows:
        for col in META_COLUMNS:
            current = r.get(col)
            if current and current not in ('Text Span', 'Not Applicable', None, '', 'nan'):
                fill_stats['real'] += 1
                continue

            # Don't use retrieval in fill_policy - it's done earlier for safe columns
            filled = apply_fill_policy(r, col, family, None)
            r[col] = filled

            if filled == DEFAULT_FILL:
                fill_stats['na'] += 1
            else:
                fill_stats['real'] += 1

    return rows
