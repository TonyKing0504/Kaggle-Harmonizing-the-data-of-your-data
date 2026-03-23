"""Central configuration for the clean pipeline."""
import os
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TRAIN_SDRF_DIR = DATA_DIR / "TrainingSDRFs"
TRAIN_TEXT_DIR = DATA_DIR / "TrainingPubText"
TEST_TEXT_DIR = DATA_DIR / "TestPubText"
SAMPLE_SUB = DATA_DIR / "SampleSubmission.csv"
SCORER_PATH = REPO_ROOT / "src" / "Scoring.py"

# Pipeline output
PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PIPELINE_DIR / "output"
CACHE_DIR = PIPELINE_DIR / "cache"
ARTIFACTS_DIR = PIPELINE_DIR / "artifacts"

for d in [OUTPUT_DIR, CACHE_DIR, ARTIFACTS_DIR]:
    d.mkdir(exist_ok=True)

# Submission columns (exact order from SampleSubmission.csv)
SUBMISSION_COLUMNS = [
    'ID', 'PXD', 'Raw Data File',
    'Characteristics[Age]', 'Characteristics[AlkylationReagent]',
    'Characteristics[AnatomicSiteTumor]', 'Characteristics[AncestryCategory]',
    'Characteristics[BMI]', 'Characteristics[Bait]',
    'Characteristics[BiologicalReplicate]', 'Characteristics[CellLine]',
    'Characteristics[CellPart]', 'Characteristics[CellType]',
    'Characteristics[CleavageAgent]', 'Characteristics[Compound]',
    'Characteristics[ConcentrationOfCompound]', 'Characteristics[Depletion]',
    'Characteristics[DevelopmentalStage]', 'Characteristics[DiseaseTreatment]',
    'Characteristics[Disease]', 'Characteristics[GeneticModification]',
    'Characteristics[Genotype]', 'Characteristics[GrowthRate]',
    'Characteristics[Label]', 'Characteristics[MaterialType]',
    'Characteristics[Modification]', 'Characteristics[Modification].1',
    'Characteristics[Modification].2', 'Characteristics[Modification].3',
    'Characteristics[Modification].4', 'Characteristics[Modification].5',
    'Characteristics[Modification].6',
    'Characteristics[NumberOfBiologicalReplicates]',
    'Characteristics[NumberOfSamples]',
    'Characteristics[NumberOfTechnicalReplicates]',
    'Characteristics[OrganismPart]', 'Characteristics[Organism]',
    'Characteristics[OriginSiteDisease]', 'Characteristics[PooledSample]',
    'Characteristics[ReductionReagent]', 'Characteristics[SamplingTime]',
    'Characteristics[Sex]', 'Characteristics[Specimen]',
    'Characteristics[SpikedCompound]', 'Characteristics[Staining]',
    'Characteristics[Strain]', 'Characteristics[SyntheticPeptide]',
    'Characteristics[Temperature]', 'Characteristics[Time]',
    'Characteristics[Treatment]', 'Characteristics[TumorCellularity]',
    'Characteristics[TumorGrade]', 'Characteristics[TumorSite]',
    'Characteristics[TumorSize]', 'Characteristics[TumorStage]',
    'Comment[AcquisitionMethod]', 'Comment[CollisionEnergy]',
    'Comment[EnrichmentMethod]', 'Comment[FlowRateChromatogram]',
    'Comment[FractionIdentifier]', 'Comment[FractionationMethod]',
    'Comment[FragmentMassTolerance]', 'Comment[FragmentationMethod]',
    'Comment[GradientTime]', 'Comment[Instrument]',
    'Comment[IonizationType]', 'Comment[MS2MassAnalyzer]',
    'Comment[NumberOfFractions]', 'Comment[NumberOfMissedCleavages]',
    'Comment[PrecursorMassTolerance]', 'Comment[Separation]',
    'FactorValue[Bait]', 'FactorValue[CellPart]',
    'FactorValue[Compound]', 'FactorValue[ConcentrationOfCompound].1',
    'FactorValue[Disease]', 'FactorValue[FractionIdentifier]',
    'FactorValue[GeneticModification]', 'FactorValue[Temperature]',
    'FactorValue[Treatment]', 'Usage',
]

# Metadata columns (exclude ID, PXD, Raw Data File, Usage)
META_COLUMNS = [c for c in SUBMISSION_COLUMNS if c not in
                ('ID', 'PXD', 'Raw Data File', 'Usage')]

# Paper-level columns (same value for all rows in a PXD)
PAPER_LEVEL_COLUMNS = [
    'Characteristics[Organism]', 'Characteristics[CleavageAgent]',
    'Characteristics[AlkylationReagent]', 'Characteristics[ReductionReagent]',
    'Characteristics[MaterialType]', 'Characteristics[Disease]',
    'Characteristics[Specimen]',
    'Characteristics[Modification]', 'Characteristics[Modification].1',
    'Characteristics[Modification].2', 'Characteristics[Modification].3',
    'Characteristics[Modification].4', 'Characteristics[Modification].5',
    'Characteristics[Modification].6',
    'Characteristics[NumberOfBiologicalReplicates]',
    'Characteristics[NumberOfSamples]',
    'Characteristics[NumberOfTechnicalReplicates]',
    'Comment[AcquisitionMethod]', 'Comment[CollisionEnergy]',
    'Comment[EnrichmentMethod]', 'Comment[FlowRateChromatogram]',
    'Comment[FractionationMethod]', 'Comment[FragmentMassTolerance]',
    'Comment[FragmentationMethod]', 'Comment[GradientTime]',
    'Comment[Instrument]', 'Comment[IonizationType]',
    'Comment[MS2MassAnalyzer]', 'Comment[NumberOfFractions]',
    'Comment[NumberOfMissedCleavages]', 'Comment[PrecursorMassTolerance]',
    'Comment[Separation]',
]

# Default fill value
DEFAULT_FILL = "Not Applicable"

# String similarity threshold (matching scorer)
SIMILARITY_THRESHOLD = 0.80
