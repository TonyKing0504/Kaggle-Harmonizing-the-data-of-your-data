# Context Handoff — V5 Pipeline Improvements

## Current Metrics
- **Local 5-Fold CV**: 0.7223 ± 0.0216 (was 0.7317 with V4)
- **Online score**: 0.17957 (V4) — new submission not yet tested online
- **Submission**: 1659 rows, 81 columns, 0 nulls, all constraints met

## What Changed (V4 → V5)
### Bug fixes that should improve online score:
1. **Cleavage agent false positive**: Fixed `"gluc" in methods` matching "glucose" → now uses word-boundary regex. Affected: PXD061090 (was "Glutamyl endopeptidase", now "Trypsin")
2. **Cell line detection**: Now uses title/abstract primarily. Methods-only cell line mentions blocked when tissue indicators present in filenames/title. Fixed false HEK293T for PXD025663 (brain tissue) and PXD061285 (mouse brain)
3. **Disease detection**: Cell-line-inferred diseases now only from title/abstract. Fixed false "adenocarcinoma" for PXD061285.
4. **TMT label for fractionated experiments**: When TMT detected but 1 row per file, label is now TMT plex type (not "label free sample"). Fixed PXD004010.
5. **Organism part**: Fixed false "lung" for PXD062877 (now "bone marrow" via BMDM detection). Added bone marrow, removed lung from filename organ map.
6. **Material type**: Tissue/cell line inference now from title/abstract only.
7. **FactorValue[Disease]**: Removed auto-propagation from Characteristics[Disease] (was F1=0.0 in CV).
8. **Modification ordering**: Fixed Carbamidomethyl as default first modification (62% of training gold). Added Carbamidomethyl default when no specific alkylation detected. Fixed Carbamyl false positive from "urea" in buffer.
9. **Treatment from filenames**: Added LIPUS, sham, mock, distal patterns.
10. **GeneticModification from filenames**: Added KO/WT detection → gene knockout/wild type fills.

### Why local CV dropped slightly (0.73 → 0.72):
- Restricting cell line/disease/material type to title/abstract loses some correct training detections that happened to work from methods
- But prevents critical false positives on test data (which drove online score down)

## Files Changed
- `pipeline.py` — all changes in the main pipeline file

## What Improved vs What Regressed
- **Improved**: False positive prevention (cell line, disease, material type, cleavage agent), TMT label handling, modification ordering, FactorValue[Disease] removal
- **Regressed**: ~1% local CV drop (acceptable tradeoff for test accuracy)

## Key Per-PXD Status
| PXD | Rows | Key Fix | Status |
|-----|------|---------|--------|
| PXD004010 | 10 | Label: TMT10plex (was label free) | Fixed |
| PXD025663 | 12 | Removed false HEK293T, MaterialType=tissue | Fixed |
| PXD061285 | 60 | Removed false HEK-293/adenocarcinoma | Fixed |
| PXD062877 | 48 | OrganismPart=bone marrow (was lung) | Fixed |
| PXD061090 | 6 | CleavageAgent=Trypsin (was Glu endopeptidase) | Fixed |
| PXD019519 | 6 | CellLine=HeLa restored via methods | Fixed |
| PXD061195 | 1376 | Largest PXD, TMT+AP-MS, looks reasonable | OK |
| PXD050621 | 9 | Still missing instrument/fragmentation | Gap |
| PXD061090 | 6 | Still missing instrument/fragmentation | Gap |

## Next 3 Actions
1. **Submit online** and measure score improvement from V4's 0.17957
2. **Improve sparse PXDs**: PXD050621, PXD061090 still missing instrument/fragmentation — check if these are in the test gold
3. **Better per-PXD retrieval**: Use nearest training PXD values more aggressively for empty technical columns (currently threshold=0.35 is too conservative)

## Risky Assumptions
- Material type for PXD040582 is "cell" but U2OS detection may be wrong (study uses MRC5 for HCMV)
- PXD062877 material type "lysate" vs "cell" — gold could go either way
- Default Carbamidomethyl modification may hurt for unusual proteomics workflows
- PXD004010 disease "Alzheimer's disease" — paper is a pipeline paper, not purely AD-focused

## Commands to Regenerate
```bash
# Validate
python pipeline.py --validate

# Generate submission
python pipeline.py

# Check submission
python -c "import pandas as pd; s=pd.read_csv('submission.csv'); print(s.shape, s.isnull().sum().sum())"
```
