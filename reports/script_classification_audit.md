# Script Classification Audit

## 1. Required Final Pipeline
These scripts are necessary to reproduce the final selected model predictions and validate them.
* `src/phase7_final_predictions.py` (Generates the final test predictions)
* `src/validate_predictions.py` (Validates the final test predictions)

## 2. Evaluation/Analysis Utility
These scripts perform offline error analysis, metric generation, or pipeline correctness checks on completed results.
* `src/phase6_error_analysis.py` (Generates error analysis artifacts)
* `src/phase8_documentation.py` (Generates the final markdown documentation and results CSVs)
* `src/phase4_pre_audit.py` (Identified duplicates across folds)
* `src/phase4_verification.py` (Verified Phase 4 numbers and folds)

## 3. Historical Experiment
These scripts contain earlier modeling phases and discarded approaches (baseline, intermediate models, fusions).
* `src/phase1_analysis.py`
* `src/phase2_baseline.py`
* `src/phase3_lexical.py`
* `src/phase4_learning.py`
* `src/phase5_fusion.py`

## 4. Temporary/Debug/Audit Script
* `scratch_audit.py` (Created during Phase 9 to check outputs and data integrity; to be removed before push).
