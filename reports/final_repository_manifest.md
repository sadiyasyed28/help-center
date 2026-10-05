# Final Repository Manifest

This manifest categorizes the files in the current local experiment directory for a future Git repository push.

## KEEP (Core Project Files)

These files constitute the core reproducible ML pipeline, testing suite, final predictions, and final documentation.

*   `src/`
    *   `src/phase1_analysis.py`
    *   `src/phase2_baseline.py`
    *   `src/phase3_lexical.py`
    *   `src/phase4_learning.py`
    *   `src/phase4_pre_audit.py`
    *   `src/phase4_verification.py`
    *   `src/phase5_fusion.py`
    *   `src/phase6_error_analysis.py`
    *   `src/phase7_final_predictions.py`
    *   `src/phase8_documentation.py`
    *   `src/validate_predictions.py`
    *   `src/utils.py` (if any, although logic is currently contained in phase scripts)
*   `tests/`
    *   `tests/__init__.py`
    *   `tests/test_evaluation.py`
*   `reports/`
    *   `reports/final_project_report.md`
    *   `reports/final_experiment_summary.md`
    *   `reports/final_results_table.csv`
    *   All markdown files documenting the phases (e.g., `reports/phase1_data_exploration.md`, `reports/phase2_baseline.md`, etc.)
*   `predictions.csv` (The final output of the project)

## OPTIONAL (Detailed Intermediate Artifacts)

These files provide in-depth details of the intermediate experiments. They could be included if full transparency and immediate artifact review on GitHub is desired, but they are not required to run the code.

*   `reports/results.csv`
*   `reports/phase4/query_groups.csv`
*   `reports/phase4/group_fold_distribution.csv`
*   `reports/phase5/v7_reference_group.csv`
*   `reports/phase6/v7_recall3_misses.csv`
*   `reports/phase6/v7_top1_misses.csv`
*   `reports/phase6/failure_categories.csv`
*   `reports/phase6/*.png`
*   Other detailed `.csv` files under `reports/`

## DO NOT PUSH (Ignored Files)

These files should be added to `.gitignore` and must never be pushed to the repository.

*   `.venv/` (Local Python virtual environment)
*   `__pycache__/` (Compiled Python bytecode)
*   `scratch_audit.py` (Temporary audit script)
*   Any IDE-specific folders (e.g., `.vscode/`, `.idea/`)
*   Data files, since the dataset is external (unless a specific `data/` directory is established for sample data).
