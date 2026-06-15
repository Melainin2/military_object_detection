# SAFE TO DELETE

This file records all files deleted during project cleanup (Phase 2).

## Runtime Temporaries
- `outputs/` — Generated inference images with bounding boxes
- `uploads/` — Temporary uploaded files (cleaned by backend on each request)
- `experiment_results/` — Experiment artifacts
- `frontend/venv/` — Misplaced virtual environment

## Server Logs & Artifacts
- `server_err.txt`, `server_out.txt`
- `server_stderr.log`, `server_stderr_new.log`
- `server_stdout.log`, `server_stdout_new.log`
- `production_test_results.json`

## Python Caches
- `backend/__pycache__/`
- `scripts/__pycache__/`

## Investigative Reports (18 files)
- `CLASS_MAPPING_REPORT.md`, `DETECTION_QUALITY_REPORT.md`
- `FAILURE_ANALYSIS.md`, `FALSE_POSITIVE_ANALYSIS.md`
- `FINAL_PRODUCTION_VALIDATION.md`, `FINAL_STATUS.md`
- `FIX_REPORT.md`, `FRONTEND_STATE_FIX.md`
- `MIGRATION_PLAN.md`, `MODEL_CONSISTENCY_REPORT.md`
- `PERFORMANCE_REPORT.md`, `PIPELINE_COMPARISON_REPORT.md`
- `PREDICT_AUDIT.md`, `PREDICTION_VALIDATION_REPORT.md`
- `PROJECT_MAP.md`, `REQUEST_TRACE_REPORT.md`
- `REGRESSION_TEST_REPORT.md`, `ROOT_CAUSE_REPORT.md`
- `SOUTENANCE_READINESS_REPORT.md`, `TRAINING_VERIFICATION_REPORT.md`

## Obsolete Debug Scripts (40 files)
- `analyze_experiments.py`, `analyze_failures.py`, `audit_explore.py`
- `batch_stress_test.py`, `check_api.py`, `check_dvid.py`
- `check_dvid_samples.py`, `check_military_classes.py`
- `classify_all.py`, `classify_military.py`, `cors_diag.py`
- `debug_vit.py`, `download_all_images.py`
- `download_more_military.py`, `download_test_images.py`
- `experiment_quality.py`, `extract_dvid_image.py`
- `extract_metrics.py`, `human_test.py`, `load_test.py`
- `local_inference.py`, `measure_timing.py`
- `measure_vit_timing.py`, `model_info.py`
- `production_test.py`, `production_test2.py`
- `production_test_final.py`, `profile_overhead.py`
- `quick_profile.py`, `test_fixed_threshold.py`
- `test_improved_direct.py`, `test_military_improved.py`
- `test_refactor.py`, `test_vit.py`, `timing.py`
- `validate_fix.py`, `validate_pipeline.py`
- `verify_api_fixes.py`, `verify_mapping.py`
- `verify_server_config.py`

## Test Result Files
- `api_stress_results.json`, `stress_test_results.json`

## Kept Scripts (for validation)
- `full_validation.py` — 17-test comprehensive validation
- `final_validation.py` — Final validation suite
- `api_stress_test.py` — 121-image production stress test
- `compare_api_vs_direct.py` — API correctness comparison
