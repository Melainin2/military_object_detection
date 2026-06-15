# PROJECT CLEANUP REPORT

## Classification Categories
- **A** = Production file
- **B** = Required runtime dependency
- **C** = Configuration
- **D** = Test file
- **E** = Debug file
- **F** = Investigation file
- **G** = Validation file
- **H** = Temporary file

## Root Directory

| File | Category | Action | Rationale |
|------|----------|--------|-----------|
| `backend/` | A | KEEP | Core backend application |
| `frontend/` | A | KEEP | Core frontend application |
| `best.pt` | B | KEEP | YOLO model weights |
| `requirements.txt` | C | KEEP | Python dependencies |
| `Dockerfile` | C | KEEP | Docker build config |
| `.dockerignore` | C | KEEP | Docker ignore rules |
| `Procfile` | C | KEEP | Heroku deployment config |
| `render.yaml` | C | KEEP | Render deployment config |
| `vercel.json` | C | KEEP | Vercel deployment config |
| `start.sh` | C | KEEP | Start script |
| `.env.example` | C | KEEP | Environment template |
| `.gitignore` | C | KEEP | Git ignore rules |
| `LICENSE` | C | KEEP | License file |
| `README.md` | C | KEEP | Project documentation |
| `test_images_military_vs_civilian/` | D | KEEP | Demo test images |
| `CLASS_MAPPING_REPORT.md` | F | SAFE_TO_DELETE | Investigation report |
| `DETECTION_QUALITY_REPORT.md` | F | SAFE_TO_DELETE | Investigation report |
| `FAILURE_ANALYSIS.md` | F | SAFE_TO_DELETE | Investigation report |
| `FALSE_POSITIVE_ANALYSIS.md` | F | SAFE_TO_DELETE | Investigation report |
| `FINAL_PRODUCTION_VALIDATION.md` | G | SAFE_TO_DELETE | Obsolete validation report |
| `FINAL_STATUS.md` | G | SAFE_TO_DELETE | Superseded by DEMO_READINESS_REPORT |
| `FIX_REPORT.md` | F | SAFE_TO_DELETE | Investigation report |
| `FRONTEND_STATE_FIX.md` | F | SAFE_TO_DELETE | Investigation report |
| `MIGRATION_PLAN.md` | F | SAFE_TO_DELETE | Investigation report |
| `MODEL_CONSISTENCY_REPORT.md` | F | SAFE_TO_DELETE | Investigation report |
| `PERFORMANCE_REPORT.md` | F | SAFE_TO_DELETE | Investigation report |
| `PIPELINE_COMPARISON_REPORT.md` | G | SAFE_TO_DELETE | Obsolete comparison report |
| `PREDICT_AUDIT.md` | F | SAFE_TO_DELETE | Investigation report |
| `PREDICTION_VALIDATION_REPORT.md` | G | SAFE_TO_DELETE | Obsolete validation report |
| `PROJECT_MAP.md` | F | SAFE_TO_DELETE | Investigation report |
| `REQUEST_TRACE_REPORT.md` | F | SAFE_TO_DELETE | Investigation report |
| `REGRESSION_TEST_REPORT.md` | G | SAFE_TO_DELETE | Superseded by DEMO_READINESS_REPORT |
| `ROOT_CAUSE_REPORT.md` | F | SAFE_TO_DELETE | Investigation report |
| `SOUTENANCE_READINESS_REPORT.md` | G | SAFE_TO_DELETE | Superseded by DEMO_READINESS_REPORT |
| `TRAINING_VERIFICATION_REPORT.md` | F | SAFE_TO_DELETE | Investigation report |
| `production_test_results.json` | H | SAFE_TO_DELETE | Temporary test artifact |
| `server_err.txt` | H | SAFE_TO_DELETE | Server log |
| `server_out.txt` | H | SAFE_TO_DELETE | Server log |
| `server_stderr.log` | H | SAFE_TO_DELETE | Server log |
| `server_stderr_new.log` | H | SAFE_TO_DELETE | Server log |
| `server_stdout.log` | H | SAFE_TO_DELETE | Server log |
| `server_stdout_new.log` | H | SAFE_TO_DELETE | Server log |
| `outputs/` | H | SAFE_TO_DELETE | Runtime generated images (regenerated on demand) |
| `uploads/` | H | SAFE_TO_DELETE | Runtime uploads (temp files deleted by backend) |
| `.git/` | C | KEEP | Git repository |
| `.github/` | C | KEEP | GitHub config |
| `.vercel/` | C | KEEP | Vercel deployment data |
| `experiment_results/` | H | SAFE_TO_DELETE | Temporary experiment artifacts |

## Backend (`backend/`)

| File | Category | Action | Rationale |
|------|----------|--------|-----------|
| `main.py` | A | KEEP | Core backend application |
| `classifier_service.py` | A | KEEP | Core classifier service |
| `Dockerfile` | C | KEEP | Docker config |
| `__pycache__/` | H | SAFE_TO_DELETE | Python bytecode cache |

## Frontend (`frontend/`)

| File | Category | Action | Rationale |
|------|----------|--------|-----------|
| `index.html` | A | KEEP | Core frontend application |
| `venv/` | H | SAFE_TO_DELETE | Virtual environment (misplaced in frontend/) |

## Scripts (`scripts/`)

| File | Category | Action | Rationale |
|------|----------|--------|-----------|
| `full_validation.py` | G | KEEP | Comprehensive 17-test validation |
| `final_validation.py` | G | KEEP | Final validation suite |
| `api_stress_test.py` | D | KEEP | 121-image production stress test |
| `compare_api_vs_direct.py` | D | KEEP | API correctness validation |
| `analyze_experiments.py` | F | SAFE_TO_DELETE | Experiment analysis |
| `analyze_failures.py` | F | SAFE_TO_DELETE | Failure analysis |
| `audit_explore.py` | F | SAFE_TO_DELETE | Exploration audit |
| `batch_stress_test.py` | D | SAFE_TO_DELETE | Duplicate of api_stress_test.py |
| `check_api.py` | E | SAFE_TO_DELETE | Debug check |
| `check_dvid.py` | F | SAFE_TO_DELETE | DVID investigation |
| `check_dvid_samples.py` | F | SAFE_TO_DELETE | DVID investigation |
| `check_military_classes.py` | F | SAFE_TO_DELETE | Class mapping investigation |
| `classify_all.py` | E | SAFE_TO_DELETE | Debug classify |
| `classify_military.py` | E | SAFE_TO_DELETE | Debug classify |
| `cors_diag.py` | E | SAFE_TO_DELETE | CORS debug |
| `debug_vit.py` | E | SAFE_TO_DELETE | ViT debug |
| `download_all_images.py` | F | SAFE_TO_DELETE | One-time download tool |
| `download_more_military.py` | F | SAFE_TO_DELETE | One-time download tool |
| `download_test_images.py` | F | SAFE_TO_DELETE | One-time download tool |
| `experiment_quality.py` | F | SAFE_TO_DELETE | Quality experiment |
| `extract_dvid_image.py` | F | SAFE_TO_DELETE | One-time extraction |
| `extract_metrics.py` | F | SAFE_TO_DELETE | Metrics extraction |
| `human_test.py` | D | SAFE_TO_DELETE | One-time human test |
| `load_test.py` | D | SAFE_TO_DELETE | Duplicate stress test |
| `local_inference.py` | E | SAFE_TO_DELETE | Debug inference |
| `measure_timing.py` | E | SAFE_TO_DELETE | Timing debug |
| `measure_vit_timing.py` | E | SAFE_TO_DELETE | ViT timing debug |
| `model_info.py` | F | SAFE_TO_DELETE | Model investigation |
| `production_test.py` | D | SAFE_TO_DELETE | Obsolete — superseded by api_stress_test |
| `production_test2.py` | D | SAFE_TO_DELETE | Obsolete — superseded by api_stress_test |
| `production_test_final.py` | D | SAFE_TO_DELETE | Obsolete — superseded by api_stress_test |
| `profile_overhead.py` | E | SAFE_TO_DELETE | Overhead profiling |
| `quick_profile.py` | E | SAFE_TO_DELETE | Quick profile |
| `test_fixed_threshold.py` | D | SAFE_TO_DELETE | One-time threshold test |
| `test_improved_direct.py` | D | SAFE_TO_DELETE | One-time direct test |
| `test_military_improved.py` | D | SAFE_TO_DELETE | One-time military test |
| `test_refactor.py` | D | SAFE_TO_DELETE | One-time refactor test |
| `test_vit.py` | D | SAFE_TO_DELETE | One-time ViT test |
| `timing.py` | E | SAFE_TO_DELETE | Timing script |
| `validate_fix.py` | G | SAFE_TO_DELETE | One-time fix validation |
| `validate_pipeline.py` | G | SAFE_TO_DELETE | Obsolete — superseded by full_validation |
| `verify_api_fixes.py` | G | SAFE_TO_DELETE | One-time API fix verification |
| `verify_mapping.py` | G | SAFE_TO_DELETE | One-time mapping verification |
| `verify_server_config.py` | G | SAFE_TO_DELETE | One-time config verification |
| `api_stress_results.json` | H | SAFE_TO_DELETE | Temporary test artifact |
| `stress_test_results.json` | H | SAFE_TO_DELETE | Temporary test artifact |
| `__pycache__/` | H | SAFE_TO_DELETE | Python bytecode cache |

## Summary

| Category | Keep | Delete |
|----------|------|--------|
| Production (A) | 4 | 0 |
| Runtime dependency (B) | 1 | 0 |
| Configuration (C) | 12 | 0 |
| Test/Validation (D/G) | 4 | 16 |
| Debug (E) | 0 | 10 |
| Investigation (F) | 0 | 13 |
| Temporary (H) | 0 | 10 |

**Total KEEP**: 21 files/dirs
**Total SAFE_TO_DELETE**: 52 files/dirs
