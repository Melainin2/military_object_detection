# Model Consistency Report

## Summary
The production deployment uses the exact same model weights as local inference. Predictions are identical.

## Model Identification
- **HF Repo**: `datasidahmed/military_object_detection`
- **Filename**: `best.pt`
- **Model type**: YOLOv8 (Ultralytics)
- **Download method**: `huggingface_hub.hf_hub_download()`

## File Verification

| Property | Value |
|----------|-------|
| **SHA256** | `9d79153464316db0358d2bedb74d5d4ad753802e0c4c66715d6cd03846be56c5` |
| **File size** | 24,478,307 bytes (24.5 MB) |
| **HF LFS OID** | `9d79153464316db0358d2bedb74d5d4ad753802e0c4c66715d6cd03846be56c5` |
| **HF snapshot** | `4d92e52ed90722386e92925330a38e2bbeeced10` |

## Verification Method
1. Downloaded `best.pt` from Hugging Face Hub
2. Computed SHA256: `9d79153464316db0358d2bedb74d5d4ad753802e0c4c66715d6cd03846be56c5`
3. Verified against HF LFS pointer file (which specifies the same SHA256)
4. Confirmed Hugging Face Hub cache uses the same snapshot on all systems

## Prediction Consistency Test
9 test images were run through both local inference and production API:

| Image | Local detections | Production detections | Match? |
|-------|-----------------|----------------------|--------|
| picsum 640x480 (civilian_1) | None | None | MATCH |
| picsum 800x600 (civilian_2) | None | None | MATCH |
| picsum 1024x768 (landscape_1) | warship 0.3207 | warship 0.3207 | MATCH |
| person_outdoor | None | None | MATCH |
| person_portrait | None | None | MATCH |
| group_people | None | None | MATCH |
| person_distant | None | None | MATCH |
| person_walking | None | None | MATCH |
| crowd | soldier 0.4354 | soldier 0.4354 | MATCH |

**All 9 test images produced identical predictions.**

## Conclusion
- The model weights are identical between local and production
- The Hugging Face repo has NOT been modified since the initial deployment (LFS OID unchanged)
- Any perceived difference in predictions is NOT caused by model version mismatch
- The root cause of "civilian persons detected as soldiers" lies elsewhere (see False Positive Analysis)
