# Pipeline Comparison Report

## Summary
Local and production prediction pipelines are architecturally identical. Both use the same image preprocessing, same model parameters, and produce byte-identical predictions.

## Pipeline Comparison

| Step | Production | Local (script) | Match? |
|------|-----------|----------------|--------|
| **Image source** | Upload via HTTP | File read / HTTP download | N/A |
| **Decode** | `cv2.imread()` → BGR (H,W,C) | `cv2.imread()` / `cv2.imdecode()` → BGR (H,W,C) | IDENTICAL |
| **Resize trigger** | `max(h,w) > MAX_INFERENCE_SIZE` (320) | Same | IDENTICAL |
| **Resize method** | `cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)` | Same | IDENTICAL |
| **Resize target** | `scale = 320 / max(h,w)` → preserves aspect ratio | Same | IDENTICAL |
| **Model input** | BGR numpy array (already resized) | Same | IDENTICAL |
| **YOLO imgsz** | `320` | `320` | IDENTICAL |
| **YOLO confidence** | `0.25` | `0.25` | IDENTICAL |
| **YOLO iou (NMS)** | `0.5` | `0.5` | IDENTICAL |
| **YOLO device** | `cpu` | `cpu` | IDENTICAL |
| **YOLO verbose** | `False` | `False` | IDENTICAL |
| **BGR→RGB conversion** | Handled by YOLO internal `_preprocess` | Same (YOLO handles it) | IDENTICAL |
| **torch.no_grad()** | Yes | Yes | IDENTICAL |
| **torch.set_num_threads(1)** | Yes | Yes | IDENTICAL |
| **Post-processing** | Iterate `results[].boxes`, filter `military_classes` | Same | IDENTICAL |

## Potential Differences Investigated

### 1. BGR/RGB Color Channel Order
- **Production**: `cv2.imread()` returns BGR → passed directly to YOLO → YOLO internally calls `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 
- **Local**: Same cv2.imread() → BGR → same YOLO internal conversion
- **Verdict**: NO DIFFERENCE — both use the identical path

### 2. Double Resize
- Our code resizes to max 320px (preserving aspect ratio), then passes to YOLO
- YOLO internally letterboxes to 320x320 (adding padding for non-square images)
- This is standard YOLO preprocessing — YOLO is designed to handle pre-resized inputs
- **Verdict**: NO DIFFERENCE — YOLO handles this correctly

### 3. Image Normalization
- Both local and production rely on YOLO's internal normalization (`img /= 255`)
- **Verdict**: IDENTICAL

### 4. Model Parameters
- Both load the same `best.pt` file (SHA256 verified)
- Both use `YOLO(model_path)` without custom overrides
- **Verdict**: IDENTICAL

## Conclusion
The prediction pipelines are ELIMINATED as a cause of false positives. Local and production produce identical predictions on all tested images.
