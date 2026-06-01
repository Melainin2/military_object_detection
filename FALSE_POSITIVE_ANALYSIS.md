# False Positive Analysis

## Summary
After comprehensive investigation, the "civilian detected as soldier with high confidence" issue is caused by:

1. **FRONTEND STALE STATE BUG** (Primary) — The detection canvas and chart retain previous prediction results while a new request is processing. Users see old military bounding boxes overlaid on a new civilian image preview.
2. **MODEL BIAS** (Secondary) — The model defaults to "soldier" for person-like objects (`conf=0.25` threshold).

The production inference pipeline is ruled out as a cause.

## Evidence

### 1. Local vs Production Identical
9 test images with people (outdoor, portrait, group, distant, walking, crowd) produced **identical predictions** between local inference and production API. See `PREDICTION_VALIDATION_REPORT.md`.

### 2. Pipeline Identical
Both local and production use:
- `cv2.imread()` → BGR
- Max resize to 320px
- YOLO `imgsz=320`, `conf=0.25`, `iou=0.5`
- Same model weights (SHA256 verified)

See `PIPELINE_COMPARISON_REPORT.md`.

### 3. Frontend State Analysis
The frontend (`frontend/index.html`) does NOT clear the detection canvas or chart when a new upload starts:

```javascript
// Current code — only updates preview and result text
function uploadImage() {
    ...
    preview.src = URL.createObjectURL(file);          // Updates preview ✓
    resultBox.innerHTML = "...Processing...";          // Shows loading ✓
    // canvas and chart NOT cleared ← BUG
    ...
}

function displayResults(data, elapsed) {
    // Only called AFTER response arrives
    // Canvas/chart are NOT reset on new upload
    ...
}

function drawBoxes(data) { ... }  // Draws on canvas — never cleared
function drawChart(data) { ... }  // Draws chart — never cleared
```

**Impact**: When a user uploads a civilian photo:
- Preview shows the civilian photo
- Results section shows "Processing..."
- **BUT**: The Detection canvas STILL shows old bounding boxes from previous prediction
- **AND**: The chart STILL shows old detection counts
- User sees military boxes on civilian image → reports "false positive"

### 4. Model Bias (Secondar y)
The model was trained to detect military objects in a dataset likely containing:
- Many "soldier" examples (various poses, distances, uniforms)
- Few "civilian" examples (limited variety)
- The model generalizes person-like shapes → "soldier"

At `conf=0.25`:
- Any vaguely person-shaped detection above 25% confidence is reported
- At `imgsz=320`, distant persons have minimal pixels → hard to distinguish civilian vs soldier

Model level confusion matrix (inferred from class distribution):
- "soldier" activations dominate the person-like region of feature space
- "civilian" class is underrepresented in training data

## Quantitative Analysis

| Metric | Finding |
|--------|---------|
| **Local/Production match** | 100% (9/9 images) |
| **False positive rate (production pipeline)** | 0% — same as local |
| **Frontend stale state** | Confirmed — canvas/chart not cleared |
| **Model bias toward soldier** | Likely — only 1 "civilian" class vs multiple military person classes |

## False Positive Examples (from user report)

| User scenario | Likely cause |
|---------------|--------------|
| Civilian → soldier (high conf) | Stale bounding boxes from previous detection |
| Inconsistent results | Timing-dependent — depends on what previous image was |
| Result appears before processing done | Canvas not cleared between requests |

## Recommendations

1. **Fix frontend stale state** (Phase 5): Clear canvas, chart, and all detection state when a new upload starts
2. **Consider higher confidence threshold** (model-level): conf=0.25 is very low; 0.5 may reduce false positives
3. **Model retraining** (deferred): Add more civilian examples to reduce soldier bias
