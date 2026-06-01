# Frontend State Fix Report

## Bug Description
When a user uploads a new image for prediction, the detection canvas and chart continue to display the **previous** prediction's bounding boxes and data until the new response arrives. This creates the illusion that the new image contains detections from the old image.

## Root Cause
The `uploadImage()` function in `frontend/index.html` only clears the result text (`resultBox.innerHTML`) but does NOT clear:
- The detection canvas (`#canvas`) — still shows old bounding boxes
- The chart (`#chart`) — still shows old detection counts
- The Chart.js instance — still holds old data

## Fix Applied

**File**: `frontend/index.html`, `uploadImage()` function

Before the fix:
```javascript
resultBox.innerHTML = "...Processing...";
```

After the fix:
```javascript
resultBox.innerHTML = "...Processing...";
const canvas = document.getElementById("canvas");
if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}
if (chart) {
    chart.destroy();
    chart = null;
}
```

This immediately clears:
1. All bounding boxes from the detection canvas
2. The Chart.js instance (including its data and rendered chart)
3. The chart variable is set to null so it can be recreated when the new response arrives

## State Lifecycle

### Before Fix (Broken)
```
Upload new image:
  [0ms]  preview.src = new image
  [0ms]  resultBox = "Processing..."
  [0ms]  canvas = OLD bounding boxes   ← BUG
  [0ms]  chart = OLD chart data        ← BUG
  [50s]  response arrives
  [50s]  displayResults → updates resultBox
  [50s]  drawBoxes → updates canvas
  [50s]  drawChart → updates chart
```

### After Fix
```
Upload new image:
  [0ms]  preview.src = new image
  [0ms]  resultBox = "Processing..."
  [0ms]  canvas.clearRect → CLEAR canvas
  [0ms]  chart.destroy() → CLEAR chart
  [50s]  response arrives
  [50s]  displayResults → updates resultBox
  [50s]  drawBoxes → draws new boxes on canvas
  [50s]  drawChart → creates new chart
```

## Verification
1. No stale bounding boxes visible during prediction
2. No stale chart data visible during prediction
3. Chart is properly recreated on new response (chart.destroy prevents "Canvas already in use" errors)
4. Canvas is blank during loading, then shows new boxes

## Files Changed
- `frontend/index.html`: Added canvas/chart clearing in `uploadImage()`
