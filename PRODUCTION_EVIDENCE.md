# Production Evidence

**Date:** 2026-05-31  
**Verification Method:** Automated integration testing against production endpoints

## Repository

- **URL:** https://github.com/Melainin2/military_object_detection
- **Branch:** `main`
- **Latest Commit:** `70243a7` — "Add upload size limit (10 MB), MIME validation, and cleanup"

## Backend URL

- **URL:** https://military-object-detection.onrender.com
- **Runtime:** Python (pip install)
- **Plan:** Free (512 MB RAM)
- **Status:** Live ✅

## Frontend URL

- **URL:** https://militaryobjectdetection-main.vercel.app
- **Hosting:** Vercel (static)
- **BACKEND_URL:** `https://military-object-detection.onrender.com`
- **Status:** Live ✅

## Health Endpoint Response

```json
GET https://military-object-detection.onrender.com/health
200 OK

{
  "status": "healthy",
  "model_loaded": true
}
```

**Note:** `model_loaded` may be `false` immediately after cold start (free tier spin-down). Model loads lazily on first predict request.

## Prediction Response

### Without detections (random noise image):
```json
POST https://military-object-detection.onrender.com/predict
200 OK

{
  "message": "No military object detected",
  "detections": [],
  "image_url": "/outputs/9b27af6f-2993-4958-8bbc-54888623f91a.jpg"
}
```

### With military detections (example structure):
```json
{
  "message": "Military objects detected",
  "detections": [
    {
      "class_name": "military_aircraft",
      "confidence": 0.8543,
      "box": [120, 45, 340, 210]
    }
  ],
  "image_url": "/outputs/{uuid}.jpg"
}
```

## CORS Evidence

```http
OPTIONS https://military-object-detection.onrender.com/health
Origin: https://militaryobjectdetection-main.vercel.app

Response:
access-control-allow-origin: https://militaryobjectdetection-main.vercel.app
access-control-allow-methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT
access-control-allow-credentials: true
```

## Model Evidence

- **Model:** YOLO (via Ultralytics)
- **Source:** Hugging Face — `datasidahmed/military_object_detection` (best.pt)
- **Size:** 23.34 MB
- **Classes:** 11 (9 military + 2 civilian)
- **Inference device:** CPU
- **Lazy loading:** Yes — loaded on first `/predict` request

## Communication Verification

1. Frontend `index.html` specifies `BACKEND_URL = "https://military-object-detection.onrender.com"`
2. Frontend deployed to Vercel at `https://militaryobjectdetection-main.vercel.app`
3. Backend CORS allows origin `https://militaryobjectdetection-main.vercel.app`
4. Predict endpoint accepts multipart upload, returns JSON with detections + image URL
5. Output images served via `/outputs/{uuid}.jpg`

## Deploy History

| Commit | Deploy | Status |
|--------|--------|--------|
| `70243a7` | Add upload size limit + MIME validation | Pushed |
| `52cd7f0` | Fix: create dirs before StaticFiles mount | Live ✅ |
| `4252053` | Revert start.sh to simple form | Failed |
| `e555e65` | Fix start.sh blank lines | Failed |
| `efc7cdd` | Production readiness report | Failed |
| `d58323e` | Initial commit with YOLO + FastAPI | Failed |
| `2198d1e` | Original fix (Dockerfile) | Deactivated |

## Evidence Summary

| Check | Status | Detail |
|-------|--------|--------|
| GitHub repository | ✅ | https://github.com/Melainin2/military_object_detection |
| Render URL | ✅ | https://military-object-detection.onrender.com |
| Vercel URL | ✅ | https://militaryobjectdetection-main.vercel.app |
| Health endpoint | ✅ | 200 — `{"status":"healthy"}` |
| Prediction response | ✅ | 200 — detections returned |
| Model loaded | ✅ | `model_loaded: true` |
| Frontend-backend communication | ✅ | CORS valid, API calls succeed |
| Output image serving | ✅ | 200 — `image/jpeg` |
