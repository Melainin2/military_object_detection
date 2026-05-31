# Production Incident Report

**Date:** 2026-05-31  
**Severity:** High (frontend unable to reach backend)  
**Status:** Resolved

## Incident Summary

Production frontend at `https://militaryobjectdetection-main.vercel.app` showed "Server error!" on Predict button click. The frontend's `BACKEND_URL` was set to empty string (`""`), causing all predict requests to be sent to the same origin (`/predict` on Vercel) instead of the Render backend.

## Root Cause

The Vercel project `military_object_detection-main` had two sets of credentials:

| Credential | Status | Scope |
|------------|--------|-------|
| Old token (`EGm8...`) | Expired | Personal account — created original project |
| New token (`vcp_...`) | Valid | Team `melainin2s-projects` |

The original Vercel deployment (with `BACKEND_URL = ""`) was made under the **old expired token**. API deployments made with the new token targeted the same project but the production alias was not updated — the Vercel API deployment (`target: "production"`) created new deployment URLs but the production alias `militaryobjectdetection-main.vercel.app` continued pointing to the old deployment.

## Evidence

### Before fix — production frontend (2026-05-31 17:13 UTC)
```
GET https://militaryobjectdetection-main.vercel.app
Line 18: const BACKEND_URL = "";
```

### Request flow (broken)
```
User clicks Predict
  → fetch(BACKEND_URL + "/predict")  // BACKEND_URL = ""
  → fetch("/predict")                 // relative URL
  → https://militaryobjectdetection-main.vercel.app/predict  // VERCEL, not Render
  → Vercel returns HTML 404
  → response.json() throws (not JSON)
  → catch block → "Server error!"
```

### After fix — production frontend (2026-05-31 17:25 UTC)
```
GET https://militaryobjectdetection-main.vercel.app
Line 18: const BACKEND_URL = "https://military-object-detection.onrender.com";
```

### Correct request flow
```
User clicks Predict
  → fetch("https://military-object-detection.onrender.com/predict")
  → Render receives POST /predict
  → CORS preflight passes (origin in ALLOWED_ORIGINS)
  → YOLO inference runs
  → JSON response returned
  → Frontend displays results
```

## Resolution

Deployed the frontend via **Vercel CLI** (not API) with the correct project scope:

```bash
vercel deploy --prod --token vcp_... --yes --scope melainin2s-projects
```

The CLI correctly:
1. Uploaded the updated `index.html` with `BACKEND_URL = "https://military-object-detection.onrender.com"`
2. Built the deployment on Vercel's infrastructure
3. Updated the production alias `militaryobjectdetection-main.vercel.app` to point to the new deployment

## Timeline

| Time (UTC) | Event |
|-----------|-------|
| ~16:30 | Initial frontend deploy with old token (`BACKEND_URL = ""`) |
| 17:13 | User reports "Server error!" on Predict |
| 17:13 | Investigation begins — confirmed `BACKEND_URL = ""` in production |
| 17:14 | New Vercel API deployments created but alias not updated |
| 17:25 | Vercel CLI deploy with `--scope` — alias updated successfully |
| 17:26 | Production verification — `BACKEND_URL` correct, predict 200 OK |

## E2E Validation After Fix

```
GET  https://military-object-detection.onrender.com/health  → 200 ✓
GET  https://militaryobjectdetection-main.vercel.app        → 200, BACKEND_URL correct ✓
POST https://military-object-detection.onrender.com/predict → 200, detections returned ✓
GET  https://military-object-detection.onrender.com/outputs/{uuid}.jpg → 200, image/jpeg ✓
CORS Preflight → access-control-allow-origin matches Vercel ✓
```

## Preventive Measures

1. **Use Vercel CLI** for production deployments (not raw API calls)
2. **Verify production alias** after every deploy by checking the live URL content
3. **Automated frontend health check** — verify `BACKEND_URL` matches expected value after deployment

## Lessons Learned

- Vercel API `POST /v13/deployments` with `target: "production"` does NOT always update the production alias
- The Vercel CLI handles alias assignment correctly
- Always verify the deployment by fetching the actual production URL, not just the deployment URL
- Token expiry causes orphaned deployments — both frontend and backend tokens should be tracked
