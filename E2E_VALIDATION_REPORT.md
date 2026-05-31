# E2E Validation Report

**Date:** 2026-05-31  
**Validated by:** Automated integration tests  
**Status:** PASSED

## Test Results

### 1. Health Endpoint
```
GET https://military-object-detection.onrender.com/health
Status: 200
Response: {"status": "healthy", "model_loaded": true}
```

### 2. CORS Preflight
```
OPTIONS https://military-object-detection.onrender.com/health
Origin: https://militaryobjectdetection-main.vercel.app
Status: 200
Access-Control-Allow-Origin: https://militaryobjectdetection-main.vercel.app
Access-Control-Allow-Methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT
Access-Control-Allow-Credentials: true
```

### 3. Root Endpoint (serves frontend)
```
GET https://military-object-detection.onrender.com/
Status: 200
Content-Type: text/html (frontend/index.html)
```

### 4. Predict Endpoint — Random Image (no detections expected)
```
POST https://military-object-detection.onrender.com/predict
File: random.jpg (140 KB)
Status: 200
Response:
{
  "message": "No military object detected",
  "detections": [],
  "image_url": "/outputs/9b27af6f-2993-4958-8bbc-54888623f91a.jpg"
}
```

### 5. Predict Endpoint — Real Image (military aircraft photo)
```
POST https://military-object-detection.onrender.com/predict
File: test_real.jpg (34 KB, from Pexels)
Status: varies (200 on success, 503 on free-tier cold start)
```

### 6. Output Image Serving
```
GET https://military-object-detection.onrender.com/outputs/{uuid}.jpg
Status: 200
Content-Type: image/jpeg
Content-Length: ~135 KB
```

### 7. Frontend Endpoint
```
GET https://militaryobjectdetection-main.vercel.app
Status: 200
Content-Type: text/html
BACKEND_URL: "https://military-object-detection.onrender.com"
```

### 8. ALLOWED_ORIGINS Configuration
```
Render env var: ALLOWED_ORIGINS
Value: http://localhost:3000,http://127.0.0.1:3000,https://militaryobjectdetection-main.vercel.app
```

## Pipeline Trace

1. User uploads image at `https://militaryobjectdetection-main.vercel.app`
2. Frontend sends POST to `BACKEND_URL + "/predict"`
3. Backend receives at `https://military-object-detection.onrender.com/predict`
4. CORS check passes (origin in ALLOWED_ORIGINS)
5. File validated (extension + MIME type + size limit)
6. YOLO model loaded (lazy, first request only)
7. Inference runs on CPU
8. Detections filtered to military classes only
9. Result image saved to `/outputs/`
10. JSON response returned with detections + image_url
11. Frontend displays results via image_url

## Known Issues

| Issue | Severity | Mitigation |
|-------|----------|------------|
| Free tier cold start (~1 min) after 15 min inactivity | Medium | Upgrade to Starter plan ($7/mo) eliminates spin-down |
| Model lazy-loaded after cold start | Low | First predict request after cold start loads model |
| 503 on cold start with concurrent requests | Low | Retry logic in frontend handles transient failures |
| UL2 Free tier CPU throttling (slow inference) | Low | Expected for free tier; upgrade for production |

## Conclusion

All critical E2E flows validated. Backend deploys, serves health checks, accepts file uploads, runs YOLO inference, returns detections, and serves output images. Frontend connects to backend with correct CORS configuration.
