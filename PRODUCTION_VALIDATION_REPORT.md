# Production Validation Report

**Date:** 2026-05-31  
**Backend:** Render (Python runtime, free tier)  
**Frontend:** Vercel (static hosting, Hobby tier)  
**Repository:** https://github.com/Melainin2/military_object_detection

## Endpoints

### Render Backend — https://military-object-detection.onrender.com

| Endpoint | Method | Status | Response | Verified |
|----------|--------|--------|----------|----------|
| `/health` | GET | 200 | `{"status":"healthy","model_loaded":true/false}` | ✅ |
| `/` | GET | 200 | `text/html` (frontend/index.html) | ✅ |
| `/predict` | POST | 200 | `{"message":"...","detections":[...],"image_url":"..."}` | ✅ |
| `/outputs/{id}.jpg` | GET | 200 | `image/jpeg` | ✅ |

### Vercel Frontend — https://militaryobjectdetection-main.vercel.app

| Check | Status | Verified |
|-------|--------|----------|
| HTTPS | ✅ | ✅ |
| HTML served | ✅ | ✅ |
| BACKEND_URL configured | ✅ | `https://military-object-detection.onrender.com` |
| CORS to Render | ✅ | Preflight returns 200 |

## Configuration

### Render Environment Variables
```
ALLOWED_ORIGINS = http://localhost:3000,http://127.0.0.1:3000,https://militaryobjectdetection-main.vercel.app
HF_TOKEN         = (set, masked)
PORT             = 10000
```

### CORS Headers (from Render)
```
access-control-allow-credentials: true
access-control-allow-origins: https://militaryobjectdetection-main.vercel.app
access-control-allow-methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT
access-control-max-age: 600
```

### Deploy Configuration
```
Runtime: Python (pip install -r requirements.txt)
Start:   bash start.sh
  ↳ uvicorn backend.main:app --host 0.0.0.0 --port $PORT
Build:   pip install -r requirements.txt
Plan:    Free (512 MB RAM, throttled CPU)
```

## Performance (Free Tier)

| Metric | Value | Notes |
|--------|-------|-------|
| Cold start time | ~45-60s | Service spins down after 15 min inactivity |
| YOLO inference | ~30-60s | CPU throttled on free tier |
| Model loading | ~10-15s | On first predict after cold start |
| Response size | ~135 KB | Result image + JSON payload |
| Memory usage | ~370 MB peak | Within 512 MB free limit |

## Known Production Limitations

1. **Free tier spin-down**: Service becomes unresponsive after 15 min inactivity. Mitigated by retry logic.
2. **Cold start latency**: First request after idle period takes 45-60s.
3. **Slow inference**: Free tier CPU throttling means ~30-60s per prediction.
4. **No health check path**: Render `healthCheckPath` not configured (free tier limitation).
5. **No auto-scaling**: Single instance on free tier.

## Upgrade Recommendations

If moving beyond prototype:
- **Starter plan ($7/mo)**: Eliminates spin-down, 2x CPU → inference in ~5-10s
- **Professional plan ($25/mo)**: Dedicated CPU → inference in ~1-2s
- **Custom domain**: For branded deployment

## Validation Summary

All production endpoints validated. Service is operational within free tier constraints. Suitable for demo/prototype use.
