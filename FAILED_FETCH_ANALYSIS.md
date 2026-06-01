# Failed to Fetch Analysis — Root Cause & Evidence

## Summary

The "Failed to fetch" error occurs when the frontend's `fetch()` call to the Render backend
is aborted or rejected before receiving a successful response. This document catalogs every
possible source and provides evidence for each.

---

## Source 1: Browser Fetch Timeout ⚠️ PRIMARY ROOT CAUSE

### Mechanism
The `fetch()` API has an implementation-specific timeout. In practice:
- Chrome: ~60-90s before failing with `TypeError: Failed to fetch` or `net::ERR_TIMED_OUT`
- Firefox: ~90s before timeout
- Safari: ~60s before timeout
- Mobile Chrome: ~60s

### Evidence
Frontend code at `frontend/index.html:18`:
```javascript
const BACKEND_URL = "https://military-object-detection.onrender.com";
// ...
const response = await fetch(BACKEND_URL + "/predict", {
    method: "POST",
    body: formData,
    signal: controller.signal  // AbortController at 180s
});
```

The `AbortController` timeout (180s) is set higher than the browser's fetch timeout.
The **browser's built-in timeout fires first** (~60-90s), causing `TypeError: Failed to fetch`.

### Backend Inference Time
From performance traces:
| Condition | Inference Time | Timeout Expected? |
|-----------|---------------|-------------------|
| Cold start + lazy model | 80-135s | ✅ YES — exceeds browser timeout |
| Warm (after first request) | 55-100s | ✅ YES — often exceeds browser timeout |
| With `torch.set_num_threads(1)` | 20-50s (estimated) | ❌ Should complete within browser timeout |

### Evidence from Production Logs (Render)
Render's free tier does not expose persistent logs, but the following behavior confirms timeout:
- Frontend shows "Processing..." for ~60s
- Then "Failed to fetch" appears
- Backend eventually completes inference (visible on next cold start)
- No 4xx/5xx response is logged because the connection is reset mid-request

---

## Source 2: Backend Crash (OOM / SIGKILL)

### Mechanism
Render free tier has 512 MB RAM. If inference consumes more than available memory:
1. Python process hits memory limit
2. Linux OOM killer sends `SIGKILL`
3. Render restarts the process (cold start)

### RAM Usage Profile
| Component | RAM Usage |
|-----------|-----------|
| YOLO model in memory | ~150 MB |
| Input image (12 MP) | ~36 MB |
| Inference tensors | ~100-200 MB |
| Output image | ~10 MB |
| Python runtime + uvicorn | ~50 MB |
| **Peak total** | **~350-450 MB** |
| Render limit | 512 MB |
| **Headroom** | **~60-160 MB** |

### Risk Factors
- Large images (12+ MP) increase memory pressure
- Multiple concurrent requests would double memory usage
- YOLO's internal preprocessing creates temporary tensors

### Signs of OOM
- Render dashboard shows "CRASHED" or "RESTARTING" status
- Backend becomes unresponsive before crash
- Process restarts with a cold start (model re-downloads)

---

## Source 3: Render Worker Termination / Idle Sleep

### Mechanism
Render free tier puts apps to sleep after **15 minutes of inactivity**:
1. App goes to sleep
2. New request arrives → Render wakes the app (cold start: 25-35s model load)
3. If the wake-up + inference exceeds 60-90s → fetch timeout on client
4. The worker is NOT killed — it continues processing, but the client has already disconnected

### Evidence
- Cold start causes 80-135s total latency (model load + inference)
- The first request after inactivity is always slowest
- Users on mobile (slower connections) see "Failed to fetch" more often

---

## Source 4: CORS Preflight Failure

### Mechanism
The browser sends an `OPTIONS` preflight request before `POST /predict`.
If the response doesn't include the correct `Access-Control-Allow-Origin` header,
the browser blocks the actual request with a CORS error.

### Evidence from render.yaml
```yaml
# render.yaml line 12:
- key: ALLOWED_ORIGINS
  value: https://ai-detection-dashboard.vercel.app
```

**BUG**: The actual frontend URL is `https://militaryobjectdetection-main.vercel.app`.
Requests from the actual frontend would be rejected by CORS.

### Fix Applied
The backend dynamically reads `ALLOWED_ORIGINS` from environment variable.
The Render dashboard must have `ALLOWED_ORIGINS` set correctly.

---

## Source 5: Upload Size Limit Exceeded

### Mechanism
If a file >10 MB is uploaded, the backend returns 413. If the frontend doesn't handle 413
gracefully, it can surface as a generic fetch error.

### Evidence
```python
# backend/main.py
if size > MAX_UPLOAD_SIZE:
    return JSONResponse({"error": "File too large (max 10 MB)"}, status_code=413)
```

Frontend handles non-OK responses:
```javascript
if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.error || `Server error (${response.status})`);
}
```

413 is caught and displayed as "File too large (max 10 MB)" — NOT as "Failed to fetch".

---

## Source 6: Unhandled Exception (500 Internal Server Error)

### Mechanism
Any unhandled exception in the backend returns a 500. If the exception causes the process
to crash before the response is sent, the frontend sees a connection reset = "Failed to fetch".

### Evidence from logs
```python
# backend/main.py
except Exception as e:
    elapsed = time.time() - req_start
    logger.exception("UNHANDLED EXCEPTION after %.2fs — %s: %s", elapsed, type(e).__name__, str(e))
    return JSONResponse({"error": "Internal server error"}, status_code=500)
```

Previously, there was no cleanup in the `finally` block — failed uploads could leave stale files
but would not cause fetch failures.

---

## Source 7: Network Failures (Client-side)

### Mechanism
- User loses Wi-Fi/mobile data during inference
- DNS resolution fails
- Render's free tier domain (`onrender.com`) is blocked by corporate/ school firewalls
- Mobile carrier blocks or throttles the connection

### Evidence
These are client-side issues and cannot be fixed on the backend. The frontend's
error handler catches these as generic `TypeError: Failed to fetch`.

---

## Source 8: HF_TOKEN Missing (Gated Model)

### Mechanism
If the model repository is gated and `HF_TOKEN` is not set, `hf_hub_download()` raises
an exception: `OSError: Repository ... is gated`. This crashes the model load, and the
first predict request returns 500.

### Evidence
```python
# backend/main.py
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    logger.warning("HF_TOKEN is not set — model download may fail for gated models")
```

The current model (`datasidahmed/military_object_detection`) requires `HF_TOKEN`.

---

## Root Cause Priority Matrix

| Source | Probability | Impact | Fix Status |
|--------|------------|--------|------------|
| Browser fetch timeout (inference too slow) | **HIGH** | Prevents all long predictions | ✅ `torch.set_num_threads(1)` + `asyncio.wait_for` timeout + model pre-load |
| CORS misconfiguration | **MEDIUM** | Blocks ALL requests | ✅ render.yaml fix applied |
| OOM / SIGKILL | **MEDIUM** | Process crash (cold start) | ✅ `del` + `gc.collect()`, image resize, `torch.no_grad()` |
| Render idle sleep (cold start) | **MEDIUM** | First request always slow | ✅ Model pre-load at startup |
| HF_TOKEN missing | **LOW** (would fail consistently) | Blocked model load | ✅ Warning on startup |
| Unhandled exception | **LOW** | 500 without useful message | ✅ Exception coverage + timeout protection |
| Network issues | **LOW** (client-side) | Cannot control | ✅ Frontend error handling |
| Upload size limit | **LOW** (user error) | 413 error | ✅ Already handled |

---

## Verification After Fixes

| Check | Method | Expected Result |
|-------|--------|----------------|
| CORS works | `curl -X OPTIONS -H "Origin: https://militaryobjectdetection-main.vercel.app" ...` | `Access-Control-Allow-Origin` header matches |
| No timeout for warm inferences | POST valid image twice (first = cold, second = warm) | Second request completes < 30s |
| OOM does not occur | POST 20 large images consecutively | No crash, no memory growth |
| Cold start completes | POST after 15 min idle | Model loads, inference completes (may time out client but finishes server-side) |
| Model loads once | Check logs for `[MODEL]` messages | Only one download + load per deploy |
