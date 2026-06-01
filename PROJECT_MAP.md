# PROJECT MAP — Military Object Detection

## SYSTEM FLOW

```
User (Browser)
  │
  ▼
Vercel (CDN, static hosting)
  ├── GET  / → frontend/index.html (SPA with Chart.js)
  │
  └── POST /predict ──────────────────────────────────┐
       https://military-object-detection.onrender.com  │
                                                       ▼
                                                 Render (Python)
                                               FastAPI + Uvicorn
                                                    │
                                                    ├── /health → {"status":"healthy"}
                                                    │
                                                    └── /predict
                                                         │
                                                         ├── 1. Validate file (ext + MIME + size)
                                                         ├── 2. Save to uploads/
                                                         ├── 3. cv2.imread → decode image
                                                         ├── 4. cv2.resize → max 640px
                                                         ├── 5. Ensure YOLO model loaded (from HuggingFace Hub)
                                                         ├── 6. YOLO inference @ imgsz=320
                                                         ├── 7. Draw bounding boxes
                                                         ├── 8. Save result to outputs/
                                                         ├── 9. Delete upload
                                                         └── 10. Return JSON + image_url
```

## TECH STACK

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Frontend | Vanilla HTML/CSS/JS | — | SPA, file upload, results display |
| Frontend | Chart.js | CDN | Detection count bar chart |
| Frontend | Vercel | — | Static hosting, CDN, SSL |
| Backend | Python 3.11 | — | Runtime |
| Backend | FastAPI | 0.136.1 | REST API framework |
| Backend | Uvicorn | 0.46.0 | ASGI server |
| Backend | Render (Starter) | — | Cloud hosting, 512 MB RAM, 0.1 CPU |
| AI | Ultralytics YOLOv8 | 8.4.41 | Object detection |
| AI | PyTorch | 2.11.0 | Deep learning framework |
| AI | HuggingFace Hub | 1.12.0 | Model distribution |
| AI | OpenCV | 4.13.0 | Image processing |
| AI Model | datasidahmed/military_object_detection | best.pt (23 MB, YOLOv8s-based) | Military object detection |

## DEPLOYMENT FLOW

```
GitHub (melainin2/military_object_detection)
  │
  ├── Push to main ──────────────────────────────────┐
  │                                                   │
  ▼                                                   ▼
Vercel (auto-deploy)                           Render (auto-deploy)
├── Build: static files                        ├── Build: pip install -r requirements.txt
├── Deploy: @vercel/static                     ├── Deploy: uvicorn backend.main:app
├── Domain: militaryobjectdetection-           ├── Domain: military-object-detection.onrender.com
│   main.vercel.app                            │
│                                              │
└── BACKEND_URL env: ──────────────────────────┘
    https://military-object-detection.onrender.com
```

## KNOWN RISKS

| Risk | Impact | Probability | Mitigation |
|------|--------|------------|------------|
| Render free tier: 0.1 CPU → slow inference | 15-35s per request | Always | `imgsz=320`, `torch.set_num_threads(1)` |
| Render free tier: 512 MB RAM → OOM risk | Process crash | Medium | `gc.collect()` before/after, smaller tensors at 320px |
| Render free tier: hibernation after 15 min | 5-40s wake delay | Always | Background model pre-load, health check wakes app |
| Render free tier: no persistent logs | Blind to crashes | Always | Added `timing:` logs to stdout |
| Browser fetch timeout (~60-90s) | "Failed to fetch" | Reduced | Inference now ~15-35s < browser timeout |
| HuggingFace model gated | Model download fails | Low | HF_TOKEN configured in Render env |
| Single-worker uvicorn | No concurrent requests | Always | Acceptable for free-tier demo |

## REQUEST FLOW (typical predict)

```
t=0s     User selects image, clicks Predict
t=0.1s   Frontend validates extension, creates FormData
t=0.2s   POST https://.../predict (fetch with AbortController 180s)
t=0.5s   CORS preflight OPTIONS (browser)
t=1s     Backend receives request, validates file
t=2s     Upload complete, image decoded
t=3s     Image resized (if >640px)
t=3s     Model check (already loaded ✓)
t=3s     YOLO inference starts @ imgsz=320
t=18-38s Inference complete
t=18.5s  Boxes drawn, output saved
t=18.5s  Upload deleted
t=18.5s  JSON response sent
t=19s    Frontend receives response
t=19s    Results displayed, chart rendered
```

## ARCHITECTURE DECISIONS

1. **imgsz=320 instead of 640**: Reduces inference computation ~4x. Training was at 640 but most detected objects (tanks, trucks, warships) are large enough to detect at 320. Small objects (distant soldiers, weapons) may be missed — acceptable trade-off for production stability.

2. **No ThreadPoolExecutor**: On 0.1 CPU with single-worker uvicorn, async threading adds complexity without benefit. Inference runs synchronously.

3. **Model pre-load at startup**: Background thread loads model from HuggingFace Hub during startup. First request doesn't pay download penalty. Thread-safe via `model_lock`.

4. **`torch.set_num_threads(1)`**: Prevents PyTorch from spawning 4+ threads that compete for 0.1 CPU. Critical for inference time predictability.

5. **No external cache/database**: Free tier constraints. All state is in-process. Uploads deleted after processing. Outputs kept for duration of deployment.
