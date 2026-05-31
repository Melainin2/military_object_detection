# DEPLOYMENT_MAP.md

## Backend Architecture

```
FastAPI Application (backend/main.py)
├── Routes
│   ├── GET  /              → Serves frontend/index.html
│   ├── POST /predict       → Accepts image, runs YOLO, returns JSON
│   └── GET  /outputs/{id}  → StaticFiles mount for result images
├── Middleware
│   └── CORSMiddleware (configured from ALLOWED_ORIGINS env)
├── Model
│   └── Lazy-loaded YOLO from HuggingFace Hub (datasidahmed/military_object_detection)
├── Storage
│   ├── uploads/  (incoming images)
│   └── outputs/  (processed result images)
└── Dependencies
    ├── FastAPI + Uvicorn (HTTP server)
    ├── Ultralytics YOLO (model inference)
    ├── OpenCV (image processing)
    ├── PyTorch (model runtime)
    └── HuggingFace Hub (model download)
```

## Frontend Architecture

```
Static HTML Page (frontend/index.html)
├── Chart.js (CDN) for bar chart
├── uploadImage() → POST /predict
├── displayResults() → renders detection list
├── drawBoxes() → renders bounding boxes on canvas
└── drawChart() → renders Chart.js bar chart
```

## Runtime Flow

```
User opens browser → http://app-url/
  → FastAPI serves frontend/index.html
  → User selects image, clicks Predict
  → POST /predict with FormData (image file)
  → [First request] Download best.pt from HuggingFace (~300MB)
  → YOLO inference on CPU (conf=0.25, iou=0.5)
  → Filter detections to military classes only
  → Draw bounding boxes via OpenCV
  → Save result image to outputs/
  → Return JSON { detections, image_url }
  → Frontend renders boxes on canvas + Chart.js bar chart
```

## API Flow

| Method | Path | Input | Output |
|--------|------|-------|--------|
| GET | `/` | — | HTML (200) |
| POST | `/predict` | `file` (multipart, jpg/png) | `{ message, detections[], image_url }` (200) or `{ error }` (400/500) |
| GET | `/outputs/{filename}` | filename | Image file (200) or 404 |

## Model Loading Flow

```
POST /predict called
  → load_model()
    → if model is None:
      → hf_hub_download(repo="datasidahmed/military_object_detection",
                         filename="best.pt",
                         token=os.getenv("HF_TOKEN"))
      → YOLO(model_path)
    → return model
  → model.predict(file_path, conf=0.25, iou=0.5, device="cpu")
```

**First request latency**: ~30-60s (download + load ~300MB model)
**Subsequent requests**: ~300-500ms inference time

## Dependency Graph

```
requirements.txt
├── fastapi==0.136.1
│   └── uvicorn==0.46.0
├── ultralytics==8.4.41
│   ├── torch==2.11.0
│   └── torchvision==0.26.0
├── opencv-python-headless==4.13.0.92
├── python-multipart==0.0.26
└── huggingface-hub==1.12.0
```

## Environment Variables

| Variable | Required | Default | Source | Used In |
|----------|----------|---------|--------|---------|
| `HF_TOKEN` | **Yes** | — | HuggingFace account | `backend/main.py:44` — model download |
| `PORT` | No | `8000` | Render assigns | `Procfile`, `start.sh`, `render.yaml` |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000,http://127.0.0.1:3000` | Configuration | `backend/main.py:13` — CORS |

## Deployment Requirements

| Requirement | Backend (Render) | Frontend (Vercel) |
|-------------|------------------|-------------------|
| Runtime | Python 3.11+ | Static |
| Build | `pip install -r requirements.txt` | None |
| Start | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` | — |
| RAM | ~2GB+ (PyTorch + YOLO model) | Minimal |
| Disk | ~500MB (model + dependencies) | Minimal |
| Secrets | `HF_TOKEN` | — |
| Env Vars | `PORT`, `ALLOWED_ORIGINS` | `BACKEND_URL` (for prod) |

## Security Risks

1. **HF_TOKEN in .env file** — must never be committed. `.gitignore` includes `.env` already.
2. **No input validation** — uploaded file size not limited, could cause OOM.
3. **No authentication** — `/predict` endpoint is publicly accessible.
4. **Uploaded files stored on disk** — `uploads/` directory accumulates files.
5. **Output files stored on disk** — `outputs/` directory accumulates files.
6. **Model downloaded at runtime** — first request may timeout on slow connections.
7. **No rate limiting** — endpoint can be called repeatedly.

## Scalability Risks

1. **Single-worker Uvicorn** — no concurrency configuration.
2. **CPU-bound inference** — YOLO inference blocks the event loop.
3. **Local disk storage** — `uploads/` and `outputs/` don't scale across instances.
4. **Lazy model loading** — first request to each instance triggers download.
5. **No caching** — result images regenerated on each request.
6. **No async inference** — `model()` call is synchronous, blocks the request.
