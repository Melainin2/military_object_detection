# OBSERVABILITY_REPORT.md

## Implementation: Structured Logging (No External Services)

**Decision**: Use Python's built-in `logging` module. No paid observability services (DataDog, NewRelic, etc.) were added per requirements.

## Logging Configuration

**File**: `backend/main.py:18-23`

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backend")
```

**Format**: `2025-05-31 12:00:00 [INFO] message`

## Log Events

| Event | Level | Location | Example |
|-------|-------|----------|---------|
| Application startup | `INFO` | `on_startup()` | `Starting AI Detection Dashboard...` |
| Directory creation | `INFO` | `on_startup()` | `Upload and output directories ready` |
| HF_TOKEN check | `INFO`/`WARNING` | `on_startup()` | `HF_TOKEN is configured` / `HF_TOKEN is not set` |
| Model download | `INFO` | `load_model()` | `Downloading model from Hugging Face Hub...` |
| Model loaded | `INFO` | `load_model()` | `Model downloaded: /path/to/best.pt` |
| Predict request | `INFO` | `predict()` | `Predict request from 127.0.0.1 — file: image.jpg` |
| Predict complete | `INFO` | `predict()` | `Processed uuid.jpg from 127.0.0.1 — 3 military detections` |
| Invalid image | `ERROR` | `predict()` | `cv2.imread failed for /path/to/file` |
| Prediction error | `ERROR` | `predict()` | `Prediction failed` (with full stack trace) |
| Application shutdown | `INFO` | `on_shutdown()` | `Shutting down AI Detection Dashboard` |

## Request Logging

Uvicorn logs all HTTP requests by default with format:
```
INFO:     <client_ip>:<port> - "<METHOD> <path> HTTP/1.1" <status_code>
```

## Health Monitoring

**Endpoint**: `GET /health`

**Response**:
```json
{"status": "healthy", "model_loaded": true}
```

Used by Render's health check system to monitor service availability.

## Error Logging

Errors are logged with `logger.exception()` which includes the full Python stack trace. To prevent information leakage, a generic `Internal server error` message is returned to the client.

## Render Logs

Render captures all stdout/stderr from the application, which includes both:
- Python logging output (`[INFO]`, `[ERROR]`, etc.)
- Uvicorn HTTP request logs

These are accessible from the Render Dashboard → Service → Logs.

## What Is NOT Implemented

| Feature | Reason |
|---------|--------|
| Metrics (prometheus, etc.) | Would require external dependencies and endpoint |
| Distributed tracing | Not needed for single-service architecture |
| Alerting | Covered by Render's built-in health check alerts |
| Log aggregation | Render Dashboard serves this purpose |
| APM | Would require paid service |
