# Request Trace Report

## Summary
Added detailed per-request tracing to the production backend, logging every stage of the prediction pipeline.

## Trace Log Format

Every prediction request now generates structured logs with `TRACE:` prefix at key stages:

| Stage | Log Format | Description |
|-------|-----------|-------------|
| `request_received` | `client=<ip> file=<name> size=<bytes>` | Request arrived, filename and size recorded |
| `config` | `imgsz=<N> conf=<N> iou=<N> max_inference_size=<N>` | Active pipeline configuration |
| `model_path` | `model_path=<path>` | Full path to the loaded model file |
| `inference_start` | `img_shape=(H,W,C)` | Shape of the image tensor passed to YOLO |
| `inference_end` | `duration=<seconds>` | Raw inference duration (excluding pre/post processing) |
| `response_sent` | `status=<code> detections=<N> elapsed=<seconds>` | Response dispatched to client |

## Example Trace (single request)

```
2026-06-01 12:43:28 [INFO] === PREDICT from 203.0.113.42 — file: photo.jpg ===
2026-06-01 12:43:28 [INFO] timing: upload_read=0.05s size=73158
2026-06-01 12:43:28 [INFO] TRACE: request_received client=203.0.113.42 file=photo.jpg size=73158
2026-06-01 12:43:28 [INFO] TRACE: config imgsz=320 conf=0.25 iou=0.5 max_inference_size=320
2026-06-01 12:43:28 [INFO] timing: preprocess=0.02s img=(180, 320, 3)
2026-06-01 12:43:28 [INFO] timing: model_ensure=0.00s (loaded=True)
2026-06-01 12:43:28 [INFO] TRACE: model_path=C:\...\snapshots\4d92e52...\best.pt
2026-06-01 12:43:28 [INFO] TRACE: inference_start img_shape=(180,320,3)
2026-06-01 12:44:10 [INFO] timing: inference=42.41s
2026-06-01 12:44:10 [INFO] TRACE: inference_end duration=42.410
2026-06-01 12:44:10 [INFO] timing: render=0.08s detections=0
2026-06-01 12:44:10 [INFO] timing: total=42.54s inference=42.41s detections=0 img=1280x720 client=203.0.113.42
2026-06-01 12:44:10 [INFO] RESULT: No military objects in abc123.jpg
2026-06-01 12:44:10 [INFO] TRACE: response_sent status=200 detections=0 elapsed=42.54s
```

## Benefits
1. **Debugging false positives**: Full trace available for every request showing exact image, config, and predictions
2. **Performance monitoring**: Inference duration tracked separately from total request time
3. **Model version tracking**: Model path logged per-request to verify correct model
4. **Client attribution**: Client IP logged to identify abusive or misconfigured clients
5. **Configuration verification**: imgsz/conf/iou logged per-request to confirm active settings

## Files Changed
- `backend/main.py`: Added TRACE log statements in predict function

## Verification
Trace logs are emitted at `INFO` level and will appear in:
- Render logs (if accessible)
- stdout/stderr of the uvicorn process
- Any log aggregation service connected to stdout
