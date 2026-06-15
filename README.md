# Military Object Detection – AI Detection Dashboard

Dual-model (ViT + YOLO) pipeline for military object detection with FastAPI backend and lightweight web UI.

## Features

- **Dual-model pipeline**: ViT (ResNet50) for scene-level classification + YOLOv8 (ONNX) for object localization
- **Smart cascade logic**: ViT acts as gatekeeper; "Soldier Detected" if either model confirms military content
- **Web UI**: Drag-and-drop image upload, bounding box visualization, confidence scores
- **FastAPI backend**: Async inference, ONNX runtime, automatic model caching
- **Ready for deployment**: Docker, Render, Hugging Face Spaces

## Technologies

- FastAPI + Uvicorn
- PyTorch + TorchVision (ResNet50 / ImageNet)
- ONNX Runtime (YOLOv8)
- OpenCV
- HTML / CSS / JavaScript + Chart.js

## Installation

```bash
git clone https://github.com/Melainin2/military_object_detection.git
cd military_object_detection
pip install -r requirements.txt
```

## Usage

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** and upload an image.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status (YOLO + ViT) |
| `/predict` | POST | Upload image, returns detections + ViT analysis |
| `/` | GET | Frontend UI |
| `/outputs/{file}` | GET | Annotated image |

### `/predict` response

```json
{
  "message": "Soldier Detected | No Soldier Detected",
  "detections": [{"class_name": "...", "confidence": 0.95, "box": [x1,y1,x2,y2]}],
  "vit_military_score": 0.96,
  "vit_confidence": 0.86,
  "vit_top_predictions": [{"class": "...", "probability": 0.86, "is_military": true}]
}
```

## Pipeline Logic

1. **ViT (ResNet50)**: Classifies image into 1000 ImageNet classes. Checks if top-20 predictions include military classes (24 ID'd via keyword/synset matching).
2. **YOLO (ONNX)**: Detects military objects among 12 classes (soldier, tank, aircraft, etc.).
3. **Decision**:
   - ViT *or* YOLO finds military content → **"Soldier Detected"**
   - Neither finds anything → **"No Soldier Detected"**
   - ViT trumps YOLO-negative (avoids false negatives)

## Security

- Keep the repository private
- Never commit `.env` or model weights
- `uploads/` and `outputs/` are gitignored

## Notes

- Models download automatically on first run (YOLO ONNX from HuggingFace, ResNet50 from PyTorch)
- ViT runs on CPU (PyTorch); GPU support available if detected
