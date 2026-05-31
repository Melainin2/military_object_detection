# AI Detection Dashboard

A production-ready AI detection dashboard for military object detection using YOLO and FastAPI. This repository contains a lightweight frontend, API back end, and deployment tooling for private GitHub hosting.

## Features

- Upload images for object detection
- Detect military-related objects using YOLO
- Bounding box overlay on uploaded images
- Detection summary dashboard with chart visualization
- FastAPI backend with secure environment support
- Docker-ready and Render-ready deployment configuration

## Technologies Used

- FastAPI
- Uvicorn
- Ultralytics YOLO
- OpenCV
- Hugging Face Hub
- HTML / CSS / JavaScript
- Chart.js
- Docker
- Render

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Melainin2/military_object_detection.git
   cd AI-Detection-Dashboard
   ```
2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the environment:
   - Windows PowerShell:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create a `.env` file from `.env.example` and set your Hugging Face token:
   ```bash
   copy .env.example .env
   ```

## Usage

Start the application locally:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open your browser at:

```text
http://127.0.0.1:8000
```

Then upload an image and submit it for object detection.

## Screenshots

> Add screenshots here after running the application locally.

- `screenshots/demo-1.png`
- `screenshots/demo-2.png`

## Deployment

### Docker

Build and run with Docker:

```bash
docker build -t ai-detection-dashboard .
docker run -p 8000:8000 ai-detection-dashboard
```

### Vercel

The frontend can be deployed on Vercel as a static site from the `frontend/` folder. Use the following settings:

- Framework preset: `Other`
- Root directory: `frontend`
- Build command: none
- Output directory: `.`

Update `frontend/index.html` to use your Render backend URL in the `BACKEND_URL` constant.

### Render

This repository includes `render.yaml` for a Render web service. Configure the `HF_TOKEN` secret in Render, then deploy using the dashboard.

Render deployment notes (Docker vs Python runtime)

- Option A — Render using Docker (common if you choose Docker on the Render dashboard):
   - Set **Root Directory** to `backend` (or the directory that contains the `Dockerfile`).
   - Set **Environment** to `Docker`.
   - Ensure there is a `Dockerfile` in that root. This repository provides `backend/Dockerfile` to support the case where the service root is `backend/` and Render builds using Docker.
   - When Root Directory is `backend`, Render will look for `Dockerfile` there; if the Dockerfile cannot be found you will get the `open Dockerfile: no such file or directory` error.

- Option B — Render using the Python runtime (no Docker):
   - Set **Environment** to `Python` (not Docker).
   - Use these commands in Render web service settings (or via `render.yaml`):

      Build Command:

      ```bash
      pip install --no-cache-dir -r requirements.txt
      ```

      Start Command:

      ```bash
      uvicorn backend.main:app --host 0.0.0.0 --port $PORT
      ```

Choose Option A if you want full control of container image; choose Option B for a simpler Render-managed Python deployment. If Render reports `open Dockerfile: no such file or directory`, either change the service to Python runtime or ensure the `Dockerfile` exists in the selected Root Directory (this repo includes `backend/Dockerfile`).

### Deployment flow

- Frontend: Vercel static site from `frontend/`
- Backend: Render service running FastAPI on `backend/main.py`
- Set `HF_TOKEN` and `ALLOWED_ORIGINS` in Render secrets/environments

## Security Recommendations

- Keep the repository private
- Never commit `.env` or model weights
- Do not push `uploads/` or `outputs/`
- Use GitHub Secrets or Render environment variables for `HF_TOKEN`

## Notes

- The backend uses FastAPI, not Flask. The application is optimized for modern async deployment with Uvicorn.
