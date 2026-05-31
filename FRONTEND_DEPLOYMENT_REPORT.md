# FRONTEND_DEPLOYMENT_REPORT.md

## Deployment Strategy: Vercel Static Site

**Decision**: Deploy as a static site on Vercel.

**Rationale**:
- Frontend is a single HTML file with no build system
- No framework dependencies (pure HTML/CSS/JS)
- Chart.js loaded from CDN
- Vercel static hosting is free and reliable

## Files Verified

| File | Status | Notes |
|------|--------|-------|
| `frontend/index.html` | ✅ Updated | Added BACKEND_URL configuration |
| `vercel.json` | ✅ Ready | Static deployment, routes to frontend/ |

## Changes Made

### 1. Configurable Backend URL
Added `BACKEND_URL` constant at top of `frontend/index.html`:
```javascript
const BACKEND_URL = "";    // local: empty (same-origin)
                           // production: "https://your-app.onrender.com"
```

### 2. API Calls Use BACKEND_URL
Changed from:
```javascript
const response = await fetch("/predict", { ... });
```
To:
```javascript
const response = await fetch(BACKEND_URL + "/predict", { ... });
```

## Vercel Configuration

**vercel.json**:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "frontend/**/*",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/frontend/$1"
    }
  ]
}
```

### Vercel Deployment Settings
| Setting | Value |
|---------|-------|
| Framework Preset | Other |
| Root Directory | `frontend` |
| Build Command | None |
| Output Directory | `.` |
| Install Command | None |

## Pre-Deployment Steps
1. Edit `frontend/index.html` line ~13: Set `BACKEND_URL` to your Render backend URL
2. Deploy to Vercel using GitHub import or `vercel` CLI

## CORS Configuration
After deploying to Vercel:
1. Note the Vercel URL (e.g., `https://ai-detection-dashboard.vercel.app`)
2. Set `ALLOWED_ORIGINS` in Render to that URL
3. This ensures the browser allows cross-origin requests from frontend to backend

## Deployment Flow
```
User → Vercel (frontend) → API call to Render (backend) → YOLO inference → Response
```

## No Build Step
The frontend is pure HTML/CSS/JS with Chart.js via CDN. No npm, webpack, or build tools are required.
