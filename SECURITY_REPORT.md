# SECURITY_REPORT.md

## Audit Scope
Full repository scan for secrets, credentials, hardcoded values, local paths, and misconfigurations.

## Findings

### 1. Secrets in Source Code
| Finding | File | Line | Severity | Status |
|---------|------|------|----------|--------|
| No hardcoded API keys | — | — | ✅ Safe | Verified |
| No hardcoded passwords | — | — | ✅ Safe | Verified |
| No hardcoded tokens | — | — | ✅ Safe | Verified |

### 2. Environment Variables
| Variable | Source | Used In | Security |
|----------|--------|---------|----------|
| `HF_TOKEN` | `.env.example`, `render.yaml` | `backend/main.py:44` | ✅ Read from env var, never hardcoded |
| `PORT` | `.env.example`, `render.yaml`, `Procfile` | Start command | ✅ Safe |
| `ALLOWED_ORIGINS` | `.env.example` | `backend/main.py:13` | ✅ Safe, has localhost defaults |

### 3. Hardcoded Localhost URLs
| Location | Code | Severity | Action |
|----------|------|----------|--------|
| `backend/main.py:13` | `os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,...")` | 🟡 Low | Default fallback only; overridden by env var in production |
| `frontend/index.html:169` | `fetch("/predict")` | 🟡 Low | Relative URL — works in dev; must be configured for production |

### 4. `.gitignore` Verification
| Pattern | Status | Notes |
|---------|--------|-------|
| `.env` | ✅ Included | Blocks .env from commit |
| `.env.*` | ✅ Included | Blocks .env.local, .env.prod, etc. |
| `!.env.example` | ✅ Included | Allows .env.example to be tracked |
| `uploads/` | ✅ Included | Prevents user uploads from entering repo |
| `outputs/` | ✅ Included | Prevents result images from entering repo |
| `*.pt` | ✅ Included | Prevents model weights from entering repo |
| `venv/` | ✅ Included | Prevents virtual environment from entering repo |
| `__pycache__/` | ✅ Included | Prevents Python cache from entering repo |

### 5. Input Validation Risks
| Risk | Location | Severity | Description |
|------|----------|----------|-------------|
| No file size limit | `backend/main.py:107-108` | 🟠 Medium | Large uploads can cause OOM or disk fill |
| No file type validation (content) | `backend/main.py:97-98` | 🟠 Medium | Only validates extension, not MIME type |
| No rate limiting | `POST /predict` | 🟠 Medium | No protection against repeated calls |
| No authentication | All endpoints | 🟠 Medium | API is fully public |

### 6. Information Leakage
| Risk | Location | Severity | Description |
|------|----------|----------|-------------|
| Stack traces in error responses | `backend/main.py:170` | 🟡 Low | `JSONResponse({"error": str(e)})` may leak internals |
| Model path in logs | `backend/main.py:47` | 🟢 Info | Prints local cache path (not exposed externally) |

### 7. `.dockerignore` Verification
| Pattern | Status |
|---------|--------|
| `.env`, `.env.*` | ✅ Included |
| `uploads`, `outputs` | ✅ Included |
| `*.pt`, `*.pth` | ✅ Included |
| `venv` | ✅ Included |

## Recommendations

### Pre-Deployment (Required)
1. ✅ Already done: `.env` is in `.gitignore`
2. ✅ Already done: `HF_TOKEN` read from environment variable
3. ⚠️ **Set `ALLOWED_ORIGINS` in Render** to the Vercel frontend URL
4. ⚠️ **Configure `HF_TOKEN`** as a Render Environment Secret (not env var)

### Recommended Improvements (Non-Blocking)
1. Add file size limit (e.g., `max_length=10485760` for 10MB)
2. Add MIME type validation (check `file.content_type`)
3. Add rate limiting middleware
4. Sanitize error messages in production (return generic error, log details)
5. Add cleanup mechanism for `uploads/` and `outputs/` directories

## Conclusion
**No critical security vulnerabilities found.** The repository follows security best practices for secrets management. Minor improvements recommended for production hardening.
