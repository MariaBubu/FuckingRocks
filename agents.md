# AI Coding Agent Guidelines & Commitments

This document defines the strict constraints, behavior policies, and working guidelines for all AI agents (including Antigravity, Claude, or other assistants) interacting with the **FuckingRocks** repository.

---

## 1. Git & GitHub Policies (CRITICAL)

> [!IMPORTANT]
> **NO UNAUTHORIZED GIT COMMIT, PUSH, OR REMOTE ACTIONS.**
> You must **never** perform any of the following operations without explicit, step-by-step review and active approval from the USER in the chat:
> - `git commit`
> - `git push`
> - `git remote` modifications
> - Deleting, moving, or modifying branches on remote repositories.
> - Staging or committing code changes.
>
> Every commit and push must be explicitly requested or approved by the user *before* execution.

---

## 2. Web Application Core Architectural Rules

1. **Instant Web App Startup (under 0.5s):**
   - The Flask/HTTP web server must bind and listen to the port instantly.
   - Any heavy library imports (e.g., `torch`, `transformers`) or model preloading must be managed to prevent startup lag or browser connection timeouts.

2. **macOS Apple Silicon & OpenMP Safety:**
   - Importing PyTorch (`torch`) in background daemon threads frequently deadlocks on macOS.
   - Always perform primary heavy imports synchronously on the main thread, or enforce single-threading via environment variables:
     ```bash
     export OMP_NUM_THREADS=1
     export MKL_NUM_THREADS=1
     export OPENBLAS_NUM_THREADS=1
     export VECLIB_MAXIMUM_THREADS=1
     export NUMEXPR_NUM_THREADS=1
     ```

3. **Separate On-Demand Large AI Models:**
   - Keep the main classification model (ResNet18) lightweight and separate.
   - Any large zero-shot detection or bounding-box models (approx. 600MB) must **never** load automatically at startup. Provide dedicated "Load Model" buttons in the UI to load them asynchronously on demand.

4. **Robust Process Management (Clean exits):**
   - Always ensure a bash `trap` is defined in startup scripts to kill background Python server processes cleanly when the user hits **Ctrl+C**:
     ```bash
     trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
     ```

---
