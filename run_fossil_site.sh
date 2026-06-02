#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Force single-threaded initialization for PyTorch backend dependencies (prevents OpenMP thread pool deadlocks on macOS)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Kill any existing server running on port 5050 to prevent stale processes from blocking
PORT=5050
if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
    echo "Port $PORT is already in use. Killing the stale server process..."
    EXISTING_PID=$(lsof -t -i :"$PORT" || true)
    if [ -n "$EXISTING_PID" ]; then
        kill -9 $EXISTING_PID 2>/dev/null || true
        sleep 1
    fi
fi

# Also kill any lingering fossil_web_app.py processes that might be stuck importing
pkill -9 -f "fossil_web_app.py" 2>/dev/null || true
sleep 0.5

echo "============================================================"
echo "🚀 Starting Fossil Web App (Instant Start Mode)"
echo "   Server starts in < 1 second."
echo "   Classifier model warms up automatically in the background."
echo "============================================================"

# Start the web server in the background
.venv/bin/python -u fossil_web_app.py &
SERVER_PID=$!

# Ensure that if the script exits (normal or Ctrl+C), it kills the Python server
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Poll for server readiness (should be < 1 second now!)
START_TIME=$(date +%s)
READY=false

for i in {1..10}; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    printf "\r⏳ Waiting for server to bind... %ds" "$ELAPSED"
    
    if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
        READY=true
        break
    fi
    sleep 0.3
done

printf "\n"

if [ "$READY" = true ]; then
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo "============================================================"
    echo "✅ Server ready in ${ELAPSED}s!"
    echo "🌐 Opening http://127.0.0.1:$PORT/"
    echo ""
    echo "   ℹ️  No PyTorch is loaded by the website."
    echo "      It uses ONNX if available, otherwise the NumPy fallback."
    echo "============================================================"
    open "http://127.0.0.1:$PORT/"
else
    echo "============================================================"
    echo "⚠️  WARNING: Server did not bind to port $PORT within 3s."
    echo "   Opening browser anyway — check terminal for errors."
    echo "============================================================"
    open "http://127.0.0.1:$PORT/"
fi

# Wait on the python process so Ctrl+C terminates it normally
wait "$SERVER_PID"
