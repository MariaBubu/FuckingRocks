#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Start the web server in the background
.venv/bin/python fossil_web_app.py &
SERVER_PID=$!

# Give the server a moment to bind to the port
sleep 1.5

# Automatically open the web app in the default macOS browser
open "http://127.0.0.1:5050/"

# Wait on the python process so Ctrl+C terminates it normally
wait "$SERVER_PID"
