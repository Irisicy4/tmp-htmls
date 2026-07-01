#!/usr/bin/env bash
# Serve this report over HTTP so it can be viewed from a browser.
#
# On this machine (h200):
#   ./serve.sh                # serves on http://127.0.0.1:8899
#
# From your laptop, to view it via SSH tunnel (no need to open a firewall port):
#   ssh -L 8899:localhost:8899 h200
#   # then in your local browser open:
#   http://localhost:8899
#
# (assumes your local ~/.ssh/config has a Host entry named "h200" pointing at
#  this machine; swap in the real hostname/user if not.)
set -euo pipefail
PORT="${1:-8899}"
cd "$(dirname "$0")"
echo "Serving $(pwd) on http://127.0.0.1:${PORT}  (Ctrl+C to stop)"
python3 -m http.server "${PORT}" --bind 127.0.0.1
