#!/bin/bash
set -e

# If arguments are passed, run them directly (e.g. ccmcp --help, ccmcp status)
if [ $# -gt 0 ]; then
    exec python -m ccmcp "$@"
fi

# Default: wait for Qdrant, init, then run
echo "Waiting for Qdrant at $QDRANT_URL..."
until curl -sf "$QDRANT_URL/healthz" >/dev/null 2>&1; do
    sleep 2
done
echo "Qdrant ready."

python -m ccmcp init
exec python -m ccmcp run
