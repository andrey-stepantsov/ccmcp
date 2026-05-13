#!/bin/bash
set -e

echo "Waiting for Qdrant at $QDRANT_URL..."
until curl -sf "$QDRANT_URL/healthz" >/dev/null 2>&1; do
    sleep 2
done
echo "Qdrant ready."

python -m ccmcp setup
exec python -m ccmcp start
