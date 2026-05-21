#!/usr/bin/env bash
# End-to-end test runner.
#
# Builds the Docker image from source, starts an isolated test stack,
# runs the E2E test suite, then tears everything down.
#
# Usage:
#   scripts/test-docker.sh              # run all e2e tests
#   scripts/test-docker.sh -k snapshot  # run only tests matching "snapshot"
#
# The stack uses:
#   - docker-compose.test.yml (build from source, isolated volumes)
#   - project name "ccmcp-test" (separate from the production stack)
#   - MCP server on port 7701 (production uses 7700)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker-compose.test.yml"
PROJECT="ccmcp-test"
MCP_PORT=7701
QDRANT_PORT=6334

COMPOSE="docker compose -p $PROJECT -f $COMPOSE_FILE"

export CCMCP_TEST_COMPOSE_FILE="$COMPOSE_FILE"
export CCMCP_TEST_PROJECT="$PROJECT"
export CCMCP_TEST_MCP_URL="http://localhost:$MCP_PORT"
export CCMCP_TEST_QDRANT_URL="http://localhost:$QDRANT_PORT"

# Always tear down on exit (success or failure)
cleanup() {
    echo ""
    echo "=== Tearing down test stack ==="
    $COMPOSE down -v --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Building image from source ==="
$COMPOSE build --quiet

echo "=== Starting test stack ==="
$COMPOSE up -d

echo "=== Waiting for stack to be healthy ==="
# Wait up to 3 minutes for the ccmcp service to pass its healthcheck
deadline=$(( $(date +%s) + 180 ))
while true; do
    status=$($COMPOSE ps ccmcp --format json 2>/dev/null \
        | python3 -c "import sys,json; data=sys.stdin.read(); \
          rows=[json.loads(l) for l in data.splitlines() if l.strip()]; \
          print(rows[0].get('Health','') if rows else '')" 2>/dev/null || echo "")
    if [ "$status" = "healthy" ]; then
        echo "Stack is healthy."
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "ERROR: Stack did not become healthy within 3 minutes."
        echo "Container logs:"
        $COMPOSE logs --tail=50 ccmcp
        exit 1
    fi
    echo "  Waiting... (status: ${status:-starting})"
    sleep 5
done

echo ""
echo "=== Running E2E tests ==="
cd "$REPO_ROOT"

# Resolve pytest: prefer the project venv, then PATH, then uv run
if [ -f "$REPO_ROOT/.venv/bin/pytest" ]; then
    PYTEST="$REPO_ROOT/.venv/bin/pytest"
elif command -v pytest >/dev/null 2>&1; then
    PYTEST="pytest"
elif command -v uv >/dev/null 2>&1; then
    PYTEST="uv run pytest"
else
    echo "ERROR: pytest not found. Run: uv pip install -e '.[dev]'"
    exit 1
fi

$PYTEST tests/test_e2e.py -m e2e -v "$@"
