#!/usr/bin/env bash
# Pull the latest trained checkpoint (and training log tail) back from CSL.
#
#   CSL_HOST=netid@somehost.cs.illinois.edu ops/csl_pull.sh
set -euo pipefail

: "${CSL_HOST:?set CSL_HOST=netid@host.cs.illinois.edu}"
REMOTE_DIR="${REMOTE_DIR:-~/tournament}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az "$CSL_HOST:$REMOTE_DIR/checkpoints/stand.pt" "$REPO_ROOT/checkpoints/stand.pt"
echo "==> checkpoint pulled; recent training log:"
ssh "$CSL_HOST" "tail -5 $REMOTE_DIR/logs/train_stand.log"
echo "==> evaluate locally with: .venv/bin/python -m sim.eval_stand"
