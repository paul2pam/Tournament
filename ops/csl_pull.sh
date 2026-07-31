#!/usr/bin/env bash
# Pull the latest trained checkpoint (and training log tail) back from CSL.
#
#   ops/csl_pull.sh [host]
set -euo pipefail

HOST="${1:-az007.csl.illinois.edu}"
CSL_USER="${CSL_USER:-kaiyan3}"
CSL_HOST="${CSL_USER}@${HOST}"
REMOTE_DIR="/home/${CSL_USER}/paulcp2/tournament"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az "$CSL_HOST:$REMOTE_DIR/checkpoints/stand.pt" "$REPO_ROOT/checkpoints/stand.pt"
echo "==> checkpoint pulled; recent training log:"
ssh "$CSL_HOST" "tail -5 $REMOTE_DIR/logs/train_stand.log"
echo "==> evaluate locally with: .venv/bin/python -m sim.eval_stand"
