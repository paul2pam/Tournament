#!/usr/bin/env bash
# Push the repo to a CSL host and launch (or resume) Phase 0 training there.
#
#   CSL_HOST=netid@somehost.cs.illinois.edu ops/csl_train.sh [extra train_stand args]
#
# Off-campus: UIUC VPN (or a campus jump host) must be up for the ssh to connect.
# Assumes a bare host you can ssh into and run long jobs on (nohup). If CSL points
# you at a Slurm queue instead, wrap the nohup line in an sbatch script.
set -euo pipefail

: "${CSL_HOST:?set CSL_HOST=netid@host.cs.illinois.edu}"
REMOTE_DIR="${REMOTE_DIR:-~/tournament}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> sync repo -> $CSL_HOST:$REMOTE_DIR"
rsync -az --delete \
  --exclude .git --exclude .venv --exclude blobs --exclude __pycache__ \
  "$REPO_ROOT/" "$CSL_HOST:$REMOTE_DIR/"
# checkpoints/ is intentionally included: ships the current stand.pt so the
# cluster resumes from wherever the laptop left off.

echo "==> bootstrap venv + launch training"
ssh "$CSL_HOST" REMOTE_DIR="$REMOTE_DIR" 'bash -s' -- "$@" <<'REMOTE'
set -euo pipefail
cd "$REMOTE_DIR"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet mujoco==3.2.7 torch numpy fastapi 'uvicorn[standard]' asyncpg httpx
fi
mkdir -p checkpoints logs
RESUME_ARGS=""
[ -f checkpoints/stand.pt ] && RESUME_ARGS="--resume checkpoints/stand.pt"
pkill -f 'sim.train_stand' 2>/dev/null || true
nohup nice .venv/bin/python -m sim.train_stand $RESUME_ARGS \
  --total-steps 20000000 --num-envs 48 --out checkpoints/stand.pt "$@" \
  > logs/train_stand.log 2>&1 &
echo "launched pid $! — tail with: ssh <host> tail -f $REMOTE_DIR/logs/train_stand.log"
REMOTE
