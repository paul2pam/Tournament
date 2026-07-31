#!/usr/bin/env bash
# Launch SAC standing training on a CSL GPU host (default cm009, to keep the
# PPO run on cm008 undisturbed). Same sandbox rules as csl_train.sh.
#
#   ops/csl_sac.sh [host] [extra train_sac args]
set -euo pipefail

HOST="${1:-cm009.csl.illinois.edu}"
shift || true
CSL_USER="${CSL_USER:-kaiyan3}"
REMOTE_DIR="/home/${CSL_USER}/paulcp2/tournament"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> sync repo -> ${CSL_USER}@${HOST}:${REMOTE_DIR}"
rsync -az --delete \
  --exclude .git --exclude .venv --exclude env --exclude blobs --exclude __pycache__ --exclude logs \
  "$REPO_ROOT/" "${CSL_USER}@${HOST}:${REMOTE_DIR}/"

echo "==> bootstrap env (CUDA torch) + launch SAC"
ssh "${CSL_USER}@${HOST}" REMOTE_DIR="$REMOTE_DIR" 'bash -s' -- "$@" <<'REMOTE'
set -euo pipefail
cd "$REMOTE_DIR"
mkdir -p logs
if command -v conda >/dev/null 2>&1 || { [ -f /etc/profile.d/conda.sh ] && source /etc/profile.d/conda.sh; }; then
  [ -d "$REMOTE_DIR/env" ] || conda create -y -p "$REMOTE_DIR/env" python=3.11 >/dev/null
else
  [ -d "$REMOTE_DIR/env" ] || python3 -m venv "$REMOTE_DIR/env"
fi
PYBIN="$REMOTE_DIR/env/bin"
# default PyPI torch wheel bundles CUDA — needed for the L40S
"$PYBIN/pip" install --quiet mujoco==3.2.7 numpy torch

# pick the GPU with the most free memory (shared box — stay out of the busy ones)
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)
echo "using GPU $GPU"

pkill -f "paulcp2/tournament.*sim.train_sac" 2>/dev/null || true
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 nohup nice "$PYBIN/python" -m sim.train_sac \
  --total-steps 3000000 --num-envs 8 --out checkpoints/stand_sac.pt "$@" \
  > logs/train_sac.log 2>&1 &
echo "launched pid $! on $(hostname)"
REMOTE

echo "==> tail: ssh ${CSL_USER}@${HOST} tail -f ${REMOTE_DIR}/logs/train_sac.log"
