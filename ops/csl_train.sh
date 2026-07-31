#!/usr/bin/env bash
# Push the repo to a CSL host and launch (or resume) Phase 0 training there.
#
#   ops/csl_train.sh [host] [extra train_stand args]
#   e.g. ops/csl_train.sh az007.csl.illinois.edu
#
# Access per Kai (shared account): login kaiyan3, ALL files must stay under
# /home/kaiyan3/paulcp2, own conda env, never touch anything outside the folder,
# never use others' conda envs. Off-campus: UIUC VPN must be up.
set -euo pipefail

HOST="${1:-az007.csl.illinois.edu}"
shift || true
CSL_USER="${CSL_USER:-kaiyan3}"
SANDBOX="/home/${CSL_USER}/paulcp2"
REMOTE_DIR="${SANDBOX}/tournament"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> sync repo -> ${CSL_USER}@${HOST}:${REMOTE_DIR}"
rsync -az --delete \
  --exclude .git --exclude .venv --exclude env --exclude env-gpu --exclude blobs \
  --exclude __pycache__ --exclude logs \
  "$REPO_ROOT/" "${CSL_USER}@${HOST}:${REMOTE_DIR}/"
# checkpoints/ ships intentionally: the cluster resumes from the laptop's stand.pt.

echo "==> bootstrap env + launch training"
ssh "${CSL_USER}@${HOST}" REMOTE_DIR="$REMOTE_DIR" 'bash -s' -- "$@" <<'REMOTE'
set -euo pipefail
cd "$REMOTE_DIR"
mkdir -p logs

# Own environment, prefix-based so it lives INSIDE our sandbox folder
# (a plain `conda create -n` would write to the shared ~/.conda — off-limits).
PYBIN=""
if command -v conda >/dev/null 2>&1 || { [ -f /etc/profile.d/conda.sh ] && source /etc/profile.d/conda.sh; }; then
  if [ ! -d "$REMOTE_DIR/env" ]; then
    conda create -y -p "$REMOTE_DIR/env" python=3.11 >/dev/null
  fi
  PYBIN="$REMOTE_DIR/env/bin"
else
  echo "conda not found on PATH; falling back to venv (still inside sandbox)"
  [ -d "$REMOTE_DIR/env" ] || python3 -m venv "$REMOTE_DIR/env"
  PYBIN="$REMOTE_DIR/env/bin"
fi

# CPU-only torch: this workload is MuJoCo-stepping-bound, and CPU wheels save
# ~2GB on a shared disk we were asked to be careful with.
"$PYBIN/pip" install --quiet mujoco==3.2.7 numpy \
  torch --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

# Kill only OUR previous run: pattern includes the sandbox path so we can never
# match another user's process on this shared account.
pkill -f "paulcp2/tournament.*sim.train_stand" 2>/dev/null || true

CORES=$(nproc)
ENVS=$(( CORES > 24 ? 48 : CORES * 2 ))
RESUME_ARGS=""
[ -f checkpoints/stand.pt ] && RESUME_ARGS="--resume checkpoints/stand.pt"
cd "$REMOTE_DIR"
OMP_NUM_THREADS=8 nohup nice "$PYBIN/python" -m sim.train_stand $RESUME_ARGS --subproc \
  --total-steps 20000000 --num-envs "$ENVS" --out checkpoints/stand.pt "$@" \
  > logs/train_stand.log 2>&1 &
echo "launched pid $! on $(hostname) with $ENVS envs ($CORES cores)"
REMOTE

echo "==> tail: ssh ${CSL_USER}@${HOST} tail -f ${REMOTE_DIR}/logs/train_stand.log"
