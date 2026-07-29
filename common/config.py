"""Shared configuration. Everything overridable via env vars; defaults match ops/docker-compose.yml."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://humanoid:humanoid@localhost:5433/humanoid"
)

BLOB_DIR = Path(os.environ.get("BLOB_DIR", REPO_ROOT / "blobs"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", REPO_ROOT / "checkpoints"))

# Server-side salt for hashing voter IPs. Not secret-critical in dev.
IP_HASH_SALT = os.environ.get("IP_HASH_SALT", "dev-salt")

# Clip/rollout geometry. 240Hz physics, control+record every 8 steps -> exactly 30fps.
SIM_TIMESTEP = 1.0 / 240.0
FRAME_SKIP = 8                     # control rate = 30 Hz
CLIP_FPS = 30
CLIP_SECONDS = 3.0
CLIP_FRAMES = int(CLIP_FPS * CLIP_SECONDS)        # 90
EPISODE_SECONDS = 10.0
EPISODE_FRAMES = int(CLIP_FPS * EPISODE_SECONDS)  # 300
SETTLE_SECONDS = 0.5               # skip reset-pose settling at episode start
CLIPS_PER_POLICY = 4

# Contest resolution (worker/contests.py)
CONTEST_MAX_COMPARISONS = 20
CONTEST_POSTERIOR_THRESHOLD = 0.95

# Challenger pool (env-overridable so the Phase 1 harness can run small/fast)
POOL_SIZE = int(os.environ.get("POOL_SIZE", 6))
PERTURB_SCALES = [0.01, 0.02, 0.05, 0.08, 0.12, 0.2]
FINETUNE_STEPS = int(os.environ.get("FINETUNE_STEPS", 40_960))

# Pair queue refill target (unconsumed rows)
QUEUE_TARGET = int(os.environ.get("QUEUE_TARGET", 200))

# Degeneracy filter
LEAK_RATE = 0.15

# Pair queue composition
QUEUE_SHARE_CONTEST = 0.80
QUEUE_SHARE_ANCESTOR = 0.15
QUEUE_SHARE_CHECK = 0.05
