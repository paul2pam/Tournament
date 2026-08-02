"""Thin MuJoCo wrapper around the standard humanoid for training and rollout.

Physics at 240 Hz, control (and frame recording) every FRAME_SKIP=8 steps -> exactly 30 Hz,
so recorded qpos frames are the clip frames with no resampling.
"""
import numpy as np
import mujoco

from common.config import SIM_TIMESTEP, FRAME_SKIP, REPO_ROOT

MODEL_PATH = str(REPO_ROOT / "sim" / "assets" / "humanoid.xml")

_MODEL = None


def load_model() -> mujoco.MjModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = mujoco.MjModel.from_xml_path(MODEL_PATH)
        _MODEL.opt.timestep = SIM_TIMESTEP
    return _MODEL


class HumanoidEnv:
    """obs = qpos[2:] ++ qvel (53,); action in [-1,1]^21.

    task='stand'   — Phase 0 pretraining: height shaping, healthy-z termination.
    task='neutral' — challenger fine-tune viability repair: alive + control cost
                     only, terminate only on physics failure. The fine-tune must
                     hold NO opinion about posture or motion style — a crawling
                     (or someday backflipping) lineage's children must not be
                     dragged back toward standing (spec §7: repair, don't steer).
    """

    HEALTHY_Z = (0.9, 2.0)
    RESET_NOISE = 0.01
    BLOWUP_QVEL = 100.0

    def __init__(self, seed: int = 0, task: str = "stand"):
        assert task in ("stand", "neutral")
        self.task = task
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.obs_dim = (self.model.nq - 2) + self.model.nv   # 26 + 27 = 53
        self.act_dim = self.model.nu                          # 21

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos[2:], self.data.qvel]).astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        n = self.RESET_NOISE
        self.data.qpos[:] += self.rng.uniform(-n, n, size=self.model.nq)
        self.data.qvel[:] += self.rng.uniform(-n, n, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

        z = self.data.qpos[2]
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        vxy = self.data.qvel[:2]

        if self.task == "stand":
            reward = (
                5.0                                   # alive bonus
                + 2.0 * min(z, 1.4)                   # height shaping, capped
                - 0.1 * float(np.square(action).sum())
                - 0.05 * float(np.square(vxy).sum())
            )
            terminated = not (finite and self.HEALTHY_Z[0] <= z <= self.HEALTHY_Z[1])
        else:  # neutral: sim-stability only
            reward = 2.0 - 0.05 * float(np.square(action).sum())
            terminated = not finite or np.abs(self.data.qvel).max() > self.BLOWUP_QVEL or z > 3.0
        return self._obs(), reward, terminated

    # Used by rollout.py: full generalized coordinates for the trajectory format.
    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()
