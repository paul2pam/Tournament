"""Degeneracy filter: high precision, narrow scope (spec §5).

Rejects ONLY mechanical failure modes — never "uninteresting":
  nan        non-finite state
  blowup     physics explosion (absurd speeds or positions)
  collapsed  the clip is essentially a body lying on the ground
  frozen     visually static — no joint motion and no root motion at all
             (note: "zero net displacement" per spec is interpreted as *frozen*, not
              "stationary" — a standing policy micro-adjusts and must pass)

The 15% leak of rejects is applied by the caller (worker), which flags leaked clips
so the filter can be audited against votes.
"""
import numpy as np

from common.config import CLIP_FPS

TORSO_DOWN_Z = 0.55       # below this the torso is on/near the ground
COLLAPSED_FRAC = 0.90     # fraction of down frames that makes a clip "collapsed"
BLOWUP_LIN_SPEED = 20.0   # m/s root speed
BLOWUP_JOINT_SPEED = 100.0  # rad/s finite-diff joint speed
BLOWUP_Z = 3.0            # m
FROZEN_JOINT_SPEED = 0.05  # rad/s — max joint speed below this over the whole clip
FROZEN_ROOT_DISP = 0.01   # m


def check_clip(frames: np.ndarray) -> str | None:
    """frames: (T, 28) qpos window. Returns rejection reason or None if clean."""
    if not np.isfinite(frames).all():
        return "nan"

    root_xyz = frames[:, 0:3]
    joints = frames[:, 7:]
    dt = 1.0 / CLIP_FPS

    lin_speed = np.linalg.norm(np.diff(root_xyz, axis=0), axis=1) / dt
    joint_speed = np.abs(np.diff(joints, axis=0)) / dt
    if (
        lin_speed.max(initial=0) > BLOWUP_LIN_SPEED
        or joint_speed.max(initial=0) > BLOWUP_JOINT_SPEED
        or np.abs(root_xyz[:, 2]).max() > BLOWUP_Z
    ):
        return "blowup"

    if (root_xyz[:, 2] < TORSO_DOWN_Z).mean() > COLLAPSED_FRAC:
        return "collapsed"

    root_disp = np.linalg.norm(root_xyz[-1] - root_xyz[0])
    if joint_speed.max(initial=0) < FROZEN_JOINT_SPEED and root_disp < FROZEN_ROOT_DISP:
        return "frozen"

    return None
