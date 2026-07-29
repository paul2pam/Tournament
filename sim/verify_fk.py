"""Assert sim/fk.py matches MuJoCo's own kinematics across random poses. Exits nonzero on drift.

Run:  python -m sim.verify_fk
"""
import sys

import mujoco
import numpy as np

from sim.env import load_model
from sim.export_skeleton import build_skeleton
from sim.fk import forward


def main():
    m = load_model()
    d = mujoco.MjData(m)
    skel = build_skeleton(m)
    rng = np.random.default_rng(7)

    worst = 0.0
    for trial in range(50):
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] += rng.uniform(-1, 1, 3)
        q = rng.normal(size=4)
        d.qpos[3:7] = q / np.linalg.norm(q)
        d.qpos[7:] += rng.uniform(-1.5, 1.5, m.nq - 7)
        mujoco.mj_forward(m, d)

        ours = forward(skel, d.qpos[0:3], d.qpos[3:7], d.qpos[7:])
        for b in range(1, m.nbody):
            pos_err = np.abs(ours[b - 1][0] - d.xpos[b]).max()
            # quaternions are sign-ambiguous
            qq = ours[b - 1][1]
            quat_err = min(np.abs(qq - d.xquat[b]).max(), np.abs(qq + d.xquat[b]).max())
            worst = max(worst, pos_err, quat_err)

    print(f"max abs error across 50 random poses x {m.nbody - 1} bodies: {worst:.2e}")
    if worst > 1e-8:
        print("FAIL: FK does not match MuJoCo")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
