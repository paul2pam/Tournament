"""Export the humanoid's kinematic tree + geoms to web/skeleton.json (one-time).

The browser reconstructs body world transforms from (root_pos, root_quat, joints[21])
via the same forward kinematics as sim/fk.py, then draws each body's geoms.

Run:  python -m sim.export_skeleton
"""
import json

import mujoco

from common.config import REPO_ROOT
from sim.env import load_model

GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
}


def build_skeleton(model=None) -> dict:
    m = model or load_model()
    bodies = []
    for b in range(1, m.nbody):  # skip world
        joints = []
        has_free = False
        for j in range(m.njnt):
            if m.jnt_bodyid[j] != b:
                continue
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                has_free = True
                continue
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, "only hinge joints supported"
            joints.append(
                {
                    "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                    "pos": m.jnt_pos[j].tolist(),
                    "axis": m.jnt_axis[j].tolist(),
                    "qpos_idx": int(m.jnt_qposadr[j]) - 7,  # index into the 21-float joints array
                }
            )
        geoms = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            geoms.append(
                {
                    "type": GEOM_TYPES[m.geom_type[g]],
                    "pos": m.geom_pos[g].tolist(),
                    "quat": m.geom_quat[g].tolist(),   # (w, x, y, z)
                    "size": m.geom_size[g].tolist(),
                }
            )
        bodies.append(
            {
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                "parent": int(m.body_parentid[b]) - 1,  # -1 => world
                "pos": m.body_pos[b].tolist(),
                "quat": m.body_quat[b].tolist(),
                "free": has_free,
                "joints": joints,
                "geoms": geoms,
            }
        )
    return {"bodies": bodies}


def main():
    out = REPO_ROOT / "web" / "skeleton.json"
    out.write_text(json.dumps(build_skeleton(), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
