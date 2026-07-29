"""Reference forward kinematics over skeleton.json — the exact math web/renderer.js ports.

Mirrors mj_kinematics for a tree of hinge joints under a free root:
  child init:  xquat = parent_xquat * body_quat ; xpos = parent_xpos + rot(parent_xquat, body_pos)
  each hinge (in listed order):
      anchor = xpos + rot(xquat, jnt_pos)
      xquat  = xquat * axisangle(jnt_axis_local, angle)
      xpos   = anchor - rot(xquat, jnt_pos)
Verified against mujoco's xpos/xquat by sim/verify_fk.py.
"""
import numpy as np


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_rot(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def forward(skel: dict, root_pos, root_quat, joints):
    """Returns list of (xpos, xquat) per body, in skeleton.json body order."""
    out = []
    for body in skel["bodies"]:
        if body["free"]:
            xpos = np.asarray(root_pos, dtype=float)
            xquat = np.asarray(root_quat, dtype=float)
        else:
            ppos, pquat = out[body["parent"]]
            xquat = quat_mul(pquat, np.asarray(body["quat"], dtype=float))
            xpos = ppos + quat_rot(pquat, np.asarray(body["pos"], dtype=float))
            for jnt in body["joints"]:
                angle = joints[jnt["qpos_idx"]]
                anchor = xpos + quat_rot(xquat, np.asarray(jnt["pos"], dtype=float))
                xquat = quat_mul(xquat, axis_angle(jnt["axis"], angle))
                xpos = anchor - quat_rot(xquat, np.asarray(jnt["pos"], dtype=float))
        out.append((xpos, xquat))
    return out
