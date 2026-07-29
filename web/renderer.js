// Skeleton rendering: forward kinematics port of sim/fk.py (verified against MuJoCo
// to 1e-16 by sim/verify_fk.py) + three.js meshes built from skeleton.json geoms.
// World is z-up, matching MuJoCo.
import * as THREE from 'three';

// ---- quaternion math on [w,x,y,z] arrays — exact mirror of sim/fk.py ----
function quatMul(a, b) {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

function quatRot(q, v) {
  const [w, x, y, z] = q;
  const u = [x, y, z];
  const c1 = cross(u, v), t = [c1[0] + w * v[0], c1[1] + w * v[1], c1[2] + w * v[2]];
  const c2 = cross(u, t);
  return [v[0] + 2 * c2[0], v[1] + 2 * c2[1], v[2] + 2 * c2[2]];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function axisAngle(axis, angle) {
  const n = Math.hypot(axis[0], axis[1], axis[2]);
  const h = angle / 2, s = Math.sin(h) / n;
  return [Math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s];
}

// FK over skeleton.json bodies; returns [{pos:[3], quat:[wxyz]}] in body order.
export function forward(skel, rootPos, rootQuat, joints) {
  const out = [];
  for (const body of skel.bodies) {
    let xpos, xquat;
    if (body.free) {
      xpos = rootPos.slice();
      xquat = rootQuat.slice();
    } else {
      const p = out[body.parent];
      xquat = quatMul(p.quat, body.quat);
      const off = quatRot(p.quat, body.pos);
      xpos = [p.pos[0] + off[0], p.pos[1] + off[1], p.pos[2] + off[2]];
      for (const jnt of body.joints) {
        const angle = joints[jnt.qpos_idx];
        const a0 = quatRot(xquat, jnt.pos);
        const anchor = [xpos[0] + a0[0], xpos[1] + a0[1], xpos[2] + a0[2]];
        xquat = quatMul(xquat, axisAngle(jnt.axis, angle));
        const a1 = quatRot(xquat, jnt.pos);
        xpos = [anchor[0] - a1[0], anchor[1] - a1[1], anchor[2] - a1[2]];
      }
    }
    out.push({ pos: xpos, quat: xquat });
  }
  return out;
}

export async function loadSkeleton() {
  const res = await fetch('skeleton.json');
  return res.json();
}

const BODY_MAT = () => new THREE.MeshStandardMaterial({ color: 0xd9a066, roughness: 0.75 });

function geomMesh(g) {
  let geometry;
  if (g.type === 'capsule') {
    geometry = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 4, 12);
    geometry.rotateX(Math.PI / 2);           // three capsule axis is Y; MuJoCo's is Z
  } else if (g.type === 'sphere') {
    geometry = new THREE.SphereGeometry(g.size[0], 16, 12);
  } else if (g.type === 'box') {
    geometry = new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]);
  } else if (g.type === 'cylinder') {
    geometry = new THREE.CylinderGeometry(g.size[0], g.size[0], 2 * g.size[1], 12);
    geometry.rotateX(Math.PI / 2);
  } else {
    geometry = new THREE.SphereGeometry(g.size[0] || 0.05, 8, 6);
  }
  const mesh = new THREE.Mesh(geometry, BODY_MAT());
  mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
  mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  return mesh;
}

// A posable humanoid inside a scene. setPose() drives it from a trajectory frame.
export class Skeleton {
  constructor(scene, skel) {
    this.skel = skel;
    this.bodyGroups = skel.bodies.map((body) => {
      const grp = new THREE.Group();
      grp.matrixAutoUpdate = false;
      for (const g of body.geoms) grp.add(geomMesh(g));
      scene.add(grp);
      return grp;
    });
    this._m = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._p = new THREE.Vector3();
  }

  setPose(rootPos, rootQuat, joints) {
    const frames = forward(this.skel, rootPos, rootQuat, joints);
    frames.forEach((f, i) => {
      this._q.set(f.quat[1], f.quat[2], f.quat[3], f.quat[0]);
      this._p.set(f.pos[0], f.pos[1], f.pos[2]);
      this.bodyGroups[i].matrix.compose(this._p, this._q, ONE);
      this.bodyGroups[i].matrixWorldNeedsUpdate = true;
    });
  }
}
const ONE = new THREE.Vector3(1, 1, 1);

export function makeScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14171c);
  const hemi = new THREE.HemisphereLight(0xffffff, 0x334, 1.1);
  hemi.position.set(0, 0, 1);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 1.4);
  dir.position.set(2, -3, 5);
  scene.add(dir);
  const grid = new THREE.GridHelper(40, 80, 0x2c333d, 0x222831);
  grid.rotation.x = Math.PI / 2;           // z-up
  scene.add(grid);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshStandardMaterial({ color: 0x181c22, roughness: 1 })
  );
  floor.position.z = -0.01;
  scene.add(floor);
  return scene;
}

// One playing clip inside its own scene, with a camera that tracks the root.
// Camera logic is identical for every instance — fairness requirement (spec §6).
export class ClipView {
  constructor(skel) {
    this.scene = makeScene();
    this.skeleton = new Skeleton(this.scene, skel);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    this.camera.up.set(0, 0, 1);
    this.trajectory = null;
    this.t0 = 0;
    this._camTarget = new THREE.Vector3(0, 0, 0.8);
  }

  setClip(trajectory) {
    this.trajectory = trajectory;
    this.t0 = performance.now();
    const r = trajectory.root_pos[0];
    this._camTarget.set(r[0], r[1], 0.8);
    this._place(0, true);
  }

  _place(frame, snap = false) {
    const tr = this.trajectory;
    this.skeleton.setPose(tr.root_pos[frame], tr.root_quat[frame], tr.joints[frame]);
    const r = tr.root_pos[frame];
    const target = this._camTarget;
    const k = snap ? 1.0 : 0.08;           // smooth follow
    target.x += (r[0] - target.x) * k;
    target.y += (r[1] - target.y) * k;
    this.camera.position.set(target.x + 2.6, target.y - 3.2, 2.0);
    this.camera.lookAt(target.x, target.y, 0.8);
  }

  tick(now) {
    if (!this.trajectory) return;
    const tr = this.trajectory;
    const frame = Math.floor(((now - this.t0) / 1000) * tr.fps) % tr.n_frames;   // loop
    this._place(frame);
  }
}

// One canvas, N viewports (spec §6: not N contexts).
export class MultiView {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.views = [];                        // {view: ClipView, el: Element}
  }

  render(now) {
    const canvas = this.renderer.domElement;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w * this.renderer.getPixelRatio() ||
        canvas.height !== h * this.renderer.getPixelRatio()) {
      this.renderer.setSize(w, h, false);
    }
    this.renderer.setScissorTest(true);
    const canvasRect = canvas.getBoundingClientRect();
    for (const { view, el } of this.views) {
      const r = el.getBoundingClientRect();
      const left = r.left - canvasRect.left, top = r.top - canvasRect.top;
      const bottom = canvasRect.height - (top + r.height);
      view.tick(now);
      view.camera.aspect = r.width / r.height;
      view.camera.updateProjectionMatrix();
      this.renderer.setViewport(left, bottom, r.width, r.height);
      this.renderer.setScissor(left, bottom, r.width, r.height);
      this.renderer.render(view.scene, view.camera);
    }
    this.renderer.setScissorTest(false);
  }
}
