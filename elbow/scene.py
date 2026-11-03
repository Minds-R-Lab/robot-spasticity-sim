"""A picture of the study: the MyoSuite elbow of world.py with a one-joint robot around it.

Only for figures and videos.  The robot here is drawn, not simulated: the session in device2.py drives the elbow,
and this scene replays the elbow angle and the muscle activity it logged.
"""
from __future__ import annotations

import numpy as np

from .world import MYO_XML, Q0_MYO

ELBOW = np.array([0.0061, -0.2904, -0.0123])      # the elbow joint, in the upper-arm frame
STEEL, MOTOR, CUFF, PAD = [.62, .66, .70, 1], [.16, .19, .23, 1], [.20, .45, .62, .55], [.12, .12, .14, 1]
REST, ACTIVE = np.array([.86, .66, .62, 1.0]), np.array([.90, .06, .06, 1.0])


def build(width=960, height=720):
    """The render model: the elbow, the robot, a background and lights.  Returns (model, data)."""
    import mujoco
    sp = mujoco.MjSpec.from_file(MYO_XML)
    sp.visual.global_.offwidth = width
    sp.visual.global_.offheight = height
    sp.visual.quality.shadowsize = 4096
    hum = sp.body("r_humerus")
    arm = sp.body("r_ulna_radius_hand")
    z = 0.075                                          # the robot sits beside the arm, on the elbow axis

    def geom(body, kind, size, pos, rgba, euler=(0, 0, 0)):
        g = body.add_geom()
        g.type = kind
        g.size = list(size) + [0] * (3 - len(size))
        g.pos = list(pos)
        g.rgba = rgba
        g.alt.euler = list(euler)
        g.contype = g.conaffinity = 0
        g.group = 1
        g.mass = 1e-6

    cyl, box, sph, cap = (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX,
                          mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE)
    e = ELBOW
    # upper arm: a cuff, a rail to the motor, the motor and its encoder
    geom(hum, cyl, (0.046, 0.022), (0.0, -0.14, 0.0), CUFF, (1.5708, 0, 0))
    geom(hum, box, (0.012, 0.075, 0.006), (0.0, -0.20, z - 0.01), STEEL)
    geom(hum, box, (0.012, 0.008, 0.035), (0.0, -0.14, 0.025), STEEL)
    geom(hum, cyl, (0.045, 0.025), (e[0], e[1], e[2] + z + 0.012), MOTOR)
    geom(hum, cyl, (0.030, 0.012), (e[0], e[1], e[2] + z + 0.048), STEEL)
    geom(hum, cyl, (0.010, 0.010), (e[0], e[1], e[2] + z + 0.066), [.9, .55, .1, 1])
    # forearm: the lever the motor turns, and its cuff
    geom(arm, box, (0.014, 0.085, 0.006), (0.0, -0.085, z - 0.025), STEEL)
    geom(arm, box, (0.012, 0.008, 0.045), (0.0, -0.165, 0.025), STEEL)
    geom(arm, cyl, (0.038, 0.022), (0.008, -0.165, 0.035), CUFF, (1.5708, 0, 0))
    # an EMG electrode on the biceps
    geom(hum, box, (0.004, 0.014, 0.012), (0.026, -0.075, 0.0), [.97, .97, .97, 1])
    geom(hum, sph, (0.005,), (0.031, -0.075, 0.0), [.85, .7, .1, 1])
    # a light background, and light
    w = sp.worldbody
    tx = sp.add_texture()
    tx.name, tx.type, tx.builtin = "sky", mujoco.mjtTexture.mjTEXTURE_SKYBOX, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    tx.rgb1, tx.rgb2, tx.width, tx.height = [1.0, 1.0, 1.0], [0.86, 0.89, 0.93], 512, 3072
    for pos, dif in (((0.6, -0.6, 2.4), 0.55), ((-0.8, 0.4, 2.2), 0.35)):
        li = w.add_light()
        li.pos = list(pos)
        li.dir = list(-np.array(pos) + np.array([0, 0.3, 1.3]))
        li.diffuse = [dif] * 3
        li.castshadow = True
    m = sp.compile()
    m.vis.headlight.ambient = [0.35] * 3
    m.vis.headlight.diffuse = [0.45] * 3
    m.tendon_width[:] = 0.006
    return m, mujoco.MjData(m)


class Painter:
    """Poses the render model at an elbow angle q (the study's convention) and colours muscles by activation."""

    def __init__(self, width=960, height=720, azimuth=345.0, elevation=-8.0, distance=0.95):
        import mujoco
        self.mj = mujoco
        self.m, self.d = build(width, height)
        self.r = mujoco.Renderer(self.m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, distance
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1
        self.opt.geomgroup[2] = 0                         # hide the wrapping surfaces
        self.opt.sitegroup[:] = 0
        self.act_tendon = {}
        for a in range(self.m.nu):
            if self.m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON:
                self.act_tendon[a] = int(self.m.actuator_trnid[a, 0])
        self.cam.lookat[:] = self.centre()

    def centre(self):
        """The middle of the arm over a stretch: shoulder, elbow, and the wrist flexed and extended."""
        mj, m, d = self.mj, self.m, self.d
        pts = []
        for q in (0.05, 0.85):
            d.qpos[0] = Q0_MYO - q
            mj.mj_forward(m, d)
            pts.append(d.site_xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "wrist")].copy())
        pts += [d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b)].copy() for b in ("r_humerus", "r_ulna_radius_hand")]
        return np.mean(pts, axis=0)

    def frame(self, q, act=None):
        self.d.qpos[0] = Q0_MYO - q
        if act is not None:
            for a, t in self.act_tendon.items():
                x = float(np.clip(act[a], 0, 1)) ** 0.45
                self.m.tendon_rgba[t] = (1 - x) * REST + x * ACTIVE
        self.mj.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, self.cam, self.opt)
        return self.r.render()
