#!/usr/bin/env python3
"""wander.py - walk around on its own, turning away from obstacles, using the
LiDAR and the depth camera together. Ends in damping mode however it ends.

Needs, in other terminals:
  start_fastlio.sh                       (LiDAR + pose)
  realsense_depth_tcp_publisher.py       (on the Jetson)
  depth_bridge_receiver.py               (mini PC, RMW cyclone)

  python3 ~/wander.py --check            # sensors only, the robot does NOT move
  python3 ~/wander.py --check --stand    # stands up (does NOT walk), sensors at walking height
  python3 ~/wander.py                    # wander for up to 3 minutes
  python3 ~/wander.py --minutes 5
  python3 ~/wander.py --no-camera        # LiDAR only

Built from the previous owner's obstacle_avoidance.py:
  * the same idea: one "nearest obstacle ahead" + "which side" from each sensor,
    merged with HIS rule (average when they agree, trust the camera when it sees
    something very close, else the LiDAR), then FORWARD / SLOW / TURNING
  * his depth-camera method (rows 80-300, 0.5-5 m, 10th percentile), widened to
    8 strips; only strips whose obstacle is inside the robot's path count as
    "ahead" (the robot is 0.47 m wide, path = +-0.33 m), and the side is where
    that obstacle sits
  * after turning, wait for several clear readings before walking again
  * same lie-down order at the end: 3 -> 1 -> 0
Changed for safety:
  * LiDAR points are straightened with the LiDAR's own IMU (gravity) before
    his height band is applied: the LiDAR is tilted ~12 deg forward, so without
    this the floor ~2 m ahead counted as an obstacle (closer while bobbing)
  * no averaging of camera and LiDAR: the LiDAR decides, the camera only stops
    the robot for something in the path closer than 0.8 m that the LiDAR
    does not see (low things below his band)
  * no sensor data = STOP (his code treated it as "clear"): a short gap makes
    it stand still and wait up to 2 s, longer than that ends the run
  * speeds 0.15 m/s and 0.15 rad/s (his script used 0.5 m/s, over the limit)
  * turning is measured by the LiDAR heading; no blind reversing; if it has
    turned a full circle and is still blocked, it stops ("boxed in")
  * stops if it is told to walk but does not move for 10 s (stuck)
  * ignores only the robot's own outline (behind the LiDAR, 0.64 m wide, plus
    the first 0.6 m straight ahead) instead of everything within 0.6 m, so
    walls and furniture right beside the robot are seen
  * ignores its own network cable: a 0.35 m strip right behind the body is
    ignored (where the cable drops off), and something only counts in the
    memory once it stays in the same place in the room while the robot walks
    20 cm (furniture does, the cable moves with the robot)
  * remembers LiDAR obstacles for 15 s on the floor (using the Fast-LIO pose),
    so something that ends up inside the sensors' close blind zone (for
    example after turning in a tight spot) is still avoided
  * the body is 86 x 47 cm, so turning in place sweeps a circle of about
    0.54 m around its centre: it veers gently away from things that close on
    one side, will not turn if something is inside that circle, and then backs
    if one side is tight and the other is open it side-steps (linear.y) into
    the open side until it has room; otherwise it backs up 0.30 m over the
    ground it has just walked (known clear); after 3 tries it stops ("too
    tight to turn")
"""
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time

import numpy as np

ROBOT_HALF_WIDTH = 0.235          # manual: standing 86.3 x 47 cm
ROBOT_HALF_LENGTH = 0.43
PATH_HALF = ROBOT_HALF_WIDTH + 0.10
LIDAR_AHEAD = 0.30                # LiDAR is 0.30 m ahead of the body centre (repo TF)
# Room needed around the body centre to turn in place. The body's corners
# really sweep ~0.49 m; the default 0.25 m (set with --turn-room) means it
# practically always just turns, like the previous owner's code.
TURN_RADIUS = 0.25
# Self-mask: where the robot's own body can show up, in LiDAR coordinates.
# Body is behind the LiDAR (centre 0.30 m back, 0.43 m half-length) and 0.47 m
# wide; ahead of the LiDAR we keep ignoring the first --xmin metres as before.
SELF_HALF_WIDTH = 0.32
SELF_BACK = LIDAR_AHEAD + ROBOT_HALF_LENGTH + 0.05                        # 0.78 m
# Tail zone: the network cable drops off the back of the robot. Ignore a strip
# 0.35 m long directly behind the body (anything there would be touching it).
TAIL_LEN = 0.35
TAIL_HALF_WIDTH = SELF_HALF_WIDTH + 0.03

# camera (from the previous owner's obstacle_avoidance.py, widened to 8 strips)
CAM_X0, CAM_X1, CAM_Y0, CAM_Y1 = 20, 620, 80, 300
CAM_MIN, CAM_MAX = 0.5, 5.0
STRIPS = 8
FX = 337.0                        # approx. focal length (px), 87 deg over 640 px
CAM_VETO = 0.8                    # camera overrides the LiDAR only for things closer than this


# ---------------------------------------------------------------- LiDAR (walk_safe.py)
def fit_floor(p, rng, cell=0.25):
    """Floor plane z = a*x + b*y + c in LiDAR coordinates, or None."""
    if len(p) < 400:
        return None, f"only {len(p)} points"
    ix = np.floor(p[:, 0] / cell).astype(np.int64) + 200
    iy = np.floor(p[:, 1] / cell).astype(np.int64) + 200
    key = ix * 1000 + iy
    order = np.lexsort((p[:, 2], key))
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    sizes = np.diff(np.r_[starts, len(ks)])
    Q = p[order[starts[sizes >= 6] + 1]].astype(np.float64)   # 2nd-lowest per cell
    nq = len(Q)
    if nq < 15:
        return None, f"only {nq} ground cells"
    idx = rng.integers(0, nq, (400, 3))
    A, B, C = Q[idx[:, 0]], Q[idx[:, 1]], Q[idx[:, 2]]
    nr = np.cross(B - A, C - A)
    ln = np.linalg.norm(nr, axis=1)
    good = ln > 1e-6
    nr = nr / np.where(good, ln, 1.0)[:, None]
    nr[nr[:, 2] < 0] *= -1
    d = -np.einsum("ij,ij->i", nr, A)
    good &= nr[:, 2] > 0.82
    good &= (d > 0.05 * nr[:, 2]) & (d < 1.2 * nr[:, 2])
    if not good.any():
        return None, f"no level plane among {nq} ground cells"
    dist = np.abs(Q @ nr.T + d)
    inl = (dist < 0.04).sum(axis=0)
    inl[~good] = 0
    k = int(np.argmax(inl))
    if inl[k] < max(20, int(0.25 * nq)):
        return None, f"best plane fits only {int(inl[k])} of {nq} ground cells"
    m = dist[:, k] < 0.04
    for _ in range(3):
        X = np.c_[Q[m, 0], Q[m, 1], np.ones(int(m.sum()))]
        coef, *_ = np.linalg.lstsq(X, Q[m, 2], rcond=None)
        res = np.abs(Q[:, 2] - (Q[:, 0] * coef[0] + Q[:, 1] * coef[1] + coef[2]))
        m = res < 0.04
        if m.sum() < 12:
            return None, "plane fit collapsed"
    G = Q[m]
    below = int((Q[:, 2] - (Q[:, 0] * coef[0] + Q[:, 1] * coef[1] + coef[2]) < -0.08).sum())
    if below > max(5, 0.1 * nq):
        return None, f"{below} ground cells lie below the best plane (not the floor)"
    near = G[np.hypot(G[:, 0], G[:, 1]) < 2.0]
    if len(near) < 8 or np.ptp(np.hypot(G[:, 0], G[:, 1])) < 0.8:
        return None, "no floor seen within 2 m"
    a, b, c = (float(v) for v in coef)
    if 1.0 / math.sqrt(1 + a * a + b * b) < 0.82 or not (-1.2 < c < -0.05):
        return None, f"implausible floor (LiDAR {-c:.2f} m up)"
    return (a, b, c), f"{int(m.sum())} of {nq} ground cells"


def nth(v, k=4):
    """k-th nearest (ignores a few stray points), or None."""
    return float(np.sort(v)[min(k, len(v) - 1)]) if len(v) >= 5 else None


def lidar_analyze(pts, a, rng, plane, keep=None):
    """One LiDAR frame -> nearest obstacle in the path ahead, its side, and how
    open the left and right are. Heights are measured above the floor.
    keep: dict that receives the obstacle points (x, y) for the memory."""
    p = np.asarray(pts, dtype=np.float32)[:, :3]
    p = p[np.isfinite(p).all(axis=1) & np.any(p != 0, axis=1)]
    now = time.time()
    r = np.hypot(p[:, 0], p[:, 1])
    if not a.floor_mode:
        # The previous owner's method (obstacle_avoidance.py): a fixed height
        # band measured from the LiDAR itself, no floor finding. Things lower
        # than the band (boxes, shoes) are left to the camera, as in his code.
        h, lo, hi, info = p[:, 2], a.band_lo, a.band_hi, "fixed height band"
        plane["t"] = now
    else:
        lo, hi = a.zmin, a.zmax
        coef, info = fit_floor(p[(p[:, 0] > a.xmin) & (r < 3.5)], rng)
        if coef is not None:
            plane["coef"], plane["t"], plane["src"] = coef, now, "fit"
        elif plane["coef"] is None:
            # The floor is learned once and then kept: the LiDAR is bolted to
            # the robot and the house floor is flat. A frame showing too little
            # floor (facing furniture) just keeps the floor it already knows.
            return dict(l_ok=False, l_info=info, l_front=None, l_side=0,
                        l_open_l=None, l_open_r=None, l_fit_t=0.0)
        ca, cb, cc = plane["coef"]
        h = p[:, 2] - (ca * p[:, 0] + cb * p[:, 1] + cc)
    own = (np.abs(p[:, 1]) < SELF_HALF_WIDTH) & (p[:, 0] > -SELF_BACK) & (p[:, 0] < a.xmin)
    own |= ((np.abs(p[:, 1]) < TAIL_HALF_WIDTH) & (p[:, 0] <= -SELF_BACK)
            & (p[:, 0] > -SELF_BACK - TAIL_LEN))                    # cable behind
    obst = p[(h > lo) & (h < hi) & ~own & (r > 0.15)]
    if keep is not None:
        keep["obst"] = obst[np.hypot(obst[:, 0], obst[:, 1]) < 2.5, :2]
        if "view" in keep:
            keep["all"], keep["band"] = p[:, :2], obst[:, :2]
    out = dict(l_ok=True, l_info=info, l_front=None, l_side=0, l_fit_t=plane["t"])
    path = obst[(obst[:, 0] > 0) & (obst[:, 0] < 4.0) & (np.abs(obst[:, 1]) < PATH_HALF)]
    if keep is not None and "view" in keep:
        keep["path"] = path[:, :2]
    out["l_front"] = nth(path[:, 0])
    if out["l_front"] is not None:
        close = path[path[:, 0] < out["l_front"] + 0.3]
        my = float(np.mean(close[:, 1]))
        out["l_side"] = 1 if my > 0.1 else -1 if my < -0.1 else 0   # +1 = on the LEFT
    # how open each side is: nearest obstacle 20-110 deg to each side, within 2.5 m
    ang = np.degrees(np.arctan2(obst[:, 1], obst[:, 0]))
    ro = np.hypot(obst[:, 0], obst[:, 1])
    near = ro < 2.5
    out["l_open_l"] = nth(ro[near & (ang > 20) & (ang < 110)])
    out["l_open_r"] = nth(ro[near & (ang < -20) & (ang > -110)])
    return out


class ObstacleMemory:
    """Short-term memory of LiDAR obstacles on the floor (10 cm squares, world
    coordinates from Fast-LIO), kept for 15 s. Both sensors are blind up close
    (LiDAR < 0.6 m, camera < 0.5 m), so something seen earlier that is now very
    close, for example after turning in a tight spot, is still known.

    Things attached to the robot (its network cable) move WITH the robot, so
    they never stay in one place in the room. A square only becomes a remembered
    obstacle once it has been seen continuously while the robot WALKED 20 cm
    (turning in place does not count: a cable curving around the robot looks
    the same while it spins). Furniture passes that within about a second and a
    half of walking; the cable does not. Squares seen while the robot has not moved yet since
    the start count at once (so a wall beside the start is known), but are
    forgotten after 3 s unless they are confirmed that way too."""

    CONFIRM_MOVE = 0.20          # m walked while the square stays in view
    PENDING_KEEP = 0.4           # s: an unconfirmed square must be seen (almost) every frame
    START_KEEP = 3.0             # s: squares from the start position

    def __init__(self, cell=0.10, keep_s=15.0, min_pts=3):
        self.cell, self.keep_s, self.min_pts = cell, keep_s, min_pts
        self.half_diag = cell * math.sqrt(0.5)          # cell centre to its corner
        self.cells = {}          # confirmed: (i, j) -> [time seen, "view" | "start"]
        self.pending = {}        # unconfirmed: (i, j) -> [time seen, x, y, yaw when first seen]
        self.start_pose = None

    def _moved(self, x0, y0, yaw0, x, y, yaw):
        return math.hypot(x - x0, y - y0) >= self.CONFIRM_MOVE

    def add(self, pts_xy, x, y, yaw, now):
        if self.start_pose is None:
            self.start_pose = (x, y, yaw)
        at_start = not self._moved(*self.start_pose, x, y, yaw)
        if len(pts_xy):
            c, s = math.cos(yaw), math.sin(yaw)
            wx = x + c * pts_xy[:, 0] - s * pts_xy[:, 1]
            wy = y + s * pts_xy[:, 0] + c * pts_xy[:, 1]
            k = np.c_[np.floor(wx / self.cell), np.floor(wy / self.cell)].astype(np.int64)
            keys, counts = np.unique(k, axis=0, return_counts=True)
            for (i, j), n in zip(keys, counts):
                if n < self.min_pts:
                    continue
                key = (int(i), int(j))
                if key in self.cells:
                    e = self.cells[key]
                    e[0] = now
                    if e[1] == "start" and not at_start:
                        e[1] = "view"            # still there after moving: real
                elif key in self.pending:
                    pe = self.pending[key]
                    if self._moved(pe[1], pe[2], pe[3], x, y, yaw):
                        self.cells[key] = [now, "view"]
                        del self.pending[key]
                    else:
                        pe[0] = now
                elif at_start:
                    self.cells[key] = [now, "start"]
                else:
                    self.pending[key] = [now, x, y, yaw]
        for kk in [kk for kk, e in self.cells.items()
                   if now - e[0] > (self.keep_s if e[1] == "view" else self.START_KEEP)]:
            del self.cells[kk]
        for kk in [kk for kk, e in self.pending.items() if now - e[0] > self.PENDING_KEEP]:
            del self.pending[kk]

    def front(self, x, y, yaw):
        """Nearest remembered obstacle in the path ahead (0-1.2 m) and its side."""
        if not self.cells:
            return None, 0
        k = np.array(list(self.cells.keys()), dtype=np.float64)
        wx, wy = (k[:, 0] + 0.5) * self.cell - x, (k[:, 1] + 0.5) * self.cell - y
        c, s = math.cos(yaw), math.sin(yaw)
        rx, ry = c * wx + s * wy, -s * wx + c * wy
        inp = (rx > 0) & (rx < 1.2) & (np.abs(ry) < PATH_HALF)
        if not inp.any():
            return None, 0
        d = float(rx[inp].min())
        my = float(ry[inp][rx[inp] < d + 0.3].mean())
        return d, (1 if my > 0.1 else -1 if my < -0.1 else 0)

    def turn_blocked(self, x, y, yaw):
        """Remembered obstacles inside the circle the body sweeps when turning in
        place (x, y = LiDAR position; the body centre is LIDAR_AHEAD behind it)."""
        if not self.cells:
            return 0
        cx, cy = x - LIDAR_AHEAD * math.cos(yaw), y - LIDAR_AHEAD * math.sin(yaw)
        k = np.array(list(self.cells.keys()), dtype=np.float64)
        d = np.hypot((k[:, 0] + 0.5) * self.cell - cx, (k[:, 1] + 0.5) * self.cell - cy)
        return int((d - self.half_diag < TURN_RADIUS).sum())

    def turn_blocked_sides(self, x, y, yaw, extra=0.0):
        """Like turn_blocked, split into (left, right) of the body; extra widens the circle."""
        if not self.cells:
            return 0, 0
        cx, cy = x - LIDAR_AHEAD * math.cos(yaw), y - LIDAR_AHEAD * math.sin(yaw)
        k = np.array(list(self.cells.keys()), dtype=np.float64)
        wx, wy = (k[:, 0] + 0.5) * self.cell - cx, (k[:, 1] + 0.5) * self.cell - cy
        c, s = math.cos(yaw), math.sin(yaw)
        rx, ry = c * wx + s * wy, -s * wx + c * wy
        inside = np.hypot(rx, ry) - self.half_diag < TURN_RADIUS + extra
        return int((inside & (ry > 0)).sum()), int((inside & (ry <= 0)).sum())

    def side_near(self, x, y, yaw, margin=0.0):
        """Remembered obstacles just outside turning room, left and right of the body."""
        if not self.cells:
            return 0, 0
        cx, cy = x - LIDAR_AHEAD * math.cos(yaw), y - LIDAR_AHEAD * math.sin(yaw)
        k = np.array(list(self.cells.keys()), dtype=np.float64)
        wx, wy = (k[:, 0] + 0.5) * self.cell - cx, (k[:, 1] + 0.5) * self.cell - cy
        c, s = math.cos(yaw), math.sin(yaw)
        rx, ry = c * wx + s * wy, -s * wx + c * wy
        near = (np.hypot(rx, ry) - self.half_diag < TURN_RADIUS + margin) & (rx > -ROBOT_HALF_LENGTH)
        return int((near & (ry > 0)).sum()), int((near & (ry < 0)).sum())


# ---------------------------------------------------------------- camera (cam_check v2)
def cam_analyze(depth_mm):
    h, w = depth_mm.shape
    roi = depth_mm[CAM_Y0:CAM_Y1, CAM_X0:CAM_X1].astype(np.float32) / 1000.0
    ok = (roi > CAM_MIN) & (roi < CAM_MAX)
    edges = np.linspace(0, roi.shape[1], STRIPS + 1).astype(int)
    strips, angles = [], []
    for k in range(STRIPS):
        s0, s1 = edges[k], edges[k + 1]
        v = roi[:, s0:s1][ok[:, s0:s1]]
        strips.append(float(np.percentile(v, 10)) if len(v) > 100 else None)
        u = CAM_X0 + (s0 + s1) / 2.0
        angles.append(math.atan((w / 2.0 - u) / FX))            # + = left
    # in the path: sideways offset (depth * tan(angle)) within the robot's path
    inpath = [v for v, an in zip(strips, angles)
              if v is not None and abs(v * math.tan(an)) < PATH_HALF]
    # side of the nearest in-path obstacle: its sideways offset (+ = LEFT)
    side, best = 0, None
    for v, an in zip(strips, angles):
        if v is not None and abs(v * math.tan(an)) < PATH_HALF and (best is None or v < best):
            best = v
            off = v * math.tan(an)
            side = 1 if off > 0.08 else -1 if off < -0.08 else 0
    if len(inpath) > 1 and max(inpath) - min(inpath) < 0.1:
        side = 0                     # flat across the path (a wall): no side
    return dict(c_front=best, c_side=side, c_strips=strips)


# ---------------------------------------------------------------- merge (his rule)
def merge(m, use_cam, cam_max_age=3.0):
    """Returns (distance or None, side, text). None = cannot see = STOP."""
    now = time.time()
    lidar_ok = m["l_ok"] and now - m["lidar_t"] < 1.0
    cam_ok = now - m["cam_t"] < cam_max_age
    if not lidar_ok:
        if now - m["lidar_t"] >= 1.0:
            return None, 0, "LiDAR data stopped arriving"
        return None, 0, f"floor never found yet ({m.get('l_info', '')})"
    if use_cam and not cam_ok:
        return None, 0, "camera lost"
    ld = m["l_front"] if m["l_front"] is not None else 9.0
    d, side, src = ld, m["l_side"], "lidar"
    # The camera sends ~2 pictures a second and a picture can be up to 2 s old.
    # A picture taken before the robot turned shows the OLD direction: ignore
    # it if the heading has changed more than 10 degrees since it was taken.
    cam_yaw = m.get("cam_yaw")
    fresh_dir = (cam_yaw is not None and m.get("yaw") is not None
                 and abs(wrap(m["yaw"] - cam_yaw)) < math.radians(10))
    # The LiDAR decides. The camera only steps in when it sees something in the
    # path closer than 0.8 m that the LiDAR does not (low things the LiDAR's
    # band ignores). No averaging: with nothing in view the camera reports the
    # floor at ~1.5 m (its limit), which is not a real distance to average.
    if use_cam and fresh_dir and m["c_front"] is not None:
        cd = m["c_front"]
        if cd < CAM_VETO and cd < ld:
            d, side, src = cd, m["c_side"], "camera"
    return d, side, src


# ---------------------------------------------------------------- sensor helper
FLOOR_FILE = os.path.expanduser("~/.wander_floor.json")   # floor as seen while standing
STATE_FILE = os.path.expanduser("~/.wander_state")        # "standing <time>" or "lying"


def read_state():
    try:
        with open(STATE_FILE) as fh:
            w = fh.read().split()
        return float(w[1]) if w and w[0] == "standing" else None
    except Exception:  # noqa
        return None


def write_state(text):
    try:
        with open(STATE_FILE, "w") as fh:
            fh.write(text)
    except Exception:  # noqa
        pass


def load_floor():
    try:
        with open(FLOOR_FILE) as fh:
            c = json.load(fh)["coef"]
        return tuple(float(v) for v in c)
    except Exception:  # noqa
        return None


def save_floor(coef):
    try:
        with open(FLOOR_FILE, "w") as fh:
            json.dump({"coef": list(coef), "saved": time.time()}, fh)
    except Exception:  # noqa
        pass


# ---------------------------------------------------------------- live view
# A small web page served by the sensor helper: open http://192.168.1.10:8080
# on the laptop. Top-down LiDAR map, camera picture with the 8 strips, what
# each sensor says, the merged decision, and a frozen snapshot of the last stop.

def png_bytes(img):
    """uint8 array (h, w) or (h, w, 3) -> PNG file bytes (no extra libraries)."""
    import struct
    import zlib
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    ctype = 2 if img.ndim == 3 else 0
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def strip_angles(width=640):
    edges = np.linspace(0, CAM_X1 - CAM_X0, STRIPS + 1)
    return [math.atan((width / 2.0 - (CAM_X0 + (edges[k] + edges[k + 1]) / 2.0)) / FX)
            for k in range(STRIPS)]


def view_info(st, a, use_cam):
    """What each sensor says and what the robot would do, for the page."""
    now = time.time()
    d, side, src = merge(st, use_cam)
    if d is None:
        act = "STOP (" + src + ")"
    else:
        act = ("STOP + TURN" if d < a.stop else "SLOW" if d < a.slow else "GO")
    cam_yaw, yaw = st.get("cam_yaw"), st.get("yaw")
    fresh_dir = (cam_yaw is not None and yaw is not None
                 and abs(wrap(yaw - cam_yaw)) < math.radians(10))
    inpath = [v is not None and abs(v * math.tan(an)) < PATH_HALF
              for v, an in zip(st["c_strips"], strip_angles())]
    return dict(t=now, action=act, merged=d, merged_src=src, merged_side=side_word(side),
                l_front=st["l_front"], l_side=side_word(st["l_side"]), l_mem=st["l_mem"],
                l_info=st["l_info"], c_front=st["c_front"], c_side=side_word(st["c_side"]),
                cam_age=now - st["cam_t"] if st["cam_t"] else None, cam_dir_ok=fresh_dir,
                c_strips=st["c_strips"], c_inpath=inpath,
                tb=st["tb"], near_l=st["near_l"], near_r=st["near_r"], tilt=st.get("tilt"),
                stop=a.stop, slow=a.slow)


def view_svg(snap, a):
    """Top-down picture around the robot (LiDAR frame): forward is up, left is left."""
    W = H = 500
    k = 100.0                                    # pixels per metre

    def X(y):
        return (2.5 - y) * k

    def Y(x):
        return (3.5 - x) * k

    def dots(xy, size):
        if xy is None or len(xy) == 0:
            return ""
        xy = xy[(xy[:, 0] > -1.5) & (xy[:, 0] < 3.5) & (np.abs(xy[:, 1]) < 2.5)]
        return "".join(f"M{X(py):.0f} {Y(px):.0f}h{size}v{size}h-{size}z" for px, py in xy)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="#fafafa"/>']
    for m in range(-1, 4):
        out.append(f'<line x1="0" y1="{Y(m):.0f}" x2="{W}" y2="{Y(m):.0f}" stroke="#e4e4e4"/>'
                   f'<text x="3" y="{Y(m) - 3:.0f}" fill="#999">{m} m</text>')
    allp = snap.get("all")
    if allp is not None and len(allp) > 3000:
        allp = allp[:: max(1, len(allp) // 3000)]
    out.append(f'<path d="{dots(allp, 2)}" fill="#c8c8c8"/>')
    out.append(f'<path d="{dots(snap.get("band"), 3)}" fill="#f0a030"/>')
    for (cx, cy) in snap.get("mem", []):
        out.append(f'<rect x="{X(cy + 0.05):.0f}" y="{Y(cx + 0.05):.0f}" width="10" height="10" '
                   f'fill="none" stroke="#3060d0" stroke-width="1.5"/>')
    out.append(f'<path d="{dots(snap.get("path"), 4)}" fill="#e02020"/>')
    # robot body, turn circle, path, stop/slow lines
    bx0, bx1 = -LIDAR_AHEAD - ROBOT_HALF_LENGTH, -LIDAR_AHEAD + ROBOT_HALF_LENGTH
    out.append(f'<rect x="{X(ROBOT_HALF_WIDTH):.0f}" y="{Y(bx1):.0f}" width="{2 * ROBOT_HALF_WIDTH * k:.0f}" '
               f'height="{(bx1 - bx0) * k:.0f}" fill="#4a7bd8" fill-opacity="0.35" stroke="#2a4b98"/>')
    out.append(f'<circle cx="{X(0):.0f}" cy="{Y(-LIDAR_AHEAD):.0f}" r="{TURN_RADIUS * k:.0f}" '
               f'fill="none" stroke="#2a4b98" stroke-dasharray="3 3"/>')
    out.append(f'<circle cx="{X(0):.0f}" cy="{Y(0):.0f}" r="4" fill="#000"/>')
    for yy in (PATH_HALF, -PATH_HALF):
        out.append(f'<line x1="{X(yy):.0f}" y1="{Y(0):.0f}" x2="{X(yy):.0f}" y2="{Y(3.5):.0f}" '
                   f'stroke="#888" stroke-dasharray="6 4"/>')
    for dist, col, lab in ((a.stop, "#e02020", "stop"), (a.slow, "#f0a030", "slow")):
        out.append(f'<line x1="{X(PATH_HALF):.0f}" y1="{Y(dist):.0f}" x2="{X(-PATH_HALF):.0f}" '
                   f'y2="{Y(dist):.0f}" stroke="{col}" stroke-width="2"/>'
                   f'<text x="{X(-PATH_HALF) + 4:.0f}" y="{Y(dist) + 4:.0f}" fill="{col}">{lab} {dist:.1f} m</text>')
    # camera strip readings (green; filled = inside the robot's path)
    info = snap.get("info", {})
    for v, an, inp in zip(info.get("c_strips", []), strip_angles(), info.get("c_inpath", [])):
        if v is not None:
            out.append(f'<circle cx="{X(v * math.tan(an)):.0f}" cy="{Y(v):.0f}" r="6" '
                       f'fill="{"#20a040" if inp else "none"}" stroke="#20a040" stroke-width="2"/>')
    out.append(f'<text x="8" y="{H - 10}" font-size="18" font-weight="bold" fill="#222">'
               f'{info.get("action", "")}</text>')
    out.append('</svg>')
    return "".join(out)


VIEW_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Hexplorer live view</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:sans-serif;margin:12px;background:#fff;color:#222}
.row{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
.cam{position:relative;width:320px;height:240px;background:#333}
.cam img{width:320px;height:240px;display:block}
.strip{position:absolute;top:40px;height:110px;border:1px solid rgba(255,255,255,.5);
 color:#fff;font-size:11px;text-align:center;box-sizing:border-box;text-shadow:0 0 3px #000}
.strip.inpath{border:2px solid #20d060}.strip.close{background:rgba(255,0,0,.35)}
table{border-collapse:collapse;font-size:14px}td{padding:2px 10px 2px 0}
h2{font-size:16px;margin:18px 0 6px}.big{font-size:22px;font-weight:bold}
.key span{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px}</style></head><body>
<div class="big" id="act">connecting...</div>
<label><input type="checkbox" id="pause"> pause</label>
<div class="row">
 <div><img id="scene" width="500" height="500">
  <div class="key"><span style="background:#c8c8c8"></span>all LiDAR points
  <span style="background:#f0a030"></span>counted as obstacle (LiDAR)
  <span style="background:#e02020"></span>obstacle in the path<br>
  <span style="border:2px solid #3060d0"></span>remembered
  <span style="background:#20a040;border-radius:5px"></span>camera strip (filled = in path)
  <span style="background:#4a7bd8"></span>robot, dashed circle = turn room</div></div>
 <div><div class="cam" id="cam"><img id="camimg"></div>
  <table id="info"></table></div>
</div>
<h2 id="stoph">Last stop: none yet</h2>
<div class="row"><img id="sscene" width="500" height="500" style="display:none">
 <div><div class="cam" id="scam" style="display:none"><img id="scamimg"></div><table id="sinfo"></table></div></div>
<script>
function f(v){return v===null||v===undefined?"--":(typeof v==="number"?v.toFixed(2):v)}
function table(el,d){el.innerHTML=
 "<tr><td><b>LiDAR</b></td><td>"+f(d.l_front)+" m "+d.l_side+(d.l_mem?" (from memory)":"")+"</td></tr>"+
 "<tr><td><b>camera</b></td><td>"+f(d.c_front)+" m "+d.c_side+", picture "+f(d.cam_age)+" s old"+
   (d.cam_dir_ok?"":" (IGNORED: taken before a turn)")+"</td></tr>"+
 "<tr><td><b>merged</b></td><td>"+f(d.merged)+" m via "+d.merged_src+", obstacle side "+d.merged_side+"</td></tr>"+
 "<tr><td><b>decision</b></td><td>"+d.action+"</td></tr>"+
 "<tr><td>turn room</td><td>"+(d.tb>=2?"TOO CLOSE TO TURN":"ok")+" (near L "+d.near_l+", R "+d.near_r+")</td></tr>"+
 "<tr><td>LiDAR mode</td><td>"+d.l_info+", LiDAR tilt "+(d.tilt===null||d.tilt===undefined?"UNKNOWN (no IMU data: not straightened)":f(d.tilt)+" deg, straightened")+"</td></tr>"}
function strips(el,d){el.querySelectorAll(".strip").forEach(e=>e.remove());
 for(let i=0;i<8;i++){const s=document.createElement("div");const v=d.c_strips[i];
  s.className="strip"+(d.c_inpath[i]?" inpath":"")+(d.c_inpath[i]&&v!==null&&v<d.stop?" close":"");
  s.style.left=(10+i*37.5)+"px";s.style.width="37.5px";s.textContent=f(v);el.appendChild(s)}}
let lastStop=0;
async function tick(){if(document.getElementById("pause").checked)return;const t=Date.now();
 try{const d=await (await fetch("info.json?"+t)).json();
  document.getElementById("act").textContent=d.action;table(document.getElementById("info"),d);
  strips(document.getElementById("cam"),d);
  document.getElementById("scene").src="scene.svg?"+t;document.getElementById("camimg").src="cam.png?"+t;
  const s=await (await fetch("stop.json?"+t)).json();
  if(s.t&&s.t!==lastStop){lastStop=s.t;
   document.getElementById("stoph").textContent="Last stop: "+new Date(s.t*1000).toLocaleTimeString()+" - "+s.action;
   table(document.getElementById("sinfo"),s);strips(document.getElementById("scam"),s);
   for(const id of ["sscene","scam"])document.getElementById(id).style.display="";
   document.getElementById("sscene").src="stop.svg?"+t;document.getElementById("scamimg").src="stop.png?"+t}
 }catch(e){document.getElementById("act").textContent="no connection to the robot"}}
setInterval(tick,500);tick();
</script></body></html>"""


class LiveView:
    """Holds the latest data (and a copy at the last stop) and serves the page."""

    def __init__(self, a):
        import threading
        self.a, self.lock = a, threading.Lock()
        self.live, self.stop_snap, self.was_stop = {}, None, False
        self.cam_img = None
        self.ok = False
        try:
            self._serve(a.view_port)
            self.ok = True
        except OSError as e:
            print(f"[live view] could not open port {a.view_port}: {e}", flush=True)

    def update(self, snap):
        with self.lock:
            snap["cam"] = self.cam_img
            self.live = snap
            stop_now = snap["info"]["action"].startswith("STOP")
            if stop_now and not self.was_stop:
                self.stop_snap = dict(snap)            # freeze this moment
            self.was_stop = stop_now

    def set_cam(self, img):
        with self.lock:
            self.cam_img = img

    def _get(self, which):
        with self.lock:
            return self.live if which == "live" else self.stop_snap

    def _serve(self, port):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        view = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0].lstrip("/")
                try:
                    if path in ("", "index.html"):
                        return self._send(VIEW_HTML.encode(), "text/html; charset=utf-8")
                    which = "stop" if path.startswith("stop") else "live"
                    snap = view._get(which) or {}
                    if path.endswith(".json"):
                        return self._send(json.dumps(snap.get("info", {})).encode(), "application/json")
                    if path.endswith(".svg"):
                        return self._send(view_svg(snap, view.a).encode(), "image/svg+xml")
                    if path.endswith(".png"):
                        img = snap.get("cam")
                        if img is None:
                            img = np.full((240, 320), 60, np.uint8)
                        return self._send(png_bytes(img), "image/png")
                    self.send_error(404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()


def robot_frame_cells(mem, x, y, yaw):
    """Remembered squares, as centres in LiDAR coordinates (for the view)."""
    if not mem.cells or yaw is None:
        return []
    k = np.array(list(mem.cells.keys()), dtype=np.float64)
    wx, wy = (k[:, 0] + 0.5) * mem.cell - x, (k[:, 1] + 0.5) * mem.cell - y
    c, s = math.cos(yaw), math.sin(yaw)
    rx, ry = c * wx + s * wy, -s * wx + c * wy
    sel = (rx > -1.5) & (rx < 3.5) & (np.abs(ry) < 2.5)
    return list(zip(rx[sel].tolist(), ry[sel].tolist()))


def level_rotation(acc):
    """Rotation that turns LiDAR coordinates into LEVEL coordinates, from the
    IMU's accelerometer (at rest it points straight up). Only tilt is removed,
    not the heading. Returns a 3x3 matrix, or None."""
    up = np.asarray(acc, dtype=np.float64)
    n = np.linalg.norm(up)
    if n < 1e-6:
        return None
    up /= n
    ez = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, ez)
    c = float(up @ ez)
    if c < 0.5:                       # tilted more than 60 deg: not plausible
        return None
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def run_sensor(a):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import PointCloud2, Image
    from nav_msgs.msg import Odometry

    rclpy.init()
    node = Node("wander_sensor")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=2)
    rng = np.random.default_rng(1)
    plane = {"coef": None, "t": 0.0, "src": None}
    fstate = {"checked": 0.0, "since": None, "saved": load_floor(), "saved_t": 0.0}
    mem = ObstacleMemory()
    st = dict(x=None, y=None, yaw=None, pose_t=0.0, lidar_t=0.0, seq=0, l_mem=False, tb=0, l_fit_t=0.0,
              near_l=0, near_r=0, tb_l=0, tb_r=0, room_l=0, room_r=0,
              l_ok=False, l_info="", l_front=None, l_side=0, l_open_l=None,
              l_open_r=None, cam_t=0.0, cam_n=0, cam_yaw=None, c_front=None, c_side=0,
              c_strips=[None] * STRIPS, tilt=None)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    view = LiveView(a) if a.view_port else None
    if view is not None and not view.ok:
        view = None
    use_cam = not a.no_camera
    imu = {"acc": None, "t": 0.0}

    def on_imu(m):
        # Average the accelerometer over ~1 s: that is the direction of
        # gravity, i.e. how the LiDAR is tilted (bobbing included, slowly).
        v = np.array([m.linear_acceleration.x, m.linear_acceleration.y,
                      m.linear_acceleration.z], dtype=np.float64)
        imu["acc"] = v if imu["acc"] is None else 0.99 * imu["acc"] + 0.01 * v
        imu["t"] = time.time()

    def on_pose(m):
        q = m.pose.pose.orientation
        st["x"], st["y"] = m.pose.pose.position.x, m.pose.pose.position.y
        st["yaw"] = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        st["pose_t"] = time.time()

    def on_cloud(m):
        step = max(m.point_step // 4, 3)
        buf = np.frombuffer(bytes(m.data), dtype=np.float32)
        buf = buf[: (len(buf) // step) * step].reshape(-1, step)
        now = time.time()
        # Straighten the points with the IMU: the LiDAR is tilted ~12 deg
        # forward, so without this the floor ~2 m ahead "rises" into the
        # obstacle band (and closer when the robot bobs while walking).
        R = level_rotation(imu["acc"]) if (imu["acc"] is not None and now - imu["t"] < 0.5) else None
        if R is not None:
            buf = buf[:, :3] @ R.T.astype(np.float32)
            st["tilt"] = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        else:
            st["tilt"] = None
        # Floor while standing: the robot's own standing floor never changes,
        # so it is saved to a file the first time it is seen and reused when a
        # run starts somewhere with no floor in view (e.g. facing a door).
        if now - fstate["checked"] > 0.5:
            fstate["checked"], fstate["since"] = now, read_state()
        since = fstate["since"]
        if (a.floor_mode and since is not None and plane["t"] < since and now - since > 1.5
                and fstate["saved"] is not None):
            plane["coef"], plane["t"], plane["src"] = fstate["saved"], now, "saved"
        keep = {"view": True} if view is not None else {}
        t_before = plane["t"]
        res = lidar_analyze(buf, a, rng, plane, keep)
        if (a.floor_mode and since is not None and plane.get("src") == "fit" and plane["t"] != t_before
                and now - since > 1.0 and now - fstate["saved_t"] > 10.0):
            save_floor(plane["coef"])
            fstate["saved"], fstate["saved_t"] = plane["coef"], now
        if plane.get("src") == "saved":
            res["l_info"] = "using the saved standing floor"
        res["l_mem"] = False
        if res["l_ok"] and st["yaw"] is not None and now - st["pose_t"] < 0.6:
            mem.add(keep.get("obst", np.zeros((0, 2))), st["x"], st["y"], st["yaw"], now)
            md, ms = mem.front(st["x"], st["y"], st["yaw"])
            if md is not None and (res["l_front"] is None or md < res["l_front"] - 0.05):
                res["l_front"], res["l_side"], res["l_mem"] = md, ms, True
            res["tb"] = mem.turn_blocked(st["x"], st["y"], st["yaw"])
            res["near_l"], res["near_r"] = mem.side_near(st["x"], st["y"], st["yaw"])
            res["tb_l"], res["tb_r"] = mem.turn_blocked_sides(st["x"], st["y"], st["yaw"])
            res["room_l"], res["room_r"] = mem.turn_blocked_sides(st["x"], st["y"], st["yaw"], 0.25)
        st.update(res)
        st["lidar_t"] = now
        st["seq"] += 1
        if view is not None:
            try:
                view.update(dict(info=view_info(st, a, use_cam), all=keep.get("all"),
                                 band=keep.get("band"), path=keep.get("path"),
                                 mem=robot_frame_cells(mem, st["x"], st["y"], st["yaw"])))
            except Exception as e:  # noqa  (the view must never stop the robot)
                print(f"[live view] {e}", flush=True)

    def on_depth(m):
        d = np.frombuffer(bytes(m.data), dtype=np.uint16).reshape(m.height, m.width)
        st.update(cam_analyze(d))
        st["cam_t"] = time.time()
        st["cam_yaw"] = st["yaw"]           # heading when this picture arrived
        st["cam_n"] += 1

    from sensor_msgs.msg import Imu
    node.create_subscription(Imu, "/livox/imu", on_imu, qos)
    node.create_subscription(Odometry, "/lidar_odometry/pose", on_pose, qos)
    node.create_subscription(PointCloud2, "/livox/lidar", on_cloud, qos)
    node.create_subscription(Image, "/camera/depth/image_raw", on_depth, qos)

    def on_color(m):
        try:
            img = np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(m.height, m.width, -1)
            img = img[::2, ::2, :3]
            if m.encoding.lower().startswith("bgr"):
                img = img[:, :, ::-1]
            view.set_cam(np.ascontiguousarray(img))
        except Exception:  # noqa
            pass

    if view is not None:
        node.create_subscription(Image, "/camera/color/image_raw", on_color, qos)
    node.create_timer(0.05, lambda: sock.sendto(json.dumps(st).encode(),
                                                ("127.0.0.1", a.port)))
    try:
        rclpy.spin(node)
    except BaseException:  # noqa  (Ctrl+C or being stopped at the end: exit quietly)
        pass


# ---------------------------------------------------------------- helpers
class Link:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.last = None

    def get(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(8192)
            except BlockingIOError:
                return self.last
            self.last = json.loads(data)


def pose_ok(m):
    return bool(m) and m["yaw"] is not None and time.time() - m["pose_t"] < 0.6


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def say(*args):
    try:
        print(*args, flush=True)
    except Exception:  # noqa
        pass


def stop_signal(signum, frame):
    raise KeyboardInterrupt


def ros_init(rclpy):
    """Start ROS WITHOUT its own Ctrl+C handler. ROS's handler shuts the
    connection down the moment Ctrl+C is pressed, so the stop / lie-down /
    damping commands that follow could not be sent ("publisher's context is
    invalid"). With it off, Ctrl+C reaches our code first and the robot is
    always stopped and put in damping before ROS is shut down."""
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError, AttributeError):
        rclpy.init()
    signal.signal(signal.SIGINT, signal.default_int_handler)   # Ctrl+C -> KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGHUP, stop_signal)


def f(v):
    return " -- " if v is None else f"{v:4.2f}"


def side_word(s):
    return {1: "LEFT", -1: "RIGHT", 0: "centre"}[s]


# ---------------------------------------------------------------- decisions
class Brain:
    """His FORWARD / TURNING state machine, measured with the LiDAR heading."""

    def __init__(self, a):
        self.a = a
        self.state = "FORWARD"
        self.turn_dir = 1
        self.turned = 0.0
        self.prev_yaw = None
        self.clear = 0
        self.last_seq = -1
        self.blocked = 0
        self.prog_t = time.time()
        self.prog_xy = None
        self.fwd_start = None            # where forward walking began (LiDAR x, y)
        self.back_from = None
        self.backups = 0
        self.side_from = None            # sidestep: start (x, y, yaw) and direction
        self.bad_since = None            # when sensor data first went missing
        self.side_dir = 0

    def choose_dir(self, m, side):
        if side != 0:
            return -side                       # obstacle on the left -> turn right
        ol = m["l_open_l"] if m["l_open_l"] is not None else 9.0
        orr = m["l_open_r"] if m["l_open_r"] is not None else 9.0
        if abs(ol - orr) > 0.2:
            return 1 if ol > orr else -1       # toward the more open side
        return self.turn_dir                   # no preference: keep the last way

    def step(self, m, use_cam):
        """Returns (vx, vy, wz, status, end_reason or None)."""
        a = self.a
        now = time.time()
        # A short gap in sensor data (the pose or LiDAR arriving late): stand
        # still and wait up to --grace seconds; longer than that: stop.
        bad = None
        if not pose_ok(m):
            bad = "pose lost"
        else:
            d, side, src = merge(m, use_cam)
            if d is None:
                bad = src
        if bad:
            if self.bad_since is None:
                self.bad_since = now
            if now - self.bad_since > a.grace:
                return 0.0, 0.0, 0.0, "", f"STOPPED: {bad} for over {a.grace:.0f} s"
            self.prog_t = now
            return 0.0, 0.0, 0.0, f"waiting: {bad}", None
        self.bad_since = None
        new_frame = m["seq"] != self.last_seq
        self.last_seq = m["seq"]

        if self.state == "FORWARD":
            if new_frame:
                self.blocked = self.blocked + 1 if d < a.stop else 0
            if self.fwd_start is None:
                self.fwd_start = (m["x"], m["y"])
            if self.blocked >= 2:
                self.prog_t = now
                if m.get("tb", 0) >= 2:
                    # 1) one side tight, the other side clear: step sideways into the clear side
                    tl, tr = m.get("tb_l", 0), m.get("tb_r", 0)
                    rl, rr = m.get("room_l", 0), m.get("room_r", 0)
                    sd = -1 if (tl >= 2 and rr < 2) else 1 if (tr >= 2 and rl < 2) else 0
                    if sd != 0 and self.backups < 3:
                        self.state, self.side_dir, self.blocked = "SIDESTEP", sd, 0
                        self.side_from = (m["x"], m["y"], m["yaw"])
                        self.backups += 1
                        say(f"OBSTACLE {d:.2f} m, too tight to turn: stepping "
                            f"{'left' if sd > 0 else 'right'} to make room")
                        return 0.0, 0.0, 0.0, "stopping", None
                    # 2) back up over ground it has just walked on (known clear)
                    walked = math.hypot(m["x"] - self.fwd_start[0], m["y"] - self.fwd_start[1])
                    if walked >= a.back + 0.05 and self.backups < 3:
                        self.state, self.back_from, self.blocked = "BACKING", (m["x"], m["y"]), 0
                        self.backups += 1
                        say(f"OBSTACLE {d:.2f} m but too tight to turn: backing up {a.back:.2f} m")
                        return 0.0, 0.0, 0.0, "stopping", None
                    return 0.0, 0.0, 0.0, "", (f"STOPPED: obstacle {d:.2f} m ahead and something within "
                                               f"{TURN_RADIUS:.2f} m of the body on both sides, too tight "
                                               f"to turn safely")
                self.state, self.turned, self.prev_yaw, self.clear = "TURNING", 0.0, m["yaw"], 0
                self.turn_dir = self.choose_dir(m, side)
                self.blocked = 0
                say(f"OBSTACLE {d:.2f} m ({src}, {side_word(side)}): turning "
                    f"{'left' if self.turn_dir > 0 else 'right'}")
                return 0.0, 0.0, 0.0, "stopping", None
            # stuck check: walking but not moving
            if self.prog_xy is None or math.hypot(m["x"] - self.prog_xy[0],
                                                  m["y"] - self.prog_xy[1]) > 0.15:
                self.prog_xy, self.prog_t = (m["x"], m["y"]), now
            elif now - self.prog_t > 10.0:
                return 0.0, 0.0, 0.0, "", "STOPPED: not moving forward for 10 s (stuck?)"
            v = a.speed if d > a.slow else a.slow_speed
            # keep turning room: veer gently away from something close on one side
            nl, nr = m.get("near_l", 0), m.get("near_r", 0)
            wz, veer = 0.0, ""
            if nl >= 2 and nr < 2:
                wz, veer = -a.veer, ", veering right"
            elif nr >= 2 and nl < 2:
                wz, veer = a.veer, ", veering left"
            return v, 0.0, wz, f"{'FORWARD' if d > a.slow else 'SLOW'} {d:.2f} m ({src}){veer}", None

        if self.state == "SIDESTEP":
            x0, y0, yaw0 = self.side_from
            dx, dy = m["x"] - x0, m["y"] - y0
            lat = -math.sin(yaw0) * dx + math.cos(yaw0) * dy        # + = moved left
            moved = lat * self.side_dir                                  # + = the way we want
            if moved < -0.04:
                return 0.0, 0.0, 0.0, "", ("STOPPED: side-step went the WRONG way (the sign of "
                                           "linear.y is opposite to what I assumed)")
            if m.get("tb", 0) < 2:
                self.state, self.blocked = "FORWARD", 2       # room now: turn at once
                say(f"Room to turn after stepping {abs(lat):.2f} m")
                return 0.0, 0.0, 0.0, "room to turn", None
            target_tight = m.get("tb_l", 0) if self.side_dir > 0 else m.get("tb_r", 0)
            if target_tight >= 2:
                return 0.0, 0.0, 0.0, "", "STOPPED: no room to step sideways either"
            if moved > a.side_max:
                return 0.0, 0.0, 0.0, "", (f"STOPPED: stepped {moved:.2f} m sideways and still "
                                           f"too tight to turn")
            if now - self.prog_t > 12.0 and moved < 0.03:
                return 0.0, 0.0, 0.0, "", "STOPPED: side-step is not moving the robot"
            return 0.0, self.side_dir * a.side_speed, 0.0, f"SIDESTEP {moved:.2f} m", None

        if self.state == "BACKING":
            gone = math.hypot(m["x"] - self.back_from[0], m["y"] - self.back_from[1])
            if gone >= a.back:
                self.state, self.blocked = "FORWARD", 2      # re-check at once: turn if room now
                return 0.0, 0.0, 0.0, "backed up", None
            if now - self.prog_t > 15.0 and gone < 0.05:
                return 0.0, 0.0, 0.0, "", "STOPPED: could not back up"
            return -a.back_speed, 0.0, 0.0, f"BACKING {gone:.2f} of {a.back:.2f} m", None

        # TURNING
        self.turned += wrap(m["yaw"] - self.prev_yaw)
        self.prev_yaw = m["yaw"]
        self.prog_xy, self.prog_t = None, now
        deg = abs(math.degrees(self.turned))
        if m.get("tb", 0) >= 2:
            return 0.0, 0.0, 0.0, "", (f"STOPPED: something came within {TURN_RADIUS:.2f} m of the "
                                  f"body while turning, too tight to keep turning")
        if deg >= 360:
            return 0.0, 0.0, 0.0, "", "STOPPED: turned a full circle and still blocked (boxed in)"
        if new_frame:
            # "clear" is judged by the LiDAR alone while turning: it sees the new
            # direction at once, the camera lags up to 2 s and made it over-turn
            ld = m["l_front"] if m["l_front"] is not None else 9.0
            self.clear = self.clear + 1 if (deg >= a.min_turn and ld > a.slow) else 0
        if self.clear >= a.clear_frames:
            self.state, self.fwd_start, self.backups = "FORWARD", None, 0
            say(f"Clear ({d:.2f} m) after turning {deg:.0f} deg: walking")
            return 0.0, 0.0, 0.0, "clear", None
        return 0.0, 0.0, self.turn_dir * a.turn_rate, f"TURNING {deg:.0f} deg, ahead {d:.2f} m", None


# ---------------------------------------------------------------- check
def run_check(a, link):
    if a.stand:
        return run_check_standing(a, link)
    print("CHECK MODE: the robot will not move. Ctrl+C to quit.")
    print_check_header(a)
    while True:
        time.sleep(0.5)
        print_check_line(a, link)


def print_check_header(a):
    print("LiDAR and camera each give: nearest obstacle in the robot's path, and which side.")
    print("'merged' is what the robot would act on; it stops below "
          f"{a.stop} m and slows below {a.slow} m.\n")


def run_check_standing(a, link):
    """--check --stand: stand up and hold still (no walking) while printing the
    check lines, so the sensors are seen at walking height. Ctrl+C: lie down,
    then damping (same safe ending as wander)."""
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from custom_msg.msg import RobotCommand

    ros_init(rclpy)
    node = Node("wander_check")
    cmd_pub = node.create_publisher(RobotCommand, "/robot_cmd", 10)
    vel_pub = node.create_publisher(Twist, "/vel_cmd", 10)

    def send(state):
        c = RobotCommand()
        c.header.stamp = node.get_clock().now().to_msg()
        c.target_state = state
        cmd_pub.publish(c)
        vel_pub.publish(Twist())              # zero velocity: it never walks

    def hold(state, seconds):
        for _ in range(int(seconds * 20)):
            send(state)
            time.sleep(0.05)

    try:
        print("CHECK MODE, STANDING: the robot stands up but does NOT walk. "
              "Ctrl+C to lie down.")
        print("Standing up (about 6 s)...")
        for s_ in (1, 2, 3):
            hold(s_, 2.0)
        write_state(f"standing {time.time()}")
        print_check_header(a)
        t_end = time.time() + 600                 # at most 10 minutes
        while time.time() < t_end:
            hold(3, 0.5)                          # keep it standing still
            print_check_line(a, link)
        print("10 minutes up.")
    except KeyboardInterrupt:
        pass
    finally:
        for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sg, signal.SIG_IGN)
        try:
            write_state("lying")
            say("\nLying down, then damping mode...")
            hold(3, 1.0)
            hold(1, 2.0)
            hold(0, 2.0)
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
    return 0


def print_check_line(a, link):
    if True:
        m = link.get()
        if not m or m["x"] is None:
            print("waiting for pose (is start_fastlio.sh running?)")
            return
        now = time.time()
        d, side, src = merge(m, not a.no_camera)
        lid = (f"LiDAR {f(m['l_front'])}{'(mem)' if m.get('l_mem') else '     '} {side_word(m['l_side']):6s}" if m["l_ok"]
               else f"LiDAR FLOOR NOT FOUND ({m['l_info']})")
        cam = (f"camera {f(m['c_front'])} {side_word(m['c_side']):6s} (age {now - m['cam_t']:.1f}s)"
               if now - m["cam_t"] < 30 else "camera: no images")
        if d is None:
            act = f"STOP ({src})"
        else:
            act = ("STOP+TURN" if d < a.stop else "SLOW" if d < a.slow else "GO")
            act += f" {d:.2f} m via {src}"
        tb = "TOO CLOSE TO TURN" if m.get("tb", 0) >= 2 else "ok"
        print(f"{lid} | {cam} | open L {f(m['l_open_l'])} R {f(m['l_open_r'])} | turn room {tb} | merged: {act}")


# ---------------------------------------------------------------- run
def run_wander(a, link):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from custom_msg.msg import RobotCommand

    use_cam = not a.no_camera
    print("Waiting for sensors...")
    t0 = time.time()
    while time.time() - t0 < 15:
        time.sleep(0.2)
        m = link.get()
        if pose_ok(m) and merge(m, use_cam)[0] is not None:
            break
    else:
        m = link.get()
        why = "no pose" if not pose_ok(m) else merge(m, use_cam)[2]
        print(f"ERROR: sensors not ready ({why}). Try --check.")
        return 1

    ros_init(rclpy)
    node = Node("wander")
    cmd_pub = node.create_publisher(RobotCommand, "/robot_cmd", 10)
    vel_pub = node.create_publisher(Twist, "/vel_cmd", 10)

    def send(state, vx=0.0, wz=0.0, vy=0.0):
        c = RobotCommand()
        c.header.stamp = node.get_clock().now().to_msg()
        c.target_state = state
        v = Twist()
        v.linear.x, v.linear.y, v.angular.z = float(vx), float(vy), float(wz)
        cmd_pub.publish(c)
        vel_pub.publish(v)

    def hold(state, seconds):
        for _ in range(int(seconds * 20)):
            send(state)
            time.sleep(0.05)

    return wander_loop(a, link, send, hold, use_cam, node, rclpy)


def wander_loop(a, link, send, hold, use_cam, node=None, rclpy=None):
    reason, standing, x0 = "unknown", False, None
    turns = 0
    try:
        print(f"Standing up (about 6 s)... then wandering for up to {a.minutes:g} min "
              f"({'LiDAR + camera' if use_cam else 'LiDAR only'})")
        standing = True
        for s in (1, 2, 3):
            hold(s, 2.0)
        # Standing raises the LiDAR ~20 cm: learn the floor again from this height.
        t_stood, t_wait = time.time(), time.time() + 10.0
        write_state(f"standing {t_stood}")
        while True:
            m = link.get()
            if m and m.get("l_fit_t", 0.0) > t_stood:
                say(f"Floor: {m.get('l_info', '')}")
                break
            if time.time() > t_wait:
                raise RuntimeError("could not see the floor after standing up, and no "
                                   "floor has been saved yet. For this first run, start "
                                   "with about 1.5 m of open floor in front of the robot")
            hold(3, 0.2)
        brain = Brain(a)
        t_end = time.time() + a.minutes * 60
        last_print = 0.0
        m = link.get()
        x0 = (m["x"], m["y"]) if m and m["x"] is not None else None
        while True:
            if time.time() > t_end:
                reason = "time limit reached"
                break
            m = link.get()
            prev = brain.state
            vx, vy, wz, status, end = brain.step(m, use_cam)
            if end:
                reason = end
                break
            if prev == "FORWARD" and brain.state == "TURNING":
                turns += 1
            send(4, vx, wz, vy)
            if status and time.time() - last_print > 1.0:
                last_print = time.time()
                say("  " + status)
            time.sleep(0.05)
    except KeyboardInterrupt:
        reason = "STOPPED: Ctrl+C"
    except Exception as e:  # noqa
        reason = f"STOPPED: error {e}"
    finally:
        for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sg, signal.SIG_IGN)
        try:
            try:
                hold(4, 1.0)                          # stop moving
                say(f"\nResult: {reason}. Turned away from obstacles {turns} times.")
                m = link.get()
                if x0 and m and m["x"] is not None:
                    say(f"Ended {math.hypot(m['x'] - x0[0], m['y'] - x0[1]):.2f} m "
                        f"from where it started (LiDAR).")
            finally:
                write_state("lying")
                if standing:
                    say("Lying down, then damping mode...")
                    hold(3, 1.0)
                    hold(1, 2.0)
                    hold(0, 2.0)
        finally:
            if node is not None:
                node.destroy_node()
                rclpy.try_shutdown()
    return 0 if reason == "time limit reached" else 2


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="sensors only, no movement")
    ap.add_argument("--stand", action="store_true",
                    help="with --check: stand up (but do not walk) while checking")
    ap.add_argument("--sensor", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--no-camera", action="store_true", help="LiDAR only")
    ap.add_argument("--speed", type=float, default=0.15)
    ap.add_argument("--slow-speed", type=float, default=0.10)
    ap.add_argument("--turn-rate", type=float, default=0.15, help="rad/s")
    ap.add_argument("--stop", type=float, default=0.8, help="stop and turn below this (m)")
    ap.add_argument("--slow", type=float, default=1.2,
                    help="slow below this; also 'clear' after a turn means beyond this")
    ap.add_argument("--min-turn", type=float, default=20.0, help="degrees")
    ap.add_argument("--veer", type=float, default=0.10, help="rad/s, to keep turning room")
    ap.add_argument("--back", type=float, default=0.30, help="m to back up when too tight to turn")
    ap.add_argument("--back-speed", type=float, default=0.08)
    ap.add_argument("--grace", type=float, default=2.0,
                    help="seconds to stand still waiting for late sensor data before stopping")
    ap.add_argument("--side-speed", type=float, default=0.06, help="m/s when side-stepping")
    ap.add_argument("--side-max", type=float, default=0.35, help="most it side-steps (m)")
    ap.add_argument("--clear-frames", type=int, default=2)
    ap.add_argument("--xmin", type=float, default=0.60)
    ap.add_argument("--band-lo", type=float, default=0.05,
                    help="LiDAR obstacles: lowest height ABOVE THE LIDAR (his value)")
    ap.add_argument("--band-hi", type=float, default=1.2,
                    help="LiDAR obstacles: highest height above the LiDAR (his value)")
    ap.add_argument("--floor-mode", action="store_true",
                    help="measure LiDAR obstacles from the floor instead (finds the floor)")
    ap.add_argument("--turn-room", type=float, default=0.25,
                    help="room (m from body centre) needed to turn in place; 0.25 = just turn")
    ap.add_argument("--zmin", type=float, default=0.10)
    ap.add_argument("--zmax", type=float, default=1.00)
    ap.add_argument("--port", type=int, default=45680)
    ap.add_argument("--view-port", type=int, default=8080,
                    help="live view in the laptop's browser: http://192.168.1.10:8080 (0 = off)")
    ap.add_argument("--sensor-rmw", default="rmw_cyclonedds_cpp")
    a = ap.parse_args()
    global TURN_RADIUS
    TURN_RADIUS = a.turn_room

    if a.sensor:
        run_sensor(a)
        return 0
    a.speed = max(0.05, min(a.speed, 0.25))
    a.slow_speed = max(0.05, min(a.slow_speed, a.speed))
    a.turn_rate = max(0.05, min(a.turn_rate, 0.30))
    a.minutes = max(0.2, min(a.minutes, 30.0))

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGHUP, stop_signal)
    write_state("lying")                 # the robot is not standing until we stand it up
    link = Link(a.port)
    env = dict(os.environ, RMW_IMPLEMENTATION=a.sensor_rmw)
    helper = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--sensor", "--port", str(a.port),
         "--xmin", str(a.xmin), "--zmin", str(a.zmin), "--zmax", str(a.zmax),
         "--band-lo", str(a.band_lo), "--band-hi", str(a.band_hi),
         "--turn-room", str(a.turn_room), "--stop", str(a.stop), "--slow", str(a.slow),
         "--view-port", str(a.view_port)]
        + (["--floor-mode"] if a.floor_mode else []) + (["--no-camera"] if a.no_camera else []),
        env=env)
    if a.view_port:
        print(f"Live view: on the laptop, open  http://192.168.1.10:{a.view_port}  in a browser")
    try:
        return run_check(a, link) if a.check else run_wander(a, link)
    except KeyboardInterrupt:
        return 0
    finally:
        helper.terminate()


if __name__ == "__main__":
    sys.exit(main())
