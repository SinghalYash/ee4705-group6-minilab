"""Task 4 search-and-approach controller (Student C).

A tick-based state machine, so the same logic runs
  * in a real-time thread against Student A's platform (goto_object.py), and
  * in lock-step with the sandbox simulator for fast, repeatable evaluation.

States
------
SEARCH   rotate in place; the target must be seen in `acquire_k` of the last
         `acquire_n` frames to be acquired. FAIL after a full turn.
TRACK    steer so the bbox centre moves to the image centre and walk forward,
         slowing down as the estimated range shrinks.
CREEP    the object is close and YOLO has dropped it, or its floor contact has
         left the image: cover the last estimated remaining distance on the
         robot's own odometry, then stop.
CONFIRM  stand still and re-detect class + colour on the live frame (C1).
STEPBACK first confirmation failure: back up to ~0.75 m (still inside the
         0.80 m limit) where YOLO sees more of the object, then CONFIRM again.
BACKOFF  later failures: back off, then re-acquire and re-approach.

Range estimation (no ground-truth object poses anywhere in this file)
--------------------------------------------------------------------
Measurement:
  "ground" classes (chair, ball): the bbox bottom edge is where the object
  meets the floor. With camera height h and downward tilt known, that pixel
  row gives a depression angle and a floor distance h / tan(angle).
  "width" classes (a raised stop sign): pinhole range f * W_real / w_px.
  A per-class depth offset turns near-face range into object-centre range.
Filter:
  YOLO sometimes boxes only the seat/backrest of a close chair, which makes
  the ground cue read "far". Each measurement is therefore gated against an
  odometry prediction (last estimate minus distance walked since); outliers
  are rejected, accepted values are blended 50/50 with the prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from perception import Detection, Detector, normalize_class


@dataclass
class CameraModel:
    """Front-camera geometry. Defaults match the example repo's dog_front_camera."""
    width: int = 640
    height: int = 480
    fovy_deg: float = 80.0
    tilt_deg: float = 14.04        # xyaxes "0 -1 0 0.242536 0 0.970143" -> atan(0.2425/0.9701)
    height_m: float = 0.49         # trunk ~0.33 m + camera z offset 0.16 m
    forward_m: float = 0.28        # camera x offset from the trunk origin

    @property
    def f_px(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.fovy_deg) / 2)


# How to range each class. depth_offset: near face -> object centre (m).
# These are object-size priors; update them to match Student A's scene.
CLASS_PRIORS = {
    "chair": {"mode": "ground", "depth_offset": 0.15},
    "sports ball": {"mode": "ground", "depth_offset": 0.12},
    "stop sign": {"mode": "width", "width_m": 0.44, "depth_offset": 0.0},
    "_default": {"mode": "ground", "depth_offset": 0.15},
}


@dataclass
class ControllerConfig:
    timeout_s: float = 60.0
    search_wz: float = 1.0            # ~0.65 rad/s with the example policy
    full_turn_deg: float = 375.0      # 360 + margin for yaw wobble
    acquire_k: int = 2
    acquire_n: int = 3
    lost_frames: int = 10             # consecutive misses before giving up the track
    center_tol: float = 0.08          # |ex| below which no turn is commanded
    k_turn: float = 1.6
    min_turn: float = 0.45            # the policy barely turns below |wz| ~ 0.4
    vx_max: float = 0.7
    vx_slow: float = 0.35
    slow_range_m: float = 1.3
    stop_range_m: float = 0.65        # stop when the object centre is this close (spec: <= 0.80)
    creep_vx: float = 0.30
    creep_max_m: float = 0.45
    near_lost_range_m: float = 1.1    # lost this close -> finish on odometry, don't spin
    near_lost_frames: int = 3
    fill_h: float = 0.85              # bbox this tall/wide (fraction of image) = already close
    fill_w: float = 0.75
    fill_creep_m: float = 0.15        # creep this far after "fills" if no range estimate
    stall_window_s: float = 1.5       # walking but odometry not advancing = blocked
    stall_min_m: float = 0.06
    gate_abs_m: float = 0.25          # range-filter gate: |z - prediction| allowed
    gate_rel: float = 0.25
    reinit_after: int = 6             # consecutive rejected measurements -> re-initialise
    confirm_window_s: float = 1.5
    confirm_k: int = 2                # matching detections needed while standing still
    confirm_far_m: float = 0.75       # step back to this range if a close view can't confirm
    backoff_s: float = 1.2
    max_retries: int = 2
    log_detect_every_s: float = 1.0


@dataclass
class TickResult:
    cmd: tuple                          # (vx, vy, wz), normalised
    status: str | None = None           # None while running, else SUCCESS / FAIL
    reason: str = ""
    found_det: Detection | None = None


class ApproachController:
    def __init__(self, detector: Detector, target_cls: str, target_color: str | None,
                 cfg: ControllerConfig | None = None, cam: CameraModel | None = None,
                 log=print):
        self.det = detector
        self.cls = normalize_class(target_cls)
        self.color = (target_color or "").strip().lower() or None
        self.cfg = cfg or ControllerConfig()
        self.cam = cam or CameraModel()
        self.prior = CLASS_PRIORS.get(self.cls, CLASS_PRIORS["_default"])
        self.log = log

        self.state = None
        self.t0 = None
        self.history: list[bool] = []
        self.misses = 0
        self.turn_dir = 1.0
        self.last_bbox = None
        self.retries = 0
        self.last_detect_log = -1e9
        self.logged_ignored: set = set()
        self.frames = 0
        self.trace: list[dict] = []   # per-tick record for plots / evaluation
        self._reset_range()

    # ----------------------------------------------------------- range filter
    def _reset_range(self):
        self.r_est = None
        self.r_ref_xy = None
        self.r_rejects = 0

    def _predicted_range(self, xy):
        if self.r_est is None:
            return None
        walked = math.hypot(xy[0] - self.r_ref_xy[0], xy[1] - self.r_ref_xy[1])
        return max(0.0, self.r_est - walked)

    def _update_range(self, z, xy):
        """Fuse a visual range measurement z (or None) with odometry. Returns estimate."""
        pred = self._predicted_range(xy)
        if z is None:
            return pred
        if pred is None:
            self.r_est, self.r_ref_xy, self.r_rejects = z, xy, 0
            return z
        gate = max(self.cfg.gate_abs_m, self.cfg.gate_rel * pred)
        if abs(z - pred) <= gate:
            self.r_est, self.r_ref_xy, self.r_rejects = 0.5 * (z + pred), xy, 0
        else:
            self.r_rejects += 1
            # Re-initialise on persistent disagreement, but never jump to a
            # larger range once close: that is the partial-box failure mode.
            may_grow = pred > self.cfg.slow_range_m
            if self.r_rejects >= self.cfg.reinit_after and (z < pred or may_grow):
                self.r_est, self.r_ref_xy, self.r_rejects = z, xy, 0
            else:
                return pred
        return self.r_est

    # ---------------------------------------------------------------- helpers
    def _matches(self, d: Detection) -> bool:
        return d.cls == self.cls and (self.color is None or d.color == self.color)

    def _pick(self, dets: list[Detection]) -> Detection | None:
        cands = [d for d in dets if self._matches(d)]
        if not cands:
            return None
        if self.last_bbox is not None:
            # data association: stay on the box nearest the one being tracked
            lx = 0.5 * (self.last_bbox[0] + self.last_bbox[2])
            return min(cands, key=lambda d: abs(d.cx - lx) - 100 * d.conf)
        return max(cands, key=lambda d: d.conf)

    def measure_range(self, d: Detection) -> tuple[float | None, bool]:
        """Visual range (m, trunk origin -> object centre) and whether the bottom is clipped."""
        cam, H = self.cam, self.cam.height
        x1, y1, x2, y2 = d.bbox
        clipped = y2 >= H - 3
        if self.prior["mode"] == "width":
            if x1 <= 2 or x2 >= cam.width - 3:
                return None, clipped
            r = cam.f_px * self.prior["width_m"] / max(1.0, x2 - x1)
            return r + cam.forward_m + self.prior["depth_offset"], clipped
        if clipped:
            return None, True
        if y1 <= 2:
            # top cut off: close objects are often boxed partially (backrest
            # only), so the bottom edge is not the floor contact any more
            return None, False
        depression = math.radians(cam.tilt_deg) + math.atan((y2 - H / 2) / cam.f_px)
        if depression <= math.radians(1.0):
            return None, False  # contact point at/above the horizon: unusable
        floor = cam.height_m / math.tan(depression)
        return floor + cam.forward_m + self.prior["depth_offset"], False

    def _steer(self, d: Detection) -> tuple[float, float]:
        ex = (d.cx - self.cam.width / 2) / (self.cam.width / 2)   # +: target to the right
        if abs(ex) < self.cfg.center_tol:
            wz = 0.0
        else:
            mag = min(1.0, max(self.cfg.min_turn, self.cfg.k_turn * abs(ex)))
            wz = -math.copysign(mag, ex)                          # right turn = negative wz
        self.turn_dir = -1.0 if ex > 0 else 1.0
        return wz, ex

    def _enter(self, state, t, **kw):
        self.state, self.state_t = state, t
        if state == "SEARCH":
            self.turned = 0.0
            self.next_turn_log = 90.0
            self.history.clear()
            self.last_bbox = None
            self._reset_range()
            self.log(f"[SEARCH] target not visible, rotating "
                     f"{'left' if self.turn_dir > 0 else 'right'}")
        elif state == "CONFIRM":
            self.confirm_hits = 0
        for k, v in kw.items():
            setattr(self, k, v)

    def _log_detect(self, d: Detection, t, force=False, note=""):
        if force or t - self.last_detect_log >= self.cfg.log_detect_every_s:
            self.log(d.log_line() + note)
            self.last_detect_log = t

    def _stalled(self, el, xy) -> bool:
        """True if we have been commanding forward motion but not moving (blocked)."""
        fwd = [s for s in self.trace if s["t"] >= el - self.cfg.stall_window_s]
        if not fwd or el - self.state_t < self.cfg.stall_window_s:
            return False
        if any(s["cmd"][0] < 0.2 for s in fwd):
            return False
        a = fwd[0]
        return math.hypot(xy[0] - a["x"], xy[1] - a["y"]) < self.cfg.stall_min_m

    def _start_creep(self, el, xy, r):
        remaining = 0.0 if r is None else r - self.cfg.stop_range_m
        remaining = float(np.clip(remaining, 0.0, self.cfg.creep_max_m))
        self._enter("CREEP", el, creep_start=xy, creep_dist=remaining)
        return self._creep_cmd(el, xy, 0.0)

    def _creep_cmd(self, el, xy, wz) -> TickResult:
        walked = math.hypot(xy[0] - self.creep_start[0], xy[1] - self.creep_start[1])
        # stall guard: if odometry stops advancing (blocked), stop anyway
        max_t = self.creep_dist / (0.8 * self.cfg.creep_vx) + 1.0
        if walked >= self.creep_dist or el - self.state_t > max_t:
            self._enter("CONFIRM", el)
            return TickResult((0.0, 0.0, 0.0))
        return TickResult((self.cfg.creep_vx, 0.0, 0.6 * wz))

    # ------------------------------------------------------------------- tick
    def tick(self, rgb: np.ndarray, t: float, yaw: float, xy: tuple) -> TickResult:
        """Process one camera frame. t: sim time (s); yaw, xy: robot odometry."""
        cfg = self.cfg
        if self.t0 is None:
            self.t0, self.prev_yaw = t, yaw
            self._enter("SEARCH", 0.0)
        el = t - self.t0
        dyaw = math.atan2(math.sin(yaw - self.prev_yaw), math.cos(yaw - self.prev_yaw))
        self.prev_yaw = yaw
        self.frames += 1

        dets = self.det.detect(rgb, classes=[self.cls])
        d = self._pick(dets)
        self.last_dets, self.last_target = dets, d
        seen = d is not None
        z, clipped = self.measure_range(d) if seen else (None, False)
        # show same-class, wrong-colour boxes once each (disambiguation evidence)
        for o in dets:
            if o.cls == self.cls and not self._matches(o) and o.color not in self.logged_ignored:
                self.logged_ignored.add(o.color)
                self.log(o.log_line() + f"  (ignored: want color={self.color})")

        state_in = self.state
        res = self._step(el, dyaw, d, z, clipped, xy)
        self.trace.append({"t": el, "state": state_in, "seen": seen, "range_meas": z,
                           "range_est": self.r_est, "yaw": yaw, "x": xy[0], "y": xy[1],
                           "cmd": res.cmd})
        return res

    def _step(self, el, dyaw, d, z, clipped, xy) -> TickResult:
        cfg = self.cfg
        seen = d is not None
        if el > cfg.timeout_s:
            return TickResult((0, 0, 0), "FAIL", "timeout")

        # ------------------------------------------------------------ SEARCH
        if self.state == "SEARCH":
            self.turned += abs(dyaw)
            if math.degrees(self.turned) >= self.next_turn_log:
                self.log(f"[SEARCH] rotated {self.next_turn_log:.0f} deg, still searching")
                self.next_turn_log += 90.0
            self.history = (self.history + [seen])[-cfg.acquire_n:]
            if seen and sum(self.history) >= cfg.acquire_k:
                self._log_detect(d, el, force=True)
                self.misses, self.last_bbox = 0, d.bbox
                self._enter("TRACK", el)
                self._update_range(z, xy)
                wz, _ = self._steer(d)
                return TickResult((0.0, 0.0, wz))
            if math.degrees(self.turned) >= cfg.full_turn_deg:
                return TickResult((0, 0, 0), "FAIL", "not_found_after_full_turn")
            # slow the spin when something promising appears so it isn't overshot
            return TickResult((0.0, 0.0, self.turn_dir * (0.6 if seen else cfg.search_wz)))

        # ------------------------------------------------------------- CREEP
        if self.state == "CREEP":
            wz = self._steer(d)[0] if seen else 0.0
            return self._creep_cmd(el, xy, wz)

        # ------------------------------------------------------------- TRACK
        if self.state == "TRACK":
            if not seen:
                self.misses += 1
                r = self._predicted_range(xy)
                if (self.misses >= cfg.near_lost_frames and r is not None
                        and r < cfg.near_lost_range_m):
                    # YOLO often drops a box that fills the frame; the range is
                    # already known, so finish the approach on odometry.
                    return self._start_creep(el, xy, r)
                if self.misses > cfg.lost_frames:
                    self.log(f"[SEARCH] target lost after {self.misses} frames")
                    self._enter("SEARCH", el)
                    return TickResult((0.0, 0.0, self.turn_dir * cfg.search_wz))
                return TickResult((0.0, 0.0, 0.0))   # hold still, wait for re-detect
            self.misses, self.last_bbox = 0, d.bbox
            self._log_detect(d, el)
            wz, ex = self._steer(d)
            r = self._update_range(z, xy)
            x1, y1, x2, y2 = d.bbox
            fills = ((y2 - y1) >= cfg.fill_h * self.cam.height
                     or (x2 - x1) >= cfg.fill_w * self.cam.width)
            if self._stalled(el, xy):
                self._enter("CONFIRM", el)
                return TickResult((0.0, 0.0, 0.0))
            if fills:
                # object fills the view: close, but finish the last bit on
                # odometry (a chair seen back-first fills the frame ~0.8 m out)
                return self._start_creep(el, xy, r if r is not None
                                         else cfg.stop_range_m + cfg.fill_creep_m)
            if r is not None and r <= cfg.stop_range_m:
                self._enter("CONFIRM", el)
                return TickResult((0.0, 0.0, 0.0))
            if clipped and self.prior["mode"] == "ground":
                return self._start_creep(el, xy, r)
            vmax = cfg.vx_slow if (r is not None and r < cfg.slow_range_m) else cfg.vx_max
            vx = vmax * max(0.0, 1.0 - abs(ex) / 0.5)
            return TickResult((vx, 0.0, wz))

        # ----------------------------------------------------------- CONFIRM
        if self.state == "CONFIRM":
            if seen:
                self.confirm_hits += 1
                if self.confirm_hits >= cfg.confirm_k:
                    self._log_detect(d, el, force=True)
                    return TickResult((0, 0, 0), "SUCCESS", "", d)
            if el - self.state_t > cfg.confirm_window_s:
                if self.retries >= cfg.max_retries:
                    return TickResult((0, 0, 0), "FAIL", "not_confirmed_at_goal")
                self.retries += 1
                pred = self._predicted_range(xy)
                if self.retries == 1 and pred is not None and pred < cfg.confirm_far_m:
                    # Very close views (e.g. a chair seen from behind) are hard
                    # for YOLO. Step back to just inside the 0.80 m limit and
                    # look again from there instead of re-approaching.
                    back = cfg.confirm_far_m - pred
                    self.log(f"[SEARCH] not confirmed at goal, stepping back "
                             f"{back:.2f} m to re-check (retry {self.retries}/{cfg.max_retries})")
                    self._enter("STEPBACK", el, back_start=xy, back_dist=back,
                                back_range=cfg.confirm_far_m)
                    return TickResult((-0.3, 0.0, 0.0))
                self.log(f"[SEARCH] not confirmed at goal, backing off "
                         f"(retry {self.retries}/{cfg.max_retries})")
                self._enter("BACKOFF", el)
                return TickResult((-0.4, 0.0, 0.0))
            return TickResult((0.0, 0.0, 0.0))

        # ---------------------------------------------------------- STEPBACK
        if self.state == "STEPBACK":
            walked = math.hypot(xy[0] - self.back_start[0], xy[1] - self.back_start[1])
            if walked >= self.back_dist or el - self.state_t > 3.0:
                # we know how far we are now: reset the range filter to it
                self.r_est, self.r_ref_xy, self.r_rejects = self.back_range, xy, 0
                self._enter("CONFIRM", el)
                return TickResult((0.0, 0.0, 0.0))
            return TickResult((-0.3, 0.0, 0.0))

        # ----------------------------------------------------------- BACKOFF
        if self.state == "BACKOFF":
            if el - self.state_t >= cfg.backoff_s:
                self._reset_range()
                if seen:
                    self.misses, self.last_bbox = 0, d.bbox
                    self._enter("TRACK", el)
                    self._update_range(z, xy)
                else:
                    self._enter("SEARCH", el)
                return TickResult((0.0, 0.0, 0.0))
            return TickResult((-0.4, 0.0, 0.0))

        raise RuntimeError(f"unknown state {self.state}")
