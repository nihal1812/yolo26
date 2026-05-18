#!/usr/bin/env python3
import os
import time
import json
import uuid
import argparse
from collections import defaultdict, deque

import zmq
import numpy as np
import cv2

from config_utils import (
    load_cfg,
    get_zmq_endpoint,
    local_connect_addr,
)


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return float(inter / (area_a + area_b - inter + 1e-6))


def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def poly_to_mask(poly_xy, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    if not poly_xy or len(poly_xy) < 3:
        return mask
    pts = np.array(poly_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def crop_resize(mask, bbox, out_hw):
    H, W = mask.shape
    x1, y1, x2, y2 = bbox
    x1 = int(clamp(x1, 0, W - 1))
    x2 = int(clamp(x2, 0, W - 1))
    y1 = int(clamp(y1, 0, H - 1))
    y2 = int(clamp(y2, 0, H - 1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(out_hw, dtype=np.uint8)
    crop = mask[y1:y2, x1:x2]
    return cv2.resize(crop, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)


def joint_heatmaps(kp_xy, kp_conf, bbox, out_hw, sigma=2.5, conf_thr=0.3):
    H, W = out_hw
    if not kp_xy or not bbox:
        return np.zeros((17, H, W), dtype=np.float32)

    x1, y1, x2, y2 = bbox
    bw = max(1.0, (x2 - x1))
    bh = max(1.0, (y2 - y1))
    J = min(17, len(kp_xy))
    maps = np.zeros((17, H, W), dtype=np.float32)

    rad = int(max(1, round(3 * sigma)))
    xs = np.arange(-rad, rad + 1)
    ys = np.arange(-rad, rad + 1)
    xx, yy = np.meshgrid(xs, ys)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)

    for j in range(J):
        c = kp_conf[j] if kp_conf and j < len(kp_conf) else 1.0
        if c < conf_thr:
            continue

        x, y = kp_xy[j]
        u = (x - x1) / bw
        v = (y - y1) / bh
        px = int(clamp(u, 0.0, 1.0) * (W - 1))
        py = int(clamp(v, 0.0, 1.0) * (H - 1))

        x0 = px - rad
        x1m = px + rad
        y0 = py - rad
        y1m = py + rad

        kx0 = 0
        ky0 = 0
        kx1 = kernel.shape[1] - 1
        ky1 = kernel.shape[0] - 1

        if x0 < 0:
            kx0 = -x0
            x0 = 0
        if y0 < 0:
            ky0 = -y0
            y0 = 0
        if x1m >= W:
            kx1 = kernel.shape[1] - 1 - (x1m - (W - 1))
            x1m = W - 1
        if y1m >= H:
            ky1 = kernel.shape[0] - 1 - (y1m - (H - 1))
            y1m = H - 1

        patch = kernel[ky0:ky1 + 1, kx0:kx1 + 1] * float(c)
        maps[j, y0:y1m + 1, x0:x1m + 1] = np.maximum(
            maps[j, y0:y1m + 1, x0:x1m + 1], patch
        )

    return maps


L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_WRI, R_WRI = 9, 10


def safe_kp(kp_xy, idx):
    if not kp_xy or idx >= len(kp_xy):
        return None
    return tuple(kp_xy[idx])


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def l2(a, b):
    return float(np.sqrt(dist2(a, b)))


def angle_shoulder_line(kp_xy):
    ls = safe_kp(kp_xy, L_SHO)
    rs = safe_kp(kp_xy, R_SHO)
    if not ls or not rs:
        return None
    return float(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))


def normalize_len(kp_xy, bbox_xyxy):
    ls = safe_kp(kp_xy, L_SHO)
    rs = safe_kp(kp_xy, R_SHO)
    lh = safe_kp(kp_xy, L_HIP)
    rh = safe_kp(kp_xy, R_HIP)

    if ls and rs and lh and rh:
        chest = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        d = l2(chest, hip)
        if d > 1e-3:
            return d

    x1, y1, x2, y2 = bbox_xyxy
    return float(max(1.0, (y2 - y1)))


def point_in_poly(pt, poly):
    if not poly or len(poly) < 3:
        return False
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            if x < x_int:
                inside = not inside
    return inside


def object_score(person_bbox, wrists_xy, obj_bbox):
    pcx, pcy = bbox_center(person_bbox)
    ocx, ocy = bbox_center(obj_bbox)
    d_center = (ocx - pcx) ** 2 + (ocy - pcy) ** 2
    if wrists_xy:
        d_wrist = min((ocx - wx) ** 2 + (ocy - wy) ** 2 for (wx, wy) in wrists_xy)
        return float(min(d_center, d_wrist))
    return float(d_center)


def pick_best_object(person_bbox, wrists_xy, obj_instances):
    if not obj_instances:
        return (None, None)
    best = (1e18, None)
    for idx, inst in enumerate(obj_instances):
        sc = object_score(person_bbox, wrists_xy, inst["bbox_xyxy"])
        if sc < best[0]:
            best = (sc, idx)
    return (best[1], best[0])


def stabilize_object_selection(pid, best_oid, best_score, last_oid, last_score, switch_state, margin_ratio=0.80, confirm_frames=3):
    pending = False
    switched = False
    countdown = 0

    if last_oid is None or last_oid < 0:
        switch_state.pop(pid, None)
        return best_oid, best_score, False, False, 0

    if best_oid == last_oid:
        switch_state.pop(pid, None)
        return last_oid, last_score, False, False, 0

    if last_score is None or best_score is None:
        switch_state.pop(pid, None)
        return best_oid, best_score, False, True, 0

    better_enough = best_score < (margin_ratio * last_score)
    st = switch_state.get(pid, {"cand_oid": None, "count": 0})

    if not better_enough:
        switch_state.pop(pid, None)
        return last_oid, last_score, False, False, 0

    if st["cand_oid"] != best_oid:
        st = {"cand_oid": best_oid, "count": 1}
    else:
        st["count"] += 1

    if st["count"] >= confirm_frames:
        switch_state.pop(pid, None)
        return best_oid, best_score, False, True, 0

    switch_state[pid] = st
    pending = True
    countdown = confirm_frames - st["count"]
    return last_oid, last_score, pending, False, countdown


def compute_velocity(vec_hist):
    if len(vec_hist) < 2:
        return None
    vxs, vys = [], []
    for i in range(1, len(vec_hist)):
        t0, x0, y0 = vec_hist[i - 1]
        t1, x1, y1 = vec_hist[i]
        dt = t1 - t0
        if dt <= 1e-4:
            continue
        vxs.append((x1 - x0) / dt)
        vys.append((y1 - y0) / dt)
    if not vxs:
        return None
    return (float(np.mean(vxs)), float(np.mean(vys)))


def cosine_sim(v1, v2):
    if v1 is None or v2 is None:
        return None
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.dot(a, b) / (na * nb))


def vec_norm(v):
    if v is None:
        return None
    return float(np.sqrt(v[0] ** 2 + v[1] ** 2))


def _as_float_list(vals):
    out = []
    for v in vals:
        if v is None:
            continue
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def agg_stats(vals):
    xs = _as_float_list(vals)
    if not xs:
        return {"mean": None, "std": None, "min": None, "max": None, "last": None}
    arr = np.array(xs, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "last": float(xs[-1]),
    }


def build_policy_features(agg, q_pose_miss, q_obj_miss):
    carry_max = None
    carry_dict = agg.get("carry_score", None)
    if isinstance(carry_dict, dict):
        carry_max = carry_dict.get("max", None)

    return {
        "contact_ratio": float(agg.get("contact_ratio", 0.0) or 0.0),
        "visibility_drop": agg.get("visibility_drop", None),
        "carry_score_max": carry_max,
        "disappeared_after_contact": bool(agg.get("disappeared_after_contact", False)),
        "heuristic_theft_score": agg.get("heuristic_theft_score", None),
        "clip_dt_s": agg.get("clip_dt_s", None),
        "missing_pose_ratio": float(q_pose_miss),
        "missing_obj_ratio": float(q_obj_miss),
    }


class FeatureBuilder:
    """
    Redis-free feature builder:
      - clip tensors saved to local storage
      - frame scalars via ZMQ PUB
      - clip/scalars_clip transport via ZMQ PUB
    """

    def __init__(self, cfg_path: str, cam_id: str, args=None, debug: bool = False):
        self.cfg = load_cfg(cfg_path)
        self.cam_id = str(cam_id)
        self.debug = bool(debug)

        if args is None:
            args = argparse.Namespace(
                T=16,
                clip_h=112,
                clip_w=112,
                sigma=2.5,
                kp_conf_thr=0.3,
                contact_use_poly=True,
                contact_dist_px=25.0,
                vis_hist=30,
                comotion_window=10,
                obj_switch_margin_ratio=0.80,
                obj_switch_confirm_frames=3,
                disappear_miss_frames=12,
                disappear_contact_min_frames=5,
                disappear_window_frames=40,
                person_state_ttl_frames=120,
                object_state_ttl_frames=240,
                emit_heuristic_theft_score=False,
                max_join_buf=512,
                clip_dir="/tmp/zono_clips",
                clip_codec="npz",
                clip_retention_s=1800,
                clip_cleanup_interval_s=60.0,
                publish_frame_scalars=True,
                publish_clip_events=True,
                publish_scalars_clip_events=True,
            )
        self.args = args

        self.publish_frame_scalars = bool(
            getattr(
                args,
                "publish_frame_scalars",
                getattr(args, "publish_frame_scalars_redis", True),
            )
        )
        self.publish_clip_events = bool(getattr(args, "publish_clip_events", True))
        self.publish_scalars_clip_events = bool(getattr(args, "publish_scalars_clip_events", True))

        frame_scalars_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "scalars")
        clips_evt_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "clips_events")
        scalars_clip_evt_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "scalars_clip_events")

        self.ctx = zmq.Context.instance()

        self.frame_scalars_bind = frame_scalars_cfg["bind"]
        self.frame_scalars_topic = frame_scalars_cfg["topic"]
        self.frame_scalars_topic_b = self.frame_scalars_topic.encode("utf-8")
        self.frame_scalars_sndhwm = int(frame_scalars_cfg.get("sndhwm", 1024))
        self.frame_scalars_pub = None
        self.frame_scalars_drop_count = 0

        self.clip_evt_bind = clips_evt_cfg["bind"]
        self.clip_evt_topic = clips_evt_cfg["topic"]
        self.clip_evt_topic_b = self.clip_evt_topic.encode("utf-8")
        self.clip_evt_sndhwm = int(clips_evt_cfg.get("sndhwm", 256))
        self.clip_pub = None
        self.clip_pub_drop_count = 0

        self.scalars_clip_evt_bind = scalars_clip_evt_cfg["bind"]
        self.scalars_clip_evt_topic = scalars_clip_evt_cfg["topic"]
        self.scalars_clip_evt_topic_b = self.scalars_clip_evt_topic.encode("utf-8")
        self.scalars_clip_evt_sndhwm = int(scalars_clip_evt_cfg.get("sndhwm", 512))
        self.scalars_clip_pub = None
        self.scalars_clip_pub_drop_count = 0

        if self.publish_frame_scalars:
            self.frame_scalars_pub = self.ctx.socket(zmq.PUB)
            self.frame_scalars_pub.setsockopt(zmq.LINGER, 0)
            self.frame_scalars_pub.setsockopt(zmq.SNDHWM, self.frame_scalars_sndhwm)
            self.frame_scalars_pub.bind(self.frame_scalars_bind)

        if self.publish_clip_events:
            self.clip_pub = self.ctx.socket(zmq.PUB)
            self.clip_pub.setsockopt(zmq.LINGER, 0)
            self.clip_pub.setsockopt(zmq.SNDHWM, self.clip_evt_sndhwm)
            self.clip_pub.bind(self.clip_evt_bind)

        if self.publish_scalars_clip_events:
            self.scalars_clip_pub = self.ctx.socket(zmq.PUB)
            self.scalars_clip_pub.setsockopt(zmq.LINGER, 0)
            self.scalars_clip_pub.setsockopt(zmq.SNDHWM, self.scalars_clip_evt_sndhwm)
            self.scalars_clip_pub.bind(self.scalars_clip_evt_bind)

        self.out_hw = (args.clip_h, args.clip_w)

        self.clip_buf = defaultdict(lambda: deque(maxlen=args.T))
        self.scalar_window = defaultdict(lambda: deque(maxlen=args.T))
        self.miss_pose_win = defaultdict(lambda: deque(maxlen=args.T))
        self.miss_obj_win = defaultdict(lambda: deque(maxlen=args.T))

        self.person_prev = {}
        self.person_last_seen = {}

        self.contact_count = defaultdict(int)
        self.last_contact_obj = {}
        self.last_contact_frame = {}
        self.last_contact_dur = {}

        self.obj_area_hist = defaultdict(lambda: deque(maxlen=args.vis_hist))
        self.obj_last_seen = {}
        self.comotion_hist = defaultdict(lambda: deque(maxlen=args.comotion_window))

        self.last_selected_obj = {}
        self.last_selected_score = {}
        self.switch_state = {}

        self.clip_dir = os.path.abspath(args.clip_dir)
        self.clip_codec = str(args.clip_codec).lower()
        self.clip_retention_s = int(args.clip_retention_s)
        self.clip_cleanup_interval_s = float(args.clip_cleanup_interval_s)
        self._last_clip_cleanup = 0.0

        os.makedirs(self.clip_dir, exist_ok=True)

        print(f"[feature_builder] cam_id={self.cam_id}")
        print(f"[feature_builder] clip storage dir={self.clip_dir} codec={self.clip_codec}")
        print(
            f"[feature_builder] frame scalars={self.frame_scalars_bind} "
            f"topic={self.frame_scalars_topic} enabled={self.publish_frame_scalars}"
        )
        print(f"[feature_builder] clip events={self.clip_evt_bind} topic={self.clip_evt_topic} enabled={self.publish_clip_events}")
        print(
            f"[feature_builder] scalars_clip events={self.scalars_clip_evt_bind} "
            f"topic={self.scalars_clip_evt_topic} enabled={self.publish_scalars_clip_events}"
        )
        print(f"[feature_builder] clip: C=19 T={args.T} HxW={self.out_hw}")

    def _jcompact(self, obj) -> str:
        return json.dumps(obj, separators=(",", ":"))

    def _clip_file_path(self, cam, pid, fid, stamp_ns):
        ext = "npz" if self.clip_codec == "npz" else "npy"
        fname = f"{cam}_pid{pid}_fid{fid}_ts{stamp_ns}_{uuid.uuid4().hex[:8]}.{ext}"
        return os.path.join(self.clip_dir, fname)

    def _save_clip_to_file(self, clip_arr, cam, pid, fid, stamp_ns):
        clip_arr = np.ascontiguousarray(clip_arr, dtype=np.float32)
        clip_path = self._clip_file_path(cam, pid, fid, stamp_ns)
        tmp_path = clip_path + ".tmp"

        with open(tmp_path, "wb") as f:
            if self.clip_codec == "npz":
                np.savez_compressed(f, clip=clip_arr)
            else:
                np.save(f, clip_arr, allow_pickle=False)

        os.replace(tmp_path, clip_path)
        return clip_path

    def _cleanup_old_clip_files(self, force=False):
        now = time.time()
        if not force and (now - self._last_clip_cleanup) < self.clip_cleanup_interval_s:
            return
        self._last_clip_cleanup = now

        cutoff = now - self.clip_retention_s
        try:
            for name in os.listdir(self.clip_dir):
                path = os.path.join(self.clip_dir, name)
                try:
                    st = os.stat(path)
                    if st.st_mtime < cutoff and os.path.isfile(path):
                        os.remove(path)
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
        except Exception:
            pass

    def _publish_clip_event(self, payload):
        if self.clip_pub is None:
            return
        try:
            self.clip_pub.send_multipart(
                [
                    self.clip_evt_topic_b,
                    payload["event_id"].encode("utf-8"),
                    self._jcompact(payload).encode("utf-8"),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            self.clip_pub_drop_count += 1
            if self.debug and (self.clip_pub_drop_count % 50 == 0):
                print(f"[feature_builder][{self.cam_id}] clip event PUB backpressure drops={self.clip_pub_drop_count}")

    def _publish_scalars_clip_event(self, payload):
        if self.scalars_clip_pub is None:
            return
        try:
            event_id = payload.get("meta", {}).get("event_id", "")
            self.scalars_clip_pub.send_multipart(
                [
                    self.scalars_clip_evt_topic_b,
                    str(event_id).encode("utf-8"),
                    self._jcompact(payload).encode("utf-8"),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            self.scalars_clip_pub_drop_count += 1
            if self.debug and (self.scalars_clip_pub_drop_count % 50 == 0):
                print(f"[feature_builder][{self.cam_id}] scalars_clip PUB backpressure drops={self.scalars_clip_pub_drop_count}")

    def _publish_clip(self, cam, pid, fid, stamp_ns, frame_w, frame_h, obj_meta, clip_arr, q_pose_miss, q_obj_miss):
        event_id = f"{cam}:{pid}:{fid}:{stamp_ns}"
        clip_path = self._save_clip_to_file(clip_arr, cam, pid, fid, stamp_ns)

        meta = {
            "type": "clip",
            "event_id": event_id,
            "cam_id": cam,
            "person_track_id": int(pid),
            "frame_id_end": int(fid),
            "stamp_ns_end": int(stamp_ns),
            "frame_w": int(frame_w),
            "frame_h": int(frame_h),
            "C": int(clip_arr.shape[0]),
            "T": int(clip_arr.shape[1]),
            "H": int(clip_arr.shape[2]),
            "W": int(clip_arr.shape[3]),
            "dtype": "float32",
            "object_track_id": int(obj_meta.get("track_id", -1)),
            "object_class_id": int(obj_meta.get("class_id", -1)),
            "mode": "heatmaps",
            "missing_pose_ratio": float(q_pose_miss),
            "missing_obj_ratio": float(q_obj_miss),
            "clip_path": clip_path,
            "clip_codec": self.clip_codec,
        }

        payload = {
            "event_id": event_id,
            "clip_path": clip_path,
            "clip_codec": self.clip_codec,
            "meta": meta,
        }

        self._publish_clip_event(payload)

    def _publish_scalars_frame(self, payload_dict):
        if self.frame_scalars_pub is None:
            return
        try:
            event_id = payload_dict.get("event_id", "")
            self.frame_scalars_pub.send_multipart(
                [
                    self.frame_scalars_topic_b,
                    str(event_id).encode("utf-8"),
                    self._jcompact(payload_dict).encode("utf-8"),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            self.frame_scalars_drop_count += 1
            if self.debug and (self.frame_scalars_drop_count % 50 == 0):
                print(
                    f"[feature_builder][{self.cam_id}] frame scalars PUB backpressure "
                    f"drops={self.frame_scalars_drop_count}"
                )

    def _publish_scalars_clip(self, cam, pid, fid, stamp_ns, obj_meta, agg_payload, q_pose_miss, q_obj_miss):
        event_id = f"{cam}:{pid}:{fid}:{stamp_ns}"
        policy_features = build_policy_features(agg_payload, q_pose_miss, q_obj_miss)
        meta = {
            "type": "scalars_clip",
            "event_id": event_id,
            "cam_id": cam,
            "person_track_id": int(pid),
            "frame_id_end": int(fid),
            "stamp_ns_end": int(stamp_ns),
            "object_track_id": int(obj_meta.get("track_id", -1)),
            "object_class_id": int(obj_meta.get("class_id", -1)),
            "T": int(self.args.T),
        }
        out = {
            "meta": meta,
            "features": agg_payload,
            "policy_features": policy_features,
        }

        self._publish_scalars_clip_event(out)

    def _compute_clip_dt_s(self, win):
        if not win or len(win) < 2:
            return None
        s0 = win[0].get("stamp_ns", None)
        s1 = win[-1].get("stamp_ns", None)
        if s0 is None or s1 is None:
            return None
        try:
            dt = (int(s1) - int(s0)) * 1e-9
            if dt <= 0:
                return None
            return float(dt)
        except Exception:
            return None

    def _compute_obj_missing_frames(self, fid_now, oid, fallback=None):
        if oid is None or oid < 0:
            return fallback
        last_seen = self.obj_last_seen.get(int(oid), None)
        if last_seen is None:
            return fallback
        try:
            return int(fid_now - int(last_seen))
        except Exception:
            return fallback

    def _cleanup_person_state(self, fid_now: int):
        stale_pids = []
        for pid, last_fid in self.person_last_seen.items():
            try:
                if (fid_now - int(last_fid)) > self.args.person_state_ttl_frames:
                    stale_pids.append(pid)
            except Exception:
                stale_pids.append(pid)

        for pid in stale_pids:
            self.clip_buf.pop(pid, None)
            self.scalar_window.pop(pid, None)
            self.miss_pose_win.pop(pid, None)
            self.miss_obj_win.pop(pid, None)
            self.person_prev.pop(pid, None)
            self.last_contact_obj.pop(pid, None)
            self.last_contact_frame.pop(pid, None)
            self.last_contact_dur.pop(pid, None)
            self.last_selected_obj.pop(pid, None)
            self.last_selected_score.pop(pid, None)
            self.switch_state.pop(pid, None)
            self.person_last_seen.pop(pid, None)

            for key in list(self.contact_count.keys()):
                if key[0] == pid:
                    self.contact_count.pop(key, None)

            for key in list(self.comotion_hist.keys()):
                if key[0] == pid:
                    self.comotion_hist.pop(key, None)

    def _cleanup_object_state(self, fid_now: int):
        stale_oids = []
        for oid, last_fid in self.obj_last_seen.items():
            try:
                if (fid_now - int(last_fid)) > self.args.object_state_ttl_frames:
                    stale_oids.append(oid)
            except Exception:
                stale_oids.append(oid)

        for oid in stale_oids:
            self.obj_last_seen.pop(oid, None)
            self.obj_area_hist.pop(oid, None)

        for key in list(self.contact_count.keys()):
            _pid, oid = key
            if oid not in self.obj_last_seen:
                self.contact_count.pop(key, None)

        for key in list(self.comotion_hist.keys()):
            _pid, oid = key
            if oid not in self.obj_last_seen:
                self.comotion_hist.pop(key, None)

    def process_pair(self, pose: dict, seg: dict):
        cam = pose.get("cam_id", self.cam_id)
        fid = int(pose.get("frame_id", 0))

        pose_stamp_ns = int(pose.get("stamp_ns", 0))
        seg_stamp_ns = int(seg.get("stamp_ns", 0))
        stamp_ns = pose_stamp_ns if pose_stamp_ns > 0 else seg_stamp_ns
        if seg_stamp_ns > 0:
            stamp_ns = min(pose_stamp_ns, seg_stamp_ns) if pose_stamp_ns > 0 else seg_stamp_ns
        if stamp_ns <= 0:
            stamp_ns = time.time_ns()

        t_s = stamp_ns * 1e-9

        frame_w = int(pose.get("frame_w", seg.get("frame_w", 0)))
        frame_h = int(pose.get("frame_h", seg.get("frame_h", 0)))
        if frame_w <= 0 or frame_h <= 0:
            return

        people = pose.get("people", [])
        instances = seg.get("instances", [])

        full_masks = [poly_to_mask(inst.get("mask_poly_xy", []), frame_h, frame_w) for inst in instances]
        person_idxs = [i for i, inst in enumerate(instances) if int(inst.get("class_id", -1)) == 0]
        obj_idxs = [i for i, inst in enumerate(instances) if int(inst.get("class_id", -1)) != 0]

        obj_by_id = {}
        for i in obj_idxs:
            oid = int(instances[i].get("track_id", -1))
            if oid >= 0:
                obj_by_id[oid] = instances[i]
                self.obj_last_seen[oid] = fid
                area = float(instances[i].get("mask_area_px", 0.0))
                if area > 0:
                    self.obj_area_hist[oid].append(area)

        zero_mask = np.zeros((frame_h, frame_w), np.uint8)

        for person in people:
            pid = int(person.get("track_id", -1))
            if pid < 0:
                continue
            pb = person.get("bbox_xyxy", None)
            if not pb:
                continue

            self.person_last_seen[pid] = fid

            kp_xy = person.get("keypoints_xy", [])
            kp_cf = person.get("keypoints_conf", [])

            lw = safe_kp(kp_xy, L_WRI)
            rw = safe_kp(kp_xy, R_WRI)
            wrists = []
            if lw:
                wrists.append(lw)
            if rw:
                wrists.append(rw)

            miss_pose = 1.0
            if kp_cf and len(kp_cf) >= 17:
                cf = np.array(kp_cf[:17], dtype=np.float32)
                miss_pose = float(np.mean(cf < float(self.args.kp_conf_thr)))
            self.miss_pose_win[pid].append(miss_pose)

            best_iou = 0.0
            best_person_inst_i = None
            for i in person_idxs:
                val = iou_xyxy(pb, instances[i]["bbox_xyxy"])
                if val > best_iou:
                    best_iou = val
                    best_person_inst_i = i
            person_full = full_masks[best_person_inst_i] if best_person_inst_i is not None else zero_mask

            obj_instances = [instances[i] for i in obj_idxs]
            best_idx, best_score = pick_best_object(pb, wrists, obj_instances)
            best_oid = int(obj_instances[best_idx].get("track_id", -1)) if best_idx is not None else -1

            last_oid = self.last_selected_obj.get(pid, None)
            last_sc = self.last_selected_score.get(pid, None)
            sel_oid, sel_sc, pending, switched, countdown = stabilize_object_selection(
                pid=pid,
                best_oid=best_oid,
                best_score=best_score,
                last_oid=last_oid,
                last_score=last_sc,
                switch_state=self.switch_state,
                margin_ratio=self.args.obj_switch_margin_ratio,
                confirm_frames=self.args.obj_switch_confirm_frames,
            )
            self.last_selected_obj[pid] = sel_oid if sel_oid is not None else -1
            self.last_selected_score[pid] = sel_sc

            if switched:
                for key in list(self.comotion_hist.keys()):
                    if key[0] == pid:
                        self.comotion_hist.pop(key, None)

            obj_meta = {"track_id": -1, "class_id": -1, "bbox_xyxy": None, "mask_poly_xy": None}
            obj_full = zero_mask

            if sel_oid is not None and sel_oid >= 0:
                inst = obj_by_id.get(sel_oid, None)
                if inst is not None:
                    obj_meta = {
                        "track_id": int(inst.get("track_id", -1)),
                        "class_id": int(inst.get("class_id", -1)),
                        "bbox_xyxy": inst.get("bbox_xyxy", None),
                        "mask_poly_xy": inst.get("mask_poly_xy", None),
                    }
                    for i in obj_idxs:
                        if int(instances[i].get("track_id", -1)) == sel_oid:
                            obj_full = full_masks[i]
                            break

            oid = int(obj_meta.get("track_id", -1))
            miss_obj = 1.0 if oid < 0 else 0.0
            self.miss_obj_win[pid].append(miss_obj)

            p_mask = crop_resize(person_full, pb, self.out_hw).astype(np.float32, copy=False)
            o_mask = crop_resize(obj_full, pb, self.out_hw).astype(np.float32, copy=False)
            hm = joint_heatmaps(kp_xy, kp_cf, pb, self.out_hw, sigma=self.args.sigma, conf_thr=self.args.kp_conf_thr)
            frame_maps = np.concatenate([p_mask[None, :, :], o_mask[None, :, :], hm], axis=0).astype(np.float32, copy=False)
            self.clip_buf[pid].append(frame_maps)

            ls = safe_kp(kp_xy, L_SHO)
            rs = safe_kp(kp_xy, R_SHO)
            lh = safe_kp(kp_xy, L_HIP)
            rh = safe_kp(kp_xy, R_HIP)

            chest = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0) if (ls and rs) else None
            hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0) if (lh and rh) else None
            norm = normalize_len(kp_xy, pb)

            hand_to_chest_L = (l2(lw, chest) / norm) if (lw and chest) else None
            hand_to_chest_R = (l2(rw, chest) / norm) if (rw and chest) else None
            hand_to_hip_L = (l2(lw, hip) / norm) if (lw and hip) else None
            hand_to_hip_R = (l2(rw, hip) / norm) if (rw and hip) else None

            ang = angle_shoulder_line(kp_xy)

            prev = self.person_prev.get(pid, None)
            v_l = v_r = None
            a_l = a_r = None
            ang_rate = None
            if prev is not None:
                dt = (stamp_ns - prev["stamp_ns"]) * 1e-9
                if dt > 1e-4:
                    if lw and prev["lw"]:
                        v_l = l2(lw, prev["lw"]) / dt
                    if rw and prev["rw"]:
                        v_r = l2(rw, prev["rw"]) / dt
                    if prev.get("v_l") is not None and v_l is not None:
                        a_l = (v_l - prev["v_l"]) / dt
                    if prev.get("v_r") is not None and v_r is not None:
                        a_r = (v_r - prev["v_r"]) / dt
                    if ang is not None and prev.get("ang") is not None:
                        da = ang - prev["ang"]
                        if da > np.pi:
                            da -= 2 * np.pi
                        if da < -np.pi:
                            da += 2 * np.pi
                        ang_rate = float(da / dt)

            guard_score = None
            if chest:
                near_L = hand_to_chest_L is not None and hand_to_chest_L < 0.35
                near_R = hand_to_chest_R is not None and hand_to_chest_R < 0.35
                slow_L = v_l is not None and v_l < 80.0
                slow_R = v_r is not None and v_r < 80.0
                move_L = v_l is not None and v_l > 120.0
                move_R = v_r is not None and v_r > 120.0
                guard_score = 1.0 if ((near_L and slow_L and move_R) or (near_R and slow_R and move_L)) else 0.0

            contact = False
            if oid >= 0 and obj_meta.get("bbox_xyxy") is not None:
                if self.args.contact_use_poly and obj_meta.get("mask_poly_xy"):
                    if lw and point_in_poly(lw, obj_meta["mask_poly_xy"]):
                        contact = True
                    if rw and point_in_poly(rw, obj_meta["mask_poly_xy"]):
                        contact = True
                else:
                    oc = bbox_center(obj_meta["bbox_xyxy"])
                    thr2 = self.args.contact_dist_px ** 2
                    if lw and dist2(lw, oc) < thr2:
                        contact = True
                    if rw and dist2(rw, oc) < thr2:
                        contact = True

            if oid >= 0 and contact:
                self.contact_count[(pid, oid)] += 1
                self.last_contact_obj[pid] = oid
                self.last_contact_frame[pid] = fid
                self.last_contact_dur[pid] = int(self.contact_count[(pid, oid)])
            else:
                if oid >= 0:
                    self.contact_count[(pid, oid)] = 0
            contact_frames = self.contact_count.get((pid, oid), 0) if oid >= 0 else 0

            visibility = None
            obj_area = None
            if oid >= 0:
                inst = obj_by_id.get(oid, None)
                if inst:
                    obj_area = float(inst.get("mask_area_px", 0.0))
                hist = self.obj_area_hist.get(oid, None)
                if hist and len(hist) >= 5:
                    med = float(np.median(np.array(hist, dtype=np.float32)))
                    if med > 1e-6 and obj_area is not None:
                        visibility = float(obj_area / med)

            disappeared_after_contact = False
            last_oid2 = self.last_contact_obj.get(pid, None)
            last_cf = self.last_contact_frame.get(pid, None)
            last_cd = int(self.last_contact_dur.get(pid, 0))

            if last_oid2 is not None and last_oid2 >= 0 and last_cf is not None:
                last_seen = self.obj_last_seen.get(last_oid2, None)
                if last_seen is not None:
                    missing = fid - last_seen
                    since_contact = fid - last_cf
                    if (
                        last_cd >= self.args.disappear_contact_min_frames
                        and since_contact <= self.args.disappear_window_frames
                        and missing >= self.args.disappear_miss_frames
                    ):
                        disappeared_after_contact = True

            pcx, pcy = bbox_center(pb)
            person_speed = None
            object_speed = None
            comotion_cos = None
            comotion_speed_ratio = None
            rel_offset_std_px = None
            carry_score = None

            if oid >= 0 and obj_meta.get("bbox_xyxy") is not None:
                ocx, ocy = bbox_center(obj_meta["bbox_xyxy"])
                key = (pid, oid)
                self.comotion_hist[key].append((t_s, pcx, pcy, ocx, ocy))

                hist = list(self.comotion_hist[key])
                if len(hist) >= 3:
                    p_traj = [(h[0], h[1], h[2]) for h in hist]
                    o_traj = [(h[0], h[3], h[4]) for h in hist]
                    v_p = compute_velocity(p_traj)
                    v_o = compute_velocity(o_traj)

                    comotion_cos = cosine_sim(v_p, v_o)
                    sp = vec_norm(v_p)
                    so = vec_norm(v_o)
                    person_speed = sp
                    object_speed = so
                    if sp is not None and so is not None and sp > 1e-6:
                        comotion_speed_ratio = float(so / sp)

                    offsets = np.array([(h[3] - h[1], h[4] - h[2]) for h in hist], dtype=np.float32)
                    if offsets.shape[0] >= 3:
                        stdx = float(np.std(offsets[:, 0]))
                        stdy = float(np.std(offsets[:, 1]))
                        rel_offset_std_px = float(np.sqrt(stdx**2 + stdy**2))

                    if comotion_cos is not None and rel_offset_std_px is not None:
                        cos_term = clamp((comotion_cos + 1.0) / 2.0, 0.0, 1.0)
                        off_term = clamp(1.0 - (rel_offset_std_px / 30.0), 0.0, 1.0)
                        carry_score = float(0.6 * cos_term + 0.4 * off_term)

            event_id = f"{cam}:{pid}:{fid}:{stamp_ns}"
            scalar_payload = {
                "type": "scalars",
                "event_id": event_id,
                "cam_id": cam,
                "frame_id": int(fid),
                "stamp_ns": int(stamp_ns),
                "frame_w": int(frame_w),
                "frame_h": int(frame_h),
                "person_track_id": int(pid),
                "object_track_id": int(oid),
                "object_class_id": int(obj_meta.get("class_id", -1)),
                "hand_to_chest_L": hand_to_chest_L,
                "hand_to_chest_R": hand_to_chest_R,
                "hand_to_hip_L": hand_to_hip_L,
                "hand_to_hip_R": hand_to_hip_R,
                "wrist_v_L": v_l,
                "wrist_v_R": v_r,
                "wrist_a_L": a_l,
                "wrist_a_R": a_r,
                "torso_angle": ang,
                "torso_angle_rate": ang_rate,
                "guard_score": guard_score,
                "contact": bool(contact),
                "contact_duration_frames": int(contact_frames),
                "obj_area_px": obj_area,
                "obj_visibility": visibility,
                "disappeared_after_contact": bool(disappeared_after_contact),
                "person_speed_px_s": person_speed,
                "object_speed_px_s": object_speed,
                "comotion_cosine": comotion_cos,
                "comotion_speed_ratio": comotion_speed_ratio,
                "rel_offset_std_px": rel_offset_std_px,
                "carry_score": carry_score,
                "obj_best_candidate_id": int(best_oid),
                "obj_best_candidate_score": float(best_score) if best_score is not None else None,
                "obj_selected_score": float(sel_sc) if sel_sc is not None else None,
                "obj_switch_pending": bool(pending),
                "obj_switched": bool(switched),
                "obj_switch_countdown": int(countdown),
            }
            self._publish_scalars_frame(scalar_payload)
            self.scalar_window[pid].append(scalar_payload)

            self.person_prev[pid] = {
                "stamp_ns": stamp_ns,
                "lw": lw,
                "rw": rw,
                "v_l": v_l,
                "v_r": v_r,
                "ang": ang,
            }

            if len(self.clip_buf[pid]) == self.args.T:
                clip = np.stack(list(self.clip_buf[pid]), axis=1).astype(np.float32, copy=False)

                q_pose_miss = (
                    float(np.mean(np.array(list(self.miss_pose_win[pid]), dtype=np.float32)))
                    if len(self.miss_pose_win[pid])
                    else 1.0
                )
                q_obj_miss = (
                    float(np.mean(np.array(list(self.miss_obj_win[pid]), dtype=np.float32)))
                    if len(self.miss_obj_win[pid])
                    else 1.0
                )

                self._publish_clip(cam, pid, fid, stamp_ns, frame_w, frame_h, obj_meta, clip, q_pose_miss, q_obj_miss)

                win = list(self.scalar_window[pid])

                def col(name):
                    return [w.get(name, None) for w in win]

                agg = {}
                for nm in [
                    "hand_to_chest_L",
                    "hand_to_chest_R",
                    "hand_to_hip_L",
                    "hand_to_hip_R",
                    "wrist_v_L",
                    "wrist_v_R",
                    "wrist_a_L",
                    "wrist_a_R",
                    "torso_angle_rate",
                    "guard_score",
                    "person_speed_px_s",
                    "object_speed_px_s",
                    "comotion_cosine",
                    "comotion_speed_ratio",
                    "rel_offset_std_px",
                    "carry_score",
                    "obj_visibility",
                ]:
                    agg[nm] = agg_stats(col(nm))

                contact_vals = [1.0 if w.get("contact", False) else 0.0 for w in win]
                agg["contact_ratio"] = float(np.mean(contact_vals)) if contact_vals else 0.0
                agg["contact_duration_frames_last"] = int(win[-1].get("contact_duration_frames", 0)) if win else 0
                agg["contact_duration_frames_max"] = int(max([w.get("contact_duration_frames", 0) for w in win], default=0))

                vis_vals = _as_float_list(col("obj_visibility"))
                if len(vis_vals) >= 2:
                    agg["visibility_drop"] = float(vis_vals[0] - vis_vals[-1])
                    agg["visibility_min"] = float(np.min(np.array(vis_vals, dtype=np.float32)))
                else:
                    agg["visibility_drop"] = None
                    agg["visibility_min"] = None

                agg["disappeared_after_contact"] = bool(win[-1].get("disappeared_after_contact", False)) if win else False
                agg["missing_pose_ratio"] = q_pose_miss
                agg["missing_obj_ratio"] = q_obj_miss

                agg["clip_dt_s"] = self._compute_clip_dt_s(win)
                agg["obj_missing_frames_last"] = self._compute_obj_missing_frames(fid, oid, fallback=None)
                last_oid_for_pid = self.last_contact_obj.get(pid, None)
                agg["last_contact_obj_missing_frames"] = self._compute_obj_missing_frames(fid, last_oid_for_pid, fallback=None)

                if self.args.emit_heuristic_theft_score:
                    cr = float(agg.get("contact_ratio", 0.0) or 0.0)
                    vd = agg.get("visibility_drop", None)
                    vd = float(vd) if vd is not None else 0.0
                    cs = agg.get("carry_score", {}).get("max", None)
                    cs = float(cs) if cs is not None else 0.0

                    cr_n = clamp01(cr)
                    vd_n = clamp01(vd / 1.0)
                    cs_n = clamp01(cs)

                    base = 0.35 * cr_n + 0.35 * vd_n + 0.30 * cs_n
                    if agg.get("disappeared_after_contact", False):
                        base = max(base, 0.85)
                    agg["heuristic_theft_score"] = float(clamp01(base))

                self._publish_scalars_clip(cam, pid, fid, stamp_ns, obj_meta, agg, q_pose_miss, q_obj_miss)

        self._cleanup_person_state(fid)
        self._cleanup_object_state(fid)
        self._cleanup_old_clip_files()

    def close(self):
        self._cleanup_old_clip_files(force=True)

        try:
            if self.frame_scalars_pub is not None:
                self.frame_scalars_pub.close(0)
        except Exception:
            pass
        self.frame_scalars_pub = None

        try:
            if self.clip_pub is not None:
                self.clip_pub.close(0)
        except Exception:
            pass
        self.clip_pub = None

        try:
            if self.scalars_clip_pub is not None:
                self.scalars_clip_pub.close(0)
        except Exception:
            pass
        self.scalars_clip_pub = None


def make_join_key(payload: dict):
    cam = str(payload.get("cam_id", ""))
    fid = int(payload.get("frame_id", -1))
    stamp_ns = int(payload.get("stamp_ns", 0))
    return (cam, fid, stamp_ns)


def build_args_from_cli(parsed):
    publish_frame_scalars = bool(
        getattr(parsed, "publish_frame_scalars", False) or getattr(parsed, "publish_frame_scalars_redis", False)
    )

    return argparse.Namespace(
        T=parsed.T,
        clip_h=parsed.clip_h,
        clip_w=parsed.clip_w,
        sigma=parsed.sigma,
        kp_conf_thr=parsed.kp_conf_thr,
        contact_use_poly=parsed.contact_use_poly,
        contact_dist_px=parsed.contact_dist_px,
        vis_hist=parsed.vis_hist,
        comotion_window=parsed.comotion_window,
        obj_switch_margin_ratio=parsed.obj_switch_margin_ratio,
        obj_switch_confirm_frames=parsed.obj_switch_confirm_frames,
        disappear_miss_frames=parsed.disappear_miss_frames,
        disappear_contact_min_frames=parsed.disappear_contact_min_frames,
        disappear_window_frames=parsed.disappear_window_frames,
        person_state_ttl_frames=parsed.person_state_ttl_frames,
        object_state_ttl_frames=parsed.object_state_ttl_frames,
        emit_heuristic_theft_score=parsed.emit_heuristic_theft_score,
        max_join_buf=parsed.max_join_buf,
        clip_dir=parsed.clip_dir,
        clip_codec=parsed.clip_codec,
        clip_retention_s=parsed.clip_retention_s,
        clip_cleanup_interval_s=parsed.clip_cleanup_interval_s,
        publish_frame_scalars=publish_frame_scalars,
        publish_clip_events=parsed.publish_clip_events,
        publish_scalars_clip_events=parsed.publish_scalars_clip_events,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--clip_h", type=int, default=112)
    ap.add_argument("--clip_w", type=int, default=112)

    ap.add_argument("--join_timeout_s", type=float, default=1.5)
    ap.add_argument("--poll_ms", type=int, default=50)
    ap.add_argument("--max_join_buf", type=int, default=512)

    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--kp_conf_thr", type=float, default=0.3)
    ap.add_argument("--contact_use_poly", action="store_true")
    ap.add_argument("--contact_dist_px", type=float, default=25.0)

    ap.add_argument("--vis_hist", type=int, default=30)
    ap.add_argument("--comotion_window", type=int, default=10)

    ap.add_argument("--obj_switch_margin_ratio", type=float, default=0.80)
    ap.add_argument("--obj_switch_confirm_frames", type=int, default=3)

    ap.add_argument("--disappear_miss_frames", type=int, default=12)
    ap.add_argument("--disappear_contact_min_frames", type=int, default=5)
    ap.add_argument("--disappear_window_frames", type=int, default=40)

    ap.add_argument("--person_state_ttl_frames", type=int, default=120)
    ap.add_argument("--object_state_ttl_frames", type=int, default=240)

    ap.add_argument("--clip_dir", default="/tmp/zono_clips")
    ap.add_argument("--clip_codec", choices=["npz", "npy"], default="npz")
    ap.add_argument("--clip_retention_s", type=int, default=1800)
    ap.add_argument("--clip_cleanup_interval_s", type=float, default=60.0)

    ap.add_argument("--publish_frame_scalars", action="store_true")
    ap.add_argument("--publish_frame_scalars_redis", action="store_true")  # backward-compatible alias
    ap.add_argument("--publish_clip_events", action="store_true")
    ap.add_argument("--publish_scalars_clip_events", action="store_true")

    ap.add_argument("--emit_heuristic_theft_score", action="store_true")
    args = ap.parse_args()

    if not args.publish_frame_scalars and not args.publish_frame_scalars_redis:
        args.publish_frame_scalars = True

    if not args.publish_clip_events:
        args.publish_clip_events = True
    if not args.publish_scalars_clip_events:
        args.publish_scalars_clip_events = True

    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    pose_cfg = get_zmq_endpoint(cfg, cam_id, "pose_features")
    seg_cfg = get_zmq_endpoint(cfg, cam_id, "seg_features")

    ctx = zmq.Context.instance()
    pose_connect = local_connect_addr(pose_cfg["bind"])
    pose_topic = pose_cfg["topic"]
    seg_connect = local_connect_addr(seg_cfg["bind"])
    seg_topic = seg_cfg["topic"]

    sub_pose = ctx.socket(zmq.SUB)
    sub_pose.setsockopt(zmq.LINGER, 0)
    sub_pose.setsockopt(zmq.RCVHWM, 1000)
    sub_pose.connect(pose_connect)
    sub_pose.setsockopt(zmq.SUBSCRIBE, pose_topic.encode("utf-8"))

    sub_seg = ctx.socket(zmq.SUB)
    sub_seg.setsockopt(zmq.LINGER, 0)
    sub_seg.setsockopt(zmq.RCVHWM, 1000)
    sub_seg.connect(seg_connect)
    sub_seg.setsockopt(zmq.SUBSCRIBE, seg_topic.encode("utf-8"))

    poller = zmq.Poller()
    poller.register(sub_pose, zmq.POLLIN)
    poller.register(sub_seg, zmq.POLLIN)

    fb = FeatureBuilder(
        cfg_path=args.config,
        cam_id=cam_id,
        args=build_args_from_cli(args),
        debug=False,
    )

    pose_buf = {}
    seg_buf = {}
    join_seen_ts = {}

    print(f"[feature_builder] SUB pose {pose_connect} topic={pose_topic}")
    print(f"[feature_builder] SUB seg  {seg_connect} topic={seg_topic}")

    try:
        while True:
            events = dict(poller.poll(timeout=args.poll_ms))

            if sub_pose in events:
                parts = sub_pose.recv_multipart()
                if len(parts) == 3:
                    _, _, payload_b = parts
                    payload = json.loads(payload_b.decode("utf-8"))
                    k = make_join_key(payload)
                    pose_buf[k] = payload
                    join_seen_ts.setdefault(k, time.time())

            if sub_seg in events:
                parts = sub_seg.recv_multipart()
                if len(parts) == 3:
                    _, _, payload_b = parts
                    payload = json.loads(payload_b.decode("utf-8"))
                    k = make_join_key(payload)
                    seg_buf[k] = payload
                    join_seen_ts.setdefault(k, time.time())

            if len(pose_buf) > args.max_join_buf:
                old_keys = sorted(pose_buf.keys(), key=lambda x: (x[1], x[2]))[: len(pose_buf) - args.max_join_buf]
                for k in old_keys:
                    pose_buf.pop(k, None)
                    join_seen_ts.pop(k, None)

            if len(seg_buf) > args.max_join_buf:
                old_keys = sorted(seg_buf.keys(), key=lambda x: (x[1], x[2]))[: len(seg_buf) - args.max_join_buf]
                for k in old_keys:
                    seg_buf.pop(k, None)
                    join_seen_ts.pop(k, None)

            common = pose_buf.keys() & seg_buf.keys()
            for k in sorted(common, key=lambda x: (x[1], x[2])):
                pose = pose_buf.pop(k)
                seg = seg_buf.pop(k)
                join_seen_ts.pop(k, None)
                fb.process_pair(pose, seg)

            now = time.time()
            stale_keys = []
            for k, ts in join_seen_ts.items():
                if (now - ts) > args.join_timeout_s:
                    stale_keys.append(k)

            for k in stale_keys:
                pose_buf.pop(k, None)
                seg_buf.pop(k, None)
                join_seen_ts.pop(k, None)

    except KeyboardInterrupt:
        print("\n[feature_builder] stopping...")
    finally:
        sub_pose.close(0)
        sub_seg.close(0)
        fb.close()


if __name__ == "__main__":
    main()
