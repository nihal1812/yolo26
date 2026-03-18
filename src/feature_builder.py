#!/usr/bin/env python3
import time
import json
import argparse
from collections import defaultdict, deque

import zmq
import redis
import numpy as np
import cv2

from config_utils import (
    load_cfg,
    get_zmq_endpoint,
    get_stream,
    local_connect_addr,
)


# ------------------------
# Geometry helpers
# ------------------------
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


# ------------------------
# Pose rasterization (heatmaps)
# ------------------------
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


# COCO-17 indices
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


# ------------------------
# Contact: point-in-polygon
# ------------------------
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


# ------------------------
# Object-of-interest scoring + hysteresis selection
# ------------------------
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


def stabilize_object_selection(
    pid,
    best_oid,
    best_score,
    last_oid,
    last_score,
    switch_state,
    margin_ratio=0.80,
    confirm_frames=3,
):
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


# ------------------------
# Co-motion helpers
# ------------------------
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


# ------------------------
# Scalar aggregation (clip-level)
# ------------------------
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


def make_join_key(payload: dict):
    cam = str(payload.get("cam_id", ""))
    fid = int(payload.get("frame_id", -1))
    stamp_ns = int(payload.get("stamp_ns", 0))
    return (cam, fid, stamp_ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--clip_h", type=int, default=112)
    ap.add_argument("--clip_w", type=int, default=112)

    ap.add_argument("--join_timeout_frames", type=int, default=10)
    ap.add_argument("--join_timeout_s", type=float, default=1.5)
    ap.add_argument("--poll_ms", type=int, default=50)

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

    ap.add_argument("--redis_host", default=None)
    ap.add_argument("--redis_port", type=int, default=None)
    ap.add_argument("--redis_db", type=int, default=None)
    ap.add_argument("--redis_pass", default=None)
    ap.add_argument("--clips_stream", default=None)
    ap.add_argument("--scalars_stream", default=None)
    ap.add_argument("--scalars_clip_stream", default=None)
    ap.add_argument("--redis_maxlen", type=int, default=20000)

    ap.add_argument("--emit_heuristic_theft_score", action="store_true")

    args = ap.parse_args()
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
    sub_pose.setsockopt(zmq.RCVHWM, 1000)
    sub_pose.connect(pose_connect)
    sub_pose.setsockopt(zmq.SUBSCRIBE, pose_topic.encode("utf-8"))

    sub_seg = ctx.socket(zmq.SUB)
    sub_seg.setsockopt(zmq.RCVHWM, 1000)
    sub_seg.connect(seg_connect)
    sub_seg.setsockopt(zmq.SUBSCRIBE, seg_topic.encode("utf-8"))

    poller = zmq.Poller()
    poller.register(sub_pose, zmq.POLLIN)
    poller.register(sub_seg, zmq.POLLIN)

    r_cfg = cfg.get("redis", {})
    r_host = args.redis_host or r_cfg.get("host", "127.0.0.1")
    r_port = args.redis_port if args.redis_port is not None else int(r_cfg.get("port", 6379))
    r_db = args.redis_db if args.redis_db is not None else int(r_cfg.get("db", 0))
    r_pass = args.redis_pass if args.redis_pass is not None else r_cfg.get("password", None)

    rdb = redis.Redis(host=r_host, port=r_port, db=r_db, password=r_pass)
    rdb.ping()

    clips_stream = args.clips_stream or get_stream(cfg, "clips", cam_id)
    scalars_stream = args.scalars_stream or get_stream(cfg, "scalars", cam_id)
    scalars_clip_stream = args.scalars_clip_stream or get_stream(cfg, "scalars_clip", cam_id)

    out_hw = (args.clip_h, args.clip_w)

    pose_buf = {}
    seg_buf = {}
    join_seen_ts = {}

    clip_buf = defaultdict(lambda: deque(maxlen=args.T))
    scalar_window = defaultdict(lambda: deque(maxlen=args.T))
    miss_pose_win = defaultdict(lambda: deque(maxlen=args.T))
    miss_obj_win = defaultdict(lambda: deque(maxlen=args.T))

    person_prev = {}
    person_last_seen = {}

    contact_count = defaultdict(int)
    last_contact_obj = {}
    last_contact_frame = {}
    last_contact_dur = {}

    obj_area_hist = defaultdict(lambda: deque(maxlen=args.vis_hist))
    obj_last_seen = {}
    comotion_hist = defaultdict(lambda: deque(maxlen=args.comotion_window))

    last_selected_obj = {}
    last_selected_score = {}
    switch_state = {}

    print(f"[feature_builder] cam_id={cam_id}")
    print(f"[feature_builder] SUB pose {pose_connect} topic={pose_topic}")
    print(f"[feature_builder] SUB seg  {seg_connect} topic={seg_topic}")
    print(f"[feature_builder] Redis clips={clips_stream} scalars={scalars_stream} scalars_clip={scalars_clip_stream}")
    print(f"[feature_builder] clip: C=19 T={args.T} HxW={out_hw}")
    if args.emit_heuristic_theft_score:
        print("[feature_builder] emitting heuristic_theft_score in scalars_clip.features")

    def publish_clip(cam, pid, fid, stamp_ns, frame_w, frame_h, obj_meta, clip_arr, q_pose_miss, q_obj_miss):
        event_id = f"{cam}:{pid}:{fid}:{stamp_ns}"
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
        }
        rdb.xadd(
            clips_stream,
            {
                "event_id": event_id,
                "cam_id": cam,
                "person_track_id": str(pid),
                "frame_id": str(fid),
                "stamp_ns": str(stamp_ns),
                "meta": json.dumps(meta),
                "clip": clip_arr.tobytes(),
            },
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def publish_scalars_frame(payload_dict):
        rdb.xadd(
            scalars_stream,
            {
                "event_id": payload_dict["event_id"],
                "cam_id": payload_dict["cam_id"],
                "person_track_id": str(payload_dict.get("person_track_id", -1)),
                "frame_id": str(payload_dict["frame_id"]),
                "stamp_ns": str(payload_dict["stamp_ns"]),
                "json": json.dumps(payload_dict),
            },
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def publish_scalars_clip(cam, pid, fid, stamp_ns, obj_meta, agg_payload, q_pose_miss, q_obj_miss):
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
            "T": int(args.T),
        }
        out = {
            "meta": meta,
            "features": agg_payload,
            "policy_features": policy_features,
        }
        rdb.xadd(
            scalars_clip_stream,
            {
                "event_id": event_id,
                "cam_id": cam,
                "person_track_id": str(pid),
                "frame_id": str(fid),
                "stamp_ns": str(stamp_ns),
                "json": json.dumps(out),
            },
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def compute_clip_dt_s(win):
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

    def compute_obj_missing_frames(fid_now, oid, fallback=None):
        if oid is None or oid < 0:
            return fallback
        last_seen = obj_last_seen.get(int(oid), None)
        if last_seen is None:
            return fallback
        try:
            return int(fid_now - int(last_seen))
        except Exception:
            return fallback

    def cleanup_person_state(fid_now: int):
        stale_pids = []
        for pid, last_fid in person_last_seen.items():
            try:
                if (fid_now - int(last_fid)) > args.person_state_ttl_frames:
                    stale_pids.append(pid)
            except Exception:
                stale_pids.append(pid)

        for pid in stale_pids:
            clip_buf.pop(pid, None)
            scalar_window.pop(pid, None)
            miss_pose_win.pop(pid, None)
            miss_obj_win.pop(pid, None)
            person_prev.pop(pid, None)
            last_contact_obj.pop(pid, None)
            last_contact_frame.pop(pid, None)
            last_contact_dur.pop(pid, None)
            last_selected_obj.pop(pid, None)
            last_selected_score.pop(pid, None)
            switch_state.pop(pid, None)
            person_last_seen.pop(pid, None)

            for key in list(contact_count.keys()):
                if key[0] == pid:
                    contact_count.pop(key, None)

            for key in list(comotion_hist.keys()):
                if key[0] == pid:
                    comotion_hist.pop(key, None)

    def cleanup_object_state(fid_now: int):
        stale_oids = []
        for oid, last_fid in obj_last_seen.items():
            try:
                if (fid_now - int(last_fid)) > args.object_state_ttl_frames:
                    stale_oids.append(oid)
            except Exception:
                stale_oids.append(oid)

        for oid in stale_oids:
            obj_last_seen.pop(oid, None)
            obj_area_hist.pop(oid, None)

        for key in list(contact_count.keys()):
            _pid, oid = key
            if oid not in obj_last_seen:
                contact_count.pop(key, None)

        for key in list(comotion_hist.keys()):
            _pid, oid = key
            if oid not in obj_last_seen:
                comotion_hist.pop(key, None)

    def cleanup_join_buffers():
        now = time.time()
        keys = set(pose_buf.keys()) | set(seg_buf.keys())
        for k in list(keys):
            first_seen = join_seen_ts.get(k, now)
            pose_payload = pose_buf.get(k)
            seg_payload = seg_buf.get(k)

            fid = None
            if pose_payload is not None:
                fid = int(pose_payload.get("frame_id", -1))
            elif seg_payload is not None:
                fid = int(seg_payload.get("frame_id", -1))

            too_old_by_time = (now - first_seen) > args.join_timeout_s
            too_old_by_frames = False
            if fid is not None:
                base_fid = fid
                too_old_by_frames = bool(False)

            if too_old_by_time or too_old_by_frames:
                pose_buf.pop(k, None)
                seg_buf.pop(k, None)
                join_seen_ts.pop(k, None)

    try:
        while True:
            events = dict(poller.poll(timeout=args.poll_ms))

            if sub_pose in events:
                _, _, payload_b = sub_pose.recv_multipart()
                payload = json.loads(payload_b.decode("utf-8"))
                k = make_join_key(payload)
                pose_buf[k] = payload
                join_seen_ts.setdefault(k, time.time())

            if sub_seg in events:
                _, _, payload_b = sub_seg.recv_multipart()
                payload = json.loads(payload_b.decode("utf-8"))
                k = make_join_key(payload)
                seg_buf[k] = payload
                join_seen_ts.setdefault(k, time.time())

            common = set(pose_buf.keys()) & set(seg_buf.keys())
            latest_common_fid = None

            for k in sorted(common, key=lambda x: (x[1], x[2])):
                pose = pose_buf.pop(k)
                seg = seg_buf.pop(k)
                join_seen_ts.pop(k, None)

                cam = pose.get("cam_id", cam_id)
                fid = int(pose.get("frame_id", 0))
                latest_common_fid = fid if latest_common_fid is None else max(latest_common_fid, fid)

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
                    continue

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
                        obj_last_seen[oid] = fid
                        area = float(instances[i].get("mask_area_px", 0.0))
                        if area > 0:
                            obj_area_hist[oid].append(area)

                for person in people:
                    pid = int(person.get("track_id", -1))
                    if pid < 0:
                        continue
                    pb = person.get("bbox_xyxy", None)
                    if not pb:
                        continue

                    person_last_seen[pid] = fid

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
                        miss_pose = float(np.mean(cf < float(args.kp_conf_thr)))
                    miss_pose_win[pid].append(miss_pose)

                    best_iou = 0.0
                    best_person_inst_i = None
                    for i in person_idxs:
                        val = iou_xyxy(pb, instances[i]["bbox_xyxy"])
                        if val > best_iou:
                            best_iou = val
                            best_person_inst_i = i
                    person_full = (
                        full_masks[best_person_inst_i]
                        if best_person_inst_i is not None
                        else np.zeros((frame_h, frame_w), np.uint8)
                    )

                    obj_instances = [instances[i] for i in obj_idxs]
                    best_idx, best_score = pick_best_object(pb, wrists, obj_instances)
                    best_oid = int(obj_instances[best_idx].get("track_id", -1)) if best_idx is not None else -1

                    last_oid = last_selected_obj.get(pid, None)
                    last_sc = last_selected_score.get(pid, None)
                    sel_oid, sel_sc, pending, switched, countdown = stabilize_object_selection(
                        pid=pid,
                        best_oid=best_oid,
                        best_score=best_score,
                        last_oid=last_oid,
                        last_score=last_sc,
                        switch_state=switch_state,
                        margin_ratio=args.obj_switch_margin_ratio,
                        confirm_frames=args.obj_switch_confirm_frames,
                    )
                    last_selected_obj[pid] = sel_oid if sel_oid is not None else -1
                    last_selected_score[pid] = sel_sc

                    if switched:
                        for key in list(comotion_hist.keys()):
                            if key[0] == pid:
                                comotion_hist.pop(key, None)

                    obj_meta = {"track_id": -1, "class_id": -1, "bbox_xyxy": None, "mask_poly_xy": None}
                    obj_full = np.zeros((frame_h, frame_w), np.uint8)

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
                    miss_obj_win[pid].append(miss_obj)

                    p_mask = crop_resize(person_full, pb, out_hw).astype(np.float32)
                    o_mask = crop_resize(obj_full, pb, out_hw).astype(np.float32)
                    hm = joint_heatmaps(
                        kp_xy, kp_cf, pb, out_hw, sigma=args.sigma, conf_thr=args.kp_conf_thr
                    )
                    frame_maps = np.concatenate([p_mask[None, :, :], o_mask[None, :, :], hm], axis=0).astype(np.float32)
                    clip_buf[pid].append(frame_maps)

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

                    prev = person_prev.get(pid, None)
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
                        if args.contact_use_poly and obj_meta.get("mask_poly_xy"):
                            if lw and point_in_poly(lw, obj_meta["mask_poly_xy"]):
                                contact = True
                            if rw and point_in_poly(rw, obj_meta["mask_poly_xy"]):
                                contact = True
                        else:
                            oc = bbox_center(obj_meta["bbox_xyxy"])
                            thr2 = args.contact_dist_px ** 2
                            if lw and dist2(lw, oc) < thr2:
                                contact = True
                            if rw and dist2(rw, oc) < thr2:
                                contact = True

                    if oid >= 0 and contact:
                        contact_count[(pid, oid)] += 1
                        last_contact_obj[pid] = oid
                        last_contact_frame[pid] = fid
                        last_contact_dur[pid] = int(contact_count[(pid, oid)])
                    else:
                        if oid >= 0:
                            contact_count[(pid, oid)] = 0
                    contact_frames = contact_count.get((pid, oid), 0) if oid >= 0 else 0

                    visibility = None
                    obj_area = None
                    if oid >= 0:
                        inst = obj_by_id.get(oid, None)
                        if inst:
                            obj_area = float(inst.get("mask_area_px", 0.0))
                        hist = obj_area_hist.get(oid, None)
                        if hist and len(hist) >= 5:
                            med = float(np.median(np.array(hist, dtype=np.float32)))
                            if med > 1e-6 and obj_area is not None:
                                visibility = float(obj_area / med)

                    disappeared_after_contact = False
                    last_oid2 = last_contact_obj.get(pid, None)
                    last_cf = last_contact_frame.get(pid, None)
                    last_cd = int(last_contact_dur.get(pid, 0))

                    if last_oid2 is not None and last_oid2 >= 0 and last_cf is not None:
                        last_seen = obj_last_seen.get(last_oid2, None)
                        if last_seen is not None:
                            missing = fid - last_seen
                            since_contact = fid - last_cf
                            if (
                                last_cd >= args.disappear_contact_min_frames
                                and since_contact <= args.disappear_window_frames
                                and missing >= args.disappear_miss_frames
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
                        comotion_hist[key].append((t_s, pcx, pcy, ocx, ocy))

                        hist = list(comotion_hist[key])
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
                    publish_scalars_frame(scalar_payload)
                    scalar_window[pid].append(scalar_payload)

                    person_prev[pid] = {
                        "stamp_ns": stamp_ns,
                        "lw": lw,
                        "rw": rw,
                        "v_l": v_l,
                        "v_r": v_r,
                        "ang": ang,
                    }

                    if len(clip_buf[pid]) == args.T:
                        clip = np.stack(list(clip_buf[pid]), axis=1)

                        q_pose_miss = (
                            float(np.mean(np.array(list(miss_pose_win[pid]), dtype=np.float32)))
                            if len(miss_pose_win[pid])
                            else 1.0
                        )
                        q_obj_miss = (
                            float(np.mean(np.array(list(miss_obj_win[pid]), dtype=np.float32)))
                            if len(miss_obj_win[pid])
                            else 1.0
                        )

                        publish_clip(cam, pid, fid, stamp_ns, frame_w, frame_h, obj_meta, clip, q_pose_miss, q_obj_miss)

                        win = list(scalar_window[pid])

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

                        agg["clip_dt_s"] = compute_clip_dt_s(win)
                        agg["obj_missing_frames_last"] = compute_obj_missing_frames(fid, oid, fallback=None)
                        last_oid_for_pid = last_contact_obj.get(pid, None)
                        agg["last_contact_obj_missing_frames"] = compute_obj_missing_frames(fid, last_oid_for_pid, fallback=None)

                        if args.emit_heuristic_theft_score:
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

                        publish_scalars_clip(cam, pid, fid, stamp_ns, obj_meta, agg, q_pose_miss, q_obj_miss)

            if latest_common_fid is not None:
                cleanup_person_state(latest_common_fid)
                cleanup_object_state(latest_common_fid)

            cleanup_join_buffers()

    except KeyboardInterrupt:
        print("\n[feature_builder] stopping...")
    finally:
        sub_pose.close()
        sub_seg.close()


if __name__ == "__main__":
    main()