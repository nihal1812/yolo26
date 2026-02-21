#!/usr/bin/env python3
"""
data_collector.py

Collects training/eval data without blocking realtime:

Consumes:
  - scores:<cam>          (model_node)
  - decisions:<cam>       (policy_node)
  - alerts:<cam>          (policy_node)
  - scalars_clip:<cam>    (feature_builder)
  - feedback:<cam>        (UI)

Optionally:
  - clips:<cam>           (pointer/meta only; clip bytes are not stored)

Produces:
  - train_events:<cam>    (unified fused event snapshots)

Key improvements vs your draft:
  1) Publish gating (avoid spamming train_events):
     - publish when "ready": score+scalars present OR feedback attached
  2) Pending feedback matching survives restart:
     - store pending alert refs in Redis ZSET per person
  3) Stable "stage" field:
     - partial / ready / labeled
"""

import argparse
import json
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import yaml
import redis


# ----------------------------
# Utils
# ----------------------------
def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def safe_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def now_ns():
    return time.time_ns()


def parse_xread(streams):
    out = []
    for sname, msgs in streams:
        s = b2s(sname)
        for mid, fields in msgs:
            out.append((s, b2s(mid), fields))
    return out


# ----------------------------
# TTL cache (in-memory fuse)
# ----------------------------
class TTLCache:
    """
    key -> (t_s, value)
    Prunes by TTL and max_items
    """
    def __init__(self, ttl_s: float, max_items: int = 20000):
        self.ttl_s = float(ttl_s)
        self.max_items = int(max_items)
        self.store = {}
        self.fifo = deque(maxlen=self.max_items)

    def put(self, key, value):
        t_s = time.time()
        self.store[key] = (t_s, value)
        self.fifo.append((t_s, key))
        self.prune()

    def get(self, key):
        item = self.store.get(key, None)
        if not item:
            return None
        t_s, val = item
        if (time.time() - t_s) > self.ttl_s:
            self.store.pop(key, None)
            return None
        return val

    def prune(self):
        now = time.time()
        while self.fifo:
            t_s, k = self.fifo[0]
            if (now - t_s) <= self.ttl_s and len(self.store) <= self.max_items:
                break
            self.fifo.popleft()
            self.store.pop(k, None)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=200)

    # Stream names overrides
    ap.add_argument("--scores_stream", default=None)
    ap.add_argument("--decisions_stream", default=None)
    ap.add_argument("--alerts_stream", default=None)
    ap.add_argument("--scalars_clip_stream", default=None)
    ap.add_argument("--feedback_stream", default=None)

    ap.add_argument("--clips_stream", default=None)
    ap.add_argument("--read_clips_stream", action="store_true")

    ap.add_argument("--train_events_stream", default=None)
    ap.add_argument("--redis_maxlen", type=int, default=50000)

    # Correlation
    ap.add_argument("--cache_ttl_s", type=float, default=30.0)

    # Pending feedback matching (persisted in Redis)
    ap.add_argument("--pending_ttl_s", type=float, default=3600.0)

    # Publishing policy
    ap.add_argument("--publish_partial", action="store_true",
                    help="If set, publish every partial update (debug mode). Default: only ready/labeled.")

    args = ap.parse_args()
    cfg = load_cfg(args.config)
    cam_id = cfg["system"]["cam_id"]

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
        decode_responses=False,
    )
    rdb.ping()

    scores_stream = args.scores_stream or r_cfg.get("scores_stream", f"scores:{cam_id}")
    decisions_stream = args.decisions_stream or r_cfg.get("decisions_stream", f"decisions:{cam_id}")
    alerts_stream = args.alerts_stream or r_cfg.get("alerts_stream", f"alerts:{cam_id}")
    scalars_clip_stream = args.scalars_clip_stream or r_cfg.get("scalars_clip_stream", f"scalars_clip:{cam_id}")
    feedback_stream = args.feedback_stream or r_cfg.get("feedback_stream", f"feedback:{cam_id}")

    clips_stream = args.clips_stream or r_cfg.get("clips_stream", f"clips:{cam_id}")
    train_events_stream = args.train_events_stream or r_cfg.get("train_events_stream", f"train_events:{cam_id}")

    # Redis keys for pending alert refs (persisted)
    # ZSET score = unix time seconds, member = json(ref)
    pending_zset_prefix = r_cfg.get("pending_zset_prefix", "train:pending")
    # Example: train:pending:cam0:pid:5
    def pending_key(cam, pid):
        return f"{pending_zset_prefix}:{cam}:pid:{pid}"

    print("[data_collector] consuming:")
    print(f"  scores        = {scores_stream}")
    print(f"  decisions     = {decisions_stream}")
    print(f"  alerts        = {alerts_stream}")
    print(f"  scalars_clip  = {scalars_clip_stream}")
    print(f"  feedback      = {feedback_stream}")
    if args.read_clips_stream:
        print(f"  clips         = {clips_stream} (READING ENABLED)")
    else:
        print(f"  clips         = {clips_stream} (not reading clip bytes; pointers only)")

    print(f"[data_collector] producing train_events = {train_events_stream}")
    print(f"[data_collector] publish_partial={args.publish_partial}")
    print(f"[data_collector] pending_ttl_s={args.pending_ttl_s}")

    # XREAD cursors
    last_ids = {
        scores_stream: "0-0",
        decisions_stream: "0-0",
        alerts_stream: "0-0",
        scalars_clip_stream: "0-0",
        feedback_stream: "0-0",
    }
    if args.read_clips_stream:
        last_ids[clips_stream] = "0-0"

    # Correlation cache keyed by (cam, pid, fid_end)
    cache = TTLCache(ttl_s=args.cache_ttl_s, max_items=50000)

    def fuse_key(cam, pid, fid_end) -> Tuple[str, int, int]:
        return (str(cam), int(pid), int(fid_end))

    def event_stage(ev: dict) -> str:
        """
        partial: missing required things
        ready: has score + scalars (good for training even without feedback)
        labeled: has feedback attached
        """
        if ev.get("feedback") is not None:
            return "labeled"
        payload = ev.get("payload", {})
        has_score = isinstance(payload.get("score", None), dict)
        has_scal = isinstance(payload.get("scalars_clip", None), dict)
        if has_score and has_scal:
            return "ready"
        return "partial"

    def publish_train_event(ev: dict):
        ev["stage"] = event_stage(ev)
        # Default gating: publish only ready/labeled (unless debug)
        if (not args.publish_partial) and ev["stage"] == "partial":
            return

        rdb.xadd(
            train_events_stream,
            {
                "cam_id": ev.get("cam_id", cam_id),
                "person_track_id": str(ev.get("person_track_id", -1)),
                "frame_id_end": str(ev.get("frame_id_end", -1)),
                "stamp_ns_end": str(ev.get("stamp_ns_end", 0)),
                "stage": ev["stage"],
                "json": json.dumps(ev),
            },
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def cache_merge(key, patch: dict) -> dict:
        base = cache.get(key) or {
            "type": "train_event",
            "cam_id": key[0],
            "person_track_id": key[1],
            "frame_id_end": key[2],
            "stamp_ns_end": 0,
            "refs": {},
            "payload": {},
            "feedback": None,
            "created_ns": now_ns(),
            "updated_ns": now_ns(),
        }

        base["updated_ns"] = now_ns()

        # merge refs/payload
        if "refs" in patch and isinstance(patch["refs"], dict):
            base["refs"].update(patch["refs"])
        if "payload" in patch and isinstance(patch["payload"], dict):
            base["payload"].update(patch["payload"])

        # stamp
        if patch.get("stamp_ns_end"):
            base["stamp_ns_end"] = int(patch["stamp_ns_end"])

        # feedback
        if patch.get("feedback") is not None:
            base["feedback"] = patch["feedback"]

        cache.put(key, base)
        return base

    # ----------------------------
    # Pending alert/decision helpers (Redis ZSET)
    # ----------------------------
    def pending_prune(cam, pid):
        k = pending_key(cam, pid)
        cutoff = time.time() - float(args.pending_ttl_s)
        # remove older-than cutoff
        rdb.zremrangebyscore(k, 0, cutoff)

    def add_pending(cam, pid, ref_obj: dict):
        pending_prune(cam, pid)
        k = pending_key(cam, pid)
        t = time.time()
        rdb.zadd(k, {json.dumps(ref_obj): t})
        # keep only last N items (avoid growth)
        rdb.zremrangebyrank(k, 0, -51)  # keep ~50 newest

    def match_pending_latest(cam, pid) -> Optional[dict]:
        pending_prune(cam, pid)
        k = pending_key(cam, pid)
        items = rdb.zrevrange(k, 0, 0)
        if not items:
            return None
        try:
            return json.loads(b2s(items[0]))
        except Exception:
            return None

    # ----------------------------
    # Main loop
    # ----------------------------
    try:
        while True:
            streams = rdb.xread(last_ids, block=args.block_ms, count=args.count)
            if not streams:
                cache.prune()
                continue

            for sname, mid, fields in parse_xread(streams):
                last_ids[sname] = mid

                obj = None
                if b"json" in fields:
                    try:
                        obj = json.loads(b2s(fields[b"json"]))
                    except Exception:
                        obj = None

                # --- scores
                if sname == scores_stream and obj:
                    cam = obj.get("cam_id", cam_id)
                    pid = safe_int(obj.get("person_track_id", -1), -1)
                    fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(obj.get("stamp_ns_end", 0), 0),
                        "refs": {"score_msg_id": mid},
                        "payload": {"score": obj},
                    })
                    publish_train_event(ev)

                # --- decisions
                elif sname == decisions_stream and obj:
                    cam = obj.get("cam_id", cam_id)
                    pid = safe_int(obj.get("person_track_id", -1), -1)
                    fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(obj.get("stamp_ns_end", 0), 0),
                        "refs": {"decision_msg_id": mid},
                        "payload": {"decision": obj},
                    })

                    # if it would alert, add pending ref for feedback matching
                    if bool(obj.get("will_alert", False)):
                        add_pending(cam, pid, {
                            "source": "decision",
                            "decision_msg_id": mid,
                            "cam_id": cam,
                            "person_track_id": pid,
                            "frame_id_end": fid_end,
                        })

                    publish_train_event(ev)

                # --- alerts
                elif sname == alerts_stream and obj:
                    cam = obj.get("cam_id", cam_id)
                    pid = safe_int(obj.get("person_track_id", -1), -1)
                    fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(obj.get("stamp_ns_end", 0), 0),
                        "refs": {"alert_msg_id": mid},
                        "payload": {"alert": obj},
                    })

                    add_pending(cam, pid, {
                        "source": "alert",
                        "alert_msg_id": mid,
                        "cam_id": cam,
                        "person_track_id": pid,
                        "frame_id_end": fid_end,
                    })

                    publish_train_event(ev)

                # --- scalars_clip
                elif sname == scalars_clip_stream and obj:
                    meta = obj.get("meta", {})
                    feats = obj.get("features", {})
                    cam = meta.get("cam_id", cam_id)
                    pid = safe_int(meta.get("person_track_id", -1), -1)
                    fid_end = safe_int(meta.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(meta.get("stamp_ns_end", 0), 0),
                        "refs": {"scalars_clip_msg_id": mid},
                        "payload": {"scalars_clip": {"meta": meta, "features": feats}},
                    })
                    publish_train_event(ev)

                # --- optional: clips stream meta only
                elif args.read_clips_stream and (sname == clips_stream):
                    meta_js = fields.get(b"meta", b"{}")
                    try:
                        meta = json.loads(b2s(meta_js))
                    except Exception:
                        continue

                    cam = meta.get("cam_id", cam_id)
                    pid = safe_int(meta.get("person_track_id", -1), -1)
                    fid_end = safe_int(meta.get("frame_id_end", -1), -1)
                    if pid < 0 or fid_end < 0:
                        continue

                    key = fuse_key(cam, pid, fid_end)
                    ev = cache_merge(key, {
                        "stamp_ns_end": safe_int(meta.get("stamp_ns_end", 0), 0),
                        "refs": {"clip_msg_id": mid},
                        "payload": {"clip_meta": meta},
                    })
                    publish_train_event(ev)

                # --- feedback
                elif sname == feedback_stream and obj:
                    cam = obj.get("cam_id", cam_id)
                    pid = safe_int(obj.get("person_track_id", -1), -1)
                    if pid < 0:
                        continue

                    label_i = safe_int(obj.get("label", None), None)
                    if label_i is None:
                        continue

                    fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                    alert_id = obj.get("alert_id", None)  # optional

                    matched_key = None
                    if fid_end is not None and fid_end >= 0:
                        matched_key = fuse_key(cam, pid, fid_end)
                    else:
                        pend = match_pending_latest(cam, pid)
                        if pend:
                            matched_key = fuse_key(
                                pend.get("cam_id", cam),
                                pend.get("person_track_id", pid),
                                pend.get("frame_id_end", -1),
                            )

                    feedback_obj = {
                        "msg_id": mid,
                        "label": int(label_i),
                        "category": obj.get("category", None),
                        "comment": obj.get("comment", None),
                        "user_id": obj.get("user_id", None),
                        "alert_id": alert_id,
                        "received_ns": now_ns(),
                    }

                    if matched_key is not None and matched_key[2] >= 0:
                        ev = cache_merge(matched_key, {
                            "refs": {"feedback_msg_id": mid},
                            "feedback": feedback_obj,
                        })
                        publish_train_event(ev)
                    else:
                        orphan = {
                            "type": "train_event",
                            "cam_id": cam,
                            "person_track_id": pid,
                            "frame_id_end": -1,
                            "stamp_ns_end": 0,
                            "refs": {"feedback_msg_id": mid},
                            "payload": {"feedback_only": obj},
                            "feedback": feedback_obj,
                            "created_ns": now_ns(),
                            "updated_ns": now_ns(),
                            "stage": "labeled",
                            "note": "unmatched_feedback",
                        }
                        publish_train_event(orphan)

            cache.prune()

    except KeyboardInterrupt:
        print("\n[data_collector] stopping...")


if __name__ == "__main__":
    main()
