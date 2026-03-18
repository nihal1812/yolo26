#!/usr/bin/env python3
"""
identity_enricher.py

Purpose:
- bridge cross-camera ReID into the rest of the pipeline
- attach global_person_id to local per-camera events

Consumes:
  - global_tracks
  - scores:{cam}
  - decisions:{cam}
  - alerts:{cam}
  - optionally clip_refs:{cam}

Produces:
  - scores_enriched:{cam}
  - decisions_enriched:{cam}
  - alerts_enriched:{cam}
  - optionally clip_refs_enriched:{cam}

How it works:
- caches mappings from (cam_id, person_track_id) -> global_person_id
- when local events arrive, enriches them with global_person_id if known
- if mapping is not yet known, it can briefly defer the event and retry

Recommended config additions:
redis:
  streams:
    reid_embeddings: "reid_embeddings:{cam}"
    global_tracks: "global_tracks"
    scores_enriched: "scores_enriched:{cam}"
    decisions_enriched: "decisions_enriched:{cam}"
    alerts_enriched: "alerts_enriched:{cam}"
    clip_refs_enriched: "clip_refs_enriched:{cam}"
"""

import argparse
import json
import time
from collections import deque
from typing import Dict, Any, Tuple, Optional, List

import redis

from config_utils import load_cfg, get_active_cams, get_stream


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def parse_xread(streams):
    out = []
    for sname, msgs in streams:
        s = b2s(sname)
        for mid, fields in msgs:
            out.append((s, b2s(mid), fields))
    return out


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


class TTLCache:
    def __init__(self, ttl_s: float, max_items: int = 50000):
        self.ttl_s = float(ttl_s)
        self.max_items = int(max_items)
        self.store: Dict[Any, Tuple[float, Any]] = {}
        self.fifo = deque(maxlen=max_items)

    def put(self, key, value):
        now = time.time()
        self.store[key] = (now, value)
        self.fifo.append((now, key))
        self.prune()

    def get(self, key):
        item = self.store.get(key, None)
        if item is None:
            return None
        ts, val = item
        if (time.time() - ts) > self.ttl_s:
            self.store.pop(key, None)
            return None
        return val

    def prune(self):
        now = time.time()
        while self.fifo:
            ts, key = self.fifo[0]
            if (now - ts) <= self.ttl_s and len(self.store) <= self.max_items:
                break
            self.fifo.popleft()
            self.store.pop(key, None)


def try_get_stream(cfg: dict, key: str, cam_id: Optional[str] = None, fallback: Optional[str] = None):
    try:
        return get_stream(cfg, key, cam_id)
    except Exception:
        if fallback is not None:
            return fallback
        raise


def make_local_key(cam_id: str, person_track_id: int) -> Tuple[str, int]:
    return (str(cam_id), int(person_track_id))


def extract_event_identity(obj: dict) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    cam_id = obj.get("cam_id", None)
    pid = safe_int(obj.get("person_track_id", None), None)
    event_id = obj.get("event_id", None)
    if cam_id is None or pid is None:
        return None, None, event_id
    return str(cam_id), int(pid), event_id


def enrich_payload(obj: dict, global_person_id: Optional[int], mapping_debug: dict) -> dict:
    out = dict(obj)
    out["global_person_id"] = int(global_person_id) if global_person_id is not None else None
    out["identity_enriched"] = bool(global_person_id is not None)
    out["identity_debug"] = mapping_debug
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--redis_maxlen", type=int, default=50000)

    ap.add_argument("--mapping_ttl_s", type=float, default=120.0)
    ap.add_argument("--pending_ttl_s", type=float, default=3.0)
    ap.add_argument("--max_pending", type=int, default=50000)

    ap.add_argument("--defer_unmapped", action="store_true")
    ap.add_argument("--include_clip_refs", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    active_cams = get_active_cams(cfg)

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
        decode_responses=False,
    )
    rdb.ping()

    global_tracks_stream = try_get_stream(cfg, "global_tracks", None, "global_tracks")

    streams_in: Dict[str, str] = {}
    streams_out: Dict[str, str] = {}

    for cam_id in active_cams:
        streams_in[f"scores:{cam_id}"] = try_get_stream(cfg, "scores", cam_id)
        streams_in[f"decisions:{cam_id}"] = try_get_stream(cfg, "decisions", cam_id)
        streams_in[f"alerts:{cam_id}"] = try_get_stream(cfg, "alerts", cam_id)

        streams_out[f"scores:{cam_id}"] = try_get_stream(cfg, "scores_enriched", cam_id, f"scores_enriched:{cam_id}")
        streams_out[f"decisions:{cam_id}"] = try_get_stream(cfg, "decisions_enriched", cam_id, f"decisions_enriched:{cam_id}")
        streams_out[f"alerts:{cam_id}"] = try_get_stream(cfg, "alerts_enriched", cam_id, f"alerts_enriched:{cam_id}")

        if args.include_clip_refs:
            streams_in[f"clip_refs:{cam_id}"] = try_get_stream(cfg, "clip_refs", cam_id)
            streams_out[f"clip_refs:{cam_id}"] = try_get_stream(cfg, "clip_refs_enriched", cam_id, f"clip_refs_enriched:{cam_id}")

    last_ids = {global_tracks_stream: "0-0"}
    for s in streams_in.values():
        last_ids[s] = "0-0"

    local_to_global = TTLCache(ttl_s=args.mapping_ttl_s, max_items=args.max_pending)
    pending_events = TTLCache(ttl_s=args.pending_ttl_s, max_items=args.max_pending)

    print(f"[identity_enricher] active_cams={active_cams}")
    print(f"[identity_enricher] global_tracks_stream={global_tracks_stream}")
    for name, s in streams_in.items():
        print(f"[identity_enricher] IN  {name} -> {s}")
    for name, s in streams_out.items():
        print(f"[identity_enricher] OUT {name} -> {s}")
    print(
        f"[identity_enricher] mapping_ttl_s={args.mapping_ttl_s} "
        f"pending_ttl_s={args.pending_ttl_s} defer_unmapped={args.defer_unmapped}"
    )

    def publish_enriched(internal_name: str, obj: dict, global_person_id: Optional[int], mapping_debug: dict):
        out_obj = enrich_payload(obj, global_person_id, mapping_debug)
        out_stream = streams_out[internal_name]

        cam_id = str(out_obj.get("cam_id", ""))
        pid = safe_int(out_obj.get("person_track_id", -1), -1)
        frame_id = safe_int(out_obj.get("frame_id_end", None), None)
        if frame_id is None:
            frame_id = safe_int(out_obj.get("frame_id", -1), -1)
        stamp_ns = safe_int(out_obj.get("stamp_ns_end", None), None)
        if stamp_ns is None:
            stamp_ns = safe_int(out_obj.get("stamp_ns", 0), 0)
        event_id = out_obj.get("event_id", "")

        fields = {
            "event_id": str(event_id or ""),
            "cam_id": cam_id,
            "person_track_id": str(pid),
            "frame_id": str(frame_id),
            "stamp_ns": str(stamp_ns),
            "json": json.dumps(out_obj),
        }
        if global_person_id is not None:
            fields["global_person_id"] = str(global_person_id)

        rdb.xadd(
            out_stream,
            fields,
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def handle_global_track(obj: dict):
        cam_id = obj.get("cam_id", None)
        pid = safe_int(obj.get("person_track_id", None), None)
        gid = safe_int(obj.get("global_person_id", None), None)
        if cam_id is None or pid is None or gid is None:
            return

        key = make_local_key(str(cam_id), int(pid))
        local_to_global.put(key, gid)

        # Try flush pending for this local track
        pend = pending_events.get(key)
        if not pend:
            return

        for internal_name, pending_obj in pend:
            publish_enriched(
                internal_name=internal_name,
                obj=pending_obj,
                global_person_id=gid,
                mapping_debug={
                    "source": "pending_flush_after_global_track",
                    "local_key": [str(cam_id), int(pid)],
                },
            )
        pending_events.put(key, [])

    def handle_local_event(internal_name: str, obj: dict):
        cam_id, pid, _event_id = extract_event_identity(obj)
        if cam_id is None or pid is None:
            return

        key = make_local_key(cam_id, pid)
        gid = local_to_global.get(key)

        if gid is not None:
            publish_enriched(
                internal_name=internal_name,
                obj=obj,
                global_person_id=gid,
                mapping_debug={
                    "source": "cache_hit",
                    "local_key": [cam_id, pid],
                },
            )
            return

        if args.defer_unmapped:
            current = pending_events.get(key)
            if current is None:
                current = []
            current.append((internal_name, obj))
            pending_events.put(key, current)
            return

        publish_enriched(
            internal_name=internal_name,
            obj=obj,
            global_person_id=None,
            mapping_debug={
                "source": "cache_miss_passthrough",
                "local_key": [cam_id, pid],
            },
        )

    while True:
        streams = rdb.xread(last_ids, block=args.block_ms, count=args.count)
        if not streams:
            local_to_global.prune()
            pending_events.prune()
            continue

        for sname, mid, fields in parse_xread(streams):
            last_ids[sname] = mid

            js = fields.get(b"json", None)
            if js is None:
                continue

            try:
                obj = json.loads(b2s(js))
            except Exception:
                continue

            if sname == global_tracks_stream:
                handle_global_track(obj)
                continue

            matched_internal_name = None
            for internal_name, stream_name in streams_in.items():
                if stream_name == sname:
                    matched_internal_name = internal_name
                    break

            if matched_internal_name is None:
                continue

            handle_local_event(matched_internal_name, obj)