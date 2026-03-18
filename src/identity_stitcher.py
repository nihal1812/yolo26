#!/usr/bin/env python3
"""
identity_stitcher.py

Consumes:
  - reid_embeddings:{cam} for all active cameras

Produces:
  - global_tracks

Purpose:
- stitch local person track_ids across many cameras into one global_person_id
- use appearance similarity + camera transition timing + optional topology weighting

Expected input message JSON from reid_node.py:
{
  "type": "reid_embedding",
  "event_id": "...",
  "cam_id": "cam0",
  "person_track_id": 12,
  "frame_id": 1234,
  "stamp_ns": 1730000000,
  "embedding_dim": 512,
  "embedding": [...],
  "bbox_xyxy": [...],
  "seg_frame_id": 1234
}

Optional config block:

cross_camera:
  max_idle_s: 30.0
  match_threshold: 0.72
  same_cam_match_threshold: 0.90
  min_transition_s_default: 0.0
  max_transition_s_default: 15.0
  prototype_momentum: 0.2

  topology:
    cam0:
      cam1: {min_s: 0.5, max_s: 8.0, weight: 1.0}
      cam2: {min_s: 1.0, max_s: 15.0, weight: 0.9}
    cam1:
      cam0: {min_s: 0.5, max_s: 8.0, weight: 1.0}
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import redis
import numpy as np

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


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class IdentityState:
    global_person_id: int
    prototype: np.ndarray
    last_cam_id: str
    last_local_track_id: int
    last_frame_id: int
    last_stamp_ns: int
    created_at_s: float
    updated_at_s: float
    seen_count: int = 1
    cameras_seen: set = field(default_factory=set)
    local_tracks: List[Tuple[str, int]] = field(default_factory=list)


class IdentityStitcher:
    def __init__(
        self,
        active_cams: List[str],
        max_idle_s: float,
        match_threshold: float,
        same_cam_match_threshold: float,
        min_transition_s_default: float,
        max_transition_s_default: float,
        topology: Dict[str, Dict[str, Dict[str, float]]],
        prototype_momentum: float = 0.2,
    ):
        self.active_cams = set(active_cams)
        self.max_idle_s = float(max_idle_s)
        self.match_threshold = float(match_threshold)
        self.same_cam_match_threshold = float(same_cam_match_threshold)
        self.min_transition_s_default = float(min_transition_s_default)
        self.max_transition_s_default = float(max_transition_s_default)
        self.topology = topology or {}
        self.prototype_momentum = float(prototype_momentum)

        self.next_global_id = 1
        self.identities: Dict[int, IdentityState] = {}

        # fast lookup: exact local track -> global id
        self.local_to_global: Dict[Tuple[str, int], int] = {}

    def _transition_cfg(self, cam_from: str, cam_to: str) -> Dict[str, float]:
        if cam_from in self.topology and cam_to in self.topology[cam_from]:
            return self.topology[cam_from][cam_to]
        return {
            "min_s": self.min_transition_s_default,
            "max_s": self.max_transition_s_default,
            "weight": 1.0,
        }

    def _transition_ok(self, prev_cam: str, new_cam: str, dt_s: float) -> Tuple[bool, float]:
        if prev_cam == new_cam:
            return True, 1.0

        cfg = self._transition_cfg(prev_cam, new_cam)
        min_s = float(cfg.get("min_s", self.min_transition_s_default))
        max_s = float(cfg.get("max_s", self.max_transition_s_default))
        weight = float(cfg.get("weight", 1.0))

        ok = (dt_s >= min_s) and (dt_s <= max_s)
        return ok, weight

    def _prune(self):
        now = time.time()
        dead = []
        for gid, st in self.identities.items():
            if (now - st.updated_at_s) > self.max_idle_s:
                dead.append(gid)

        for gid in dead:
            st = self.identities.pop(gid, None)
            if st is None:
                continue
            for key in list(self.local_to_global.keys()):
                if self.local_to_global.get(key) == gid:
                    self.local_to_global.pop(key, None)

    def _create_identity(
        self,
        cam_id: str,
        local_track_id: int,
        frame_id: int,
        stamp_ns: int,
        emb: np.ndarray,
    ) -> IdentityState:
        gid = self.next_global_id
        self.next_global_id += 1

        now = time.time()
        st = IdentityState(
            global_person_id=gid,
            prototype=emb.copy(),
            last_cam_id=cam_id,
            last_local_track_id=local_track_id,
            last_frame_id=frame_id,
            last_stamp_ns=stamp_ns,
            created_at_s=now,
            updated_at_s=now,
            seen_count=1,
            cameras_seen={cam_id},
            local_tracks=[(cam_id, local_track_id)],
        )
        self.identities[gid] = st
        self.local_to_global[(cam_id, local_track_id)] = gid
        return st

    def _update_identity(
        self,
        st: IdentityState,
        cam_id: str,
        local_track_id: int,
        frame_id: int,
        stamp_ns: int,
        emb: np.ndarray,
    ) -> IdentityState:
        alpha = self.prototype_momentum
        st.prototype = l2_normalize((1.0 - alpha) * st.prototype + alpha * emb)

        st.last_cam_id = cam_id
        st.last_local_track_id = local_track_id
        st.last_frame_id = frame_id
        st.last_stamp_ns = stamp_ns
        st.updated_at_s = time.time()
        st.seen_count += 1
        st.cameras_seen.add(cam_id)

        lt = (cam_id, local_track_id)
        if lt not in st.local_tracks:
            st.local_tracks.append(lt)
        self.local_to_global[lt] = st.global_person_id
        return st

    def assign(
        self,
        cam_id: str,
        local_track_id: int,
        frame_id: int,
        stamp_ns: int,
        emb: np.ndarray,
    ) -> Tuple[IdentityState, Dict[str, Any]]:
        self._prune()

        emb = l2_normalize(emb.astype(np.float32))
        key = (cam_id, local_track_id)

        # Fast path: exact same local track already known
        if key in self.local_to_global:
            gid = self.local_to_global[key]
            st = self.identities.get(gid, None)
            if st is not None:
                sim = cosine_sim(st.prototype, emb)
                self._update_identity(st, cam_id, local_track_id, frame_id, stamp_ns, emb)
                dbg = {
                    "matched": True,
                    "reason": "existing_local_track_mapping",
                    "candidate_global_id": gid,
                    "appearance_sim": float(sim),
                }
                return st, dbg

        best_gid = None
        best_score = -1e9
        best_sim = None
        best_dt_s = None
        best_transition_weight = None

        ts_s = stamp_ns * 1e-9 if stamp_ns > 0 else time.time()

        for gid, st in self.identities.items():
            prev_ts_s = st.last_stamp_ns * 1e-9 if st.last_stamp_ns > 0 else ts_s
            dt_s = max(0.0, ts_s - prev_ts_s)

            sim = cosine_sim(st.prototype, emb)

            if st.last_cam_id == cam_id:
                thr = self.same_cam_match_threshold
                transition_ok = True
                transition_weight = 1.0
            else:
                thr = self.match_threshold
                transition_ok, transition_weight = self._transition_ok(st.last_cam_id, cam_id, dt_s)

            if not transition_ok:
                continue
            if sim < thr:
                continue

            score = sim * transition_weight
            if score > best_score:
                best_score = score
                best_gid = gid
                best_sim = sim
                best_dt_s = dt_s
                best_transition_weight = transition_weight

        if best_gid is None:
            st = self._create_identity(cam_id, local_track_id, frame_id, stamp_ns, emb)
            dbg = {
                "matched": False,
                "reason": "new_global_identity",
                "candidate_global_id": st.global_person_id,
                "appearance_sim": None,
                "dt_s": None,
                "transition_weight": None,
                "match_score": None,
            }
            return st, dbg

        st = self.identities[best_gid]
        self._update_identity(st, cam_id, local_track_id, frame_id, stamp_ns, emb)
        dbg = {
            "matched": True,
            "reason": "appearance_topology_match",
            "candidate_global_id": best_gid,
            "appearance_sim": float(best_sim),
            "dt_s": float(best_dt_s),
            "transition_weight": float(best_transition_weight),
            "match_score": float(best_score),
        }
        return st, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--global_tracks_stream", default=None)
    ap.add_argument("--redis_maxlen", type=int, default=50000)
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

    ccfg = cfg.get("cross_camera", {})
    max_idle_s = float(ccfg.get("max_idle_s", 30.0))
    match_threshold = float(ccfg.get("match_threshold", 0.72))
    same_cam_match_threshold = float(ccfg.get("same_cam_match_threshold", 0.90))
    min_transition_s_default = float(ccfg.get("min_transition_s_default", 0.0))
    max_transition_s_default = float(ccfg.get("max_transition_s_default", 15.0))
    prototype_momentum = float(ccfg.get("prototype_momentum", 0.2))
    topology = ccfg.get("topology", {}) or {}

    try:
        global_tracks_stream = args.global_tracks_stream or get_stream(cfg, "global_tracks")
    except Exception:
        global_tracks_stream = "global_tracks"

    stitcher = IdentityStitcher(
        active_cams=active_cams,
        max_idle_s=max_idle_s,
        match_threshold=match_threshold,
        same_cam_match_threshold=same_cam_match_threshold,
        min_transition_s_default=min_transition_s_default,
        max_transition_s_default=max_transition_s_default,
        topology=topology,
        prototype_momentum=prototype_momentum,
    )

    stream_map = {}
    last_ids = {}
    for cam_id in active_cams:
        try:
            s = get_stream(cfg, "reid_embeddings", cam_id)
        except Exception:
            s = f"reid_embeddings:{cam_id}"
        stream_map[cam_id] = s
        last_ids[s] = "0-0"

    print(f"[identity_stitcher] active_cams={active_cams}")
    for cam, stream in stream_map.items():
        print(f"[identity_stitcher] {cam} -> {stream}")
    print(f"[identity_stitcher] global_tracks_stream={global_tracks_stream}")
    print(
        f"[identity_stitcher] match_threshold={match_threshold} "
        f"same_cam_match_threshold={same_cam_match_threshold} "
        f"max_idle_s={max_idle_s}"
    )

    while True:
        streams = rdb.xread(last_ids, block=args.block_ms, count=args.count)
        if not streams:
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

            cam_id = str(obj.get("cam_id", "")).strip()
            local_track_id = int(obj.get("person_track_id", -1))
            frame_id = int(obj.get("frame_id", -1))
            stamp_ns = int(obj.get("stamp_ns", 0))
            event_id = obj.get("event_id", "")
            emb_list = obj.get("embedding", None)

            if not cam_id or local_track_id < 0 or emb_list is None:
                continue

            try:
                emb = np.asarray(emb_list, dtype=np.float32)
            except Exception:
                continue

            st, dbg = stitcher.assign(
                cam_id=cam_id,
                local_track_id=local_track_id,
                frame_id=frame_id,
                stamp_ns=stamp_ns,
                emb=emb,
            )

            out = {
                "type": "global_track",
                "event_id": event_id,
                "cam_id": cam_id,
                "person_track_id": int(local_track_id),
                "global_person_id": int(st.global_person_id),
                "frame_id": int(frame_id),
                "stamp_ns": int(stamp_ns),
                "seen_count": int(st.seen_count),
                "cameras_seen": sorted(list(st.cameras_seen)),
                "debug": dbg,
            }

            rdb.xadd(
                global_tracks_stream,
                {
                    "event_id": str(event_id or ""),
                    "cam_id": str(cam_id),
                    "person_track_id": str(local_track_id),
                    "global_person_id": str(st.global_person_id),
                    "frame_id": str(frame_id),
                    "stamp_ns": str(stamp_ns),
                    "json": json.dumps(out),
                },
                maxlen=args.redis_maxlen,
                approximate=True,
            )

            print(
                f"[identity_stitcher] cam={cam_id} local_pid={local_track_id} "
                f"-> global_pid={st.global_person_id} reason={dbg.get('reason')}"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[identity_stitcher] stopping...")