#!/usr/bin/env python3

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import numpy as np


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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class NullGlobalTrackPublisher:
    def publish_global_track(self, payload: dict):
        return


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
    cameras_seen: Set[str] = field(default_factory=set)
    local_tracks: List[Tuple[str, int]] = field(default_factory=list)


class IdentityStitcher:
    def __init__(
        self,
        active_cams: List[str],
        max_idle_s: float = 90.0,
        match_threshold: float = 0.72,
        same_cam_match_threshold: float = 0.88,
        same_cam_local_reuse_threshold: float = 0.75,
        min_transition_s_default: float = 0.0,
        max_transition_s_default: float = 30.0,
        topology: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        prototype_momentum: float = 0.08,
        max_local_tracks_per_identity: int = 32,
        dedupe_ttl_s: float = 10.0,
        publisher=None,
        same_cam_reuse_max_dt_s: float = 8.0,
        min_cross_cam_dt_s: float = 0.10,
        debug_log: bool = False,
        health_log_every_s: float = 10.0,
        weak_keep_threshold: float = 0.50,
        strong_keep_threshold: float = 0.78,
        gid_lock_min_age_s: float = 3.0,
        remap_strong_sim: float = 0.92,
        remap_margin: float = 0.12,
        remap_cooldown_s: float = 8.0,
    ):
        self.active_cams = set(str(c) for c in active_cams)

        self.max_idle_s = float(max_idle_s)
        self.match_threshold = float(match_threshold)
        self.same_cam_match_threshold = float(same_cam_match_threshold)
        self.same_cam_local_reuse_threshold = float(same_cam_local_reuse_threshold)

        self.min_transition_s_default = float(min_transition_s_default)
        self.max_transition_s_default = float(max_transition_s_default)
        self.topology = topology or {}

        self.prototype_momentum = float(clamp(prototype_momentum, 0.0, 1.0))
        self.max_local_tracks_per_identity = int(max_local_tracks_per_identity)
        self.dedupe_ttl_s = float(dedupe_ttl_s)

        self.publisher = publisher if publisher is not None else NullGlobalTrackPublisher()

        self.same_cam_reuse_max_dt_s = float(same_cam_reuse_max_dt_s)
        self.min_cross_cam_dt_s = float(min_cross_cam_dt_s)

        self.debug_log = bool(debug_log)
        self.health_log_every_s = float(health_log_every_s)

        # Keeps same-camera local tracks stable during weak crops / blur / occlusion.
        self.weak_keep_threshold = float(weak_keep_threshold)

        # Keeps an existing local-track mapping if similarity is strong,
        # even if the global identity was recently updated by another camera.
        self.strong_keep_threshold = float(strong_keep_threshold)
        # Production hardening: prevent unstable local-track gid remaps.
        self.gid_lock_min_age_s = float(gid_lock_min_age_s)
        self.remap_strong_sim = float(remap_strong_sim)
        self.remap_margin = float(remap_margin)
        self.remap_cooldown_s = float(remap_cooldown_s)

        self.next_global_id = 1

        self.identities: Dict[int, IdentityState] = {}
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        self.seen_event_ids: Dict[str, float] = {}

        self.stats = {
            "embeddings": 0,
            "new_gid": 0,
            "matched_gid": 0,
            "duplicates": 0,
            "invalid": 0,
            "cross_cam_matches": 0,
            "same_cam_matches": 0,
            "local_mapping_drops": 0,
            "same_camera_remaps": 0,
            "repeated_local_track_remaps": 0,
            "possible_id_switches": 0,
            "last_health_log_s": time.time(),
            "last_health_new_gid": 0,
            "last_health_local_mapping_drops": 0,
            "last_health_same_camera_remaps": 0,
        }
        self.local_last_gid: Dict[Tuple[str, int], int] = {}
        self.local_remap_counts: Dict[Tuple[str, int], int] = {}
        self.local_first_assigned_at_s: Dict[Tuple[str, int], float] = {}
        self.local_last_remap_at_s: Dict[Tuple[str, int], float] = {}

        # Production hardening: rate-limit noisy identity lock logs.
        # Behavior is unchanged; this only reduces repeated debug output.
        self.identity_log_every_s = 5.0
        self._identity_log_last: Dict[Tuple[str, str, int], float] = {}

    def _log(self, msg: str):
        if self.debug_log:
            print(f"[identity_stitcher] {msg}")

    def _rate_limited_identity_log(self, kind: str, cam_id: str, local_track_id: int, msg: str, every_s: float = None):
        """
        Rate-limit repeated identity hardening logs per (kind, cam, local_track_id).
        This does not change assignment behavior.
        """
        if not self.debug_log:
            return False

        if every_s is None:
            every_s = float(getattr(self, "identity_log_every_s", 5.0))

        if every_s <= 0:
            self._log(msg)
            return True

        key = (str(kind), str(cam_id), int(local_track_id))
        now = time.time()
        last = float(self._identity_log_last.get(key, 0.0))

        if (now - last) >= every_s:
            self._identity_log_last[key] = now
            self._log(msg)
            return True

        return False


    def _maybe_log_health(self):
        now = time.time()

        if (now - self.stats["last_health_log_s"]) < self.health_log_every_s:
            return

        total = max(1, self.stats["embeddings"])
        matched = self.stats["matched_gid"]
        gid_coverage = 100.0 * matched / total
        dt = max(1e-6, now - self.stats["last_health_log_s"])
        new_gid_delta = int(self.stats["new_gid"] - self.stats["last_health_new_gid"])
        drops_delta = int(self.stats["local_mapping_drops"] - self.stats["last_health_local_mapping_drops"])
        remap_delta = int(self.stats["same_camera_remaps"] - self.stats["last_health_same_camera_remaps"])

        print(
            f"[reid_health] embeddings={self.stats['embeddings']} "
            f"matched_gid={matched} new_gid={self.stats['new_gid']} "
            f"gid_coverage={gid_coverage:.1f}% "
            f"new_gid_per_s={new_gid_delta / dt:.2f} "
            f"same_cam_matches={self.stats['same_cam_matches']} "
            f"cross_cam_matches={self.stats['cross_cam_matches']} "
            f"duplicates={self.stats['duplicates']} "
            f"invalid={self.stats['invalid']} "
            f"local_mapping_drops={self.stats['local_mapping_drops']} "
            f"local_mapping_drops_per_s={drops_delta / dt:.2f} "
            f"same_camera_remaps={self.stats['same_camera_remaps']} "
            f"same_camera_remaps_per_s={remap_delta / dt:.2f} "
            f"repeated_local_track_remaps={self.stats['repeated_local_track_remaps']} "
            f"possible_id_switches={self.stats['possible_id_switches']} "
            f"active_identities={len(self.identities)}"
        )

        self.stats["last_health_log_s"] = now
        self.stats["last_health_new_gid"] = self.stats["new_gid"]
        self.stats["last_health_local_mapping_drops"] = self.stats["local_mapping_drops"]
        self.stats["last_health_same_camera_remaps"] = self.stats["same_camera_remaps"]

    def _record_local_assignment(self, cam_id: str, local_track_id: int, gid: int, reason: str):
        key = (str(cam_id), int(local_track_id))
        prev_gid = self.local_last_gid.get(key)
        if prev_gid is not None and int(prev_gid) != int(gid):
            self.stats["same_camera_remaps"] += 1
            self.stats["possible_id_switches"] += 1
            count = int(self.local_remap_counts.get(key, 0)) + 1
            self.local_remap_counts[key] = count
            if count > 1:
                self.stats["repeated_local_track_remaps"] += 1
            self._log(
                f"possible_id_switch cam={cam_id} ltid={local_track_id} "
                f"prev_gid={prev_gid} new_gid={gid} reason={reason} remap_count={count}"
            )
        self.local_last_gid[key] = int(gid)

    def _prune_seen_events(self):
        now = time.time()

        dead = [
            eid
            for eid, ts in self.seen_event_ids.items()
            if (now - ts) > self.dedupe_ttl_s
        ]

        for eid in dead:
            self.seen_event_ids.pop(eid, None)

    def _mark_seen_event(self, event_id: str):
        if event_id:
            self.seen_event_ids[str(event_id)] = time.time()

    def _already_seen_event(self, event_id: str) -> bool:
        if not event_id:
            return False

        self._prune_seen_events()
        return str(event_id) in self.seen_event_ids

    def _transition_cfg(self, cam_from: str, cam_to: str) -> Dict[str, float]:
        if cam_from in self.topology and cam_to in self.topology[cam_from]:
            cfg = self.topology[cam_from][cam_to] or {}

            return {
                "min_s": float(cfg.get("min_s", self.min_transition_s_default)),
                "max_s": float(cfg.get("max_s", self.max_transition_s_default)),
                "weight": float(cfg.get("weight", 1.0)),
            }

        return {
            "min_s": self.min_transition_s_default,
            "max_s": self.max_transition_s_default,
            "weight": 1.0,
        }

    def _transition_ok(self, prev_cam: str, new_cam: str, dt_s: float):
        if prev_cam == new_cam:
            return True, 1.0

        cfg = self._transition_cfg(prev_cam, new_cam)

        min_s = float(cfg.get("min_s", self.min_transition_s_default))
        max_s = float(cfg.get("max_s", self.max_transition_s_default))
        weight = max(0.0, float(cfg.get("weight", 1.0)))

        ok = (dt_s >= max(min_s, self.min_cross_cam_dt_s)) and (dt_s <= max_s)

        return ok, weight

    def _prune(self):
        now = time.time()

        dead_gids = []

        for gid, st in self.identities.items():
            if (now - st.updated_at_s) > self.max_idle_s:
                dead_gids.append(gid)

        for gid in dead_gids:
            self.identities.pop(gid, None)

            dead_keys = [
                key
                for key, mapped_gid in self.local_to_global.items()
                if mapped_gid == gid
            ]

            for key in dead_keys:
                self.local_to_global.pop(key, None)
                self.local_last_gid.pop(key, None)
                self.local_remap_counts.pop(key, None)
                self.local_first_assigned_at_s.pop(key, None)
                self.local_last_remap_at_s.pop(key, None)

        self._prune_seen_events()

    def _create_identity(
        self,
        cam_id: str,
        local_track_id: int,
        frame_id: int,
        stamp_ns: int,
        emb: np.ndarray,
    ) -> IdentityState:
        local_key = (cam_id, local_track_id)

        existing_gid = self.local_to_global.get(local_key)
        if existing_gid is None:
            existing_gid = self.local_last_gid.get(local_key)

        if existing_gid is not None:
            existing_st = self.identities.get(int(existing_gid))
            if existing_st is not None:
                self.stats["local_mapping_drops"] += 1
                self._rate_limited_identity_log(
                    "create_identity_blocked_existing_local",
                    cam_id,
                    local_track_id,
                    (
                        f"create_identity_blocked_existing_local cam={cam_id} "
                        f"ltid={local_track_id} existing_gid={existing_gid}"
                    ),
                )
                return self._keep_identity_without_prototype_update(
                    existing_st,
                    cam_id,
                    local_track_id,
                    frame_id,
                    stamp_ns,
                    int(existing_gid),
                )
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
        self.local_to_global[local_key] = gid
        self.local_first_assigned_at_s[local_key] = now
        self._record_local_assignment(cam_id, local_track_id, gid, "create_identity")

        return st


    def _remap_allowed(
        self,
        local_key,
        old_gid: int,
        new_gid: int,
        emb,
        now_s: float,
    ):
        """
        Production hard lock:
        Once a local track has a gid, do not remap it during that local-track lifetime.
        This prevents gid bouncing like gid 5 -> 6 -> 5.
        """
        old_st = self.identities.get(int(old_gid))
        new_st = self.identities.get(int(new_gid))

        old_sim = -1.0
        new_sim = -1.0
        margin = 0.0

        try:
            if old_st is not None:
                old_sim = cosine_sim(emb, old_st.prototype)
            if new_st is not None:
                new_sim = cosine_sim(emb, new_st.prototype)
            margin = float(new_sim - old_sim)
        except Exception:
            pass

        first_seen = float(self.local_first_assigned_at_s.get(local_key, now_s))
        local_age_s = max(0.0, now_s - first_seen)

        debug = {
            "old_gid": int(old_gid),
            "new_gid": int(new_gid),
            "old_sim": float(old_sim),
            "new_sim": float(new_sim),
            "margin": float(margin),
            "local_age_s": float(local_age_s),
        }

        return False, "hard_local_gid_lock", debug


    def _update_identity(
        self,
        st: IdentityState,
        cam_id: str,
        local_track_id: int,
        frame_id: int,
        stamp_ns: int,
        emb: np.ndarray,
    ) -> IdentityState:
        local_key = (cam_id, local_track_id)
        now_s = time.time()

        old_gid = self.local_to_global.get(local_key)
        if old_gid is None:
            old_gid = self.local_last_gid.get(local_key)

        new_gid = int(st.global_person_id)

        if old_gid is not None and int(old_gid) != new_gid:
            allowed, reason, dbg = self._remap_allowed(
                local_key,
                int(old_gid),
                new_gid,
                emb,
                now_s,
            )

            if not allowed:
                old_st = self.identities.get(int(old_gid))
                if old_st is not None:
                    self.stats["local_mapping_drops"] += 1
                    self._rate_limited_identity_log(
                        "gid_remap_blocked",
                        cam_id,
                        local_track_id,
                        (
                            f"gid_remap_blocked cam={cam_id} ltid={local_track_id} "
                            f"old_gid={old_gid} new_gid={new_gid} reason={reason} "
                            f"old_sim={dbg.get('old_sim', -1.0):.3f} "
                            f"new_sim={dbg.get('new_sim', -1.0):.3f} "
                            f"margin={dbg.get('margin', 0.0):.3f} "
                            f"age={dbg.get('local_age_s', 0.0):.2f}"
                        ),
                    )
                    return self._keep_identity_without_prototype_update(
                        old_st,
                        cam_id,
                        local_track_id,
                        frame_id,
                        stamp_ns,
                        int(old_gid),
                    )

            self.local_last_remap_at_s[local_key] = now_s

        alpha = self.prototype_momentum
        st.prototype = l2_normalize((1.0 - alpha) * st.prototype + alpha * emb)
        st.last_cam_id = cam_id
        st.last_local_track_id = local_track_id
        st.last_frame_id = frame_id
        st.last_stamp_ns = stamp_ns
        st.updated_at_s = now_s
        st.seen_count += 1
        st.cameras_seen.add(cam_id)

        if local_key not in st.local_tracks:
            st.local_tracks.append(local_key)

        if len(st.local_tracks) > self.max_local_tracks_per_identity:
            st.local_tracks = st.local_tracks[-self.max_local_tracks_per_identity:]

        self.local_to_global[local_key] = st.global_person_id
        self.local_first_assigned_at_s.setdefault(local_key, now_s)
        self._record_local_assignment(cam_id, local_track_id, st.global_person_id, "update_identity")
        return st


    def _keep_identity_without_prototype_update(
        self,
        st: IdentityState,
        cam_id: str,
        local_track_id: int,
        frame_id: int,
        stamp_ns: int,
        gid: int,
    ) -> IdentityState:
        """
        Keep the old gid, but do not update the appearance prototype.

        Useful when the local track is probably the same person, but the crop quality
        or cross-camera timing makes the embedding unsafe for prototype update.
        """
        st.last_cam_id = cam_id
        st.last_local_track_id = local_track_id
        st.last_frame_id = frame_id
        st.last_stamp_ns = stamp_ns
        st.updated_at_s = time.time()
        st.seen_count += 1
        st.cameras_seen.add(cam_id)

        local_key = (cam_id, local_track_id)

        if local_key not in st.local_tracks:
            st.local_tracks.append(local_key)

        if len(st.local_tracks) > self.max_local_tracks_per_identity:
            st.local_tracks = st.local_tracks[-self.max_local_tracks_per_identity:]

        self.local_to_global[local_key] = gid
        self._record_local_assignment(cam_id, local_track_id, gid, "keep_identity")

        return st

    def _drop_local_mapping(self, cam_id: str, local_track_id: int):
        local_key = (cam_id, local_track_id)

        # Keep local_last_gid on purpose.
        # It is the safety memory that prevents _create_identity from assigning
        # a new gid to the same local track after a temporary mapping drop.
        self.local_to_global.pop(local_key, None)
        self.stats["local_mapping_drops"] += 1

    def assign(
        self,
        cam_id: str,
        local_track_id: int,
        frame_id: int,
        stamp_ns: int,
        emb: np.ndarray,
    ):
        self._prune()

        cam_id = str(cam_id)
        local_track_id = int(local_track_id)
        frame_id = int(frame_id)
        stamp_ns = int(stamp_ns)

        emb = l2_normalize(emb.astype(np.float32))

        key = (cam_id, local_track_id)
        ts_s = stamp_ns * 1e-9 if stamp_ns > 0 else time.time()

        # ------------------------------------------------------------
        # 1. Existing local-track mapping path
        # ------------------------------------------------------------
        if key in self.local_to_global:
            gid = self.local_to_global[key]
            st = self.identities.get(gid, None)

            if st is not None:
                prev_cam_id = st.last_cam_id
                prev_ts_s = st.last_stamp_ns * 1e-9 if st.last_stamp_ns > 0 else ts_s
                dt_s = max(0.0, ts_s - prev_ts_s)
                sim = cosine_sim(st.prototype, emb)

                # Strong local-track ownership safety:
                # If this exact local track is already mapped to this gid and the
                # appearance is strong, keep it even if the identity's last camera
                # was updated by another camera.
                if sim >= self.strong_keep_threshold and dt_s <= self.same_cam_reuse_max_dt_s:
                    self._keep_identity_without_prototype_update(
                        st,
                        cam_id,
                        local_track_id,
                        frame_id,
                        stamp_ns,
                        gid,
                    )

                    dbg = {
                        "matched": True,
                        "reason": "existing_local_track_mapping_strong_keep",
                        "candidate_global_id": gid,
                        "appearance_sim": float(sim),
                        "dt_s": float(dt_s),
                        "transition_weight": 1.0,
                        "match_score": float(sim),
                        "prev_cam_id": prev_cam_id,
                        "new_cam_id": cam_id,
                        "cross_camera": bool(prev_cam_id != cam_id),
                    }

                    self._log(
                        f"cam={cam_id} ltid={local_track_id} gid={gid} "
                        f"reason={dbg['reason']} sim={sim:.3f} dt={dt_s:.3f} "
                        f"cross_camera={dbg['cross_camera']}"
                    )

                    return st, dbg

                # Normal same-camera local reuse:
                # update prototype only when similarity is confidently above threshold.
                if st.last_cam_id == cam_id and dt_s <= self.same_cam_reuse_max_dt_s:
                    if sim >= self.same_cam_local_reuse_threshold:
                        self._update_identity(
                            st,
                            cam_id,
                            local_track_id,
                            frame_id,
                            stamp_ns,
                            emb,
                        )

                        dbg = {
                            "matched": True,
                            "reason": "existing_local_track_mapping_validated",
                            "candidate_global_id": gid,
                            "appearance_sim": float(sim),
                            "dt_s": float(dt_s),
                            "transition_weight": 1.0,
                            "match_score": float(sim),
                            "prev_cam_id": prev_cam_id,
                            "new_cam_id": cam_id,
                            "cross_camera": False,
                        }

                        self._log(
                            f"cam={cam_id} ltid={local_track_id} gid={gid} "
                            f"reason={dbg['reason']} sim={sim:.3f} dt={dt_s:.3f}"
                        )

                        return st, dbg

                    # Weak same-camera local reuse:
                    # keep gid but avoid prototype update.
                    if sim >= self.weak_keep_threshold:
                        self._keep_identity_without_prototype_update(
                            st,
                            cam_id,
                            local_track_id,
                            frame_id,
                            stamp_ns,
                            gid,
                        )

                        dbg = {
                            "matched": True,
                            "reason": "existing_local_track_mapping_weak_keep",
                            "candidate_global_id": gid,
                            "appearance_sim": float(sim),
                            "dt_s": float(dt_s),
                            "transition_weight": 1.0,
                            "match_score": float(sim),
                            "prev_cam_id": prev_cam_id,
                            "new_cam_id": cam_id,
                            "cross_camera": False,
                        }

                        self._log(
                            f"cam={cam_id} ltid={local_track_id} gid={gid} "
                            f"reason={dbg['reason']} sim={sim:.3f} dt={dt_s:.3f}"
                        )

                        return st, dbg

                self._log(
                    f"dropping stale local map cam={cam_id} ltid={local_track_id} "
                    f"gid={gid} sim={sim:.3f} dt={dt_s:.3f}"
                )

            # If mapped gid is missing, or validation failed, remove local mapping
            # and continue to global identity search.
            self._drop_local_mapping(cam_id, local_track_id)

        # ------------------------------------------------------------
        # 2. Search among existing identities
        # ------------------------------------------------------------
        best_gid = None
        best_score = -1e9
        best_sim = None
        best_dt_s = None
        best_transition_weight = None
        best_prev_cam = None

        for gid, st in self.identities.items():
            prev_ts_s = st.last_stamp_ns * 1e-9 if st.last_stamp_ns > 0 else ts_s
            dt_s = max(0.0, ts_s - prev_ts_s)

            sim = cosine_sim(st.prototype, emb)

            if st.last_cam_id == cam_id:
                threshold = self.same_cam_match_threshold
                transition_ok = dt_s <= self.same_cam_reuse_max_dt_s
                transition_weight = 1.0
            else:
                threshold = self.match_threshold
                transition_ok, transition_weight = self._transition_ok(
                    st.last_cam_id,
                    cam_id,
                    dt_s,
                )

            if not transition_ok:
                continue

            if sim < threshold:
                continue

            score = sim * transition_weight

            if score > best_score:
                best_score = score
                best_gid = gid
                best_sim = sim
                best_dt_s = dt_s
                best_transition_weight = transition_weight
                best_prev_cam = st.last_cam_id

        # ------------------------------------------------------------
        # 3. No match found: create new global identity
        # ------------------------------------------------------------
        if best_gid is None:
            st = self._create_identity(
                cam_id,
                local_track_id,
                frame_id,
                stamp_ns,
                emb,
            )

            dbg = {
                "matched": False,
                "reason": "new_global_identity",
                "candidate_global_id": st.global_person_id,
                "appearance_sim": None,
                "dt_s": None,
                "transition_weight": None,
                "match_score": None,
                "prev_cam_id": None,
                "new_cam_id": cam_id,
                "cross_camera": False,
            }

            self._log(
                f"cam={cam_id} ltid={local_track_id} gid={st.global_person_id} "
                f"reason={dbg['reason']}"
            )

            return st, dbg

        # ------------------------------------------------------------
        # 4. Match found: update existing global identity
        # ------------------------------------------------------------
        st = self.identities[best_gid]
        prev_cam_id = best_prev_cam
        cross_camera = prev_cam_id != cam_id

        self._update_identity(
            st,
            cam_id,
            local_track_id,
            frame_id,
            stamp_ns,
            emb,
        )

        dbg = {
            "matched": True,
            "reason": "appearance_topology_match",
            "candidate_global_id": best_gid,
            "appearance_sim": float(best_sim),
            "dt_s": float(best_dt_s),
            "transition_weight": float(best_transition_weight),
            "match_score": float(best_score),
            "prev_cam_id": prev_cam_id,
            "new_cam_id": cam_id,
            "cross_camera": bool(cross_camera),
        }

        self._log(
            f"cam={cam_id} ltid={local_track_id} gid={best_gid} "
            f"reason={dbg['reason']} sim={best_sim:.3f} dt={best_dt_s:.3f} "
            f"tw={best_transition_weight:.3f} score={best_score:.3f} "
            f"cross_camera={cross_camera}"
        )

        return st, dbg

    def process_embedding(self, emb_obj: dict) -> dict:
        self.stats["embeddings"] += 1

        event_id = str(emb_obj.get("event_id", "")).strip()
        match_keys = emb_obj.get("match_keys", [])

        if not isinstance(match_keys, list):
            match_keys = []

        clean_match_keys = sorted(set(str(k) for k in match_keys if k))

        if event_id and event_id not in clean_match_keys:
            clean_match_keys.append(event_id)

        clean_match_keys = sorted(set(clean_match_keys))

        if self._already_seen_event(event_id):
            self.stats["duplicates"] += 1
            self._maybe_log_health()

            return {
                "type": "global_track",
                "event_id": event_id,
                "cam_id": emb_obj.get("cam_id", ""),
                "person_track_id": int(emb_obj.get("person_track_id", -1)),
                "global_person_id": None,
                "frame_id": int(emb_obj.get("frame_id", -1)),
                "stamp_ns": int(emb_obj.get("stamp_ns", 0)),
                "seen_count": None,
                "cameras_seen": [],
                "match_keys": clean_match_keys,
                "debug": {
                    "matched": False,
                    "reason": "duplicate_event_ignored",
                    "candidate_global_id": None,
                },
                "appearance_sim": None,
                "match_reason": "duplicate_event_ignored",
                "match_score": None,
                "transition_dt_s": None,
                "prev_cam_id": None,
                "new_cam_id": emb_obj.get("cam_id", ""),
                "cross_camera": False,
            }

        cam_id = str(emb_obj.get("cam_id", "")).strip()
        local_track_id = int(emb_obj.get("person_track_id", -1))
        frame_id = int(emb_obj.get("frame_id", -1))
        stamp_ns = int(emb_obj.get("stamp_ns", 0))
        emb_list = emb_obj.get("embedding", None)

        if not cam_id or local_track_id < 0 or emb_list is None:
            self.stats["invalid"] += 1
            self._maybe_log_health()
            raise ValueError("Invalid embedding object")

        emb = np.asarray(emb_list, dtype=np.float32)

        st, dbg = self.assign(
            cam_id=cam_id,
            local_track_id=local_track_id,
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            emb=emb,
        )
        t_identity_ns = time.time_ns()

        if dbg.get("matched"):
            self.stats["matched_gid"] += 1

            if dbg.get("cross_camera"):
                self.stats["cross_cam_matches"] += 1
            else:
                self.stats["same_cam_matches"] += 1
        else:
            self.stats["new_gid"] += 1

        self._maybe_log_health()

        out = {
            "type": "global_track",
            "event_id": event_id,
            "cam_id": cam_id,
            "person_track_id": int(local_track_id),
            "global_person_id": int(st.global_person_id),
            "frame_id": int(frame_id),
            "stamp_ns": int(stamp_ns),
            "t_capture_ns": int(emb_obj.get("t_capture_ns", stamp_ns)),
            "t_pose_ns": int(emb_obj.get("t_pose_ns", 0) or 0),
            "t_seg_ns": int(emb_obj.get("t_seg_ns", 0) or 0),
            "t_reid_ns": int(emb_obj.get("t_reid_ns", 0) or 0),
            "t_identity_ns": int(t_identity_ns),
            "seen_count": int(st.seen_count),
            "cameras_seen": sorted(list(st.cameras_seen)),
            "match_keys": clean_match_keys,
            "debug": dbg,
            "appearance_sim": dbg.get("appearance_sim"),
            "match_reason": dbg.get("reason"),
            "match_score": dbg.get("match_score"),
            "transition_dt_s": dbg.get("dt_s"),
            "prev_cam_id": dbg.get("prev_cam_id"),
            "new_cam_id": dbg.get("new_cam_id"),
            "cross_camera": dbg.get("cross_camera", False),
        }

        self._mark_seen_event(event_id)

        try:
            self.publisher.publish_global_track(out)
        except Exception:
            pass

        return out
