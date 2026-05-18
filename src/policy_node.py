#!/usr/bin/env python3
"""
policy_node.py
(policy + local alert emitter, split-score + global identity aware)

Transport model:
  - subscribe to scores via ZMQ
  - publish decisions via ZMQ
  - publish alerts via ZMQ
  - optionally read policy_updates from Redis as a control path

Local alerts are still emitted and are intended to be consumed by:
- identity_enricher
- incident_builder

So alerts from this node are "local_preincident" alerts, not final cross-camera incidents.
"""

import argparse
import json
import time
import math
from collections import defaultdict, deque

import zmq

try:
    import redis
except Exception:
    redis = None

from config_utils import load_cfg, get_stream, get_zmq_endpoint, local_connect_addr


def parse_xread_messages(streams):
    out = []
    for _sname, msgs in streams:
        for mid, fields in msgs:
            out.append((mid.decode() if isinstance(mid, bytes) else str(mid), fields))
    return out


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def safe_bool(v, default=False):
    if isinstance(v, bool):
        return bool(v)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def load_policy_cfg(cfg: dict) -> dict:
    """
    Policy settings can come from either:

      policy:
        threshold: ...

    or:

      brain:
        policy_node_args:
          threshold: ...

    brain.policy_node_args wins if both are present.
    """
    policy_cfg = {}

    top_policy = cfg.get("policy", {})
    if isinstance(top_policy, dict):
        policy_cfg.update(top_policy)

    brain_policy = cfg.get("brain", {}).get("policy_node_args", {})
    if isinstance(brain_policy, dict):
        policy_cfg.update(brain_policy)

    return policy_cfg


class PolicyWorker:
    def __init__(self, args):
        self.args = args
        self.cfg = load_cfg(args.config)
        self.cam_id = str(args.cam_id)

        self.rdb = None
        self.policy_updates_stream = None
        self.last_upd_id = "0-0"

        disable_redis_updates = getattr(args, "disable_redis_updates", False)

        if redis is not None and not disable_redis_updates:
            try:
                r_cfg = self.cfg.get("redis", {})
                self.rdb = redis.Redis(
                    host=r_cfg.get("host", "127.0.0.1"),
                    port=int(r_cfg.get("port", 6379)),
                    db=int(r_cfg.get("db", 0)),
                    password=r_cfg.get("password", None),
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                    health_check_interval=30,
                )
                self.rdb.ping()
                self.policy_updates_stream = getattr(args, "policy_updates_stream", None) or get_stream(
                    self.cfg,
                    "policy_updates",
                )
            except Exception as e:
                print(f"[policy_node] Redis policy updates disabled: {e}")
                self.rdb = None
                self.policy_updates_stream = None

        scores_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "scores")
        decisions_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "decisions")
        alerts_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "alerts")

        self.scores_connect = getattr(args, "scores_connect", None) or local_connect_addr(scores_cfg["bind"])
        self.scores_topic = getattr(args, "scores_topic", None) or scores_cfg["topic"]

        self.decisions_bind = getattr(args, "decisions_bind", None) or decisions_cfg["bind"]
        self.decisions_topic = getattr(args, "decisions_topic", None) or decisions_cfg["topic"]

        self.alerts_bind = getattr(args, "alerts_bind", None) or alerts_cfg["bind"]
        self.alerts_topic = getattr(args, "alerts_topic", None) or alerts_cfg["topic"]

        self.scores_topic_b = self.scores_topic.encode("utf-8")
        self.decisions_topic_b = self.decisions_topic.encode("utf-8")
        self.alerts_topic_b = self.alerts_topic.encode("utf-8")

        self.ctx = zmq.Context.instance()

        self.sub_scores = self.ctx.socket(zmq.SUB)
        self.sub_scores.setsockopt(zmq.LINGER, 0)
        self.sub_scores.setsockopt(zmq.RCVHWM, int(getattr(args, "zmq_rcvhwm", 256)))
        self.sub_scores.connect(self.scores_connect)
        self.sub_scores.setsockopt(zmq.SUBSCRIBE, self.scores_topic_b)

        self.pub_decisions = self.ctx.socket(zmq.PUB)
        self.pub_decisions.setsockopt(zmq.LINGER, 0)
        self.pub_decisions.setsockopt(zmq.SNDHWM, int(decisions_cfg.get("sndhwm", 1000)))
        self.pub_decisions.bind(self.decisions_bind)

        self.pub_alerts = self.ctx.socket(zmq.PUB)
        self.pub_alerts.setsockopt(zmq.LINGER, 0)
        self.pub_alerts.setsockopt(zmq.SNDHWM, int(alerts_cfg.get("sndhwm", 1000)))
        self.pub_alerts.bind(self.alerts_bind)

        policy_cfg = load_policy_cfg(self.cfg)

        self.policy = {
            "threshold": float(policy_cfg.get("threshold", getattr(args, "threshold", 0.75))),
            "M": int(policy_cfg.get("M", getattr(args, "M", 8))),
            "K": int(policy_cfg.get("K", getattr(args, "K", 4))),
            "cooldown_s": float(policy_cfg.get("cooldown_s", getattr(args, "cooldown_s", 10.0))),

            "max_missing_pose_ratio": float(
                policy_cfg.get("max_missing_pose_ratio", getattr(args, "max_missing_pose_ratio", 0.55))
            ),
            "max_missing_obj_ratio": float(
                policy_cfg.get("max_missing_obj_ratio", getattr(args, "max_missing_obj_ratio", 0.75))
            ),
            "drop_if_scalar_missing": bool(
                getattr(args, "drop_if_scalar_missing", False)
                or safe_bool(policy_cfg.get("drop_if_scalar_missing", False))
            ),

            "use_suspicion_policy": bool(
                getattr(args, "use_suspicion_policy", False)
                or safe_bool(policy_cfg.get("use_suspicion_policy", False))
            ),
            "suspicion_tau_s": float(
                policy_cfg.get("suspicion_tau_s", getattr(args, "suspicion_tau_s", 6.0))
            ),
            "suspicion_alert_thr": float(
                policy_cfg.get("suspicion_alert_thr", getattr(args, "suspicion_alert_thr", 0.85))
            ),
            "suspicion_persist_clips": int(
                policy_cfg.get("suspicion_persist_clips", getattr(args, "suspicion_persist_clips", 2))
            ),
            "suspicion_gain": float(
                policy_cfg.get("suspicion_gain", getattr(args, "suspicion_gain", 0.55))
            ),
            "suspicion_model_weight": float(
                policy_cfg.get("suspicion_model_weight", getattr(args, "suspicion_model_weight", 0.55))
            ),
            "suspicion_votes_weight": float(
                policy_cfg.get("suspicion_votes_weight", getattr(args, "suspicion_votes_weight", 0.45))
            ),

            "min_votes_to_alert": int(
                policy_cfg.get("min_votes_to_alert", getattr(args, "min_votes_to_alert", 1))
            ),
            "vote_use_heuristic_score": bool(
                getattr(args, "vote_use_heuristic_score", False)
                or safe_bool(policy_cfg.get("vote_use_heuristic_score", False))
            ),
            "heuristic_vote_thr": float(
                policy_cfg.get("heuristic_vote_thr", getattr(args, "heuristic_vote_thr", 0.80))
            ),
            "vote_contact_ratio_thr": float(
                policy_cfg.get("vote_contact_ratio_thr", getattr(args, "vote_contact_ratio_thr", 0.35))
            ),
            "vote_visibility_drop_thr": float(
                policy_cfg.get("vote_visibility_drop_thr", getattr(args, "vote_visibility_drop_thr", 0.25))
            ),
            "vote_carry_score_thr": float(
                policy_cfg.get("vote_carry_score_thr", getattr(args, "vote_carry_score_thr", 0.70))
            ),
            "vote_disappeared_after_contact": bool(
                getattr(args, "vote_disappeared_after_contact", False)
                or safe_bool(policy_cfg.get("vote_disappeared_after_contact", False))
            ),

            "use_global_identity_state": bool(
                getattr(args, "use_global_identity_state", False)
                or safe_bool(policy_cfg.get("use_global_identity_state", False))
            ),

            # If true, normal classic/suspicion alerts are suppressed until global_person_id exists.
            "require_global_id_for_alert": bool(
                getattr(args, "require_global_id_for_alert", False)
                or safe_bool(policy_cfg.get("require_global_id_for_alert", False))
            ),

            # Middle-ground safety path:
            # allow a very strong gid=None event to alert, instead of fully blocking all gid-less alerts.
            "allow_gid_none_strong_alert": bool(
                getattr(args, "allow_gid_none_strong_alert", False)
                or safe_bool(policy_cfg.get("allow_gid_none_strong_alert", False))
            ),
            "gid_none_strong_score_thr": float(
                policy_cfg.get("gid_none_strong_score_thr", getattr(args, "gid_none_strong_score_thr", 0.97))
            ),
            "gid_none_min_votes": int(
                policy_cfg.get("gid_none_min_votes", getattr(args, "gid_none_min_votes", 2))
            ),
            "gid_none_require_suspicion": bool(
                getattr(args, "gid_none_require_suspicion", False)
                or safe_bool(policy_cfg.get("gid_none_require_suspicion", False))
            ),
            "gid_none_min_suspicion": float(
                policy_cfg.get("gid_none_min_suspicion", getattr(args, "gid_none_min_suspicion", 0.70))
            ),

            "score_field": str(policy_cfg.get("score_field", getattr(args, "score_field", "score"))),
            "prefer_fused_if_present": bool(
                getattr(args, "prefer_fused_if_present", False)
                or safe_bool(policy_cfg.get("prefer_fused_if_present", False))
            ),
        }

        if self.policy["K"] > self.policy["M"]:
            print(
                f"[policy_node] correcting invalid K>M: K={self.policy['K']} M={self.policy['M']}"
            )
            self.policy["K"] = self.policy["M"]

        def make_score_deque():
            return deque(maxlen=self.policy["M"])

        self.make_score_deque = make_score_deque
        self.score_win = defaultdict(self.make_score_deque)
        self.last_seen = {}
        self.last_alert_ts = {}

        self.S = defaultdict(float)
        self.S_persist = defaultdict(int)
        self.last_s_update_ts = defaultdict(float)

        self.poller = zmq.Poller()
        self.poller.register(self.sub_scores, zmq.POLLIN)

        self.loop_n = 0

        print(f"[policy_node] cam_id={self.cam_id}")
        print(f"[policy_node] scores_sub={self.scores_connect} topic={self.scores_topic}")
        print(f"[policy_node] decisions_pub={self.decisions_bind} topic={self.decisions_topic}")
        print(f"[policy_node] alerts_pub={self.alerts_bind} topic={self.alerts_topic}")
        print(f"[policy_node] policy_updates_stream={self.policy_updates_stream}")
        print(f"[policy_node] loaded_policy_cfg={policy_cfg}")
        print(f"[policy_node] policy={self.policy}")

    def _jcompact(self, obj) -> str:
        return json.dumps(obj, separators=(",", ":"))

    def entity_key_for(self, cam, pid, gid):
        if self.policy["use_global_identity_state"] and gid is not None:
            return f"gid:{int(gid)}"
        return f"cam:{cam}:pid:{int(pid)}"

    def emit_decision(self, payload: dict):
        header = {
            "type": "decision",
            "event_id": payload.get("event_id"),
            "cam_id": payload.get("cam_id"),
            "person_track_id": payload.get("person_track_id"),
            "global_person_id": payload.get("global_person_id"),
            "frame_id": payload.get("frame_id_end"),
            "stamp_ns": payload.get("stamp_ns_end", 0),
        }
        self.pub_decisions.send_multipart([
            self.decisions_topic_b,
            self._jcompact(header).encode("utf-8"),
            self._jcompact(payload).encode("utf-8"),
        ])

    def emit_alert(self, payload: dict):
        header = {
            "type": "alert",
            "event_id": payload.get("event_id"),
            "cam_id": payload.get("cam_id"),
            "person_track_id": payload.get("person_track_id"),
            "global_person_id": payload.get("global_person_id"),
            "frame_id": payload.get("frame_id_end"),
            "stamp_ns": payload.get("stamp_ns_end", 0),
        }
        self.pub_alerts.send_multipart([
            self.alerts_topic_b,
            self._jcompact(header).encode("utf-8"),
            self._jcompact(payload).encode("utf-8"),
        ])

    def prune_state(self):
        now = time.time()
        ttl_s = float(getattr(self.args, "state_ttl_s", 120.0))
        drop = [ek for ek, ts in self.last_seen.items() if (now - ts) > ttl_s]

        for ek in drop:
            self.score_win.pop(ek, None)
            self.last_seen.pop(ek, None)
            self.last_alert_ts.pop(ek, None)
            self.S.pop(ek, None)
            self.S_persist.pop(ek, None)
            self.last_s_update_ts.pop(ek, None)

    def resize_windows_if_needed(self, old_M: int, new_M: int):
        if old_M == new_M:
            return

        for ek, dq in list(self.score_win.items()):
            kept = list(dq)[-new_M:]
            self.score_win[ek] = deque(kept, maxlen=new_M)

    def apply_policy_update(self, update_obj: dict):
        target = update_obj.get("target", None)
        accept_target = getattr(self.args, "policy_target", None) or self.cam_id

        if target not in (None, "all", accept_target):
            return False, f"ignored update target={target} accept={accept_target}"

        old_M = self.policy["M"]

        alias_map = {
            "recommended_threshold": "threshold",
            "carry_thr": "vote_carry_score_thr",
            "visibility_drop_thr": "vote_visibility_drop_thr",
            "contact_ratio_thr": "vote_contact_ratio_thr",
            "heuristic_thr": "heuristic_vote_thr",
            "susp_gain": "suspicion_gain",
        }

        for raw_key, raw_val in update_obj.items():
            map_key = alias_map.get(raw_key, raw_key)
            if map_key not in self.policy:
                continue

            cur = self.policy.get(map_key)

            if isinstance(cur, bool):
                self.policy[map_key] = safe_bool(raw_val, cur)
            elif isinstance(cur, int):
                v = safe_int(raw_val, None)
                if v is not None:
                    self.policy[map_key] = int(v)
            else:
                v = safe_float(raw_val, None)
                if v is not None:
                    self.policy[map_key] = float(v)

        new_M = self.policy["M"]
        self.resize_windows_if_needed(old_M, new_M)

        if self.policy["K"] > self.policy["M"]:
            self.policy["K"] = self.policy["M"]

        return True, f"applied policy={self.policy}"

    def compute_votes(self, policy_feats: dict):
        votes = 0
        reasons = []

        if not isinstance(policy_feats, dict):
            return 0, reasons

        if self.policy["vote_disappeared_after_contact"]:
            if bool(policy_feats.get("disappeared_after_contact", False)):
                votes += 1
                reasons.append("vote_disappeared_after_contact")

        cr = safe_float(policy_feats.get("contact_ratio", None), None)
        if cr is not None and cr >= self.policy["vote_contact_ratio_thr"]:
            votes += 1
            reasons.append("vote_contact_ratio")

        vd = safe_float(policy_feats.get("visibility_drop", None), None)
        if vd is not None and vd >= self.policy["vote_visibility_drop_thr"]:
            votes += 1
            reasons.append("vote_visibility_drop")

        cs = safe_float(policy_feats.get("carry_score_max", None), None)
        if cs is not None and cs >= self.policy["vote_carry_score_thr"]:
            votes += 1
            reasons.append("vote_carry_score")

        if self.policy["vote_use_heuristic_score"]:
            hs = safe_float(policy_feats.get("heuristic_theft_score", None), None)
            if hs is not None and hs >= self.policy["heuristic_vote_thr"]:
                votes += 1
                reasons.append("vote_heuristic_theft_score")

        return votes, reasons

    def select_score(self, obj: dict):
        if self.policy["prefer_fused_if_present"]:
            sf = safe_float(obj.get("score_fused", None), None)
            if sf is not None:
                return sf, "score_fused"

        preferred = str(self.policy["score_field"])
        if preferred in obj:
            v = safe_float(obj.get(preferred, None), None)
            if v is not None:
                return v, preferred

        for k in ["score", "score_fused", "score_fused_raw", "score_cnn", "score_mlp"]:
            v = safe_float(obj.get(k, None), None)
            if v is not None:
                return v, k

        return None, None

    def update_suspicion(self, ek: str, score: float, votes: int, policy_feats: dict):
        now = time.time()
        last_t = self.last_s_update_ts.get(ek, now)
        dt = max(0.0, now - last_t)

        clip_dt = safe_float(policy_feats.get("clip_dt_s", None), None) if isinstance(policy_feats, dict) else None
        if clip_dt is not None and clip_dt > 1e-3:
            dt = clip_dt

        tau = max(0.25, float(self.policy["suspicion_tau_s"]))
        decay = math.exp(-dt / tau)

        thr = float(self.policy["threshold"])
        norm_score = 0.0
        if score is not None:
            norm_score = clamp01((float(score) - thr) / max(1e-6, (1.0 - thr)))

        mv = max(1, int(self.policy["min_votes_to_alert"]))
        norm_votes = clamp01(float(votes) / float(mv))

        evidence = (
            float(self.policy["suspicion_model_weight"]) * norm_score
            + float(self.policy["suspicion_votes_weight"]) * norm_votes
        )

        bump = 0.0
        if isinstance(policy_feats, dict) and bool(policy_feats.get("disappeared_after_contact", False)):
            bump = 0.20

        evidence = clamp01(evidence + bump)

        prevS = float(self.S.get(ek, 0.0))
        newS = clamp01(prevS * decay + float(self.policy["suspicion_gain"]) * evidence)

        self.S[ek] = newS
        self.last_s_update_ts[ek] = now

        dbg = {
            "dt_s": float(dt),
            "decay": float(decay),
            "norm_score": float(norm_score),
            "norm_votes": float(norm_votes),
            "evidence": float(evidence),
            "prevS": float(prevS),
            "newS": float(newS),
        }
        return newS, dbg

    def pull_updates(self):
        if self.rdb is None or self.policy_updates_stream is None:
            return

        if int(getattr(self.args, "check_updates_every_loops", 1)) <= 0:
            return

        if (self.loop_n % int(getattr(self.args, "check_updates_every_loops", 1))) != 0:
            return

        try:
            upd_streams = self.rdb.xread(
                {self.policy_updates_stream: self.last_upd_id},
                block=int(getattr(self.args, "block_ms", 1000)),
                count=10,
            )
        except Exception as e:
            print(f"[policy_node] policy_updates Redis error: {e}")
            return

        if upd_streams:
            for mid, fields in parse_xread_messages(upd_streams):
                self.last_upd_id = mid
                update_obj = {}

                if b"json" in fields:
                    try:
                        update_obj = json.loads(fields[b"json"].decode())
                    except Exception:
                        update_obj = {}
                else:
                    for k, v in fields.items():
                        kk = k.decode() if isinstance(k, bytes) else str(k)
                        update_obj[kk] = b2s(v)

                ok, msg = self.apply_policy_update(update_obj)
                print(f"[policy_node] policy_update mid={mid} ok={ok} msg={msg}")

    def _handle_score_obj(self, obj: dict):
        cam = obj.get("cam_id", self.cam_id)
        pid = safe_int(obj.get("person_track_id", -1), -1)
        gid = safe_int(obj.get("global_person_id", None), None)
        fid_end = safe_int(obj.get("frame_id_end", -1), -1)
        stamp_ns_end = safe_int(obj.get("stamp_ns_end", 0), 0)

        score, score_source = self.select_score(obj)
        event_id = obj.get("event_id") or f"{cam}:{pid}:{fid_end}:{stamp_ns_end}"

        if pid < 0 or fid_end < 0 or score is None:
            return

        miss_pose = safe_float(obj.get("missing_pose_ratio", 0.0), 0.0)
        miss_obj = safe_float(obj.get("missing_obj_ratio", 0.0), 0.0)
        scalar_missing = bool(obj.get("scalar_missing", False))

        gate_ok = True
        gate_reasons = []

        if miss_pose is not None and miss_pose > self.policy["max_missing_pose_ratio"]:
            gate_ok = False
            gate_reasons.append("pose_missing_high")

        if miss_obj is not None and miss_obj > self.policy["max_missing_obj_ratio"]:
            gate_ok = False
            gate_reasons.append("obj_missing_high")

        if self.policy["drop_if_scalar_missing"] and scalar_missing:
            gate_ok = False
            gate_reasons.append("scalar_missing")

        gid_missing = gid is None

        if self.policy["require_global_id_for_alert"] and gid_missing:
            # Middle-ground behavior:
            # Do not hard-block gate_ok here.
            # Normal alerts will be suppressed later, but emergency strong gid-less alerts can still pass.
            gate_reasons.append("missing_global_id")

        ek = self.entity_key_for(cam, pid, gid)

        self.score_win[ek].append(float(score))
        self.last_seen[ek] = time.time()

        win = list(self.score_win[ek])
        hits = sum(1 for s in win if s >= self.policy["threshold"])
        k_of_m_ok = (hits >= self.policy["K"]) and (len(win) >= self.policy["M"])

        now = time.time()
        last_a = self.last_alert_ts.get(ek, 0.0)
        cd_ok = (now - last_a) >= self.policy["cooldown_s"]

        policy_feats = obj.get("policy_features", {}) or {}
        votes, vote_reasons = self.compute_votes(policy_feats)

        S_val = None
        S_dbg = {}
        suspicion_ok = False

        if self.policy["use_suspicion_policy"]:
            S_val, S_dbg = self.update_suspicion(ek, float(score), int(votes), policy_feats)

            if S_val >= self.policy["suspicion_alert_thr"]:
                self.S_persist[ek] = int(self.S_persist.get(ek, 0)) + 1
            else:
                self.S_persist[ek] = 0

            suspicion_ok = (
                (S_val >= self.policy["suspicion_alert_thr"])
                and (self.S_persist[ek] >= self.policy["suspicion_persist_clips"])
                and (votes >= self.policy["min_votes_to_alert"])
            )

        trigger_classic = bool(gate_ok and k_of_m_ok and cd_ok)
        trigger_suspicion = bool(gate_ok and suspicion_ok and cd_ok)

        normal_alert_blocked_by_gid = bool(
            self.policy["require_global_id_for_alert"] and gid_missing
        )

        gidless_score_ok = score is not None and float(score) >= float(
            self.policy["gid_none_strong_score_thr"]
        )

        gidless_votes_ok = int(votes) >= int(self.policy["gid_none_min_votes"])

        gidless_suspicion_ok = (
            not self.policy["gid_none_require_suspicion"]
            or (
                S_val is not None
                and float(S_val) >= float(self.policy["gid_none_min_suspicion"])
            )
        )

        trigger_gidless_strong = bool(
            normal_alert_blocked_by_gid
            and self.policy["allow_gid_none_strong_alert"]
            and gate_ok
            and cd_ok
            and gidless_score_ok
            and gidless_votes_ok
            and gidless_suspicion_ok
        )

        trigger_normal = bool(
            (trigger_classic or trigger_suspicion)
            and not normal_alert_blocked_by_gid
        )

        trigger = bool(trigger_normal or trigger_gidless_strong)

        emitted_at_ns = time.time_ns()

        decision = {
            "type": "decision",
            "event_id": event_id,
            "cam_id": cam,
            "person_track_id": pid,
            "global_person_id": gid,
            "entity_key": ek,
            "frame_id_end": fid_end,
            "stamp_ns_end": stamp_ns_end,
            "emitted_at_ns": emitted_at_ns,

            "score": float(score),
            "score_source": score_source,
            "score_fused": safe_float(obj.get("score_fused", None), None),
            "score_fused_raw": safe_float(obj.get("score_fused_raw", None), None),
            "score_cnn": safe_float(obj.get("score_cnn", None), None),
            "score_mlp": safe_float(obj.get("score_mlp", None), None),
            "score_identity_memory": safe_float(obj.get("score_identity_memory", None), None),

            "threshold": float(self.policy["threshold"]),
            "window_M": int(self.policy["M"]),
            "required_K": int(self.policy["K"]),
            "hits_in_window": int(hits),
            "window_len": int(len(win)),
            "gate_ok": bool(gate_ok),
            "gate_reasons": gate_reasons,
            "cooldown_ok": bool(cd_ok),
            "cooldown_s": float(self.policy["cooldown_s"]),

            "votes": int(votes),
            "vote_reasons": vote_reasons,

            "use_suspicion_policy": bool(self.policy["use_suspicion_policy"]),
            "S": float(S_val) if S_val is not None else None,
            "S_persist": int(self.S_persist.get(ek, 0)),
            "S_debug": S_dbg,

            "will_alert": bool(trigger),
            "trigger_classic": bool(trigger_classic),
            "trigger_suspicion": bool(trigger_suspicion),
            "trigger_gidless_strong": bool(trigger_gidless_strong),
            "trigger_normal": bool(trigger_normal),
            "normal_alert_blocked_by_gid": bool(normal_alert_blocked_by_gid),

            "gidless_score_ok": bool(gidless_score_ok),
            "gidless_votes_ok": bool(gidless_votes_ok),
            "gidless_suspicion_ok": bool(gidless_suspicion_ok),

            "model_version": obj.get("model_version", "unknown"),
            "object_track_id": obj.get("object_track_id", -1),
            "object_class_id": obj.get("object_class_id", -1),

            # Clip reference info from model_node.
            "clip_path": obj.get("clip_path"),
            "clip_codec": obj.get("clip_codec"),
            "local_clip_path": obj.get("local_clip_path", obj.get("clip_path")),

            "missing_pose_ratio": miss_pose,
            "missing_obj_ratio": miss_obj,
            "scalar_missing": scalar_missing,

            "policy_features": policy_feats,
            "identity_features": obj.get("identity_features", {}),

            "incident_candidate": bool(trigger or (S_val is not None and S_val >= 0.50)),
            "incident_builder_expected": True,
        }

        self.emit_decision(decision)

        if trigger:
            self.last_alert_ts[ek] = now

            alert_policy = "gidless_strong" if trigger_gidless_strong else (
                "classic" if trigger_classic else "suspicion"
            )

            alert = {
                "type": "alert",
                "alert_scope": "local_preincident",
                "event_id": event_id,
                "cam_id": cam,
                "person_track_id": pid,
                "global_person_id": gid,
                "entity_key": ek,
                "frame_id_end": fid_end,
                "stamp_ns_end": stamp_ns_end,
                "emitted_at_ns": emitted_at_ns,

                "score": float(score),
                "score_source": score_source,
                "score_fused": safe_float(obj.get("score_fused", None), None),
                "score_fused_raw": safe_float(obj.get("score_fused_raw", None), None),
                "score_cnn": safe_float(obj.get("score_cnn", None), None),
                "score_mlp": safe_float(obj.get("score_mlp", None), None),
                "score_identity_memory": safe_float(obj.get("score_identity_memory", None), None),

                "suspicion": float(S_val) if S_val is not None else float(score),
                "reason": {
                    "policy": alert_policy,
                    "hits_in_window": int(hits),
                    "window_M": int(self.policy["M"]),
                    "required_K": int(self.policy["K"]),
                    "threshold": float(self.policy["threshold"]),
                    "gate_reasons": gate_reasons,
                    "votes": int(votes),
                    "vote_reasons": vote_reasons,
                    "S": float(S_val) if S_val is not None else None,
                    "S_persist": int(self.S_persist.get(ek, 0)),
                    "trigger_classic": bool(trigger_classic),
                    "trigger_suspicion": bool(trigger_suspicion),
                    "trigger_gidless_strong": bool(trigger_gidless_strong),
                },

                "model_version": obj.get("model_version", "unknown"),
                "object_track_id": obj.get("object_track_id", -1),
                "object_class_id": obj.get("object_class_id", -1),

                # Clip reference info from model_node.
                "clip_path": obj.get("clip_path"),
                "clip_codec": obj.get("clip_codec"),
                "local_clip_path": obj.get("local_clip_path", obj.get("clip_path")),

                "policy_features": policy_feats,
                "identity_features": obj.get("identity_features", {}),
                "incident_candidate": True,
                "incident_builder_expected": True,
            }

            self.emit_alert(alert)

            print(
                f"[policy_node] ALERT cam={cam} pid={pid} gid={gid} fid_end={fid_end} "
                f"score={score:.3f} src={score_source} policy={alert_policy} "
                f"classic={trigger_classic} suspicion={trigger_suspicion} "
                f"gidless_strong={trigger_gidless_strong} votes={votes} "
                f"S={S_val} gate_reasons={gate_reasons}"
            )

    def pull_scores(self):
        events = dict(self.poller.poll(timeout=int(getattr(self.args, "block_ms", 1000))))

        if self.sub_scores not in events:
            self.prune_state()
            return

        drained = 0
        max_drain = int(getattr(self.args, "max_drain_per_step", 64))

        while drained < max_drain:
            try:
                parts = self.sub_scores.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                print(f"[policy_node] scores ZMQ error: {e}")
                break

            drained += 1

            if len(parts) != 3:
                continue

            try:
                _, _, payload_b = parts
                obj = json.loads(payload_b.decode("utf-8"))
            except Exception:
                continue

            self._handle_score_obj(obj)

        self.prune_state()

    def step(self):
        self.loop_n += 1
        self.pull_updates()
        self.pull_scores()

    def run_forever(self, sleep_s=0.001):
        try:
            while True:
                self.step()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\n[policy_node] stopping...")
        finally:
            for sock in [self.sub_scores, self.pub_decisions, self.pub_alerts]:
                try:
                    sock.close(0)
                except Exception:
                    pass


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--scores_connect", default=None)
    ap.add_argument("--scores_topic", default=None)
    ap.add_argument("--decisions_bind", default=None)
    ap.add_argument("--decisions_topic", default=None)
    ap.add_argument("--alerts_bind", default=None)
    ap.add_argument("--alerts_topic", default=None)

    ap.add_argument("--policy_updates_stream", default=None)
    ap.add_argument("--policy_target", default=None)
    ap.add_argument("--disable_redis_updates", action="store_true")

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--zmq_rcvhwm", type=int, default=256)
    ap.add_argument("--max_drain_per_step", type=int, default=64)

    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--cooldown_s", type=float, default=10.0)

    ap.add_argument("--max_missing_pose_ratio", type=float, default=0.55)
    ap.add_argument("--max_missing_obj_ratio", type=float, default=0.75)
    ap.add_argument("--drop_if_scalar_missing", action="store_true")

    ap.add_argument("--use_suspicion_policy", action="store_true")
    ap.add_argument("--suspicion_tau_s", type=float, default=6.0)
    ap.add_argument("--suspicion_alert_thr", type=float, default=0.85)
    ap.add_argument("--suspicion_persist_clips", type=int, default=2)
    ap.add_argument("--suspicion_gain", type=float, default=0.55)
    ap.add_argument("--suspicion_model_weight", type=float, default=0.55)
    ap.add_argument("--suspicion_votes_weight", type=float, default=0.45)

    ap.add_argument("--min_votes_to_alert", type=int, default=1)
    ap.add_argument("--vote_use_heuristic_score", action="store_true")
    ap.add_argument("--heuristic_vote_thr", type=float, default=0.80)

    ap.add_argument("--vote_contact_ratio_thr", type=float, default=0.35)
    ap.add_argument("--vote_visibility_drop_thr", type=float, default=0.25)
    ap.add_argument("--vote_carry_score_thr", type=float, default=0.70)
    ap.add_argument("--vote_disappeared_after_contact", action="store_true")

    ap.add_argument("--state_ttl_s", type=float, default=120.0)
    ap.add_argument("--check_updates_every_loops", type=int, default=1)

    ap.add_argument("--use_global_identity_state", action="store_true")
    ap.add_argument("--require_global_id_for_alert", action="store_true")
    ap.add_argument("--allow_gid_none_strong_alert", action="store_true")
    ap.add_argument("--gid_none_strong_score_thr", type=float, default=0.97)
    ap.add_argument("--gid_none_min_votes", type=int, default=2)
    ap.add_argument("--gid_none_require_suspicion", action="store_true")
    ap.add_argument("--gid_none_min_suspicion", type=float, default=0.70)

    ap.add_argument("--score_field", type=str, default="score")
    ap.add_argument("--prefer_fused_if_present", action="store_true")

    return ap


def main():
    args = build_arg_parser().parse_args()
    worker = PolicyWorker(args)
    worker.run_forever()


if __name__ == "__main__":
    main()
