#!/usr/bin/env python3
"""
policy_node.py
(policy + local alert emitter, split-score + global identity aware)

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

import redis

from config_utils import load_cfg, get_stream


def parse_xread_messages(streams):
    out = []
    for _sname, msgs in streams:
        for mid, fields in msgs:
            out.append((mid.decode(), fields))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--scores_stream", default=None)
    ap.add_argument("--alerts_stream", default=None)
    ap.add_argument("--decisions_stream", default=None)

    ap.add_argument("--policy_updates_stream", default=None)
    ap.add_argument("--policy_target", default=None)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=50)

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
    ap.add_argument("--redis_maxlen", type=int, default=20000)
    ap.add_argument("--check_updates_every_loops", type=int, default=1)

    ap.add_argument("--use_global_identity_state", action="store_true")
    ap.add_argument("--score_field", type=str, default="score")
    ap.add_argument("--prefer_fused_if_present", action="store_true")

    args = ap.parse_args()
    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
    )
    rdb.ping()

    scores_stream = args.scores_stream or get_stream(cfg, "scores", cam_id)
    alerts_stream = args.alerts_stream or get_stream(cfg, "alerts", cam_id)
    decisions_stream = args.decisions_stream or get_stream(cfg, "decisions", cam_id)
    policy_updates_stream = args.policy_updates_stream or get_stream(cfg, "policy_updates")

    policy_cfg = cfg.get("policy", {})

    policy = {
        "threshold": float(args.threshold),
        "M": int(args.M),
        "K": int(args.K),
        "cooldown_s": float(args.cooldown_s),
        "max_missing_pose_ratio": float(args.max_missing_pose_ratio),
        "max_missing_obj_ratio": float(args.max_missing_obj_ratio),
        "drop_if_scalar_missing": bool(args.drop_if_scalar_missing),

        "use_suspicion_policy": bool(args.use_suspicion_policy or safe_bool(policy_cfg.get("use_suspicion_policy", False))),
        "suspicion_tau_s": float(policy_cfg.get("suspicion_tau_s", args.suspicion_tau_s)),
        "suspicion_alert_thr": float(policy_cfg.get("suspicion_alert_thr", args.suspicion_alert_thr)),
        "suspicion_persist_clips": int(policy_cfg.get("suspicion_persist_clips", args.suspicion_persist_clips)),
        "suspicion_gain": float(policy_cfg.get("suspicion_gain", args.suspicion_gain)),
        "suspicion_model_weight": float(policy_cfg.get("suspicion_model_weight", args.suspicion_model_weight)),
        "suspicion_votes_weight": float(policy_cfg.get("suspicion_votes_weight", args.suspicion_votes_weight)),

        "min_votes_to_alert": int(policy_cfg.get("min_votes_to_alert", args.min_votes_to_alert)),
        "vote_use_heuristic_score": bool(args.vote_use_heuristic_score or safe_bool(policy_cfg.get("vote_use_heuristic_score", False))),
        "heuristic_vote_thr": float(policy_cfg.get("heuristic_vote_thr", args.heuristic_vote_thr)),
        "vote_contact_ratio_thr": float(policy_cfg.get("vote_contact_ratio_thr", args.vote_contact_ratio_thr)),
        "vote_visibility_drop_thr": float(policy_cfg.get("vote_visibility_drop_thr", args.vote_visibility_drop_thr)),
        "vote_carry_score_thr": float(policy_cfg.get("vote_carry_score_thr", args.vote_carry_score_thr)),
        "vote_disappeared_after_contact": bool(
            args.vote_disappeared_after_contact or safe_bool(policy_cfg.get("vote_disappeared_after_contact", False))
        ),

        "use_global_identity_state": bool(
            args.use_global_identity_state or safe_bool(policy_cfg.get("use_global_identity_state", False))
        ),
        "score_field": str(policy_cfg.get("score_field", args.score_field)),
        "prefer_fused_if_present": bool(
            args.prefer_fused_if_present or safe_bool(policy_cfg.get("prefer_fused_if_present", False))
        ),
    }

    def make_score_deque():
        return deque(maxlen=policy["M"])

    score_win = defaultdict(make_score_deque)
    last_seen = {}
    last_alert_ts = {}

    S = defaultdict(float)
    S_persist = defaultdict(int)
    last_s_update_ts = defaultdict(float)

    last_score_id = "0-0"
    last_upd_id = "0-0"
    loop_n = 0

    def entity_key_for(cam, pid, gid):
        if policy["use_global_identity_state"] and gid is not None:
            return f"gid:{int(gid)}"
        return f"cam:{cam}:pid:{int(pid)}"

    def emit_decision(payload: dict):
        rdb.xadd(
            decisions_stream,
            {
                "event_id": payload["event_id"],
                "cam_id": payload["cam_id"],
                "person_track_id": str(payload["person_track_id"]),
                "global_person_id": "" if payload.get("global_person_id") is None else str(payload["global_person_id"]),
                "frame_id": str(payload["frame_id_end"]),
                "stamp_ns": str(payload.get("stamp_ns_end", 0)),
                "json": json.dumps(payload),
            },
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def emit_alert(payload: dict):
        rdb.xadd(
            alerts_stream,
            {
                "event_id": payload["event_id"],
                "cam_id": payload["cam_id"],
                "person_track_id": str(payload["person_track_id"]),
                "global_person_id": "" if payload.get("global_person_id") is None else str(payload["global_person_id"]),
                "frame_id": str(payload["frame_id_end"]),
                "stamp_ns": str(payload.get("stamp_ns_end", 0)),
                "json": json.dumps(payload),
            },
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def prune_state():
        now = time.time()
        drop = [ek for ek, ts in last_seen.items() if (now - ts) > args.state_ttl_s]
        for ek in drop:
            score_win.pop(ek, None)
            last_seen.pop(ek, None)
            last_alert_ts.pop(ek, None)
            S.pop(ek, None)
            S_persist.pop(ek, None)
            last_s_update_ts.pop(ek, None)

    def resize_windows_if_needed(old_M: int, new_M: int):
        if old_M == new_M:
            return
        for ek, dq in list(score_win.items()):
            kept = list(dq)[-new_M:]
            score_win[ek] = deque(kept, maxlen=new_M)

    def apply_policy_update(update_obj: dict):
        target = update_obj.get("target", None)
        accept_target = args.policy_target or cam_id
        if target not in (None, "all", accept_target):
            return False, f"ignored update target={target} accept={accept_target}"

        old_M = policy["M"]

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
            if map_key not in policy:
                continue

            cur = policy.get(map_key)

            if isinstance(cur, bool):
                policy[map_key] = safe_bool(raw_val, cur)
            elif isinstance(cur, int):
                v = safe_int(raw_val, None)
                if v is not None:
                    policy[map_key] = int(v)
            else:
                v = safe_float(raw_val, None)
                if v is not None:
                    policy[map_key] = float(v)

        new_M = policy["M"]
        resize_windows_if_needed(old_M, new_M)

        if policy["K"] > policy["M"]:
            policy["K"] = policy["M"]

        return True, f"applied policy={policy}"

    def compute_votes(policy_feats: dict):
        votes = 0
        reasons = []

        if not isinstance(policy_feats, dict):
            return 0, reasons

        if policy["vote_disappeared_after_contact"]:
            if bool(policy_feats.get("disappeared_after_contact", False)):
                votes += 1
                reasons.append("vote_disappeared_after_contact")

        cr = safe_float(policy_feats.get("contact_ratio", None), None)
        if cr is not None and cr >= policy["vote_contact_ratio_thr"]:
            votes += 1
            reasons.append("vote_contact_ratio")

        vd = safe_float(policy_feats.get("visibility_drop", None), None)
        if vd is not None and vd >= policy["vote_visibility_drop_thr"]:
            votes += 1
            reasons.append("vote_visibility_drop")

        cs = safe_float(policy_feats.get("carry_score_max", None), None)
        if cs is not None and cs >= policy["vote_carry_score_thr"]:
            votes += 1
            reasons.append("vote_carry_score")

        if policy["vote_use_heuristic_score"]:
            hs = safe_float(policy_feats.get("heuristic_theft_score", None), None)
            if hs is not None and hs >= policy["heuristic_vote_thr"]:
                votes += 1
                reasons.append("vote_heuristic_theft_score")

        return votes, reasons

    def select_score(obj: dict):
        if policy["prefer_fused_if_present"]:
            sf = safe_float(obj.get("score_fused", None), None)
            if sf is not None:
                return sf, "score_fused"

        preferred = str(policy["score_field"])
        if preferred in obj:
            v = safe_float(obj.get(preferred, None), None)
            if v is not None:
                return v, preferred

        for k in ["score", "score_fused", "score_fused_raw", "score_cnn", "score_mlp"]:
            v = safe_float(obj.get(k, None), None)
            if v is not None:
                return v, k

        return None, None

    def update_suspicion(ek: str, score: float, votes: int, policy_feats: dict):
        now = time.time()
        last_t = last_s_update_ts.get(ek, now)
        dt = max(0.0, now - last_t)

        clip_dt = safe_float(policy_feats.get("clip_dt_s", None), None) if isinstance(policy_feats, dict) else None
        if clip_dt is not None and clip_dt > 1e-3:
            dt = clip_dt

        tau = max(0.25, float(policy["suspicion_tau_s"]))
        decay = math.exp(-dt / tau)

        thr = float(policy["threshold"])
        norm_score = 0.0
        if score is not None:
            norm_score = clamp01((float(score) - thr) / max(1e-6, (1.0 - thr)))

        mv = max(1, int(policy["min_votes_to_alert"]))
        norm_votes = clamp01(float(votes) / float(mv))

        evidence = (
            float(policy["suspicion_model_weight"]) * norm_score +
            float(policy["suspicion_votes_weight"]) * norm_votes
        )

        bump = 0.0
        if isinstance(policy_feats, dict) and bool(policy_feats.get("disappeared_after_contact", False)):
            bump = 0.20

        evidence = clamp01(evidence + bump)

        prevS = float(S.get(ek, 0.0))
        newS = clamp01(prevS * decay + float(policy["suspicion_gain"]) * evidence)

        S[ek] = newS
        last_s_update_ts[ek] = now

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

    try:
        while True:
            loop_n += 1

            if args.check_updates_every_loops > 0 and (loop_n % args.check_updates_every_loops == 0):
                upd_streams = rdb.xread({policy_updates_stream: last_upd_id}, block=1, count=10)
                if upd_streams:
                    for mid, fields in parse_xread_messages(upd_streams):
                        last_upd_id = mid
                        update_obj = {}
                        if b"json" in fields:
                            try:
                                update_obj = json.loads(fields[b"json"].decode())
                            except Exception:
                                update_obj = {}
                        else:
                            for k, v in fields.items():
                                update_obj[k.decode()] = b2s(v)

                        ok, msg = apply_policy_update(update_obj)
                        print(f"[policy_node] policy_update mid={mid} ok={ok} msg={msg}")

            streams = rdb.xread({scores_stream: last_score_id}, block=args.block_ms, count=args.count)
            if not streams:
                prune_state()
                continue

            for mid, fields in parse_xread_messages(streams):
                last_score_id = mid

                js = fields.get(b"json", b"{}")
                try:
                    obj = json.loads(b2s(js))
                except Exception:
                    continue

                cam = obj.get("cam_id", cam_id)
                pid = safe_int(obj.get("person_track_id", -1), -1)
                gid = safe_int(obj.get("global_person_id", None), None)
                fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                stamp_ns_end = safe_int(obj.get("stamp_ns_end", 0), 0)

                score, score_source = select_score(obj)
                event_id = obj.get("event_id") or f"{cam}:{pid}:{fid_end}:{stamp_ns_end}"

                if pid < 0 or fid_end < 0 or score is None:
                    continue

                miss_pose = safe_float(obj.get("missing_pose_ratio", 0.0), 0.0)
                miss_obj = safe_float(obj.get("missing_obj_ratio", 0.0), 0.0)
                scalar_missing = bool(obj.get("scalar_missing", False))

                gate_ok = True
                gate_reasons = []
                if miss_pose is not None and miss_pose > policy["max_missing_pose_ratio"]:
                    gate_ok = False
                    gate_reasons.append("pose_missing_high")
                if miss_obj is not None and miss_obj > policy["max_missing_obj_ratio"]:
                    gate_ok = False
                    gate_reasons.append("obj_missing_high")
                if policy["drop_if_scalar_missing"] and scalar_missing:
                    gate_ok = False
                    gate_reasons.append("scalar_missing")

                ek = entity_key_for(cam, pid, gid)

                score_win[ek].append(float(score))
                last_seen[ek] = time.time()

                win = list(score_win[ek])
                hits = sum(1 for s in win if s >= policy["threshold"])
                k_of_m_ok = (hits >= policy["K"]) and (len(win) >= policy["M"])

                now = time.time()
                last_a = last_alert_ts.get(ek, 0.0)
                cd_ok = (now - last_a) >= policy["cooldown_s"]

                policy_feats = obj.get("policy_features", {}) or {}
                votes, vote_reasons = compute_votes(policy_feats)

                S_val = None
                S_dbg = {}
                suspicion_ok = False
                if policy["use_suspicion_policy"]:
                    S_val, S_dbg = update_suspicion(ek, float(score), int(votes), policy_feats)

                    if S_val >= policy["suspicion_alert_thr"]:
                        S_persist[ek] = int(S_persist.get(ek, 0)) + 1
                    else:
                        S_persist[ek] = 0

                    suspicion_ok = (
                        (S_val >= policy["suspicion_alert_thr"]) and
                        (S_persist[ek] >= policy["suspicion_persist_clips"]) and
                        (votes >= policy["min_votes_to_alert"])
                    )

                trigger_classic = bool(gate_ok and k_of_m_ok and cd_ok)
                trigger_suspicion = bool(gate_ok and suspicion_ok and cd_ok)
                trigger = bool(trigger_classic or trigger_suspicion)

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

                    "threshold": float(policy["threshold"]),
                    "window_M": int(policy["M"]),
                    "required_K": int(policy["K"]),
                    "hits_in_window": int(hits),
                    "window_len": int(len(win)),
                    "gate_ok": bool(gate_ok),
                    "gate_reasons": gate_reasons,
                    "cooldown_ok": bool(cd_ok),
                    "cooldown_s": float(policy["cooldown_s"]),
                    "votes": int(votes),
                    "vote_reasons": vote_reasons,
                    "use_suspicion_policy": bool(policy["use_suspicion_policy"]),
                    "S": float(S_val) if S_val is not None else None,
                    "S_persist": int(S_persist.get(ek, 0)),
                    "S_debug": S_dbg,
                    "will_alert": bool(trigger),
                    "trigger_classic": bool(trigger_classic),
                    "trigger_suspicion": bool(trigger_suspicion),
                    "model_version": obj.get("model_version", "unknown"),
                    "object_track_id": obj.get("object_track_id", -1),
                    "object_class_id": obj.get("object_class_id", -1),
                    "missing_pose_ratio": miss_pose,
                    "missing_obj_ratio": miss_obj,
                    "scalar_missing": scalar_missing,
                    "policy_features": policy_feats,
                    "identity_features": obj.get("identity_features", {}),
                    "incident_candidate": bool(trigger or (S_val is not None and S_val >= 0.50)),
                    "incident_builder_expected": True,
                }
                emit_decision(decision)

                if trigger:
                    last_alert_ts[ek] = now
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
                            "policy": "classic" if trigger_classic else "suspicion",
                            "hits_in_window": int(hits),
                            "window_M": int(policy["M"]),
                            "required_K": int(policy["K"]),
                            "threshold": float(policy["threshold"]),
                            "gate_reasons": gate_reasons,
                            "votes": int(votes),
                            "vote_reasons": vote_reasons,
                            "S": float(S_val) if S_val is not None else None,
                            "S_persist": int(S_persist.get(ek, 0)),
                        },
                        "model_version": obj.get("model_version", "unknown"),
                        "object_track_id": obj.get("object_track_id", -1),
                        "object_class_id": obj.get("object_class_id", -1),
                        "policy_features": policy_feats,
                        "identity_features": obj.get("identity_features", {}),
                        "incident_candidate": True,
                        "incident_builder_expected": True,
                    }
                    emit_alert(alert)
                    print(
                        f"[policy_node] ALERT cam={cam} pid={pid} gid={gid} fid_end={fid_end} "
                        f"score={score:.3f} src={score_source} classic={trigger_classic} "
                        f"suspicion={trigger_suspicion} votes={votes} S={S_val}"
                    )

            prune_state()

    except KeyboardInterrupt:
        print("\n[policy_node] stopping...")


if __name__ == "__main__":
    main()