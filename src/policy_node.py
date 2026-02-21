#!/usr/bin/env python3
"""
policy_node.py (policy + alert emitter)

Consumes:
  - Redis Stream: scores:<cam>  (raw scores per clip from model_node)

Produces:
  - Redis Stream: alerts:<cam>
  - Redis Stream: decisions:<cam>

Listens:
  - Redis Stream: policy_updates  (dynamic threshold/K/M/cooldown/gates updates)

Policy:
  - threshold
  - K-of-M smoothing
  - cooldown per person_track_id
  - quality gating
"""

import argparse
import json
import time
from collections import defaultdict, deque

import yaml
import redis


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    # Redis read
    ap.add_argument("--scores_stream", default=None)
    ap.add_argument("--alerts_stream", default=None)
    ap.add_argument("--decisions_stream", default=None)

    # NEW: policy updates stream
    ap.add_argument("--policy_updates_stream", default="policy_updates")
    ap.add_argument("--policy_target", default=None, help="If set, only accept updates for this target (e.g. cam0).")

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=50)

    # Core policy defaults
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--M", type=int, default=8, help="window size")
    ap.add_argument("--K", type=int, default=4, help="must have >=K hits in window")

    # Cooldown (seconds)
    ap.add_argument("--cooldown_s", type=float, default=10.0)

    # Quality gates
    ap.add_argument("--max_missing_pose_ratio", type=float, default=0.55)
    ap.add_argument("--max_missing_obj_ratio", type=float, default=0.75)
    ap.add_argument("--drop_if_scalar_missing", action="store_true")

    # Housekeeping
    ap.add_argument("--state_ttl_s", type=float, default=120.0, help="forget inactive person IDs after TTL")
    ap.add_argument("--redis_maxlen", type=int, default=20000)

    # How often to poll updates
    ap.add_argument("--check_updates_every_loops", type=int, default=1)

    args = ap.parse_args()
    cfg = load_cfg(args.config)
    cam_id = cfg["system"]["cam_id"]

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
    )
    rdb.ping()

    scores_stream = args.scores_stream or r_cfg.get("scores_stream", f"scores:{cam_id}")
    alerts_stream = args.alerts_stream or r_cfg.get("alerts_stream", f"alerts:{cam_id}")
    decisions_stream = args.decisions_stream or r_cfg.get("decisions_stream", f"decisions:{cam_id}")
    policy_updates_stream = args.policy_updates_stream or r_cfg.get("policy_updates_stream", "policy_updates")

    # Policy is now mutable (updated at runtime)
    policy = {
        "threshold": float(args.threshold),
        "M": int(args.M),
        "K": int(args.K),
        "cooldown_s": float(args.cooldown_s),
        "max_missing_pose_ratio": float(args.max_missing_pose_ratio),
        "max_missing_obj_ratio": float(args.max_missing_obj_ratio),
        "drop_if_scalar_missing": bool(args.drop_if_scalar_missing),
    }

    def make_score_deque():
        return deque(maxlen=policy["M"])

    score_win = defaultdict(make_score_deque)  # pid -> deque[float]
    last_seen = {}      # pid -> time.time()
    last_alert_ts = {}  # pid -> time.time()

    last_score_id = "0-0"
    last_upd_id = "0-0"
    loop_n = 0

    print(f"[policy_node] scores={scores_stream}")
    print(f"[policy_node] alerts={alerts_stream}")
    print(f"[policy_node] decisions={decisions_stream}")
    print(f"[policy_node] policy_updates={policy_updates_stream} policy_target={args.policy_target or cam_id}")
    print(f"[policy_node] initial policy={policy}")

    def emit_decision(payload: dict):
        rdb.xadd(
            decisions_stream,
            {
                "cam_id": payload["cam_id"],
                "person_track_id": str(payload["person_track_id"]),
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
                "cam_id": payload["cam_id"],
                "person_track_id": str(payload["person_track_id"]),
                "frame_id": str(payload["frame_id_end"]),
                "stamp_ns": str(payload.get("stamp_ns_end", 0)),
                "json": json.dumps(payload),
            },
            maxlen=args.redis_maxlen,
            approximate=True,
        )

    def prune_state():
        now = time.time()
        drop = [pid for pid, ts in last_seen.items() if (now - ts) > args.state_ttl_s]
        for pid in drop:
            score_win.pop(pid, None)
            last_seen.pop(pid, None)
            last_alert_ts.pop(pid, None)

    def resize_windows_if_needed(old_M: int, new_M: int):
        if old_M == new_M:
            return
        # Rebuild all deques with new maxlen, preserving most recent values
        for pid, dq in list(score_win.items()):
            kept = list(dq)[-new_M:]
            score_win[pid] = deque(kept, maxlen=new_M)

    def apply_policy_update(update_obj: dict):
        """
        Expected update_obj example (from trainer):
          {
            "type":"policy_update",
            "target":"cam0",
            "recommended_threshold":"0.82",
            "cooldown_s":"10",
            "K":"4",
            "M":"8",
            "max_missing_pose_ratio":"0.55",
            ...
          }
        """
        target = update_obj.get("target", None)
        accept_target = args.policy_target or cam_id

        # Accept if:
        # - update target matches cam
        # - or target is "all"
        # - or target missing (we accept)
        if target not in (None, "all", accept_target):
            return False, f"ignored update target={target} accept={accept_target}"

        old_M = policy["M"]

        # threshold
        thr = update_obj.get("recommended_threshold", None)
        if thr is None:
            thr = update_obj.get("threshold", None)
        if thr is not None:
            v = safe_float(thr, None)
            if v is not None:
                policy["threshold"] = float(max(0.01, min(0.99, v)))

        # K/M
        m = update_obj.get("M", None)
        k = update_obj.get("K", None)
        if m is not None:
            mv = safe_int(m, None)
            if mv is not None and mv >= 1:
                policy["M"] = int(mv)
        if k is not None:
            kv = safe_int(k, None)
            if kv is not None and kv >= 1:
                policy["K"] = int(kv)

        # cooldown
        cd = update_obj.get("cooldown_s", None)
        if cd is not None:
            cv = safe_float(cd, None)
            if cv is not None and cv >= 0:
                policy["cooldown_s"] = float(cv)

        # gates
        mp = update_obj.get("max_missing_pose_ratio", None)
        mo = update_obj.get("max_missing_obj_ratio", None)
        if mp is not None:
            v = safe_float(mp, None)
            if v is not None:
                policy["max_missing_pose_ratio"] = float(max(0.0, min(1.0, v)))
        if mo is not None:
            v = safe_float(mo, None)
            if v is not None:
                policy["max_missing_obj_ratio"] = float(max(0.0, min(1.0, v)))

        dsm = update_obj.get("drop_if_scalar_missing", None)
        if dsm is not None:
            # accept bool-like strings
            if isinstance(dsm, bool):
                policy["drop_if_scalar_missing"] = bool(dsm)
            else:
                s = str(dsm).strip().lower()
                policy["drop_if_scalar_missing"] = s in ("1", "true", "yes", "y")

        new_M = policy["M"]
        resize_windows_if_needed(old_M, new_M)

        # keep K in range
        if policy["K"] > policy["M"]:
            policy["K"] = policy["M"]

        return True, f"applied policy={policy}"

    try:
        while True:
            loop_n += 1

            # ---- 0) poll policy updates
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
                            # decode fields
                            for k, v in fields.items():
                                update_obj[k.decode()] = b2s(v)

                        ok, msg = apply_policy_update(update_obj)
                        print(f"[policy_node] policy_update mid={mid} ok={ok} msg={msg}")

            # ---- 1) consume scores
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
                fid_end = safe_int(obj.get("frame_id_end", -1), -1)
                stamp_ns_end = safe_int(obj.get("stamp_ns_end", 0), 0)

                score = safe_float(obj.get("score", None), None)
                if pid < 0 or fid_end < 0 or score is None:
                    continue

                # Quality fields
                miss_pose = safe_float(obj.get("missing_pose_ratio", 0.0), 0.0)
                miss_obj = safe_float(obj.get("missing_obj_ratio", 0.0), 0.0)
                scalar_missing = bool(obj.get("scalar_missing", False))

                # gates
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

                # update rolling window
                score_win[pid].append(float(score))
                last_seen[pid] = time.time()

                # K-of-M hits (use live policy values)
                win = list(score_win[pid])
                hits = sum(1 for s in win if s >= policy["threshold"])
                k_of_m_ok = (hits >= policy["K"]) and (len(win) >= policy["M"])

                # cooldown
                now = time.time()
                last_a = last_alert_ts.get(pid, 0.0)
                cd_ok = (now - last_a) >= policy["cooldown_s"]

                trigger = bool(gate_ok and k_of_m_ok and cd_ok)

                decision = {
                    "type": "decision",
                    "cam_id": cam,
                    "person_track_id": pid,
                    "frame_id_end": fid_end,
                    "stamp_ns_end": stamp_ns_end,

                    "score": float(score),
                    "threshold": float(policy["threshold"]),
                    "window_M": int(policy["M"]),
                    "required_K": int(policy["K"]),
                    "hits_in_window": int(hits),
                    "window_len": int(len(win)),

                    "gate_ok": bool(gate_ok),
                    "gate_reasons": gate_reasons,
                    "cooldown_ok": bool(cd_ok),
                    "cooldown_s": float(policy["cooldown_s"]),
                    "will_alert": bool(trigger),

                    "model_version": obj.get("model_version", "unknown"),
                    "object_track_id": obj.get("object_track_id", -1),
                    "object_class_id": obj.get("object_class_id", -1),
                    "missing_pose_ratio": miss_pose,
                    "missing_obj_ratio": miss_obj,
                    "scalar_missing": scalar_missing,
                }
                emit_decision(decision)

                if trigger:
                    last_alert_ts[pid] = now
                    alert = {
                        "type": "alert",
                        "cam_id": cam,
                        "person_track_id": pid,
                        "frame_id_end": fid_end,
                        "stamp_ns_end": stamp_ns_end,
                        "score": float(score),
                        "reason": {
                            "policy": "k_of_m_threshold",
                            "hits_in_window": int(hits),
                            "window_M": int(policy["M"]),
                            "required_K": int(policy["K"]),
                            "threshold": float(policy["threshold"]),
                            "gate_reasons": gate_reasons,
                        },
                        "model_version": obj.get("model_version", "unknown"),
                        "object_track_id": obj.get("object_track_id", -1),
                        "object_class_id": obj.get("object_class_id", -1),
                    }
                    emit_alert(alert)
                    print(f"[policy_node] ALERT cam={cam} pid={pid} fid_end={fid_end} score={score:.3f} hits={hits}/{policy['M']}")

            prune_state()

    except KeyboardInterrupt:
        print("\n[policy_node] stopping...")


if __name__ == "__main__":
    main()
