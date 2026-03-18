#!/usr/bin/env python3
"""
incident_builder.py

Consumes:
  - decisions_enriched:{cam}
  - alerts_enriched:{cam}

Produces:
  - incidents
  - incidents_enriched

Purpose:
- build one cross-camera incident timeline per global_person_id
- merge suspicious evidence coming from multiple cameras
- emit:
    1) incident state updates
    2) incident alerts when escalation threshold is reached

Design:
- entity key is global_person_id when present
- falls back to cam_id + person_track_id if global id is missing
- accumulates evidence from enriched decisions and alerts
- maintains incident state with TTL
- emits a new incident when:
    - score/suspicion gets high enough, or
    - alert events arrive, or
    - evidence spans multiple cameras with enough confidence
"""

import argparse
import json
import time
from collections import defaultdict, deque
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


def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def try_get_stream(cfg: dict, key: str, cam_id: Optional[str] = None, fallback: Optional[str] = None):
    try:
        return get_stream(cfg, key, cam_id)
    except Exception:
        if fallback is not None:
            return fallback
        raise


def now_s():
    return time.time()


class IncidentBuilder:
    def __init__(
        self,
        incident_ttl_s: float,
        evidence_window_s: float,
        min_cams_for_cross_camera: int,
        incident_open_thr: float,
        incident_alert_thr: float,
        incident_alert_on_any_alert: bool,
        max_history: int,
    ):
        self.incident_ttl_s = float(incident_ttl_s)
        self.evidence_window_s = float(evidence_window_s)
        self.min_cams_for_cross_camera = int(min_cams_for_cross_camera)
        self.incident_open_thr = float(incident_open_thr)
        self.incident_alert_thr = float(incident_alert_thr)
        self.incident_alert_on_any_alert = bool(incident_alert_on_any_alert)
        self.max_history = int(max_history)

        self.incidents: Dict[str, Dict[str, Any]] = {}
        self.next_incident_id = 1

    def _make_entity_key(self, obj: dict) -> str:
        gid = safe_int(obj.get("global_person_id", None), None)
        if gid is not None:
            return f"gid:{gid}"

        cam = str(obj.get("cam_id", "unknown"))
        pid = safe_int(obj.get("person_track_id", -1), -1)
        return f"cam:{cam}:pid:{pid}"

    def _prune(self):
        t = now_s()
        dead = []
        for k, st in self.incidents.items():
            if (t - float(st.get("updated_at_s", 0.0))) > self.incident_ttl_s:
                dead.append(k)
        for k in dead:
            self.incidents.pop(k, None)

    def _new_incident_state(self, entity_key: str, obj: dict) -> Dict[str, Any]:
        gid = safe_int(obj.get("global_person_id", None), None)
        cam = str(obj.get("cam_id", "unknown"))
        pid = safe_int(obj.get("person_track_id", -1), -1)

        t = now_s()
        incident_id = f"incident_{self.next_incident_id}"
        self.next_incident_id += 1

        return {
            "incident_id": incident_id,
            "entity_key": entity_key,
            "global_person_id": gid,
            "fallback_local_refs": [{"cam_id": cam, "person_track_id": pid}] if gid is None else [],
            "created_at_s": t,
            "updated_at_s": t,
            "status": "open",
            "cams_seen": set(),
            "local_refs": set(),
            "evidence": deque(maxlen=self.max_history),
            "event_ids": deque(maxlen=self.max_history),
            "max_score": 0.0,
            "max_suspicion": 0.0,
            "alert_count": 0,
            "decision_count": 0,
            "last_alert_at_s": None,
            "last_decision_at_s": None,
            "incident_alerted": False,
            "incident_open_reason": None,
            "incident_alert_reason": None,
        }

    def _normalize_signal(self, obj: dict, source_type: str) -> Dict[str, Any]:
        score = safe_float(obj.get("score", None), None)
        suspicion = safe_float(obj.get("suspicion", None), None)

        if suspicion is None:
            suspicion = safe_float(obj.get("S", None), None)
        if suspicion is None:
            suspicion = score
        if suspicion is None:
            suspicion = 0.0

        if score is None:
            score = suspicion

        cam = str(obj.get("cam_id", "unknown"))
        pid = safe_int(obj.get("person_track_id", -1), -1)
        gid = safe_int(obj.get("global_person_id", None), None)
        fid = safe_int(obj.get("frame_id_end", None), None)
        if fid is None:
            fid = safe_int(obj.get("frame_id", -1), -1)
        stamp_ns = safe_int(obj.get("stamp_ns_end", None), None)
        if stamp_ns is None:
            stamp_ns = safe_int(obj.get("stamp_ns", 0), 0)

        signal = {
            "source_type": source_type,
            "event_id": obj.get("event_id"),
            "cam_id": cam,
            "person_track_id": pid,
            "global_person_id": gid,
            "frame_id": fid,
            "stamp_ns": stamp_ns,
            "score": float(score),
            "suspicion": float(clamp01(suspicion)),
            "will_alert": bool(obj.get("will_alert", False)),
            "policy_features": obj.get("policy_features", {}) if isinstance(obj.get("policy_features", {}), dict) else {},
            "reason": obj.get("reason", {}) if isinstance(obj.get("reason", {}), dict) else {},
            "identity_debug": obj.get("identity_debug", {}) if isinstance(obj.get("identity_debug", {}), dict) else {},
            "received_at_s": now_s(),
        }
        return signal

    def _trim_old_evidence(self, st: Dict[str, Any]):
        t = now_s()
        fresh = deque(maxlen=self.max_history)
        for ev in st["evidence"]:
            if (t - float(ev.get("received_at_s", t))) <= self.evidence_window_s:
                fresh.append(ev)
        st["evidence"] = fresh

    def _compute_summary(self, st: Dict[str, Any]) -> Dict[str, Any]:
        self._trim_old_evidence(st)

        cams = sorted({ev.get("cam_id") for ev in st["evidence"] if ev.get("cam_id") is not None})
        suspicions = [float(ev.get("suspicion", 0.0)) for ev in st["evidence"]]
        scores = [float(ev.get("score", 0.0)) for ev in st["evidence"]]
        alert_events = [ev for ev in st["evidence"] if ev.get("source_type") == "alert"]
        decision_events = [ev for ev in st["evidence"] if ev.get("source_type") == "decision"]

        max_susp = max(suspicions) if suspicions else 0.0
        mean_susp = sum(suspicions) / len(suspicions) if suspicions else 0.0
        max_score = max(scores) if scores else 0.0

        cross_camera = len(cams) >= self.min_cams_for_cross_camera

        return {
            "cams_seen_recent": cams,
            "num_cams_recent": len(cams),
            "num_evidence_recent": len(st["evidence"]),
            "num_alerts_recent": len(alert_events),
            "num_decisions_recent": len(decision_events),
            "max_suspicion_recent": float(max_susp),
            "mean_suspicion_recent": float(mean_susp),
            "max_score_recent": float(max_score),
            "cross_camera_active": bool(cross_camera),
        }

    def ingest(self, obj: dict, source_type: str) -> Tuple[Dict[str, Any], bool, Optional[str], Dict[str, Any]]:
        self._prune()

        entity_key = self._make_entity_key(obj)
        st = self.incidents.get(entity_key)
        if st is None:
            st = self._new_incident_state(entity_key, obj)
            self.incidents[entity_key] = st

        signal = self._normalize_signal(obj, source_type)

        st["updated_at_s"] = now_s()
        st["cams_seen"].add(signal["cam_id"])
        st["local_refs"].add((signal["cam_id"], signal["person_track_id"]))
        st["evidence"].append(signal)

        ev_id = signal.get("event_id")
        if ev_id is not None:
            st["event_ids"].append(ev_id)

        st["max_score"] = max(float(st.get("max_score", 0.0)), float(signal["score"]))
        st["max_suspicion"] = max(float(st.get("max_suspicion", 0.0)), float(signal["suspicion"]))

        if source_type == "alert":
            st["alert_count"] += 1
            st["last_alert_at_s"] = now_s()
        else:
            st["decision_count"] += 1
            st["last_decision_at_s"] = now_s()

        summary = self._compute_summary(st)

        opened_now = False
        alert_now = False
        alert_reason = None

        if st["incident_open_reason"] is None:
            if summary["max_suspicion_recent"] >= self.incident_open_thr:
                st["incident_open_reason"] = "suspicion_open_threshold"
                opened_now = True
            elif summary["cross_camera_active"] and summary["mean_suspicion_recent"] >= max(0.40, self.incident_open_thr * 0.65):
                st["incident_open_reason"] = "cross_camera_multi_evidence"
                opened_now = True
            elif source_type == "alert" and self.incident_alert_on_any_alert:
                st["incident_open_reason"] = "incoming_alert"
                opened_now = True

        if not st["incident_alerted"]:
            if self.incident_alert_on_any_alert and source_type == "alert":
                alert_now = True
                alert_reason = "source_alert"
            elif summary["max_suspicion_recent"] >= self.incident_alert_thr:
                alert_now = True
                alert_reason = "incident_alert_threshold"
            elif summary["cross_camera_active"] and summary["mean_suspicion_recent"] >= max(0.65, self.incident_alert_thr * 0.80):
                alert_now = True
                alert_reason = "cross_camera_escalation"

            if alert_now:
                st["incident_alerted"] = True
                st["incident_alert_reason"] = alert_reason

        latest_signal = signal
        incident_obj = {
            "type": "incident",
            "incident_id": st["incident_id"],
            "entity_key": st["entity_key"],
            "global_person_id": st["global_person_id"],
            "cams_seen_all": sorted(list(st["cams_seen"])),
            "local_refs_all": sorted([{"cam_id": c, "person_track_id": p} for (c, p) in st["local_refs"]], key=lambda x: (x["cam_id"], x["person_track_id"])),
            "status": st["status"],
            "created_at_s": float(st["created_at_s"]),
            "updated_at_s": float(st["updated_at_s"]),
            "incident_open_reason": st["incident_open_reason"],
            "incident_alerted": bool(st["incident_alerted"]),
            "incident_alert_reason": st["incident_alert_reason"],
            "max_score": float(st["max_score"]),
            "max_suspicion": float(st["max_suspicion"]),
            "summary": summary,
            "latest_signal": latest_signal,
            "event_ids_recent": list(st["event_ids"]),
        }

        return incident_obj, alert_now, alert_reason, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--redis_maxlen", type=int, default=50000)

    ap.add_argument("--incident_ttl_s", type=float, default=120.0)
    ap.add_argument("--evidence_window_s", type=float, default=20.0)
    ap.add_argument("--min_cams_for_cross_camera", type=int, default=2)

    ap.add_argument("--incident_open_thr", type=float, default=0.70)
    ap.add_argument("--incident_alert_thr", type=float, default=0.90)
    ap.add_argument("--incident_alert_on_any_alert", action="store_true")

    ap.add_argument("--max_history", type=int, default=100)

    ap.add_argument("--incidents_stream", default=None)
    ap.add_argument("--incidents_alerts_stream", default=None)

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

    incidents_stream = args.incidents_stream or try_get_stream(cfg, "incidents", None, "incidents")
    incidents_alerts_stream = args.incidents_alerts_stream or try_get_stream(cfg, "incidents_alerts", None, "incidents_alerts")

    input_streams = {}
    last_ids = {}

    for cam_id in active_cams:
        dec_s = try_get_stream(cfg, "decisions_enriched", cam_id, f"decisions_enriched:{cam_id}")
        al_s = try_get_stream(cfg, "alerts_enriched", cam_id, f"alerts_enriched:{cam_id}")

        input_streams[dec_s] = ("decision", cam_id)
        input_streams[al_s] = ("alert", cam_id)
        last_ids[dec_s] = "0-0"
        last_ids[al_s] = "0-0"

    builder = IncidentBuilder(
        incident_ttl_s=args.incident_ttl_s,
        evidence_window_s=args.evidence_window_s,
        min_cams_for_cross_camera=args.min_cams_for_cross_camera,
        incident_open_thr=args.incident_open_thr,
        incident_alert_thr=args.incident_alert_thr,
        incident_alert_on_any_alert=args.incident_alert_on_any_alert,
        max_history=args.max_history,
    )

    print(f"[incident_builder] active_cams={active_cams}")
    for s, (stype, cam) in input_streams.items():
        print(f"[incident_builder] IN {stype} cam={cam} -> {s}")
    print(f"[incident_builder] OUT incidents={incidents_stream}")
    print(f"[incident_builder] OUT incident_alerts={incidents_alerts_stream}")

    try:
        while True:
            streams = rdb.xread(last_ids, block=args.block_ms, count=args.count)
            if not streams:
                builder._prune()
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

                meta = input_streams.get(sname, None)
                if meta is None:
                    continue

                source_type, cam_id = meta
                incident_obj, alert_now, alert_reason, summary = builder.ingest(obj, source_type)

                latest = incident_obj.get("latest_signal", {})
                event_id = latest.get("event_id", "")

                rdb.xadd(
                    incidents_stream,
                    {
                        "incident_id": str(incident_obj["incident_id"]),
                        "event_id": str(event_id or ""),
                        "cam_id": str(latest.get("cam_id", cam_id)),
                        "person_track_id": str(latest.get("person_track_id", -1)),
                        "global_person_id": "" if incident_obj.get("global_person_id") is None else str(incident_obj["global_person_id"]),
                        "frame_id": str(latest.get("frame_id", -1)),
                        "stamp_ns": str(latest.get("stamp_ns", 0)),
                        "json": json.dumps(incident_obj),
                    },
                    maxlen=args.redis_maxlen,
                    approximate=True,
                )

                if alert_now:
                    incident_alert = {
                        "type": "incident_alert",
                        "incident_id": incident_obj["incident_id"],
                        "global_person_id": incident_obj.get("global_person_id"),
                        "entity_key": incident_obj["entity_key"],
                        "reason": alert_reason,
                        "summary": summary,
                        "latest_signal": incident_obj.get("latest_signal", {}),
                        "cams_seen_all": incident_obj.get("cams_seen_all", []),
                        "local_refs_all": incident_obj.get("local_refs_all", []),
                        "event_ids_recent": incident_obj.get("event_ids_recent", []),
                        "created_at_s": incident_obj.get("created_at_s"),
                        "updated_at_s": incident_obj.get("updated_at_s"),
                    }

                    rdb.xadd(
                        incidents_alerts_stream,
                        {
                            "incident_id": str(incident_obj["incident_id"]),
                            "event_id": str(event_id or ""),
                            "cam_id": str(latest.get("cam_id", cam_id)),
                            "person_track_id": str(latest.get("person_track_id", -1)),
                            "global_person_id": "" if incident_obj.get("global_person_id") is None else str(incident_obj["global_person_id"]),
                            "frame_id": str(latest.get("frame_id", -1)),
                            "stamp_ns": str(latest.get("stamp_ns", 0)),
                            "json": json.dumps(incident_alert),
                        },
                        maxlen=args.redis_maxlen,
                        approximate=True,
                    )

                    print(
                        f"[incident_builder] INCIDENT ALERT incident_id={incident_obj['incident_id']} "
                        f"gid={incident_obj.get('global_person_id')} "
                        f"cams={summary.get('cams_seen_recent')} "
                        f"reason={alert_reason}"
                    )

    except KeyboardInterrupt:
        print("\n[incident_builder] stopping...")


if __name__ == "__main__":
    main()