#!/usr/bin/env python3
"""
incident_builder.py

Shared in-process incident stage.

Consumes:
- enriched decisions / enriched alerts directly

Produces:
- incident object
- optional incident_alert object

Scalable design:
- in-process only
- bounded evidence/history
- duplicate event protection
- transport-agnostic optional publisher hook
"""

import time
from collections import deque
from typing import Dict, Any, Optional, Tuple, Set


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


def now_s():
    return time.time()


class NullIncidentPublisher:
    def publish_incident(self, incident_obj: dict):
        return

    def publish_incident_alert(self, alert_obj: dict):
        return


class IncidentBuilder:
    def __init__(
        self,
        incident_ttl_s: float = 120.0,
        evidence_window_s: float = 20.0,
        min_cams_for_cross_camera: int = 2,
        incident_open_thr: float = 0.70,
        incident_alert_thr: float = 0.90,
        incident_alert_on_any_alert: bool = False,
        max_history: int = 100,
        dedupe_ttl_s: float = 60.0,
        max_fallback_local_refs: int = 32,
        publisher=None,
    ):
        self.incident_ttl_s = float(incident_ttl_s)
        self.evidence_window_s = float(evidence_window_s)
        self.min_cams_for_cross_camera = int(min_cams_for_cross_camera)
        self.incident_open_thr = float(incident_open_thr)
        self.incident_alert_thr = float(incident_alert_thr)
        self.incident_alert_on_any_alert = bool(incident_alert_on_any_alert)
        self.max_history = int(max_history)
        self.dedupe_ttl_s = float(dedupe_ttl_s)
        self.max_fallback_local_refs = int(max_fallback_local_refs)
        self.publisher = publisher if publisher is not None else NullIncidentPublisher()

        self.incidents: Dict[str, Dict[str, Any]] = {}
        self.next_incident_id = 1

        # event_id -> seen_time_s
        self.seen_event_ids: Dict[str, float] = {}

    def _make_entity_key(self, obj: dict) -> str:
        gid = safe_int(obj.get("global_person_id", None), None)
        if gid is not None:
            return f"gid:{gid}"

        cam = str(obj.get("cam_id", "unknown"))
        pid = safe_int(obj.get("person_track_id", -1), -1)
        return f"cam:{cam}:pid:{pid}"

    def _prune_seen_events(self):
        t = now_s()
        dead = [eid for eid, ts in self.seen_event_ids.items() if (t - ts) > self.dedupe_ttl_s]
        for eid in dead:
            self.seen_event_ids.pop(eid, None)

    def _already_seen_event(self, event_id: Optional[str]) -> bool:
        if not event_id:
            return False
        self._prune_seen_events()
        return str(event_id) in self.seen_event_ids

    def _mark_seen_event(self, event_id: Optional[str]):
        if event_id:
            self.seen_event_ids[str(event_id)] = now_s()

    def _prune(self):
        t = now_s()
        dead = []
        for k, st in self.incidents.items():
            if (t - float(st.get("updated_at_s", 0.0))) > self.incident_ttl_s:
                dead.append(k)
        for k in dead:
            self.incidents.pop(k, None)

        self._prune_seen_events()

    def _new_incident_state(self, entity_key: str, obj: dict):
        gid = safe_int(obj.get("global_person_id", None), None)
        cam = str(obj.get("cam_id", "unknown"))
        pid = safe_int(obj.get("person_track_id", -1), -1)

        t = now_s()
        incident_id = f"incident_{self.next_incident_id}"
        self.next_incident_id += 1

        fallback_local_refs = []
        if gid is None:
            fallback_local_refs.append({"cam_id": cam, "person_track_id": pid})

        return {
            "incident_id": incident_id,
            "entity_key": entity_key,
            "global_person_id": gid,
            "fallback_local_refs": fallback_local_refs,
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

    def _normalize_signal(self, obj: dict, source_type: str):
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

        return {
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

    def _trim_old_evidence(self, st):
        t = now_s()
        fresh = deque(maxlen=self.max_history)
        fresh_event_ids = deque(maxlen=self.max_history)

        for ev in st["evidence"]:
            if (t - float(ev.get("received_at_s", t))) <= self.evidence_window_s:
                fresh.append(ev)
                ev_id = ev.get("event_id", None)
                if ev_id is not None:
                    fresh_event_ids.append(ev_id)

        st["evidence"] = fresh
        st["event_ids"] = fresh_event_ids

    def _compute_summary(self, st):
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

    def ingest(self, obj: dict, source_type: str):
        self._prune()

        incoming_event_id = obj.get("event_id", None)
        if self._already_seen_event(incoming_event_id):
            return None, None

        entity_key = self._make_entity_key(obj)
        st = self.incidents.get(entity_key)
        if st is None:
            st = self._new_incident_state(entity_key, obj)
            self.incidents[entity_key] = st

        signal = self._normalize_signal(obj, source_type)

        st["updated_at_s"] = now_s()
        st["cams_seen"].add(signal["cam_id"])
        st["local_refs"].add((signal["cam_id"], signal["person_track_id"]))

        # maintain bounded fallback refs only when gid is absent
        if st["global_person_id"] is None:
            ref = {"cam_id": signal["cam_id"], "person_track_id": signal["person_track_id"]}
            if ref not in st["fallback_local_refs"]:
                st["fallback_local_refs"].append(ref)
                if len(st["fallback_local_refs"]) > self.max_fallback_local_refs:
                    st["fallback_local_refs"] = st["fallback_local_refs"][-self.max_fallback_local_refs:]

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

        alert_now = False
        alert_reason = None

        if st["incident_open_reason"] is None:
            if summary["max_suspicion_recent"] >= self.incident_open_thr:
                st["incident_open_reason"] = "suspicion_open_threshold"
            elif summary["cross_camera_active"] and summary["mean_suspicion_recent"] >= max(0.40, self.incident_open_thr * 0.65):
                st["incident_open_reason"] = "cross_camera_multi_evidence"
            elif source_type == "alert" and self.incident_alert_on_any_alert:
                st["incident_open_reason"] = "incoming_alert"

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

        incident_obj = {
            "type": "incident",
            "incident_id": st["incident_id"],
            "entity_key": st["entity_key"],
            "global_person_id": st["global_person_id"],
            "cams_seen_all": sorted(list(st["cams_seen"])),
            "local_refs_all": sorted(
                [{"cam_id": c, "person_track_id": p} for (c, p) in st["local_refs"]],
                key=lambda x: (x["cam_id"], x["person_track_id"]),
            ),
            "fallback_local_refs": list(st["fallback_local_refs"]),
            "status": st["status"],
            "created_at_s": float(st["created_at_s"]),
            "updated_at_s": float(st["updated_at_s"]),
            "incident_open_reason": st["incident_open_reason"],
            "incident_alerted": bool(st["incident_alerted"]),
            "incident_alert_reason": st["incident_alert_reason"],
            "max_score": float(st["max_score"]),
            "max_suspicion": float(st["max_suspicion"]),
            "summary": summary,
            "latest_signal": signal,
            "event_ids_recent": list(st["event_ids"]),
        }

        incident_alert = None
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
                "fallback_local_refs": incident_obj.get("fallback_local_refs", []),
                "event_ids_recent": incident_obj.get("event_ids_recent", []),
                "created_at_s": incident_obj.get("created_at_s"),
                "updated_at_s": incident_obj.get("updated_at_s"),
            }

        self._mark_seen_event(incoming_event_id)

        try:
            self.publisher.publish_incident(incident_obj)
        except Exception:
            pass

        if incident_alert is not None:
            try:
                self.publisher.publish_incident_alert(incident_alert)
            except Exception:
                pass

        return incident_obj, incident_alert
