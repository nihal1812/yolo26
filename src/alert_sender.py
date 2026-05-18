#!/usr/bin/env python3
import argparse
import json
import time

import zmq
import requests

from config_utils import load_cfg, get_active_cams, get_zmq_endpoint, local_connect_addr


def resolve_alerts_endpoint(cfg, cam_id: str):
    try:
        return get_zmq_endpoint(cfg, cam_id, "alerts_enriched")
    except Exception:
        return get_zmq_endpoint(cfg, cam_id, "alerts")


def resolve_incident_alerts_endpoint(cfg, cam_id: str):
    try:
        return get_zmq_endpoint(cfg, cam_id, "incident_alerts")
    except Exception:
        return None


def post_with_retry(url, payload, timeout_s, retries=3, base_sleep=0.5):
    last_err = None

    for attempt in range(retries):
        try:
            headers = {}
            idem_key = payload.get("incident_id") or payload.get("event_id")
            if idem_key:
                headers["Idempotency-Key"] = str(idem_key)

            r = requests.post(url, json=payload, timeout=timeout_s, headers=headers)
            if 200 <= r.status_code < 300:
                return True, r.status_code, None

            last_err = f"status={r.status_code} body={r.text[:300]}"
        except Exception as e:
            last_err = str(e)

        if attempt < retries - 1:
            time.sleep(base_sleep * (2 ** attempt))

    return False, None, last_err


def make_sub_socket(ctx, connect_addr, topic, rcvhwm=1000, latest_only=False):
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, int(rcvhwm))

    if latest_only:
        sub.setsockopt(zmq.CONFLATE, 1)

    sub.connect(connect_addr)
    sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    return sub


class AlertSenderStage:
    def __init__(
        self,
        cfg,
        cam_id: str,
        *,
        poll_ms: int = 100,
        source_mode: str = "local",
        incident_leader_cam: str = None,
        latest_only: bool = False,
        rcvhwm: int = 1000,
        external_mode: bool = False,
    ):
        self.cfg = cfg
        self.cam_id = str(cam_id)
        self.poll_ms = int(poll_ms)
        self.source_mode = str(source_mode)
        self.latest_only = bool(latest_only)
        self.rcvhwm = int(rcvhwm)
        self.external_mode = bool(external_mode)

        self.clip_refs_by_event = {}
        self.clip_refs_by_track = {}
        self.clip_ref_ttl_s = 90.0

        active_cams = get_active_cams(cfg)

        ui_cfg = cfg.get("ui_pipeline", {})
        wh = ui_cfg.get("webhooks", {})
        self.timeout_s = float(wh.get("timeout_s", 3.0))
        self.webhooks_enabled = bool(wh.get("enabled", False))

        self.leader_cam = incident_leader_cam or (
            active_cams[0] if active_cams else self.cam_id
        )

        if self.source_mode == "local":
            self.alerts_url = wh.get("alerts_url")
            if self.webhooks_enabled and not self.alerts_url:
                raise RuntimeError("ui_pipeline.webhooks.alerts_url missing")
        else:
            self.alerts_url = (
                wh.get("incident_alerts_url")
                or wh.get("incidents_url")
                or wh.get("alerts_url")
            )
            if self.webhooks_enabled and not self.alerts_url:
                raise RuntimeError("incident alert webhook URL missing")

        self.ctx = zmq.Context.instance()
        self.poller = zmq.Poller()
        self.sub = None
        self.topic = None

        self._build_subscriber()

        print(f"[alert_sender] cam_id={self.cam_id}")
        print(f"[alert_sender] source_mode={self.source_mode}")
        print(f"[alert_sender] external_mode={self.external_mode}")
        print(f"[alert_sender] webhooks_enabled={self.webhooks_enabled}")
        print(f"[alert_sender] POST {self.alerts_url}")

    def _build_subscriber(self):
        if self.external_mode:
            return

        if self.source_mode == "incident" and self.cam_id != self.leader_cam:
            return

        if self.source_mode == "local":
            ep = resolve_alerts_endpoint(self.cfg, self.cam_id)
        else:
            ep = resolve_incident_alerts_endpoint(self.cfg, self.leader_cam)
            if ep is None:
                raise RuntimeError("incident_alerts ZMQ endpoint missing")

        connect_addr = local_connect_addr(ep["bind"])
        topic = ep["topic"]

        self.sub = make_sub_socket(
            self.ctx,
            connect_addr=connect_addr,
            topic=topic,
            rcvhwm=int(ep.get("rcvhwm", self.rcvhwm)),
            latest_only=bool(ep.get("latest_only", self.latest_only)),
        )

        self.topic = topic
        self.poller.register(self.sub, zmq.POLLIN)

    def _clip_track_key(self, cam_id, person_track_id, global_person_id=None):
        if global_person_id is not None:
            return f"{cam_id}:gid:{global_person_id}"
        return f"{cam_id}:pid:{person_track_id}"

    def _prune_clip_refs(self):
        now = time.time()

        for k in list(self.clip_refs_by_event.keys()):
            if now - self.clip_refs_by_event[k].get("_cached_at", 0) > self.clip_ref_ttl_s:
                self.clip_refs_by_event.pop(k, None)

        for k in list(self.clip_refs_by_track.keys()):
            if now - self.clip_refs_by_track[k].get("_cached_at", 0) > self.clip_ref_ttl_s:
                self.clip_refs_by_track.pop(k, None)

    def handle_clip_ref_obj(self, ref_obj: dict):
        if not isinstance(ref_obj, dict):
            return False

        self._prune_clip_refs()

        ref = dict(ref_obj)
        ref["_cached_at"] = time.time()

        event_id = ref.get("event_id")
        cam_id = ref.get("cam_id", self.cam_id)
        person_track_id = ref.get("person_track_id", -1)
        global_person_id = ref.get("global_person_id")

        if event_id:
            self.clip_refs_by_event[str(event_id)] = ref

        track_key = self._clip_track_key(cam_id, person_track_id, global_person_id)
        self.clip_refs_by_track[track_key] = ref

        print(
            f"[alert_sender] cached clip_ref cam={cam_id} "
            f"event_id={event_id} clip={ref.get('local_clip_path')}"
        )

        return True

    def _find_clip_ref_for_payload(self, payload: dict):
        self._prune_clip_refs()

        # Highest priority: an already-attached final clipRef.
        direct_ref = payload.get("clipRef")
        if isinstance(direct_ref, dict):
            return direct_ref

        event_id = payload.get("event_id")
        if event_id and str(event_id) in self.clip_refs_by_event:
            return self.clip_refs_by_event[str(event_id)]

        cam_id = payload.get("cam_id", self.cam_id)
        person_track_id = payload.get("person_track_id", -1)
        global_person_id = payload.get("global_person_id")

        key = self._clip_track_key(cam_id, person_track_id, global_person_id)
        cached_ref = self.clip_refs_by_track.get(key)
        if cached_ref:
            return cached_ref

        # No fallback to policy/model local_clip_path here.
        # If bbox_overlay fails to create an annotated MP4, clip will remain None,
        # making the failure visible in logs/dashboard instead of silently using raw NPZ.
        return None

    def _attach_clip_ref(self, payload: dict):
        ref = self._find_clip_ref_for_payload(payload)
        if not ref:
            return payload

        payload = dict(payload)

        local_clip_path = ref.get("local_clip_path") or ref.get("clipPath")
        clip_url = ref.get("clip_url") or ref.get("clipUrl") or ""

        payload["clipPath"] = local_clip_path
        payload["clipUrl"] = clip_url
        payload["clipRef"] = ref
        payload["storage_status"] = ref.get("storage_status")

        return payload

    def _build_payload(self, obj: dict):
        if self.source_mode == "local":
            alert = obj

            cam_id = alert.get("cam_id", self.cam_id)
            person_track_id = alert.get("person_track_id", -1)
            score = alert.get("score", alert.get("suspicion"))

            return {
                "event": "suspicious_activity",
                "alert_type": "local_preincident",

                "event_id": alert.get("event_id"),
                "cam_id": cam_id,
                "cameraId": cam_id,

                "person_track_id": person_track_id,
                "trackId": person_track_id,

                "global_person_id": alert.get("global_person_id"),
                "frame_id_end": alert.get("frame_id_end", alert.get("frame_id", -1)),
                "stamp_ns_end": alert.get("stamp_ns_end", alert.get("stamp_ns", 0)),

                "score": score,
                "confidence": score,
                "suspicion": alert.get("suspicion", score),

                "reason": alert.get("reason", {}),
                "model_version": alert.get("model_version", "unknown"),

                "object_track_id": alert.get("object_track_id", -1),
                "object_class_id": alert.get("object_class_id", -1),

                "policy_features": alert.get("policy_features", {}),
                "identity_enriched": bool(alert.get("identity_enriched", False)),
                "identity_debug": alert.get("identity_debug", {}),
                "incident_candidate": bool(alert.get("incident_candidate", False)),

                # Model clip reference fields carried by policy_node.
                # The final dashboard clipRef is generated by bbox_overlay as annotated MP4.
                # for debugging/fallback and downstream visibility.
                "clip_path": alert.get("clip_path"),
                "local_clip_path": alert.get("local_clip_path", alert.get("clip_path")),
                "clip_codec": alert.get("clip_codec"),

                "bbox_xyxy": alert.get("bbox_xyxy"),
                "bbox": alert.get("bbox") or alert.get("bbox_xyxy"),
                "label": alert.get("label", "person"),

                "raw": alert,
            }

        incident = obj
        latest_signal = incident.get("latest_signal", {})
        if not isinstance(latest_signal, dict):
            latest_signal = {}

        cam_id = latest_signal.get("cam_id")
        person_track_id = latest_signal.get("person_track_id", -1)
        score = latest_signal.get("score", latest_signal.get("suspicion"))

        return {
            "event": "suspicious_incident",
            "alert_type": "incident",

            "incident_id": incident.get("incident_id"),
            "event_id": latest_signal.get("event_id"),

            "cam_id": cam_id,
            "cameraId": cam_id,

            "person_track_id": person_track_id,
            "trackId": person_track_id,

            "global_person_id": incident.get("global_person_id"),
            "frame_id_end": latest_signal.get("frame_id", -1),
            "stamp_ns_end": latest_signal.get("stamp_ns", 0),

            "score": score,
            "confidence": score,
            "suspicion": latest_signal.get("suspicion", score),

            "reason": incident.get("reason"),
            "summary": incident.get("summary", {}),
            "cams_seen_all": incident.get("cams_seen_all", []),
            "local_refs_all": incident.get("local_refs_all", []),
            "fallback_local_refs": incident.get("fallback_local_refs", []),

            # Clip reference fields if incident_builder carries them in latest_signal.
            "clip_path": latest_signal.get("clip_path"),
            "local_clip_path": latest_signal.get("local_clip_path", latest_signal.get("clip_path")),
            "clip_codec": latest_signal.get("clip_codec"),

            "bbox_xyxy": latest_signal.get("bbox_xyxy"),
            "bbox": latest_signal.get("bbox") or latest_signal.get("bbox_xyxy"),
            "label": latest_signal.get("label", "person"),

            "raw": incident,
        }

    def handle_alert_obj(self, obj: dict):
        payload = self._build_payload(obj)
        payload = self._attach_clip_ref(payload)

        if self.webhooks_enabled:
            ok, status_code, err = post_with_retry(
                self.alerts_url,
                payload,
                self.timeout_s,
                retries=3,
            )
        else:
            ok = True
            status_code = None
            err = None

        if ok:
            print(
                f"[alert_sender] cam={self.cam_id} sent {payload.get('alert_type')} "
                f"ok=True status={status_code} "
                f"gid={payload.get('global_person_id')} "
                f"clip={payload.get('clipPath')}"
            )
        else:
            print(f"[alert_sender] cam={self.cam_id} failed alert: {err}")

        return ok

    def poll_and_send(self, max_messages=None):
        if self.external_mode:
            return 0

        if self.source_mode == "incident" and self.cam_id != self.leader_cam:
            return 0

        if self.sub is None:
            return 0

        events = dict(self.poller.poll(timeout=self.poll_ms))
        if self.sub not in events:
            return 0

        processed = 0

        while True:
            try:
                parts = self.sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:
                break

            if len(parts) != 3:
                continue

            _topic_b, _header_b, payload_b = parts

            try:
                obj = json.loads(payload_b.decode("utf-8"))
            except Exception:
                continue

            self.handle_alert_obj(obj)
            processed += 1

            if max_messages is not None and processed >= max_messages:
                break

        return processed

    def run_forever(self, sleep_s=0.01):
        while True:
            n = self.poll_and_send()
            if n <= 0:
                time.sleep(sleep_s)

    def close(self):
        try:
            if self.sub is not None:
                self.poller.unregister(self.sub)
        except Exception:
            pass

        try:
            if self.sub is not None:
                self.sub.close(0)
        except Exception:
            pass

        self.sub = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)
    ap.add_argument("--poll_ms", type=int, default=100)
    ap.add_argument("--source_mode", choices=["local", "incident"], default="local")
    ap.add_argument("--incident_leader_cam", default=None)
    ap.add_argument("--latest_only", action="store_true")
    ap.add_argument("--rcvhwm", type=int, default=1000)
    ap.add_argument("--external_mode", action="store_true")

    args = ap.parse_args()

    cfg = load_cfg(args.config)

    stage = AlertSenderStage(
        cfg,
        args.cam_id,
        poll_ms=args.poll_ms,
        source_mode=args.source_mode,
        incident_leader_cam=args.incident_leader_cam,
        latest_only=args.latest_only,
        rcvhwm=args.rcvhwm,
        external_mode=args.external_mode,
    )

    try:
        stage.run_forever(sleep_s=0.01)
    finally:
        stage.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[alert_sender] stopping...")
