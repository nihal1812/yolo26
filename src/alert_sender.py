#!/usr/bin/env python3
import argparse
import json
import time

import redis
import requests

from config_utils import load_cfg, get_stream, get_active_cams


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def parse_xread(streams):
    out = []
    for _s, msgs in streams:
        for mid, fields in msgs:
            out.append((b2s(mid), fields))
    return out


def resolve_alerts_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "alerts_enriched", cam_id)
    except Exception:
        return get_stream(cfg, "alerts", cam_id)


def resolve_incident_alerts_stream(cfg):
    try:
        return get_stream(cfg, "incidents_alerts")
    except Exception:
        return "incidents_alerts"


def post_with_retry(url, payload, timeout_s, retries=3, base_sleep=0.5):
    last_err = None
    for attempt in range(retries):
        try:
            headers = {}
            idem_key = payload.get("incident_id") or payload.get("event_id")
            if idem_key:
                headers["Idempotency-Key"] = str(idem_key)

            r = requests.post(url, json=payload, timeout=timeout_s, headers=headers)
            ok = 200 <= r.status_code < 300
            if ok:
                return True, r.status_code, None
            last_err = f"status={r.status_code}"
        except Exception as e:
            last_err = str(e)

        if attempt < (retries - 1):
            time.sleep(base_sleep * (2 ** attempt))

    return False, None, last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=50)

    ap.add_argument("--alerts_stream", default=None)
    ap.add_argument("--incident_alerts_stream", default=None)

    ap.add_argument("--source_mode", choices=["local", "incident"], default="local")
    ap.add_argument("--incident_leader_cam", default=None)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)
    active_cams = get_active_cams(cfg)

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
    )
    rdb.ping()

    p3 = cfg.get("pipeline3", {})
    wh = p3.get("webhooks", {})
    timeout_s = float(wh.get("timeout_s", 3.0))

    leader_cam = args.incident_leader_cam or (active_cams[0] if active_cams else cam_id)

    if args.source_mode == "local":
        alerts_url = wh.get("alerts_url")
        stream_name = args.alerts_stream or resolve_alerts_stream(cfg, cam_id)
        if not alerts_url:
            raise RuntimeError("pipeline3.webhooks.alerts_url missing in config")
    else:
        alerts_url = wh.get("incident_alerts_url") or wh.get("alerts_url")
        stream_name = args.incident_alerts_stream or resolve_incident_alerts_stream(cfg)
        if not alerts_url:
            raise RuntimeError("pipeline3.webhooks.incident_alerts_url or alerts_url missing in config")

    last_id = "0-0"

    print(f"[alert_sender] cam_id={cam_id}")
    print(f"[alert_sender] source_mode={args.source_mode}")
    print(f"[alert_sender] stream={stream_name}")
    print(f"[alert_sender] POST {alerts_url}")

    if args.source_mode == "incident" and cam_id != leader_cam:
        print(f"[alert_sender] incident mode inactive on cam={cam_id}; leader_cam={leader_cam}")
        try:
            while True:
                time.sleep(5.0)
        except KeyboardInterrupt:
            print("\n[alert_sender] stopping...")
        return

    while True:
        streams = rdb.xread({stream_name: last_id}, block=args.block_ms, count=args.count)
        if not streams:
            continue

        for mid, fields in parse_xread(streams):
            last_id = mid
            js = fields.get(b"json", b"{}")
            try:
                obj = json.loads(b2s(js))
            except Exception:
                continue

            if args.source_mode == "local":
                alert = obj
                payload = {
                    "alert_type": "local_preincident",
                    "event_id": alert.get("event_id"),
                    "cam_id": alert.get("cam_id", cam_id),
                    "person_track_id": alert.get("person_track_id", -1),
                    "global_person_id": alert.get("global_person_id", None),
                    "frame_id_end": alert.get("frame_id_end", -1),
                    "stamp_ns_end": alert.get("stamp_ns_end", 0),
                    "score": alert.get("score", None),
                    "suspicion": alert.get("suspicion", alert.get("score", None)),
                    "reason": alert.get("reason", {}),
                    "model_version": alert.get("model_version", "unknown"),
                    "object_track_id": alert.get("object_track_id", -1),
                    "object_class_id": alert.get("object_class_id", -1),
                    "policy_features": alert.get("policy_features", {}),
                    "identity_enriched": bool(alert.get("identity_enriched", False)),
                    "identity_debug": alert.get("identity_debug", {}),
                    "incident_candidate": bool(alert.get("incident_candidate", False)),
                    "raw": alert,
                }
            else:
                incident = obj
                latest_signal = incident.get("latest_signal", {}) if isinstance(incident.get("latest_signal", {}), dict) else {}
                payload = {
                    "alert_type": "incident",
                    "incident_id": incident.get("incident_id"),
                    "event_id": latest_signal.get("event_id"),
                    "cam_id": latest_signal.get("cam_id"),
                    "person_track_id": latest_signal.get("person_track_id", -1),
                    "global_person_id": incident.get("global_person_id", None),
                    "frame_id_end": latest_signal.get("frame_id", -1),
                    "stamp_ns_end": latest_signal.get("stamp_ns", 0),
                    "score": latest_signal.get("score", None),
                    "suspicion": latest_signal.get("suspicion", latest_signal.get("score", None)),
                    "reason": incident.get("reason"),
                    "summary": incident.get("summary", {}),
                    "cams_seen_all": incident.get("cams_seen_all", []),
                    "local_refs_all": incident.get("local_refs_all", []),
                    "raw": incident,
                }

            ok, status_code, err = post_with_retry(alerts_url, payload, timeout_s, retries=3)
            if ok:
                print(
                    f"[alert_sender] cam={cam_id} sent {payload.get('alert_type')} mid={mid} "
                    f"status={status_code} ok=True gid={payload.get('global_person_id')}"
                )
            else:
                print(f"[alert_sender] cam={cam_id} failed alert mid={mid}: {err}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[alert_sender] stopping...")