#!/usr/bin/env python3
import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import redis

from config_utils import load_cfg, get_active_cams, get_stream


def safe_int(v, d=None):
    try:
        return int(v)
    except Exception:
        return d


class Handler(BaseHTTPRequestHandler):
    rdb = None
    active_cams = set()
    feedback_streams = {}
    secret = None
    path = "/feedback"

    dedup_prefix = "feedback:dedup"
    dedup_ttl_s = 7 * 24 * 3600  # 7 days

    def log_message(self, format, *args):
        return

    def _send(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _unauth(self):
        return self._send(401, {"ok": False, "error": "unauthorized"})

    def _pick_stream(self, cam_id: str):
        return Handler.feedback_streams.get(cam_id)

    def do_POST(self):
        if self.path != Handler.path:
            return self._send(404, {"ok": False, "error": "not_found"})

        if Handler.secret:
            got = self.headers.get("X-Secret", "")
            if got != Handler.secret:
                return self._unauth()

        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)

        try:
            obj = json.loads(body.decode("utf-8"))
        except Exception:
            return self._send(400, {"ok": False, "error": "bad_json"})

        cam = str(obj.get("cam_id", "")).strip()
        if not cam or cam not in Handler.active_cams:
            return self._send(
                400,
                {
                    "ok": False,
                    "error": "invalid_cam_id",
                    "allowed_cams": sorted(list(Handler.active_cams)),
                    "got": cam,
                },
            )

        stream = self._pick_stream(cam)
        if not stream:
            return self._send(
                500,
                {"ok": False, "error": "feedback_stream_not_configured", "cam_id": cam},
            )

        pid = safe_int(obj.get("person_track_id", -1), -1)
        fid = safe_int(obj.get("frame_id_end", -1), -1)
        label = safe_int(obj.get("label", None), None)
        stamp_ns_end = safe_int(obj.get("stamp_ns_end", 0), 0)

        alert_id = obj.get("alert_id", None)
        decision_id = obj.get("decision_id", None)
        event_id = obj.get("event_id", None)

        feedback_id = obj.get("feedback_id", None) or str(uuid.uuid4())

        free_text = ""
        for k in ["comment", "description", "feedback_text", "text", "notes", "reason", "explanation", "summary"]:
            v = obj.get(k, None)
            if v is not None and str(v).strip():
                free_text = str(v).strip()
                break

        has_structured_label = label in (0, 1)
        has_match_ref = (fid >= 0) or bool(alert_id) or bool(decision_id) or bool(event_id)
        has_text = bool(free_text)

        if pid < 0 or (not has_match_ref) or (not has_structured_label and not has_text):
            return self._send(
                400,
                {
                    "ok": False,
                    "error": "missing_fields",
                    "need": [
                        "person_track_id",
                        "cam_id",
                        "one_of: label(0/1) OR free-text comment/description/text/notes/reason",
                        "one_of: frame_id_end OR alert_id OR decision_id OR event_id",
                    ],
                    "got": {
                        "cam_id": cam,
                        "person_track_id": pid,
                        "frame_id_end": fid,
                        "alert_id": bool(alert_id),
                        "decision_id": bool(decision_id),
                        "event_id": bool(event_id),
                        "label": label,
                        "has_text": has_text,
                    },
                },
            )

        dedup_key = f"{Handler.dedup_prefix}:{cam}:{feedback_id}"
        if not Handler.rdb.set(dedup_key, "1", nx=True, ex=int(Handler.dedup_ttl_s)):
            return self._send(200, {"ok": True, "dedup": True, "feedback_id": feedback_id})

        payload = {
            "type": "feedback",
            "cam_id": cam,
            "person_track_id": pid,
            "frame_id_end": fid,
            "stamp_ns_end": stamp_ns_end,
            "label": int(label) if has_structured_label else None,
            "event_id": event_id,
            "alert_id": alert_id,
            "decision_id": decision_id,
            "feedback_id": feedback_id,
            "category": obj.get("category", None),
            "comment": obj.get("comment", ""),
            "description": obj.get("description", None),
            "feedback_text": obj.get("feedback_text", None),
            "text": obj.get("text", None),
            "notes": obj.get("notes", None),
            "reason": obj.get("reason", None),
            "explanation": obj.get("explanation", None),
            "summary": obj.get("summary", None),
            "user_id": obj.get("user_id", None),
            "confidence": obj.get("confidence", None),
            "needs_review": obj.get("needs_review", None),
            "reasoning_summary": obj.get("reasoning_summary", None),
            "received_ns": time.time_ns(),
            "raw": obj,
        }

        Handler.rdb.xadd(
            stream,
            {
                "event_id": str(event_id or ""),
                "cam_id": cam,
                "person_track_id": str(pid),
                "frame_id": str(fid),
                "stamp_ns": str(stamp_ns_end),
                "json": json.dumps(payload),
            },
            maxlen=20000,
            approximate=True,
        )

        return self._send(
            200,
            {
                "ok": True,
                "feedback_id": feedback_id,
                "cam_id": cam,
                "stream": stream,
            },
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    active_cams = get_active_cams(cfg)

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
    )
    rdb.ping()

    fb_cfg = cfg.get("feedback_webhook", {})
    host = fb_cfg.get("host", "0.0.0.0")
    port = int(fb_cfg.get("port", 8090))
    path = fb_cfg.get("path", "/feedback")
    secret = fb_cfg.get("secret", None)

    feedback_streams = {cam: get_stream(cfg, "feedback", cam) for cam in active_cams}

    Handler.rdb = rdb
    Handler.active_cams = set(active_cams)
    Handler.feedback_streams = feedback_streams
    Handler.secret = secret
    Handler.path = path

    httpd = HTTPServer((host, port), Handler)
    print(f"[feedback_receiver] listening http://{host}:{port}{path}")
    print(f"[feedback_receiver] active_cams={active_cams}")
    for cam, stream in feedback_streams.items():
        print(f"[feedback_receiver] {cam} -> Redis {stream}")

    httpd.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[feedback_receiver] stopping...")