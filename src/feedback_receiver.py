#!/usr/bin/env python3
import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml
import redis


def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_int(v, d=None):
    try:
        return int(v)
    except Exception:
        return d


def b2s(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


class Handler(BaseHTTPRequestHandler):
    rdb = None
    cam_id = None
    stream = None
    secret = None
    path = "/feedback"

    # idempotency
    dedup_prefix = "feedback:dedup"
    dedup_ttl_s = 7 * 24 * 3600  # 7 days

    def _send(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _unauth(self):
        return self._send(401, {"ok": False, "error": "unauthorized"})

    def do_POST(self):
        if self.path != Handler.path:
            return self._send(404, {"ok": False, "error": "not_found"})

        # optional shared secret
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

        cam = obj.get("cam_id", Handler.cam_id)
        pid = safe_int(obj.get("person_track_id", -1), -1)
        fid = safe_int(obj.get("frame_id_end", -1), -1)
        label = safe_int(obj.get("label", None), None)

        alert_id = obj.get("alert_id", None)        # recommended
        decision_id = obj.get("decision_id", None)  # optional
        stamp_ns_end = safe_int(obj.get("stamp_ns_end", 0), 0)

        # Feedback id (idempotency)
        feedback_id = obj.get("feedback_id", None) or str(uuid.uuid4())

        # Accept if pid valid and label valid and (frame_id_end OR alert_id OR decision_id)
        if pid < 0 or label not in (0, 1) or (fid < 0 and not alert_id and not decision_id):
            return self._send(
                400,
                {
                    "ok": False,
                    "error": "missing_fields",
                    "need": [
                        "person_track_id",
                        "label(0/1)",
                        "one_of: frame_id_end OR alert_id OR decision_id"
                    ],
                    "got": {
                        "person_track_id": pid,
                        "frame_id_end": fid,
                        "alert_id": bool(alert_id),
                        "decision_id": bool(decision_id),
                        "label": label,
                    }
                },
            )

        # Deduplicate retries
        dedup_key = f"{Handler.dedup_prefix}:{cam}:{feedback_id}"
        # SETNX
        if not Handler.rdb.set(dedup_key, "1", nx=True, ex=int(Handler.dedup_ttl_s)):
            return self._send(200, {"ok": True, "dedup": True, "feedback_id": feedback_id})

        payload = {
            "type": "feedback",
            "cam_id": cam,
            "person_track_id": pid,
            "frame_id_end": fid,  # may be -1 if unknown
            "stamp_ns_end": stamp_ns_end,
            "label": int(label),

            # matching helpers
            "alert_id": alert_id,
            "decision_id": decision_id,

            # metadata
            "feedback_id": feedback_id,
            "category": obj.get("category", None),
            "comment": obj.get("comment", ""),
            "user_id": obj.get("user_id", None),

            "received_ns": time.time_ns(),
            "raw": obj,
        }

        Handler.rdb.xadd(
            Handler.stream,
            {
                "cam_id": cam,
                "person_track_id": str(pid),
                "frame_id": str(fid),
                "stamp_ns": str(stamp_ns_end),
                "json": json.dumps(payload),
            },
            maxlen=20000,
            approximate=True,
        )

        return self._send(200, {"ok": True, "feedback_id": feedback_id})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
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

    fb_cfg = cfg.get("feedback_webhook", {})
    host = fb_cfg.get("host", "0.0.0.0")
    port = int(fb_cfg.get("port", 8090))
    path = fb_cfg.get("path", "/feedback")
    secret = fb_cfg.get("secret", None)

    feedback_stream = r_cfg.get("feedback_stream", f"feedback:{cam_id}")

    Handler.rdb = rdb
    Handler.cam_id = cam_id
    Handler.stream = feedback_stream
    Handler.secret = secret
    Handler.path = path

    httpd = HTTPServer((host, port), Handler)
    print(f"[feedback_receiver] listening http://{host}:{port}{path} -> Redis {feedback_stream}")
    httpd.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[feedback_receiver] stopping...")
