#!/usr/bin/env python3
import argparse, json, time
import yaml, redis
import requests

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

def parse_xread(streams):
    out = []
    for _s, msgs in streams:
        for mid, fields in msgs:
            out.append((b2s(mid), fields))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=50)
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

    p3 = cfg.get("pipeline3", {})
    wh = p3.get("webhooks", {})
    alerts_url = wh.get("alerts_url")
    timeout_s = float(wh.get("timeout_s", 3.0))

    alerts_stream = r_cfg.get("alerts_stream", f"alerts:{cam_id}")
    if not alerts_url:
        raise RuntimeError("pipeline3.webhooks.alerts_url missing in config")

    last_id = "0-0"
    print(f"[alert_sender] alerts_stream={alerts_stream}")
    print(f"[alert_sender] POST {alerts_url}")

    while True:
        streams = rdb.xread({alerts_stream: last_id}, block=args.block_ms, count=args.count)
        if not streams:
            continue

        for mid, fields in parse_xread(streams):
            last_id = mid
            js = fields.get(b"json", b"{}")
            try:
                alert = json.loads(b2s(js))
            except Exception:
                continue

            payload = {
                "cam_id": alert.get("cam_id", cam_id),
                "person_track_id": alert.get("person_track_id", -1),
                "frame_id_end": alert.get("frame_id_end", -1),
                "stamp_ns_end": alert.get("stamp_ns_end", 0),
                "score": alert.get("score", None),
                "reason": alert.get("reason", {}),
                "model_version": alert.get("model_version", "unknown"),
                "object_track_id": alert.get("object_track_id", -1),
                "object_class_id": alert.get("object_class_id", -1),
                "raw": alert,
            }

            try:
                r = requests.post(alerts_url, json=payload, timeout=timeout_s)
                ok = (200 <= r.status_code < 300)
                print(f"[alert_sender] sent alert mid={mid} status={r.status_code} ok={ok}")
            except Exception as e:
                print(f"[alert_sender] webhook error mid={mid}: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[alert_sender] stopping...")
