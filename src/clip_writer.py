#!/usr/bin/env python3
import argparse, json, os, time, threading
from pathlib import Path

import yaml, redis
import boto3
from botocore.client import Config

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

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_s():
    return time.time()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--scan_every_s", type=float, default=10.0)
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

    clips_stream = r_cfg.get("clips_stream", f"clips:{cam_id}")

    p3 = cfg.get("pipeline3", {})
    clips_cfg = p3.get("clips", {})
    out_dir = Path(clips_cfg.get("out_dir", "clips_cache"))
    retention_h = float(clips_cfg.get("retention_hours", 48))
    retention_s = retention_h * 3600.0
    fmt = clips_cfg.get("write_format", "npz").lower()

    safe_mkdir(out_dir)

    # S3 config
    s3_cfg = p3.get("s3", {})
    s3_enabled = bool(s3_cfg.get("enabled", False))
    bucket = s3_cfg.get("bucket")
    prefix = s3_cfg.get("prefix", f"{cam_id}/")
    region = s3_cfg.get("region", None)
    endpoint_url = s3_cfg.get("endpoint_url", None)
    access_key = s3_cfg.get("access_key", None)
    secret_key = s3_cfg.get("secret_key", None)
    verify_ssl = bool(s3_cfg.get("verify_ssl", True))

    s3 = None
    if s3_enabled:
        if not bucket:
            raise RuntimeError("pipeline3.s3.bucket missing while s3.enabled=true")
        session = boto3.session.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        s3 = session.client(
            "s3",
            endpoint_url=endpoint_url,
            verify=verify_ssl,
            config=Config(signature_version="s3v4"),
        )

    print(f"[clip_writer] clips_stream={clips_stream}")
    print(f"[clip_writer] out_dir={out_dir} retention_hours={retention_h} format={fmt}")
    print(f"[clip_writer] s3_enabled={s3_enabled} bucket={bucket} prefix={prefix}")

    last_id = "0-0"
    pending_upload = set()
    lock = threading.Lock()

    def clip_filename(meta: dict):
        cam = meta.get("cam_id", cam_id)
        pid = meta.get("person_track_id", -1)
        fid = meta.get("frame_id_end", -1)
        tsn = meta.get("stamp_ns_end", 0)
        base = f"{cam}_pid{pid}_fid{fid}_ts{tsn}"
        return base + (".npz" if fmt == "npz" else ".bin")

    def write_clip_file(meta: dict, clip_bytes: bytes):
        fn = clip_filename(meta)
        path = out_dir / fn
        if fmt == "npz":
            # store meta.json + raw float32 bytes
            import numpy as np
            C = int(meta.get("C", 19)); T = int(meta.get("T", 16)); H = int(meta.get("H", 112)); W = int(meta.get("W", 112))
            arr = np.frombuffer(clip_bytes, dtype=np.float32)
            if arr.size == C*T*H*W:
                arr = arr.reshape((C, T, H, W))
                np.savez_compressed(path, meta_json=json.dumps(meta), clip=arr)
            else:
                # fallback
                with open(path, "wb") as f:
                    f.write(clip_bytes)
        else:
            # raw bytes only, meta in sidecar json
            with open(path, "wb") as f:
                f.write(clip_bytes)
            with open(str(path) + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

        return path

    def s3_key_for(path: Path):
        return f"{prefix}{path.name}"

    def upload_worker():
        while True:
            time.sleep(0.2)
            if not s3_enabled or s3 is None:
                continue

            with lock:
                items = list(pending_upload)

            for p in items:
                p = Path(p)
                if not p.exists():
                    with lock:
                        pending_upload.discard(str(p))
                    continue
                try:
                    key = s3_key_for(p)
                    s3.upload_file(str(p), bucket, key)
                    # success -> delete local
                    p.unlink(missing_ok=True)
                    # also delete sidecar if exists
                    side = Path(str(p) + ".json")
                    side.unlink(missing_ok=True)

                    with lock:
                        pending_upload.discard(str(p))
                    print(f"[clip_writer] uploaded+deleted {p.name} -> s3://{bucket}/{key}")
                except Exception as e:
                    # keep for retry until TTL cleanup
                    print(f"[clip_writer] upload failed {p.name}: {e}")

    def cleanup_old_files():
        cutoff = now_s() - retention_s
        for p in out_dir.glob("*"):
            try:
                if not p.is_file():
                    continue
                mtime = p.stat().st_mtime
                if mtime < cutoff:
                    p.unlink(missing_ok=True)
                    print(f"[clip_writer] TTL deleted {p.name}")
            except Exception:
                pass

    # start uploader thread
    t = threading.Thread(target=upload_worker, daemon=True)
    t.start()

    last_cleanup = 0.0
    while True:
        streams = rdb.xread({clips_stream: last_id}, block=args.block_ms, count=args.count)
        if not streams:
            if (now_s() - last_cleanup) > args.scan_every_s:
                cleanup_old_files()
                last_cleanup = now_s()
            continue

        for mid, fields in parse_xread(streams):
            last_id = mid
            meta_js = fields.get(b"meta", b"{}")
            clip_bytes = fields.get(b"clip", None)
            if clip_bytes is None:
                continue
            try:
                meta = json.loads(b2s(meta_js))
            except Exception:
                meta = {}

            path = write_clip_file(meta, clip_bytes)
            with lock:
                pending_upload.add(str(path))

        if (now_s() - last_cleanup) > args.scan_every_s:
            cleanup_old_files()
            last_cleanup = now_s()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[clip_writer] stopping...")
