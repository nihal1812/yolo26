#!/usr/bin/env python3
import argparse
import json
import time
import threading
from pathlib import Path

import redis
import boto3
import requests
from botocore.client import Config

from config_utils import load_cfg, get_stream


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


def resolve_clip_refs_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "clip_refs_enriched", cam_id)
    except Exception:
        try:
            return get_stream(cfg, "clip_refs", cam_id)
        except Exception:
            return f"clip_refs:{cam_id}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)
    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--scan_every_s", type=float, default=10.0)
    ap.add_argument("--clip_refs_stream", default=None)
    ap.add_argument("--frontend_notify_retries", type=int, default=3)
    ap.add_argument("--frontend_notify_backoff_s", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    r_cfg = cfg.get("redis", {})
    redis_maxlen = int(r_cfg.get("maxlen", 50000)) if "maxlen" in r_cfg else 50000

    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
    )
    rdb.ping()

    clips_stream = get_stream(cfg, "clips", cam_id)
    clip_refs_stream = args.clip_refs_stream or resolve_clip_refs_stream(cfg, cam_id)

    p3 = cfg.get("pipeline3", {})
    clips_cfg = p3.get("clips", {})
    out_dir = Path(clips_cfg.get("out_dir", "clips_cache")) / cam_id
    retention_h = float(clips_cfg.get("retention_hours", 48))
    retention_s = retention_h * 3600.0
    early_delete_h = float(clips_cfg.get("early_delete_after_success_hours", 24))
    early_delete_s = early_delete_h * 3600.0
    fmt = str(clips_cfg.get("write_format", "npz")).lower()

    safe_mkdir(out_dir)

    wh = p3.get("webhooks", {})
    clip_notify_url = wh.get("clips_url", None) or wh.get("clip_url", None)
    timeout_s = float(wh.get("timeout_s", 3.0))

    s3_cfg = p3.get("s3", {})
    s3_enabled = bool(s3_cfg.get("enabled", False))
    bucket = s3_cfg.get("bucket")
    prefix = str(s3_cfg.get("prefix", "{cam}/")).format(cam=cam_id)
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
            region_name=region,
        )
        s3 = session.client(
            "s3",
            endpoint_url=endpoint_url,
            verify=verify_ssl,
            config=Config(signature_version="s3v4"),
        )

    print(f"[clip_writer] cam_id={cam_id}")
    print(f"[clip_writer] clips_stream={clips_stream}")
    print(f"[clip_writer] clip_refs_stream={clip_refs_stream}")
    print(
        f"[clip_writer] out_dir={out_dir} retention_hours={retention_h} "
        f"early_delete_after_success_hours={early_delete_h} format={fmt}"
    )
    print(f"[clip_writer] s3_enabled={s3_enabled} bucket={bucket} prefix={prefix}")
    print(f"[clip_writer] clip_notify_url={clip_notify_url}")

    last_id = "0-0"
    pending_upload = set()
    lock = threading.Lock()

    def clip_filename(meta: dict):
        cam = meta.get("cam_id", cam_id)
        pid = meta.get("person_track_id", -1)
        fid = meta.get("frame_id_end", -1)
        tsn = meta.get("stamp_ns_end", 0)
        ext = ".npz" if fmt == "npz" else ".bin"
        return f"{cam}_pid{pid}_fid{fid}_ts{tsn}{ext}"

    def sidecar_path_for(path: Path) -> Path:
        return Path(str(path) + ".state.json")

    def meta_path_for(path: Path) -> Path:
        return Path(str(path) + ".meta.json")

    def load_state(path: Path) -> dict:
        sp = sidecar_path_for(path)
        if not sp.exists():
            return {}
        try:
            with open(sp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_state(path: Path, state: dict):
        sp = sidecar_path_for(path)
        tmp = Path(str(sp) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        tmp.replace(sp)

    def update_state(path: Path, **patch):
        st = load_state(path)
        st.update(patch)
        save_state(path, st)
        return st

    def write_clip_file(meta: dict, clip_bytes: bytes):
        fn = clip_filename(meta)
        path = out_dir / fn

        if fmt == "npz":
            import numpy as np

            c = int(meta.get("C", 19))
            t = int(meta.get("T", 16))
            h = int(meta.get("H", 112))
            w = int(meta.get("W", 112))
            arr = np.frombuffer(clip_bytes, dtype=np.float32)

            if arr.size == c * t * h * w:
                arr = arr.reshape((c, t, h, w))
                np.savez_compressed(path, meta_json=json.dumps(meta), clip=arr)
            else:
                with open(path, "wb") as f:
                    f.write(clip_bytes)
        else:
            with open(path, "wb") as f:
                f.write(clip_bytes)
            with open(str(path) + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

        return path

    def s3_key_for(path: Path):
        return f"{prefix}{path.name}"

    def read_meta(path: Path):
        mp = meta_path_for(path)
        if not mp.exists():
            return {}
        try:
            with open(mp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def build_clip_ref(meta: dict, local_path: Path, clip_url: str = "", storage_status: str = "saved_local"):
        state = load_state(local_path)
        return {
            "type": "clip_ref",
            "event_id": meta.get("event_id"),
            "cam_id": meta.get("cam_id", cam_id),
            "person_track_id": meta.get("person_track_id", -1),
            "global_person_id": meta.get("global_person_id", None),
            "frame_id_end": meta.get("frame_id_end", -1),
            "stamp_ns_end": meta.get("stamp_ns_end", 0),
            "object_track_id": meta.get("object_track_id", -1),
            "object_class_id": meta.get("object_class_id", -1),
            "local_clip_path": str(local_path),
            "clip_filename": local_path.name,
            "clip_url": clip_url,
            "storage_status": storage_status,
            "format": fmt,
            "upload_ok": bool(state.get("uploaded", False)),
            "frontend_notify_ok": bool(state.get("frontend_notified", False)),
            "saved_at_s": state.get("saved_at_s", None),
            "uploaded_at_s": state.get("uploaded_at_s", None),
            "frontend_notified_at_s": state.get("frontend_notified_at_s", None),
            "raw_meta": meta,
        }

    def publish_clip_ref(ref_obj: dict):
        try:
            fields = {
                "event_id": str(ref_obj.get("event_id", "")),
                "cam_id": str(ref_obj.get("cam_id", cam_id)),
                "person_track_id": str(ref_obj.get("person_track_id", -1)),
                "frame_id": str(ref_obj.get("frame_id_end", -1)),
                "stamp_ns": str(ref_obj.get("stamp_ns_end", 0)),
                "json": json.dumps(ref_obj),
            }
            gid = ref_obj.get("global_person_id", None)
            if gid is not None:
                fields["global_person_id"] = str(gid)

            rdb.xadd(
                clip_refs_stream,
                fields,
                maxlen=redis_maxlen,
                approximate=True,
            )
        except Exception as e:
            print(f"[clip_writer] cam={cam_id} failed to publish clip_ref: {e}")

    def notify_frontend_once(ref_obj: dict):
        if not clip_notify_url:
            return False

        headers = {}
        event_id = ref_obj.get("event_id")
        if event_id:
            headers["Idempotency-Key"] = str(event_id)

        try:
            r = requests.post(clip_notify_url, json=ref_obj, timeout=timeout_s, headers=headers)
            ok = 200 <= r.status_code < 300
            print(
                f"[clip_writer] cam={cam_id} notified frontend "
                f"event_id={ref_obj.get('event_id')} status={r.status_code} ok={ok}"
            )
            return ok
        except Exception as e:
            print(f"[clip_writer] cam={cam_id} frontend notify failed event_id={ref_obj.get('event_id')}: {e}")
            return False

    def notify_frontend(ref_obj: dict):
        for attempt in range(args.frontend_notify_retries):
            ok = notify_frontend_once(ref_obj)
            if ok:
                return True
            if attempt < args.frontend_notify_retries - 1:
                time.sleep(args.frontend_notify_backoff_s * (2 ** attempt))
        return False

    def upload_worker():
        while True:
            time.sleep(0.2)
            if not s3_enabled or s3 is None:
                continue

            with lock:
                items = list(pending_upload)

            for p_str in items:
                p = Path(p_str)
                if not p.exists():
                    with lock:
                        pending_upload.discard(p_str)
                    continue

                meta = read_meta(p)
                st = load_state(p)
                if st.get("uploaded", False):
                    with lock:
                        pending_upload.discard(p_str)
                    continue

                try:
                    key = s3_key_for(p)
                    s3.upload_file(str(p), bucket, key)

                    update_state(
                        p,
                        uploaded=True,
                        uploaded_at_s=now_s(),
                        s3_bucket=bucket,
                        s3_key=key,
                    )

                    with lock:
                        pending_upload.discard(p_str)

                    print(f"[clip_writer] uploaded {p.name} -> s3://{bucket}/{key}")

                    ref_obj = build_clip_ref(
                        meta=meta,
                        local_path=p,
                        clip_url="",
                        storage_status="uploaded",
                    )
                    publish_clip_ref(ref_obj)

                    if not st.get("frontend_notified", False):
                        notify_ok = notify_frontend(ref_obj)
                        if notify_ok:
                            update_state(
                                p,
                                frontend_notified=True,
                                frontend_notified_at_s=now_s(),
                            )
                            ref_obj2 = build_clip_ref(
                                meta=meta,
                                local_path=p,
                                clip_url="",
                                storage_status="uploaded_and_notified",
                            )
                            publish_clip_ref(ref_obj2)

                except Exception as e:
                    print(f"[clip_writer] upload failed {p.name}: {e}")

    def should_delete_early(state: dict, age_s: float) -> bool:
        return (
            bool(state.get("uploaded", False))
            and bool(state.get("frontend_notified", False))
            and age_s >= early_delete_s
        )

    def delete_clip_group(path: Path):
        siblings = [
            path,
            meta_path_for(path),
            sidecar_path_for(path),
            Path(str(path) + ".json"),
        ]
        for p in siblings:
            try:
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception:
                pass

    def cleanup_old_files():
        now = now_s()
        for p in out_dir.glob("*"):
            try:
                if not p.is_file():
                    continue
                if p.name.endswith(".meta.json") or p.name.endswith(".state.json") or p.name.endswith(".json"):
                    continue

                st = load_state(p)
                saved_at_s = st.get("saved_at_s", None)
                if saved_at_s is None:
                    age_s = now - p.stat().st_mtime
                else:
                    age_s = now - float(saved_at_s)

                if age_s >= retention_s:
                    delete_clip_group(p)
                    print(f"[clip_writer] hard TTL deleted {p.name}")
                    continue

                if should_delete_early(st, age_s):
                    delete_clip_group(p)
                    print(f"[clip_writer] early deleted after success {p.name}")

            except Exception:
                pass

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

            try:
                with open(meta_path_for(path), "w", encoding="utf-8") as f:
                    json.dump(meta, f)
            except Exception:
                pass

            save_state(path, {
                "event_id": meta.get("event_id"),
                "saved_local": True,
                "saved_at_s": now_s(),
                "uploaded": False,
                "uploaded_at_s": None,
                "frontend_notified": False,
                "frontend_notified_at_s": None,
            })

            ref_obj = build_clip_ref(
                meta=meta,
                local_path=path,
                clip_url="",
                storage_status="saved_local",
            )
            publish_clip_ref(ref_obj)

            notify_ok = notify_frontend(ref_obj)
            if notify_ok:
                update_state(
                    path,
                    frontend_notified=True,
                    frontend_notified_at_s=now_s(),
                )
                ref_obj2 = build_clip_ref(
                    meta=meta,
                    local_path=path,
                    clip_url="",
                    storage_status="saved_local_and_notified",
                )
                publish_clip_ref(ref_obj2)

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