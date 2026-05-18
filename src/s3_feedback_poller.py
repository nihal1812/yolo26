#!/usr/bin/env python3
"""
s3_feedback_poller.py

Polls labeled training samples from S3 and publishes each new sample over ZMQ.

Important:
- This script does NOT save training JSON payloads anymore.
- data_collector_llm.py is responsible for saving, batching, and publishing to trainer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import boto3
import zmq
import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def now_ns() -> int:
    return time.time_ns()


def safe_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def load_seen(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return set(str(x) for x in data)
        if isinstance(data, dict) and isinstance(data.get("seen"), list):
            return set(str(x) for x in data["seen"])
    except Exception:
        pass
    return set()


def save_seen(path: Path, seen: set):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"updated_ns": now_ns(), "seen": sorted(seen)}, indent=2))
    tmp.replace(path)


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def local_connect_addr(bind_addr: str) -> str:
    if bind_addr.startswith("tcp://*:"):
        return bind_addr.replace("tcp://*:", "tcp://127.0.0.1:", 1)
    if bind_addr.startswith("tcp://0.0.0.0:"):
        return bind_addr.replace("tcp://0.0.0.0:", "tcp://127.0.0.1:", 1)
    return bind_addr


class ZmqPublisher:
    def __init__(self, bind: Optional[str], connect: Optional[str], topic: str, sndhwm: int = 1000):
        if not bind and not connect:
            raise ValueError("ZMQ publisher needs zmq_bind or zmq_connect")

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SNDHWM, int(sndhwm))

        self.topic = str(topic)
        self.topic_b = self.topic.encode("utf-8")

        self.bind = bind
        self.connect = connect

        if bind:
            self.sock.bind(bind)
        if connect:
            self.sock.connect(connect)

        # PUB/SUB needs a small warmup to avoid first-message loss.
        time.sleep(0.5)

        print(f"[s3_feedback] ZMQ PUB bind={bind} connect={connect} topic={self.topic}")

    def publish(self, payload: dict) -> bool:
        header = {
            "type": "s3_feedback",
            "event_id": payload.get("event_id"),
            "sample_id": payload.get("sample_id"),
            "dataset_class": payload.get("dataset_class"),
            "label": payload.get("label"),
            "s3_uri": payload.get("s3_uri"),
            "stamp_ns": now_ns(),
        }
        try:
            self.sock.send_multipart(
                [
                    self.topic_b,
                    json.dumps(header, sort_keys=True).encode("utf-8"),
                    json.dumps(payload, sort_keys=True).encode("utf-8"),
                ],
                flags=zmq.NOBLOCK,
            )
            return True
        except zmq.Again:
            print(f"[s3_feedback] ZMQ HWM drop sample={payload.get('sample_id')}")
            return False
        except Exception as exc:
            print(f"[s3_feedback] ZMQ publish failed: {type(exc).__name__}: {exc}")
            return False

    def close(self):
        try:
            self.sock.close(0)
        except Exception:
            pass


def list_s3_objects(s3, bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key")
            if key:
                yield obj


def parse_dataset_key(key: str, base_prefix: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Expected:
      datasets/training/2026-05-08/not_theft/clips/<sample>.mp4
      datasets/training/2026-05-08/not_theft/frames/<sample>/frame_0001.jpg

    Returns:
      dataset_date, dataset_class, media_type, sample_id
    """
    base = base_prefix.strip("/").split("/")
    parts = key.strip("/").split("/")

    if len(parts) < len(base) + 4:
        return None
    if parts[: len(base)] != base:
        return None

    rest = parts[len(base):]
    dataset_date = rest[0]
    dataset_class = rest[1]
    media_type = rest[2]

    if media_type == "clips":
        sample_file = rest[3]
        sample_id = sample_file.rsplit(".", 1)[0]
        return dataset_date, dataset_class, media_type, sample_id

    if media_type == "frames":
        sample_id = rest[3]
        return dataset_date, dataset_class, media_type, sample_id

    return None


def build_payload(
    *,
    bucket: str,
    region: str,
    key: str,
    dataset_date: str,
    dataset_class: str,
    sample_id: str,
    class_meta: dict,
) -> dict:
    label = class_meta.get("label")
    confidence = class_meta.get("confidence", 1.0)
    needs_review = safe_bool(class_meta.get("needs_review", False), False)

    uri = s3_uri(bucket, key)
    event_id = f"s3:{dataset_date}:{dataset_class}:{sample_id}"

    frames_prefix = key.replace(f"/clips/{sample_id}.mp4", f"/frames/{sample_id}/")
    if frames_prefix == key:
        frames_prefix = None

    return {
        "type": "feedback",
        "source": "s3_dataset",
        "dataset_mode": True,

        "event_id": event_id,
        "sample_id": sample_id,
        "dataset_date": dataset_date,
        "dataset_class": dataset_class,
        "media_type": "clips",

        "label": label,
        "confidence": confidence,
        "needs_review": needs_review,
        "category": dataset_class,

        "s3_bucket": bucket,
        "s3_region": region,
        "s3_key": key,
        "s3_uri": uri,
        "frames_prefix": frames_prefix,

        # Synthetic placeholders. data_collector_llm accepts these for dataset mode.
        "cam_id": "s3_dataset",
        "person_track_id": -1,
        "frame_id_end": -1,
        "stamp_ns_end": now_ns(),

        "comment": f"S3 dataset feedback: {dataset_class} / {sample_id}",
    }


def scan_once(cfg: dict, publisher: ZmqPublisher, replay: bool = False) -> int:
    fb_cfg = cfg.get("feedback_s3") or cfg.get("s3_feedback") or {}

    if not bool(fb_cfg.get("enabled", False)):
        print("[s3_feedback] disabled in config")
        return 0

    bucket = fb_cfg.get("bucket")
    region = fb_cfg.get("region", "eu-central-1")
    root_prefix = str(fb_cfg.get("prefix", "datasets/training")).strip("/")
    dataset_date = str(fb_cfg.get("date", "")).strip()

    if not bucket:
        raise ValueError("feedback_s3.bucket missing")
    if not dataset_date:
        raise ValueError("feedback_s3.date missing")

    scan_prefix = f"{root_prefix}/{dataset_date}/"
    state_file = Path(
        fb_cfg.get(
            "state_file",
            "/home/yahboom/zono/yolo26/logs/s3_feedback_seen.json",
        )
    )

    classes = fb_cfg.get("classes", {}) or {}
    if not classes:
        raise ValueError("feedback_s3.classes missing")

    seen = set() if replay else load_seen(state_file)

    print(
        f"[s3_feedback] started bucket={bucket} prefix={scan_prefix} "
        f"region={region} state={state_file}"
    )

    s3 = boto3.client("s3", region_name=region)

    objects_scanned = 0
    clips_seen = 0
    new_count = 0
    published_count = 0

    for obj in list_s3_objects(s3, bucket, scan_prefix):
        objects_scanned += 1
        key = obj["Key"]
        parsed = parse_dataset_key(key, root_prefix)
        if not parsed:
            continue

        d_date, d_class, media_type, sample_id = parsed

        if d_date != dataset_date:
            continue
        if d_class not in classes:
            print(f"[s3_feedback] skipping unknown class key={key}")
            continue
        if media_type != "clips":
            continue
        if not key.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".npz")):
            continue

        clips_seen += 1
        class_meta = classes[d_class] or {}

        payload = build_payload(
            bucket=bucket,
            region=region,
            key=key,
            dataset_date=d_date,
            dataset_class=d_class,
            sample_id=sample_id,
            class_meta=class_meta,
        )

        dedup_id = payload["event_id"]

        if dedup_id in seen:
            continue

        new_count += 1

        ok = publisher.publish(payload)
        if ok:
            published_count += 1
            seen.add(dedup_id)
            print(
                f"[s3_feedback] published class={d_class} sample={sample_id} "
                f"label={payload.get('label')} s3={payload.get('s3_uri')}"
            )

    if not replay:
        save_seen(state_file, seen)

    print(
        f"[s3_feedback] scan complete objects={objects_scanned} "
        f"clips={clips_seen} new={new_count} published={published_count}"
    )

    return published_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--replay", action="store_true", help="Ignore state file and republish matching samples.")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    fb_cfg = cfg.get("feedback_s3") or cfg.get("s3_feedback") or {}

    if not bool(fb_cfg.get("enabled", False)):
        print("[s3_feedback] disabled in config")
        return

    bind = fb_cfg.get("zmq_bind", "tcp://*:5692")
    connect = fb_cfg.get("zmq_connect")
    topic = fb_cfg.get("topic", "feedback.s3_dataset")
    sndhwm = int(fb_cfg.get("sndhwm", 1000))

    pub = ZmqPublisher(bind=bind, connect=connect, topic=topic, sndhwm=sndhwm)

    try:
        while True:
            try:
                scan_once(cfg, pub, replay=args.replay)
            except Exception as exc:
                print(f"[s3_feedback] ERROR: {type(exc).__name__}: {exc}")

            if args.once:
                break

            poll_s = float(fb_cfg.get("poll_interval_s", 60))
            time.sleep(max(1.0, poll_s))
    finally:
        pub.close()


if __name__ == "__main__":
    main()
