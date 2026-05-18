#!/usr/bin/env python3
"""
data_collector_llm.py

Pipeline data collector for the theft-detection learning system.

This worker is still created per camera by brain.py because raw realtime inputs
(scores, decisions, alerts, feedback, clip references, and scalar-clip events)
are naturally camera-specific.

Important production design:
- The collector is per camera.
- The trainer is NOT per camera.
- Every collector publishes training events to ONE shared ZMQ topic, for example
  train_events.theft.
- The fleet/global TrainerWorker subscribes once to that shared topic and builds
  one shared model across cameras/stores.
- cam_id and site_id/store_id are kept as metadata only. They are useful for
  debugging, per-camera/per-store reports, filtering bad cameras later, and
  threshold overrides, but they are not used as model identities.

Clip-storage design:
- Clips should be uploaded by the upstream clip/data collector pipeline to S3.
- This collector does not keep clips on device.
- Each training event should carry the unique clip location in:

    payload.clip_ref.s3_uri

  or payload.clip_ref.bucket + payload.clip_ref.key.

- The trainer will download that clip temporarily from S3 only when needed,
  load it, and delete the local temporary file.

Typical data flow:

    Redis/ZMQ camera inputs
        -> DataCollectorWorker(cam_id)
        -> shared ZMQ topic train_events.theft
        -> TrainerWorker(target_id=global_retail, model_scope=fleet)
        -> S3 shared model registry

Recommended config.yaml sections:

  data_collector_args:
    # Existing per-camera input settings can remain here.
    use_scalars_clip_zmq: true

    # New shared output to trainer. With multiple collectors, prefer connect
    # here and let the trainer SUB socket bind, or use a single proxy.
    train_events_zmq_enabled: true
    train_events_zmq_mode: pub
    train_events_zmq_bind: null
    train_events_zmq_connect: tcp://127.0.0.1:5690
    train_events_topic: train_events.theft
    publish_train_events_redis: false

    # Optional metadata added to every event.
    site_id: store_001
    fleet_id: global_retail
    target_id: global_retail

  trainer_node:
    target_id: global_retail
    model_scope: fleet
    require_site_id: false
    zmq:
      input_mode: sub
      input_bind: tcp://*:5690
      input_connect: null
      input_topic: train_events.theft

Inputs consumed/subscribed by this collector:
- Redis stream: scores_enriched:{cam} or scores:{cam}
- Redis stream: decisions_enriched:{cam} or decisions:{cam}
- Redis stream: alerts_enriched:{cam} or alerts:{cam}
- Redis stream: feedback:{cam}
- Redis stream: clip_refs_enriched:{cam} or clip_refs:{cam}
- Optional ZMQ SUB: scalars_clip_events topic from feature_builder

Output produced by this collector:
- Preferred: ZMQ PUB/PUSH training events on train_events_topic
- Optional legacy compatibility: Redis stream train_events:{cam}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path
from collections import deque
from typing import Any, Dict, Optional, Tuple, List

import redis
import zmq

try:
    from .config_utils import (
        load_cfg,
        get_stream,
        get_zmq_endpoint,
        local_connect_addr,
    )
except Exception:
    from config_utils import (
        load_cfg,
        get_stream,
        get_zmq_endpoint,
        local_connect_addr,
    )


LOG = logging.getLogger("data_collector")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper(), logging.INFO), format="%(message)s")


def log_event(level: int, message: str, **fields: Any) -> None:
    payload = {"ts": time.time(), "level": logging.getLevelName(level), "component": "data_collector", "message": message}
    payload.update(fields)
    LOG.log(level, json.dumps(payload, sort_keys=True))


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


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


def safe_bool(v, default=False):
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


def now_ns():
    return time.time_ns()


def parse_xread(streams):
    out = []
    for sname, msgs in streams:
        s = b2s(sname)
        for mid, fields in msgs:
            out.append((s, b2s(mid), fields))
    return out


def resolve_scores_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "scores_enriched", cam_id)
    except Exception:
        return get_stream(cfg, "scores", cam_id)


def resolve_decisions_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "decisions_enriched", cam_id)
    except Exception:
        return get_stream(cfg, "decisions", cam_id)


def resolve_alerts_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "alerts_enriched", cam_id)
    except Exception:
        return get_stream(cfg, "alerts", cam_id)


def resolve_clip_refs_stream(cfg, cam_id: str):
    try:
        return get_stream(cfg, "clip_refs_enriched", cam_id)
    except Exception:
        try:
            return get_stream(cfg, "clip_refs", cam_id)
        except Exception:
            return f"clip_refs:{cam_id}"


def extract_feedback_text(obj: dict) -> str:
    if not isinstance(obj, dict):
        return ""
    candidates = [
        obj.get("comment"), obj.get("description"), obj.get("feedback_text"),
        obj.get("text"), obj.get("notes"), obj.get("reason"),
        obj.get("explanation"), obj.get("summary"),
    ]
    parts = []
    for item in candidates:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            parts.append(s)
    return "\n".join(parts).strip()


def summarize_reasoning_steps(text: str) -> List[str]:
    if not text:
        return []
    parts = []
    for block in text.splitlines():
        block = block.strip(" \t-")
        if not block:
            continue
        sub = [x.strip() for x in block.replace(";", ".").split(".")]
        for s in sub:
            if len(s) >= 6:
                parts.append(" ".join(s.split()))
    return parts[:8]


def validate_feedback_schema(x: dict, allowed_categories: set) -> Optional[dict]:
    if not isinstance(x, dict):
        return None
    label = safe_int(x.get("label", None), None)
    if label not in (0, 1):
        return None
    category = x.get("category", None)
    if category is not None:
        category = str(category).strip().lower()
        if allowed_categories and category not in allowed_categories:
            category = "needs_review" if "needs_review" in allowed_categories else None
    comment = x.get("comment", None)
    if comment is not None:
        comment = str(comment).strip()
    confidence = safe_float(x.get("confidence", 0.0), 0.0)
    confidence = max(0.0, min(1.0, confidence))
    needs_review = bool(x.get("needs_review", False))
    reasoning_summary = x.get("reasoning_summary", [])
    if not isinstance(reasoning_summary, list):
        reasoning_summary = []
    reasoning_summary = [str(v).strip() for v in reasoning_summary if str(v).strip()]
    if not reasoning_summary and comment:
        reasoning_summary = summarize_reasoning_steps(comment)
    return {
        "label": int(label),
        "category": category,
        "comment": comment,
        "confidence": confidence,
        "needs_review": needs_review,
        "reasoning_summary": reasoning_summary,
    }


def heuristic_feedback_parse(feedback_text: str, allowed_categories: set) -> Optional[dict]:
    text = (feedback_text or "").strip()
    if not text:
        return None
    low = text.lower()
    theft_cues = ["theft", "steal", "stole", "shoplift", "shoplifting", "conceal", "concealed", "hid", "hidden", "without paying", "did not pay", "didn't pay", "passed checkout", "left with item"]
    benign_cues = ["not theft", "false alarm", "normal", "benign", "no theft", "returned the item", "put it back", "paid for it", "already owned", "own bag", "customer checked"]
    uncertain_cues = ["not sure", "unclear", "maybe", "possibly", "might have", "uncertain", "ambiguous"]
    theft_hits = sum(1 for k in theft_cues if k in low)
    benign_hits = sum(1 for k in benign_cues if k in low)
    uncertain_hits = sum(1 for k in uncertain_cues if k in low)
    reasoning_summary = summarize_reasoning_steps(text)
    if theft_hits > benign_hits:
        return validate_feedback_schema({"label": 1, "category": "theft" if "theft" in allowed_categories else None, "comment": "Operator text indicates theft-like sequence of events", "confidence": 0.70 if uncertain_hits else 0.84, "needs_review": bool(uncertain_hits), "reasoning_summary": reasoning_summary}, allowed_categories)
    if benign_hits > theft_hits:
        category = "false_alarm" if "false_alarm" in allowed_categories else ("benign" if "benign" in allowed_categories else None)
        return validate_feedback_schema({"label": 0, "category": category, "comment": "Operator text indicates benign / false-alarm behavior", "confidence": 0.70 if uncertain_hits else 0.84, "needs_review": bool(uncertain_hits), "reasoning_summary": reasoning_summary}, allowed_categories)
    return validate_feedback_schema({"label": 0, "category": "needs_review" if "needs_review" in allowed_categories else None, "comment": "Ambiguous operator feedback; manual review recommended", "confidence": 0.30, "needs_review": True, "reasoning_summary": reasoning_summary}, allowed_categories)


def build_feedback_json_schema(allowed_categories: List[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "integer", "enum": [0, 1]},
            "category": {"type": "string", "enum": allowed_categories},
            "comment": {"type": "string"},
            "reasoning_summary": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 8},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "needs_review": {"type": "boolean"},
        },
        "required": ["label", "category", "comment", "reasoning_summary", "confidence", "needs_review"],
    }


def get_openai_client(api_key: Optional[str], base_url: Optional[str], timeout_s: float):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("OpenAI Python SDK is not installed. Install it with: pip install openai") from exc
    kwargs = {"timeout": timeout_s}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def extract_response_text(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    try:
        output = getattr(resp, "output", None)
        if not output:
            return ""
        chunks = []
        for item in output:
            content = getattr(item, "content", None) or []
            for c in content:
                c_text = getattr(c, "text", None)
                if isinstance(c_text, str) and c_text.strip():
                    chunks.append(c_text.strip())
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def call_openai_feedback_parser(feedback_text: str, allowed_categories: set, model: str, timeout_s: float, api_key: Optional[str], base_url: Optional[str]) -> Optional[dict]:
    text = (feedback_text or "").strip()
    if not text:
        return None
    ordered_categories = sorted(list(allowed_categories))
    client = get_openai_client(api_key=api_key, base_url=base_url, timeout_s=timeout_s)
    system_prompt = (
        "You normalize human operator feedback for a retail theft-detection training pipeline. "
        "Return only JSON following the schema. label=1 means theft/confirmed suspicious event. "
        "label=0 means not theft/false alarm/benign. If uncertain, choose the best label, "
        "set needs_review=true, and lower confidence."
    )
    user_prompt = f"Allowed categories: {ordered_categories}\n\nHuman operator feedback:\n{text}"
    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0,
        text={"format": {"type": "json_schema", "name": "feedback_normalization", "schema": build_feedback_json_schema(ordered_categories), "strict": True}},
    )
    raw = extract_response_text(resp)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return validate_feedback_schema(parsed, allowed_categories)


def parse_feedback_with_llm(feedback_text: str, allowed_categories: set, model: str, timeout_s: float, api_key: Optional[str], base_url: Optional[str], enable_fallback: bool = True) -> Optional[dict]:
    try:
        parsed = call_openai_feedback_parser(feedback_text, allowed_categories, model, timeout_s, api_key, base_url)
        if parsed is not None:
            return parsed
    except Exception as exc:
        log_event(logging.WARNING, "llm_feedback_parse_failed", error=str(exc))
    if enable_fallback:
        return heuristic_feedback_parse(feedback_text, allowed_categories)
    return None


class TTLCache:
    def __init__(self, ttl_s: float, max_items: int = 20000):
        self.ttl_s = float(ttl_s)
        self.max_items = int(max_items)
        self.store = {}
        self.fifo = deque(maxlen=self.max_items)

    def put(self, key, value):
        t_s = time.time()
        self.store[key] = (t_s, value)
        self.fifo.append((t_s, key))
        self.prune()

    def get(self, key):
        item = self.store.get(key, None)
        if not item:
            return None
        t_s, val = item
        if (time.time() - t_s) > self.ttl_s:
            self.store.pop(key, None)
            return None
        return val

    def prune(self):
        now = time.time()
        while self.fifo:
            t_s, k = self.fifo[0]
            if (now - t_s) <= self.ttl_s and len(self.store) <= self.max_items:
                break
            self.fifo.popleft()
            self.store.pop(k, None)


def normalize_clip_ref(obj: dict) -> Optional[dict]:
    """Normalize any clip reference shape into payload.clip_ref for the trainer."""
    if not isinstance(obj, dict):
        return None

    # Already in the preferred shape.
    if isinstance(obj.get("clip_ref"), dict):
        return dict(obj["clip_ref"])

    # Common S3 shapes.
    s3_uri = obj.get("s3_uri") or obj.get("clip_s3_uri") or obj.get("s3_path")
    if isinstance(s3_uri, str) and s3_uri.strip():
        return {"storage": "s3", "s3_uri": s3_uri.strip(), "tensor_key": obj.get("tensor_key", "clip"), "format": obj.get("format")}

    bucket = obj.get("bucket") or obj.get("s3_bucket")
    key = obj.get("key") or obj.get("s3_key")
    if bucket and key:
        return {"storage": "s3", "bucket": str(bucket), "key": str(key).lstrip("/"), "s3_uri": f"s3://{bucket}/{str(key).lstrip('/')}", "tensor_key": obj.get("tensor_key", "clip"), "format": obj.get("format")}

    # Development fallback only. The trainer can be configured to reject this.
    local_path = obj.get("local_clip_path") or obj.get("clip_path") or obj.get("path")
    if isinstance(local_path, str) and local_path.strip():
        return {"storage": "local", "local_clip_path": local_path.strip(), "tensor_key": obj.get("tensor_key", "clip"), "format": obj.get("format")}

    return None



def safe_name(s: Any) -> str:
    s = str(s or "").strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:220] or f"item_{now_ns()}"


class DatasetFeedbackBatcher:
    """
    Handles S3 dataset feedback samples.

    Design:
      - s3_feedback_poller publishes one sample at a time over ZMQ.
      - data_collector_llm receives each sample.
      - this batcher saves each normalized training event locally.
      - it publishes ONE train_batch only after min_batch_size or max_wait_s.
    """

    def __init__(self, worker: "DataCollectorWorker", cfg: dict):
        self.worker = worker
        self.cfg = cfg or {}

        self.enabled = bool(self.cfg.get("enabled", False))
        self.save_dir = Path(
            self.cfg.get(
                "save_dir",
                "/home/yahboom/zono/yolo26/logs/dataset_feedback_collector",
            )
        )
        self.pending_dir = self.save_dir / "pending"
        self.published_dir = self.save_dir / "published"
        self.rejected_dir = self.save_dir / "rejected"

        self.min_batch_size = int(self.cfg.get("min_batch_size", 20))
        self.max_wait_s = float(self.cfg.get("max_wait_s", 300.0))
        self.min_confidence = float(
            self.cfg.get(
                "min_confidence",
                getattr(worker.args, "feedback_min_confidence", 0.85),
            )
        )

        self.first_pending_s = None
        self.samples_seen = 0
        self.samples_saved = 0
        self.samples_rejected = 0
        self.batches_published = 0

        if self.enabled:
            self.pending_dir.mkdir(parents=True, exist_ok=True)
            self.published_dir.mkdir(parents=True, exist_ok=True)
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
            if self.pending_count() > 0:
                self.first_pending_s = time.time()

    def pending_files(self) -> List[Path]:
        if not self.pending_dir.exists():
            return []
        return sorted(self.pending_dir.glob("*.json"))

    def pending_count(self) -> int:
        return len(self.pending_files())

    def reject(self, obj: dict, reason: str):
        self.samples_rejected += 1
        try:
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
            event_id = obj.get("event_id") or obj.get("sample_id") or now_ns()
            out = self.rejected_dir / f"{safe_name(event_id)}__{safe_name(reason)}.json"
            out.write_text(
                json.dumps(
                    {
                        "reason": reason,
                        "received_ns": now_ns(),
                        "raw": obj,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        except Exception as exc:
            log_event(logging.WARNING, "dataset_feedback_reject_save_failed", error=str(exc), reason=reason)

    def normalize_dataset_feedback(self, obj: dict) -> Optional[dict]:
        if not isinstance(obj, dict):
            return None

        label_i = safe_int(obj.get("label", None), None)
        if label_i not in (0, 1):
            # suspicious/needs_review with label=null should not train automatically.
            self.reject(obj, "missing_or_unsupported_label")
            return None

        clip_ref = normalize_clip_ref(obj)
        if not clip_ref or clip_ref.get("storage") != "s3":
            self.reject(obj, "missing_s3_clip_ref")
            return None

        confidence = safe_float(obj.get("confidence", 1.0), 1.0)
        needs_review = safe_bool(obj.get("needs_review", False), False)

        event_id = (
            obj.get("event_id")
            or f"s3:{obj.get('dataset_date','unknown')}:{obj.get('dataset_class','unknown')}:{obj.get('sample_id', now_ns())}"
        )

        dataset_class = str(obj.get("dataset_class") or obj.get("label_name") or "").strip()
        category = obj.get("category", None) or dataset_class or ("theft" if label_i == 1 else "not_theft")

        comment = (
            obj.get("comment")
            or obj.get("feedback_text")
            or f"S3 dataset feedback: {dataset_class or category} / {obj.get('sample_id', '')}"
        )

        train_ok = (not needs_review) and confidence >= self.min_confidence

        feedback_obj = {
            "msg_id": obj.get("_zmq_msg_id") or f"s3_feedback:{event_id}",
            "event_id": event_id,
            "label": int(label_i),
            "category": category,
            "comment": str(comment),
            "reasoning_summary": obj.get("reasoning_summary", []),
            "raw_feedback_text": extract_feedback_text(obj),
            "user_id": obj.get("user_id", "s3_dataset"),
            "alert_id": obj.get("alert_id"),
            "received_ns": now_ns(),
            "llm_confidence": confidence,
            "needs_review": needs_review,
            "raw_feedback": obj,
        }

        ev = {
            "type": "train_event",
            "source": "s3_dataset_feedback",
            "dataset_mode": True,
            "event_id": event_id,
            "cam_id": obj.get("cam_id", "s3_dataset"),
            "site_id": self.worker.site_id,
            "fleet_id": self.worker.fleet_id,
            "target_id": self.worker.target_id,
            "person_track_id": safe_int(obj.get("person_track_id", -1), -1),
            "global_person_id": obj.get("global_person_id"),
            "frame_id_end": safe_int(obj.get("frame_id_end", -1), -1),
            "stamp_ns_end": safe_int(obj.get("stamp_ns_end", 0), 0),
            "schema_version": "train_event_v2",
            "created_ns": now_ns(),
            "updated_ns": now_ns(),
            "stage": "labeled",
            "train_ok": bool(train_ok),
            "train_ok_reasons": [] if train_ok else ["feedback_needs_review_or_low_confidence"],
            "refs": {
                "feedback_source": "s3_dataset",
                "s3_uri": clip_ref.get("s3_uri"),
            },
            "payload": {
                "clip_ref": clip_ref,
                "s3_dataset_feedback": {
                    "sample_id": obj.get("sample_id"),
                    "dataset_date": obj.get("dataset_date"),
                    "dataset_class": dataset_class,
                    "media_type": obj.get("media_type", "clips"),
                    "s3_bucket": obj.get("s3_bucket") or clip_ref.get("bucket"),
                    "s3_region": obj.get("s3_region"),
                    "s3_key": obj.get("s3_key") or clip_ref.get("key"),
                    "s3_uri": obj.get("s3_uri") or clip_ref.get("s3_uri"),
                    "frames_prefix": obj.get("frames_prefix"),
                    "raw": obj,
                },
            },
            "feedback": feedback_obj,
        }

        return ev

    def save_event(self, ev: dict) -> bool:
        event_id = ev.get("event_id")
        out = self.pending_dir / f"{safe_name(event_id)}.json"

        # Dedup by filename. If the sample already exists in pending, keep it.
        if out.exists():
            return False

        out.write_text(json.dumps(ev, indent=2, sort_keys=True))
        self.samples_saved += 1

        if self.first_pending_s is None:
            self.first_pending_s = time.time()

        return True

    def process(self, obj: dict):
        if not self.enabled:
            return

        self.samples_seen += 1

        ev = self.normalize_dataset_feedback(obj)
        if ev is None:
            return

        saved = self.save_event(ev)
        if saved:
            log_event(
                logging.INFO,
                "dataset_feedback_saved",
                cam_id=self.worker.cam_id,
                event_id=ev.get("event_id"),
                label=ev.get("feedback", {}).get("label"),
                pending=self.pending_count(),
            )

        self.flush_if_ready(force=False)

    def should_flush(self) -> bool:
        n = self.pending_count()
        if n <= 0:
            return False
        if n >= self.min_batch_size:
            return True
        if self.first_pending_s is not None and (time.time() - self.first_pending_s) >= self.max_wait_s:
            return True
        return False

    def flush_if_ready(self, force: bool = False):
        if not self.enabled:
            return
        if not force and not self.should_flush():
            return

        files = self.pending_files()
        if not files:
            self.first_pending_s = None
            return

        events = []
        used_files = []

        for f in files:
            try:
                events.append(json.loads(f.read_text()))
                used_files.append(f)
            except Exception as exc:
                log_event(logging.WARNING, "dataset_feedback_pending_read_failed", file=str(f), error=str(exc))

        if not events:
            return

        batch_id = f"s3_feedback_batch_{int(time.time())}_{len(events)}"

        batch = {
            "type": "train_batch",
            "source": "s3_dataset_feedback",
            "schema_version": "train_batch_v1",
            "batch_id": batch_id,
            "cam_id": self.worker.cam_id,
            "site_id": self.worker.site_id,
            "fleet_id": self.worker.fleet_id,
            "target_id": self.worker.target_id,
            "created_ns": now_ns(),
            "num_events": len(events),
            "events": events,
        }

        ok = self.worker.publish_train_batch(batch)
        if not ok:
            log_event(
                logging.WARNING,
                "dataset_feedback_batch_publish_failed",
                batch_id=batch_id,
                pending=len(events),
            )
            return

        batch_dir = self.published_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "batch.json").write_text(json.dumps(batch, indent=2, sort_keys=True))

        for f in used_files:
            try:
                f.rename(batch_dir / f.name)
            except Exception as exc:
                log_event(logging.WARNING, "dataset_feedback_pending_move_failed", file=str(f), error=str(exc))

        self.batches_published += 1
        self.first_pending_s = time.time() if self.pending_count() > 0 else None

        log_event(
            logging.INFO,
            "dataset_feedback_batch_published",
            batch_id=batch_id,
            num_events=len(events),
            topic=self.worker.train_events_topic,
        )


class DataCollectorWorker:
    def __init__(self, args):
        self.args = args
        self.cfg = load_cfg(args.config)
        self.cam_id = str(args.cam_id)

        self.site_id = getattr(args, "site_id", None) or self.cfg.get("site_id") or self.cfg.get("store_id")
        self.fleet_id = getattr(args, "fleet_id", None) or getattr(args, "target_id", None) or (self.cfg.get("trainer_node", {}) or {}).get("target_id", "global_retail")
        self.target_id = getattr(args, "target_id", None) or self.fleet_id

        r_cfg = self.cfg.get("redis", {})
        self.rdb = redis.Redis(
            host=r_cfg.get("host", "127.0.0.1"),
            port=int(r_cfg.get("port", 6379)),
            db=int(r_cfg.get("db", 0)),
            password=r_cfg.get("password", None),
            decode_responses=False,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            health_check_interval=30,
        )
        self.rdb.ping()

        self.scores_stream = args.scores_stream or resolve_scores_stream(self.cfg, self.cam_id)
        self.decisions_stream = args.decisions_stream or resolve_decisions_stream(self.cfg, self.cam_id)
        self.alerts_stream = args.alerts_stream or resolve_alerts_stream(self.cfg, self.cam_id)
        self.feedback_stream = args.feedback_stream or get_stream(self.cfg, "feedback", self.cam_id)
        self.clip_refs_stream = args.clip_refs_stream or resolve_clip_refs_stream(self.cfg, self.cam_id)
        self.train_events_stream = args.train_events_stream or get_stream(self.cfg, "train_events", self.cam_id)

        self.use_scalars_clip_zmq = bool(args.use_scalars_clip_zmq)
        self.ctx = zmq.Context.instance()

        self.sub_scalars = None
        self.scalars_topic_b = None
        self.scalars_connect = None
        self.scalars_topic = None
        if self.use_scalars_clip_zmq:
            scalars_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "scalars_clip_events")
            self.scalars_connect = args.scalars_connect or local_connect_addr(scalars_cfg["bind"])
            self.scalars_topic = args.scalars_topic or scalars_cfg["topic"]
            self.sub_scalars = self.ctx.socket(zmq.SUB)
            self.sub_scalars.setsockopt(zmq.LINGER, 0)
            self.sub_scalars.setsockopt(zmq.RCVHWM, int(args.zmq_rcvhwm))
            self.sub_scalars.connect(self.scalars_connect)
            self.sub_scalars.setsockopt(zmq.SUBSCRIBE, self.scalars_topic.encode("utf-8"))
            self.scalars_topic_b = self.scalars_topic.encode("utf-8")

        # Optional S3 dataset feedback ZMQ input.
        # Only one collector should enable this, usually cam0.
        dataset_cfg = self.cfg.get("feedback_dataset_collector", {}) or {}
        feedback_s3_cfg = self.cfg.get("feedback_s3", {}) or {}

        collector_cam_id = str(dataset_cfg.get("collector_cam_id", "cam0"))
        self.use_s3_feedback_zmq = bool(
            getattr(args, "use_s3_feedback_zmq", False)
            or dataset_cfg.get("enabled", False)
        ) and self.cam_id == collector_cam_id

        self.sub_s3_feedback = None
        self.s3_feedback_topic = None
        self.s3_feedback_connect = None

        if self.use_s3_feedback_zmq:
            self.s3_feedback_topic = (
                getattr(args, "s3_feedback_topic", None)
                or dataset_cfg.get("topic")
                or feedback_s3_cfg.get("topic")
                or "feedback.s3_dataset"
            )
            self.s3_feedback_connect = (
                getattr(args, "s3_feedback_zmq_connect", None)
                or dataset_cfg.get("zmq_connect")
                or feedback_s3_cfg.get("zmq_connect")
            )

            if not self.s3_feedback_connect:
                bind = feedback_s3_cfg.get("zmq_bind") or dataset_cfg.get("zmq_bind") or "tcp://*:5692"
                self.s3_feedback_connect = local_connect_addr(bind)

            self.sub_s3_feedback = self.ctx.socket(zmq.SUB)
            self.sub_s3_feedback.setsockopt(zmq.LINGER, 0)
            self.sub_s3_feedback.setsockopt(zmq.RCVHWM, int(args.zmq_rcvhwm))
            self.sub_s3_feedback.connect(self.s3_feedback_connect)
            self.sub_s3_feedback.setsockopt(zmq.SUBSCRIBE, self.s3_feedback_topic.encode("utf-8"))

            log_event(
                logging.INFO,
                "s3_feedback_zmq_subscribed",
                cam_id=self.cam_id,
                connect=self.s3_feedback_connect,
                topic=self.s3_feedback_topic,
            )


        # New shared ZMQ output to the fleet/global trainer.
        self.train_events_zmq_enabled = bool(getattr(args, "train_events_zmq_enabled", True))
        self.train_events_zmq_mode = str(getattr(args, "train_events_zmq_mode", "pub")).lower()
        self.train_events_zmq_bind = getattr(args, "train_events_zmq_bind", None)
        self.train_events_zmq_connect = getattr(args, "train_events_zmq_connect", None)
        self.train_events_topic = str(getattr(args, "train_events_topic", "train_events.theft"))
        self.train_events_topic_b = self.train_events_topic.encode("utf-8")
        self.publish_train_events_redis = bool(getattr(args, "publish_train_events_redis", False))
        self.train_pub = None

        if self.train_events_zmq_enabled:
            sock_type = zmq.PUB if self.train_events_zmq_mode == "pub" else zmq.PUSH
            self.train_pub = self.ctx.socket(sock_type)
            self.train_pub.setsockopt(zmq.LINGER, 0)
            self.train_pub.setsockopt(zmq.SNDHWM, int(getattr(args, "zmq_sndhwm", 512)))
            if self.train_events_zmq_bind:
                self.train_pub.bind(self.train_events_zmq_bind)
            if self.train_events_zmq_connect:
                self.train_pub.connect(self.train_events_zmq_connect)
            if not self.train_events_zmq_bind and not self.train_events_zmq_connect:
                raise ValueError("train_events_zmq_enabled=True but no train_events_zmq_bind/connect was provided")

        self.dataset_feedback_batcher = DatasetFeedbackBatcher(self, dataset_cfg)
        self.llm_api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
        self.pending_zset_prefix = r_cfg.get("pending_zset_prefix", "train:pending")
        self.last_ids = {
            self.scores_stream: "0-0",
            self.decisions_stream: "0-0",
            self.alerts_stream: "0-0",
            self.feedback_stream: "0-0",
            self.clip_refs_stream: "0-0",
        }
        self.cache = TTLCache(ttl_s=args.cache_ttl_s, max_items=50000)
        self.events_published_total = 0
        self.events_dropped_partial_total = 0
        self.events_dropped_no_clip_ref_total = 0

        log_event(logging.INFO, "collector_started", cam_id=self.cam_id, site_id=self.site_id, fleet_id=self.fleet_id, scores=self.scores_stream, decisions=self.decisions_stream, alerts=self.alerts_stream, feedback=self.feedback_stream, clip_refs=self.clip_refs_stream, train_events_topic=self.train_events_topic, train_events_zmq_enabled=self.train_events_zmq_enabled, publish_train_events_redis=self.publish_train_events_redis)

    def pending_key(self, cam, pid):
        return f"{self.pending_zset_prefix}:{cam}:pid:{pid}"

    def fuse_key(self, cam, pid, fid_end):
        return (str(cam), int(pid), int(fid_end))

    def stage_from_event(self, ev: dict) -> str:
        if ev.get("feedback") is not None:
            return "labeled"
        payload = ev.get("payload", {})
        has_score = isinstance(payload.get("score", None), dict)
        has_scal = isinstance(payload.get("scalars_clip", None), dict)
        if has_score and has_scal:
            return "ready"
        return "partial"

    def goldish_ok_for_unlabeled(self, ev: dict) -> Tuple[bool, List[str]]:
        reasons = []
        payload = ev.get("payload", {})
        dec = payload.get("decision", None)
        score_obj = payload.get("score", None)
        scal = payload.get("scalars_clip", None)
        if not isinstance(score_obj, dict):
            reasons.append("no_score")
        if not isinstance(scal, dict):
            reasons.append("no_scalars_clip")
        if reasons:
            return False, reasons
        miss_pose = None
        miss_obj = None
        if isinstance(dec, dict):
            miss_pose = safe_float(dec.get("missing_pose_ratio", None), None)
            miss_obj = safe_float(dec.get("missing_obj_ratio", None), None)
        if miss_pose is None:
            miss_pose = safe_float(score_obj.get("missing_pose_ratio", None), None)
        if miss_obj is None:
            miss_obj = safe_float(score_obj.get("missing_obj_ratio", None), None)
        if miss_pose is not None and miss_pose > float(self.args.gold_max_missing_pose_ratio):
            reasons.append("pose_missing_high")
        if miss_obj is not None and miss_obj > float(self.args.gold_max_missing_obj_ratio):
            reasons.append("obj_missing_high")
        votes = safe_int(dec.get("votes", None), None) if isinstance(dec, dict) else None
        S = safe_float(dec.get("S", None), None) if isinstance(dec, dict) else None
        gate_ok = dec.get("gate_ok", None) if isinstance(dec, dict) else None
        if self.args.gold_require_gate_ok and gate_ok is not True:
            reasons.append("gate_not_ok")
        ok_by_votes = (votes is not None and votes >= int(self.args.gold_min_votes))
        ok_by_S = (S is not None and S >= float(self.args.gold_min_S))
        if not (ok_by_votes or ok_by_S):
            reasons.append("not_strong_enough_votes_or_S")
        return (len(reasons) == 0), reasons

    def ensure_training_metadata(self, ev: dict) -> dict:
        ev.setdefault("type", "train_event")
        ev.setdefault("cam_id", self.cam_id)
        if self.site_id and not ev.get("site_id") and not ev.get("store_id"):
            ev["site_id"] = self.site_id
        if self.fleet_id and not ev.get("fleet_id"):
            ev["fleet_id"] = self.fleet_id
        if self.target_id and not ev.get("target_id"):
            ev["target_id"] = self.target_id
        ev.setdefault("schema_version", "train_event_v2")
        ev.setdefault("published_ns", now_ns())

        payload = ev.setdefault("payload", {})
        # If old score messages carried the clip pointer, lift it into the new
        # payload.clip_ref shape expected by the fleet/global trainer.
        if not isinstance(payload.get("clip_ref"), dict):
            for src_name in ("clip_ref", "score", "clip_meta"):
                src = payload.get(src_name)
                cr = normalize_clip_ref(src) if isinstance(src, dict) else None
                if cr:
                    payload["clip_ref"] = cr
                    break
        return ev

    def publish_train_event(self, ev: dict):
        ev = self.ensure_training_metadata(ev)
        ev["stage"] = self.stage_from_event(ev)
        if (not self.args.publish_partial) and ev["stage"] == "partial":
            self.events_dropped_partial_total += 1
            return
        if "train_ok" not in ev:
            ev["train_ok"] = True
        if "train_ok_reasons" not in ev:
            ev["train_ok_reasons"] = []
        if self.args.gold_filter_unlabeled_ready and ev["stage"] == "ready" and ev.get("feedback") is None:
            ok, reasons = self.goldish_ok_for_unlabeled(ev)
            ev["train_ok"] = bool(ok)
            ev["train_ok_reasons"] = reasons
            if not ok:
                return

        # Preferred output for the new TrainerWorker.
        if self.train_pub is not None:
            body = json.dumps(ev, sort_keys=True).encode("utf-8")
            try:
                if self.train_events_zmq_mode == "pub":
                    self.train_pub.send_multipart([self.train_events_topic_b, body], flags=zmq.NOBLOCK)
                else:
                    self.train_pub.send(body, flags=zmq.NOBLOCK)
                self.events_published_total += 1
            except zmq.Again:
                log_event(logging.WARNING, "train_event_zmq_hwm_drop", cam_id=self.cam_id, event_id=ev.get("event_id"))
            except Exception as exc:
                log_event(logging.ERROR, "train_event_zmq_publish_failed", cam_id=self.cam_id, event_id=ev.get("event_id"), error=str(exc))

        # Optional legacy Redis output. Keep disabled in production when the new
        # trainer is subscribed to ZMQ.
        if self.publish_train_events_redis:
            fields = {
                "event_id": ev.get("event_id", ""),
                "cam_id": ev.get("cam_id", self.cam_id),
                "site_id": ev.get("site_id", ""),
                "fleet_id": ev.get("fleet_id", self.fleet_id or ""),
                "person_track_id": str(ev.get("person_track_id", -1)),
                "frame_id_end": str(ev.get("frame_id_end", -1)),
                "stamp_ns_end": str(ev.get("stamp_ns_end", 0)),
                "stage": ev["stage"],
                "train_ok": "1" if ev.get("train_ok", True) else "0",
                "json": json.dumps(ev),
            }
            gid = ev.get("global_person_id", None)
            if gid is not None:
                fields["global_person_id"] = str(gid)
            self.rdb.xadd(self.train_events_stream, fields, maxlen=self.args.redis_maxlen, approximate=True)

    def cache_merge(self, key, patch: dict) -> dict:
        base = self.cache.get(key) or {
            "type": "train_event",
            "event_id": None,
            "cam_id": key[0],
            "site_id": self.site_id,
            "fleet_id": self.fleet_id,
            "target_id": self.target_id,
            "person_track_id": key[1],
            "frame_id_end": key[2],
            "stamp_ns_end": 0,
            "global_person_id": None,
            "refs": {},
            "payload": {},
            "feedback": None,
            "created_ns": now_ns(),
            "updated_ns": now_ns(),
        }
        base["updated_ns"] = now_ns()
        if patch.get("event_id"):
            base["event_id"] = patch["event_id"]
        if "refs" in patch and isinstance(patch["refs"], dict):
            base["refs"].update(patch["refs"])
        if "payload" in patch and isinstance(patch["payload"], dict):
            base["payload"].update(patch["payload"])
        if patch.get("stamp_ns_end"):
            base["stamp_ns_end"] = int(patch["stamp_ns_end"])
        if patch.get("feedback") is not None:
            base["feedback"] = patch["feedback"]
        if "global_person_id" in patch and patch["global_person_id"] is not None:
            base["global_person_id"] = patch["global_person_id"]
        if "train_ok" in patch:
            base["train_ok"] = patch["train_ok"]
        if "train_ok_reasons" in patch:
            base["train_ok_reasons"] = patch["train_ok_reasons"]
        self.cache.put(key, base)
        return base

    def pending_prune(self, cam, pid):
        k = self.pending_key(cam, pid)
        cutoff = time.time() - float(self.args.pending_ttl_s)
        self.rdb.zremrangebyscore(k, 0, cutoff)

    def add_pending(self, cam, pid, ref_obj: dict):
        self.pending_prune(cam, pid)
        k = self.pending_key(cam, pid)
        t = time.time()
        self.rdb.zadd(k, {json.dumps(ref_obj): t})
        self.rdb.zremrangebyrank(k, 0, -51)

    def match_pending_latest(self, cam, pid) -> Optional[dict]:
        self.pending_prune(cam, pid)
        k = self.pending_key(cam, pid)
        items = self.rdb.zrevrange(k, 0, 0)
        if not items:
            return None
        try:
            return json.loads(b2s(items[0]))
        except Exception:
            return None

    def match_event_key(self, obj: dict):
        cam = obj.get("cam_id", self.cam_id)
        pid = safe_int(obj.get("person_track_id", -1), -1)
        fid_end = safe_int(obj.get("frame_id_end", -1), -1)
        stamp_ns_end = safe_int(obj.get("stamp_ns_end", 0), 0)
        event_id = obj.get("event_id") or (f"{cam}:{pid}:{fid_end}:{stamp_ns_end}" if pid >= 0 and fid_end >= 0 else None)
        if pid < 0 or fid_end < 0:
            return None, event_id, cam, pid, fid_end, stamp_ns_end
        return self.fuse_key(cam, pid, fid_end), event_id, cam, pid, fid_end, stamp_ns_end

    def publish_train_batch(self, batch: dict) -> bool:
        """
        Publish a batched set of labeled S3 dataset samples to trainer.

        This intentionally bypasses publish_train_event(), because publish_train_event()
        is for one realtime fused event. S3 dataset feedback should be batched.
        """
        if self.train_pub is None:
            return False

        try:
            body = json.dumps(batch, sort_keys=True).encode("utf-8")
            if self.train_events_zmq_mode == "pub":
                self.train_pub.send_multipart([self.train_events_topic_b, body], flags=zmq.NOBLOCK)
            else:
                self.train_pub.send(body, flags=zmq.NOBLOCK)
            self.events_published_total += 1
            return True
        except zmq.Again:
            log_event(
                logging.WARNING,
                "train_batch_zmq_hwm_drop",
                cam_id=self.cam_id,
                batch_id=batch.get("batch_id"),
            )
            return False
        except Exception as exc:
            log_event(
                logging.ERROR,
                "train_batch_zmq_publish_failed",
                cam_id=self.cam_id,
                batch_id=batch.get("batch_id"),
                error=str(exc),
            )
            return False

    def pump_s3_feedback_zmq(self, max_messages: int = 128):
        if self.sub_s3_feedback is None:
            return

        n = 0
        while n < max_messages:
            try:
                parts = self.sub_s3_feedback.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                log_event(logging.WARNING, "s3_feedback_zmq_recv_failed", cam_id=self.cam_id, error=str(exc))
                break

            n += 1
            if len(parts) < 2:
                continue

            try:
                payload_b = parts[-1]
                obj = json.loads(payload_b.decode("utf-8"))
                obj["_zmq_topic"] = b2s(parts[0])
                obj["_zmq_received_ns"] = now_ns()
            except Exception as exc:
                log_event(logging.WARNING, "s3_feedback_zmq_bad_json", cam_id=self.cam_id, error=str(exc))
                continue

            self.dataset_feedback_batcher.process(obj)


    def process_scalars_clip_event(self, obj: dict):
        meta = obj.get("meta", {})
        feats = obj.get("features", {})
        policy_feats = obj.get("policy_features", {})
        cam = meta.get("cam_id", self.cam_id)
        pid = safe_int(meta.get("person_track_id", -1), -1)
        fid_end = safe_int(meta.get("frame_id_end", -1), -1)
        stamp_ns_end = safe_int(meta.get("stamp_ns_end", 0), 0)
        event_id = meta.get("event_id") or f"{cam}:{pid}:{fid_end}:{stamp_ns_end}"
        if pid < 0 or fid_end < 0:
            return
        key = self.fuse_key(cam, pid, fid_end)
        ev = self.cache_merge(key, {"event_id": event_id, "stamp_ns_end": stamp_ns_end, "refs": {"scalars_clip_event": "zmq"}, "payload": {"scalars_clip": {"meta": meta, "features": feats, "policy_features": policy_feats}}})
        self.publish_train_event(ev)

    def pump_scalars_zmq(self, max_messages: int = 64):
        if self.sub_scalars is None:
            return
        n = 0
        while n < max_messages:
            try:
                parts = self.sub_scalars.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:
                break
            n += 1
            if len(parts) < 2:
                continue
            try:
                payload_b = parts[-1]
                obj = json.loads(payload_b.decode("utf-8"))
            except Exception:
                continue
            self.process_scalars_clip_event(obj)

    def process_stream_message(self, sname, mid, fields):
        self.last_ids[sname] = mid
        obj = None
        if b"json" in fields:
            try:
                obj = json.loads(b2s(fields[b"json"]))
            except Exception:
                obj = None
        if not obj:
            return

        if sname == self.scores_stream:
            key, event_id, cam, pid, fid_end, stamp_ns_end = self.match_event_key(obj)
            if key is None:
                return
            refs = {"score_msg_id": mid}
            clip_ref = normalize_clip_ref(obj)
            if clip_ref:
                refs["clip_ref"] = clip_ref
            ev = self.cache_merge(key, {"event_id": event_id, "stamp_ns_end": stamp_ns_end, "global_person_id": obj.get("global_person_id", None), "refs": refs, "payload": {"score": obj, **({"clip_ref": clip_ref} if clip_ref else {})}})
            self.publish_train_event(ev)

        elif sname == self.decisions_stream:
            key, event_id, cam, pid, fid_end, stamp_ns_end = self.match_event_key(obj)
            if key is None:
                return
            ev = self.cache_merge(key, {"event_id": event_id, "stamp_ns_end": stamp_ns_end, "global_person_id": obj.get("global_person_id", None), "refs": {"decision_msg_id": mid}, "payload": {"decision": obj}})
            if bool(obj.get("will_alert", False)):
                self.add_pending(cam, pid, {"source": "decision", "event_id": event_id, "decision_msg_id": mid, "cam_id": cam, "person_track_id": pid, "global_person_id": obj.get("global_person_id", None), "frame_id_end": fid_end, "stamp_ns_end": stamp_ns_end})
            self.publish_train_event(ev)

        elif sname == self.alerts_stream:
            key, event_id, cam, pid, fid_end, stamp_ns_end = self.match_event_key(obj)
            if key is None:
                return
            ev = self.cache_merge(key, {"event_id": event_id, "stamp_ns_end": stamp_ns_end, "global_person_id": obj.get("global_person_id", None), "refs": {"alert_msg_id": mid}, "payload": {"alert": obj}})
            self.add_pending(cam, pid, {"source": "alert", "event_id": event_id, "alert_msg_id": mid, "cam_id": cam, "person_track_id": pid, "global_person_id": obj.get("global_person_id", None), "frame_id_end": fid_end, "stamp_ns_end": stamp_ns_end})
            self.publish_train_event(ev)

        elif sname == self.clip_refs_stream:
            key, event_id, cam, pid, fid_end, stamp_ns_end = self.match_event_key(obj)
            if key is None:
                return
            clip_ref = normalize_clip_ref(obj)
            payload = {"clip_ref_raw": obj}
            if clip_ref:
                payload["clip_ref"] = clip_ref
            ev = self.cache_merge(key, {"event_id": event_id, "stamp_ns_end": stamp_ns_end, "global_person_id": obj.get("global_person_id", None), "refs": {"clip_ref_msg_id": mid}, "payload": payload})
            self.publish_train_event(ev)

        elif sname == self.feedback_stream:
            self.process_feedback_message(mid, obj)

    def process_feedback_message(self, mid: str, obj: dict):
        cam = obj.get("cam_id", self.cam_id)
        pid = safe_int(obj.get("person_track_id", -1), -1)
        if pid < 0:
            return
        allowed_categories = set(x.strip().lower() for x in str(self.args.llm_allowed_categories).split(",") if x.strip())
        fid_end = safe_int(obj.get("frame_id_end", -1), -1)
        stamp_ns_end = safe_int(obj.get("stamp_ns_end", 0), 0)
        feedback_event_id = obj.get("event_id", None)
        alert_id = obj.get("alert_id", None)
        matched_key = None
        matched_event_id = feedback_event_id
        if feedback_event_id:
            parts = str(feedback_event_id).split(":")
            if len(parts) >= 4:
                try:
                    f_cam = parts[0]
                    f_pid = int(parts[1])
                    f_fid = int(parts[2])
                    matched_key = self.fuse_key(f_cam, f_pid, f_fid)
                except Exception:
                    matched_key = None
        if matched_key is None and fid_end is not None and fid_end >= 0:
            matched_key = self.fuse_key(cam, pid, fid_end)
            matched_event_id = matched_event_id or f"{cam}:{pid}:{fid_end}:{stamp_ns_end}"
        if matched_key is None:
            pend = self.match_pending_latest(cam, pid)
            if pend:
                matched_key = self.fuse_key(pend.get("cam_id", cam), pend.get("person_track_id", pid), pend.get("frame_id_end", -1))
                matched_event_id = matched_event_id or pend.get("event_id")
        parsed_feedback = None
        label_i = safe_int(obj.get("label", None), None)
        if label_i in (0, 1):
            parsed_feedback = validate_feedback_schema({"label": label_i, "category": obj.get("category", None), "comment": obj.get("comment", None) or extract_feedback_text(obj), "confidence": safe_float(obj.get("confidence", 1.0), 1.0), "needs_review": bool(obj.get("needs_review", False)), "reasoning_summary": obj.get("reasoning_summary", [])}, allowed_categories)
        if parsed_feedback is None and self.args.enable_llm_feedback_parse:
            free_text = extract_feedback_text(obj)
            if free_text:
                parsed_feedback = parse_feedback_with_llm(feedback_text=free_text, allowed_categories=allowed_categories, model=self.args.llm_model, timeout_s=self.args.llm_timeout_s, api_key=self.llm_api_key, base_url=self.args.llm_base_url, enable_fallback=(not self.args.llm_disable_fallback))
        if parsed_feedback is None:
            return
        free_text = extract_feedback_text(obj)
        feedback_obj = {"msg_id": mid, "event_id": matched_event_id, "label": int(parsed_feedback["label"]), "category": parsed_feedback.get("category"), "comment": parsed_feedback.get("comment"), "reasoning_summary": parsed_feedback.get("reasoning_summary", []), "raw_feedback_text": free_text, "user_id": obj.get("user_id", None), "alert_id": alert_id, "received_ns": now_ns(), "llm_confidence": parsed_feedback.get("confidence", 1.0), "needs_review": parsed_feedback.get("needs_review", False), "raw_feedback": obj}
        feedback_train_ok = ((not feedback_obj["needs_review"]) and (float(feedback_obj.get("llm_confidence", 0.0)) >= float(self.args.feedback_min_confidence)))
        gid_from_feedback = obj.get("global_person_id", None)
        if matched_key is not None and matched_key[2] >= 0:
            ev = self.cache_merge(matched_key, {"event_id": matched_event_id, "global_person_id": gid_from_feedback, "refs": {"feedback_msg_id": mid}, "feedback": feedback_obj, "train_ok": feedback_train_ok, "train_ok_reasons": [] if feedback_train_ok else ["feedback_needs_review_or_low_confidence"]})
            self.publish_train_event(ev)
        else:
            orphan = {"type": "train_event", "event_id": matched_event_id, "cam_id": cam, "site_id": self.site_id, "fleet_id": self.fleet_id, "target_id": self.target_id, "person_track_id": pid, "global_person_id": gid_from_feedback, "frame_id_end": -1, "stamp_ns_end": 0, "refs": {"feedback_msg_id": mid}, "payload": {"feedback_only": obj}, "feedback": feedback_obj, "created_ns": now_ns(), "updated_ns": now_ns(), "stage": "labeled", "train_ok": feedback_train_ok, "train_ok_reasons": [] if feedback_train_ok else ["feedback_needs_review_or_low_confidence"], "note": "unmatched_feedback"}
            self.publish_train_event(orphan)

    def step(self):
        try:
            self.pump_scalars_zmq()
            self.pump_s3_feedback_zmq()
            self.dataset_feedback_batcher.flush_if_ready(force=False)
            streams = self.rdb.xread(self.last_ids, block=self.args.block_ms, count=self.args.count)
            if not streams:
                self.cache.prune()
                return
            for sname, mid, fields in parse_xread(streams):
                self.process_stream_message(sname, mid, fields)
            self.pump_scalars_zmq()
            self.pump_s3_feedback_zmq()
            self.dataset_feedback_batcher.flush_if_ready(force=False)
            self.cache.prune()
        except redis.RedisError as exc:
            log_event(logging.ERROR, "redis_error", cam_id=self.cam_id, error=str(exc))
            time.sleep(0.5)
        except Exception as exc:
            log_event(logging.ERROR, "collector_step_failed", cam_id=self.cam_id, error=str(exc), traceback=traceback.format_exc())
            time.sleep(0.2)

    def run_forever(self, sleep_s=0.002):
        try:
            while True:
                self.step()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            log_event(logging.INFO, "collector_stopping", cam_id=self.cam_id)
        finally:
            self.close()

    def close(self):
        for sock in (self.sub_scalars, self.sub_s3_feedback, self.train_pub):
            try:
                if sock is not None:
                    sock.close(0)
            except Exception:
                pass


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)
    ap.add_argument("--site_id", default=None)
    ap.add_argument("--fleet_id", default=None)
    ap.add_argument("--target_id", default=None)
    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--scores_stream", default=None)
    ap.add_argument("--decisions_stream", default=None)
    ap.add_argument("--alerts_stream", default=None)
    ap.add_argument("--feedback_stream", default=None)
    ap.add_argument("--clip_refs_stream", default=None)
    ap.add_argument("--train_events_stream", default=None)
    ap.add_argument("--redis_maxlen", type=int, default=50000)
    ap.add_argument("--cache_ttl_s", type=float, default=30.0)
    ap.add_argument("--pending_ttl_s", type=float, default=3600.0)
    ap.add_argument("--publish_partial", action="store_true")
    ap.add_argument("--gold_filter_unlabeled_ready", action="store_true")
    ap.add_argument("--gold_min_votes", type=int, default=2)
    ap.add_argument("--gold_min_S", type=float, default=0.85)
    ap.add_argument("--gold_require_gate_ok", action="store_true")
    ap.add_argument("--gold_max_missing_pose_ratio", type=float, default=0.55)
    ap.add_argument("--gold_max_missing_obj_ratio", type=float, default=0.75)
    ap.add_argument("--enable_llm_feedback_parse", action="store_true")
    ap.add_argument("--llm_model", type=str, default="gpt-5.4")
    ap.add_argument("--llm_timeout_s", type=float, default=8.0)
    ap.add_argument("--feedback_min_confidence", type=float, default=0.80)
    ap.add_argument("--llm_allowed_categories", type=str, default="theft,false_alarm,benign,uncertain,needs_review")
    ap.add_argument("--llm_api_key", type=str, default=None)
    ap.add_argument("--llm_base_url", type=str, default=None)
    ap.add_argument("--llm_disable_fallback", action="store_true")
    ap.add_argument("--use_scalars_clip_zmq", action="store_true")
    ap.add_argument("--scalars_connect", default=None)
    ap.add_argument("--scalars_topic", default=None)
    ap.add_argument("--zmq_rcvhwm", type=int, default=512)
    ap.add_argument("--zmq_sndhwm", type=int, default=512)
    ap.add_argument("--train_events_zmq_enabled", action="store_true", default=True)
    ap.add_argument("--no_train_events_zmq", dest="train_events_zmq_enabled", action="store_false")
    ap.add_argument("--train_events_zmq_mode", default="pub", choices=["pub", "push"])
    ap.add_argument("--train_events_zmq_bind", default=None)
    ap.add_argument("--train_events_zmq_connect", default=None)
    ap.add_argument("--train_events_topic", default="train_events.theft")
    ap.add_argument("--use_s3_feedback_zmq", action="store_true")
    ap.add_argument("--s3_feedback_zmq_connect", default=None)
    ap.add_argument("--s3_feedback_topic", default=None)
    ap.add_argument("--publish_train_events_redis", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    return ap



def apply_data_collector_config_args(args):
    """
    Merge config.yaml:data_collector_args into CLI args.

    CLI still wins when a value was explicitly provided.
    For boolean store_true/store_false flags, config fills the intended runtime
    default so direct execution behaves like brain.py/service execution.
    """
    try:
        cfg = load_cfg(args.config)
    except Exception as exc:
        log_event(logging.WARNING, "config_args_load_failed", error=str(exc))
        return args

    dc = cfg.get("data_collector_args", {}) or {}
    if not isinstance(dc, dict):
        return args

    parser_defaults = {
        action.dest: action.default
        for action in build_arg_parser()._actions
        if action.dest and action.dest != "help"
    }

    for key, value in dc.items():
        if not hasattr(args, key):
            continue

        current = getattr(args, key)
        default = parser_defaults.get(key, None)

        # If CLI left the parser default, config may override it.
        # If user explicitly passed a different CLI value, keep CLI.
        if current == default:
            setattr(args, key, value)

    return args

def main():
    args = build_arg_parser().parse_args()
    setup_logging(args.log_level)
    args = apply_data_collector_config_args(args)
    worker = DataCollectorWorker(args)
    worker.run_forever()


if __name__ == "__main__":
    main()
