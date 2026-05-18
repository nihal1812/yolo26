#!/usr/bin/env python3
"""
trainer_node.py

Production-hardened fleet/global trainer component for the theft-detection
learning pipeline.

This file is meant to be imported by brain.py as part of the pipeline. The CLI
at the bottom is only for quick manual testing or maintenance.

Pipeline role:
- Subscribe to ONE shared ZMQ training topic, for example train_events.theft.
- Accept training events from all cameras and all stores that publish to that
  topic.
- Train ONE shared fleet/global theft model so new stores can start from the
  cloud champion instead of starting from scratch.
- Keep site_id/store_id and cam_id only as metadata for reporting, debugging,
  dataset balance checks, and future local overrides. They are not the model
  partition key.
- Upload model, registry, and dataset artifacts to the common S3 bucket/prefixes.
- Publish model_update, policy_update, and trainer_health messages back to the
  rest of the pipeline.

Important clip-storage behavior:
- Production mode expects clips to live in S3, not long-term local disk.
- Each incoming event must carry its own clip S3 location, because every sample
  points to a different S3 object.
- The trainer downloads a clip to a temporary file only when it validates or
  trains on that sample, loads it into memory, and deletes the temporary file
  immediately after loading.
- Local clip paths are supported only as a development fallback and must be
  sandboxed with allowed_clip_root.

Recommended incoming training event:
{
  "event_id": "evt_123",
  "site_id": "store_001",
  "cam_id": "cam01",
  "fleet_id": "global_retail",
  "train_ok": true,
  "feedback": {"label": 0|1, "llm_confidence": 0.95, "needs_review": false},
  "payload": {
    "clip_ref": {
      "storage": "s3",
      "s3_uri": "s3://zono-model/datasets/theft/global_retail/clips/evt_123.npz",
      "tensor_key": "clip"
    },
    "score": {...},
    "scalars_clip": {"features": {...}},
    "decision": {...}
  }
}

Recommended config.yaml block:
trainer_node:
  target_id: global_retail        # model/registry target, not a camera/store
  model_scope: fleet              # fleet/global shared model
  require_site_id: false          # site_id is metadata; do not filter stores

  zmq:
    input_mode: sub
    input_connect: tcp://127.0.0.1:5690
    input_bind: null
    input_topic: train_events.theft
    output_mode: pub
    output_bind: tcp://*:5691
    output_connect: null
    output_topic_prefix: trainer

  storage:
    out_dir: models
    require_s3_clips: true
    clip_temp_dir: null
    max_clip_mb: 256
    require_s3_upload: false

model_cloud:
  enabled: true
  registry_mode: s3
  bucket: zono-model
  region: eu-central-1
  prefix_models: models/theft/
  prefix_registry: registry/theft/
  prefix_datasets: datasets/theft/

S3 artifact layout for cross-store reuse:
  s3://zono-model/models/theft/global_retail/<model_version>.pt
  s3://zono-model/registry/theft/global_retail/champion.json
  s3://zono-model/datasets/theft/global_retail/dataset_<version>.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import zmq
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyzmq is required. Install with: pip install pyzmq") from exc

try:
    from config_utils import load_cfg
except Exception:
    def load_cfg(path: str) -> dict:
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        if p.suffix.lower() in {".json"}:
            return json.loads(p.read_text(encoding="utf-8"))
        try:
            import yaml
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("trainer_node")


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts": time.time(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if hasattr(record, "extra_fields"):
                payload.update(getattr(record, "extra_fields"))
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(payload, sort_keys=True)

    if json_logs:
        for handler in logging.getLogger().handlers:
            handler.setFormatter(JsonFormatter())


def log_event(level: int, message: str, **fields: Any) -> None:
    LOGGER.log(level, message, extra={"extra_fields": fields})


# ---------------------------------------------------------------------------
# Safe helpers
# ---------------------------------------------------------------------------

def safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return default


def safe_bool(v: Any, default: bool = False) -> bool:
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


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return copy.deepcopy(default)


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def stable_unit_float(s: str, seed: int = 42) -> float:
    h = hashlib.sha1(f"{seed}:{s}".encode("utf-8")).hexdigest()
    val = int(h[:12], 16)
    return (val % 10_000_000) / 10_000_000.0


def now_ns() -> int:
    return time.time_ns()


# ---------------------------------------------------------------------------
# Feature extraction and model code
# ---------------------------------------------------------------------------

def compute_scalar_norm(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0).astype(np.float32)
    std = np.std(X, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def normalize_X(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _get_stat(d: dict, key: str, stat: str) -> Any:
    try:
        v = d.get(key, None)
        if isinstance(v, dict):
            return v.get(stat, None)
    except Exception:
        pass
    return None


def default_scalar_schema() -> List[Tuple[str, str]]:
    schema: List[Tuple[str, str]] = []
    for nm in ["hand_to_chest_L", "hand_to_chest_R", "hand_to_hip_L", "hand_to_hip_R"]:
        for st in ["mean", "max", "last"]:
            schema.append((nm, st))
    for nm in ["wrist_v_L", "wrist_v_R"]:
        for st in ["mean", "max", "last"]:
            schema.append((nm, st))
    schema.append(("torso_angle_rate", "mean"))
    schema.append(("torso_angle_rate", "max"))
    schema.append(("guard_score", "mean"))
    schema.append(("guard_score", "max"))
    schema.append(("guard_score", "last"))
    for nm in ["obj_visibility", "carry_score", "comotion_cosine", "rel_offset_std_px"]:
        for st in ["mean", "min", "max", "last"]:
            schema.append((nm, st))
    for nm in ["person_speed_px_s", "object_speed_px_s"]:
        for st in ["mean", "max", "last"]:
            schema.append((nm, st))
    return schema


def build_scalar_vector(agg_features: dict, schema: list, fill_value: float = 0.0) -> Tuple[np.ndarray, int]:
    vec: List[float] = []
    missing = 0
    for fname, stat in schema:
        val = _get_stat(agg_features, fname, stat)
        if val is None:
            vec.append(float(fill_value))
            missing += 1
        else:
            vec.append(float(val))
    return np.array(vec, dtype=np.float32), missing


def append_special_scalars(vec: np.ndarray, agg_features: dict) -> np.ndarray:
    def fget(k: str, default: float = 0.0) -> float:
        v = agg_features.get(k, None)
        if v is None:
            return float(default)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        try:
            return float(v)
        except Exception:
            return float(default)

    extras = np.array(
        [
            fget("contact_ratio", 0.0),
            fget("contact_duration_frames_last", 0.0),
            fget("contact_duration_frames_max", 0.0),
            fget("visibility_drop", 0.0),
            fget("visibility_min", 0.0),
            fget("disappeared_after_contact", 0.0),
            fget("missing_pose_ratio", 1.0),
            fget("missing_obj_ratio", 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([vec, extras], axis=0)


def identity_feature_names() -> List[str]:
    return [
        "id_seen_recently",
        "id_seen_other_cam_recently",
        "id_time_since_last_seen_s",
        "id_recent_cam_count",
        "id_recent_event_count",
        "id_prior_score_ema",
        "id_prior_score_max",
        "id_prior_suspicion_mean",
    ]


def extract_identity_feature_vector_from_score_obj(score_obj: dict, use_identity_features: bool) -> np.ndarray:
    if not use_identity_features:
        return np.zeros((0,), dtype=np.float32)
    feats = score_obj.get("identity_features", {})
    if not isinstance(feats, dict):
        feats = {}
    vals = [float(safe_float(feats.get(name, 0.0), 0.0)) for name in identity_feature_names()]
    return np.asarray(vals, dtype=np.float32)


def default_model_runtime_meta(cfg: dict) -> dict:
    mr = (cfg or {}).get("model_runtime", {}) if isinstance(cfg, dict) else {}
    return {
        "task_type": str(mr.get("task_type", "theft_detection")),
        "label_space": str(mr.get("label_space", "binary_theft")),
        "domain_type": str(mr.get("domain_type", "retail")),
        "environment_family": str(mr.get("environment_family", "generic")),
        "site_scope": str(mr.get("site_scope", "global")),
        "schema_version": str(mr.get("schema_version", "scalars_v1")),
        "identity_feature_version": str(mr.get("identity_feature_version", "idfeat_v1")),
        "runtime_compat_version": str(mr.get("runtime_compat_version", "model_node_v1")),
        "rollout_mode": str(mr.get("rollout_mode", "shadow")),
        "allow_global_fallback": bool(safe_bool(mr.get("allow_global_fallback", True), True)),
    }


class CNNBackbone3D(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ScalarMLP(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BinaryHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FusionModel(nn.Module):
    def __init__(self, clip_channels: int, scalar_dim: int, cnn_emb: int = 256, s_emb: int = 64):
        super().__init__()
        self.clip_channels = int(clip_channels)
        self.scalar_dim = int(scalar_dim)
        self.cnn = CNNBackbone3D(clip_channels, emb_dim=cnn_emb)
        self.smlp = ScalarMLP(scalar_dim, emb_dim=s_emb)
        self.cnn_head = BinaryHead(cnn_emb)
        self.mlp_head = BinaryHead(s_emb)
        self.fusion_head = BinaryHead(cnn_emb + s_emb)

    def forward(self, clip: torch.Tensor, scalars: torch.Tensor) -> dict:
        zc = self.cnn(clip)
        zs = self.smlp(scalars)
        return {
            "cnn_logits": self.cnn_head(zc),
            "mlp_logits": self.mlp_head(zs),
            "fused_logits": self.fusion_head(torch.cat([zc, zs], dim=1)),
            "cnn_emb": zc,
            "mlp_emb": zs,
        }


def load_checkpoint(weights_path: str, device: torch.device) -> Tuple[dict, dict]:
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        state_dict = ckpt
        meta = {}
    return state_dict, meta


# ---------------------------------------------------------------------------
# Clip security and sample extraction
# ---------------------------------------------------------------------------

def resolve_clip_uri_from_event(payload: dict) -> Optional[str]:
    """Return the clip location carried by a training event.

    Production events should provide an S3 URI. Local paths are accepted only for
    development compatibility. The returned value is a URI-like string; it is not
    necessarily a local path.
    """
    if not isinstance(payload, dict):
        return None

    # Preferred production location. The collector/uploader should put the clip
    # in S3 first, then publish a training event pointing at that S3 object.
    clip_ref = payload.get("clip_ref", None)
    if isinstance(clip_ref, dict):
        for k in (
            "s3_uri",
            "clip_s3_uri",
            "uri",
            "s3_path",
            "local_clip_path",  # backward-compatible local fallback
            "clip_path",
            "path",
        ):
            v = clip_ref.get(k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # Alternate shape sometimes used by uploaders.
        bucket = clip_ref.get("bucket") or clip_ref.get("s3_bucket")
        key = clip_ref.get("key") or clip_ref.get("s3_key")
        if bucket and key:
            return f"s3://{bucket}/{str(key).lstrip('/')}"

    # Backward-compatible fallbacks from older scoring payloads.
    score_obj = payload.get("score", None)
    if isinstance(score_obj, dict):
        for k in ("clip_s3_uri", "s3_uri", "clip_path"):
            v = score_obj.get(k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

    clip_meta = payload.get("clip_meta", None)
    if isinstance(clip_meta, dict):
        for k in ("clip_s3_uri", "s3_uri", "clip_path"):
            v = clip_meta.get(k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return None


# Backward-compatible function name. Existing code that calls
# resolve_clip_path_from_event still works, but the value may now be an S3 URI.
def resolve_clip_path_from_event(payload: dict) -> Optional[str]:
    return resolve_clip_uri_from_event(payload)


def is_s3_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.lower().startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not is_s3_uri(uri):
        raise ValueError(f"not an s3 uri: {uri}")
    rest = uri[5:]
    bucket, sep, key = rest.partition("/")
    if not bucket or not sep or not key:
        raise ValueError(f"invalid s3 uri: {uri}")
    return bucket, key


def path_is_under(path: Path, roots: List[Path]) -> bool:
    try:
        resolved = path.resolve(strict=False)
        for root in roots:
            rr = root.resolve(strict=False)
            if resolved == rr or rr in resolved.parents:
                return True
    except Exception:
        return False
    return False


def validate_npz_size(path: Path, max_bytes: int) -> bool:
    try:
        total = 0
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                total += int(info.file_size)
                if total > max_bytes:
                    return False
        return True
    except Exception:
        return False


def load_clip_tensor_safe(
    local_clip_path: str,
    *,
    expected_channels: int = 19,
    allowed_roots: Optional[List[Path]] = None,
    max_clip_bytes: int = 256 * 1024 * 1024,
    expected_ndim: int = 4,
) -> Optional[np.ndarray]:
    if not local_clip_path:
        return None

    path = Path(local_clip_path).expanduser()
    try:
        path = path.resolve(strict=True)
    except Exception:
        return None

    if allowed_roots and not path_is_under(path, allowed_roots):
        log_event(logging.WARNING, "clip_path_rejected_outside_allowed_roots", clip_path=str(path))
        return None

    try:
        if path.stat().st_size > max_clip_bytes:
            log_event(logging.WARNING, "clip_path_rejected_size", clip_path=str(path), bytes=path.stat().st_size)
            return None
    except Exception:
        return None

    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            mm = np.load(str(path), allow_pickle=False, mmap_mode="r")
            if mm.ndim != expected_ndim or mm.shape[0] != expected_channels:
                return None
            if int(np.prod(mm.shape)) * np.dtype(mm.dtype).itemsize > max_clip_bytes:
                return None
            clip = np.asarray(mm, dtype=np.float32)
        elif suffix == ".npz":
            if not validate_npz_size(path, max_clip_bytes=max_clip_bytes):
                return None
            with np.load(str(path), allow_pickle=False) as data:
                if "clip" not in data:
                    return None
                arr = data["clip"]
                if arr.ndim != expected_ndim or arr.shape[0] != expected_channels:
                    return None
                if int(np.prod(arr.shape)) * np.dtype(arr.dtype).itemsize > max_clip_bytes:
                    return None
                clip = arr.astype(np.float32)
        else:
            return None

        if not np.all(np.isfinite(clip)):
            return None
        return np.ascontiguousarray(clip, dtype=np.float32)
    except Exception as exc:
        log_event(logging.WARNING, "clip_load_failed", clip_path=str(path), error=str(exc))
        return None




class ClipProvider:
    """Loads training clips from S3 or local development paths.

    The production path is S3. A clip is downloaded into a temporary file,
    loaded into a NumPy tensor, and the temporary file is deleted immediately.
    This keeps edge devices from becoming long-term clip stores.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        allowed_roots: Optional[List[Path]],
        max_clip_bytes: int,
        temp_dir: Optional[str] = None,
        require_s3_clips: bool = False,
    ):
        self.cfg = cfg or {}
        self.allowed_roots = allowed_roots or []
        self.max_clip_bytes = int(max_clip_bytes)
        self.temp_dir = Path(temp_dir).expanduser() if temp_dir else None
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.require_s3_clips = bool(require_s3_clips)
        self._s3_client = None

    def _client(self):
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3  # type: ignore
            cloud_cfg = self.cfg.get("model_cloud", {}) if isinstance(self.cfg, dict) else {}
            if not isinstance(cloud_cfg, dict):
                cloud_cfg = {}
            session_kwargs = {}
            access_key = cloud_cfg.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID")
            secret_key = cloud_cfg.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
            if access_key and secret_key:
                session_kwargs["aws_access_key_id"] = access_key
                session_kwargs["aws_secret_access_key"] = secret_key
            region = cloud_cfg.get("region") or os.environ.get("AWS_DEFAULT_REGION")
            if region:
                session_kwargs["region_name"] = region
            session = boto3.session.Session(**session_kwargs)
            self._s3_client = session.client(
                "s3",
                endpoint_url=cloud_cfg.get("endpoint_url") or None,
                verify=bool(safe_bool(cloud_cfg.get("verify_ssl", True), True)),
            )
            return self._s3_client
        except Exception as exc:
            raise RuntimeError(f"failed to create S3 client for clip loading: {exc}") from exc

    def load_clip_tensor(self, clip_uri: str, *, expected_channels: int) -> Optional[np.ndarray]:
        if not clip_uri:
            return None
        if is_s3_uri(clip_uri):
            return self._load_s3_clip(clip_uri, expected_channels=expected_channels)
        if self.require_s3_clips:
            log_event(logging.WARNING, "local_clip_rejected_s3_required", clip_uri=clip_uri)
            return None
        if not self.allowed_roots:
            log_event(logging.WARNING, "local_clip_rejected_no_allowed_roots", clip_uri=clip_uri)
            return None
        return load_clip_tensor_safe(
            clip_uri,
            expected_channels=expected_channels,
            allowed_roots=self.allowed_roots,
            max_clip_bytes=self.max_clip_bytes,
        )

    def _load_s3_clip(self, s3_uri: str, *, expected_channels: int) -> Optional[np.ndarray]:
        bucket, key = parse_s3_uri(s3_uri)
        suffix = Path(key).suffix.lower()
        if suffix not in {".npy", ".npz"}:
            log_event(logging.WARNING, "s3_clip_rejected_suffix", s3_uri=s3_uri, suffix=suffix)
            return None

        tmp_path = None
        try:
            client = self._client()
            try:
                head = client.head_object(Bucket=bucket, Key=key)
                size = int(head.get("ContentLength", 0))
                if size <= 0 or size > self.max_clip_bytes:
                    log_event(logging.WARNING, "s3_clip_rejected_size", s3_uri=s3_uri, bytes=size)
                    return None
            except Exception as exc:
                log_event(logging.WARNING, "s3_clip_head_failed", s3_uri=s3_uri, error=str(exc))
                return None

            # NamedTemporaryFile is closed before np.load so it works consistently
            # on Linux and Windows. The file is deleted in finally.
            fd, name = tempfile.mkstemp(prefix="trainer_clip_", suffix=suffix, dir=str(self.temp_dir) if self.temp_dir else None)
            os.close(fd)
            tmp_path = Path(name)
            client.download_file(bucket, key, str(tmp_path))
            clip = load_clip_tensor_safe(
                str(tmp_path),
                expected_channels=expected_channels,
                allowed_roots=None,
                max_clip_bytes=self.max_clip_bytes,
            )
            return clip
        except Exception as exc:
            log_event(logging.WARNING, "s3_clip_load_failed", s3_uri=s3_uri, error=str(exc))
            return None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
def extract_training_sample(
    train_event: Dict[str, Any],
    *,
    schema: list,
    scalar_dim: int,
    base_scalar_dim: int,
    identity_dim: int,
    use_identity_features: bool,
    expected_clip_channels: int,
    filter_bad_quality: bool,
    max_missing_pose_ratio: float,
    max_missing_obj_ratio: float,
    require_gate_ok: bool,
    require_train_ok: bool,
    skip_feedback_needs_review: bool,
    min_feedback_confidence: Optional[float],
    clip_provider: ClipProvider,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, dict, str]]:
    reject_reason = "unknown"

    fb = train_event.get("feedback", None)
    if not isinstance(fb, dict):
        return None
    y = safe_int(fb.get("label", None), None)
    if y is None or y not in (0, 1):
        return None

    if require_train_ok and (train_event.get("train_ok", True) is not True):
        return None

    if skip_feedback_needs_review and bool(fb.get("needs_review", False)):
        return None

    fb_conf = safe_float(fb.get("llm_confidence", None), None)
    if min_feedback_confidence is not None:
        if fb_conf is None or fb_conf < float(min_feedback_confidence):
            return None

    payload = train_event.get("payload", {})
    score_obj = payload.get("score", None)
    scal_obj = payload.get("scalars_clip", None)
    dec_obj = payload.get("decision", None)

    if not isinstance(score_obj, dict) or not isinstance(scal_obj, dict):
        return None

    clip_uri = resolve_clip_uri_from_event(payload)
    clip_np = clip_provider.load_clip_tensor(
        clip_uri or "",
        expected_channels=expected_clip_channels,
    )
    if clip_np is None:
        return None

    feats = scal_obj.get("features", {})
    if not isinstance(feats, dict):
        feats = {}

    miss_pose = None
    miss_obj = None
    gate_ok = None
    if isinstance(dec_obj, dict):
        miss_pose = safe_float(dec_obj.get("missing_pose_ratio", None), None)
        miss_obj = safe_float(dec_obj.get("missing_obj_ratio", None), None)
        gate_ok = dec_obj.get("gate_ok", None)
    if miss_pose is None:
        miss_pose = safe_float(score_obj.get("missing_pose_ratio", None), None)
    if miss_obj is None:
        miss_obj = safe_float(score_obj.get("missing_obj_ratio", None), None)

    if filter_bad_quality:
        if miss_pose is not None and miss_pose > float(max_missing_pose_ratio):
            return None
        if miss_obj is not None and miss_obj > float(max_missing_obj_ratio):
            return None
    if require_gate_ok and (gate_ok is not True):
        return None

    svec, miss_fields = build_scalar_vector(feats, schema, fill_value=0.0)
    x = append_special_scalars(svec, feats)
    if x.size != base_scalar_dim:
        return None

    if use_identity_features:
        id_vec = extract_identity_feature_vector_from_score_obj(score_obj, use_identity_features=True)
        if id_vec.size != identity_dim:
            return None
        x = np.concatenate([x, id_vec], axis=0)

    if x.size != scalar_dim:
        return None
    if not np.all(np.isfinite(x)):
        return None

    meta = {
        "event_id": train_event.get("event_id"),
        "cam_id": train_event.get("cam_id"),
        "person_track_id": train_event.get("person_track_id"),
        "global_person_id": train_event.get("global_person_id"),
        # clip_uri is kept because production clips live in S3. Do not assume this
        # is a local filesystem path.
        "clip_uri": clip_uri,
        "clip_path": None if (clip_uri and is_s3_uri(clip_uri)) else (str(Path(clip_uri).expanduser().resolve(strict=False)) if clip_uri else None),
        "label": int(y),
        "feedback_confidence": fb_conf,
        "feedback_needs_review": bool(fb.get("needs_review", False)),
        "feedback_category": fb.get("category"),
        "feedback_comment": fb.get("comment"),
        "missing_pose_ratio": miss_pose,
        "missing_obj_ratio": miss_obj,
        "gate_ok": gate_ok,
        "scalar_missing_fields": int(miss_fields),
    }
    return clip_np.astype(np.float32), x.astype(np.float32), int(y), meta, "accepted"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def suggest_threshold_balanced(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.05,
    min_recall: float = 0.40,
    min_alert_rate: float = 0.03,
    max_threshold: float = 0.95,
) -> float:
    if scores.size == 0:
        return 0.8
    candidates = np.unique(np.clip(scores, 0.0, 1.0))
    candidates = np.concatenate([
        np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90], dtype=np.float32),
        candidates.astype(np.float32),
    ])
    candidates = np.unique(np.clip(candidates, 0.05, max_threshold))
    candidates = np.sort(candidates)

    pos_mask = labels == 1
    neg_mask = labels == 0
    best = None
    fallback = None

    for thr in candidates:
        pred = scores >= thr
        alert_rate = float(np.mean(pred)) if pred.size else 0.0
        recall = float(np.mean(pred[pos_mask])) if np.any(pos_mask) else 0.0
        fpr = float(np.mean(pred[neg_mask])) if np.any(neg_mask) else 0.0
        feasible = (fpr <= target_fpr) and (recall >= min_recall) and (alert_rate >= min_alert_rate)
        quality = (
            abs(fpr - target_fpr)
            + 0.30 * max(0.0, min_recall - recall)
            + 0.20 * max(0.0, min_alert_rate - alert_rate)
        )
        if feasible:
            cand = (quality, float(thr), fpr, recall, alert_rate)
            if best is None or cand < best:
                best = cand

        fallback_quality = (
            abs(fpr - target_fpr)
            + 0.50 * max(0.0, min_recall - recall)
            + 0.35 * max(0.0, min_alert_rate - alert_rate)
        )
        cand_fb = (fallback_quality, float(thr), fpr, recall, alert_rate)
        if fallback is None or cand_fb < fallback:
            fallback = cand_fb

    chosen = best if best is not None else fallback
    return 0.8 if chosen is None else float(chosen[1])


def compute_binary_metrics(scores: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> dict:
    pred = (scores >= thr).astype(np.int64)
    acc = float((pred == labels).mean()) if labels.size else 0.0
    pos_mask = labels == 1
    neg_mask = labels == 0
    recall = float(pred[pos_mask].mean()) if np.any(pos_mask) else 0.0
    fpr = float(pred[neg_mask].mean()) if np.any(neg_mask) else 0.0
    precision = float(np.sum((pred == 1) & (labels == 1)) / max(1, np.sum(pred == 1))) if labels.size else 0.0
    return {
        "acc": acc,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "alert_rate": float(np.mean(pred)) if labels.size else 0.0,
        "threshold": float(thr),
    }


def balanced_quality(metrics: dict, target_fpr: float) -> float:
    recall = float(metrics.get("recall", 0.0))
    fpr = float(metrics.get("fpr", 1.0))
    precision = float(metrics.get("precision", 0.0))
    return (1.8 * recall) + (0.7 * precision) - (1.2 * abs(fpr - target_fpr)) - (0.8 * fpr)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    if n <= 0:
        return None
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return [float((centre - margin) / denom), float((centre + margin) / denom)]


def metrics_with_intervals(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    m = compute_binary_metrics(scores, labels, thr=thr)
    pred = scores >= thr
    pos = labels == 1
    neg = labels == 0
    tp = int(np.sum(pred & pos))
    pos_n = int(np.sum(pos))
    fp = int(np.sum(pred & neg))
    neg_n = int(np.sum(neg))
    pp = int(np.sum(pred))
    correct_pos_pred = int(np.sum(pred & pos))
    m["recall_ci95"] = wilson_interval(tp, pos_n)
    m["fpr_ci95"] = wilson_interval(fp, neg_n)
    m["precision_ci95"] = wilson_interval(correct_pos_pred, pp)
    m["pos_n"] = pos_n
    m["neg_n"] = neg_n
    return m


def build_dataset_version(event_ids: List[str]) -> str:
    ids = sorted([str(x) for x in event_ids if x])
    return sha1_text("\n".join(ids))[:16] if ids else "empty"


def build_persistent_split(event_ids: List[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[str]]:
    train_ids, val_ids, test_ids = [], [], []
    for eid in sorted(event_ids):
        u = stable_unit_float(eid, seed=seed)
        if u < train_ratio:
            train_ids.append(eid)
        elif u < (train_ratio + val_ratio):
            val_ids.append(eid)
        else:
            test_ids.append(eid)
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def count_labels(ids: List[str], labels_by_event: Dict[str, int]) -> dict:
    ys = [labels_by_event[i] for i in ids if i in labels_by_event]
    return {"total": len(ys), "pos": int(sum(1 for y in ys if y == 1)), "neg": int(sum(1 for y in ys if y == 0))}


def enforce_split_class_coverage(split: Dict[str, List[str]], labels_by_event: Dict[str, int]) -> Dict[str, List[str]]:
    out = {k: list(v) for k, v in split.items()}

    def move_one(label_needed: int, src_names: List[str], dst_name: str) -> bool:
        for src_name in src_names:
            for eid in list(out[src_name]):
                if labels_by_event.get(eid) == label_needed:
                    out[src_name].remove(eid)
                    out[dst_name].append(eid)
                    return True
        return False

    for bucket in ["train", "val", "test"]:
        counts = count_labels(out[bucket], labels_by_event)
        if counts["neg"] == 0:
            move_one(0, [b for b in ["train", "val", "test"] if b != bucket], bucket)
        counts = count_labels(out[bucket], labels_by_event)
        if counts["pos"] == 0:
            move_one(1, [b for b in ["train", "val", "test"] if b != bucket], bucket)
    return out


def compute_label_quality(sample_metas: List[dict]) -> dict:
    n = len(sample_metas)
    if n == 0:
        return {
            "num_samples": 0,
            "avg_feedback_confidence": None,
            "low_confidence_ratio": None,
            "needs_review_ratio": None,
            "pos_ratio": None,
            "neg_ratio": None,
            "avg_missing_pose_ratio": None,
            "avg_missing_obj_ratio": None,
        }

    confs = [safe_float(m.get("feedback_confidence", None), None) for m in sample_metas]
    confs = [c for c in confs if c is not None]
    needs_review = [1.0 if bool(m.get("feedback_needs_review", False)) else 0.0 for m in sample_metas]
    ys = [safe_int(m.get("label", None), None) for m in sample_metas]
    miss_pose = [safe_float(m.get("missing_pose_ratio", None), None) for m in sample_metas]
    miss_pose = [v for v in miss_pose if v is not None]
    miss_obj = [safe_float(m.get("missing_obj_ratio", None), None) for m in sample_metas]
    miss_obj = [v for v in miss_obj if v is not None]
    pos = sum(1 for y in ys if y == 1)
    neg = sum(1 for y in ys if y == 0)
    low_conf_ratio = float(np.mean(np.array(confs, dtype=np.float32) < 0.80)) if confs else None

    return {
        "num_samples": n,
        "avg_feedback_confidence": float(np.mean(confs)) if confs else None,
        "low_confidence_ratio": low_conf_ratio,
        "needs_review_ratio": float(np.mean(needs_review)) if needs_review else 0.0,
        "pos_ratio": float(pos / max(1, n)),
        "neg_ratio": float(neg / max(1, n)),
        "avg_missing_pose_ratio": float(np.mean(miss_pose)) if miss_pose else None,
        "avg_missing_obj_ratio": float(np.mean(miss_obj)) if miss_obj else None,
    }


def compute_feature_drift_score(X_raw: np.ndarray, reference_norm: Optional[dict]) -> Optional[float]:
    if reference_norm is None:
        return None
    mean = reference_norm.get("mean", None)
    std = reference_norm.get("std", None)
    if mean is None or std is None:
        return None
    try:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        curr_mean = np.mean(X_raw, axis=0).astype(np.float32)
        z_shift = np.abs((curr_mean - mean) / std)
        return float(np.mean(z_shift))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Optional S3 artifact registry
# ---------------------------------------------------------------------------

class CloudArtifactStore:
    """Small S3 uploader for models, registries, and dataset manifests.

    Config is read from cfg["model_cloud"] and matches the shape you showed:
      model_cloud:
        enabled: true
        registry_mode: s3
        bucket: zono-model
        region: eu-central-1
        endpoint_url: null
        access_key: null
        secret_key: null
        verify_ssl: true
        prefix_models: models/theft/
        prefix_registry: registry/theft/
        prefix_datasets: datasets/theft/

    Artifacts are stored under <prefix>/<target_id>/... so one bucket can serve
    all stores and deployments through one shared cloud champion.
    """

    def __init__(self, cfg: dict, *, target_id: str, model_scope: str, require_upload: bool = False):
        cloud_cfg = (cfg or {}).get("model_cloud", {}) if isinstance(cfg, dict) else {}
        self.cfg = cloud_cfg if isinstance(cloud_cfg, dict) else {}
        self.target_id = str(target_id)
        self.model_scope = str(model_scope)
        self.require_upload = bool(require_upload)
        self.enabled = bool(safe_bool(self.cfg.get("enabled", False), False))
        self.registry_mode = str(self.cfg.get("registry_mode", "local")).lower()
        self.bucket = self.cfg.get("bucket")
        self.client = None

        self.prefix_models = str(self.cfg.get("prefix_models", "models/theft/"))
        self.prefix_registry = str(self.cfg.get("prefix_registry", "registry/theft/"))
        self.prefix_datasets = str(self.cfg.get("prefix_datasets", "datasets/theft/"))

        if not self.enabled:
            return
        if self.registry_mode != "s3":
            log_event(logging.WARNING, "cloud_registry_disabled_unsupported_mode", registry_mode=self.registry_mode)
            self.enabled = False
            return
        if not self.bucket:
            msg = "model_cloud.enabled is true, but model_cloud.bucket is empty"
            if self.require_upload:
                raise ValueError(msg)
            log_event(logging.WARNING, "cloud_registry_disabled_missing_bucket", error=msg)
            self.enabled = False
            return

        try:
            import boto3  # type: ignore
            session_kwargs = {}
            access_key = self.cfg.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID")
            secret_key = self.cfg.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
            if access_key and secret_key:
                session_kwargs["aws_access_key_id"] = access_key
                session_kwargs["aws_secret_access_key"] = secret_key
            region = self.cfg.get("region") or os.environ.get("AWS_DEFAULT_REGION")
            if region:
                session_kwargs["region_name"] = region
            session = boto3.session.Session(**session_kwargs)
            self.client = session.client(
                "s3",
                endpoint_url=self.cfg.get("endpoint_url") or None,
                verify=bool(safe_bool(self.cfg.get("verify_ssl", True), True)),
            )
            log_event(logging.INFO, "cloud_registry_enabled", bucket=self.bucket, region=region, target_id=self.target_id, model_scope=self.model_scope)
        except Exception as exc:
            if self.require_upload:
                raise
            log_event(logging.WARNING, "cloud_registry_disabled_init_failed", error=str(exc))
            self.enabled = False
            self.client = None

    def _prefix_for(self, family: str) -> str:
        if family == "model":
            return self.prefix_models
        if family == "registry":
            return self.prefix_registry
        if family == "dataset":
            return self.prefix_datasets
        raise ValueError(f"unknown artifact family: {family}")

    def key_for(self, family: str, filename: str) -> str:
        prefix = self._prefix_for(family).strip("/")
        # Avoid double appending the target_id if a user already put it in the prefix.
        parts = [p for p in prefix.split("/") if p]
        if not parts or parts[-1] != self.target_id:
            parts.append(self.target_id)
        parts.append(Path(filename).name)
        return "/".join(parts)

    def uri_for(self, family: str, filename: str) -> Optional[str]:
        if not self.enabled or not self.bucket:
            return None
        return f"s3://{self.bucket}/{self.key_for(family, filename)}"

    def upload_file(self, local_path: str | Path, family: str) -> Optional[str]:
        if not self.enabled or self.client is None or not self.bucket:
            return None
        local_path = Path(local_path)
        key = self.key_for(family, local_path.name)
        try:
            self.client.upload_file(str(local_path), self.bucket, key)
            uri = f"s3://{self.bucket}/{key}"
            log_event(logging.INFO, "cloud_artifact_uploaded", family=family, uri=uri)
            return uri
        except Exception as exc:
            log_event(logging.ERROR, "cloud_artifact_upload_failed", family=family, path=str(local_path), error=str(exc))
            if self.require_upload:
                raise
            return None


# ---------------------------------------------------------------------------
# ZMQ transport
# ---------------------------------------------------------------------------

class ZmqTransport:
    def __init__(
        self,
        *,
        input_mode: str,
        input_bind: Optional[str],
        input_connect: Optional[str],
        output_mode: str,
        output_bind: Optional[str],
        output_connect: Optional[str],
        input_topic: str,
        output_topic_prefix: str,
        recv_timeout_ms: int,
        sndhwm: int = 1000,
        rcvhwm: int = 1000,
    ):
        self.ctx = zmq.Context.instance()
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.input_topic = input_topic
        self.output_topic_prefix = output_topic_prefix
        self.recv_timeout_ms = int(recv_timeout_ms)

        if input_mode == "pull":
            self.in_sock = self.ctx.socket(zmq.PULL)
        elif input_mode == "sub":
            self.in_sock = self.ctx.socket(zmq.SUB)
            self.in_sock.setsockopt_string(zmq.SUBSCRIBE, input_topic)
        else:
            raise ValueError("input_mode must be pull or sub")

        self.in_sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self.in_sock.setsockopt(zmq.RCVHWM, int(rcvhwm))

        if input_bind:
            self.in_sock.bind(input_bind)
        if input_connect:
            self.in_sock.connect(input_connect)

        if output_mode == "pub":
            self.out_sock = self.ctx.socket(zmq.PUB)
        elif output_mode == "push":
            self.out_sock = self.ctx.socket(zmq.PUSH)
        else:
            raise ValueError("output_mode must be pub or push")

        self.out_sock.setsockopt(zmq.SNDHWM, int(sndhwm))
        if output_bind:
            self.out_sock.bind(output_bind)
        if output_connect:
            self.out_sock.connect(output_connect)

        log_event(
            logging.INFO,
            "zmq_transport_started",
            input_mode=input_mode,
            input_bind=input_bind,
            input_connect=input_connect,
            output_mode=output_mode,
            output_bind=output_bind,
            output_connect=output_connect,
        )

    def recv_event(self) -> Optional[dict]:
        try:
            if self.input_mode == "sub":
                parts = self.in_sock.recv_multipart()
                if len(parts) == 1:
                    payload_b = parts[0]
                else:
                    payload_b = parts[-1]
            else:
                payload_b = self.in_sock.recv()
        except zmq.Again:
            return None

        try:
            return json.loads(payload_b.decode("utf-8"))
        except Exception as exc:
            log_event(logging.WARNING, "bad_zmq_message", error=str(exc))
            return None

    def publish(self, kind: str, payload: dict) -> None:
        payload = dict(payload)
        payload.setdefault("type", kind)
        payload.setdefault("ts_ns", str(now_ns()))
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        if self.output_mode == "pub":
            topic = f"{self.output_topic_prefix}.{kind}".encode("utf-8")
            self.out_sock.send_multipart([topic, body])
        else:
            self.out_sock.send(body)


# ---------------------------------------------------------------------------
# Trainer worker
# ---------------------------------------------------------------------------

@dataclass
class DatasetPolicy:
    min_train_samples: int
    min_val_samples: int
    min_test_samples: int
    min_train_pos: int
    min_train_neg: int
    min_val_pos: int
    min_val_neg: int
    min_test_pos: int
    min_test_neg: int


class TrainerWorkerZMQ:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = load_cfg(args.config) if args.config else {}
        # Production model identity: one shared fleet/global model by default.
        # site_id/store_id and cam_id remain event metadata only. They are useful
        # for reports and monitoring, but they do not partition the model.
        self.target_id = str(args.target_id or args.fleet_id or "global_retail")
        self.fleet_id = str(args.fleet_id or self.target_id)
        self.site_id = str(args.site_id or "")
        self.model_scope = str(args.model_scope or "fleet")
        self.scope_id = self.target_id

        self.transport = ZmqTransport(
            input_mode=args.input_mode,
            input_bind=args.input_bind,
            input_connect=args.input_connect,
            output_mode=args.output_mode,
            output_bind=args.output_bind,
            output_connect=args.output_connect,
            input_topic=args.input_topic,
            output_topic_prefix=args.output_topic_prefix,
            recv_timeout_ms=args.recv_timeout_ms,
            sndhwm=args.sndhwm,
            rcvhwm=args.rcvhwm,
        )

        self.root_dir = Path(args.out_dir).resolve()
        self.scope_dir = self.root_dir / self.model_scope / self.target_id
        self.manifests_dir = self.scope_dir / "manifests"
        self.checkpoints_dir = self.scope_dir / "checkpoints"
        self.registry_dir = self.scope_dir / "registry"
        self.rollback_dir = self.scope_dir / "rollback"
        for p in [self.manifests_dir, self.checkpoints_dir, self.registry_dir, self.rollback_dir]:
            p.mkdir(parents=True, exist_ok=True)

        self.champion_registry_path = self.registry_dir / "champion.json"
        self.champion_history_path = self.registry_dir / "champion_history.json"
        self.sample_cache_path = self.registry_dir / "samples.json"
        self.trainer_state_path = self.registry_dir / "trainer_state.json"
        self.metrics_path = self.registry_dir / "metrics.json"

        self.schema = default_scalar_schema()
        self.base_scalar_dim = len(self.schema) + 8
        self.use_identity_features = bool(args.use_identity_features)
        self.identity_dim = len(identity_feature_names()) if self.use_identity_features else 0
        self.scalar_dim = self.base_scalar_dim + self.identity_dim
        self.clip_channels = int(args.clip_channels)
        self.device = torch.device(args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu")

        self.runtime_meta = default_model_runtime_meta(self.cfg)
        self.default_rollout_mode = str(args.initial_rollout_state or self.runtime_meta.get("rollout_mode", "shadow"))
        if self.default_rollout_mode not in {"shadow", "canary", "active"}:
            self.default_rollout_mode = "shadow"

        self.allowed_clip_roots = [Path(x).expanduser().resolve(strict=False) for x in args.allowed_clip_root]
        self.max_clip_bytes = int(args.max_clip_mb * 1024 * 1024)
        self.clip_provider = ClipProvider(
            self.cfg,
            allowed_roots=self.allowed_clip_roots,
            max_clip_bytes=self.max_clip_bytes,
            temp_dir=args.clip_temp_dir,
            require_s3_clips=bool(args.require_s3_clips),
        )

        self.cloud = CloudArtifactStore(
            self.cfg,
            target_id=self.target_id,
            model_scope=self.model_scope,
            require_upload=bool(args.require_s3_upload),
        )

        self.dataset_policy = DatasetPolicy(
            min_train_samples=args.min_train_samples,
            min_val_samples=args.min_val_samples,
            min_test_samples=args.min_test_samples,
            min_train_pos=args.min_train_pos,
            min_train_neg=args.min_train_neg,
            min_val_pos=args.min_val_pos,
            min_val_neg=args.min_val_neg,
            min_test_pos=args.min_test_pos,
            min_test_neg=args.min_test_neg,
        )

        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

        persisted_samples = load_json(self.sample_cache_path, default={"samples": []})
        self.sample_records: Dict[str, dict] = {}
        self.seen_event_ids = set()
        for rec in persisted_samples.get("samples", []):
            eid = rec.get("event_id")
            if eid:
                self.sample_records[eid] = rec
                self.seen_event_ids.add(eid)

        trainer_state = load_json(self.trainer_state_path, default={})
        self.last_event_id = trainer_state.get("last_event_id")
        self.last_train_t = float(trainer_state.get("last_train_t", 0.0))
        self.events_seen_total = int(trainer_state.get("events_seen_total", 0))
        self.events_accepted_total = int(trainer_state.get("events_accepted_total", 0))
        self.events_rejected_total = int(trainer_state.get("events_rejected_total", 0))
        self.training_runs_total = int(trainer_state.get("training_runs_total", 0))

        self.champion_registry = load_json(self.champion_registry_path, default={})
        self.champion_history = load_json(self.champion_history_path, default={"history": []})

        self.sample_lock = threading.RLock()
        self.train_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.training_thread: Optional[threading.Thread] = None
        self.training_requested = threading.Event()
        self.last_metrics_publish_t = 0.0

        log_event(
            logging.INFO,
            "trainer_started",
            target_id=self.target_id,
            fleet_id=self.fleet_id,
            scope_id=self.scope_id,
            device=str(self.device),
            scalar_dim=self.scalar_dim,
            clip_channels=self.clip_channels,
            allowed_clip_roots=[str(p) for p in self.allowed_clip_roots],
            rollout_mode=self.default_rollout_mode,
            current_champion=self.champion_registry.get("model_version"),
        )

    def runtime_meta_fields(self) -> dict:
        return {
            "target_id": self.target_id,
            "fleet_id": self.fleet_id,
            "model_scope": self.model_scope,
            "task_type": self.runtime_meta["task_type"],
            "label_space": self.runtime_meta["label_space"],
            "domain_type": self.runtime_meta["domain_type"],
            "environment_family": self.runtime_meta["environment_family"],
            "site_scope": self.runtime_meta["site_scope"],
            "schema_version": self.runtime_meta["schema_version"],
            "identity_feature_version": self.runtime_meta["identity_feature_version"],
            "runtime_compat_version": self.runtime_meta["runtime_compat_version"],
        }

    def save_sample_cache(self) -> None:
        with self.sample_lock:
            payload = {"samples": sorted(self.sample_records.values(), key=lambda x: str(x.get("event_id", "")))}
        atomic_write_json(self.sample_cache_path, payload)

    def save_trainer_state(self) -> None:
        atomic_write_json(
            self.trainer_state_path,
            {
                "last_event_id": self.last_event_id,
                "last_train_t": float(self.last_train_t),
                "events_seen_total": self.events_seen_total,
                "events_accepted_total": self.events_accepted_total,
                "events_rejected_total": self.events_rejected_total,
                "training_runs_total": self.training_runs_total,
                "current_champion": self.champion_registry.get("model_version"),
                "updated_at_ns": now_ns(),
            },
        )

    def save_metrics(self, extra: Optional[dict] = None) -> None:
        with self.sample_lock:
            total = len(self.sample_records)
            pos = sum(1 for rec in self.sample_records.values() if int(rec["label"]) == 1)
            neg = total - pos
            label_quality = compute_label_quality(list(self.sample_records.values()))
        metrics = {
            "target_id": self.target_id,
            "fleet_id": self.fleet_id,
            "model_scope": self.model_scope,
            "target": self.scope_id,
            "updated_at_ns": now_ns(),
            "samples_total": total,
            "samples_pos": pos,
            "samples_neg": neg,
            "events_seen_total": self.events_seen_total,
            "events_accepted_total": self.events_accepted_total,
            "events_rejected_total": self.events_rejected_total,
            "training_runs_total": self.training_runs_total,
            "is_training": self.training_thread is not None and self.training_thread.is_alive(),
            "last_event_id": self.last_event_id,
            "last_train_t": self.last_train_t,
            "current_champion": self.champion_registry.get("model_version"),
            "current_rollout_state": self.champion_registry.get("rollout_state"),
            "label_quality": label_quality,
        }
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            except Exception:
                pass
        if extra:
            metrics.update(extra)
        atomic_write_json(self.metrics_path, metrics)

    def publish_health_if_due(self) -> None:
        now = time.time()
        if now - self.last_metrics_publish_t < float(self.args.metrics_every_s):
            return
        self.last_metrics_publish_t = now
        self.save_metrics()
        try:
            self.transport.publish("trainer_health", load_json(self.metrics_path, default={}))
        except Exception as exc:
            log_event(logging.WARNING, "health_publish_failed", error=str(exc))

    def build_model(self) -> FusionModel:
        m = FusionModel(clip_channels=self.clip_channels, scalar_dim=self.scalar_dim).to(self.device)
        m.train()
        return m

    def run_model_scores(self, model: nn.Module, C: np.ndarray, Xn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            clip_t = torch.from_numpy(C).to(self.device)
            X_t = torch.from_numpy(Xn).to(self.device)
            out = model(clip_t, X_t)
            fused_scores = torch.sigmoid(out["fused_logits"]).squeeze(1).cpu().numpy()
            cnn_scores = torch.sigmoid(out["cnn_logits"]).squeeze(1).cpu().numpy()
            mlp_scores = torch.sigmoid(out["mlp_logits"]).squeeze(1).cpu().numpy()
        return fused_scores, cnn_scores, mlp_scores

    def evaluate_model_bundle(self, model: nn.Module, C: np.ndarray, Xn: np.ndarray, y: np.ndarray, threshold: Optional[float] = None) -> dict:
        fused_scores, cnn_scores, mlp_scores = self.run_model_scores(model, C, Xn)
        if threshold is None:
            threshold = suggest_threshold_balanced(
                fused_scores,
                y,
                target_fpr=float(self.args.target_fpr),
                min_recall=float(self.args.min_recall_floor),
                min_alert_rate=float(self.args.min_alert_rate_floor),
                max_threshold=float(self.args.max_recommended_threshold),
            )
        fused_bal = metrics_with_intervals(fused_scores, y, thr=float(threshold))
        return {
            "fused_scores": fused_scores,
            "cnn_scores": cnn_scores,
            "mlp_scores": mlp_scores,
            "thr_balanced": float(threshold),
            "fused_at_05": metrics_with_intervals(fused_scores, y, thr=0.5),
            "cnn_at_05": metrics_with_intervals(cnn_scores, y, thr=0.5),
            "mlp_at_05": metrics_with_intervals(mlp_scores, y, thr=0.5),
            "fused_at_balanced": fused_bal,
            "fused_quality": float(balanced_quality(fused_bal, target_fpr=float(self.args.target_fpr))),
        }

    def _eval_without_arrays(self, eval_dict: dict) -> dict:
        out = dict(eval_dict)
        for k in ["fused_scores", "cnn_scores", "mlp_scores"]:
            if k in out:
                out[k] = out[k].tolist()
        return out

    def split_policy_ok(self, train_ids: List[str], val_ids: List[str], test_ids: List[str], labels_by_event: Dict[str, int]) -> Tuple[bool, List[str], dict]:
        counts = {
            "train": count_labels(train_ids, labels_by_event),
            "val": count_labels(val_ids, labels_by_event),
            "test": count_labels(test_ids, labels_by_event),
        }
        reasons: List[str] = []
        p = self.dataset_policy

        if counts["train"]["total"] < p.min_train_samples:
            reasons.append("train_samples_below_min")
        if counts["val"]["total"] < p.min_val_samples:
            reasons.append("val_samples_below_min")
        if counts["test"]["total"] < p.min_test_samples:
            reasons.append("test_samples_below_min")

        if counts["train"]["pos"] < p.min_train_pos:
            reasons.append("train_pos_below_min")
        if counts["train"]["neg"] < p.min_train_neg:
            reasons.append("train_neg_below_min")
        if counts["val"]["pos"] < p.min_val_pos:
            reasons.append("val_pos_below_min")
        if counts["val"]["neg"] < p.min_val_neg:
            reasons.append("val_neg_below_min")
        if counts["test"]["pos"] < p.min_test_pos:
            reasons.append("test_pos_below_min")
        if counts["test"]["neg"] < p.min_test_neg:
            reasons.append("test_neg_below_min")

        warnings = []
        for split_name, c in counts.items():
            if c["pos"] < 30 or c["neg"] < 30:
                warnings.append(f"{split_name}_low_class_count_confidence_intervals_wide")
        counts["warnings"] = warnings
        return len(reasons) == 0, reasons, counts

    def load_split_arrays(self, ids: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]]:
        clips, Xs, ys, metas = [], [], [], []
        with self.sample_lock:
            records = {eid: self.sample_records.get(eid) for eid in ids}

        for eid, rec in records.items():
            if not rec:
                continue
            clip_np = self.clip_provider.load_clip_tensor(
                rec.get("clip_uri") or rec.get("clip_path", ""),
                expected_channels=self.clip_channels,
            )
            if clip_np is None:
                continue
            x = np.asarray(rec["x"], dtype=np.float32)
            y = int(rec["label"])
            clips.append(clip_np.astype(np.float32))
            Xs.append(x.astype(np.float32))
            ys.append(y)
            metas.append(rec)

        if not clips:
            return None
        return (
            np.stack(clips, axis=0).astype(np.float32),
            np.stack(Xs, axis=0).astype(np.float32),
            np.array(ys, dtype=np.int64),
            metas,
        )

    def trigger_training_if_due(self, force: bool = False) -> None:
        if self.training_thread is not None and self.training_thread.is_alive():
            return
        now = time.time()
        with self.sample_lock:
            n = len(self.sample_records)
        if not force:
            if (now - self.last_train_t) < float(self.args.train_every_s):
                return
            if n < int(self.args.min_labeled):
                return
        self.training_thread = threading.Thread(target=self._training_thread_entry, daemon=True)
        self.training_thread.start()

    def _training_thread_entry(self) -> None:
        if not self.train_lock.acquire(blocking=False):
            return
        started = time.time()
        try:
            self.training_runs_total += 1
            self.train_and_publish()
        except torch.cuda.OutOfMemoryError as exc:
            log_event(logging.ERROR, "training_cuda_oom", error=str(exc))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            log_event(logging.ERROR, "training_failed", error=str(exc), traceback=traceback.format_exc())
        finally:
            self.last_train_t = time.time()
            duration = time.time() - started
            self.save_trainer_state()
            self.save_metrics({"last_training_duration_s": duration})
            self.train_lock.release()

    def train_and_publish(self) -> None:
        started = time.time()

        with self.sample_lock:
            all_event_ids = sorted(self.sample_records.keys())
            sample_records_snapshot = copy.deepcopy(self.sample_records)

        if len(all_event_ids) < int(self.args.min_labeled):
            log_event(logging.INFO, "skip_training_min_labeled", samples=len(all_event_ids))
            return

        dataset_version = build_dataset_version(all_event_ids)
        split_path = self.manifests_dir / f"split_{dataset_version}.json"
        dataset_path = self.manifests_dir / f"dataset_{dataset_version}.json"
        report_path = self.manifests_dir / f"report_{dataset_version}.json"

        labels_by_event = {eid: int(sample_records_snapshot[eid]["label"]) for eid in all_event_ids}

        split = load_json(split_path, default=None)
        if split is None:
            split = build_persistent_split(
                all_event_ids,
                train_ratio=float(self.args.train_ratio),
                val_ratio=float(self.args.val_ratio),
                seed=int(self.args.seed),
            )
            split = enforce_split_class_coverage(split, labels_by_event)
            atomic_write_json(split_path, split)

        train_ids = [eid for eid in split.get("train", []) if eid in sample_records_snapshot]
        val_ids = [eid for eid in split.get("val", []) if eid in sample_records_snapshot]
        test_ids = [eid for eid in split.get("test", []) if eid in sample_records_snapshot]

        policy_ok, policy_reasons, split_counts = self.split_policy_ok(train_ids, val_ids, test_ids, labels_by_event)
        if not policy_ok:
            log_event(logging.INFO, "skip_training_dataset_policy", reasons=policy_reasons, split_counts=split_counts)
            self.write_rejected_manifest(
                dataset_path,
                dataset_version,
                all_event_ids,
                split_path,
                accepted=False,
                reasons=policy_reasons,
                split_counts=split_counts,
            )
            return

        tr_pack = self.load_split_arrays(train_ids)
        va_pack = self.load_split_arrays(val_ids)
        te_pack = self.load_split_arrays(test_ids)
        if tr_pack is None or va_pack is None or te_pack is None:
            log_event(logging.WARNING, "skip_training_empty_split_after_clip_reload")
            return

        Ctr, Xtr_raw, ytr, _mtr = tr_pack
        Cva, Xva_raw, yva, _mva = va_pack
        Cte, Xte_raw, yte, _mte = te_pack

        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2 or len(np.unique(yte)) < 2:
            log_event(logging.INFO, "skip_training_missing_class_coverage")
            return

        x_mean, x_std = compute_scalar_norm(Xtr_raw)
        Xtr = normalize_X(Xtr_raw, x_mean, x_std)
        Xva = normalize_X(Xva_raw, x_mean, x_std)
        Xte = normalize_X(Xte_raw, x_mean, x_std)

        model = self.build_model()
        if bool(self.args.freeze_cnn):
            for p in model.cnn.parameters():
                p.requires_grad = False

        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(self.args.lr))
        pos = max(1, int((ytr == 1).sum()))
        neg = max(1, int((ytr == 0).sum()))
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_va_loss = float("inf")

        def run_epoch(Cb: np.ndarray, Xb: np.ndarray, yb: np.ndarray, train: bool) -> float:
            model.train(train)
            bs = int(self.args.batch_size)
            losses: List[float] = []
            # Shuffle only during training.
            order = np.arange(Xb.shape[0])
            if train:
                np.random.shuffle(order)
            for start_idx in range(0, Xb.shape[0], bs):
                idx = order[start_idx:start_idx + bs]
                clip_b = torch.from_numpy(Cb[idx]).to(self.device)
                xb = torch.from_numpy(Xb[idx]).to(self.device)
                yb_t = torch.from_numpy(yb[idx].astype(np.float32)).to(self.device).unsqueeze(1)
                with torch.set_grad_enabled(train):
                    out = model(clip_b, xb)
                    loss_fused = loss_fn(out["fused_logits"], yb_t)
                    loss_cnn = loss_fn(out["cnn_logits"], yb_t)
                    loss_mlp = loss_fn(out["mlp_logits"], yb_t)
                    loss = (
                        float(self.args.fusion_loss_weight) * loss_fused
                        + float(self.args.cnn_loss_weight) * loss_cnn
                        + float(self.args.mlp_loss_weight) * loss_mlp
                    )
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(self.args.max_grad_norm))
                        opt.step()
                losses.append(float(loss.item()))
            return float(np.mean(losses)) if losses else 0.0

        epoch_hist = []
        for ep in range(int(self.args.epochs)):
            tr_loss = run_epoch(Ctr, Xtr, ytr, train=True)
            va_loss = run_epoch(Cva, Xva, yva, train=False)
            epoch_hist.append({"epoch": ep + 1, "tr_loss": tr_loss, "va_loss": va_loss})
            log_event(logging.INFO, "training_epoch", epoch=ep + 1, train_loss=tr_loss, val_loss=va_loss)
            if va_loss < best_va_loss:
                best_va_loss = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            log_event(logging.WARNING, "candidate_rejected_no_best_state")
            return

        model.load_state_dict(best_state, strict=True)

        # Critical change: threshold is selected on validation only.
        challenger_val_eval = self.evaluate_model_bundle(model, Cva, Xva, yva, threshold=None)
        chosen_threshold = float(challenger_val_eval["thr_balanced"])

        # Test is evaluated with the validation-chosen threshold, and remains report-only.
        challenger_test_eval = self.evaluate_model_bundle(model, Cte, Xte, yte, threshold=chosen_threshold)

        val_metrics = challenger_val_eval["fused_at_balanced"]
        test_metrics = challenger_test_eval["fused_at_balanced"]

        label_quality = compute_label_quality([sample_records_snapshot[eid] for eid in all_event_ids])
        drift_score = compute_feature_drift_score(Xtr_raw, self.champion_registry.get("scalar_norm")) if self.champion_registry else None

        ts = int(time.time())
        version = f"fusion_{self.model_scope}_{self.target_id}_{ts}"
        weights_path = os.path.abspath(os.path.join(self.checkpoints_dir, f"{version}.pt"))
        weights_uri = self.cloud.uri_for("model", f"{version}.pt") or ""
        registry_uri = self.cloud.uri_for("registry", "champion.json") or ""
        history_uri = self.cloud.uri_for("registry", "champion_history.json") or ""
        dataset_uri = self.cloud.uri_for("dataset", f"dataset_{dataset_version}.json") or ""
        report_uri = self.cloud.uri_for("dataset", f"report_{dataset_version}.json") or ""

        ckpt = {
            "state_dict": model.state_dict(),
            "model_type": "split_fusion",
            "branch_outputs": ["cnn", "mlp", "fusion"],
            "model_version": version,
            "clip_channels": int(self.clip_channels),
            "scalar_dim": int(self.scalar_dim),
            "base_scalar_dim": int(self.base_scalar_dim),
            "identity_dim": int(self.identity_dim),
            "use_identity_features": bool(self.use_identity_features),
            "identity_feature_names": identity_feature_names() if self.use_identity_features else [],
            "schema_len": int(len(self.schema)),
            "scalar_schema": [{"name": n, "stat": s} for (n, s) in self.schema],
            **self.runtime_meta_fields(),
            "rollout_state": self.default_rollout_mode,
            "trained_at_ns": now_ns(),
            "target_id": self.target_id,
            "fleet_id": self.fleet_id,
            "model_scope": self.model_scope,
            "target": self.scope_id,
            "notes": "Split-output FusionModel trained by ZMQ background trainer.",
            "freeze_cnn": bool(self.args.freeze_cnn),
            "seed": int(self.args.seed),
            "split_version": dataset_version,
            "dataset_version": dataset_version,
            "train_samples": int(len(ytr)),
            "val_samples": int(len(yva)),
            "test_samples": int(len(yte)),
            "split_counts": split_counts,
            "loss_weights": {
                "fusion": float(self.args.fusion_loss_weight),
                "cnn": float(self.args.cnn_loss_weight),
                "mlp": float(self.args.mlp_loss_weight),
            },
            "epoch_history": epoch_hist,
            "best_val_loss": float(best_va_loss),
            "recommended_threshold_fused": chosen_threshold,
            "threshold_source": "validation",
            "weights_uri": weights_uri or "",
            "registry_uri": registry_uri or "",
            "history_uri": history_uri or "",
            "dataset_uri": dataset_uri or "",
            "report_uri": report_uri or "",
            "scalar_norm": {"mean": x_mean.tolist(), "std": x_std.tolist()},
            "label_quality": label_quality,
            "drift_score_vs_champion": drift_score,
            "val_eval": self._eval_without_arrays(challenger_val_eval),
            "test_eval_report_only": self._eval_without_arrays(challenger_test_eval),
            "candidate_status": "candidate",
        }
        torch.save(ckpt, weights_path)
        uploaded_weights_uri = self.cloud.upload_file(weights_path, "model")
        if uploaded_weights_uri:
            weights_uri = uploaded_weights_uri

        champion_eval = None
        champion_metrics = None
        champion_quality = None
        if self.champion_registry and self.champion_registry.get("weights_path") and os.path.exists(self.champion_registry["weights_path"]):
            try:
                champ_model = self.build_model()
                sd, _meta = load_checkpoint(self.champion_registry["weights_path"], self.device)
                champ_model.load_state_dict(sd, strict=False)
                champ_norm = self.champion_registry.get("scalar_norm", None)
                if champ_norm and champ_norm.get("mean") is not None and champ_norm.get("std") is not None:
                    Xva_champ = normalize_X(Xva_raw, np.asarray(champ_norm["mean"], dtype=np.float32), np.asarray(champ_norm["std"], dtype=np.float32))
                else:
                    Xva_champ = Xva
                champion_eval = self.evaluate_model_bundle(champ_model, Cva, Xva_champ, yva, threshold=None)
                champion_metrics = champion_eval["fused_at_balanced"]
                champion_quality = champion_eval["fused_quality"]
            except Exception as exc:
                log_event(logging.WARNING, "champion_eval_failed", error=str(exc))

        accepted, acceptance_reasons, acceptance_details = self.accept_candidate(
            challenger_metrics=val_metrics,
            challenger_quality=challenger_val_eval["fused_quality"],
            champion_metrics=champion_metrics,
            champion_quality=champion_quality,
            drift_score=drift_score,
        )

        dataset_manifest = {
            "dataset_version": dataset_version,
            "target_id": self.target_id,
            "fleet_id": self.fleet_id,
            "model_scope": self.model_scope,
            "target": self.scope_id,
            "created_at_ns": now_ns(),
            "num_samples_total": len(all_event_ids),
            "num_train": len(train_ids),
            "num_val": len(val_ids),
            "num_test": len(test_ids),
            "split_counts": split_counts,
            "event_ids": all_event_ids,
            "split_file": str(split_path),
            "label_quality": label_quality,
            "drift_score_vs_champion": drift_score,
            "challenger_version": version,
            "challenger_weights_path": weights_path,
            **self.runtime_meta_fields(),
            "rollout_state": self.default_rollout_mode,
            "threshold_source": "validation",
            "challenger_val_eval": self._eval_without_arrays(challenger_val_eval),
            "challenger_test_eval_report_only": self._eval_without_arrays(challenger_test_eval),
            "champion_registry_before": self.champion_registry,
            "champion_eval_on_current_val": self._eval_without_arrays(champion_eval) if champion_eval else None,
            "accepted": bool(accepted),
            "acceptance_reasons": acceptance_reasons,
            "acceptance_details": acceptance_details,
            "training_duration_s": time.time() - started,
        }
        atomic_write_json(dataset_path, dataset_manifest)
        atomic_write_json(report_path, dataset_manifest)
        uploaded_dataset_uri = self.cloud.upload_file(dataset_path, "dataset")
        uploaded_report_uri = self.cloud.upload_file(report_path, "dataset")
        if uploaded_dataset_uri:
            dataset_uri = uploaded_dataset_uri
        if uploaded_report_uri:
            report_uri = uploaded_report_uri

        if not accepted and not bool(self.args.publish_rejected_candidates):
            log_event(logging.INFO, "candidate_rejected", model_version=version, reasons=acceptance_reasons)
            return

        if accepted:
            self.publish_accepted_candidate(
                version=version,
                weights_path=weights_path,
                dataset_version=dataset_version,
                threshold=chosen_threshold,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                challenger_quality=challenger_val_eval["fused_quality"],
                x_mean=x_mean,
                x_std=x_std,
                label_quality=label_quality,
                drift_score=drift_score,
                split_counts=split_counts,
                weights_uri=weights_uri,
                dataset_uri=dataset_uri,
                report_uri=report_uri,
            )
        elif bool(self.args.publish_rejected_candidates):
            self.transport.publish("model_update", {
                "target": self.scope_id,
                "weights_path": weights_path,
                "weights_uri": weights_uri or "",
                "dataset_uri": dataset_uri or "",
                "report_uri": report_uri or "",
                "model_version": version,
                "model_type": "split_fusion",
                "candidate_status": "rejected",
                "dataset_version": dataset_version,
                "reject_reasons": json.dumps(acceptance_reasons),
                **self.runtime_meta_fields(),
                "rollout_state": "rejected",
            })

        self.cleanup_old_manifests()
        log_event(
            logging.INFO,
            "training_complete",
            model_version=version,
            accepted=accepted,
            val_recall=val_metrics.get("recall"),
            val_fpr=val_metrics.get("fpr"),
            test_recall_report_only=test_metrics.get("recall"),
            test_fpr_report_only=test_metrics.get("fpr"),
            threshold=chosen_threshold,
            duration_s=time.time() - started,
        )

    def write_rejected_manifest(
        self,
        dataset_path: Path,
        dataset_version: str,
        all_event_ids: List[str],
        split_path: Path,
        accepted: bool,
        reasons: List[str],
        split_counts: dict,
    ) -> None:
        atomic_write_json(
            dataset_path,
            {
                "dataset_version": dataset_version,
                "target_id": self.target_id,
            "fleet_id": self.fleet_id,
            "model_scope": self.model_scope,
            "target": self.scope_id,
                "created_at_ns": now_ns(),
                "num_samples_total": len(all_event_ids),
                "event_ids": all_event_ids,
                "split_file": str(split_path),
                "accepted": bool(accepted),
                "acceptance_reasons": reasons,
                "split_counts": split_counts,
                "status": "training_skipped_dataset_policy",
            },
        )

    def accept_candidate(
        self,
        *,
        challenger_metrics: dict,
        challenger_quality: float,
        champion_metrics: Optional[dict],
        champion_quality: Optional[float],
        drift_score: Optional[float],
    ) -> Tuple[bool, List[str], dict]:
        accepted = True
        reasons: List[str] = []
        details: dict = {}

        # Candidate checks are based on validation metrics only.
        if challenger_metrics["recall"] < float(self.args.champion_min_recall):
            accepted = False
            reasons.append("challenger_val_recall_below_min")
        if challenger_metrics["fpr"] > float(self.args.champion_max_fpr):
            accepted = False
            reasons.append("challenger_val_fpr_above_max")
        if drift_score is not None and drift_score > float(self.args.max_allowed_drift_score):
            accepted = False
            reasons.append("drift_score_too_high")

        if champion_metrics is not None and champion_quality is not None:
            quality_gain = float(challenger_quality - champion_quality)
            recall_drop = float(champion_metrics["recall"] - challenger_metrics["recall"])
            fpr_increase = float(challenger_metrics["fpr"] - champion_metrics["fpr"])
            details.update({
                "quality_gain_vs_champion": quality_gain,
                "recall_drop_vs_champion": recall_drop,
                "fpr_increase_vs_champion": fpr_increase,
            })
            if quality_gain < float(self.args.accept_min_quality_gain):
                accepted = False
                reasons.append("val_quality_gain_too_small")
            if recall_drop > float(self.args.accept_max_recall_drop):
                accepted = False
                reasons.append("val_recall_drop_too_large")
            if fpr_increase > float(self.args.accept_max_fpr_increase):
                accepted = False
                reasons.append("val_fpr_increase_too_large")
        else:
            details.update({
                "quality_gain_vs_champion": None,
                "recall_drop_vs_champion": None,
                "fpr_increase_vs_champion": None,
            })

        return accepted, reasons, details

    def publish_accepted_candidate(
        self,
        *,
        version: str,
        weights_path: str,
        dataset_version: str,
        threshold: float,
        val_metrics: dict,
        test_metrics: dict,
        challenger_quality: float,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        label_quality: dict,
        drift_score: Optional[float],
        split_counts: dict,
        weights_uri: Optional[str] = None,
        dataset_uri: Optional[str] = None,
        report_uri: Optional[str] = None,
    ) -> None:
        rollout_state = self.default_rollout_mode
        registry_uri = self.cloud.uri_for("registry", "champion.json")
        history_uri = self.cloud.uri_for("registry", "champion_history.json")

        model_update_msg = {
            "target": self.scope_id,
            "weights_path": weights_path,
            "weights_uri": weights_uri or "",
            "registry_uri": registry_uri or "",
            "dataset_uri": dataset_uri or "",
            "report_uri": report_uri or "",
            "model_version": version,
            "model_type": "split_fusion",
            "clip_channels": str(self.clip_channels),
            "scalar_dim": str(self.scalar_dim),
            "base_scalar_dim": str(self.base_scalar_dim),
            "identity_dim": str(self.identity_dim),
            "use_identity_features": "1" if self.use_identity_features else "0",
            "dataset_version": dataset_version,
            "candidate_status": "accepted",
            "rollout_state": rollout_state,
            "threshold_source": "validation",
            **self.runtime_meta_fields(),
        }
        self.transport.publish("model_update", model_update_msg)

        policy_msg = {
            "target": self.scope_id,
            "recommended_threshold": str(threshold),
            "threshold_source": "validation",
            "target_fpr": str(float(self.args.target_fpr)),
            "min_recall_floor": str(float(self.args.min_recall_floor)),
            "min_alert_rate_floor": str(float(self.args.min_alert_rate_floor)),
            "model_version": version,
            "registry_uri": registry_uri or "",
            "dataset_uri": dataset_uri or "",
            "report_uri": report_uri or "",
            "dataset_version": dataset_version,
            "candidate_status": "accepted",
            "rollout_state": rollout_state,
            "trained_samples": str(split_counts["train"]["total"] + split_counts["val"]["total"] + split_counts["test"]["total"]),
            "val_acc_fused": str(val_metrics["acc"]),
            "val_recall_fused": str(val_metrics["recall"]),
            "val_fpr_fused": str(val_metrics["fpr"]),
            "val_precision_fused": str(val_metrics["precision"]),
            "test_acc_fused_report_only": str(test_metrics["acc"]),
            "test_recall_fused_report_only": str(test_metrics["recall"]),
            "test_fpr_fused_report_only": str(test_metrics["fpr"]),
            "test_precision_fused_report_only": str(test_metrics["precision"]),
            "drift_score_vs_champion": "" if drift_score is None else str(drift_score),
            **self.runtime_meta_fields(),
        }

        optional_policy_fields = {
            "K": self.args.recommend_K,
            "M": self.args.recommend_M,
            "cooldown_s": self.args.recommend_cooldown_s,
            "use_suspicion_policy": self.args.recommend_use_suspicion_policy,
            "vote_use_heuristic_score": self.args.recommend_vote_use_heuristic_score,
            "vote_carry_score_thr": self.args.recommend_carry_thr,
            "vote_visibility_drop_thr": self.args.recommend_visibility_drop_thr,
            "vote_contact_ratio_thr": self.args.recommend_contact_ratio_thr,
            "heuristic_vote_thr": self.args.recommend_heuristic_thr,
            "suspicion_gain": self.args.recommend_susp_gain,
        }
        for k, v in optional_policy_fields.items():
            if v is not None:
                policy_msg[k] = str(v)

        self.transport.publish("policy_update", policy_msg)

        old_champion = copy.deepcopy(self.champion_registry)
        new_registry = {
            "target_id": self.target_id,
            "fleet_id": self.fleet_id,
            "model_scope": self.model_scope,
            "target": self.scope_id,
            "model_version": version,
            "weights_path": weights_path,
            "dataset_version": dataset_version,
            "val_metrics": val_metrics,
            "test_metrics_report_only": test_metrics,
            "val_quality": challenger_quality,
            "threshold_fused": float(threshold),
            "threshold_source": "validation",
            "weights_uri": weights_uri or "",
            "registry_uri": registry_uri or "",
            "history_uri": history_uri or "",
            "dataset_uri": dataset_uri or "",
            "report_uri": report_uri or "",
            "scalar_norm": {"mean": x_mean.tolist(), "std": x_std.tolist()},
            "accepted_at_ns": now_ns(),
            "label_quality": label_quality,
            "split_counts": split_counts,
            **self.runtime_meta_fields(),
            "rollout_state": rollout_state,
            "previous_champion": old_champion.get("model_version") if old_champion else None,
        }

        if old_champion:
            hist = self.champion_history.get("history", [])
            hist.append(old_champion)
            hist = hist[-int(self.args.keep_last_n_champions):]
            self.champion_history = {"history": hist}
            atomic_write_json(self.champion_history_path, self.champion_history)

        self.champion_registry = new_registry
        atomic_write_json(self.champion_registry_path, self.champion_registry)
        self.cloud.upload_file(self.champion_registry_path, "registry")
        self.cloud.upload_file(self.champion_history_path, "registry")

        log_event(
            logging.INFO,
            "candidate_accepted",
            model_version=version,
            rollout_state=rollout_state,
            threshold=threshold,
            val_recall=val_metrics.get("recall"),
            val_fpr=val_metrics.get("fpr"),
        )

    def rollback_to_model_version(self, model_version: str) -> bool:
        history = self.champion_history.get("history", [])
        candidate = None
        for item in reversed(history):
            if item.get("model_version") == model_version:
                candidate = item
                break
        if candidate is None:
            log_event(logging.ERROR, "rollback_model_not_found", model_version=model_version)
            return False
        if not candidate.get("weights_path") or not os.path.exists(candidate["weights_path"]):
            log_event(logging.ERROR, "rollback_weights_missing", model_version=model_version, weights_path=candidate.get("weights_path"))
            return False

        current = copy.deepcopy(self.champion_registry)
        if current:
            history.append(current)
        self.champion_history = {"history": history[-int(self.args.keep_last_n_champions):]}
        atomic_write_json(self.champion_history_path, self.champion_history)

        candidate["rollout_state"] = str(self.args.rollback_rollout_state)
        candidate["rollback_at_ns"] = now_ns()
        candidate["rollback_from"] = current.get("model_version") if current else None
        self.champion_registry = candidate
        atomic_write_json(self.champion_registry_path, self.champion_registry)

        self.transport.publish("model_update", {
            "target": self.scope_id,
            "weights_path": candidate["weights_path"],
            "model_version": candidate["model_version"],
            "model_type": candidate.get("model_type", "split_fusion"),
            "candidate_status": "rollback",
            "rollout_state": candidate["rollout_state"],
            "dataset_version": candidate.get("dataset_version"),
            **self.runtime_meta_fields(),
        })
        self.transport.publish("policy_update", {
            "target": self.scope_id,
            "recommended_threshold": str(candidate.get("threshold_fused", 0.8)),
            "threshold_source": candidate.get("threshold_source", "validation"),
            "model_version": candidate["model_version"],
            "candidate_status": "rollback",
            "rollout_state": candidate["rollout_state"],
            "dataset_version": candidate.get("dataset_version"),
            **self.runtime_meta_fields(),
        })
        log_event(logging.INFO, "rollback_published", model_version=model_version, rollout_state=candidate["rollout_state"])
        return True

    def promote_current(self, rollout_state: str) -> bool:
        if rollout_state not in {"shadow", "canary", "active"}:
            raise ValueError("--promote_to must be one of shadow, canary, active")
        if not self.champion_registry:
            log_event(logging.ERROR, "promote_failed_no_champion")
            return False
        self.champion_registry["rollout_state"] = rollout_state
        self.champion_registry["promoted_at_ns"] = now_ns()
        atomic_write_json(self.champion_registry_path, self.champion_registry)
        self.transport.publish("model_update", {
            "target": self.scope_id,
            "weights_path": self.champion_registry["weights_path"],
            "model_version": self.champion_registry["model_version"],
            "model_type": "split_fusion",
            "candidate_status": "promoted",
            "rollout_state": rollout_state,
            "dataset_version": self.champion_registry.get("dataset_version"),
            **self.runtime_meta_fields(),
        })
        self.transport.publish("policy_update", {
            "target": self.scope_id,
            "recommended_threshold": str(self.champion_registry.get("threshold_fused", 0.8)),
            "threshold_source": self.champion_registry.get("threshold_source", "validation"),
            "model_version": self.champion_registry["model_version"],
            "candidate_status": "promoted",
            "rollout_state": rollout_state,
            "dataset_version": self.champion_registry.get("dataset_version"),
            **self.runtime_meta_fields(),
        })
        log_event(logging.INFO, "champion_promoted", model_version=self.champion_registry["model_version"], rollout_state=rollout_state)
        return True

    def cleanup_old_manifests(self) -> None:
        manifests = sorted(self.manifests_dir.glob("dataset_*.json"))
        if len(manifests) > int(self.args.keep_last_n_manifests):
            for p in manifests[:-int(self.args.keep_last_n_manifests)]:
                try:
                    p.unlink()
                except Exception:
                    pass

    def handle_event(self, ev: dict) -> None:
        self.events_seen_total += 1

        event_id = ev.get("event_id")
        if not event_id:
            self.events_rejected_total += 1
            return

        self.last_event_id = str(event_id)

        # site_id/store_id is metadata only for the fleet/global trainer. We do
        # not reject events from other stores because the purpose of this node is
        # cross-store learning. If require_site_id is enabled, we only require
        # the metadata to exist so reports can show per-store distribution.
        event_site_id = ev.get("site_id") or ev.get("store_id")
        if bool(self.args.require_site_id) and not event_site_id:
            self.events_rejected_total += 1
            log_event(logging.WARNING, "event_rejected_missing_site_id", event_id=event_id)
            return

        # Optional safety filter: if an event explicitly names a different
        # fleet/target, reject it. This is off unless the field is present.
        event_fleet_id = ev.get("fleet_id") or ev.get("target_id")
        if event_fleet_id and str(event_fleet_id) != self.fleet_id and str(event_fleet_id) != self.target_id:
            self.events_rejected_total += 1
            log_event(logging.INFO, "event_rejected_wrong_fleet", event_id=event_id, event_fleet_id=event_fleet_id, trainer_target_id=self.target_id)
            return

        with self.sample_lock:
            if event_id in self.seen_event_ids:
                return

        sample = extract_training_sample(
            ev,
            schema=self.schema,
            scalar_dim=self.scalar_dim,
            base_scalar_dim=self.base_scalar_dim,
            identity_dim=self.identity_dim,
            use_identity_features=self.use_identity_features,
            expected_clip_channels=self.clip_channels,
            filter_bad_quality=bool(self.args.filter_bad_quality),
            max_missing_pose_ratio=float(self.args.filter_max_missing_pose_ratio),
            max_missing_obj_ratio=float(self.args.filter_max_missing_obj_ratio),
            require_gate_ok=bool(self.args.filter_require_gate_ok),
            require_train_ok=bool(self.args.require_train_ok),
            skip_feedback_needs_review=bool(self.args.skip_feedback_needs_review),
            min_feedback_confidence=self.args.min_feedback_confidence,
            clip_provider=self.clip_provider,
        )

        if sample is None:
            self.events_rejected_total += 1
            self.save_trainer_state()
            return

        clip_np, x, yv, meta, _status = sample
        rec = dict(meta)
        rec["event_id"] = event_id
        rec["label"] = int(yv)
        rec["x"] = x.astype(np.float32).tolist()
        rec["clip_shape"] = list(clip_np.shape)

        with self.sample_lock:
            self.sample_records[event_id] = rec
            self.seen_event_ids.add(event_id)
            total = len(self.sample_records)
            pos = sum(1 for r in self.sample_records.values() if int(r["label"]) == 1)
            neg = total - pos

        self.events_accepted_total += 1

        if self.events_accepted_total % int(self.args.sample_cache_every_n) == 0:
            self.save_sample_cache()
        self.save_trainer_state()

        log_event(logging.INFO, "training_sample_accepted", event_id=event_id, samples_total=total, pos=pos, neg=neg)

    def run_forever(self) -> None:
        backoff = float(self.args.initial_backoff_s)

        while not self.stop_event.is_set():
            try:
                ev = self.transport.recv_event()
                if ev is not None:
                    self.handle_event(ev)
                    if self.events_accepted_total % int(self.args.sample_cache_every_n) == 0:
                        self.save_sample_cache()

                self.trigger_training_if_due(force=False)
                self.publish_health_if_due()
                backoff = float(self.args.initial_backoff_s)

            except KeyboardInterrupt:
                break
            except Exception as exc:
                log_event(logging.ERROR, "service_loop_error", error=str(exc), traceback=traceback.format_exc())
                time.sleep(backoff)
                backoff = min(float(self.args.max_backoff_s), backoff * 2.0)

        self.shutdown()

    def shutdown(self) -> None:
        log_event(logging.INFO, "trainer_stopping")
        self.stop_event.set()
        self.save_sample_cache()
        self.save_trainer_state()
        self.save_metrics()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=float(self.args.shutdown_join_s))
        log_event(logging.INFO, "trainer_stopped")



# ---------------------------------------------------------------------------
# Pipeline configuration helpers
# ---------------------------------------------------------------------------

# This node is expected to be part of a larger ZMQ pipeline. In production, keep
# these settings in config.yaml and let the pipeline manager instantiate the
# worker from config. The long CLI command is only for manual testing.
#
# Recommended config.yaml section:
#
# trainer_node:
#   target_id: global_retail
#   fleet_id: global_retail
#   model_scope: fleet
#   require_site_id: false
#
#   zmq:
#     input_mode: sub
#     input_connect: tcp://127.0.0.1:5690
#     input_bind: null
#     input_topic: train_events.theft
#     output_mode: pub
#     output_bind: tcp://*:5691
#     output_connect: null
#     output_topic_prefix: trainer
#
#   storage:
#     out_dir: models
#     require_s3_clips: true
#     clip_temp_dir: null
#     max_clip_mb: 256
#     require_s3_upload: false
#
#   training:
#     min_labeled: 500
#     train_every_s: 600
#     device: cuda:0
#     epochs: 8
#     batch_size: 16
#     lr: 0.001
#     freeze_cnn: false
#     fusion_loss_weight: 1.0
#     cnn_loss_weight: 0.30
#     mlp_loss_weight: 0.30
#     max_grad_norm: 5.0
#     seed: 42
#     clip_channels: 19
#
# The actual clip location belongs inside each training event, not only in the
# config, because every event points to a different S3 object:
#
# {
#   "event_id": "evt_123",
#   "site_id": "store_001",
#   "cam_id": "cam02",
#   "fleet_id": "global_retail",
#   "payload": {
#     "clip_ref": {
#       "storage": "s3",
#       "s3_uri": "s3://zono-model/datasets/theft/global_retail/clips/evt_123.npz",
#       "tensor_key": "clip"
#     }
#   },
#   "feedback": {"label": 1, "llm_confidence": 0.92, "needs_review": false}
# }


def _cfg_section(cfg: dict) -> dict:
    """Return the trainer section while supporting a few legacy names."""
    if not isinstance(cfg, dict):
        return {}
    for key in ("trainer_node", "trainer", "trainer_node", "learning_trainer"):
        section = cfg.get(key)
        if isinstance(section, dict):
            return section
    return {}


def _set_if_present(args: argparse.Namespace, section: dict, attr: str, key: Optional[str] = None) -> None:
    if isinstance(section, dict):
        k = key or attr
        if k in section:
            setattr(args, attr, section[k])


def _set_many(args: argparse.Namespace, section: dict, names: List[str]) -> None:
    for name in names:
        _set_if_present(args, section, name)


def build_args_from_config(config: Optional[str | dict] = None, overrides: Optional[dict] = None) -> argparse.Namespace:
    """
    Build the same argparse.Namespace used by TrainerWorkerZMQ, but from config.

    Use this from your pipeline instead of passing a long command:

        from trainer_node import create_trainer_worker_from_config
        worker = create_trainer_worker_from_config("config.yaml")
        worker.run_forever()

    or, if the pipeline already owns the subscriber:

        worker = create_trainer_worker_from_config(cfg, start_transport=False)
        worker.handle_event(event_dict)

    The second pattern is useful when a central pipeline manager receives ZMQ
    messages and dispatches events directly to the trainer component.
    """
    parser = build_arg_parser()
    args = parser.parse_args([])

    if isinstance(config, dict):
        cfg = config
        args.config = None
    elif isinstance(config, str) and config:
        cfg = load_cfg(config)
        args.config = config
    else:
        cfg = {}
        args.config = None

    tcfg = _cfg_section(cfg)
    zcfg = tcfg.get("zmq", {}) if isinstance(tcfg.get("zmq", {}), dict) else {}
    scfg = tcfg.get("storage", {}) if isinstance(tcfg.get("storage", {}), dict) else {}
    train_cfg = tcfg.get("training", {}) if isinstance(tcfg.get("training", {}), dict) else {}
    dcfg = tcfg.get("dataset", {}) if isinstance(tcfg.get("dataset", {}), dict) else {}
    pcfg = tcfg.get("promotion", {}) if isinstance(tcfg.get("promotion", {}), dict) else {}
    qcfg = tcfg.get("quality", {}) if isinstance(tcfg.get("quality", {}), dict) else {}
    polcfg = tcfg.get("policy", {}) if isinstance(tcfg.get("policy", {}), dict) else {}
    ocfg = tcfg.get("observability", {}) if isinstance(tcfg.get("observability", {}), dict) else {}
    svcfg = tcfg.get("service", {}) if isinstance(tcfg.get("service", {}), dict) else {}
    admin_cfg = tcfg.get("admin", {}) if isinstance(tcfg.get("admin", {}), dict) else {}

    # Top-level trainer identity. The model is fleet/global by default; site_id
    # and cam_id remain metadata inside individual events.
    _set_many(args, tcfg, ["target_id", "fleet_id", "model_scope", "site_id", "cam_id", "require_site_id"])

    # ZMQ input/output. For your pipeline this is usually one SUB topic from the
    # data collector, e.g. train_events.theft.
    _set_many(args, zcfg, [
        "input_mode", "input_bind", "input_connect", "input_topic",
        "output_mode", "output_bind", "output_connect", "output_topic_prefix",
        "recv_timeout_ms", "sndhwm", "rcvhwm",
    ])

    # Storage. Production clips are in S3 and passed per event as clip_ref.s3_uri.
    _set_many(args, scfg, [
        "out_dir", "allowed_clip_root", "require_s3_clips", "clip_temp_dir",
        "max_clip_mb", "require_s3_upload",
    ])

    _set_many(args, train_cfg, [
        "min_labeled", "train_every_s", "device", "epochs", "batch_size", "lr",
        "freeze_cnn", "fusion_loss_weight", "cnn_loss_weight", "mlp_loss_weight",
        "max_grad_norm", "seed", "clip_channels",
    ])

    _set_many(args, dcfg, [
        "train_ratio", "val_ratio", "test_ratio", "min_train_samples", "min_val_samples",
        "min_test_samples", "min_train_pos", "min_train_neg", "min_val_pos", "min_val_neg",
        "min_test_pos", "min_test_neg",
    ])

    _set_many(args, pcfg, [
        "target_fpr", "min_recall_floor", "min_alert_rate_floor", "max_recommended_threshold",
        "initial_rollout_state", "publish_rejected_candidates", "champion_min_recall",
        "champion_max_fpr", "accept_min_quality_gain", "accept_max_recall_drop",
        "accept_max_fpr_increase", "max_allowed_drift_score",
    ])

    _set_many(args, qcfg, [
        "filter_bad_quality", "filter_require_gate_ok", "filter_max_missing_pose_ratio",
        "filter_max_missing_obj_ratio", "require_train_ok", "skip_feedback_needs_review",
        "min_feedback_confidence", "use_identity_features",
    ])

    _set_many(args, polcfg, [
        "recommend_K", "recommend_M", "recommend_cooldown_s", "recommend_use_suspicion_policy",
        "recommend_vote_use_heuristic_score", "recommend_carry_thr", "recommend_visibility_drop_thr",
        "recommend_contact_ratio_thr", "recommend_heuristic_thr", "recommend_susp_gain",
    ])

    _set_many(args, ocfg, [
        "log_level", "json_logs", "metrics_every_s", "sample_cache_every_n",
        "keep_last_n_manifests", "keep_last_n_champions",
    ])

    _set_many(args, svcfg, ["initial_backoff_s", "max_backoff_s", "shutdown_join_s"])
    _set_many(args, admin_cfg, ["force_train_once", "rollback_to_model_version", "rollback_rollout_state", "promote_to"])

    # Direct override dictionary wins over config. This is useful in tests or when
    # the parent pipeline injects runtime-specific socket endpoints.
    if overrides:
        for key, value in overrides.items():
            setattr(args, key, value)

    # argparse action="append" expects a list for allowed_clip_root.
    if args.allowed_clip_root is None:
        args.allowed_clip_root = []
    elif isinstance(args.allowed_clip_root, str):
        args.allowed_clip_root = [args.allowed_clip_root]

    return args


def create_trainer_worker_from_config(config: Optional[str | dict] = None, overrides: Optional[dict] = None) -> "TrainerWorkerZMQ":
    """
    Create the trainer as an importable pipeline component.

    This is the recommended production entrypoint. It avoids the long CLI command
    entirely; all ZMQ, S3, dataset, training, and rollout settings live in config.
    """
    args = build_args_from_config(config, overrides=overrides)
    setup_logging(str(args.log_level), bool(args.json_logs))
    validate_args(args)
    return TrainerWorkerZMQ(args)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default=None)
    ap.add_argument("--target_id", default="global_retail", help="Fleet/global model target, e.g. global_retail. This is the model/registry identity.")
    ap.add_argument("--fleet_id", default=None, help="Optional fleet identifier. Defaults to target_id.")
    ap.add_argument("--site_id", default=None, help="Optional local site metadata only. Not used to partition the model.")
    ap.add_argument("--model_scope", default="fleet", choices=["fleet", "global", "shared", "store", "site"], help="Training/model scope. Use fleet/global/shared for the common model.")
    ap.add_argument("--cam_id", default=None, help="Deprecated compatibility alias; retained only as event metadata.")
    ap.add_argument("--require_site_id", action="store_true", help="Require site_id/store_id metadata in events, but do not filter by one store.")

    # ZMQ
    ap.add_argument("--input_mode", choices=["pull", "sub"], default="pull")
    ap.add_argument("--input_bind", default="tcp://*:5690")
    ap.add_argument("--input_connect", default=None)
    ap.add_argument("--input_topic", default="train_event")
    ap.add_argument("--output_mode", choices=["pub", "push"], default="pub")
    ap.add_argument("--output_bind", default="tcp://*:5691")
    ap.add_argument("--output_connect", default=None)
    ap.add_argument("--output_topic_prefix", default="trainer")
    ap.add_argument("--recv_timeout_ms", type=int, default=50)
    ap.add_argument("--sndhwm", type=int, default=1000)
    ap.add_argument("--rcvhwm", type=int, default=1000)

    # Storage and safety
    ap.add_argument("--out_dir", default="models")
    ap.add_argument(
        "--allowed_clip_root",
        action="append",
        default=[],
        help="Development/local fallback only. Production clips should use S3 URIs.",
    )
    ap.add_argument(
        "--require_s3_clips",
        action="store_true",
        help="Reject local clip paths and accept only s3:// clip URIs from training events.",
    )
    ap.add_argument(
        "--clip_temp_dir",
        default=None,
        help="Optional temporary directory for short-lived S3 clip downloads. Files are deleted after loading.",
    )
    ap.add_argument("--max_clip_mb", type=int, default=256)
    ap.add_argument("--require_s3_upload", action="store_true", help="Fail training if model_cloud S3 upload fails.")

    # Training schedule
    ap.add_argument("--min_labeled", type=int, default=500)
    ap.add_argument("--train_every_s", type=float, default=600.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze_cnn", action="store_true")
    ap.add_argument("--fusion_loss_weight", type=float, default=1.0)
    ap.add_argument("--cnn_loss_weight", type=float, default=0.30)
    ap.add_argument("--mlp_loss_weight", type=float, default=0.30)
    ap.add_argument("--max_grad_norm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip_channels", type=int, default=19)

    # Threshold and promotion
    ap.add_argument("--target_fpr", type=float, default=0.05)
    ap.add_argument("--min_recall_floor", type=float, default=0.40)
    ap.add_argument("--min_alert_rate_floor", type=float, default=0.03)
    ap.add_argument("--max_recommended_threshold", type=float, default=0.95)
    ap.add_argument("--initial_rollout_state", choices=["shadow", "canary", "active"], default="shadow")
    ap.add_argument("--publish_rejected_candidates", action="store_true")

    # Dataset splits and stronger requirements
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--min_train_samples", type=int, default=250)
    ap.add_argument("--min_val_samples", type=int, default=100)
    ap.add_argument("--min_test_samples", type=int, default=100)
    ap.add_argument("--min_train_pos", type=int, default=50)
    ap.add_argument("--min_train_neg", type=int, default=50)
    ap.add_argument("--min_val_pos", type=int, default=20)
    ap.add_argument("--min_val_neg", type=int, default=20)
    ap.add_argument("--min_test_pos", type=int, default=20)
    ap.add_argument("--min_test_neg", type=int, default=20)

    # Candidate acceptance
    ap.add_argument("--champion_min_recall", type=float, default=0.35)
    ap.add_argument("--champion_max_fpr", type=float, default=0.20)
    ap.add_argument("--accept_min_quality_gain", type=float, default=0.01)
    ap.add_argument("--accept_max_recall_drop", type=float, default=0.03)
    ap.add_argument("--accept_max_fpr_increase", type=float, default=0.03)
    ap.add_argument("--max_allowed_drift_score", type=float, default=6.0)

    # Label and quality filtering
    ap.add_argument("--filter_bad_quality", action="store_true", default=True)
    ap.add_argument("--filter_require_gate_ok", action="store_true")
    ap.add_argument("--filter_max_missing_pose_ratio", type=float, default=0.55)
    ap.add_argument("--filter_max_missing_obj_ratio", type=float, default=0.75)
    ap.add_argument("--require_train_ok", action="store_true", default=True)
    ap.add_argument("--skip_feedback_needs_review", action="store_true", default=True)
    ap.add_argument("--min_feedback_confidence", type=float, default=0.80)
    ap.add_argument("--use_identity_features", action="store_true")

    # Policy recommendations
    ap.add_argument("--recommend_K", type=int, default=None)
    ap.add_argument("--recommend_M", type=int, default=None)
    ap.add_argument("--recommend_cooldown_s", type=float, default=None)
    ap.add_argument("--recommend_use_suspicion_policy", type=str, default=None)
    ap.add_argument("--recommend_vote_use_heuristic_score", type=str, default=None)
    ap.add_argument("--recommend_carry_thr", type=float, default=None)
    ap.add_argument("--recommend_visibility_drop_thr", type=float, default=None)
    ap.add_argument("--recommend_contact_ratio_thr", type=float, default=None)
    ap.add_argument("--recommend_heuristic_thr", type=float, default=None)
    ap.add_argument("--recommend_susp_gain", type=float, default=None)

    # Observability and retention
    ap.add_argument("--log_level", default="INFO")
    ap.add_argument("--json_logs", action="store_true")
    ap.add_argument("--metrics_every_s", type=float, default=30.0)
    ap.add_argument("--sample_cache_every_n", type=int, default=10)
    ap.add_argument("--keep_last_n_manifests", type=int, default=20)
    ap.add_argument("--keep_last_n_champions", type=int, default=10)

    # Service handling
    ap.add_argument("--initial_backoff_s", type=float, default=0.5)
    ap.add_argument("--max_backoff_s", type=float, default=30.0)
    ap.add_argument("--shutdown_join_s", type=float, default=10.0)

    # Admin actions
    ap.add_argument("--force_train_once", action="store_true")
    ap.add_argument("--rollback_to_model_version", default=None)
    ap.add_argument("--rollback_rollout_state", choices=["shadow", "canary", "active"], default="shadow")
    ap.add_argument("--promote_to", choices=["shadow", "canary", "active"], default=None)

    return ap


def validate_args(args: argparse.Namespace) -> None:
    if not getattr(args, "target_id", None):
        args.target_id = getattr(args, "fleet_id", None) or "global_retail"
    if not getattr(args, "fleet_id", None):
        args.fleet_id = args.target_id

    total = float(args.train_ratio) + float(args.val_ratio) + float(args.test_ratio)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if not args.input_bind and not args.input_connect:
        raise ValueError("Set either --input_bind or --input_connect")

    if not args.output_bind and not args.output_connect:
        raise ValueError("Set either --output_bind or --output_connect")

    if not args.allowed_clip_root:
        env_root = os.environ.get("TRAINER_ALLOWED_CLIP_ROOT")
        if env_root:
            args.allowed_clip_root = [env_root]

    # In production the collector should upload clips to S3 and publish s3:// URIs.
    # Local clips are allowed only as a dev fallback, and then they must be under
    # an allowed root to avoid path traversal / arbitrary file reads.
    if not args.require_s3_clips and not args.allowed_clip_root:
        log_event(
            logging.WARNING,
            "local_clip_fallback_disabled_no_allowed_root",
            note="Use --require_s3_clips for production or set --allowed_clip_root for local development.",
        )


def main() -> None:
    # CLI is a convenience wrapper. In the real pipeline, prefer:
    #     worker = create_trainer_worker_from_config("config.yaml")
    #     worker.run_forever()
    #
    # This CLI still supports old flags, but when --config is provided the config
    # values are loaded first. Explicit CLI overrides can be passed by your test
    # command only when needed.
    ap = build_arg_parser()
    raw_args = ap.parse_args()

    if raw_args.config:
        # Load production defaults from config.yaml so the service does not need
        # a long command. For manual tests, explicit CLI values still exist on
        # raw_args, but config should be the source of truth in the pipeline.
        args = build_args_from_config(raw_args.config)
        # Preserve admin one-shot CLI actions because they are convenient during
        # maintenance even when the node normally runs from config.
        for name in ("force_train_once", "rollback_to_model_version", "rollback_rollout_state", "promote_to"):
            setattr(args, name, getattr(raw_args, name))
    else:
        args = raw_args

    setup_logging(args.log_level, args.json_logs)
    validate_args(args)

    worker = TrainerWorkerZMQ(args)

    def _handle_signal(signum, _frame):
        log_event(logging.INFO, "signal_received", signum=signum)
        worker.stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if args.promote_to:
        worker.promote_current(args.promote_to)
        worker.shutdown()
        return

    if args.rollback_to_model_version:
        worker.rollback_to_model_version(args.rollback_to_model_version)
        worker.shutdown()
        return

    if args.force_train_once:
        worker.trigger_training_if_due(force=True)
        if worker.training_thread:
            worker.training_thread.join()
        worker.shutdown()
        return

    worker.run_forever()


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Compatibility layer for brain.py orchestration.
# brain.py expects:
#   from trainer_node import TrainerWorker
#   trainer.step()
#
# The ZMQ trainer implementation may internally be named TrainerWorkerZMQ.
# These aliases keep the pipeline interface stable.
# ---------------------------------------------------------------------

try:
    TrainerWorker
except NameError:
    try:
        TrainerWorker = TrainerWorkerZMQ
    except NameError:
        pass


def _trainer_worker_step_compat(self):
    """
    Single non-blocking orchestration step for brain.py.

    Receives at most one training event, handles it, then triggers background
    training/health publishing if the implementation provides those methods.
    """
    ev = None

    try:
        if hasattr(self, "transport") and hasattr(self.transport, "recv_event"):
            ev = self.transport.recv_event()
        elif hasattr(self, "recv_event"):
            ev = self.recv_event()
    except Exception:
        ev = None

    if ev is not None and hasattr(self, "handle_event"):
        self.handle_event(ev)

    for method_name in (
        "trigger_training_if_due",
        "maybe_train_and_publish",
        "publish_health_if_due",
    ):
        method = getattr(self, method_name, None)
        if callable(method):
            try:
                method()
            except TypeError:
                try:
                    method(force=False)
                except TypeError:
                    pass

    return ev is not None


try:
    if "TrainerWorker" in globals() and not hasattr(TrainerWorker, "step"):
        TrainerWorker.step = _trainer_worker_step_compat
except Exception:
    pass
