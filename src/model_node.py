#!/usr/bin/env python3
"""
model_node.py

Phase 2 performance-focused version.

This version builds on the Phase 1 safe refactor and keeps the same external workflow:
  - same ZMQ inputs
  - same ZMQ score output topic/message structure
  - same Redis model-update workflow
  - same model architecture
  - same score JSON output keys
  - same default behavior unless Phase 2 flags are enabled

Phase 2 additions:
  - optional low-latency clip draining with --low_latency
  - scalar-event batch processing without dropping scalar cache events
  - optional Redis update throttling with --redis_update_interval_s
  - better timing/stats visibility
  - safer metadata conversion in output
  - CUDA warmup and inference_mode retained

Important:
  - By default, behavior remains very close to Phase 1.
  - --low_latency may drop old clip events and score only the newest clip event in the socket queue.
  - Scalar events are not dropped in low-latency mode; they are batch-processed into the scalar cache.
"""

import argparse
import json
import time
from pathlib import Path
from collections import deque
from contextlib import nullcontext

import zmq
import numpy as np
import torch
import torch.nn as nn

try:
    import redis
except Exception:
    redis = None

from config_utils import (
    load_cfg,
    get_stream,
    get_zmq_endpoint,
    local_connect_addr,
)


DEFAULT_CLIP_CHANNELS = 19
DEFAULT_CLIP_T = 16
DEFAULT_CLIP_H = 112
DEFAULT_CLIP_W = 112
DEFAULT_CLIP_SHAPE = (
    DEFAULT_CLIP_CHANNELS,
    DEFAULT_CLIP_T,
    DEFAULT_CLIP_H,
    DEFAULT_CLIP_W,
)

MODEL_META_DEFAULTS = {
    "task_type": "theft_detection",
    "label_space": "binary_theft",
    "domain_type": "retail",
    "environment_family": "generic",
    "site_scope": "global",
    "schema_version": "scalars_v1",
    "identity_feature_version": "idfeat_v1",
    "runtime_compat_version": "model_node_v1",
}


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def parse_xread_messages(streams):
    out = []
    for _sname, msgs in streams:
        for mid, fields in msgs:
            out.append((b2s(mid), fields))
    return out


def safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=None):
    try:
        return int(v)
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


def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def now_mono():
    return time.monotonic()


def now_wall():
    return time.time()


def parse_zmq_json_payload(parts):
    if len(parts) != 3:
        return None
    try:
        return json.loads(parts[2].decode("utf-8"))
    except Exception:
        return None


def make_sub_socket(ctx, connect_addr, topic_b, rcvhwm, rcvbuf_mb=0):
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, int(rcvhwm))

    if rcvbuf_mb and float(rcvbuf_mb) > 0:
        try:
            sock.setsockopt(zmq.RCVBUF, int(float(rcvbuf_mb) * 1024 * 1024))
        except Exception:
            pass

    try:
        sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
        sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 30)
        sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 10)
    except Exception:
        pass

    sock.connect(connect_addr)
    sock.setsockopt(zmq.SUBSCRIBE, topic_b)
    return sock


def maybe_cuda_synchronize(device, enabled):
    if enabled and device.type == "cuda":
        torch.cuda.synchronize()


def sigmoid_item(x):
    return torch.sigmoid(x)[0, 0].item()


def normalize_scalar_vector(vec: np.ndarray, norm_meta: dict):
    if not isinstance(norm_meta, dict):
        return vec

    mean = norm_meta.get("mean", None)
    std = norm_meta.get("std", None)
    if mean is None or std is None:
        return vec

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if mean.shape != vec.shape or std.shape != vec.shape:
        return vec

    std = np.where(std < 1e-6, 1.0, std)
    return (vec - mean) / std


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

    def forward(self, x):
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

    def forward(self, x):
        return self.net(x)


class BinaryHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, z):
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

    def forward(self, clip, scalars):
        zc = self.cnn(clip)
        zs = self.smlp(scalars)

        cnn_logits = self.cnn_head(zc)
        mlp_logits = self.mlp_head(zs)
        fused_logits = self.fusion_head(torch.cat([zc, zs], dim=1))

        return {
            "cnn_logits": cnn_logits,
            "mlp_logits": mlp_logits,
            "fused_logits": fused_logits,
            "cnn_emb": zc,
            "mlp_emb": zs,
        }


def _get_stat(d, key, stat):
    try:
        v = d.get(key, None)
        if isinstance(v, dict):
            return v.get(stat, None)
    except Exception:
        pass
    return None


def build_scalar_vector(features: dict, schema: list, fill_value: float = 0.0):
    vec = []
    miss = 0
    for (fname, stat) in schema:
        val = _get_stat(features, fname, stat)
        if val is None:
            vec.append(float(fill_value))
            miss += 1
        else:
            vec.append(float(val))
    return np.array(vec, dtype=np.float32), miss


def default_scalar_schema():
    s = []
    for nm in ["hand_to_chest_L", "hand_to_chest_R", "hand_to_hip_L", "hand_to_hip_R"]:
        for st in ["mean", "max", "last"]:
            s.append((nm, st))
    for nm in ["wrist_v_L", "wrist_v_R"]:
        for st in ["mean", "max", "last"]:
            s.append((nm, st))

    s.append(("torso_angle_rate", "mean"))
    s.append(("torso_angle_rate", "max"))

    s.append(("guard_score", "mean"))
    s.append(("guard_score", "max"))
    s.append(("guard_score", "last"))

    for nm in ["obj_visibility", "carry_score", "comotion_cosine", "rel_offset_std_px"]:
        for st in ["mean", "min", "max", "last"]:
            s.append((nm, st))

    for nm in ["person_speed_px_s", "object_speed_px_s"]:
        for st in ["mean", "max", "last"]:
            s.append((nm, st))

    return s


def append_special_scalars(vec: np.ndarray, agg_features: dict):
    def fget(k, default=0.0):
        v = agg_features.get(k, None)
        if v is None:
            return float(default)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        try:
            return float(v)
        except Exception:
            return float(default)

    extras = np.array([
        fget("contact_ratio", 0.0),
        fget("contact_duration_frames_last", 0.0),
        fget("contact_duration_frames_max", 0.0),
        fget("visibility_drop", 0.0),
        fget("visibility_min", 0.0),
        fget("disappeared_after_contact", 0.0),
        fget("missing_pose_ratio", 1.0),
        fget("missing_obj_ratio", 1.0),
    ], dtype=np.float32)

    return np.concatenate([vec, extras], axis=0)


def identity_feature_names():
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


def build_identity_features(global_person_id, id_state, now_s, recent_window_s=30.0):
    names = identity_feature_names()

    if global_person_id is None:
        return np.zeros((len(names),), dtype=np.float32), {k: 0.0 for k in names}

    st = id_state.get(global_person_id, None)
    if not st:
        return np.zeros((len(names),), dtype=np.float32), {k: 0.0 for k in names}

    last_seen_s = st.get("last_seen_s", None)
    cams_recent = st.get("cams_recent", set()) or set()
    scores_recent = st.get("scores_recent", deque())
    suspicion_recent = st.get("suspicion_recent", deque())

    id_seen_recently = 1.0
    id_seen_other_cam_recently = 1.0 if len(cams_recent) > 1 else 0.0

    if last_seen_s is None:
        dt = recent_window_s
    else:
        dt = max(0.0, float(now_s - last_seen_s))

    id_time_since_last_seen_s = min(dt, recent_window_s) / max(1e-6, recent_window_s)
    id_recent_cam_count = min(len(cams_recent), 8) / 8.0
    id_recent_event_count = min(len(scores_recent), 20) / 20.0
    id_prior_score_ema = clamp01(st.get("score_ema", 0.0))
    id_prior_score_max = clamp01(max(scores_recent) if len(scores_recent) > 0 else 0.0)
    id_prior_suspicion_mean = clamp01(float(np.mean(suspicion_recent)) if len(suspicion_recent) > 0 else 0.0)

    feats = np.array([
        id_seen_recently,
        id_seen_other_cam_recently,
        id_time_since_last_seen_s,
        id_recent_cam_count,
        id_recent_event_count,
        id_prior_score_ema,
        id_prior_score_max,
        id_prior_suspicion_mean,
    ], dtype=np.float32)

    feat_dict = dict(zip(names, feats.tolist()))
    return feats, feat_dict


def compute_identity_memory_score(global_person_id, id_state):
    if global_person_id is None:
        return 0.0

    st = id_state.get(global_person_id, None)
    if not st:
        return 0.0

    score_ema = clamp01(st.get("score_ema", 0.0))
    suspicion_recent = st.get("suspicion_recent", deque())
    prior_susp_mean = clamp01(float(np.mean(suspicion_recent)) if len(suspicion_recent) > 0 else 0.0)
    multi_cam_bonus = 0.10 if len(st.get("cams_recent", set()) or set()) > 1 else 0.0

    mem = 0.65 * score_ema + 0.35 * prior_susp_mean + multi_cam_bonus
    return clamp01(mem)


def update_identity_state(id_state, global_person_id, cam, event_time_s, final_score, raw_fused_score, max_recent=20, ema_alpha=0.25):
    if global_person_id is None:
        return

    st = id_state.get(global_person_id, None)
    if st is None:
        st = {
            "last_seen_s": None,
            "cams_recent": set(),
            "score_ema": 0.0,
            "scores_recent": deque(maxlen=max_recent),
            "suspicion_recent": deque(maxlen=max_recent),
            "history": deque(maxlen=max_recent),
        }
        id_state[global_person_id] = st

    prev_ema = float(st.get("score_ema", 0.0))
    st["score_ema"] = (1.0 - ema_alpha) * prev_ema + ema_alpha * float(raw_fused_score)
    st["last_seen_s"] = float(event_time_s)
    st["cams_recent"].add(str(cam))
    st["scores_recent"].append(float(raw_fused_score))
    st["suspicion_recent"].append(float(final_score))
    st["history"].append({
        "cam_id": str(cam),
        "seen_at_s": float(event_time_s),
        "raw_fused_score": float(raw_fused_score),
        "final_score": float(final_score),
    })


def prune_identity_state(id_state, now_s, max_idle_s):
    dead = []
    for gid, st in id_state.items():
        last_seen_s = st.get("last_seen_s", None)
        if last_seen_s is None:
            dead.append(gid)
            continue
        if (now_s - float(last_seen_s)) > float(max_idle_s):
            dead.append(gid)

    for gid in dead:
        id_state.pop(gid, None)


def load_checkpoint(weights_path: str, device: torch.device):
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        print("[model_node] WARNING torch.load weights_only unsupported; loading trusted local checkpoint")
        ckpt = torch.load(weights_path, map_location=device)
    except Exception as exc:
        print(f"[model_node] WARNING safe checkpoint load failed; falling back for trusted local path: {exc}")
        ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        state_dict = ckpt
        meta = {}
    return state_dict, meta


def build_model(clip_channels: int, scalar_dim: int, device: torch.device):
    m = FusionModel(clip_channels=clip_channels, scalar_dim=scalar_dim)
    m.to(device)
    m.eval()
    return m


def validate_model_forward(model: nn.Module, clip_shape, scalar_dim: int, device: torch.device, use_amp: bool):
    c, t, h, w = clip_shape
    clip = torch.zeros((1, c, t, h, w), dtype=torch.float32, device=device)
    scalars = torch.zeros((1, scalar_dim), dtype=torch.float32, device=device)

    with torch.inference_mode():
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(clip, scalars)
        else:
            out = model(clip, scalars)

    required = ["cnn_logits", "mlp_logits", "fused_logits"]
    for k in required:
        if k not in out:
            raise RuntimeError(f"Validation failed: missing output {k}.")
        v = out[k]
        if v is None or v.numel() != 1:
            raise RuntimeError(f"Validation failed: bad shape for {k}.")
        if torch.isnan(v).any():
            raise RuntimeError(f"Validation failed: NaNs in {k}.")
    return True


def extract_policy_features(obj: dict, agg_feats: dict):
    pf = obj.get("policy_features", {})
    if isinstance(pf, dict) and pf:
        return pf

    carry_max = None
    carry_dict = agg_feats.get("carry_score", None)
    if isinstance(carry_dict, dict):
        carry_max = carry_dict.get("max", None)

    return {
        "contact_ratio": agg_feats.get("contact_ratio", None),
        "visibility_drop": agg_feats.get("visibility_drop", None),
        "carry_score_max": carry_max,
        "disappeared_after_contact": agg_feats.get("disappeared_after_contact", None),
        "heuristic_theft_score": agg_feats.get("heuristic_theft_score", None),
        "clip_dt_s": agg_feats.get("clip_dt_s", None),
        "missing_pose_ratio": agg_feats.get("missing_pose_ratio", None),
        "missing_obj_ratio": agg_feats.get("missing_obj_ratio", None),
    }


def load_champion_registry(registry_path: Path):
    if not registry_path.exists():
        return None
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def default_model_meta(policy: dict):
    return {k: str(policy.get(k, v)) for k, v in MODEL_META_DEFAULTS.items()}


def merge_model_meta(meta: dict, policy: dict):
    base = default_model_meta(policy)
    if not isinstance(meta, dict):
        return base
    out = dict(base)
    for k in base.keys():
        v = meta.get(k, None)
        if v is not None:
            out[k] = str(v)
    return out


class ModelWorker:
    def __init__(self, args):
        self.args = args
        self.cfg = load_cfg(args.config)
        self.cam_id = str(args.cam_id)

        model_node_cfg = self.cfg.get("brain", {}).get("model_node_args", {})

        def cfg_default(name, default):
            return model_node_cfg.get(name, getattr(args, name, default))

        self.args.disable_redis_updates = cfg_default("disable_redis_updates", False)
        self.args.clip_connect = cfg_default("clip_connect", None)
        self.args.clip_topic = cfg_default("clip_topic", None)
        self.args.scalars_connect = cfg_default("scalars_connect", None)
        self.args.scalars_topic = cfg_default("scalars_topic", None)
        self.args.global_tracks_connect = cfg_default("global_tracks_connect", None)
        self.args.global_tracks_topic = cfg_default("global_tracks_topic", None)
        self.args.scores_bind = cfg_default("scores_bind", None)
        self.args.scores_topic = cfg_default("scores_topic", None)
        self.args.updates_stream = cfg_default("updates_stream", None)
        self.args.updates_target = cfg_default("updates_target", None)
        self.args.allow_runtime_model_selection = cfg_default("allow_runtime_model_selection", False)
        self.args.enable_identity_memory_fusion = cfg_default("enable_identity_memory_fusion", False)
        self.args.identity_memory_weight = float(cfg_default("identity_memory_weight", 0.20))
        self.args.identity_recent_window_s = float(cfg_default("identity_recent_window_s", 30.0))
        self.args.identity_state_idle_s = float(cfg_default("identity_state_idle_s", 60.0))
        self.args.global_cache_ttl_s = float(cfg_default("global_cache_ttl_s", 60.0))
        self.args.wait_for_global_id_s = float(cfg_default("wait_for_global_id_s", 1.0))
        self.args.global_id_frame_tolerance = int(cfg_default("global_id_frame_tolerance", 12))
        self.args.scalar_cache_size = int(cfg_default("scalar_cache_size", 5000))
        self.args.scalar_cache_ttl_s = float(cfg_default("scalar_cache_ttl_s", 5.0))
        self.args.wait_for_scalars_s = float(
            cfg_default("wait_for_scalars_s", getattr(args, "wait_for_scalars_s", 0.20))
        )
        self.args.max_drain_per_step = int(cfg_default("max_drain_per_step", 64))
        self.args.check_updates_every_loops = int(cfg_default("check_updates_every_loops", 1))
        self.args.prefer_champion_registry = cfg_default("prefer_champion_registry", False)
        self.args.champion_registry_dir = cfg_default("champion_registry_dir", "models")
        self.args.weights = cfg_default("weights", None)
        self.args.model_version = str(cfg_default("model_version", "bootstrap_v0"))
        self.args.device = str(cfg_default("device", "cuda:0"))
        self.args.fp16 = bool(cfg_default("fp16", False))
        self.args.zmq_rcvhwm = int(cfg_default("zmq_rcvhwm", 256))
        self.args.block_ms = int(cfg_default("block_ms", 1000))

        self.args.debug = bool(cfg_default("debug", getattr(args, "debug", False)))
        self.args.debug_timing = bool(cfg_default("debug_timing", getattr(args, "debug_timing", False)))
        self.args.debug_schema = bool(cfg_default("debug_schema", getattr(args, "debug_schema", False)))
        self.args.stats_every_s = float(cfg_default("stats_every_s", getattr(args, "stats_every_s", 0.0)))

        self.args.low_latency = bool(cfg_default("low_latency", getattr(args, "low_latency", False)))
        self.args.max_clip_drain = int(cfg_default("max_clip_drain", getattr(args, "max_clip_drain", 32)))
        self.args.max_scalar_batch = int(cfg_default("max_scalar_batch", getattr(args, "max_scalar_batch", 64)))
        self.args.redis_update_interval_s = float(cfg_default("redis_update_interval_s", getattr(args, "redis_update_interval_s", 0.0)))
        self.args.zmq_rcvbuf_mb = float(cfg_default("zmq_rcvbuf_mb", getattr(args, "zmq_rcvbuf_mb", 0.0)))

        self.stats = {
            "scalar_events_received": 0,
            "clip_events_received": 0,
            "clip_events_scored": 0,
            "clip_events_skipped": 0,
            "clip_events_dropped_low_latency": 0,
            "global_updates_received": 0,
            "scalar_missing": 0,
            "model_updates_seen": 0,
            "model_updates_applied": 0,
            "model_updates_rejected": 0,
        }
        self.last_stats_print_s = now_mono()
        self.last_redis_update_check_s = 0.0

        self.runtime_policy = self.cfg.get("model_runtime", {})
        self.compat_policy = self.cfg.get("model_compatibility", {})
        self.cloud_policy = self.cfg.get("model_cloud", {})

        self.runtime_policy_defaults = {
            "task_type": "theft_detection",
            "label_space": "binary_theft",
            "domain_type": "retail",
            "environment_family": "generic",
            "site_scope": "global",
            "schema_version": "scalars_v1",
            "identity_feature_version": "idfeat_v1",
            "runtime_compat_version": "model_node_v1",
            "allow_global_fallback": True,
            "rollout_mode": "active",
            "blend_alpha": 0.35,
            "shadow_score_guard_enabled": False,
            "score_shift_guard": 0.35,
            "max_alert_rate_delta": 0.20,
        }
        for k, v in self.runtime_policy_defaults.items():
            self.runtime_policy.setdefault(k, v)

        self.rdb = None
        self.updates_stream = None
        self.last_upd_id = "0-0"

        self._setup_redis()
        self._setup_zmq()
        self._setup_device_and_model_state()
        self._initial_model_selection()
        self._print_banner()

    def _jcompact(self, obj):
        return json.dumps(obj, separators=(",", ":"))

    def log_debug(self, msg):
        if getattr(self.args, "debug", False):
            print(msg)

    def maybe_print_stats(self):
        if self.args.stats_every_s <= 0:
            return
        now = now_mono()
        if (now - self.last_stats_print_s) < self.args.stats_every_s:
            return
        self.last_stats_print_s = now
        print(f"[model_node] stats={self.stats}")

    def _print_banner(self):
        print(f"[model_node] cam_id={self.cam_id}")
        print(f"[model_node] device={self.device} amp={self.use_amp}")
        print(f"[model_node] clip_events={self.clip_connect} topic={self.clip_topic}")
        print(f"[model_node] scalars_events={self.scalars_connect} topic={self.scalars_topic}")
        print(f"[model_node] global_tracks={self.gt_connect} topic={self.gt_topic}")
        print(f"[model_node] scores_pub={self.scores_bind} topic={self.scores_topic}")
        print(f"[model_node] updates_stream={self.updates_stream} updates_target={self.args.updates_target or self.cam_id}")
        print(f"[model_node] base_scalar_dim={self.base_scalar_dim} scalar_dim={self.scalar_dim} identity_dim={self.identity_dim}")
        print(f"[model_node] model_version={self.args.model_version}")
        print(f"[model_node] identity_memory_fusion={self.args.enable_identity_memory_fusion} weight={self.args.identity_memory_weight}")
        print(f"[model_node] global_cache_ttl_s={self.args.global_cache_ttl_s} wait_for_global_id_s={self.args.wait_for_global_id_s} frame_tol={self.args.global_id_frame_tolerance}")
        print(f"[model_node] rollout_mode={self.rollout_mode} blend_alpha={self.blend_alpha}")
        print(f"[model_node] low_latency={self.args.low_latency} max_clip_drain={self.args.max_clip_drain} max_scalar_batch={self.args.max_scalar_batch}")
        print(f"[model_node] wait_for_scalars_s={self.args.wait_for_scalars_s}")
        print(f"[model_node] redis_update_interval_s={self.args.redis_update_interval_s}")
        print(f"[model_node] runtime_policy={self.runtime_policy}")
        print(f"[model_node] cloud_policy={self.cloud_policy}")
        print(f"[model_node] active_model_info={self.active_model_info}")
        if self.args.debug_schema:
            print(f"[model_node] scalar_schema_len={len(self.schema)}")
            print(f"[model_node] scalar_schema={self.schema}")

    def _setup_redis(self):
        if redis is not None and not self.args.disable_redis_updates:
            try:
                r_cfg = self.cfg.get("redis", {})
                self.rdb = redis.Redis(
                    host=r_cfg.get("host", "127.0.0.1"),
                    port=int(r_cfg.get("port", 6379)),
                    db=int(r_cfg.get("db", 0)),
                    password=r_cfg.get("password", None),
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                    health_check_interval=30,
                )
                self.rdb.ping()
                self.updates_stream = self.args.updates_stream or get_stream(self.cfg, "model_updates")
            except Exception as e:
                print(f"[model_node] Redis updates disabled: {e}")
                self.rdb = None
                self.updates_stream = None

    def _setup_zmq(self):
        clip_evt_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "clips_events")
        scal_evt_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "scalars_clip_events")
        gt_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "global_tracks")
        scores_cfg = get_zmq_endpoint(self.cfg, self.cam_id, "scores")

        self.clip_connect = self.args.clip_connect or local_connect_addr(clip_evt_cfg["bind"])
        self.clip_topic = self.args.clip_topic or clip_evt_cfg["topic"]
        self.scalars_connect = self.args.scalars_connect or local_connect_addr(scal_evt_cfg["bind"])
        self.scalars_topic = self.args.scalars_topic or scal_evt_cfg["topic"]
        self.gt_connect = self.args.global_tracks_connect or local_connect_addr(gt_cfg["bind"])
        self.gt_topic = self.args.global_tracks_topic or gt_cfg["topic"]
        self.scores_bind = self.args.scores_bind or scores_cfg["bind"]
        self.scores_topic = self.args.scores_topic or scores_cfg["topic"]

        self.clip_topic_b = self.clip_topic.encode("utf-8")
        self.scalars_topic_b = self.scalars_topic.encode("utf-8")
        self.gt_topic_b = self.gt_topic.encode("utf-8")
        self.scores_topic_b = self.scores_topic.encode("utf-8")

        self.ctx = zmq.Context.instance()

        self.sub_clip = make_sub_socket(self.ctx, self.clip_connect, self.clip_topic_b, self.args.zmq_rcvhwm, self.args.zmq_rcvbuf_mb)
        self.sub_scalars = make_sub_socket(self.ctx, self.scalars_connect, self.scalars_topic_b, self.args.zmq_rcvhwm, self.args.zmq_rcvbuf_mb)
        self.sub_global = make_sub_socket(self.ctx, self.gt_connect, self.gt_topic_b, self.args.zmq_rcvhwm, self.args.zmq_rcvbuf_mb)

        self.pub_scores = self.ctx.socket(zmq.PUB)
        self.pub_scores.setsockopt(zmq.LINGER, 0)
        self.pub_scores.setsockopt(zmq.SNDHWM, int(scores_cfg.get("sndhwm", 1000)))
        self.pub_scores.bind(self.scores_bind)

        self.poller = zmq.Poller()
        self.poller.register(self.sub_clip, zmq.POLLIN)
        self.poller.register(self.sub_scalars, zmq.POLLIN)
        self.poller.register(self.sub_global, zmq.POLLIN)

    def _setup_device_and_model_state(self):
        requested_cuda = str(self.args.device).startswith("cuda")
        if requested_cuda and not torch.cuda.is_available():
            print("[model_node] WARNING: CUDA requested but not available, falling back to CPU")

        self.device = torch.device(self.args.device if (requested_cuda and torch.cuda.is_available()) else "cpu")
        self.use_amp = bool(self.args.fp16 and self.device.type == "cuda")

        self.schema = default_scalar_schema()
        self.base_scalar_dim = len(self.schema) + 8
        self.identity_dim = len(identity_feature_names())
        self.scalar_dim = self.base_scalar_dim
        self.clip_channels_default = DEFAULT_CLIP_CHANNELS

        self.model = build_model(self.clip_channels_default, self.scalar_dim, self.device)
        self.model_norm_meta = {}

        self.active_model_info = {
            "model_version": str(self.args.model_version),
            "weights_path": None,
            "dataset_version": None,
            "candidate_status": "bootstrap",
            "test_quality": None,
            "loaded_from": "bootstrap_random_init",
            "task_type": self.runtime_policy["task_type"],
            "label_space": self.runtime_policy["label_space"],
            "domain_type": self.runtime_policy["domain_type"],
            "environment_family": self.runtime_policy["environment_family"],
            "site_scope": self.runtime_policy["site_scope"],
            "schema_version": self.runtime_policy["schema_version"],
            "identity_feature_version": self.runtime_policy["identity_feature_version"],
            "runtime_compat_version": self.runtime_policy["runtime_compat_version"],
            "rollout_state": "bootstrap",
        }

        self.shadow_model = None
        self.shadow_model_info = None
        self.shadow_model_norm_meta = {}
        self.rollout_mode = str(self.runtime_policy.get("rollout_mode", "active"))
        self.blend_alpha = clamp01(self.runtime_policy.get("blend_alpha", 0.35))
        self.allow_global_fallback = safe_bool(self.runtime_policy.get("allow_global_fallback", True), True)

        self.score_history_active = deque(maxlen=500)
        self.score_history_shadow = deque(maxlen=500)
        self.scalar_cache = {}
        self.scalar_fifo = deque(maxlen=self.args.scalar_cache_size)
        self.global_id_cache = {}
        self.global_id_fifo = deque(maxlen=10000)
        self.id_state = {}
        self.loop_n = 0
        self.last_clip_shape_seen = DEFAULT_CLIP_SHAPE

    def _initial_model_selection(self):
        startup_loaded = False
        if self.args.weights:
            ok, msg = self.try_load_model(self.args.weights, source="cli_weights", declared_version=self.args.model_version)
            print(f"[model_node] startup weights load ok={ok} msg={msg}")
            startup_loaded = bool(ok)

        if (not startup_loaded) and self.args.prefer_champion_registry:
            champ_path = Path(self.args.champion_registry_dir) / self.cam_id / "registry" / "champion.json"
            champ = load_champion_registry(champ_path)
            if champ and champ.get("weights_path"):
                ok, msg = self.try_load_model(
                    champ.get("weights_path"),
                    source="champion_registry",
                    declared_version=champ.get("model_version", None),
                )
                print(f"[model_node] champion registry load ok={ok} msg={msg}")
                startup_loaded = bool(ok)

        if not startup_loaded:
            print("[model_node] no startup weights loaded; running bootstrap random-init model")

    def cache_put(self, key, payload):
        now = now_mono()
        self.scalar_cache[key] = (now, payload)
        self.scalar_fifo.append((now, key))

    def cache_get(self, key):
        item = self.scalar_cache.get(key, None)
        if not item:
            return None
        t0, payload = item
        if (now_mono() - t0) > self.args.scalar_cache_ttl_s:
            self.scalar_cache.pop(key, None)
            return None
        return payload

    def wait_for_scalar_event(self, event_id: str):
        """
        Wait briefly for the matching scalars_clip_event.

        This prevents model_node from scoring a clip with empty/default scalar
        features when the clip event arrives slightly before the scalar event.
        """
        got = self.cache_get(event_id)
        if got is not None:
            return got

        timeout_s = float(getattr(self.args, "wait_for_scalars_s", 0.20))
        if timeout_s <= 0:
            return None

        t0 = now_mono()

        while (now_mono() - t0) < timeout_s:
            # Drain pending scalar events into scalar_cache.
            self._receive_scalar_batch()

            got = self.cache_get(event_id)
            if got is not None:
                return got

            time.sleep(0.005)

        return None

    def cache_prune(self):
        now = now_mono()
        while self.scalar_fifo:
            t0, k = self.scalar_fifo[0]
            item = self.scalar_cache.get(k, None)
            if item is None:
                self.scalar_fifo.popleft()
                continue
            current_t0, _payload = item
            if current_t0 != t0:
                self.scalar_fifo.popleft()
                continue
            expired = (now - t0) > self.args.scalar_cache_ttl_s
            too_large = len(self.scalar_cache) > self.args.scalar_cache_size
            if not expired and not too_large:
                break
            self.scalar_fifo.popleft()
            self.scalar_cache.pop(k, None)

    def gid_cache_put(self, event_id, global_person_id, alt_keys=None, cam=None, pid=None, fid=None):
        now = now_mono()
        keys = [event_id] if event_id else []
        if alt_keys:
            keys.extend(list(alt_keys))
        if cam is not None and pid is not None:
            keys.append(f"{cam}:{pid}:latest")

        for k in keys:
            if not k:
                continue
            self.global_id_cache[str(k)] = (now, int(global_person_id))
            self.global_id_fifo.append((now, str(k)))

        if cam is not None and pid is not None and fid is not None:
            frame_key = f"{cam}:{pid}:frames"
            old_item = self.global_id_cache.get(frame_key, None)
            frames = old_item[1] if old_item and isinstance(old_item[1], list) else []
            frames.append((int(fid), int(global_person_id), now))
            frames = frames[-30:]
            self.global_id_cache[frame_key] = (now, frames)

    def gid_cache_get(self, keys, cam=None, pid=None, fid=None):
        now = now_mono()

        for k in keys:
            if not k:
                continue
            item = self.global_id_cache.get(str(k), None)
            if not item:
                continue
            t0, gid = item
            if isinstance(gid, list):
                continue
            if (now - t0) > self.args.global_cache_ttl_s:
                self.global_id_cache.pop(str(k), None)
                continue
            return int(gid)

        if cam is not None and pid is not None and fid is not None:
            frame_key = f"{cam}:{pid}:frames"
            item = self.global_id_cache.get(frame_key, None)
            if item:
                _, frames = item
                best_gid = None
                best_gap = 10**9
                for f, gid, t0 in frames:
                    if (now - t0) > self.args.global_cache_ttl_s:
                        continue
                    gap = abs(int(fid) - int(f))
                    if gap < best_gap:
                        best_gap = gap
                        best_gid = gid
                if best_gid is not None and best_gap <= int(self.args.global_id_frame_tolerance):
                    return int(best_gid)

        if cam is not None and pid is not None:
            latest_key = f"{cam}:{pid}:latest"
            item = self.global_id_cache.get(latest_key, None)
            if item:
                t0, gid = item
                if not isinstance(gid, list) and (now - t0) <= self.args.global_cache_ttl_s:
                    return int(gid)
        return None

    def gid_cache_prune(self):
        now = now_mono()
        while self.global_id_fifo:
            t0, k = self.global_id_fifo[0]
            if (now - t0) <= self.args.global_cache_ttl_s:
                break
            self.global_id_fifo.popleft()
            item = self.global_id_cache.get(k, None)
            if item and not isinstance(item[1], list):
                self.global_id_cache.pop(k, None)

        dead_frame_keys = []
        for k, item in list(self.global_id_cache.items()):
            if not str(k).endswith(":frames"):
                continue
            _, frames = item
            if not isinstance(frames, list):
                dead_frame_keys.append(k)
                continue
            frames = [(f, gid, t0) for (f, gid, t0) in frames if (now - t0) <= self.args.global_cache_ttl_s]
            if not frames:
                dead_frame_keys.append(k)
            else:
                self.global_id_cache[k] = (now, frames[-30:])
        for k in dead_frame_keys:
            self.global_id_cache.pop(k, None)

    def check_model_metadata_compat(self, incoming_meta: dict):
        mm = merge_model_meta(incoming_meta, self.runtime_policy)

        if mm["task_type"] != str(self.runtime_policy["task_type"]):
            return False, f"task_type mismatch {mm['task_type']} != {self.runtime_policy['task_type']}", mm
        if mm["label_space"] != str(self.runtime_policy["label_space"]):
            return False, f"label_space mismatch {mm['label_space']} != {self.runtime_policy['label_space']}", mm
        if mm["domain_type"] != str(self.runtime_policy["domain_type"]):
            return False, f"domain_type mismatch {mm['domain_type']} != {self.runtime_policy['domain_type']}", mm
        if mm["schema_version"] != str(self.runtime_policy["schema_version"]):
            return False, f"schema_version mismatch {mm['schema_version']} != {self.runtime_policy['schema_version']}", mm
        if mm["identity_feature_version"] != str(self.runtime_policy["identity_feature_version"]):
            return False, f"identity_feature_version mismatch {mm['identity_feature_version']} != {self.runtime_policy['identity_feature_version']}", mm
        if mm["runtime_compat_version"] != str(self.runtime_policy["runtime_compat_version"]):
            return False, f"runtime_compat_version mismatch {mm['runtime_compat_version']} != {self.runtime_policy['runtime_compat_version']}", mm

        incoming_site = str(mm["site_scope"])
        incoming_env = str(mm["environment_family"])
        my_site = str(self.runtime_policy["site_scope"])
        my_env = str(self.runtime_policy["environment_family"])

        if incoming_site == my_site:
            return True, "metadata compatible: exact site", mm
        if incoming_env == my_env:
            return True, "metadata compatible: same environment family", mm
        if incoming_site == "global" and self.allow_global_fallback:
            return True, "metadata compatible: global fallback", mm

        return False, f"site/environment mismatch incoming_site={incoming_site} incoming_env={incoming_env} device_site={my_site} device_env={my_env}", mm

    def warmup_model(self, model, scalar_dim):
        if self.device.type != "cuda":
            return
        try:
            for _ in range(3):
                validate_model_forward(model, DEFAULT_CLIP_SHAPE, scalar_dim, self.device, self.use_amp)
            torch.cuda.synchronize()
            self.log_debug("[model_node] CUDA model warmup complete")
        except Exception as e:
            print(f"[model_node] WARNING model warmup failed: {e}")

    def try_load_model(self, weights_path: str, source: str, declared_version: str = None, as_shadow: bool = False):
        if not weights_path:
            return False, "missing weights_path"
        p = Path(weights_path)
        if not p.exists():
            return False, f"weights file not found: {weights_path}"

        try:
            sd, meta = load_checkpoint(str(p), self.device)
            if not isinstance(meta, dict):
                meta = {}

            ok_meta, meta_msg, merged_meta = self.check_model_metadata_compat(meta)
            if not ok_meta:
                return False, meta_msg

            ckpt_scalar_dim = safe_int(meta.get("scalar_dim", None), None)
            next_scalar_dim = int(ckpt_scalar_dim) if ckpt_scalar_dim is not None and ckpt_scalar_dim > 0 else int(self.scalar_dim)
            ckpt_clip_channels = safe_int(meta.get("clip_channels", None), self.clip_channels_default)
            if int(ckpt_clip_channels) != int(self.clip_channels_default):
                return False, f"incompatible clip_channels {ckpt_clip_channels} != {self.clip_channels_default}"

            new_model = build_model(self.clip_channels_default, next_scalar_dim, self.device)
            missing_keys, unexpected_keys = new_model.load_state_dict(sd, strict=False)
            if missing_keys or unexpected_keys:
                print(f"[model_node] WARNING checkpoint partial load missing_keys={missing_keys} unexpected_keys={unexpected_keys}")

            validate_model_forward(new_model, DEFAULT_CLIP_SHAPE, next_scalar_dim, self.device, self.use_amp)
            self.warmup_model(new_model, next_scalar_dim)

            resolved_version = declared_version or meta.get("model_version") or f"loaded_{int(now_wall())}"
            info = {
                "model_version": str(resolved_version),
                "weights_path": str(p),
                "dataset_version": meta.get("dataset_version", None),
                "candidate_status": meta.get("candidate_status", "accepted"),
                "test_quality": meta.get("test_eval", {}).get("fused_quality", None) if isinstance(meta.get("test_eval", None), dict) else meta.get("test_quality", None),
                "loaded_from": source,
                "task_type": merged_meta["task_type"],
                "label_space": merged_meta["label_space"],
                "domain_type": merged_meta["domain_type"],
                "environment_family": merged_meta["environment_family"],
                "site_scope": merged_meta["site_scope"],
                "schema_version": merged_meta["schema_version"],
                "identity_feature_version": merged_meta["identity_feature_version"],
                "runtime_compat_version": merged_meta["runtime_compat_version"],
                "rollout_state": "shadow" if as_shadow else "active",
            }

            if as_shadow:
                self.shadow_model = new_model
                self.shadow_model_info = info
                self.shadow_model_norm_meta = meta.get("scalar_norm", {}) if isinstance(meta, dict) else {}
                return True, f"loaded SHADOW model_version={resolved_version} scalar_dim={next_scalar_dim} from={source}"

            self.model = new_model
            self.scalar_dim = int(next_scalar_dim)
            self.model_norm_meta = meta.get("scalar_norm", {}) if isinstance(meta, dict) else {}
            self.active_model_info = info
            self.args.model_version = str(resolved_version)
            return True, f"loaded ACTIVE model_version={resolved_version} scalar_dim={self.scalar_dim} from={source}"

        except Exception as e:
            return False, f"failed to load weights {weights_path}: {e}"

    def should_accept_update(self, update_obj: dict):
        if not self.args.allow_runtime_model_selection:
            return True, "runtime selection disabled -> accept"

        incoming_status = str(update_obj.get("candidate_status", "accepted"))
        incoming_version = str(update_obj.get("model_version", "unknown"))
        current_version = str(self.active_model_info.get("model_version", "unknown"))

        if incoming_status not in ("accepted", "champion", "promoted"):
            return False, f"skip candidate_status={incoming_status}"
        if incoming_version == current_version:
            return False, "same model_version already active"

        incoming_site = str(update_obj.get("site_scope", "global"))
        incoming_env = str(update_obj.get("environment_family", "generic"))
        my_site = str(self.runtime_policy["site_scope"])
        my_env = str(self.runtime_policy["environment_family"])

        if incoming_site == my_site:
            return True, "exact site match"
        if incoming_env == my_env:
            return True, "environment family match"
        if incoming_site == "global" and self.allow_global_fallback:
            return True, "global fallback accepted"

        return False, f"scope mismatch incoming_site={incoming_site} incoming_env={incoming_env} my_site={my_site} my_env={my_env}"

    def try_apply_update(self, update_fields: dict, last_clip_shape=DEFAULT_CLIP_SHAPE):
        target = update_fields.get("target", None)
        accept_target = self.args.updates_target or self.cam_id
        if target not in (accept_target, "all", None):
            return False, f"ignored update for target={target}"

        accept, why = self.should_accept_update(update_fields)
        if not accept:
            return False, why

        weights_path = update_fields.get("weights_path") or update_fields.get("weights")
        if not weights_path:
            return False, "update missing weights_path"

        rollout_state = str(update_fields.get("rollout_state", self.rollout_mode))
        if rollout_state not in ("active", "shadow", "blended"):
            rollout_state = "active"

        if rollout_state == "shadow":
            ok, msg = self.try_load_model(str(weights_path), "runtime_model_update", update_fields.get("model_version", None), as_shadow=True)
            return ok, f"shadow update: {msg}"

        ok, msg = self.try_load_model(str(weights_path), "runtime_model_update", update_fields.get("model_version", None), as_shadow=False)
        if not ok:
            return False, msg

        validate_model_forward(self.model, last_clip_shape, self.scalar_dim, self.device, self.use_amp)
        if update_fields.get("dataset_version") is not None:
            self.active_model_info["dataset_version"] = update_fields.get("dataset_version")
        if update_fields.get("candidate_status") is not None:
            self.active_model_info["candidate_status"] = update_fields.get("candidate_status")
        self.active_model_info["rollout_state"] = rollout_state
        return True, f"applied update: {msg} rollout={rollout_state}"

    def pull_global_track_updates(self):
        drained = 0
        while drained < self.args.max_drain_per_step:
            try:
                parts = self.sub_global.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                print(f"[model_node] global_tracks ZMQ error: {e}")
                break

            drained += 1
            obj = parse_zmq_json_payload(parts)
            if obj is None:
                continue

            self.stats["global_updates_received"] += 1
            event_id = obj.get("event_id", None)
            cam = obj.get("cam_id", None)
            pid = safe_int(obj.get("person_track_id", None), None)
            fid = safe_int(obj.get("frame_id", None), None)
            fid_end = safe_int(obj.get("frame_id_end", None), None)
            gid = safe_int(obj.get("global_person_id", None), None)
            if gid is None:
                continue

            alt_keys = []
            if cam is not None and pid is not None and fid is not None:
                alt_keys.append(f"{cam}:{pid}:{fid}")
            if cam is not None and pid is not None and fid_end is not None:
                alt_keys.append(f"{cam}:{pid}:{fid_end}")

            self.gid_cache_put(event_id, gid, alt_keys=alt_keys, cam=cam, pid=pid, fid=fid if fid is not None else fid_end)

        self.gid_cache_prune()

    def pull_runtime_updates(self):
        if self.rdb is None or self.updates_stream is None:
            return
        if self.args.check_updates_every_loops <= 0:
            return
        if (self.loop_n % self.args.check_updates_every_loops) != 0:
            return

        if self.args.redis_update_interval_s > 0:
            now = now_mono()
            if (now - self.last_redis_update_check_s) < self.args.redis_update_interval_s:
                return
            self.last_redis_update_check_s = now

        try:
            upd_streams = self.rdb.xread({self.updates_stream: self.last_upd_id}, block=1, count=5)
        except Exception as e:
            print(f"[model_node] model_updates Redis error: {e}")
            return

        if upd_streams:
            for mid, fields in parse_xread_messages(upd_streams):
                self.last_upd_id = mid
                update_obj = {}
                self.stats["model_updates_seen"] += 1

                if b"json" in fields:
                    try:
                        update_obj = json.loads(b2s(fields[b"json"]))
                    except Exception:
                        update_obj = {}
                else:
                    for k, v in fields.items():
                        update_obj[b2s(k)] = b2s(v)

                ok, msg = self.try_apply_update(update_obj, last_clip_shape=self.last_clip_shape_seen)
                if ok:
                    self.stats["model_updates_applied"] += 1
                else:
                    self.stats["model_updates_rejected"] += 1
                print(f"[model_node] model_update mid={mid} ok={ok} msg={msg}")

    def _load_clip_from_path(self, clip_path: str, clip_codec: str = None):
        if not clip_path:
            return None, "missing clip_path"
        p = Path(clip_path)
        if not p.exists():
            return None, f"clip file not found: {clip_path}"

        codec = (clip_codec or p.suffix.lstrip(".") or "").lower()
        try:
            if codec == "npz":
                with np.load(str(p), allow_pickle=False) as data:
                    if "clip" not in data:
                        return None, f"npz missing clip key: {clip_path}"
                    clip_np = data["clip"]
            elif codec == "npy":
                clip_np = np.load(str(p), allow_pickle=False)
            else:
                return None, f"unsupported clip codec: {codec}"

            if not isinstance(clip_np, np.ndarray):
                return None, "clip is not ndarray"
            if clip_np.dtype != np.float32:
                clip_np = clip_np.astype(np.float32, copy=False)
            if clip_np.ndim != 4:
                return None, f"clip ndim must be 4, got {clip_np.ndim}"
            return np.ascontiguousarray(clip_np), None
        except Exception as e:
            return None, f"failed loading clip: {e}"

    def _handle_scalars_event(self, payload):
        meta = payload.get("meta", {})
        feats = payload.get("features", {})
        policy_features = payload.get("policy_features", {})
        cam = meta.get("cam_id", self.cam_id)
        pid = safe_int(meta.get("person_track_id", -1), -1)
        fid_end = safe_int(meta.get("frame_id_end", -1), -1)
        event_id = meta.get("event_id") or f"{cam}:{pid}:{fid_end}:{meta.get('stamp_ns_end', 0)}"

        self.cache_put(event_id, {
            "event_id": event_id,
            "meta": meta,
            "features": feats,
            "policy_features": policy_features,
        })

    def _publish_score(self, out: dict):
        header = {
            "type": "score",
            "event_id": out.get("event_id"),
            "cam_id": out.get("cam_id"),
            "person_track_id": out.get("person_track_id"),
            "global_person_id": out.get("global_person_id"),
            "frame_id": out.get("frame_id_end"),
            "stamp_ns": out.get("stamp_ns_end"),
        }
        self.pub_scores.send_multipart([
            self.scores_topic_b,
            self._jcompact(header).encode("utf-8"),
            self._jcompact(out).encode("utf-8"),
        ])

    def forward_scores_with_model(self, model, norm_meta, clip_np, vec_np):
        vec_use = normalize_scalar_vector(vec_np, norm_meta)
        if vec_use.dtype != np.float32:
            vec_use = vec_use.astype(np.float32, copy=False)
        if clip_np.dtype != np.float32:
            clip_np = clip_np.astype(np.float32, copy=False)

        clip_t = torch.from_numpy(np.ascontiguousarray(clip_np)).unsqueeze(0).to(self.device, non_blocking=True)
        scalars_t = torch.from_numpy(np.ascontiguousarray(vec_use)).unsqueeze(0).to(self.device, non_blocking=True)

        sync_timing = bool(getattr(self.args, "debug_timing", False))
        maybe_cuda_synchronize(self.device, sync_timing)

        with torch.inference_mode():
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with amp_ctx:
                out_model = model(clip_t, scalars_t)

        maybe_cuda_synchronize(self.device, sync_timing)

        raw_fused = sigmoid_item(out_model["fused_logits"])
        score_cnn = sigmoid_item(out_model["cnn_logits"])
        score_mlp = sigmoid_item(out_model["mlp_logits"])
        return raw_fused, score_cnn, score_mlp

    def apply_rollout_restraint(self, active_score, shadow_score=None):
        rollout_state = str(self.active_model_info.get("rollout_state", self.rollout_mode))
        if rollout_state == "active" or shadow_score is None:
            return clamp01(active_score), "active"
        if rollout_state == "blended":
            a = clamp01(self.blend_alpha)
            fused = (1.0 - a) * float(active_score) + a * float(shadow_score)
            return clamp01(fused), "blended"
        if rollout_state == "shadow":
            return clamp01(active_score), "shadow_active_kept"
        return clamp01(active_score), "active"

    def _lookup_global_person_id_for_clip(self, event_id, cam, pid, fid_end):
        gid_keys = [event_id, f"{cam}:{pid}:{fid_end}"]
        global_person_id = self.gid_cache_get(gid_keys, cam=cam, pid=pid, fid=fid_end)
        if global_person_id is not None:
            return global_person_id
        if self.args.wait_for_global_id_s <= 0:
            return None

        t_wait0 = now_mono()
        while (now_mono() - t_wait0) < float(self.args.wait_for_global_id_s):
            self.pull_global_track_updates()
            global_person_id = self.gid_cache_get(gid_keys, cam=cam, pid=pid, fid=fid_end)
            if global_person_id is not None:
                return global_person_id
            time.sleep(0.01)
        return None

    def _parse_clip_event_meta(self, payload):
        meta = payload.get("meta", payload)
        cam = meta.get("cam_id", self.cam_id)
        pid = safe_int(meta.get("person_track_id", -1), -1)
        fid_end = safe_int(meta.get("frame_id_end", -1), -1)
        stamp_ns_end = safe_int(meta.get("stamp_ns_end", 0), 0)
        event_id = meta.get("event_id") or f"{cam}:{pid}:{fid_end}:{stamp_ns_end}"

        c = safe_int(meta.get("C", DEFAULT_CLIP_CHANNELS), DEFAULT_CLIP_CHANNELS)
        t = safe_int(meta.get("T", DEFAULT_CLIP_T), DEFAULT_CLIP_T)
        h = safe_int(meta.get("H", DEFAULT_CLIP_H), DEFAULT_CLIP_H)
        w = safe_int(meta.get("W", DEFAULT_CLIP_W), DEFAULT_CLIP_W)
        clip_path = payload.get("clip_path", None) or meta.get("clip_path", None)
        clip_codec = payload.get("clip_codec", None) or meta.get("clip_codec", None)

        return {
            "meta": meta,
            "cam": cam,
            "pid": pid,
            "fid_end": fid_end,
            "stamp_ns_end": stamp_ns_end,
            "event_id": event_id,
            "shape": (c, t, h, w),
            "clip_path": clip_path,
            "clip_codec": clip_codec,
        }

    def _prepare_scalar_vector(self, agg_feats, id_feat_vec):
        vec, miss_cnt = build_scalar_vector(agg_feats, self.schema, fill_value=0.0)
        vec = append_special_scalars(vec, agg_feats)

        if self.scalar_dim > self.base_scalar_dim:
            extra_needed = self.scalar_dim - self.base_scalar_dim
            if extra_needed <= len(id_feat_vec):
                id_append = id_feat_vec[:extra_needed]
            else:
                pad = np.zeros((extra_needed - len(id_feat_vec),), dtype=np.float32)
                id_append = np.concatenate([id_feat_vec, pad], axis=0)
            vec = np.concatenate([vec, id_append], axis=0)
        elif self.scalar_dim < self.base_scalar_dim:
            vec = vec[:self.scalar_dim]

        return vec.astype(np.float32), miss_cnt

    def _handle_clip_event(self, payload, dropped_before=0):
        self.stats["clip_events_received"] += 1
        if dropped_before > 0:
            self.stats["clip_events_dropped_low_latency"] += int(dropped_before)

        timing_enabled = bool(getattr(self.args, "debug_timing", False))
        t_total0 = time.perf_counter()
        t_clip_load = 0.0
        t_gid = 0.0
        t_infer = 0.0
        t_shadow = 0.0

        parsed = self._parse_clip_event_meta(payload)
        meta = parsed["meta"]
        cam = parsed["cam"]
        pid = parsed["pid"]
        fid_end = parsed["fid_end"]
        stamp_ns_end = parsed["stamp_ns_end"]
        event_id = parsed["event_id"]
        c, t, h, w = parsed["shape"]
        clip_path = parsed["clip_path"]
        clip_codec = parsed["clip_codec"]
        self.last_clip_shape_seen = (c, t, h, w)

        if c != self.clip_channels_default:
            self.stats["clip_events_skipped"] += 1
            return

        t0 = time.perf_counter()
        clip_np, err = self._load_clip_from_path(clip_path, clip_codec=clip_codec)
        t_clip_load = time.perf_counter() - t0
        if clip_np is None:
            self.stats["clip_events_skipped"] += 1
            print(f"[model_node] skip clip event_id={event_id} reason={err}")
            return

        if clip_np.shape != (c, t, h, w):
            self.stats["clip_events_skipped"] += 1
            print(f"[model_node] skip clip event_id={event_id} bad_shape={clip_np.shape} expected={(c, t, h, w)}")
            return

        got = self.wait_for_scalar_event(event_id)

        if got is None:
            self.stats["scalar_missing"] += 1
            self.stats["clip_events_skipped"] += 1
            print(f"[model_node] skip event_id={event_id} reason=missing_scalars")
            return

        scalar_missing = False
        agg_feats = got.get("features", {}) or {}
        policy_features = extract_policy_features(got, agg_feats)

        t0 = time.perf_counter()
        global_person_id = self._lookup_global_person_id_for_clip(event_id, cam, pid, fid_end)
        t_gid = time.perf_counter() - t0

        event_time_s = (stamp_ns_end * 1e-9) if stamp_ns_end > 0 else now_wall()
        id_feat_vec, id_feat_dict = build_identity_features(
            global_person_id=global_person_id,
            id_state=self.id_state,
            now_s=event_time_s,
            recent_window_s=self.args.identity_recent_window_s,
        )

        vec_base, miss_cnt = self._prepare_scalar_vector(agg_feats, id_feat_vec)

        t0 = time.perf_counter()
        raw_score_fused, score_cnn, score_mlp = self.forward_scores_with_model(self.model, self.model_norm_meta, clip_np, vec_base)
        t_infer = time.perf_counter() - t0

        shadow_raw_score_fused = None
        shadow_score_cnn = None
        shadow_score_mlp = None
        if self.shadow_model is not None:
            try:
                t0 = time.perf_counter()
                shadow_raw_score_fused, shadow_score_cnn, shadow_score_mlp = self.forward_scores_with_model(
                    self.shadow_model,
                    self.shadow_model_norm_meta,
                    clip_np,
                    vec_base,
                )
                t_shadow = time.perf_counter() - t0
            except Exception as e:
                print(f"[model_node] shadow scoring failed: {e}")
                shadow_raw_score_fused = None
                shadow_score_cnn = None
                shadow_score_mlp = None

        rollout_score_raw, rollout_mode_applied = self.apply_rollout_restraint(active_score=raw_score_fused, shadow_score=shadow_raw_score_fused)
        identity_memory_score = compute_identity_memory_score(global_person_id, self.id_state)
        t_score_ns = time.time_ns()

        if self.args.enable_identity_memory_fusion and global_person_id is not None:
            w_mem = clamp01(self.args.identity_memory_weight)
            w_raw = 1.0 - w_mem
            score_fused = clamp01(w_raw * rollout_score_raw + w_mem * identity_memory_score)
        else:
            score_fused = rollout_score_raw

        update_identity_state(self.id_state, global_person_id, cam, event_time_s, score_fused, raw_score_fused)
        prune_identity_state(self.id_state, now_wall(), self.args.identity_state_idle_s)

        self.score_history_active.append(float(raw_score_fused))
        if shadow_raw_score_fused is not None:
            self.score_history_shadow.append(float(shadow_raw_score_fused))

        out = {
            "type": "score",
            "event_id": event_id,
            "cam_id": cam,
            "person_track_id": pid,
            "global_person_id": int(global_person_id) if global_person_id is not None else None,
            "identity_enriched": bool(global_person_id is not None),
            "frame_id_end": fid_end,
            "stamp_ns_end": stamp_ns_end,
            "t_capture_ns": int(meta.get("t_capture_ns", stamp_ns_end)),
            "t_reid_ns": int(meta.get("t_reid_ns", 0) or 0),
            "t_identity_ns": int(meta.get("t_identity_ns", 0) or 0),
            "t_score_ns": int(t_score_ns),

            # Clip reference info carried forward for policy_node -> alert -> clip_writer.
            "clip_path": clip_path,
            "clip_codec": clip_codec,
            "local_clip_path": clip_path,

            "score": float(score_fused),
            "score_fused": float(score_fused),
            "score_fused_raw": float(raw_score_fused),
            "score_cnn": float(score_cnn),
            "score_mlp": float(score_mlp),
            "score_identity_memory": float(identity_memory_score),
            "model_version": self.active_model_info.get("model_version", self.args.model_version),
            "model_loaded_from": self.active_model_info.get("loaded_from"),
            "model_dataset_version": self.active_model_info.get("dataset_version"),
            "model_candidate_status": self.active_model_info.get("candidate_status"),
            "task_type": self.active_model_info.get("task_type"),
            "label_space": self.active_model_info.get("label_space"),
            "domain_type": self.active_model_info.get("domain_type"),
            "environment_family": self.active_model_info.get("environment_family"),
            "site_scope": self.active_model_info.get("site_scope"),
            "schema_version": self.active_model_info.get("schema_version"),
            "identity_feature_version": self.active_model_info.get("identity_feature_version"),
            "runtime_compat_version": self.active_model_info.get("runtime_compat_version"),
            "rollout_state": self.active_model_info.get("rollout_state", self.rollout_mode),
            "rollout_mode_applied": rollout_mode_applied,
            "shadow_model_version": self.shadow_model_info.get("model_version") if self.shadow_model_info else None,
            "shadow_score_fused_raw": float(shadow_raw_score_fused) if shadow_raw_score_fused is not None else None,
            "shadow_score_cnn": float(shadow_score_cnn) if shadow_score_cnn is not None else None,
            "shadow_score_mlp": float(shadow_score_mlp) if shadow_score_mlp is not None else None,
            "scalar_missing": bool(scalar_missing),
            "scalar_missing_fields": int(miss_cnt),
            "object_track_id": safe_int(meta.get("object_track_id", -1), -1),
            "object_class_id": safe_int(meta.get("object_class_id", -1), -1),
            "missing_pose_ratio": safe_float(meta.get("missing_pose_ratio", 0.0), 0.0),
            "missing_obj_ratio": safe_float(meta.get("missing_obj_ratio", 0.0), 0.0),
            "policy_features": policy_features,
            "identity_features": id_feat_dict,
            "identity_feature_names": identity_feature_names(),
            "identity_memory_fusion_enabled": bool(self.args.enable_identity_memory_fusion),
            "identity_memory_weight": float(self.args.identity_memory_weight),
        }

        self._publish_score(out)
        self.stats["clip_events_scored"] += 1

        s_val = safe_float(policy_features.get("heuristic_theft_score", None), None)
        s_txt = f" heuristic={s_val:.3f}" if s_val is not None else ""
        gid_txt = f" gid={global_person_id}" if global_person_id is not None else " gid=None"
        shadow_txt = f" shadow_raw={shadow_raw_score_fused:.3f}" if shadow_raw_score_fused is not None else ""
        drop_txt = f" dropped_old={dropped_before}" if dropped_before > 0 else ""

        print(
            f"[model_node] cam={cam} pid={pid}{gid_txt} fid_end={fid_end} "
            f"fused={score_fused:.3f} raw={raw_score_fused:.3f} "
            f"cnn={score_cnn:.3f} mlp={score_mlp:.3f} mem={identity_memory_score:.3f} "
            f"ver={self.active_model_info.get('model_version', self.args.model_version)} "
            f"rollout={rollout_mode_applied}"
            f"{shadow_txt}{s_txt}{drop_txt}"
        )

        if timing_enabled:
            t_total = time.perf_counter() - t_total0
            capture_ns = safe_int(out.get("t_capture_ns", 0), 0)
            capture_to_score_ms = ((t_score_ns - capture_ns) / 1e6) if capture_ns > 0 else None
            capture_txt = f" capture_to_score_ms={capture_to_score_ms:.1f}" if capture_to_score_ms is not None else ""
            print(
                f"[model_node] timing event_id={event_id} "
                f"clip_load_ms={t_clip_load * 1000:.1f} "
                f"gid_wait_ms={t_gid * 1000:.1f} "
                f"infer_ms={t_infer * 1000:.1f} "
                f"shadow_ms={t_shadow * 1000:.1f} "
                f"total_ms={t_total * 1000:.1f}"
                f"{capture_txt}"
            )

    def _receive_one_scalar_event(self):
        try:
            parts = self.sub_scalars.recv_multipart()
            payload = parse_zmq_json_payload(parts)
            if payload is not None:
                self.stats["scalar_events_received"] += 1
                self._handle_scalars_event(payload)
        except Exception as e:
            print(f"[model_node] scalars event error: {e}")

    def _receive_scalar_batch(self):
        processed = 0
        while processed < max(1, int(self.args.max_scalar_batch)):
            try:
                parts = self.sub_scalars.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                print(f"[model_node] scalars batch error: {e}")
                break

            payload = parse_zmq_json_payload(parts)
            if payload is not None:
                self.stats["scalar_events_received"] += 1
                self._handle_scalars_event(payload)
            processed += 1

    def _receive_one_clip_event(self):
        try:
            parts = self.sub_clip.recv_multipart()
            payload = parse_zmq_json_payload(parts)
            if payload is not None:
                self._handle_clip_event(payload)
        except Exception as e:
            print(f"[model_node] clip event error: {e}")

    def _receive_latest_clip_event(self):
        latest_payload = None
        received = 0
        max_drain = max(1, int(self.args.max_clip_drain))

        while received < max_drain:
            try:
                parts = self.sub_clip.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                print(f"[model_node] clip drain error: {e}")
                break

            payload = parse_zmq_json_payload(parts)
            if payload is not None:
                latest_payload = payload
                received += 1

        if latest_payload is not None:
            dropped_before = max(0, received - 1)
            self._handle_clip_event(latest_payload, dropped_before=dropped_before)

    def pull_feature_events(self):
        events = dict(self.poller.poll(timeout=self.args.block_ms))

        if self.sub_scalars in events and (events[self.sub_scalars] & zmq.POLLIN):
            if self.args.low_latency:
                self._receive_scalar_batch()
            else:
                self._receive_one_scalar_event()

        if self.sub_clip in events and (events[self.sub_clip] & zmq.POLLIN):
            if self.args.low_latency:
                self._receive_latest_clip_event()
            else:
                self._receive_one_clip_event()

        self.cache_prune()

    def step(self):
        self.loop_n += 1
        self.pull_global_track_updates()
        self.pull_runtime_updates()
        self.pull_feature_events()
        self.maybe_print_stats()

    def close(self):
        for sock in [self.sub_clip, self.sub_scalars, self.sub_global, self.pub_scores]:
            try:
                sock.close(0)
            except Exception:
                pass

    def run_forever(self, sleep_s=0.001):
        try:
            while True:
                self.step()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\n[model_node] stopping...")
        finally:
            self.close()


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--updates_stream", default=None)
    ap.add_argument("--updates_target", default=None)
    ap.add_argument("--check_updates_every_loops", type=int, default=1)
    ap.add_argument("--disable_redis_updates", action="store_true")

    ap.add_argument("--clip_connect", default=None)
    ap.add_argument("--clip_topic", default=None)
    ap.add_argument("--scalars_connect", default=None)
    ap.add_argument("--scalars_topic", default=None)
    ap.add_argument("--global_tracks_connect", default=None)
    ap.add_argument("--global_tracks_topic", default=None)

    ap.add_argument("--scores_bind", default=None)
    ap.add_argument("--scores_topic", default=None)

    ap.add_argument("--zmq_rcvhwm", type=int, default=256)
    ap.add_argument("--zmq_rcvbuf_mb", type=float, default=0.0)
    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--max_drain_per_step", type=int, default=64)

    ap.add_argument("--scalar_cache_size", type=int, default=5000)
    ap.add_argument("--scalar_cache_ttl_s", type=float, default=5.0)
    ap.add_argument("--wait_for_scalars_s", type=float, default=0.20)

    ap.add_argument("--weights", default=None)
    ap.add_argument("--model_version", default="bootstrap_v0")

    ap.add_argument("--prefer_champion_registry", action="store_true")
    ap.add_argument("--champion_registry_dir", default="models")
    ap.add_argument("--allow_runtime_model_selection", action="store_true")

    ap.add_argument("--global_cache_ttl_s", type=float, default=60.0)
    ap.add_argument("--wait_for_global_id_s", type=float, default=1.0)
    ap.add_argument("--global_id_frame_tolerance", type=int, default=12)

    ap.add_argument("--identity_recent_window_s", type=float, default=30.0)
    ap.add_argument("--identity_state_idle_s", type=float, default=60.0)
    ap.add_argument("--enable_identity_memory_fusion", action="store_true")
    ap.add_argument("--identity_memory_weight", type=float, default=0.20)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_timing", action="store_true")
    ap.add_argument("--debug_schema", action="store_true")
    ap.add_argument("--stats_every_s", type=float, default=0.0)

    # Phase 2 performance options.
    ap.add_argument("--low_latency", action="store_true")
    ap.add_argument("--max_clip_drain", type=int, default=32)
    ap.add_argument("--max_scalar_batch", type=int, default=64)
    ap.add_argument("--redis_update_interval_s", type=float, default=0.0)

    return ap


def main():
    args = build_arg_parser().parse_args()
    worker = ModelWorker(args)
    worker.run_forever()


if __name__ == "__main__":
    main()
