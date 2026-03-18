#!/usr/bin/env python3
"""
model_node.py (split-output CNN + MLP fusion + hot model updates + identity-aware scoring)

Publishes:
  - score_fused
  - score_cnn
  - score_mlp

Backward compatibility:
  - score == score_fused

Identity support added:
  1) Metadata propagation:
     - global_person_id is attached when available
  2) Identity feature augmentation:
     - extra identity features are appended only if model scalar_dim expects them
  3) Identity memory:
     - lightweight cross-camera memory prior can be fused into final score
"""

import argparse
import json
import time
from collections import deque, defaultdict

import redis
import numpy as np
import torch
import torch.nn as nn

from config_utils import load_cfg, get_stream


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


def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


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


# ============================================================
# Model
# ============================================================

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


# ============================================================
# Scalar helpers
# ============================================================

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


# ============================================================
# Identity helpers
# ============================================================

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


# ============================================================
# Checkpoint / model loading
# ============================================================

def load_checkpoint(weights_path: str, device: torch.device):
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
    with torch.no_grad():
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


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--clips_stream", default=None)
    ap.add_argument("--scalars_clip_stream", default=None)
    ap.add_argument("--scores_stream", default=None)

    ap.add_argument("--updates_stream", default=None)
    ap.add_argument("--updates_target", default=None)
    ap.add_argument("--check_updates_every_loops", type=int, default=1)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=8)

    ap.add_argument("--scalar_cache_size", type=int, default=5000)
    ap.add_argument("--scalar_cache_ttl_s", type=float, default=5.0)

    ap.add_argument("--weights", default=None)
    ap.add_argument("--model_version", default="v0")

    # identity support
    ap.add_argument("--global_tracks_stream", default=None)
    ap.add_argument("--global_cache_ttl_s", type=float, default=30.0)
    ap.add_argument("--identity_recent_window_s", type=float, default=30.0)
    ap.add_argument("--identity_state_idle_s", type=float, default=60.0)

    # identity memory fusion
    ap.add_argument("--enable_identity_memory_fusion", action="store_true")
    ap.add_argument("--identity_memory_weight", type=float, default=0.20)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
    )
    rdb.ping()

    clips_stream = args.clips_stream or get_stream(cfg, "clips", cam_id)
    scalars_clip_stream = args.scalars_clip_stream or get_stream(cfg, "scalars_clip", cam_id)
    scores_stream = args.scores_stream or get_stream(cfg, "scores", cam_id)
    updates_stream = args.updates_stream or get_stream(cfg, "model_updates")

    try:
        global_tracks_stream = args.global_tracks_stream or get_stream(cfg, "global_tracks")
    except Exception:
        global_tracks_stream = "global_tracks"

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    schema = default_scalar_schema()
    base_scalar_dim = len(schema) + 8
    identity_dim = len(identity_feature_names())
    scalar_dim = base_scalar_dim
    clip_channels_default = 19

    model = build_model(clip_channels_default, scalar_dim, device)
    model_norm_meta = {}

    if args.weights:
        sd, meta = load_checkpoint(args.weights, device)

        ckpt_scalar_dim = safe_int(meta.get("scalar_dim", None), None)
        if ckpt_scalar_dim is not None and ckpt_scalar_dim > 0:
            scalar_dim = int(ckpt_scalar_dim)
            model = build_model(clip_channels_default, scalar_dim, device)

        model.load_state_dict(sd, strict=False)
        args.model_version = str(meta.get("model_version", args.model_version))
        model_norm_meta = meta.get("scalar_norm", {})

    print(f"[model_node] cam_id={cam_id}")
    print(f"[model_node] device={device} amp={use_amp}")
    print(f"[model_node] clips={clips_stream} scalars_clip={scalars_clip_stream} scores={scores_stream}")
    print(f"[model_node] updates_stream={updates_stream} updates_target={args.updates_target or cam_id}")
    print(f"[model_node] global_tracks_stream={global_tracks_stream}")
    print(f"[model_node] base_scalar_dim={base_scalar_dim} scalar_dim={scalar_dim} identity_dim={identity_dim}")
    print(f"[model_node] model_version={args.model_version}")
    print(f"[model_node] identity_memory_fusion={args.enable_identity_memory_fusion} weight={args.identity_memory_weight}")

    last_clip_id = "0-0"
    last_scal_id = "0-0"
    last_upd_id = "0-0"
    last_gid_id = "0-0"

    scalar_cache = {}
    scalar_fifo = deque(maxlen=args.scalar_cache_size)

    global_id_cache = {}
    global_id_fifo = deque(maxlen=10000)

    # global identity memory
    id_state = {}

    def cache_put(key, payload):
        now = time.time()
        scalar_cache[key] = (now, payload)
        scalar_fifo.append((now, key))

    def cache_get(key):
        item = scalar_cache.get(key, None)
        if not item:
            return None
        t0, payload = item
        if (time.time() - t0) > args.scalar_cache_ttl_s:
            scalar_cache.pop(key, None)
            return None
        return payload

    def cache_prune():
        now = time.time()
        while scalar_fifo:
            t0, k = scalar_fifo[0]
            if (now - t0) <= args.scalar_cache_ttl_s and len(scalar_cache) <= args.scalar_cache_size:
                break
            scalar_fifo.popleft()
            scalar_cache.pop(k, None)

    def gid_cache_put(event_id, global_person_id, alt_keys=None):
        now = time.time()
        keys = [event_id] if event_id else []
        if alt_keys:
            keys.extend(list(alt_keys))

        for k in keys:
            if not k:
                continue
            global_id_cache[str(k)] = (now, int(global_person_id))
            global_id_fifo.append((now, str(k)))

    def gid_cache_get(keys):
        now = time.time()
        for k in keys:
            if not k:
                continue
            item = global_id_cache.get(str(k), None)
            if not item:
                continue
            t0, gid = item
            if (now - t0) > args.global_cache_ttl_s:
                global_id_cache.pop(str(k), None)
                continue
            return int(gid)
        return None

    def gid_cache_prune():
        now = time.time()
        while global_id_fifo:
            t0, k = global_id_fifo[0]
            if (now - t0) <= args.global_cache_ttl_s:
                break
            global_id_fifo.popleft()
            global_id_cache.pop(k, None)

    def pull_global_track_updates():
        nonlocal last_gid_id

        streams = rdb.xread({global_tracks_stream: last_gid_id}, block=1, count=100)
        if not streams:
            return

        for mid, fields in parse_xread_messages(streams):
            last_gid_id = mid

            obj = {}
            if b"json" in fields:
                try:
                    obj = json.loads(b2s(fields[b"json"]))
                except Exception:
                    obj = {}
            else:
                for k, v in fields.items():
                    obj[b2s(k)] = b2s(v)

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

            gid_cache_put(event_id, gid, alt_keys=alt_keys)

        gid_cache_prune()

    def try_apply_update(update_fields: dict, last_clip_shape=(19, 16, 112, 112)):
        nonlocal model
        nonlocal clip_channels_default
        nonlocal scalar_dim
        nonlocal model_norm_meta

        weights_path = update_fields.get("weights_path") or update_fields.get("weights")
        if not weights_path:
            return False, "update missing weights_path"

        target = update_fields.get("target", None)
        accept_target = args.updates_target or cam_id
        if target not in (accept_target, "all", None):
            return False, f"ignored update for target={target}"

        upd_clip_channels = update_fields.get("clip_channels", None)
        upd_scalar_dim = update_fields.get("scalar_dim", None)

        if upd_clip_channels is not None:
            try:
                upd_clip_channels = int(upd_clip_channels)
            except Exception:
                upd_clip_channels = None

        if upd_scalar_dim is not None:
            try:
                upd_scalar_dim = int(upd_scalar_dim)
            except Exception:
                upd_scalar_dim = None

        if upd_clip_channels is not None and upd_clip_channels != clip_channels_default:
            return False, f"incompatible clip_channels {upd_clip_channels} != {clip_channels_default}"

        if upd_scalar_dim is None:
            upd_scalar_dim = scalar_dim

        new_model = build_model(clip_channels_default, int(upd_scalar_dim), device)
        sd, meta = load_checkpoint(weights_path, device)
        new_model.load_state_dict(sd, strict=False)

        validate_model_forward(new_model, last_clip_shape, int(upd_scalar_dim), device, use_amp)

        model = new_model
        scalar_dim = int(upd_scalar_dim)
        model_norm_meta = meta.get("scalar_norm", {})
        new_version = update_fields.get("model_version") or meta.get("model_version") or "unknown"
        args.model_version = str(new_version)

        return True, f"applied model_version={args.model_version} scalar_dim={scalar_dim}"

    loop_n = 0
    last_clip_shape_seen = (19, 16, 112, 112)

    try:
        while True:
            loop_n += 1

            pull_global_track_updates()

            if args.check_updates_every_loops > 0 and (loop_n % args.check_updates_every_loops == 0):
                upd_streams = rdb.xread({updates_stream: last_upd_id}, block=1, count=5)
                if upd_streams:
                    for mid, fields in parse_xread_messages(upd_streams):
                        last_upd_id = mid
                        update_obj = {}

                        if b"json" in fields:
                            try:
                                update_obj = json.loads(b2s(fields[b"json"]))
                            except Exception:
                                update_obj = {}
                        else:
                            for k, v in fields.items():
                                update_obj[b2s(k)] = b2s(v)

                        ok, msg = try_apply_update(update_obj, last_clip_shape=last_clip_shape_seen)
                        print(f"[model_node] model_update mid={mid} ok={ok} msg={msg}")

            sc_streams = rdb.xread({scalars_clip_stream: last_scal_id}, block=1, count=100)
            if sc_streams:
                for mid, fields in parse_xread_messages(sc_streams):
                    last_scal_id = mid
                    try:
                        js = b2s(fields.get(b"json", b"{}"))
                        obj = json.loads(js)

                        meta = obj.get("meta", {})
                        feats = obj.get("features", {})
                        policy_features = obj.get("policy_features", {})

                        cam = meta.get("cam_id", cam_id)
                        pid = int(meta.get("person_track_id", -1))
                        fid_end = int(meta.get("frame_id_end", -1))
                        event_id = meta.get("event_id") or f"{cam}:{pid}:{fid_end}:{meta.get('stamp_ns_end', 0)}"

                        key = event_id
                        cache_put(key, {
                            "event_id": event_id,
                            "meta": meta,
                            "features": feats,
                            "policy_features": policy_features,
                        })
                    except Exception:
                        pass
                cache_prune()

            streams = rdb.xread({clips_stream: last_clip_id}, block=args.block_ms, count=args.count)
            if not streams:
                prune_identity_state(id_state, time.time(), args.identity_state_idle_s)
                continue

            for mid, fields in parse_xread_messages(streams):
                last_clip_id = mid

                meta_js = b2s(fields.get(b"meta", b"{}"))
                try:
                    meta = json.loads(meta_js)
                except Exception:
                    continue

                clip_bytes = fields.get(b"clip", None)
                if clip_bytes is None:
                    continue

                cam = meta.get("cam_id", cam_id)
                pid = int(meta.get("person_track_id", -1))
                fid_end = int(meta.get("frame_id_end", -1))
                stamp_ns_end = int(meta.get("stamp_ns_end", 0))
                event_id = meta.get("event_id") or f"{cam}:{pid}:{fid_end}:{stamp_ns_end}"

                c = int(meta.get("C", 19))
                t = int(meta.get("T", 16))
                h = int(meta.get("H", 112))
                w = int(meta.get("W", 112))
                last_clip_shape_seen = (c, t, h, w)

                if c != clip_channels_default:
                    continue

                clip_np = np.frombuffer(clip_bytes, dtype=np.float32)
                if clip_np.size != (c * t * h * w):
                    continue
                clip_np = clip_np.reshape((c, t, h, w))
                clip = torch.from_numpy(clip_np).unsqueeze(0).to(device)

                key = event_id
                got = cache_get(key)
                scalar_missing = False
                agg_feats = {}
                policy_features = {}
                if got is None:
                    scalar_missing = True
                else:
                    agg_feats = got.get("features", {}) or {}
                    policy_features = extract_policy_features(got, agg_feats)

                # -----------------------------
                # Resolve global identity
                # -----------------------------
                gid_keys = [
                    event_id,
                    f"{cam}:{pid}:{fid_end}",
                ]
                global_person_id = gid_cache_get(gid_keys)

                event_time_s = (stamp_ns_end * 1e-9) if stamp_ns_end > 0 else time.time()
                id_feat_vec, id_feat_dict = build_identity_features(
                    global_person_id=global_person_id,
                    id_state=id_state,
                    now_s=event_time_s,
                    recent_window_s=args.identity_recent_window_s,
                )

                # -----------------------------
                # Build scalar vector
                # -----------------------------
                vec, miss_cnt = build_scalar_vector(agg_feats, schema, fill_value=0.0)
                vec = append_special_scalars(vec, agg_feats)

                # Append identity features only if model expects them
                if scalar_dim > base_scalar_dim:
                    extra_needed = scalar_dim - base_scalar_dim
                    if extra_needed <= len(id_feat_vec):
                        id_append = id_feat_vec[:extra_needed]
                    else:
                        pad = np.zeros((extra_needed - len(id_feat_vec),), dtype=np.float32)
                        id_append = np.concatenate([id_feat_vec, pad], axis=0)
                    vec = np.concatenate([vec, id_append], axis=0)
                elif scalar_dim < base_scalar_dim:
                    vec = vec[:scalar_dim]

                vec = normalize_scalar_vector(vec, model_norm_meta)
                scalars = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0).to(device)

                # -----------------------------
                # Forward pass
                # -----------------------------
                with torch.no_grad():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out_model = model(clip, scalars)
                    else:
                        out_model = model(clip, scalars)

                raw_score_fused = torch.sigmoid(out_model["fused_logits"])[0, 0].item()
                score_cnn = torch.sigmoid(out_model["cnn_logits"])[0, 0].item()
                score_mlp = torch.sigmoid(out_model["mlp_logits"])[0, 0].item()

                identity_memory_score = compute_identity_memory_score(global_person_id, id_state)

                if args.enable_identity_memory_fusion and global_person_id is not None:
                    w_mem = clamp01(args.identity_memory_weight)
                    w_raw = 1.0 - w_mem
                    score_fused = clamp01(w_raw * raw_score_fused + w_mem * identity_memory_score)
                else:
                    score_fused = raw_score_fused

                # -----------------------------
                # Update identity memory after scoring
                # -----------------------------
                update_identity_state(
                    id_state=id_state,
                    global_person_id=global_person_id,
                    cam=cam,
                    event_time_s=event_time_s,
                    final_score=score_fused,
                    raw_fused_score=raw_score_fused,
                )

                prune_identity_state(id_state, time.time(), args.identity_state_idle_s)

                out = {
                    "type": "score",
                    "event_id": event_id,
                    "cam_id": cam,
                    "person_track_id": pid,
                    "global_person_id": int(global_person_id) if global_person_id is not None else None,
                    "frame_id_end": fid_end,
                    "stamp_ns_end": stamp_ns_end,

                    # backward compatible main score
                    "score": float(score_fused),

                    # split outputs
                    "score_fused": float(score_fused),
                    "score_fused_raw": float(raw_score_fused),
                    "score_cnn": float(score_cnn),
                    "score_mlp": float(score_mlp),
                    "score_identity_memory": float(identity_memory_score),

                    "model_version": args.model_version,
                    "scalar_missing": bool(scalar_missing),
                    "scalar_missing_fields": int(miss_cnt),
                    "object_track_id": int(meta.get("object_track_id", -1)),
                    "object_class_id": int(meta.get("object_class_id", -1)),
                    "missing_pose_ratio": float(meta.get("missing_pose_ratio", 0.0)),
                    "missing_obj_ratio": float(meta.get("missing_obj_ratio", 0.0)),
                    "policy_features": policy_features,

                    # identity traceability
                    "identity_features": id_feat_dict,
                    "identity_feature_names": identity_feature_names(),
                    "identity_memory_fusion_enabled": bool(args.enable_identity_memory_fusion),
                    "identity_memory_weight": float(args.identity_memory_weight),
                }

                rdb.xadd(
                    scores_stream,
                    {
                        "event_id": event_id,
                        "cam_id": cam,
                        "person_track_id": str(pid),
                        "global_person_id": "" if global_person_id is None else str(global_person_id),
                        "frame_id": str(fid_end),
                        "stamp_ns": str(stamp_ns_end),
                        "json": json.dumps(out),
                    },
                    maxlen=int(cfg.get("runtime", {}).get("redis_maxlen", 20000)),
                    approximate=True,
                )

                s_val = safe_float(policy_features.get("heuristic_theft_score", None), None)
                s_txt = f" heuristic={s_val:.3f}" if s_val is not None else ""
                gid_txt = f" gid={global_person_id}" if global_person_id is not None else " gid=None"

                print(
                    f"[model_node] cam={cam} pid={pid}{gid_txt} fid_end={fid_end} "
                    f"fused={score_fused:.3f} raw={raw_score_fused:.3f} "
                    f"cnn={score_cnn:.3f} mlp={score_mlp:.3f} mem={identity_memory_score:.3f} "
                    f"ver={args.model_version}{s_txt}"
                )

    except KeyboardInterrupt:
        print("\n[model_node] stopping...")


if __name__ == "__main__":
    main()