#!/usr/bin/env python3
"""
model_node.py  (CNN + MLP fusion + hot model updates)

Consumes:
  - Redis Stream: clips:<cam>  (float32 clip bytes + meta json)
  - Redis Stream: scalars_clip:<cam> (clip-aligned aggregated scalars json)

Produces:
  - Redis Stream: scores:<cam>  (raw model scores per clip)

Listens:
  - Redis Stream: model_updates (hot reload weights + version)

Safe hot reload:
  - Load new weights into a NEW model instance
  - Run a dummy forward pass to validate
  - Swap the model reference atomically
"""

import argparse
import json
import time
from collections import deque

import yaml
import redis
import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Config
# ----------------------------
def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


# ----------------------------
# Model (reference implementation)
# ----------------------------
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


class FusionHead(nn.Module):
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
        self.head = FusionHead(cnn_emb + s_emb)

    def forward(self, clip, scalars):
        zc = self.cnn(clip)
        zs = self.smlp(scalars)
        z = torch.cat([zc, zs], dim=1)
        logits = self.head(z)
        return logits


# ----------------------------
# Scalar schema helpers
# ----------------------------
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
    S = []
    for nm in ["hand_to_chest_L", "hand_to_chest_R", "hand_to_hip_L", "hand_to_hip_R"]:
        for st in ["mean", "max", "last"]:
            S.append((nm, st))
    for nm in ["wrist_v_L", "wrist_v_R"]:
        for st in ["mean", "max", "last"]:
            S.append((nm, st))

    S.append(("torso_angle_rate", "mean"))
    S.append(("torso_angle_rate", "max"))

    S.append(("guard_score", "mean"))
    S.append(("guard_score", "max"))
    S.append(("guard_score", "last"))

    for nm in ["obj_visibility", "carry_score", "comotion_cosine", "rel_offset_std_px"]:
        for st in ["mean", "min", "max", "last"]:
            S.append((nm, st))

    for nm in ["person_speed_px_s", "object_speed_px_s"]:
        for st in ["mean", "max", "last"]:
            S.append((nm, st))

    return S


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


# ----------------------------
# Redis helpers
# ----------------------------
def parse_xread_messages(streams):
    out = []
    for _sname, msgs in streams:
        for mid, fields in msgs:
            out.append((b2s(mid), fields))
    return out


# ----------------------------
# Hot reload helpers
# ----------------------------
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
    """
    clip_shape: (C,T,H,W) for a single sample
    """
    C, T, H, W = clip_shape
    clip = torch.zeros((1, C, T, H, W), dtype=torch.float32, device=device)
    scalars = torch.zeros((1, scalar_dim), dtype=torch.float32, device=device)
    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(clip, scalars)
        else:
            logits = model(clip, scalars)

    if logits is None or logits.numel() != 1:
        raise RuntimeError("Validation failed: logits shape unexpected.")
    if torch.isnan(logits).any():
        raise RuntimeError("Validation failed: logits contain NaNs.")
    return True


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--clips_stream", default=None)
    ap.add_argument("--scalars_clip_stream", default=None)
    ap.add_argument("--scores_stream", default=None)

    # Hot updates
    ap.add_argument("--updates_stream", default="model_updates")
    ap.add_argument("--updates_target", default=None, help="e.g. cam0 or all. If None, accept all.")
    ap.add_argument("--check_updates_every_loops", type=int, default=1)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=8)

    ap.add_argument("--scalar_cache_size", type=int, default=5000)
    ap.add_argument("--scalar_cache_ttl_s", type=float, default=5.0)

    ap.add_argument("--weights", default=None)
    ap.add_argument("--model_version", default="v0")

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

    clips_stream = args.clips_stream or r_cfg.get("clips_stream", f"clips:{cam_id}")
    scalars_clip_stream = args.scalars_clip_stream or r_cfg.get("scalars_clip_stream", f"scalars_clip:{cam_id}")
    scores_stream = args.scores_stream or r_cfg.get("scores_stream", f"scores:{cam_id}")

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    schema = default_scalar_schema()
    scalar_dim = len(schema) + 8

    clip_channels_default = 19
    model = build_model(clip_channels_default, scalar_dim, device)

    if args.weights:
        sd, meta = load_checkpoint(args.weights, device)
        model.load_state_dict(sd, strict=False)
        args.model_version = str(meta.get("model_version", args.model_version))

    print(f"[model_node] device={device} amp={use_amp}")
    print(f"[model_node] clips={clips_stream} scalars_clip={scalars_clip_stream} scores={scores_stream}")
    print(f"[model_node] updates_stream={args.updates_stream} updates_target={args.updates_target}")
    print(f"[model_node] scalar_dim={scalar_dim} model_version={args.model_version}")

    last_clip_id = "0-0"
    last_scal_id = "0-0"
    last_upd_id = "0-0"

    scalar_cache = {}
    scalar_fifo = deque(maxlen=args.scalar_cache_size)

    def cache_put(key, meta, features):
        now = time.time()
        scalar_cache[key] = (now, meta, features)
        scalar_fifo.append((now, key))

    def cache_get(key):
        item = scalar_cache.get(key, None)
        if not item:
            return None
        t0, meta, features = item
        if (time.time() - t0) > args.scalar_cache_ttl_s:
            scalar_cache.pop(key, None)
            return None
        return (meta, features)

    def cache_prune():
        now = time.time()
        while scalar_fifo:
            t0, k = scalar_fifo[0]
            if (now - t0) <= args.scalar_cache_ttl_s and len(scalar_cache) <= args.scalar_cache_size:
                break
            scalar_fifo.popleft()
            scalar_cache.pop(k, None)

    def try_apply_update(update_fields: dict, last_clip_shape=(19, 16, 112, 112)):
        """
        update_fields: dict from Redis fields or json payload
        expects weights_path and model_version
        """
        nonlocal model
        nonlocal clip_channels_default
        nonlocal scalar_dim

        weights_path = update_fields.get("weights_path") or update_fields.get("weights")
        if not weights_path:
            return False, "update missing weights_path"

        target = update_fields.get("target", None)
        if args.updates_target:
            if target not in (args.updates_target, "all", None):
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
        if upd_scalar_dim is not None and upd_scalar_dim != scalar_dim:
            return False, f"incompatible scalar_dim {upd_scalar_dim} != {scalar_dim}"

        new_model = build_model(clip_channels_default, scalar_dim, device)
        sd, meta = load_checkpoint(weights_path, device)
        new_model.load_state_dict(sd, strict=False)

        validate_model_forward(new_model, last_clip_shape, scalar_dim, device, use_amp)

        model = new_model
        new_version = update_fields.get("model_version") or meta.get("model_version") or "unknown"
        args.model_version = str(new_version)

        return True, f"applied model_version={args.model_version}"

    loop_n = 0
    last_clip_shape_seen = (19, 16, 112, 112)

    try:
        while True:
            loop_n += 1

            # 0) Check model_updates periodically (non-blocking)
            if args.check_updates_every_loops > 0 and (loop_n % args.check_updates_every_loops == 0):
                upd_streams = rdb.xread({args.updates_stream: last_upd_id}, block=1, count=5)
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

            # 1) Pull scalars_clip into cache
            sc_streams = rdb.xread({scalars_clip_stream: last_scal_id}, block=1, count=100)
            if sc_streams:
                for mid, fields in parse_xread_messages(sc_streams):
                    last_scal_id = mid
                    try:
                        js = b2s(fields.get(b"json", b"{}"))
                        obj = json.loads(js)
                        meta = obj.get("meta", {})
                        feats = obj.get("features", {})
                        cam = meta.get("cam_id", cam_id)
                        pid = int(meta.get("person_track_id", -1))
                        fid_end = int(meta.get("frame_id_end", -1))
                        key = (cam, pid, fid_end)
                        cache_put(key, meta, feats)
                    except Exception:
                        pass
                cache_prune()

            # 2) Read clips (blocking)
            streams = rdb.xread({clips_stream: last_clip_id}, block=args.block_ms, count=args.count)
            if not streams:
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

                C = int(meta.get("C", 19))
                T = int(meta.get("T", 16))
                H = int(meta.get("H", 112))
                W = int(meta.get("W", 112))
                last_clip_shape_seen = (C, T, H, W)

                if C != clip_channels_default:
                    continue

                clip_np = np.frombuffer(clip_bytes, dtype=np.float32)
                if clip_np.size != (C * T * H * W):
                    continue
                clip_np = clip_np.reshape((C, T, H, W))
                clip = torch.from_numpy(clip_np).unsqueeze(0).to(device)  # (1,C,T,H,W)

                key = (cam, pid, fid_end)
                got = cache_get(key)
                scalar_missing = False
                agg_feats = {}
                if got is None:
                    scalar_missing = True
                else:
                    _m, agg_feats = got

                vec, miss_cnt = build_scalar_vector(agg_feats, schema, fill_value=0.0)
                vec = append_special_scalars(vec, agg_feats)
                scalars = torch.from_numpy(vec).unsqueeze(0).to(device)  # (1,S)

                with torch.no_grad():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = model(clip, scalars)
                    else:
                        logits = model(clip, scalars)
                    score = torch.sigmoid(logits)[0, 0].item()

                out = {
                    "type": "score",
                    "cam_id": cam,
                    "person_track_id": pid,
                    "frame_id_end": fid_end,
                    "stamp_ns_end": stamp_ns_end,
                    "score": float(score),
                    "model_version": args.model_version,
                    "scalar_missing": bool(scalar_missing),
                    "scalar_missing_fields": int(miss_cnt),
                    "object_track_id": int(meta.get("object_track_id", -1)),
                    "object_class_id": int(meta.get("object_class_id", -1)),
                    "missing_pose_ratio": float(meta.get("missing_pose_ratio", 0.0)),
                    "missing_obj_ratio": float(meta.get("missing_obj_ratio", 0.0)),
                }

                rdb.xadd(
                    scores_stream,
                    {
                        "cam_id": cam,
                        "person_track_id": str(pid),
                        "frame_id": str(fid_end),
                        "stamp_ns": str(stamp_ns_end),
                        "json": json.dumps(out),
                    },
                    maxlen=int(cfg.get("runtime", {}).get("redis_maxlen", 20000)),
                    approximate=True,
                )

                print(f"[model_node] cam={cam} pid={pid} fid_end={fid_end} score={score:.3f} ver={args.model_version}")

    except KeyboardInterrupt:
        print("\n[model_node] stopping...")


if __name__ == "__main__":
    main()
