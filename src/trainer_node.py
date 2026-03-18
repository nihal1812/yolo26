#!/usr/bin/env python3
"""
trainer_node.py

Stronger lifecycle trainer for split-output FusionModel:
- dataset versioning
- persistent train/val/test split
- challenger vs champion evaluation
- acceptance / rejection gating
- rollback safety (only accepted challengers are published)
- label quality monitoring
- basic drift monitoring

Consumes:
  - train_events:{cam}

Produces:
  - model_updates
  - policy_updates
"""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import redis
import numpy as np
import torch
import torch.nn as nn

from config_utils import load_cfg, get_stream


def b2s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def parse_xread(streams):
    out = []
    for sname, msgs in streams:
        s = b2s(sname)
        for mid, fields in msgs:
            out.append((s, b2s(mid), fields))
    return out


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


def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def atomic_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


def load_json(path: Path, default=None):
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


def compute_scalar_norm(X: np.ndarray):
    mean = np.mean(X, axis=0).astype(np.float32)
    std = np.std(X, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def normalize_X(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)


def _get_stat(d, key, stat):
    try:
        v = d.get(key, None)
        if isinstance(v, dict):
            return v.get(stat, None)
    except Exception:
        pass
    return None


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


def build_scalar_vector(agg_features: dict, schema: list, fill_value: float = 0.0) -> Tuple[np.ndarray, int]:
    vec = []
    miss = 0
    for (fname, stat) in schema:
        val = _get_stat(agg_features, fname, stat)
        if val is None:
            vec.append(float(fill_value))
            miss += 1
        else:
            vec.append(float(val))
    return np.array(vec, dtype=np.float32), miss


def append_special_scalars(vec: np.ndarray, agg_features: dict) -> np.ndarray:
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


def extract_identity_feature_vector_from_score_obj(score_obj: dict, use_identity_features: bool) -> np.ndarray:
    names = identity_feature_names()
    if not use_identity_features:
        return np.zeros((0,), dtype=np.float32)

    feats = score_obj.get("identity_features", {})
    if not isinstance(feats, dict):
        feats = {}

    vals = []
    for name in names:
        vals.append(float(safe_float(feats.get(name, 0.0), 0.0)))
    return np.asarray(vals, dtype=np.float32)


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


def load_checkpoint(weights_path: str, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        state_dict = ckpt
        meta = {}
    return state_dict, meta


# ----------------------------
# Clip loading
# ----------------------------
# Kept exactly as requested: no changes to where/how clips are loaded.
def load_clip_tensor(local_clip_path: str, expected_channels: int = 19) -> Optional[np.ndarray]:
    if not local_clip_path:
        return None
    if not os.path.exists(local_clip_path):
        return None
    if not local_clip_path.lower().endswith(".npz"):
        return None

    try:
        data = np.load(local_clip_path, allow_pickle=True)
        if "clip" not in data:
            return None
        clip = data["clip"].astype(np.float32)
        if clip.ndim != 4:
            return None
        if clip.shape[0] != expected_channels:
            return None
        return clip
    except Exception:
        return None


def extract_training_sample(
    train_event: Dict[str, Any],
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
) -> Optional[Tuple[np.ndarray, np.ndarray, int, dict]]:
    fb = train_event.get("feedback", None)
    if not isinstance(fb, dict):
        return None

    y = safe_int(fb.get("label", None), None)
    if y is None:
        return None

    if require_train_ok and (train_event.get("train_ok", True) is not True):
        return None

    if skip_feedback_needs_review and bool(fb.get("needs_review", False)):
        return None

    fb_conf = safe_float(fb.get("llm_confidence", None), None)
    if min_feedback_confidence is not None:
        if fb_conf is not None and fb_conf < float(min_feedback_confidence):
            return None

    payload = train_event.get("payload", {})
    score_obj = payload.get("score", None)
    scal_obj = payload.get("scalars_clip", None)
    dec_obj = payload.get("decision", None)
    clip_ref = payload.get("clip_ref", None)

    if not isinstance(score_obj, dict) or not isinstance(scal_obj, dict) or not isinstance(clip_ref, dict):
        return None

    clip_path = clip_ref.get("local_clip_path", None)
    clip_np = load_clip_tensor(clip_path, expected_channels=expected_clip_channels)
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

    meta = {
        "event_id": train_event.get("event_id"),
        "cam_id": train_event.get("cam_id"),
        "person_track_id": train_event.get("person_track_id"),
        "global_person_id": train_event.get("global_person_id"),
        "clip_path": clip_path,
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

    return clip_np.astype(np.float32), x.astype(np.float32), int(y), meta


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
            abs(fpr - target_fpr) +
            0.30 * max(0.0, min_recall - recall) +
            0.20 * max(0.0, min_alert_rate - alert_rate)
        )

        if feasible:
            cand = (quality, float(thr), fpr, recall, alert_rate)
            if best is None or cand < best:
                best = cand

        fallback_quality = (
            abs(fpr - target_fpr) +
            0.50 * max(0.0, min_recall - recall) +
            0.35 * max(0.0, min_alert_rate - alert_rate)
        )
        cand_fb = (fallback_quality, float(thr), fpr, recall, alert_rate)
        if fallback is None or cand_fb < fallback:
            fallback = cand_fb

    chosen = best if best is not None else fallback
    if chosen is None:
        return 0.8
    return float(chosen[1])


def compute_binary_metrics(scores: np.ndarray, labels: np.ndarray, thr: float = 0.5):
    pred = (scores >= thr).astype(np.int64)
    acc = float((pred == labels).mean()) if labels.size else 0.0
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    recall = float(pred[pos_mask].mean()) if np.any(pos_mask) else 0.0
    fpr = float(pred[neg_mask].mean()) if np.any(neg_mask) else 0.0
    precision = float(np.sum((pred == 1) & (labels == 1)) / max(1, np.sum(pred == 1))) if labels.size else 0.0
    return {
        "acc": acc,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "alert_rate": float(np.mean(pred)) if labels.size else 0.0,
    }


def balanced_quality(metrics: dict, target_fpr: float) -> float:
    recall = float(metrics.get("recall", 0.0))
    fpr = float(metrics.get("fpr", 1.0))
    precision = float(metrics.get("precision", 0.0))
    return (1.8 * recall) + (0.7 * precision) - (1.2 * abs(fpr - target_fpr)) - (0.8 * fpr)


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


def enforce_split_class_coverage(split: Dict[str, List[str]], labels_by_event: Dict[str, int]) -> Dict[str, List[str]]:
    out = {k: list(v) for k, v in split.items()}

    def has_both(ids: List[str]) -> bool:
        ys = [labels_by_event[i] for i in ids if i in labels_by_event]
        return (0 in ys) and (1 in ys)

    if has_both(out["train"]) and has_both(out["val"]) and has_both(out["test"]):
        return out

    pool = sorted(labels_by_event.keys())

    def move_one(label_needed: int, src_names: List[str], dst_name: str):
        for src_name in src_names:
            for eid in list(out[src_name]):
                if labels_by_event.get(eid) == label_needed:
                    out[src_name].remove(eid)
                    out[dst_name].append(eid)
                    return True
        return False

    for bucket in ["train", "val", "test"]:
        ys = [labels_by_event[i] for i in out[bucket] if i in labels_by_event]
        if 0 not in ys:
            move_one(0, [b for b in ["train", "val", "test"] if b != bucket], bucket)
        ys = [labels_by_event[i] for i in out[bucket] if i in labels_by_event]
        if 1 not in ys:
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

    low_conf_ratio = None
    if confs:
        low_conf_ratio = float(np.mean(np.array(confs, dtype=np.float32) < 0.80))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cam_id", required=True)

    ap.add_argument("--train_events_stream", default=None)
    ap.add_argument("--model_updates_stream", default=None)
    ap.add_argument("--policy_updates_stream", default=None)

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=500)

    ap.add_argument("--min_labeled", type=int, default=200)
    ap.add_argument("--train_every_s", type=float, default=120.0)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze_cnn", action="store_true")

    ap.add_argument("--fusion_loss_weight", type=float, default=1.0)
    ap.add_argument("--cnn_loss_weight", type=float, default=0.30)
    ap.add_argument("--mlp_loss_weight", type=float, default=0.30)

    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--target_fpr", type=float, default=0.05)
    ap.add_argument("--min_recall_floor", type=float, default=0.40)
    ap.add_argument("--min_alert_rate_floor", type=float, default=0.03)
    ap.add_argument("--max_recommended_threshold", type=float, default=0.95)

    ap.add_argument("--clip_channels", type=int, default=19)

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

    ap.add_argument("--filter_bad_quality", action="store_true")
    ap.add_argument("--filter_require_gate_ok", action="store_true")
    ap.add_argument("--filter_max_missing_pose_ratio", type=float, default=0.55)
    ap.add_argument("--filter_max_missing_obj_ratio", type=float, default=0.75)

    ap.add_argument("--require_train_ok", action="store_true")
    ap.add_argument("--skip_feedback_needs_review", action="store_true")
    ap.add_argument("--min_feedback_confidence", type=float, default=None)

    ap.add_argument("--use_identity_features", action="store_true")

    # lifecycle / production-ish controls
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)

    ap.add_argument("--min_train_samples", type=int, default=40)
    ap.add_argument("--min_val_samples", type=int, default=20)
    ap.add_argument("--min_test_samples", type=int, default=20)

    ap.add_argument("--champion_min_recall", type=float, default=0.20)
    ap.add_argument("--champion_max_fpr", type=float, default=0.25)
    ap.add_argument("--accept_min_quality_gain", type=float, default=0.01)
    ap.add_argument("--accept_max_recall_drop", type=float, default=0.03)
    ap.add_argument("--accept_max_fpr_increase", type=float, default=0.03)
    ap.add_argument("--max_allowed_drift_score", type=float, default=6.0)

    ap.add_argument("--publish_rejected_candidates", action="store_true")
    ap.add_argument("--keep_last_n_manifests", type=int, default=20)

    args = ap.parse_args()
    cfg = load_cfg(args.config)
    cam_id = str(args.cam_id)

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    r_cfg = cfg.get("redis", {})
    rdb = redis.Redis(
        host=r_cfg.get("host", "127.0.0.1"),
        port=int(r_cfg.get("port", 6379)),
        db=int(r_cfg.get("db", 0)),
        password=r_cfg.get("password", None),
    )
    rdb.ping()

    train_events_stream = args.train_events_stream or get_stream(cfg, "train_events", cam_id)
    model_updates_stream = args.model_updates_stream or get_stream(cfg, "model_updates")
    policy_updates_stream = args.policy_updates_stream or get_stream(cfg, "policy_updates")

    root_dir = Path(args.out_dir).resolve()
    cam_dir = root_dir / cam_id
    manifests_dir = cam_dir / "manifests"
    checkpoints_dir = cam_dir / "checkpoints"
    registry_dir = cam_dir / "registry"
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(registry_dir, exist_ok=True)

    champion_registry_path = registry_dir / "champion.json"
    sample_cache_path = registry_dir / "samples.json"
    trainer_state_path = registry_dir / "trainer_state.json"

    schema = default_scalar_schema()
    base_scalar_dim = len(schema) + 8
    identity_dim = len(identity_feature_names()) if args.use_identity_features else 0
    scalar_dim = base_scalar_dim + identity_dim

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    print(f"[trainer_node] cam_id={cam_id}")
    print(f"[trainer_node] train_events={train_events_stream}")
    print(f"[trainer_node] model_updates={model_updates_stream}")
    print(f"[trainer_node] policy_updates={policy_updates_stream}")
    print(f"[trainer_node] device={device} scalar_dim={scalar_dim} clip_channels={args.clip_channels}")
    print(f"[trainer_node] base_scalar_dim={base_scalar_dim} identity_dim={identity_dim} use_identity_features={args.use_identity_features}")

    last_id = "0-0"
    seen_event_ids = set()

    persisted_samples = load_json(sample_cache_path, default={"samples": []})
    sample_records: Dict[str, dict] = {}
    for rec in persisted_samples.get("samples", []):
        eid = rec.get("event_id")
        if eid:
            sample_records[eid] = rec
            seen_event_ids.add(eid)

    trainer_state = load_json(trainer_state_path, default={"last_train_t": 0.0})
    last_train_t = float(trainer_state.get("last_train_t", 0.0))

    champion_registry = load_json(champion_registry_path, default={})

    def save_sample_cache():
        payload = {"samples": sorted(sample_records.values(), key=lambda x: str(x.get("event_id", "")))}
        atomic_write_json(sample_cache_path, payload)

    def save_trainer_state():
        atomic_write_json(trainer_state_path, {"last_train_t": float(last_train_t)})

    def build_model():
        m = FusionModel(clip_channels=args.clip_channels, scalar_dim=scalar_dim).to(device)
        m.train()
        return m

    def run_model_scores(model: nn.Module, C: np.ndarray, Xn: np.ndarray):
        model.eval()
        with torch.no_grad():
            clip_t = torch.from_numpy(C).to(device)
            X_t = torch.from_numpy(Xn).to(device)
            out = model(clip_t, X_t)

            fused_scores = torch.sigmoid(out["fused_logits"]).squeeze(1).cpu().numpy()
            cnn_scores = torch.sigmoid(out["cnn_logits"]).squeeze(1).cpu().numpy()
            mlp_scores = torch.sigmoid(out["mlp_logits"]).squeeze(1).cpu().numpy()

        return fused_scores, cnn_scores, mlp_scores

    def evaluate_model_bundle(model: nn.Module, C: np.ndarray, Xn: np.ndarray, y: np.ndarray, target_fpr: float):
        fused_scores, cnn_scores, mlp_scores = run_model_scores(model, C, Xn)

        fused_05 = compute_binary_metrics(fused_scores, y, thr=0.5)
        cnn_05 = compute_binary_metrics(cnn_scores, y, thr=0.5)
        mlp_05 = compute_binary_metrics(mlp_scores, y, thr=0.5)

        thr = suggest_threshold_balanced(
            fused_scores,
            y,
            target_fpr=target_fpr,
            min_recall=args.min_recall_floor,
            min_alert_rate=args.min_alert_rate_floor,
            max_threshold=args.max_recommended_threshold,
        )
        fused_bal = compute_binary_metrics(fused_scores, y, thr=thr)

        return {
            "fused_scores": fused_scores,
            "cnn_scores": cnn_scores,
            "mlp_scores": mlp_scores,
            "thr_balanced": float(thr),
            "fused_at_05": fused_05,
            "cnn_at_05": cnn_05,
            "mlp_at_05": mlp_05,
            "fused_at_balanced": fused_bal,
            "fused_quality": float(balanced_quality(fused_bal, target_fpr=target_fpr)),
        }

    def maybe_train_and_publish():
        nonlocal last_train_t, champion_registry

        now = time.time()
        if (now - last_train_t) < args.train_every_s:
            return

        if len(sample_records) < args.min_labeled:
            return

        all_event_ids = sorted(sample_records.keys())
        dataset_version = build_dataset_version(all_event_ids)

        split_path = manifests_dir / f"split_{dataset_version}.json"
        dataset_path = manifests_dir / f"dataset_{dataset_version}.json"
        report_path = manifests_dir / f"report_{dataset_version}.json"

        labels_by_event = {eid: int(sample_records[eid]["label"]) for eid in all_event_ids}
        split = load_json(split_path, default=None)
        if split is None:
            split = build_persistent_split(
                all_event_ids,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
            split = enforce_split_class_coverage(split, labels_by_event)
            atomic_write_json(split_path, split)

        train_ids = [eid for eid in split.get("train", []) if eid in sample_records]
        val_ids = [eid for eid in split.get("val", []) if eid in sample_records]
        test_ids = [eid for eid in split.get("test", []) if eid in sample_records]

        if len(train_ids) < args.min_train_samples or len(val_ids) < args.min_val_samples or len(test_ids) < args.min_test_samples:
            print(
                f"[trainer_node] skip training: split sizes too small "
                f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
            )
            last_train_t = now
            save_trainer_state()
            return

        def load_split_arrays(ids: List[str]):
            clips = []
            Xs = []
            ys = []
            metas = []
            for eid in ids:
                rec = sample_records.get(eid)
                if not rec:
                    continue
                clip_np = load_clip_tensor(rec.get("clip_path"), expected_channels=args.clip_channels)
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

        tr_pack = load_split_arrays(train_ids)
        va_pack = load_split_arrays(val_ids)
        te_pack = load_split_arrays(test_ids)

        if tr_pack is None or va_pack is None or te_pack is None:
            print("[trainer_node] skip training: one of train/val/test packs is empty after clip reload")
            last_train_t = now
            save_trainer_state()
            return

        Ctr, Xtr_raw, ytr, mtr = tr_pack
        Cva, Xva_raw, yva, mva = va_pack
        Cte, Xte_raw, yte, mte = te_pack

        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2 or len(np.unique(yte)) < 2:
            print("[trainer_node] skip training: missing class coverage in train/val/test")
            last_train_t = now
            save_trainer_state()
            return

        x_mean, x_std = compute_scalar_norm(Xtr_raw)
        Xtr = normalize_X(Xtr_raw, x_mean, x_std)
        Xva = normalize_X(Xva_raw, x_mean, x_std)
        Xte = normalize_X(Xte_raw, x_mean, x_std)

        model = build_model()

        if args.freeze_cnn:
            for p in model.cnn.parameters():
                p.requires_grad = False

        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

        pos = max(1, int((ytr == 1).sum()))
        neg = max(1, int((ytr == 0).sum()))
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_va_loss = float("inf")

        def run_epoch(Cb, Xb, yb, train: bool):
            model.train(train)
            bs = args.batch_size
            losses = []

            for i in range(0, Xb.shape[0], bs):
                clip_b = torch.from_numpy(Cb[i:i + bs]).to(device)
                xb = torch.from_numpy(Xb[i:i + bs]).to(device)
                yb_t = torch.from_numpy(yb[i:i + bs].astype(np.float32)).to(device).unsqueeze(1)

                with torch.set_grad_enabled(train):
                    out = model(clip_b, xb)

                    loss_fused = loss_fn(out["fused_logits"], yb_t)
                    loss_cnn = loss_fn(out["cnn_logits"], yb_t)
                    loss_mlp = loss_fn(out["mlp_logits"], yb_t)

                    loss = (
                        float(args.fusion_loss_weight) * loss_fused +
                        float(args.cnn_loss_weight) * loss_cnn +
                        float(args.mlp_loss_weight) * loss_mlp
                    )

                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()

                losses.append(float(loss.item()))
            return float(np.mean(losses)) if losses else 0.0

        epoch_hist = []
        for ep in range(args.epochs):
            tr_loss = run_epoch(Ctr, Xtr, ytr, train=True)
            va_loss = run_epoch(Cva, Xva, yva, train=False)
            epoch_hist.append({"epoch": ep + 1, "tr_loss": tr_loss, "va_loss": va_loss})
            print(f"[trainer_node] epoch {ep + 1}/{args.epochs} tr_loss={tr_loss:.4f} va_loss={va_loss:.4f}")

            if va_loss < best_va_loss:
                best_va_loss = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            print("[trainer_node] candidate rejected: no best state captured")
            last_train_t = now
            save_trainer_state()
            return

        model.load_state_dict(best_state, strict=True)

        challenger_val_eval = evaluate_model_bundle(model, Cva, Xva, yva, target_fpr=args.target_fpr)
        challenger_test_eval = evaluate_model_bundle(model, Cte, Xte, yte, target_fpr=args.target_fpr)

        val_fused = challenger_val_eval["fused_at_balanced"]
        test_fused = challenger_test_eval["fused_at_balanced"]

        print(
            f"[trainer_node] challenger val fused "
            f"acc={val_fused['acc']:.4f} recall={val_fused['recall']:.4f} "
            f"fpr={val_fused['fpr']:.4f} thr={challenger_val_eval['thr_balanced']:.3f}"
        )
        print(
            f"[trainer_node] challenger test fused "
            f"acc={test_fused['acc']:.4f} recall={test_fused['recall']:.4f} "
            f"fpr={test_fused['fpr']:.4f} thr={challenger_test_eval['thr_balanced']:.3f}"
        )

        label_quality = compute_label_quality([sample_records[eid] for eid in all_event_ids])
        drift_score = compute_feature_drift_score(Xtr_raw, champion_registry.get("scalar_norm")) if champion_registry else None

        ts = int(time.time())
        version = f"fusion_split_{cam_id}_{ts}"
        weights_path = os.path.abspath(os.path.join(checkpoints_dir, f"{version}.pt"))

        ckpt = {
            "state_dict": model.state_dict(),
            "model_type": "split_fusion",
            "branch_outputs": ["cnn", "mlp", "fusion"],
            "model_version": version,
            "clip_channels": int(args.clip_channels),
            "scalar_dim": int(scalar_dim),
            "base_scalar_dim": int(base_scalar_dim),
            "identity_dim": int(identity_dim),
            "use_identity_features": bool(args.use_identity_features),
            "identity_feature_names": identity_feature_names() if args.use_identity_features else [],
            "schema_len": int(len(schema)),
            "scalar_schema": [{"name": n, "stat": s} for (n, s) in schema],
            "trained_at_ns": time.time_ns(),
            "cam_id": cam_id,
            "notes": "Split-output FusionModel: cnn_head + mlp_head + fusion_head (+ optional identity features).",
            "freeze_cnn": bool(args.freeze_cnn),
            "seed": int(args.seed),
            "split_version": dataset_version,
            "train_samples": int(len(ytr)),
            "val_samples": int(len(yva)),
            "test_samples": int(len(yte)),
            "loss_weights": {
                "fusion": float(args.fusion_loss_weight),
                "cnn": float(args.cnn_loss_weight),
                "mlp": float(args.mlp_loss_weight),
            },
            "epoch_history": epoch_hist,
            "best_val_loss": float(best_va_loss),
            "recommended_threshold_fused": float(challenger_test_eval["thr_balanced"]),
            "scalar_norm": {
                "mean": x_mean.tolist(),
                "std": x_std.tolist(),
            },
            "label_quality": label_quality,
            "drift_score_vs_champion": drift_score,
            "val_eval": challenger_val_eval,
            "test_eval": challenger_test_eval,
        }
        torch.save(ckpt, weights_path)

        champion_eval = None
        champion_metrics = None
        champion_quality = None

        if champion_registry and champion_registry.get("weights_path") and os.path.exists(champion_registry["weights_path"]):
            try:
                champ_model = build_model()
                sd, meta = load_checkpoint(champion_registry["weights_path"], device)
                champ_model.load_state_dict(sd, strict=False)
                champ_scalar_norm = champion_registry.get("scalar_norm", None)

                if champ_scalar_norm and champ_scalar_norm.get("mean") is not None and champ_scalar_norm.get("std") is not None:
                    Xm_test = normalize_X(
                        Xte_raw,
                        np.asarray(champ_scalar_norm["mean"], dtype=np.float32),
                        np.asarray(champ_scalar_norm["std"], dtype=np.float32),
                    )
                else:
                    Xm_test = Xte

                champion_eval = evaluate_model_bundle(champ_model, Cte, Xm_test, yte, target_fpr=args.target_fpr)
                champion_metrics = champion_eval["fused_at_balanced"]
                champion_quality = champion_eval["fused_quality"]

                print(
                    f"[trainer_node] champion test fused "
                    f"acc={champion_metrics['acc']:.4f} recall={champion_metrics['recall']:.4f} "
                    f"fpr={champion_metrics['fpr']:.4f} thr={champion_eval['thr_balanced']:.3f}"
                )
            except Exception as e:
                print(f"[trainer_node] failed champion evaluation: {e}")

        challenger_quality = challenger_test_eval["fused_quality"]
        challenger_metrics = challenger_test_eval["fused_at_balanced"]

        acceptance_reasons = []
        accepted = True

        if challenger_metrics["recall"] < float(args.champion_min_recall):
            accepted = False
            acceptance_reasons.append("challenger_recall_below_min")

        if challenger_metrics["fpr"] > float(args.champion_max_fpr):
            accepted = False
            acceptance_reasons.append("challenger_fpr_above_max")

        if drift_score is not None and drift_score > float(args.max_allowed_drift_score):
            accepted = False
            acceptance_reasons.append("drift_score_too_high")

        if champion_metrics is not None and champion_quality is not None:
            quality_gain = challenger_quality - champion_quality
            recall_drop = champion_metrics["recall"] - challenger_metrics["recall"]
            fpr_increase = challenger_metrics["fpr"] - champion_metrics["fpr"]

            if quality_gain < float(args.accept_min_quality_gain):
                accepted = False
                acceptance_reasons.append("quality_gain_too_small")
            if recall_drop > float(args.accept_max_recall_drop):
                accepted = False
                acceptance_reasons.append("recall_drop_too_large")
            if fpr_increase > float(args.accept_max_fpr_increase):
                accepted = False
                acceptance_reasons.append("fpr_increase_too_large")
        else:
            quality_gain = None
            recall_drop = None
            fpr_increase = None

        dataset_manifest = {
            "dataset_version": dataset_version,
            "cam_id": cam_id,
            "created_at_ns": time.time_ns(),
            "num_samples_total": len(all_event_ids),
            "num_train": len(train_ids),
            "num_val": len(val_ids),
            "num_test": len(test_ids),
            "event_ids": all_event_ids,
            "split_file": str(split_path),
            "label_quality": label_quality,
            "drift_score_vs_champion": drift_score,
            "challenger_version": version,
            "challenger_weights_path": weights_path,
            "challenger_val_eval": challenger_val_eval,
            "challenger_test_eval": challenger_test_eval,
            "champion_registry_before": champion_registry,
            "champion_eval_on_current_test": champion_eval,
            "accepted": bool(accepted),
            "acceptance_reasons": acceptance_reasons,
            "quality_gain_vs_champion": quality_gain,
            "recall_drop_vs_champion": recall_drop,
            "fpr_increase_vs_champion": fpr_increase,
        }

        atomic_write_json(dataset_path, dataset_manifest)
        atomic_write_json(report_path, dataset_manifest)

        if not accepted and not args.publish_rejected_candidates:
            print(f"[trainer_node] challenger rejected: {acceptance_reasons}")
            last_train_t = now
            save_trainer_state()
            return

        if accepted:
            thr = float(challenger_test_eval["thr_balanced"])

            model_update_msg = {
                "type": "model_update",
                "target": cam_id,
                "weights_path": weights_path,
                "model_version": version,
                "model_type": "split_fusion",
                "clip_channels": str(args.clip_channels),
                "scalar_dim": str(scalar_dim),
                "base_scalar_dim": str(base_scalar_dim),
                "identity_dim": str(identity_dim),
                "use_identity_features": "1" if args.use_identity_features else "0",
                "dataset_version": dataset_version,
                "candidate_status": "accepted",
                "ts_ns": str(time.time_ns()),
            }
            rdb.xadd(model_updates_stream, model_update_msg)

            policy_msg = {
                "type": "policy_update",
                "target": cam_id,
                "recommended_threshold": str(thr),
                "target_fpr": str(args.target_fpr),
                "min_recall_floor": str(args.min_recall_floor),
                "min_alert_rate_floor": str(args.min_alert_rate_floor),
                "model_version": version,
                "dataset_version": dataset_version,
                "candidate_status": "accepted",
                "trained_samples": str(len(all_event_ids)),
                "pos": str(int(sum(1 for rec in sample_records.values() if int(rec["label"]) == 1))),
                "neg": str(int(sum(1 for rec in sample_records.values() if int(rec["label"]) == 0))),
                "test_acc_fused": str(challenger_metrics["acc"]),
                "test_recall_fused": str(challenger_metrics["recall"]),
                "test_fpr_fused": str(challenger_metrics["fpr"]),
                "test_precision_fused": str(challenger_metrics["precision"]),
                "drift_score_vs_champion": "" if drift_score is None else str(drift_score),
                "ts_ns": str(time.time_ns()),
            }

            if args.recommend_K is not None:
                policy_msg["K"] = str(int(args.recommend_K))
            if args.recommend_M is not None:
                policy_msg["M"] = str(int(args.recommend_M))
            if args.recommend_cooldown_s is not None:
                policy_msg["cooldown_s"] = str(float(args.recommend_cooldown_s))

            if args.recommend_use_suspicion_policy is not None:
                policy_msg["use_suspicion_policy"] = str(args.recommend_use_suspicion_policy)
            if args.recommend_vote_use_heuristic_score is not None:
                policy_msg["vote_use_heuristic_score"] = str(args.recommend_vote_use_heuristic_score)

            if args.recommend_carry_thr is not None:
                policy_msg["vote_carry_score_thr"] = str(float(args.recommend_carry_thr))
            if args.recommend_visibility_drop_thr is not None:
                policy_msg["vote_visibility_drop_thr"] = str(float(args.recommend_visibility_drop_thr))
            if args.recommend_contact_ratio_thr is not None:
                policy_msg["vote_contact_ratio_thr"] = str(float(args.recommend_contact_ratio_thr))
            if args.recommend_heuristic_thr is not None:
                policy_msg["heuristic_vote_thr"] = str(float(args.recommend_heuristic_thr))
            if args.recommend_susp_gain is not None:
                policy_msg["suspicion_gain"] = str(float(args.recommend_susp_gain))

            rdb.xadd(policy_updates_stream, policy_msg)

            champion_registry = {
                "cam_id": cam_id,
                "model_version": version,
                "weights_path": weights_path,
                "dataset_version": dataset_version,
                "test_metrics": challenger_metrics,
                "test_quality": challenger_quality,
                "threshold_fused": thr,
                "scalar_norm": {
                    "mean": x_mean.tolist(),
                    "std": x_std.tolist(),
                },
                "accepted_at_ns": time.time_ns(),
                "label_quality": label_quality,
            }
            atomic_write_json(champion_registry_path, champion_registry)

            print(f"[trainer_node] ACCEPTED champion={version}")
            print(f"[trainer_node] published model_update={version}")
            print(f"[trainer_node] recommended fused threshold={thr:.3f}")
        else:
            if args.publish_rejected_candidates:
                rdb.xadd(model_updates_stream, {
                    "type": "model_update",
                    "target": cam_id,
                    "weights_path": weights_path,
                    "model_version": version,
                    "model_type": "split_fusion",
                    "candidate_status": "rejected",
                    "dataset_version": dataset_version,
                    "reject_reasons": json.dumps(acceptance_reasons),
                    "ts_ns": str(time.time_ns()),
                })
            print(f"[trainer_node] challenger rejected: {acceptance_reasons}")

        manifests = sorted(manifests_dir.glob("dataset_*.json"))
        if len(manifests) > args.keep_last_n_manifests:
            for p in manifests[:-args.keep_last_n_manifests]:
                try:
                    p.unlink()
                except Exception:
                    pass

        last_train_t = now
        save_trainer_state()

    try:
        while True:
            streams = rdb.xread({train_events_stream: last_id}, block=args.block_ms, count=args.count)
            if not streams:
                maybe_train_and_publish()
                continue

            new_added = 0
            for _s, mid, fields in parse_xread(streams):
                last_id = mid
                js = fields.get(b"json", None)
                if js is None:
                    continue
                try:
                    ev = json.loads(b2s(js))
                except Exception:
                    continue

                event_id = ev.get("event_id", None)
                if not event_id:
                    continue
                if event_id in seen_event_ids:
                    continue

                sample = extract_training_sample(
                    ev,
                    schema=schema,
                    scalar_dim=scalar_dim,
                    base_scalar_dim=base_scalar_dim,
                    identity_dim=identity_dim,
                    use_identity_features=bool(args.use_identity_features),
                    expected_clip_channels=args.clip_channels,
                    filter_bad_quality=bool(args.filter_bad_quality),
                    max_missing_pose_ratio=float(args.filter_max_missing_pose_ratio),
                    max_missing_obj_ratio=float(args.filter_max_missing_obj_ratio),
                    require_gate_ok=bool(args.filter_require_gate_ok),
                    require_train_ok=bool(args.require_train_ok),
                    skip_feedback_needs_review=bool(args.skip_feedback_needs_review),
                    min_feedback_confidence=args.min_feedback_confidence,
                )
                if sample is None:
                    continue

                clip_np, x, yv, meta = sample

                rec = dict(meta)
                rec["event_id"] = event_id
                rec["label"] = int(yv)
                rec["x"] = x.astype(np.float32).tolist()
                rec["clip_shape"] = list(clip_np.shape)

                sample_records[event_id] = rec
                seen_event_ids.add(event_id)
                new_added += 1

            if new_added > 0:
                save_sample_cache()

            total = len(sample_records)
            if total > 0 and (total % 25 == 0 or new_added > 0):
                pos = sum(1 for rec in sample_records.values() if int(rec["label"]) == 1)
                neg = total - pos
                q = compute_label_quality(list(sample_records.values()))
                print(
                    f"[trainer_node] labeled={total} (pos={pos}, neg={neg}) "
                    f"avg_conf={q.get('avg_feedback_confidence')} "
                    f"needs_review_ratio={q.get('needs_review_ratio')}"
                )

            maybe_train_and_publish()

    except KeyboardInterrupt:
        print("\n[trainer_node] stopping...")


if __name__ == "__main__":
    main()