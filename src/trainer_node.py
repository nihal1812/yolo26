#!/usr/bin/env python3
"""
trainer_node.py (FusionModel trainer + policy updater)

Consumes:
  - train_events:<cam>

Produces:
  - model_updates
  - policy_updates

NEW:
  - Optional filtering using decision quality (gate_ok / missing ratios / have_scalars_clip)
  - Can publish extended policy params for suspicion/voting system
"""

import argparse
import json
import os
import time
from typing import Dict, Any, List, Tuple, Optional

import yaml
import redis
import numpy as np

import torch
import torch.nn as nn


# ----------------------------
# Config helpers
# ----------------------------
def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


# ----------------------------
# Scalar schema (MUST match model_node)
# ----------------------------
def _get_stat(d, key, stat):
    try:
        v = d.get(key, None)
        if isinstance(v, dict):
            return v.get(stat, None)
    except Exception:
        pass
    return None


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


# ----------------------------
# FusionModel (same as model_node)
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
# Training sample extraction
# ----------------------------
def extract_training_sample(
    train_event: Dict[str, Any],
    schema: list,
    scalar_dim: int,
    filter_bad_quality: bool,
    max_missing_pose_ratio: float,
    max_missing_obj_ratio: float,
    require_gate_ok: bool,
    require_train_ok: bool,
    skip_feedback_needs_review: bool,
    min_feedback_confidence: Optional[float],
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Returns (x_vec, y_label) or None if sample not usable.

    Requires:
      - feedback.label (0/1)
      - payload.score.score (raw model score)
      - payload.scalars_clip.features (aggregated scalars)

    Optional quality filtering via payload.decision fields.
    """
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

    if min_feedback_confidence is not None:
        fb_conf = safe_float(fb.get("llm_confidence", None), None)
        if fb_conf is not None and fb_conf < float(min_feedback_confidence):
            return None

    payload = train_event.get("payload", {})
    score_obj = payload.get("score", None)
    scal_obj = payload.get("scalars_clip", None)
    dec_obj = payload.get("decision", None)

    if not isinstance(score_obj, dict) or not isinstance(scal_obj, dict):
        return None

    raw_score = safe_float(score_obj.get("score", None), None)
    if raw_score is None:
        return None

    feats = scal_obj.get("features", {})
    if not isinstance(feats, dict):
        feats = {}

    # Optional: skip bad quality labels
    if filter_bad_quality:
        miss_pose = None
        miss_obj = None
        gate_ok = None
        have_feats = None

        if isinstance(dec_obj, dict):
            miss_pose = safe_float(dec_obj.get("missing_pose_ratio", None), None)
            miss_obj = safe_float(dec_obj.get("missing_obj_ratio", None), None)
            gate_ok = dec_obj.get("gate_ok", None)
            have_feats = dec_obj.get("have_scalars_clip", None)

        if miss_pose is None:
            miss_pose = safe_float(score_obj.get("missing_pose_ratio", None), None)
        if miss_obj is None:
            miss_obj = safe_float(score_obj.get("missing_obj_ratio", None), None)

        if have_feats is False:
            return None
        if miss_pose is not None and miss_pose > float(max_missing_pose_ratio):
            return None
        if miss_obj is not None and miss_obj > float(max_missing_obj_ratio):
            return None
        if require_gate_ok and (gate_ok is not True):
            return None

    svec, _miss = build_scalar_vector(feats, schema, fill_value=0.0)
    svec = append_special_scalars(svec, feats)

    # Keep dims stable (must match model_node)
    # We inject raw_score as first feature and truncate/pad to scalar_dim.
    x_full = np.concatenate([np.array([raw_score], dtype=np.float32), svec], axis=0)

    if x_full.size >= scalar_dim:
        x = x_full[:scalar_dim]
    else:
        pad = np.zeros((scalar_dim - x_full.size,), dtype=np.float32)
        x = np.concatenate([x_full, pad], axis=0)

    return (x.astype(np.float32), int(y))


# ----------------------------
# Policy suggestion
# ----------------------------
def suggest_threshold(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.05) -> float:
    neg = scores[labels == 0]
    if neg.size < 20:
        return 0.8
    thr = float(np.quantile(neg, 1.0 - target_fpr))
    return max(0.05, min(0.99, thr))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--train_events_stream", default=None)

    ap.add_argument("--model_updates_stream", default="model_updates")
    ap.add_argument("--policy_updates_stream", default="policy_updates")

    ap.add_argument("--block_ms", type=int, default=1000)
    ap.add_argument("--count", type=int, default=500)

    # Training schedule
    ap.add_argument("--min_labeled", type=int, default=200)
    ap.add_argument("--train_every_s", type=float, default=120.0)

    # Model/training
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze_cnn", action="store_true")

    # Output directory
    ap.add_argument("--out_dir", default="models")

    # Threshold suggestion target
    ap.add_argument("--target_fpr", type=float, default=0.05)

    # Metadata: must match model_node defaults
    ap.add_argument("--clip_channels", type=int, default=19)

    # Optional legacy recommendations
    ap.add_argument("--recommend_K", type=int, default=None)
    ap.add_argument("--recommend_M", type=int, default=None)
    ap.add_argument("--recommend_cooldown_s", type=float, default=None)

    # -------------------------
    # NEW: optional extended policy recommendations
    # (only sent if provided)
    # -------------------------
    ap.add_argument("--recommend_use_suspicion_policy", type=str, default=None,
                    help="If set (true/false), publish use_suspicion_policy toggle.")
    ap.add_argument("--recommend_vote_use_heuristic_score", type=str, default=None,
                    help="If set (true/false), publish vote_use_heuristic_score toggle.")

    ap.add_argument("--recommend_carry_thr", type=float, default=None)
    ap.add_argument("--recommend_visibility_drop_thr", type=float, default=None)
    ap.add_argument("--recommend_contact_ratio_thr", type=float, default=None)
    ap.add_argument("--recommend_heuristic_thr", type=float, default=None)

    ap.add_argument("--recommend_susp_gain", type=float, default=None)
    ap.add_argument("--recommend_susp_decay", type=float, default=None)
    ap.add_argument("--recommend_suspicious_enter", type=float, default=None)
    ap.add_argument("--recommend_suspicious_exit", type=float, default=None)
    ap.add_argument("--recommend_alert_enter", type=float, default=None)
    ap.add_argument("--recommend_alert_exit", type=float, default=None)

    ap.add_argument("--recommend_votes_M", type=int, default=None)
    ap.add_argument("--recommend_votes_K", type=int, default=None)
    ap.add_argument("--recommend_strong_votes_min", type=int, default=None)

    # -------------------------
    # NEW: quality filtering for labeled training samples
    # -------------------------
    ap.add_argument("--filter_bad_quality", action="store_true",
                    help="If set, skip labeled samples with poor quality based on decision/score missing ratios.")
    ap.add_argument("--filter_require_gate_ok", action="store_true",
                    help="If set, require decision.gate_ok==True for labeled samples.")
    ap.add_argument("--filter_max_missing_pose_ratio", type=float, default=0.55)
    ap.add_argument("--filter_max_missing_obj_ratio", type=float, default=0.75)

    # Optional LLM-feedback-aware filtering
    ap.add_argument("--require_train_ok", action="store_true",
                    help="If set, only train on events explicitly marked train_ok=True.")
    ap.add_argument("--skip_feedback_needs_review", action="store_true",
                    help="If set, skip feedback with feedback.needs_review==True.")
    ap.add_argument("--min_feedback_confidence", type=float, default=None,
                    help="If set, skip samples with feedback.llm_confidence below this value. If confidence is absent, sample is kept.")

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

    train_events_stream = args.train_events_stream or r_cfg.get("train_events_stream", f"train_events:{cam_id}")
    os.makedirs(args.out_dir, exist_ok=True)

    schema = default_scalar_schema()
    scalar_dim = len(schema) + 8  # MUST match model_node

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    print(f"[trainer_node] train_events={train_events_stream}")
    print(f"[trainer_node] device={device} scalar_dim={scalar_dim} clip_channels={args.clip_channels}")
    print(f"[trainer_node] min_labeled={args.min_labeled} train_every_s={args.train_every_s} freeze_cnn={args.freeze_cnn}")
    print(f"[trainer_node] filter_bad_quality={args.filter_bad_quality} filter_require_gate_ok={args.filter_require_gate_ok}")
    print(f"[trainer_node] require_train_ok={args.require_train_ok} skip_feedback_needs_review={args.skip_feedback_needs_review} min_feedback_confidence={args.min_feedback_confidence}")

    last_id = "0-0"
    labeled_X: List[np.ndarray] = []
    labeled_y: List[int] = []
    last_train_t = 0.0

    def build_model():
        m = FusionModel(clip_channels=args.clip_channels, scalar_dim=scalar_dim).to(device)
        m.train()
        return m

    def maybe_train_and_publish():
        nonlocal labeled_X, labeled_y, last_train_t

        now = time.time()
        if (now - last_train_t) < args.train_every_s:
            return
        if len(labeled_y) < args.min_labeled:
            return

        X = np.stack(labeled_X, axis=0).astype(np.float32)
        y = np.array(labeled_y, dtype=np.int64)

        n = X.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        split = int(0.8 * n)
        tr, va = idx[:split], idx[split:]

        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        model = build_model()

        if args.freeze_cnn:
            for p in model.cnn.parameters():
                p.requires_grad = False

        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        dummy_clip = torch.zeros((args.batch_size, args.clip_channels, 16, 112, 112), dtype=torch.float32, device=device)

        def run_epoch(Xb, yb, train: bool):
            model.train(train)
            bs = args.batch_size
            losses = []
            for i in range(0, Xb.shape[0], bs):
                xb = torch.from_numpy(Xb[i:i+bs]).to(device)
                yb_t = torch.from_numpy(yb[i:i+bs].astype(np.float32)).to(device).unsqueeze(1)
                clip_b = dummy_clip[: xb.shape[0]]

                with torch.set_grad_enabled(train):
                    logits = model(clip_b, xb)
                    loss = loss_fn(logits, yb_t)
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()

                losses.append(float(loss.item()))
            return float(np.mean(losses)) if losses else 0.0

        for ep in range(args.epochs):
            tr_loss = run_epoch(Xtr, ytr, train=True)
            va_loss = run_epoch(Xva, yva, train=False)
            print(f"[trainer_node] epoch {ep+1}/{args.epochs} tr_loss={tr_loss:.4f} va_loss={va_loss:.4f}")

        # Evaluate calibrated scores
        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).to(device)
            clip_t = torch.zeros((X.shape[0], args.clip_channels, 16, 112, 112), dtype=torch.float32, device=device)
            logits = model(clip_t, X_t).squeeze(1)
            calib_scores = torch.sigmoid(logits).cpu().numpy()

        thr = suggest_threshold(calib_scores, y, target_fpr=args.target_fpr)

        # Save weights
        ts = int(time.time())
        version = f"fusion_{cam_id}_{ts}"
        weights_path = os.path.abspath(os.path.join(args.out_dir, f"{version}.pt"))

        ckpt = {
            "state_dict": model.state_dict(),
            "model_version": version,
            "clip_channels": int(args.clip_channels),
            "scalar_dim": int(scalar_dim),
            "schema_len": int(len(schema)),
            "trained_at_ns": time.time_ns(),
            "notes": "FusionModel update (trained fast on scalars-only path; clip input zeroed).",
            "freeze_cnn": bool(args.freeze_cnn),
        }
        torch.save(ckpt, weights_path)

        # Publish model update
        rdb.xadd(args.model_updates_stream, {
            "type": "model_update",
            "target": cam_id,
            "weights_path": weights_path,
            "model_version": version,
            "clip_channels": str(args.clip_channels),
            "scalar_dim": str(scalar_dim),
            "ts_ns": str(time.time_ns()),
        })

        # Publish policy update suggestion (extended)
        policy_msg = {
            "type": "policy_update",
            "target": cam_id,
            "recommended_threshold": str(thr),
            "target_fpr": str(args.target_fpr),
            "model_version": version,
            "trained_samples": str(n),
            "pos": str(int((y == 1).sum())),
            "neg": str(int((y == 0).sum())),
            "ts_ns": str(time.time_ns()),
        }

        # legacy optional
        if args.recommend_K is not None:
            policy_msg["K"] = str(int(args.recommend_K))
        if args.recommend_M is not None:
            policy_msg["M"] = str(int(args.recommend_M))
        if args.recommend_cooldown_s is not None:
            policy_msg["cooldown_s"] = str(float(args.recommend_cooldown_s))

        # new optional toggles
        if args.recommend_use_suspicion_policy is not None:
            policy_msg["use_suspicion_policy"] = str(args.recommend_use_suspicion_policy)
        if args.recommend_vote_use_heuristic_score is not None:
            policy_msg["vote_use_heuristic_score"] = str(args.recommend_vote_use_heuristic_score)

        # new optional thresholds/params
        if args.recommend_carry_thr is not None:
            policy_msg["carry_thr"] = str(float(args.recommend_carry_thr))
        if args.recommend_visibility_drop_thr is not None:
            policy_msg["visibility_drop_thr"] = str(float(args.recommend_visibility_drop_thr))
        if args.recommend_contact_ratio_thr is not None:
            policy_msg["contact_ratio_thr"] = str(float(args.recommend_contact_ratio_thr))
        if args.recommend_heuristic_thr is not None:
            policy_msg["heuristic_thr"] = str(float(args.recommend_heuristic_thr))

        if args.recommend_susp_gain is not None:
            policy_msg["susp_gain"] = str(float(args.recommend_susp_gain))
        if args.recommend_susp_decay is not None:
            policy_msg["susp_decay"] = str(float(args.recommend_susp_decay))
        if args.recommend_suspicious_enter is not None:
            policy_msg["suspicious_enter"] = str(float(args.recommend_suspicious_enter))
        if args.recommend_suspicious_exit is not None:
            policy_msg["suspicious_exit"] = str(float(args.recommend_suspicious_exit))
        if args.recommend_alert_enter is not None:
            policy_msg["alert_enter"] = str(float(args.recommend_alert_enter))
        if args.recommend_alert_exit is not None:
            policy_msg["alert_exit"] = str(float(args.recommend_alert_exit))

        if args.recommend_votes_M is not None:
            policy_msg["votes_M"] = str(int(args.recommend_votes_M))
        if args.recommend_votes_K is not None:
            policy_msg["votes_K"] = str(int(args.recommend_votes_K))
        if args.recommend_strong_votes_min is not None:
            policy_msg["strong_votes_min"] = str(int(args.recommend_strong_votes_min))

        rdb.xadd(args.policy_updates_stream, policy_msg)

        print(f"[trainer_node] published model_update={version}")
        print(f"[trainer_node] recommended threshold={thr:.3f} (target_fpr={args.target_fpr})")

        last_train_t = now

    try:
        while True:
            streams = rdb.xread({train_events_stream: last_id}, block=args.block_ms, count=args.count)
            if not streams:
                maybe_train_and_publish()
                continue

            for _s, mid, fields in parse_xread(streams):
                last_id = mid
                js = fields.get(b"json", None)
                if js is None:
                    continue
                try:
                    ev = json.loads(b2s(js))
                except Exception:
                    continue

                sample = extract_training_sample(
                    ev,
                    schema=schema,
                    scalar_dim=scalar_dim,
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
                x, yv = sample
                labeled_X.append(x)
                labeled_y.append(yv)

            if len(labeled_y) % 50 == 0 and len(labeled_y) > 0:
                pos = sum(1 for v in labeled_y if v == 1)
                neg = len(labeled_y) - pos
                print(f"[trainer_node] labeled={len(labeled_y)} (pos={pos}, neg={neg})")

            maybe_train_and_publish()

    except KeyboardInterrupt:
        print("\n[trainer_node] stopping...")


if __name__ == "__main__":
    main()