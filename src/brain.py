#!/usr/bin/env python3
"""
brain.py (Orchestrator for the brain pipeline)

Starts:
  - model_node.py
  - policy_node.py
  - data_collector.py / data_collector_llm.py
  - trainer_node.py / trainer_node_updated.py

Behavior:
  - Always passes --config to every node.
  - Reads optional extra CLI flags for policy / collector / trainer from YAML.
  - Prefers updated node files when they exist, but falls back to the original names.
"""

import sys
import time
import yaml
import signal
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def popen_node(py_exe, script_path: Path, cfg_path: str, extra_args=None):
    cmd = [py_exe, str(script_path), "--config", cfg_path]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[brain] exec: {' '.join(map(str, cmd))}")
    return subprocess.Popen(cmd, cwd=str(HERE))


def terminate_all(procs, grace_s=3.0):
    for p in procs:
        try:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        except Exception:
            pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if all(p.poll() is not None for p in procs):
            return
        time.sleep(0.1)

    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


def as_bool(v, default=False):
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def resolve_node(preferred_name: str, fallback_name: str = None) -> Path:
    preferred = HERE / preferred_name
    if preferred.exists():
        return preferred
    if fallback_name:
        fallback = HERE / fallback_name
        if fallback.exists():
            return fallback
    return preferred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--no-collector", action="store_true")
    ap.add_argument("--no-trainer", action="store_true")

    ap.add_argument("--model-device", default=None)
    ap.add_argument("--trainer-device", default=None)

    # Optional hard override if you want to force original vs updated files
    ap.add_argument("--collector-script", default=None,
                    help="Optional explicit collector script path/name.")
    ap.add_argument("--trainer-script", default=None,
                    help="Optional explicit trainer script path/name.")

    args = ap.parse_args()
    cfg_path = args.config
    cfg = load_cfg(cfg_path) or {}

    py = sys.executable

    model_node_path = HERE / "model_node.py"
    policy_node_path = HERE / "policy_node.py"

    if args.collector_script:
        collector_path = (HERE / args.collector_script).resolve() if not Path(args.collector_script).is_absolute() else Path(args.collector_script)
    else:
        collector_path = resolve_node("data_collector_llm.py", "data_collector.py")

    if args.trainer_script:
        trainer_path = (HERE / args.trainer_script).resolve() if not Path(args.trainer_script).is_absolute() else Path(args.trainer_script)
    else:
        trainer_path = resolve_node("trainer_node_updated.py", "trainer_node.py")

    # ---------------------------
    # Existing pass-through args
    # ---------------------------
    model_args = []
    if args.model_device:
        model_args += ["--device", args.model_device]

    trainer_args = []
    if args.trainer_device:
        trainer_args += ["--device", args.trainer_device]

    # ---------------------------
    # Policy config-driven flags
    # ---------------------------
    policy_cfg = cfg.get("policy", {}) if isinstance(cfg, dict) else {}
    policy_args = []

    if as_bool(policy_cfg.get("use_suspicion_policy", False)):
        policy_args.append("--use_suspicion_policy")
    if as_bool(policy_cfg.get("vote_use_heuristic_score", False)):
        policy_args.append("--vote_use_heuristic_score")

    # Optional threshold pass-throughs if the policy node supports them
    for key in [
        "carry_thr",
        "visibility_drop_thr",
        "contact_ratio_thr",
        "heuristic_thr",
        "susp_gain",
        "susp_decay",
        "suspicious_enter",
        "suspicious_exit",
        "alert_enter",
        "alert_exit",
        "votes_M",
        "votes_K",
        "strong_votes_min",
        "K",
        "M",
        "cooldown_s",
    ]:
        if key in policy_cfg and policy_cfg[key] is not None:
            policy_args += [f"--{key}", str(policy_cfg[key])]

    # ---------------------------
    # Collector config-driven flags
    # ---------------------------
    collector_cfg = cfg.get("data_collector", {}) if isinstance(cfg, dict) else {}
    collector_args = []

    # Existing collector gating
    if as_bool(collector_cfg.get("gold_filter_unlabeled_ready", False)):
        collector_args.append("--gold_filter_unlabeled_ready")
    if as_bool(collector_cfg.get("gold_require_gate_ok", False)):
        collector_args.append("--gold_require_gate_ok")
    if as_bool(collector_cfg.get("publish_partial", False)):
        collector_args.append("--publish_partial")
    if as_bool(collector_cfg.get("read_clips_stream", False)):
        collector_args.append("--read_clips_stream")

    for key, cast in [
        ("gold_min_votes", int),
        ("gold_min_S", float),
        ("gold_max_missing_pose_ratio", float),
        ("gold_max_missing_obj_ratio", float),
        ("block_ms", int),
        ("count", int),
        ("redis_maxlen", int),
        ("cache_ttl_s", float),
        ("pending_ttl_s", float),
    ]:
        if key in collector_cfg and collector_cfg[key] is not None:
            collector_args += [f"--{key}", str(cast(collector_cfg[key]))]

    for key in [
        "scores_stream",
        "decisions_stream",
        "alerts_stream",
        "scalars_clip_stream",
        "feedback_stream",
        "clips_stream",
        "train_events_stream",
    ]:
        if key in collector_cfg and collector_cfg[key]:
            collector_args += [f"--{key}", str(collector_cfg[key])]

    # New collector LLM flags (supported by the LLM-enabled collector)
    if as_bool(collector_cfg.get("enable_llm_feedback_parse", False)):
        collector_args.append("--enable_llm_feedback_parse")
    if as_bool(collector_cfg.get("llm_disable_fallback", False)):
        collector_args.append("--llm_disable_fallback")

    for key, cast in [
        ("llm_model", str),
        ("llm_timeout_s", float),
        ("feedback_min_confidence", float),
        ("llm_allowed_categories", str),
        ("llm_api_key", str),
        ("llm_base_url", str),
    ]:
        if key in collector_cfg and collector_cfg[key] is not None:
            collector_args += [f"--{key}", str(cast(collector_cfg[key]))]

    # ---------------------------
    # Trainer config-driven flags
    # ---------------------------
    trainer_cfg = cfg.get("trainer", {}) if isinstance(cfg, dict) else {}

    # Existing trainer filtering toggles
    if as_bool(trainer_cfg.get("filter_bad_quality", False)):
        trainer_args.append("--filter_bad_quality")
    if as_bool(trainer_cfg.get("filter_require_gate_ok", False)):
        trainer_args.append("--filter_require_gate_ok")

    for key, cast in [
        ("filter_max_missing_pose_ratio", float),
        ("filter_max_missing_obj_ratio", float),
    ]:
        if key in trainer_cfg and trainer_cfg[key] is not None:
            trainer_args += [f"--{key}", str(cast(trainer_cfg[key]))]

    # New trainer filters for LLM-labeled data (supported by trainer_node_updated.py)
    if as_bool(trainer_cfg.get("require_train_ok", False)):
        trainer_args.append("--require_train_ok")
    if as_bool(trainer_cfg.get("skip_feedback_needs_review", False)):
        trainer_args.append("--skip_feedback_needs_review")
    if "min_feedback_confidence" in trainer_cfg and trainer_cfg["min_feedback_confidence"] is not None:
        trainer_args += ["--min_feedback_confidence", str(float(trainer_cfg["min_feedback_confidence"]))]

    # Existing/common trainer passthroughs
    for key, cast in [
        ("train_events_stream", str),
        ("model_updates_stream", str),
        ("policy_updates_stream", str),
        ("block_ms", int),
        ("count", int),
        ("min_labeled", int),
        ("train_every_s", float),
        ("epochs", int),
        ("batch_size", int),
        ("lr", float),
        ("out_dir", str),
        ("target_fpr", float),
        ("clip_channels", int),
        ("recommend_K", int),
        ("recommend_M", int),
        ("recommend_cooldown_s", float),
        ("recommend_use_suspicion_policy", str),
        ("recommend_vote_use_heuristic_score", str),
        ("recommend_carry_thr", float),
        ("recommend_visibility_drop_thr", float),
        ("recommend_contact_ratio_thr", float),
        ("recommend_heuristic_thr", float),
        ("recommend_susp_gain", float),
        ("recommend_susp_decay", float),
        ("recommend_suspicious_enter", float),
        ("recommend_suspicious_exit", float),
        ("recommend_alert_enter", float),
        ("recommend_alert_exit", float),
        ("recommend_votes_M", int),
        ("recommend_votes_K", int),
        ("recommend_strong_votes_min", int),
    ]:
        if key in trainer_cfg and trainer_cfg[key] is not None:
            trainer_args += [f"--{key}", str(cast(trainer_cfg[key]))]

    if as_bool(trainer_cfg.get("freeze_cnn", False)):
        trainer_args.append("--freeze_cnn")

    procs = []
    try:
        print(f"[brain] config={cfg_path}")
        print(f"[brain] collector_script={collector_path.name}")
        print(f"[brain] trainer_script={trainer_path.name}")

        print("[brain] starting model_node...")
        procs.append(popen_node(py, model_node_path, cfg_path, extra_args=model_args))

        time.sleep(0.5)

        if not args.no_policy:
            print("[brain] starting policy_node...")
            if policy_args:
                print(f"[brain] policy_node extra args: {policy_args}")
            procs.append(popen_node(py, policy_node_path, cfg_path, extra_args=policy_args))

        time.sleep(0.3)

        if not args.no_collector:
            print("[brain] starting data_collector...")
            if collector_args:
                print(f"[brain] data_collector extra args: {collector_args}")
            procs.append(popen_node(py, collector_path, cfg_path, extra_args=collector_args))

        time.sleep(0.3)

        if not args.no_trainer:
            print("[brain] starting trainer_node...")
            if trainer_args:
                print(f"[brain] trainer_node extra args: {trainer_args}")
            procs.append(popen_node(py, trainer_path, cfg_path, extra_args=trainer_args))

        print("[brain] all processes started. Ctrl+C to stop.")

        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None:
                    raise RuntimeError(f"Process exited unexpectedly: pid={p.pid} code={rc}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[brain] stopping...")
    except Exception as e:
        print(f"[brain] error: {e}")
    finally:
        terminate_all(procs)
        print("[brain] done.")


if __name__ == "__main__":
    main()
