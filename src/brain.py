#!/usr/bin/env python3
"""
brain.py (Orchestrator for the brain pipeline)

Starts per camera:
  - model_node.py
  - policy_node.py
  - data_collector_llm.py / data_collector.py

Starts once globally:
  - trainer_node.py / trainer_node_updated.py   (optional)

Behavior:
  - Always passes --config to every node
  - Passes --cam_id to per-camera nodes
  - Reads optional extra CLI flags from YAML
  - Prefers updated node files when they exist, but falls back to original names
"""

import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

from config_utils import load_cfg, get_active_cams, get_brain_args, format_cam_dict, get_stream

HERE = Path(__file__).resolve().parent


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


def dict_to_cli_args(d: dict):
    """
    Converts a flat dict into CLI args:
      {"device":"cuda:0", "freeze_cnn":True, "epochs":3}
    -> ["--device","cuda:0","--freeze_cnn","--epochs","3"]
    """
    args = []
    for k, v in (d or {}).items():
        key = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(key)
        elif v is not None:
            args.extend([key, str(v)])
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--no-collector", action="store_true")
    ap.add_argument("--no-trainer", action="store_true")

    ap.add_argument("--model-device", default=None)
    ap.add_argument("--trainer-device", default=None)

    ap.add_argument("--collector-script", default=None)
    ap.add_argument("--trainer-script", default=None)

    args = ap.parse_args()
    cfg_path = args.config
    cfg = load_cfg(cfg_path) or {}
    active_cams = get_active_cams(cfg)

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

    brain_cfg = cfg.get("brain", {}) if isinstance(cfg, dict) else {}
    stagger_s = float(brain_cfg.get("stagger_s", 0.3))

    model_cfg_args = get_brain_args(cfg, "model_node_args")
    policy_cfg_args = get_brain_args(cfg, "policy_node_args")
    collector_cfg_args = get_brain_args(cfg, "data_collector_args")
    trainer_cfg_args = get_brain_args(cfg, "trainer_node_args")

    procs = []
    try:
        print(f"[brain] config={cfg_path}")
        print(f"[brain] active_cams={active_cams}")
        print(f"[brain] collector_script={collector_path.name}")
        print(f"[brain] trainer_script={trainer_path.name}")

        # ---------------------------
        # Per-camera nodes
        # ---------------------------
        for cam_id in active_cams:
            print(f"[brain] starting camera brain nodes for {cam_id} ...")

            # -------- model_node args
            model_args_dict = format_cam_dict(model_cfg_args, cam_id)
            model_args = ["--cam_id", cam_id]
            model_args.extend(dict_to_cli_args(model_args_dict))
            if args.model_device:
                model_args.extend(["--device", args.model_device])

            print(f"[brain] starting model_node for {cam_id} ...")
            procs.append(popen_node(py, model_node_path, cfg_path, extra_args=model_args))
            time.sleep(stagger_s)

            # -------- policy_node args
            if not args.no_policy:
                policy_args_dict = format_cam_dict(policy_cfg_args, cam_id)
                policy_args = ["--cam_id", cam_id]
                policy_args.extend(dict_to_cli_args(policy_args_dict))

                print(f"[brain] starting policy_node for {cam_id} ...")
                procs.append(popen_node(py, policy_node_path, cfg_path, extra_args=policy_args))
                time.sleep(stagger_s)

            # -------- collector args
            if not args.no_collector:
                collector_args_dict = format_cam_dict(collector_cfg_args, cam_id)
                collector_args = ["--cam_id", cam_id]
                collector_args.extend(dict_to_cli_args(collector_args_dict))

                print(f"[brain] starting data_collector for {cam_id} ...")
                procs.append(popen_node(py, collector_path, cfg_path, extra_args=collector_args))
                time.sleep(stagger_s)

        # ---------------------------
        # Global trainer (single process)
        # ---------------------------
        if not args.no_trainer:
            trainer_args_dict = dict(trainer_cfg_args or {})

            # trainer consumes one stream, so choose strategy:
            # current version starts one trainer per camera? No.
            # here we start one global trainer only if exactly one active cam.
            # for multi-cam, start one trainer per cam to avoid breaking current trainer implementation.
            if len(active_cams) == 1:
                cam_id = active_cams[0]
                trainer_args_dict = format_cam_dict(trainer_args_dict, cam_id)
                trainer_args = ["--cam_id", cam_id]
                trainer_args.extend(dict_to_cli_args(trainer_args_dict))
                if args.trainer_device:
                    trainer_args.extend(["--device", args.trainer_device])

                print(f"[brain] starting trainer_node for {cam_id} ...")
                procs.append(popen_node(py, trainer_path, cfg_path, extra_args=trainer_args))
                time.sleep(stagger_s)
            else:
                # Start one trainer per camera because current trainer_node is per-stream/per-cam
                for cam_id in active_cams:
                    trainer_args_cam = format_cam_dict(trainer_args_dict, cam_id)
                    trainer_args = ["--cam_id", cam_id]
                    trainer_args.extend(dict_to_cli_args(trainer_args_cam))
                    if args.trainer_device:
                        trainer_args.extend(["--device", args.trainer_device])

                    print(f"[brain] starting trainer_node for {cam_id} ...")
                    procs.append(popen_node(py, trainer_path, cfg_path, extra_args=trainer_args))
                    time.sleep(stagger_s)

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