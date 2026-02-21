#!/usr/bin/env python3
"""
brain.py  (Orchestrator for your "brain pipeline")

Starts and supervises:
  - model_node.py
  - policy_node.py
  - data_collector.py
  - trainer_node.py

Behavior:
  - Spawns all nodes as subprocesses
  - Staggers startup slightly (so Redis consumers don't race on first connect)
  - If any child exits unexpectedly -> stops everything
  - Ctrl+C -> graceful shutdown (SIGINT), then hard kill if needed

Usage:
  python brain.py --config path/to/config.yaml

Notes:
  - This assumes the node filenames are in the same folder as brain.py
  - If you keep nodes in subfolders, adjust NODE_PATHS below.
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
    return subprocess.Popen(cmd, cwd=str(HERE))


def terminate_all(procs, grace_s=3.0):
    # SIGINT first
    for p in procs:
        try:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        except Exception:
            pass

    # wait a bit
    t0 = time.time()
    while time.time() - t0 < grace_s:
        if all(p.poll() is not None for p in procs):
            return
        time.sleep(0.1)

    # hard kill
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    # Optional: allow disabling nodes for debugging
    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--no-collector", action="store_true")
    ap.add_argument("--no-trainer", action="store_true")

    # Optional: pass through device flags to model/trainer
    ap.add_argument("--model-device", default=None)   # e.g. cuda:0
    ap.add_argument("--trainer-device", default=None) # e.g. cuda:0

    args = ap.parse_args()
    cfg_path = args.config
    _cfg = load_cfg(cfg_path)  # just validates config exists

    py = sys.executable

    # ---- Node paths (edit here if your filenames differ)
    model_node_path = HERE / "model_node.py"
    policy_node_path = HERE / "policy_node.py"
    collector_path = HERE / "data_collector.py"
    trainer_path = HERE / "trainer_node.py"

    # ---- Extra args (optional)
    model_args = []
    if args.model_device:
        model_args += ["--device", args.model_device]

    trainer_args = []
    if args.trainer_device:
        trainer_args += ["--device", args.trainer_device]

    procs = []
    try:
        print("[brain] starting model_node...")
        procs.append(popen_node(py, model_node_path, cfg_path, extra_args=model_args))

        time.sleep(0.5)

        if not args.no_policy:
            print("[brain] starting policy_node...")
            procs.append(popen_node(py, policy_node_path, cfg_path))

        time.sleep(0.3)

        if not args.no_collector:
            print("[brain] starting data_collector...")
            procs.append(popen_node(py, collector_path, cfg_path))

        time.sleep(0.3)

        if not args.no_trainer:
            print("[brain] starting trainer_node...")
            procs.append(popen_node(py, trainer_path, cfg_path, extra_args=trainer_args))

        print("[brain] all processes started. Ctrl+C to stop.")

        while True:
            # If any process exits unexpectedly, stop everything
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
