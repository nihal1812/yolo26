#!/usr/bin/env python3
"""
main.py (top-level orchestrator)

Starts four pipelines:
  1) perception.py
  2) brain.py
  3) ui_pipeline.py
  4) reid_pipeline.py

Behavior:
  - starts subprocesses
  - monitors children
  - restarts failed child pipelines
  - Ctrl+C stops everything cleanly
"""

import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def popen_node(py_exe, script_path: Path, cfg_path: str, extra_args=None):
    cmd = [py_exe, str(script_path), "--config", cfg_path]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[main] exec: {' '.join(map(str, cmd))}")
    return subprocess.Popen(cmd, cwd=str(HERE))


def terminate_proc(p, grace_s=3.0):
    if p is None:
        return
    try:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    except Exception:
        return

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if p.poll() is not None:
            return
        time.sleep(0.1)

    try:
        if p.poll() is None:
            p.kill()
    except Exception:
        pass


def terminate_all(procs, grace_s=3.0):
    for p in procs:
        terminate_proc(p, grace_s=grace_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--no-perception", action="store_true")
    ap.add_argument("--no-brain", action="store_true")
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--no-reid-pipeline", action="store_true")

    # pass-through flags to ui_pipeline
    ap.add_argument("--no-alert-sender", action="store_true")
    ap.add_argument("--no-clip-writer", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--enable-local-alert-senders", action="store_true")
    ap.add_argument("--incident-leader-cam", default=None)

    # top-level restart behavior
    ap.add_argument("--restart_delay_s", type=float, default=2.0)
    ap.add_argument("--max_restarts_per_pipeline", type=int, default=20)
    ap.add_argument("--restart_backoff_step_s", type=float, default=1.0)
    ap.add_argument("--restart_backoff_cap_s", type=float, default=20.0)

    args = ap.parse_args()

    py = sys.executable

    perception = HERE / "perception.py"
    brain = HERE / "brain.py"
    ui_pipe = HERE / "ui_pipeline.py"
    reid_pipe = HERE / "reid_pipeline.py"

    proc_specs = {}
    procs = {}
    restart_counts = {}

    def register_proc(name: str, script_path: Path, extra_args=None):
        proc_specs[name] = {
            "script": script_path,
            "extra_args": list(extra_args) if extra_args else [],
        }
        restart_counts[name] = 0

    if not args.no_perception:
        register_proc("perception", perception, [])

    if not args.no_brain:
        register_proc("brain", brain, [])

    if not args.no_ui:
        ui_extra = []
        if args.no_alert_sender:
            ui_extra.append("--no-alert-sender")
        if args.no_clip_writer:
            ui_extra.append("--no-clip-writer")
        if args.no_overlay:
            ui_extra.append("--no-overlay")
        if args.no_feedback:
            ui_extra.append("--no-feedback")
        if args.enable_local_alert_senders:
            ui_extra.append("--enable-local-alert-senders")
        if args.incident_leader_cam:
            ui_extra.extend(["--incident-leader-cam", str(args.incident_leader_cam)])

        register_proc("ui_pipeline", ui_pipe, ui_extra)

    if not args.no_reid_pipeline:
        register_proc("reid_pipeline", reid_pipe, [])

    if not proc_specs:
        raise RuntimeError("Nothing to start (all pipelines disabled).")

    def start_one(name: str):
        spec = proc_specs[name]
        p = popen_node(py, spec["script"], args.config, extra_args=spec["extra_args"])
        procs[name] = p
        return p

    try:
        # start in dependency-friendly order
        if "perception" in proc_specs:
            print("[main] starting perception pipeline...")
            start_one("perception")
            time.sleep(0.5)

        if "brain" in proc_specs:
            print("[main] starting brain pipeline...")
            start_one("brain")
            time.sleep(0.5)

        if "ui_pipeline" in proc_specs:
            print("[main] starting UI pipeline...")
            start_one("ui_pipeline")
            time.sleep(0.5)

        if "reid_pipeline" in proc_specs:
            print("[main] starting ReID pipeline...")
            start_one("reid_pipeline")
            time.sleep(0.5)

        print("[main] all pipelines started. Ctrl+C to stop.")

        while True:
            for name, p in list(procs.items()):
                rc = p.poll()
                if rc is None:
                    continue

                print(f"[main] child exited: pipeline={name} code={rc}")
                terminate_proc(p)
                procs.pop(name, None)

                restart_counts[name] = int(restart_counts.get(name, 0)) + 1
                if restart_counts[name] > args.max_restarts_per_pipeline:
                    raise RuntimeError(
                        f"[main] pipeline {name} exceeded max restarts "
                        f"({args.max_restarts_per_pipeline})"
                    )

                delay = min(
                    args.restart_delay_s + (restart_counts[name] - 1) * args.restart_backoff_step_s,
                    args.restart_backoff_cap_s,
                )
                print(f"[main] restarting pipeline {name} in {delay:.1f}s ...")
                time.sleep(delay)
                start_one(name)
                time.sleep(0.5)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[main] stopping...")
    except Exception as e:
        print(f"[main] error: {e}")
    finally:
        terminate_all(list(procs.values()))
        print("[main] done.")


if __name__ == "__main__":
    main()