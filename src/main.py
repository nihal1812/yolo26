#!/usr/bin/env python3
"""
main.py (top-level orchestrator)

Starts three pipelines in parallel:
  1) perception.py   (rtsp_stream + pose_track + seg_track [+ feature_builder])
  2) brain.py        (model_node + policy_node + data_collector + trainer_node)
  3) ui_pipeline.py  (alert_sender + clip_writer + bbox_overlay + feedback_receiver)

Behavior:
  - Starts subprocesses in parallel (small stagger for clarity)
  - Monitors children; if any exits unexpectedly -> shutdown all
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
    return subprocess.Popen(cmd, cwd=str(HERE))


def terminate_all(procs, grace_s=3.0):
    # soft stop
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

    # hard stop
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    # optional switches
    ap.add_argument("--no-perception", action="store_true")
    ap.add_argument("--no-brain", action="store_true")
    ap.add_argument("--no-ui", action="store_true")

    # pass-through flags to ui_pipeline
    ap.add_argument("--no-alert-sender", action="store_true")
    ap.add_argument("--no-clip-writer", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-feedback", action="store_true")

    args = ap.parse_args()

    py = sys.executable

    perception = HERE / "perception.py"
    brain      = HERE / "brain.py"
    ui_pipe    = HERE / "ui_pipeline.py"

    procs = []
    try:
        if not args.no_perception:
            print("[main] starting perception pipeline...")
            procs.append(popen_node(py, perception, args.config))
            time.sleep(0.5)

        if not args.no_brain:
            print("[main] starting brain pipeline...")
            procs.append(popen_node(py, brain, args.config))
            time.sleep(0.5)

        if not args.no_ui:
            print("[main] starting UI pipeline...")
            ui_extra = []
            if args.no_alert_sender: ui_extra.append("--no-alert-sender")
            if args.no_clip_writer:  ui_extra.append("--no-clip-writer")
            if args.no_overlay:      ui_extra.append("--no-overlay")
            if args.no_feedback:     ui_extra.append("--no-feedback")
            procs.append(popen_node(py, ui_pipe, args.config, extra_args=ui_extra))
            time.sleep(0.5)

        if not procs:
            raise RuntimeError("Nothing to start (all pipelines disabled).")

        print("[main] all pipelines started. Ctrl+C to stop.")

        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None:
                    raise RuntimeError(f"Process exited unexpectedly: pid={p.pid} code={rc}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[main] stopping...")
    except Exception as e:
        print(f"[main] error: {e}")
    finally:
        terminate_all(procs)
        print("[main] done.")


if __name__ == "__main__":
    main()
