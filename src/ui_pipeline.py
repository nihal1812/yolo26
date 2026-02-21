#!/usr/bin/env python3
"""
ui_pipeline.py (Pipeline 3 orchestrator)

Starts:
  - alert_sender.py
  - clip_writer.py
  - bbox_overlay.py
  - feedback_receiver.py

Same style as perception.py/brain.py:
  - start subprocesses in parallel
  - monitor children
  - shutdown all on Ctrl+C or if one dies
"""

import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def popen_node(py_exe, script_path: Path, cfg_path: str, extra_args=None):
    if not script_path.exists():
        raise FileNotFoundError(f"Node script not found: {script_path}")

    cmd = [py_exe, str(script_path), "--config", cfg_path]
    if extra_args:
        cmd.extend(extra_args)

    print("[ui_pipeline] CMD:", " ".join(cmd))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--no-alert-sender", action="store_true")
    ap.add_argument("--no-clip-writer", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-feedback", action="store_true")

    # Optional extra args passthrough (rarely needed but handy)
    ap.add_argument("--overlay-args", nargs="*", default=None, help="Extra args for bbox_overlay.py")
    ap.add_argument("--clip-args", nargs="*", default=None, help="Extra args for clip_writer.py")
    ap.add_argument("--alert-args", nargs="*", default=None, help="Extra args for alert_sender.py")
    ap.add_argument("--feedback-args", nargs="*", default=None, help="Extra args for feedback_receiver.py")

    args = ap.parse_args()

    py = sys.executable

    alert_sender = HERE / "alert_sender.py"
    clip_writer  = HERE / "clip_writer.py"
    bbox_overlay = HERE / "bbox_overlay.py"
    feedback_rx  = HERE / "feedback_receiver.py"

    procs = []
    try:
        # Nice startup order:
        # 1) feedback receiver (so UI can post immediately)
        # 2) alert sender / overlay / clip writer (consumers)
        if not args.no_feedback:
            print("[ui_pipeline] starting feedback_receiver...")
            procs.append(popen_node(py, feedback_rx, args.config, args.feedback_args))
            time.sleep(0.2)

        if not args.no_alert_sender:
            print("[ui_pipeline] starting alert_sender...")
            procs.append(popen_node(py, alert_sender, args.config, args.alert_args))
            time.sleep(0.2)

        if not args.no_overlay:
            print("[ui_pipeline] starting bbox_overlay...")
            procs.append(popen_node(py, bbox_overlay, args.config, args.overlay_args))
            time.sleep(0.2)

        if not args.no_clip_writer:
            print("[ui_pipeline] starting clip_writer...")
            procs.append(popen_node(py, clip_writer, args.config, args.clip_args))
            time.sleep(0.2)

        print("[ui_pipeline] all processes started. Ctrl+C to stop.")

        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None:
                    raise RuntimeError(f"Process exited unexpectedly: pid={p.pid} code={rc}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[ui_pipeline] stopping...")
    except Exception as e:
        print(f"[ui_pipeline] error: {e}")
    finally:
        terminate_all(procs)
        print("[ui_pipeline] done.")


if __name__ == "__main__":
    main()
