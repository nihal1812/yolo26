#!/usr/bin/env python3
"""
ui_pipeline.py

Starts:
  - alert_sender.py          (global incident sender, optional local per-camera senders)
  - clip_writer.py           (per camera)
  - bbox_overlay.py          (per camera)
  - feedback_receiver.py     (single global process)

Behavior:
  - starts subprocesses
  - monitors children
  - restarts failed children
  - clean shutdown on Ctrl+C
"""

import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

from config_utils import load_cfg, get_active_cams

HERE = Path(__file__).resolve().parent


def popen_node(py_exe, script_path: Path, cfg_path: str, extra_args=None):
    if not script_path.exists():
        raise FileNotFoundError(f"Node script not found: {script_path}")

    cmd = [py_exe, str(script_path), "--config", cfg_path]
    if extra_args:
        cmd.extend(extra_args)

    print("[ui_pipeline] CMD:", " ".join(map(str, cmd)))
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

    ap.add_argument("--no-alert-sender", action="store_true")
    ap.add_argument("--no-clip-writer", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-feedback", action="store_true")

    ap.add_argument("--enable-local-alert-senders", action="store_true")
    ap.add_argument("--incident-leader-cam", default=None)

    ap.add_argument("--overlay-args", nargs="*", default=None)
    ap.add_argument("--clip-args", nargs="*", default=None)
    ap.add_argument("--alert-args", nargs="*", default=None)
    ap.add_argument("--feedback-args", nargs="*", default=None)

    ap.add_argument("--restart_delay_s", type=float, default=2.0)
    ap.add_argument("--max_restarts_per_child", type=int, default=50)
    ap.add_argument("--restart_backoff_step_s", type=float, default=1.0)
    ap.add_argument("--restart_backoff_cap_s", type=float, default=15.0)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    active_cams = get_active_cams(cfg)
    if not active_cams:
        raise RuntimeError("No active cameras configured.")

    leader_cam = args.incident_leader_cam or active_cams[0]

    py = sys.executable

    alert_sender = HERE / "alert_sender.py"
    clip_writer = HERE / "clip_writer.py"
    bbox_overlay = HERE / "bbox_overlay.py"
    feedback_rx = HERE / "feedback_receiver.py"

    proc_specs = {}
    procs = {}
    restart_counts = {}

    def register_proc(name: str, script_path: Path, extra_args=None):
        proc_specs[name] = {
            "script": script_path,
            "extra_args": list(extra_args) if extra_args else [],
        }
        restart_counts[name] = 0

    if not args.no_feedback:
        register_proc("feedback_receiver", feedback_rx, args.feedback_args or [])

    if not args.no_alert_sender:
        incident_args = [
            "--cam_id", leader_cam,
            "--source_mode", "incident",
            "--incident_leader_cam", leader_cam,
        ]
        if args.alert_args:
            incident_args.extend(args.alert_args)
        register_proc("alert_sender:incident", alert_sender, incident_args)

        if args.enable_local_alert_senders:
            for cam_id in active_cams:
                local_args = [
                    "--cam_id", cam_id,
                    "--source_mode", "local",
                ]
                if args.alert_args:
                    local_args.extend(args.alert_args)
                register_proc(f"alert_sender:local:{cam_id}", alert_sender, local_args)

    if not args.no_overlay:
        for cam_id in active_cams:
            extra = ["--cam_id", cam_id]
            if args.overlay_args:
                extra.extend(args.overlay_args)
            register_proc(f"bbox_overlay:{cam_id}", bbox_overlay, extra)

    if not args.no_clip_writer:
        for cam_id in active_cams:
            extra = ["--cam_id", cam_id]
            if args.clip_args:
                extra.extend(args.clip_args)
            register_proc(f"clip_writer:{cam_id}", clip_writer, extra)

    if not proc_specs:
        raise RuntimeError("Nothing to start (all UI components disabled).")

    def start_one(name: str):
        spec = proc_specs[name]
        print(f"[ui_pipeline] starting {name} ...")
        p = popen_node(py, spec["script"], args.config, spec["extra_args"])
        procs[name] = p
        return p

    try:
        print(f"[ui_pipeline] active_cams={active_cams}")
        print(f"[ui_pipeline] incident_leader_cam={leader_cam}")
        print(f"[ui_pipeline] enable_local_alert_senders={args.enable_local_alert_senders}")

        # start order
        if "feedback_receiver" in proc_specs:
            start_one("feedback_receiver")
            time.sleep(0.3)

        if "alert_sender:incident" in proc_specs:
            start_one("alert_sender:incident")
            time.sleep(0.3)

        for cam_id in active_cams:
            local_name = f"alert_sender:local:{cam_id}"
            if local_name in proc_specs:
                start_one(local_name)
                time.sleep(0.2)

        for cam_id in active_cams:
            name = f"bbox_overlay:{cam_id}"
            if name in proc_specs:
                start_one(name)
                time.sleep(0.2)

        for cam_id in active_cams:
            name = f"clip_writer:{cam_id}"
            if name in proc_specs:
                start_one(name)
                time.sleep(0.2)

        print("[ui_pipeline] all processes started. Ctrl+C to stop.")

        while True:
            for name, p in list(procs.items()):
                rc = p.poll()
                if rc is None:
                    continue

                print(f"[ui_pipeline] child exited: name={name} code={rc}")
                terminate_proc(p)
                procs.pop(name, None)

                restart_counts[name] = int(restart_counts.get(name, 0)) + 1
                if restart_counts[name] > args.max_restarts_per_child:
                    raise RuntimeError(
                        f"[ui_pipeline] child {name} exceeded max restarts "
                        f"({args.max_restarts_per_child})"
                    )

                delay = min(
                    args.restart_delay_s + (restart_counts[name] - 1) * args.restart_backoff_step_s,
                    args.restart_backoff_cap_s,
                )
                print(f"[ui_pipeline] restarting {name} in {delay:.1f}s ...")
                time.sleep(delay)
                start_one(name)
                time.sleep(0.2)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[ui_pipeline] stopping...")
    except Exception as e:
        print(f"[ui_pipeline] error: {e}")
    finally:
        terminate_all(list(procs.values()))
        print("[ui_pipeline] done.")


if __name__ == "__main__":
    main()